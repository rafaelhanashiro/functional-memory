"""Native Triton scalar attention with an inlined value/derivative callback.

ScalarKernel.function is a @triton.jit function (s, parameters, WITH_GRAD)
returning (f(s), f'(s)). Parameters are immutable compile-time constants.
The JAX custom VJP differentiates Q/K/V and the optional per-head RMSNorm gain;
function parameters are fixed. No arbitrary Python callable is executed on GPU.

Streaming accumulation keeps Y = exp(log_scale) * accumulator with a positive
scale 1 + sum(abs(f)). Fused RMSNorm uses epsilon * exp(-2*log_scale), including
in its analytic backward. Signed weights and zero outputs are supported.
The exponential uses score shifting before evaluation; other callbacks must
return finite values on valid scores. This does not repair cancellation error.
"""
from dataclasses import dataclass
from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ScalarKernel:
    name: str
    function: Any
    parameters: tuple = ()
    exponential: bool = False


def exponential_kernel():
    from .triton_scalar_kernels import exponential
    return ScalarKernel("exp", exponential, exponential=True)


def polynomial_kernel(coefficients):
    from .triton_scalar_kernels import polynomial
    values = tuple(float(x) for x in coefficients)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Polynomial coefficients must be nonempty and finite")
    return ScalarKernel("polynomial", polynomial, values)


def basis_kernel(projection):
    """Lower an existing frozen named-basis projection without refitting it."""
    from .triton_scalar_kernels import basis_sum
    from ..basis import BasisFunction
    terms = []
    for coefficient, function in zip(projection.coefficients, projection.basis, strict=True):
        if not isinstance(function, BasisFunction):
            raise ValueError("Fused basis attention needs named functions or an explicit ScalarKernel")
        p = dict(function.parameters)
        parameters = ((p["center"], p["width"]) if function.kind == "rbf" else
                      (p["power"],) if function.kind == "monomial" else
                      (p["frequency"], p["phase"]))
        terms.append((function.kind, float(coefficient), parameters))
    if not terms:
        raise ValueError("A scalar basis must be nonempty")
    return ScalarKernel("basis", basis_sum, tuple(terms))


@dataclass(frozen=True)
class Tuning:
    forward_q: int = 128
    forward_k: int = 64
    forward_warps: int = 4
    stages: int = 3
    backward_q: int = 128
    backward_k: int = 32
    kv_q: int = 32
    kv_k: int = 128


def _call(fn, *args, shapes, grid, name, **meta):
    from .ejkernel_bridge import triton_call
    return triton_call(*args, kernel=fn, out_shape=shapes, grid=grid,
                       name=f"scalar::{name}", enable_fp_fusion=False, **meta)


def _common(q, v, kernel, mode, precise):
    return dict(T=q.shape[1], H=q.shape[2], DQ=q.shape[-1], DV=v.shape[-1],
                SCALE=q.shape[-1] ** -0.5, MODE=mode, FN=kernel.function,
                PARAMS=kernel.parameters, EXP=kernel.exponential, PRECISE=precise)


def _forward(q, k, v, gain, kernel, mode, streaming, precise, eps, tuning):
    from . import triton_scalar_kernels as kernels
    shapes = (jax.ShapeDtypeStruct(v.shape, v.dtype),
              jax.ShapeDtypeStruct(v.shape, jnp.float32),
              jax.ShapeDtypeStruct((*v.shape[:3], 2), jnp.float32))
    return _call(kernels.forward, q, k, v, gain, shapes=shapes,
                 grid=((q.shape[1] + tuning.forward_q - 1) // tuning.forward_q, q.shape[0] * q.shape[2]),
                 name=f"{kernel.name}_forward", **_common(q, v, kernel, mode, precise),
                 EPS=eps, STREAM=streaming, BM=tuning.forward_q, BN=tuning.forward_k,
                 num_warps=tuning.forward_warps, num_stages=tuning.stages)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _attention(q, k, v, gain, kernel, mode, streaming, precise, eps, tuning):
    return _forward(q, k, v, gain, kernel, mode, streaming, precise, eps, tuning)[0]


def _attention_fwd(q, k, v, gain, kernel, mode, streaming, precise, eps, tuning):
    out, n, stats = _forward(q, k, v, gain, kernel, mode, streaming, precise, eps, tuning)
    return out, (q, k, v, gain, n, stats)


def _attention_bwd(kernel, mode, streaming, precise, eps, tuning, residuals, do):
    from . import triton_scalar_kernels as kernels
    q, k, v, gain, n, stats = residuals
    batch, time, heads, width = v.shape
    tiles = (time + 127) // 128
    adj, delta, d_gain = _call(
        kernels.preprocess, n, stats, gain, do,
        shapes=(jax.ShapeDtypeStruct(v.shape, jnp.float32),
                jax.ShapeDtypeStruct(v.shape[:3], jnp.float32),
                jax.ShapeDtypeStruct((batch, heads, tiles, width), jnp.float32)),
        grid=(tiles, batch * heads), name="preprocess", T=time, H=heads, DV=width,
        MODE=mode, BM=128, num_warps=4)
    meta = _common(q, v, kernel, mode, precise)
    dq = _call(kernels.backward_q, q, k, v, adj, delta, stats,
               shapes=jax.ShapeDtypeStruct(q.shape, q.dtype),
               grid=((time + tuning.backward_q - 1) // tuning.backward_q, batch * heads),
               name=f"{kernel.name}_dq", **meta, BM=tuning.backward_q, BN=tuning.backward_k,
               num_warps=4, num_stages=3)
    dk, dv = _call(kernels.backward_kv, q, k, v, adj, delta, stats,
                  shapes=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
                  grid=((time + tuning.kv_k - 1) // tuning.kv_k, batch * heads),
                  name=f"{kernel.name}_dkv", **meta, BM=tuning.kv_q, BN=tuning.kv_k,
                  num_warps=4, num_stages=3)
    dg = jnp.sum(d_gain, axis=(0, 2)).astype(gain.dtype) if mode == 2 else jnp.zeros_like(gain)
    return dq, dk, dv, dg


_attention.defvjp(_attention_fwd, _attention_bwd)


def scalar_attention(q, k, v, *, kernel: ScalarKernel, attention_normalization=False,
                     output_norm_scale=None, eps=1e-6, streaming=None,
                     precision="bf16", tuning=Tuning()):
    """Causal f(QKᵀ/sqrt(dK))V, optionally normalized or fused with head RMSNorm.

    BTHD layout, common BF16/FP32 input dtype, head widths 16/32/64/128.
    Precision 'bf16' rounds weights and RMS adjoints for BF16 tensor products;
    'tf32x3' retains FP32 operands. Outputs and Q/K/V gradients use input dtype.
    The first implementation targets H100 (sm90); other GPUs are rejected.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Expected BTHD Q/K and matching BTH dimensions for V")
    if min(*q.shape, v.shape[-1]) <= 0 or q.shape[-1] not in (16, 32, 64, 128) or v.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("Scalar Triton attention supports positive sizes and widths 16/32/64/128")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError("Expected a common BF16 or FP32 dtype")
    if precision not in ("bf16", "tf32x3") or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid precision or RMSNorm epsilon")
    if attention_normalization and output_norm_scale is not None:
        raise ValueError("Fused RMSNorm currently consumes unnormalized attention")
    if output_norm_scale is not None and output_norm_scale.shape != v.shape[-2:]:
        raise ValueError("RMSNorm gain must have shape [heads, value_width]")
    if not isinstance(kernel, ScalarKernel):
        raise TypeError("kernel must be a ScalarKernel")
    if tuning.forward_q % tuning.forward_k or tuning.forward_q not in (64, 128) or tuning.forward_k not in (32, 64):
        raise ValueError("Invalid forward tile sizes")
    device = jax.devices()[0]
    if device.platform != "gpu" or str(device.compute_capability).split(".")[0] != "9":
        raise ValueError("Scalar Triton attention currently requires H100/sm90")
    mode = 2 if output_norm_scale is not None else 1 if attention_normalization else 0
    if streaming is None:
        streaming = mode != 0
    gain = output_norm_scale if output_norm_scale is not None else jnp.ones(v.shape[-2:], jnp.float32)
    return _attention(q, k, v, gain, kernel, mode, bool(streaming), precision == "tf32x3", eps, tuning)
