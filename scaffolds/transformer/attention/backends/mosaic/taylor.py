"""Normalized causal Taylor attention on Mosaic GPU kernels (Hopper and Blackwell).

Same mathematics and failure behavior as ``fused_taylor.fused_polynomial_attention``: FP32
scores, polynomials, sums, and accumulators; FP32 weights and score gradients enter the
tensor cores as three BF16 components; a row whose sum is nonfinite, nonpositive (even
degree) or zero (odd degree), or whose output overflows, is NaN. The output is divided by
its row sum once, after accumulation, so signed odd degrees need a single pass.

Supported: BF16 Q/K/V of shape [batch, time, heads, channels], Q/K width 8, 16, 32, 64, or
128 (widths below 64 are zero-padded, which leaves every score unchanged), value width 64,
degree >= 1, and time a multiple of 128 (H100) or 256 (B300). Anything else, and any GPU
other than compute capability 9.0 or 10.x, raises ValueError.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp

WIDTHS = (8, 16, 32, 64, 128)
VALUE_WIDTH = 64


def _kernels():
    capability = jax.devices()[0].compute_capability
    major = int(capability.split(".")[0])
    if capability == "9.0":
        from scaffolds.transformer.attention.backends.mosaic import hopper
        return "hopper", hopper
    if major == 10:
        from scaffolds.transformer.attention.backends.mosaic import blackwell
        return "blackwell", blackwell
    raise ValueError(f"Mosaic GPU Taylor attention supports compute capability 9.0 and 10.x, not {capability}")


def check(q, k, v, degree):
    """Raise ValueError unless the Mosaic kernels support these inputs."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError(f"Mosaic Taylor attention requires an integer degree >= 1, got {degree!r}")
    if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q and K must share the shape [batch, time, heads, channels], with V matching to channels")
    if not q.dtype == k.dtype == v.dtype == jnp.bfloat16:
        raise ValueError(f"Mosaic Taylor attention requires BF16 Q/K/V, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.shape[-1] not in WIDTHS:
        raise ValueError(f"Mosaic Taylor attention supports Q/K widths {WIDTHS}, got {q.shape[-1]}")
    if v.shape[-1] != VALUE_WIDTH:
        raise ValueError(f"Mosaic Taylor attention supports value width {VALUE_WIDTH}, got {v.shape[-1]}")
    arch, _ = _kernels()
    multiple = 128 if arch == "hopper" else 256
    if q.shape[1] % multiple:
        raise ValueError(f"Mosaic Taylor attention on {arch} requires time to be a multiple of {multiple}, got {q.shape[1]}")


def _pad(x):
    width = x.shape[-1]
    return x if width >= 64 else jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 64 - width)))


def _forward(q, k, v, degree):
    arch, kernels = _kernels()
    scale = 1.0 / math.sqrt(q.shape[-1])
    out, norm = kernels.forward(_pad(q), _pad(k), v, degree=degree, scale=scale)
    return out, norm


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _attention(q, k, v, degree):
    return _forward(q, k, v, degree)[0].astype(v.dtype)


def _attention_fwd(q, k, v, degree):
    out, norm = _forward(q, k, v, degree)
    return out.astype(v.dtype), (q, k, v, out, norm)


def _attention_bwd(degree, residuals, do):
    q, k, v, out, norm = residuals
    arch, kernels = _kernels()
    width = q.shape[-1]
    scale = 1.0 / math.sqrt(width)
    qp, kp = _pad(q), _pad(k)
    delta = jnp.sum(do.astype(jnp.float32) * out, axis=-1).transpose(0, 2, 1)
    inv_z = (1.0 / norm).transpose(0, 2, 1)
    args = (qp, kp, v, do, delta, inv_z)
    if arch == "hopper":
        grad_q = kernels.dq(*args, degree=degree, scale=scale)
        grad_k, grad_v = kernels.dkv(*args, degree=degree, scale=scale)
    else:
        stages = 2 if width > 64 else 3  # shared memory at width 128
        grad_q = kernels.dq(*args, degree=degree, scale=scale, stages=stages)
        grad_k = kernels.dkv(*args, which="dk", degree=degree, scale=scale, stages=stages)
        grad_v = kernels.dkv(*args, which="dv", degree=degree, scale=scale)
    return grad_q[..., :width], grad_k[..., :width], grad_v


_attention.defvjp(_attention_fwd, _attention_bwd)


def taylor_attention(q, k, v, *, attention_degree: int):
    """Normalized causal T_m(QK^T / sqrt(d_K)) V for [batch, time, heads, channels] arrays."""
    check(q, k, v, attention_degree)
    return _attention(q, k, v, attention_degree)
