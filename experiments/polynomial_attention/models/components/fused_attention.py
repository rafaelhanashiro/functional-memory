"""Fused causal Taylor attention and its reverse-mode derivative on CUDA.

Each Pallas/Triton program owns a query tile (forward/dQ) or a key tile
(dK/dV). Only causal tiles are visited. Scores and polynomial values stay in
registers/shared memory; neither pass materializes a sequence-square array.
Unshifted Horner arithmetic, normalization, and accumulators are FP32. FP32
matrix products explicitly use TF32x3, never single-TF32 or BF16 weights.
The backward saves the *unrounded* FP32 output for the normalization derivative.
"""

from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


BLOCK_Q = 64
BLOCK_K = 64
_F32_DOT = lax.DotAlgorithmPreset.TF32_TF32_F32_X3


def _dot(a, b):
    precision = _F32_DOT if a.dtype == jnp.float32 else lax.Precision.HIGHEST
    return pl.dot(a, b, precision=precision)


def _polynomial(scores, degree):
    value = jnp.ones_like(scores)
    for order in range(degree, 0, -1):
        value = 1.0 + scores * (value / order)
    return value


def _polynomial_and_derivative(scores, degree):
    # Differentiate the actual Horner recurrence, rather than evaluating a
    # separately rounded p_(degree-1). Both are identical in exact arithmetic.
    value = jnp.ones_like(scores)
    derivative = jnp.zeros_like(scores)
    for order in range(degree, 0, -1):
        derivative = value / order + scores * (derivative / order)
        value = 1.0 + scores * (value / order)
    return value, derivative


def _load(ref, rows, width, time):
    columns = jnp.arange(ref.shape[-1])
    return plt.load(ref.at[rows, :],
                    mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)


def _forward_kernel(q_ref, k_ref, v_ref, out_ref, norm_ref, *, degree, time, dq, dv, block_q, block_k):
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    initial = (jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32),
               jnp.zeros((block_q,), jnp.float32), jnp.zeros((block_q,), jnp.bool_))

    def body(tile, carry):
        output, normalizer, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (ki[None, :] < time)
        values = _polynomial(jnp.where(causal, scores, 0.0), degree)
        invalid_entries = causal & (~jnp.isfinite(scores) | ~jnp.isfinite(values) | (values <= 0))
        invalid |= jnp.max(invalid_entries.astype(jnp.int32), axis=1) != 0
        values = jnp.where(causal, values, 0.0)
        updated = normalizer + jnp.sum(values, axis=1)
        # Online normalized accumulation avoids overflow in an unnormalized
        # sum(p(s)*V) when p(s), sum(p(s)), and the final output are finite.
        denominator = jnp.where(updated > 0, updated, 1.0)
        output = output * (normalizer / denominator)[:, None]
        output += _dot(values / denominator[:, None], v.astype(jnp.float32))
        return output, updated, invalid

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    output, normalizer, invalid = lax.fori_loop(0, end, body, initial)
    invalid |= ~jnp.isfinite(normalizer) | (normalizer <= 0)
    output = jnp.where(invalid[:, None], jnp.nan, output)
    normalizer = jnp.where(invalid, jnp.nan, normalizer)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))
    plt.store(norm_ref, normalizer, mask=qi < time)


def _dq_kernel(q_ref, k_ref, v_ref, do_ref, out_ref, norm_ref, dq_ref, *, degree, time, dq, dv, block_q, block_k):
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0).astype(jnp.float32)
    output = plt.load(out_ref, mask=vmask, other=0)
    normalizer = plt.load(norm_ref, mask=qi < time, other=1)
    delta = jnp.sum(do * output, axis=1)

    def body(tile, grad_q):
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (ki[None, :] < time)
        _, derivative = _polynomial_and_derivative(jnp.where(causal, scores, 0.0), degree)
        dp = _dot(do, v.astype(jnp.float32).T) - delta[:, None]
        ds = jnp.where(causal, dp * (derivative / normalizer[:, None]), 0.0)
        return grad_q + _dot(ds, k.astype(jnp.float32)) * (1.0 / math.sqrt(dq))

    grad_q = jnp.zeros(q_ref.shape, jnp.float32)
    if degree:
        end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
        grad_q = lax.fori_loop(0, end, body, grad_q)
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)


def _dkv_kernel(q_ref, k_ref, v_ref, do_ref, out_ref, norm_ref, dk_ref, dv_ref, *, degree, time, dq, dv, block_q, block_k):
    ki = pl.program_id(0) * block_k + jnp.arange(block_k)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k = plt.load(k_ref, mask=kmask, other=0)
    v = plt.load(v_ref, mask=vmask, other=0).astype(jnp.float32)
    initial = (jnp.zeros(k_ref.shape, jnp.float32), jnp.zeros(v_ref.shape, jnp.float32))

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * block_q + jnp.arange(block_q)
        q = _load(q_ref, qi, dq, time)
        do = _load(do_ref, qi, dv, time).astype(jnp.float32)
        output = _load(out_ref, qi, dv, time)
        normalizer = plt.load(norm_ref.at[qi], mask=qi < time, other=1)
        delta = jnp.sum(do * output, axis=1)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        values, derivative = _polynomial_and_derivative(jnp.where(causal, scores, 0.0), degree)
        weights = jnp.where(causal, values / normalizer[:, None], 0.0)
        grad_v += _dot(weights.T, do)
        if degree:
            dp = _dot(do, v.T) - delta[:, None]
            ds = jnp.where(causal, dp * (derivative / normalizer[:, None]), 0.0)
            grad_k += _dot(ds.T, q.astype(jnp.float32)) * (1.0 / math.sqrt(dq))
        return grad_k, grad_v

    start = lax.div(pl.program_id(0) * block_k, block_q)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, block_q), body, initial)
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


def _spec(tile, width):
    return pl.BlockSpec((None, tile, None, max(16, pl.next_power_of_2(width))),
                        lambda i, b, h: (b, i, h, 0))


def _full_spec(time, width):
    return pl.BlockSpec((None, time, None, max(16, pl.next_power_of_2(width))),
                        lambda i, b, h: (b, 0, h, 0))


def _norm_spec(tile):
    return pl.BlockSpec((None, tile, None), lambda i, b, h: (b, i, h))


def _tile_sizes(dq, dv):
    # Wider heads need smaller tiles to keep TF32x3 backward within H100 SRAM.
    return (BLOCK_Q, BLOCK_K) if max(dq, dv) <= 64 else (32, 32)


def _forward(q, k, v, degree):
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(v.shape, jnp.float32),
                   jax.ShapeDtypeStruct((batch, time, heads), jnp.float32)),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv)),
        out_specs=(_spec(block_q, dv), _norm_spec(block_q)),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=2),
        name=f"taylor_{degree}_forward",
    )(q, k, v)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _attention(q, k, v, degree):
    output, _ = _forward(q, k, v, degree)
    return output.astype(v.dtype)


def _attention_fwd(q, k, v, degree):
    output, norm = _forward(q, k, v, degree)
    return output.astype(v.dtype), (q, k, v, output, norm)


def _attention_bwd(degree, residuals, do):
    q, k, v, output, norm = residuals
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    params = plt.CompilerParams(num_warps=4, num_stages=2)
    grad_q = pl.pallas_call(
        partial(_dq_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  _spec(block_q, dv), _spec(block_q, dv), _norm_spec(block_q)),
        out_specs=_spec(block_q, dq), compiler_params=params,
        name=f"taylor_{degree}_dq",
    )(q, k, v, do, output, norm)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  _full_spec(time, dv), _full_spec(time, dv),
                  pl.BlockSpec((None, time, None), lambda i, b, h: (b, 0, h))),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)), compiler_params=params,
        name=f"taylor_{degree}_dkv",
    )(q, k, v, do, output, norm)
    return grad_q, grad_k, grad_v


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_polynomial_attention(q, k, v, *, attention_degree: int):
    """Causal polynomial attention for [batch, time, heads, channels] arrays.

    Supports BF16 and FP32 Q/K/V with one common dtype, positive dimensions, and head widths up to 128.
    GPU execution requires an Ampere-or-newer NVIDIA device (TF32x3 support).
    The blocked implementation remains the scalar-diagnostics reference.
    """
    if (not isinstance(attention_degree, int) or isinstance(attention_degree, bool)
            or attention_degree < 0 or attention_degree % 2):
        raise ValueError("attention_degree must be a nonnegative even integer")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch, time, heads, channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    if min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    if max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Fused attention requires a common float32 or bfloat16 dtype")
    return _attention(q, k, v, attention_degree)
