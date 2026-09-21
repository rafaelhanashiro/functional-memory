"""Fused unnormalized exponential attention and its derivative on CUDA.

Each Pallas/Triton program owns a query tile (forward/dQ) or a key tile
(dK/dV). Only causal tiles are visited. Scores and exponential weights stay in
registers/shared memory; neither pass materializes a sequence-square array.
Unshifted exponentials and accumulators are FP32. Mixed
FP32/BF16 products decompose the FP32 operand into three BF16 components;
FP32/FP32 products use TF32x3. Weights are never rounded to a single BF16 value.
For Y = exp(S)V, dS = exp(S) * (dY V^T). There is no softmax derivative term,
row denominator, or max shift. The backward recomputes weights from Q and K;
only Q/K/V are retained by the custom VJP.
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
NUM_WARPS = 4
NUM_STAGES = 2
_F32_DOT = lax.DotAlgorithmPreset.TF32_TF32_F32_X3


def _bf16_components(value):
    """Split FP32 into BF16 components without rounding the high part to inf.

    Truncating each leading component leaves a same-sign residual. Three
    components retain the FP32 significand away from BF16 underflow; products
    and their sums still have ordinary FP32 accumulation error.
    """
    def truncate(operand):
        bits = lax.bitcast_convert_type(operand, jnp.uint32)
        return lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32)

    high = truncate(value)
    residual = value - high
    middle = truncate(residual)
    low = residual - middle
    return high.astype(jnp.bfloat16), middle.astype(jnp.bfloat16), low.astype(jnp.bfloat16)


def _dot(a, b):
    if a.dtype == jnp.float32 and b.dtype == jnp.bfloat16:
        high, middle, low = _bf16_components(a)
        return (pl.dot(high, b, precision=lax.Precision.HIGHEST)
                + pl.dot(middle, b, precision=lax.Precision.HIGHEST)
                + pl.dot(low, b, precision=lax.Precision.HIGHEST))
    if a.dtype == b.dtype == jnp.bfloat16:
        return pl.dot(a, b, precision=lax.Precision.HIGHEST)
    return pl.dot(a.astype(jnp.float32), b.astype(jnp.float32), precision=_F32_DOT)


def _load(ref, rows, width, time):
    columns = jnp.arange(ref.shape[-1])
    return plt.load(ref.at[rows, :],
                    mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)


def _forward_kernel(q_ref, k_ref, v_ref, out_ref, *, time, dq, dv, block_q, block_k):
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    initial = (jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32),
               jnp.zeros((block_q,), jnp.bool_))

    def body(tile, carry):
        output, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (ki[None, :] < time)
        values = jnp.exp(jnp.where(causal, scores, 0.0))
        invalid_entries = causal & (~jnp.isfinite(scores) | ~jnp.isfinite(values))
        invalid |= jnp.max(invalid_entries.astype(jnp.int32), axis=1) != 0
        values = jnp.where(causal, values, 0.0)
        output += _dot(values, v)
        return output, invalid

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    output, invalid = lax.fori_loop(0, end, body, initial)
    output = jnp.where(invalid[:, None], jnp.nan, output)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))


def _dq_kernel(q_ref, k_ref, v_ref, do_ref, dq_ref, *, time, dq, dv, block_q, block_k):
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)

    def body(tile, grad_q):
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (ki[None, :] < time)
        weights = jnp.exp(jnp.where(causal, scores, 0.0))
        dp = _dot(do, v.T)
        ds = jnp.where(causal, dp * weights, 0.0)
        return grad_q + _dot(ds, k) * (1.0 / math.sqrt(dq))

    grad_q = jnp.zeros(q_ref.shape, jnp.float32)
    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    grad_q = lax.fori_loop(0, end, body, grad_q)
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)


def _dkv_kernel(q_ref, k_ref, v_ref, do_ref, dk_ref, dv_ref, *, time, dq, dv, block_q, block_k):
    ki = pl.program_id(0) * block_k + jnp.arange(block_k)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k = plt.load(k_ref, mask=kmask, other=0)
    v = plt.load(v_ref, mask=vmask, other=0)
    initial = (jnp.zeros(k_ref.shape, jnp.float32), jnp.zeros(v_ref.shape, jnp.float32))

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * block_q + jnp.arange(block_q)
        q = _load(q_ref, qi, dq, time)
        do = _load(do_ref, qi, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        weights = jnp.where(causal, jnp.exp(jnp.where(causal, scores, 0.0)), 0.0)
        grad_v += _dot(weights.T, do)
        dp = _dot(do, v.T)
        ds = jnp.where(causal, dp * weights, 0.0)
        grad_k += _dot(ds.T, q) * (1.0 / math.sqrt(dq))
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


def _tile_sizes(dq, dv):
    # Wider heads need smaller tiles to keep backward within H100 SRAM.
    return (BLOCK_Q, BLOCK_K) if max(dq, dv) <= 64 else (32, 32)


def _forward(q, k, v):
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv)),
        out_specs=_spec(block_q, dv),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        name="exp_unnormalized_forward",
    )(q, k, v)


@jax.custom_vjp
def _attention(q, k, v):
    output = _forward(q, k, v)
    return output.astype(v.dtype)


def _attention_fwd(q, k, v):
    output = _forward(q, k, v)
    return output.astype(v.dtype), (q, k, v)


def _attention_bwd(residuals, do):
    q, k, v = residuals
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    params = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    grad_q = pl.pallas_call(
        partial(_dq_kernel, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  _spec(block_q, dv)),
        out_specs=_spec(block_q, dq), compiler_params=params,
        name="exp_unnormalized_dq",
    )(q, k, v, do)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  _full_spec(time, dv)),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)), compiler_params=params,
        name="exp_unnormalized_dkv",
    )(q, k, v, do)
    return grad_q, grad_k, grad_v


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_unnormalized_attention(q, k, v):
    """Causal exp(QK^T / sqrt(d_k))V, without a softmax denominator.

    Supports BF16 and FP32 Q/K/V with one common dtype, positive dimensions, and head widths up to 128.
    GPU execution requires an Ampere-or-newer NVIDIA device (TF32x3 support).
    The blocked implementation remains the scalar-diagnostics reference.
    """
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
    return _attention(q, k, v)
