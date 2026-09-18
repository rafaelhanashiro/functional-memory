"""Streamed learned quadratic attention with coefficient gradients on CUDA.

Both families first compute the complete causal normalizer, then accumulate
normalized values in a second pass. Signed kernels therefore never divide by
partial tile sums. The backward stores independent coefficient-gradient
partials per query tile, batch, and head and reduces them without atomics.
"""

from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from experiments.polynomial_attention.models.components.fused_attention import (
    NUM_STAGES, NUM_WARPS, _dot, _full_spec, _load, _norm_spec, _spec,
    _tile_sizes,
)


def _quadratic(scores, first, second, mode):
    if mode == "positive":
        shifted = first * scores + second
        return 0.5 * (1.0 + shifted * shifted)
    return 1.0 + scores * (first + second * scores)


def _derivatives(scores, first, second, mode):
    if mode == "positive":
        shifted = first * scores + second
        return first * shifted, scores * shifted, shifted
    return first + 2.0 * second * scores, scores, scores * scores


def _forward_kernel(q_ref, k_ref, v_ref, first_ref, second_ref, out_ref, norm_ref,
                    *, mode, time, dq, dv, block_q, block_k):
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) &
                 (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    first, second = first_ref[...], second_ref[...]
    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)

    def normalization_tile(tile, carry):
        normalizer, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k = _load(k_ref, ki, dq, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = ((qi[:, None] >= ki[None, :]) &
                  (qi[:, None] < time) & (ki[None, :] < time))
        values = _quadratic(jnp.where(causal, scores, 0.0), first, second, mode)
        bad = ~jnp.isfinite(scores) | ~jnp.isfinite(values)
        if mode == "positive":
            bad |= values <= 0
        invalid |= jnp.max((causal & bad).astype(jnp.int32), axis=1) != 0
        normalizer += jnp.sum(jnp.where(causal, values, 0.0), axis=1)
        return normalizer, invalid

    normalizer, invalid = lax.fori_loop(
        0, end, normalization_tile,
        (jnp.zeros((block_q,), jnp.float32), jnp.zeros((block_q,), jnp.bool_)),
    )
    invalid |= ~jnp.isfinite(normalizer) | (normalizer == 0)
    if mode == "positive":
        invalid |= normalizer < 0
    # The placeholder is used only for rows that are subsequently poisoned.
    denominator = jnp.where(invalid, 1.0, normalizer)

    def output_tile(tile, output):
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = ((qi[:, None] >= ki[None, :]) &
                  (qi[:, None] < time) & (ki[None, :] < time))
        values = _quadratic(jnp.where(causal, scores, 0.0), first, second, mode)
        weights = jnp.where(causal, values / denominator[:, None], 0.0)
        return output + _dot(weights, v)

    output = lax.fori_loop(
        0, end, output_tile,
        jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32),
    )
    output = jnp.where(invalid[:, None], jnp.nan, output)
    normalizer = jnp.where(invalid, jnp.nan, normalizer)
    plt.store(out_ref, output, mask=(qi[:, None] < time) &
              (jnp.arange(out_ref.shape[-1])[None, :] < dv))
    plt.store(norm_ref, normalizer, mask=qi < time)


def _dq_coefficient_kernel(q_ref, k_ref, v_ref, first_ref, second_ref, do_ref,
                           out_ref, norm_ref, dq_ref, grad_first_ref, grad_second_ref,
                           *, mode, time, dq, dv, block_q, block_k):
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)
    output = plt.load(out_ref, mask=vmask, other=0)
    normalizer = plt.load(norm_ref, mask=qi < time, other=1)
    delta = jnp.sum(do.astype(jnp.float32) * output, axis=1)
    first, second = first_ref[...], second_ref[...]

    def body(tile, carry):
        grad_q, grad_first, grad_second = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = ((qi[:, None] >= ki[None, :]) &
                  (qi[:, None] < time) & (ki[None, :] < time))
        derivative, first_derivative, second_derivative = _derivatives(
            jnp.where(causal, scores, 0.0), first, second, mode,
        )
        dp = _dot(do, v.T) - delta[:, None]
        ds = jnp.where(causal, dp * (derivative / normalizer[:, None]), 0.0)
        grad_q += _dot(ds, k) * (1.0 / math.sqrt(dq))
        grad_first += jnp.sum(jnp.where(
            causal, dp * (first_derivative / normalizer[:, None]), 0.0,
        ))
        grad_second += jnp.sum(jnp.where(
            causal, dp * (second_derivative / normalizer[:, None]), 0.0,
        ))
        return grad_q, grad_first, grad_second

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    grad_q, grad_first, grad_second = lax.fori_loop(
        0, end, body,
        (jnp.zeros(q_ref.shape, jnp.float32),
         jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)),
    )
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    grad_first_ref[...] = grad_first
    grad_second_ref[...] = grad_second


def _dkv_kernel(q_ref, k_ref, v_ref, first_ref, second_ref, do_ref, out_ref,
                norm_ref, dk_ref, dv_ref, *, mode, time, dq, dv, block_q, block_k):
    ki = pl.program_id(0) * block_k + jnp.arange(block_k)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k = plt.load(k_ref, mask=kmask, other=0)
    v = plt.load(v_ref, mask=vmask, other=0)
    first, second = first_ref[...], second_ref[...]

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * block_q + jnp.arange(block_q)
        q = _load(q_ref, qi, dq, time)
        do = _load(do_ref, qi, dv, time)
        output = _load(out_ref, qi, dv, time)
        normalizer = plt.load(norm_ref.at[qi], mask=qi < time, other=1)
        delta = jnp.sum(do.astype(jnp.float32) * output, axis=1)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = ((qi[:, None] >= ki[None, :]) &
                  (qi[:, None] < time) & (ki[None, :] < time))
        safe_scores = jnp.where(causal, scores, 0.0)
        values = _quadratic(safe_scores, first, second, mode)
        derivative, _, _ = _derivatives(safe_scores, first, second, mode)
        weights = jnp.where(causal, values / normalizer[:, None], 0.0)
        grad_v += _dot(weights.T, do)
        dp = _dot(do, v.T) - delta[:, None]
        ds = jnp.where(causal, dp * (derivative / normalizer[:, None]), 0.0)
        grad_k += _dot(ds.T, q) * (1.0 / math.sqrt(dq))
        return grad_k, grad_v

    start = lax.div(pl.program_id(0) * block_k, block_q)
    grad_k, grad_v = lax.fori_loop(
        start, pl.cdiv(time, block_q), body,
        (jnp.zeros(k_ref.shape, jnp.float32), jnp.zeros(v_ref.shape, jnp.float32)),
    )
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


def _coefficient_spec():
    return pl.BlockSpec((None,), lambda i, b, h: (h,))


def _forward(q, k, v, coefficients, mode, interpret):
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, mode=mode, time=time, dq=dq, dv=dv,
                block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(v.shape, jnp.float32),
                   jax.ShapeDtypeStruct((batch, time, heads), jnp.float32)),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  _coefficient_spec(), _coefficient_spec()),
        out_specs=(_spec(block_q, dv), _norm_spec(block_q)),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        interpret=interpret, name=f"quadratic_{mode}_forward",
    )(q, k, v, coefficients[:, 0], coefficients[:, 1])


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _attention(q, k, v, coefficients, mode, interpret=False):
    output, _ = _forward(q, k, v, coefficients, mode, interpret)
    return output.astype(v.dtype)


def _attention_fwd(q, k, v, coefficients, mode, interpret):
    output, norm = _forward(q, k, v, coefficients, mode, interpret)
    return output.astype(v.dtype), (q, k, v, coefficients, output, norm)


def _attention_bwd(mode, interpret, residuals, do):
    q, k, v, coefficients, output, norm = residuals
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    query_tiles = pl.cdiv(time, block_q)
    compiler = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    grad_q, first_partials, second_partials = pl.pallas_call(
        partial(_dq_coefficient_kernel, mode=mode, time=time, dq=dq, dv=dv,
                block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype),
                   jax.ShapeDtypeStruct((query_tiles, batch, heads), jnp.float32),
                   jax.ShapeDtypeStruct((query_tiles, batch, heads), jnp.float32)),
        grid=(query_tiles, batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  _coefficient_spec(), _coefficient_spec(),
                  _spec(block_q, dv), _spec(block_q, dv),
                  _norm_spec(block_q)),
        out_specs=(_spec(block_q, dq),
                   pl.BlockSpec((None, None, None), lambda i, b, h: (i, b, h)),
                   pl.BlockSpec((None, None, None), lambda i, b, h: (i, b, h))),
        compiler_params=compiler, interpret=interpret, name=f"quadratic_{mode}_dq_coefficients",
    )(q, k, v, coefficients[:, 0], coefficients[:, 1], do, output, norm)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, mode=mode, time=time, dq=dq, dv=dv,
                block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype),
                   jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  _coefficient_spec(), _coefficient_spec(),
                  _full_spec(time, dv), _full_spec(time, dv),
                  pl.BlockSpec((None, time, None), lambda i, b, h: (b, 0, h))),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)),
        compiler_params=compiler, interpret=interpret, name=f"quadratic_{mode}_dkv",
    )(q, k, v, coefficients[:, 0], coefficients[:, 1], do, output, norm)
    grad_coefficients = jnp.stack((
        jnp.sum(first_partials, axis=(0, 1), dtype=jnp.float32),
        jnp.sum(second_partials, axis=(0, 1), dtype=jnp.float32),
    ), axis=-1)
    return grad_q, grad_k, grad_v, grad_coefficients


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_quadratic_attention(q, k, v, coefficients, *, quadratic_mode):
    """Attend to a causal prefix with FP32 per-head coefficient pairs.

    Q/K/V use [batch,time,heads,channels] and a common FP32/BF16 dtype.
    Positive pairs are (a,b) in .5*(1+(a*s+b)^2); unrestricted pairs are
    (b,c) in 1+b*s+c*s^2. Only the unrestricted mode permits signed weights
    and negative nonzero row sums. Zero/nonfinite sums poison their rows.
    """
    if quadratic_mode not in ("positive", "unrestricted"):
        raise ValueError("quadratic_mode must be positive or unrestricted")
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
    if coefficients.shape != (q.shape[2], 2) or coefficients.dtype != jnp.float32:
        raise ValueError("Quadratic coefficients must be float32 with shape [heads, 2]")
    return _attention(q, k, v, coefficients, quadratic_mode, False)
