"""Fused causal polynomials with per-query coefficients and per-key factors.

For Y_t = sum_(i<=t) p_ti(s_ti) V_i with p_ti(s) = sum_r c_(t,r) f_(i,r) s^r, every tile forms
the rank-one pair coefficients c_(t,r) f_(i,r) inside a single Horner evaluation, so all orders
share one score tile and the output is accumulated in FP32 and rounded once. Nothing pairwise
is stored. The backward query pass computes dQ and dc_(t,r) = sum_(i<=t) f_(i,r) s_ti^r dP_ti;
the key pass computes dK, dV, and df_(i,r) = sum_(t>=i) c_(t,r) s_ti^r dP_ti, with
dP_ti = dY_t . V_i. This is the query-polynomial kernel with the key factors added; with every
factor one it computes the same function.
"""

from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from scaffolds.transformer.attention.backends.fused_unnormalized import (
    NUM_STAGES, NUM_WARPS, _dot, _full_spec, _load, _spec, _tile_sizes,
)


def _pair_values(scores, coefficients, factors):
    """Horner with pair coefficients c_r[q] f_r[k]; coefficients [BQ] and factors [BK] per order."""
    value = jnp.zeros_like(scores) + coefficients[-1][:, None] * factors[-1][None, :]
    for coefficient, factor in zip(coefficients[-2::-1], factors[-2::-1]):
        value = value * scores + coefficient[:, None] * factor[None, :]
    return value


def _pair_values_and_derivative(scores, coefficients, factors):
    value = jnp.zeros_like(scores) + coefficients[-1][:, None] * factors[-1][None, :]
    derivative = jnp.zeros_like(scores)
    for coefficient, factor in zip(coefficients[-2::-1], factors[-2::-1]):
        derivative = derivative * scores + value
        value = value * scores + coefficient[:, None] * factor[None, :]
    return value, derivative


def _forward_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    coefficient_refs, factor_refs, out_ref = refs[:count], refs[count:2 * count], refs[2 * count]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    coefficients = tuple(plt.load(ref, mask=qi < time, other=0) for ref in coefficient_refs)
    initial = (jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32), jnp.zeros((block_q,), jnp.bool_))

    def body(tile, carry):
        output, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        factors = tuple(plt.load(ref.at[ki], mask=ki < time, other=0) for ref in factor_refs)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        values = _pair_values(jnp.where(causal, scores, 0.0), coefficients, factors)
        invalid_entries = causal & (~jnp.isfinite(scores) | ~jnp.isfinite(values))
        invalid |= jnp.max(invalid_entries.astype(jnp.int32), axis=1) != 0
        output += _dot(jnp.where(causal, values, 0.0), v)
        return output, invalid

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    output, invalid = lax.fori_loop(0, end, body, initial)
    output = jnp.where(invalid[:, None], jnp.nan, output)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))


def _dq_coefficient_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    coefficient_refs, factor_refs = refs[:count], refs[count:2 * count]
    do_ref, dq_ref = refs[2 * count], refs[2 * count + 1]
    grad_refs = refs[2 * count + 2:]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)
    coefficients = tuple(plt.load(ref, mask=qi < time, other=0) for ref in coefficient_refs)

    def body(tile, carry):
        grad_q, grads = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        factors = tuple(plt.load(ref.at[ki], mask=ki < time, other=0) for ref in factor_refs)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        safe_scores = jnp.where(causal, scores, 0.0)
        _, derivative = _pair_values_and_derivative(safe_scores, coefficients, factors)
        dp = jnp.where(causal, _dot(do, v.T), 0.0)
        grad_q += _dot(dp * derivative, k) * (1.0 / math.sqrt(dq))
        power, updated = jnp.ones_like(safe_scores), []
        for grad, factor in zip(grads, factors, strict=True):
            updated.append(grad + jnp.sum(dp * power * factor[None, :], axis=1))
            power = power * safe_scores
        return grad_q, tuple(updated)

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    grad_q, grads = lax.fori_loop(
        0, end, body,
        (jnp.zeros(q_ref.shape, jnp.float32), tuple(jnp.zeros((block_q,), jnp.float32) for _ in range(count))),
    )
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    for grad_ref, grad in zip(grad_refs, grads, strict=True):
        plt.store(grad_ref, grad, mask=qi < time)


def _dkv_factor_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    coefficient_refs, factor_refs = refs[:count], refs[count:2 * count]
    do_ref, dk_ref, dv_ref = refs[2 * count:2 * count + 3]
    grad_refs = refs[2 * count + 3:]
    ki = pl.program_id(0) * block_k + jnp.arange(block_k)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k = plt.load(k_ref, mask=kmask, other=0)
    v = plt.load(v_ref, mask=vmask, other=0)
    factors = tuple(plt.load(ref, mask=ki < time, other=0) for ref in factor_refs)
    initial = (jnp.zeros(k_ref.shape, jnp.float32), jnp.zeros(v_ref.shape, jnp.float32),
               tuple(jnp.zeros((block_k,), jnp.float32) for _ in range(count)))

    def body(tile, carry):
        grad_k, grad_v, grads = carry
        qi = tile * block_q + jnp.arange(block_q)
        q = _load(q_ref, qi, dq, time)
        do = _load(do_ref, qi, dv, time)
        coefficients = tuple(plt.load(ref.at[qi], mask=qi < time, other=0) for ref in coefficient_refs)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        safe_scores = jnp.where(causal, scores, 0.0)
        values, derivative = _pair_values_and_derivative(safe_scores, coefficients, factors)
        grad_v += _dot(jnp.where(causal, values, 0.0).T, do)
        dp = jnp.where(causal, _dot(do, v.T), 0.0)
        grad_k += _dot((dp * derivative).T, q) * (1.0 / math.sqrt(dq))
        power, updated = jnp.ones_like(safe_scores), []
        for grad, coefficient in zip(grads, coefficients, strict=True):
            updated.append(grad + jnp.sum(dp * power * coefficient[:, None], axis=0))
            power = power * safe_scores
        return grad_k, grad_v, tuple(updated)

    start = lax.div(pl.program_id(0) * block_k, block_q)
    grad_k, grad_v, grads = lax.fori_loop(start, pl.cdiv(time, block_q), body, initial)
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)
    for grad_ref, grad in zip(grad_refs, grads, strict=True):
        plt.store(grad_ref, grad, mask=ki < time)


def _block_spec(block):
    return pl.BlockSpec((None, block, None), lambda i, b, h: (b, i, h))


def _full_column_spec(time):
    return pl.BlockSpec((None, time, None), lambda i, b, h: (b, 0, h))


def _columns(array):
    return tuple(array[..., order] for order in range(array.shape[-1]))


def _backward_tile_sizes(dq, dv, *, degree, time, dtype, interpret):
    """Qualified smaller backward tiles; other shapes/devices retain the original geometry.

    At the production H100/B300 context, m=8/12's per-degree gradient
    accumulators make 64x64 backward tiles substantially slower than 32x32.
    Forward keeps its existing tiles. Precision and reductions are unchanged.
    """
    if (not interpret and dtype == jnp.bfloat16 and dq == dv == 64
            and time == 2048 and degree in (8, 12)):
        devices = jax.local_devices()
        h100 = all(getattr(device, "compute_capability", None) == "9.0"
                   and getattr(device, "device_kind", "").startswith("NVIDIA H100")
                   for device in devices)
        b300 = all(getattr(device, "compute_capability", None) == "10.3"
                   and getattr(device, "device_kind", "").startswith("NVIDIA B300")
                   for device in devices)
        if devices and (h100 or b300):
            return 32, 32
    return _tile_sizes(dq, dv)


def _forward(q, k, v, coefficients, factors, interpret):
    batch, time, heads, dq = q.shape
    dv, degree = v.shape[-1], coefficients.shape[-1] - 1
    count = degree + 1
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_block_spec(block_q),) * count, *(_full_column_spec(time),) * count),
        out_specs=_spec(block_q, dv),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        interpret=interpret, name=f"query_key_polynomial_{degree}_unnormalized_forward",
    )(q, k, v, *_columns(coefficients), *_columns(factors))


def _cast_output(output, dtype):
    output = output.astype(dtype)
    return jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _attention(q, k, v, coefficients, factors, interpret):
    return _cast_output(_forward(q, k, v, coefficients, factors, interpret), v.dtype)


def _attention_fwd(q, k, v, coefficients, factors, interpret):
    output = _cast_output(_forward(q, k, v, coefficients, factors, interpret), v.dtype)
    return output, (q, k, v, coefficients, factors)


def _attention_bwd(interpret, residuals, do):
    q, k, v, coefficients, factors = residuals
    batch, time, heads, dq = q.shape
    dv, degree = v.shape[-1], coefficients.shape[-1] - 1
    count = degree + 1
    block_q, block_k = _backward_tile_sizes(dq, dv, degree=degree, time=time, dtype=q.dtype, interpret=interpret)
    compiler = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    coefficient_columns, factor_columns = _columns(coefficients), _columns(factors)
    grad_q, *grad_coefficients = pl.pallas_call(
        partial(_dq_coefficient_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype),
                   *(jax.ShapeDtypeStruct(coefficients.shape[:3], jnp.float32),) * count),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_block_spec(block_q),) * count, *(_full_column_spec(time),) * count, _spec(block_q, dv)),
        out_specs=(_spec(block_q, dq), *(_block_spec(block_q),) * count),
        compiler_params=compiler, interpret=interpret,
        name=f"query_key_polynomial_{degree}_unnormalized_dq_coefficients",
    )(q, k, v, *coefficient_columns, *factor_columns, do)
    grad_k, grad_v, *grad_factors = pl.pallas_call(
        partial(_dkv_factor_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype),
                   *(jax.ShapeDtypeStruct(factors.shape[:3], jnp.float32),) * count),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  *(_full_column_spec(time),) * count, *(_block_spec(block_k),) * count, _full_spec(time, dv)),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv), *(_block_spec(block_k),) * count),
        compiler_params=compiler, interpret=interpret,
        name=f"query_key_polynomial_{degree}_unnormalized_dkv_factors",
    )(q, k, v, *coefficient_columns, *factor_columns, do)
    return grad_q, grad_k, grad_v, jnp.stack(grad_coefficients, axis=-1), jnp.stack(grad_factors, axis=-1)


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_query_key_polynomial_attention(q, k, v, coefficients, factors, interpret=False):
    """Unnormalized causal sum_r c_(t,r) f_(i,r) s_ti^r V_i with c, f [B, T, H, m+1].

    Q/K/V share FP32 or BF16 dtype and head widths up to 128. Coefficients and factors are
    FP32 with degree m >= 1. Arithmetic and their gradients are FP32; the output and Q/K/V
    gradients have their operand dtypes. A nonfinite causal score or polynomial value poisons
    the complete output row. ``interpret=True`` runs the Pallas CPU interpreter.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch,time,heads,channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    if min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    if max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Fused attention requires a common float32 or bfloat16 dtype")
    for name, array in (("coefficients", coefficients), ("factors", factors)):
        if (array.ndim != 4 or array.shape[:3] != q.shape[:3]
                or array.shape[-1] < 2 or array.dtype != jnp.float32):
            raise ValueError(f"Query-key {name} must be float32 with shape [batch,time,heads,m+1], m>=1")
    if coefficients.shape != factors.shape:
        raise ValueError("Query coefficients and key factors must have the same shape")
    return _attention(q, k, v, coefficients, factors, interpret)
