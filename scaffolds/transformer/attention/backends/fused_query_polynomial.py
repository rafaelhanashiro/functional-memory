"""Fused causal polynomials with FP32 coefficients for every query and head.

For Y_t = sum_(i<=t) p_t(s_ti) V_i, p_t(s) = sum_r c_(t,r) s^r,
all orders share a single score tile and a single Horner evaluation. No basis
outputs or sequence-square arrays are materialized. The backward query pass
computes dQ and dc_(t,r) = sum_(i<=t) s_ti^r (dY_t . V_i); the key pass computes
dK and dV. Coefficient gradients remain per query instead of reducing across
queries as the static-coefficient backend does.
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


def _values(scores, coefficients):
    value = jnp.zeros_like(scores) + coefficients[-1][:, None]
    for coefficient in coefficients[-2::-1]:
        value = value * scores + coefficient[:, None]
    return value


def _values_and_derivative(scores, coefficients):
    value = jnp.zeros_like(scores) + coefficients[-1][:, None]
    derivative = jnp.zeros_like(scores)
    for coefficient in coefficients[-2::-1]:
        derivative = derivative * scores + value
        value = value * scores + coefficient[:, None]
    return value, derivative


def _forward_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    out_ref = refs[count]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    coefficients = tuple(plt.load(ref, mask=qi < time, other=0) for ref in refs[:count])
    initial = (jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32), jnp.zeros((block_q,), jnp.bool_))

    def body(tile, carry):
        output, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        values = _values(jnp.where(causal, scores, 0.0), coefficients)
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
    do_ref, dq_ref = refs[count], refs[count + 1]
    grad_refs = refs[count + 2:]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)
    coefficients = tuple(plt.load(ref, mask=qi < time, other=0) for ref in refs[:count])

    def body(tile, carry):
        grad_q, grads = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        safe_scores = jnp.where(causal, scores, 0.0)
        _, derivative = _values_and_derivative(safe_scores, coefficients)
        dp = jnp.where(causal, _dot(do, v.T), 0.0)
        grad_q += _dot(dp * derivative, k) * (1.0 / math.sqrt(dq))
        power, updated = jnp.ones_like(safe_scores), []
        for grad in grads:
            updated.append(grad + jnp.sum(dp * power, axis=1))
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


def _dkv_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    do_ref, dk_ref, dv_ref = refs[count:count + 3]
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
        coefficients = tuple(plt.load(ref.at[qi], mask=qi < time, other=0) for ref in refs[:count])
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        values, derivative = _values_and_derivative(jnp.where(causal, scores, 0.0), coefficients)
        grad_v += _dot(jnp.where(causal, values, 0.0).T, do)
        ds = jnp.where(causal, _dot(do, v.T) * derivative, 0.0)
        grad_k += _dot(ds.T, q) * (1.0 / math.sqrt(dq))
        return grad_k, grad_v

    start = lax.div(pl.program_id(0) * block_k, block_q)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, block_q), body, initial)
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


def _coefficient_spec(block_q):
    return pl.BlockSpec((None, block_q, None), lambda i, b, h: (b, i, h))


def _full_coefficient_spec(time):
    return pl.BlockSpec((None, time, None), lambda i, b, h: (b, 0, h))


def _columns(coefficients):
    return tuple(coefficients[..., order] for order in range(coefficients.shape[-1]))


def _forward(q, k, v, coefficients, interpret):
    batch, time, heads, dq = q.shape
    dv, degree = v.shape[-1], coefficients.shape[-1] - 1
    count = degree + 1
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(block_q),) * count),
        out_specs=_spec(block_q, dv),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        interpret=interpret, name=f"query_polynomial_{degree}_unnormalized_forward",
    )(q, k, v, *_columns(coefficients))


def _cast_output(output, dtype):
    output = output.astype(dtype)
    return jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _attention(q, k, v, coefficients, interpret):
    return _cast_output(_forward(q, k, v, coefficients, interpret), v.dtype)


def _attention_fwd(q, k, v, coefficients, interpret):
    return _cast_output(_forward(q, k, v, coefficients, interpret), v.dtype), (q, k, v, coefficients)


def _attention_bwd(interpret, residuals, do):
    q, k, v, coefficients = residuals
    batch, time, heads, dq = q.shape
    dv, degree = v.shape[-1], coefficients.shape[-1] - 1
    count = degree + 1
    block_q, block_k = _tile_sizes(dq, dv)
    compiler = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    columns = _columns(coefficients)
    grad_q, *grad_columns = pl.pallas_call(
        partial(_dq_coefficient_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype),
                   *(jax.ShapeDtypeStruct(coefficients.shape[:3], jnp.float32),) * count),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(block_q),) * count, _spec(block_q, dv)),
        out_specs=(_spec(block_q, dq), *(_coefficient_spec(block_q),) * count),
        compiler_params=compiler, interpret=interpret,
        name=f"query_polynomial_{degree}_unnormalized_dq_coefficients",
    )(q, k, v, *columns, do)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  *(_full_coefficient_spec(time),) * count, _full_spec(time, dv)),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)),
        compiler_params=compiler, interpret=interpret,
        name=f"query_polynomial_{degree}_unnormalized_dkv",
    )(q, k, v, *columns, do)
    return grad_q, grad_k, grad_v, jnp.stack(grad_columns, axis=-1)


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_query_polynomial_attention(q, k, v, coefficients, interpret=False):
    """Unnormalized causal p_(b,t,h)(QK^T/sqrt(d_K))V with c[B,T,H,m+1].

    Q/K/V share FP32 or BF16 dtype and head widths up to 128. Coefficients are
    FP32 with degree m>=1. Arithmetic and coefficient gradients are FP32; output
    and Q/K/V gradients have their respective operand dtypes. Nonfinite causal
    scores or polynomial values poison the complete output row, even if V is
    zero. ``interpret=True`` runs the Pallas CPU interpreter for qualification.
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
    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3]
            or coefficients.shape[-1] < 2 or coefficients.dtype != jnp.float32):
        raise ValueError("Query polynomial coefficients must be float32 with shape [batch,time,heads,m+1], m>=1")
    return _attention(q, k, v, coefficients, interpret)
