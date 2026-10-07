"""Fused unnormalized attention with learned per-head polynomial coefficients on CUDA.

The kernel is p_h(S) = sum_{r<=m} c_{h,r} S^r with FP32 coefficients per head,
evaluated by Horner's rule on unshifted scores; weights may be signed and are
never divided by a row sum. Tiles, products, masking, and FP32 accumulation are
those of ``fused_unnormalized``: only causal tiles are visited and no pass
materializes a sequence-square array. For Y = p(S)V, dS = p'(S) * (dY V^T), and
dc_{h,r} = sum_{i>=j} S_ij^r (dY_i . V_j). The dQ pass stores each coefficient's
partial sum per query tile, batch, and head, reduced afterwards without atomics.
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
    value = jnp.zeros_like(scores) + coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        value = value * scores + coefficient
    return value


def _values_and_derivative(scores, coefficients):
    # Differentiate the Horner recurrence that produced the forward values.
    value = jnp.zeros_like(scores) + coefficients[-1]
    derivative = jnp.zeros_like(scores)
    for coefficient in coefficients[-2::-1]:
        derivative = derivative * scores + value
        value = value * scores + coefficient
    return value, derivative


def _forward_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k, query_guard=False):
    coefficients = tuple(ref[...] for ref in refs[:degree + 1])
    out_ref = refs[degree + 1 + int(query_guard)]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    if query_guard:
        guard = plt.load(refs[degree + 1], mask=qi[:, None] < time, other=0)
    initial = (jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32),
               jnp.zeros((block_q,), jnp.bool_))

    def body(tile, carry):
        output, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (ki[None, :] < time)
        values = _values(jnp.where(causal, scores, 0.0), coefficients)
        invalid_entries = causal & (~jnp.isfinite(scores) | ~jnp.isfinite(values))
        if query_guard:
            actual_kernel = guard[:, 0, None] + guard[:, 1, None] * jnp.where(causal, scores, 0.0)
            invalid_entries |= causal & ~jnp.isfinite(actual_kernel)
        invalid |= jnp.max(invalid_entries.astype(jnp.int32), axis=1) != 0
        output += _dot(jnp.where(causal, values, 0.0), v)
        return output, invalid

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    output, invalid = lax.fori_loop(0, end, body, initial)
    output = jnp.where(invalid[:, None], jnp.nan, output)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))


def _dq_coefficient_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    coefficients = tuple(ref[...] for ref in refs[:count])
    do_ref, dq_ref = refs[count], refs[count + 1]
    grad_refs = refs[count + 2:]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)

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
            updated.append(grad + jnp.sum(dp * power))
            power = power * safe_scores
        return grad_q, tuple(updated)

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    grad_q, grads = lax.fori_loop(
        0, end, body,
        (jnp.zeros(q_ref.shape, jnp.float32), tuple(jnp.zeros((), jnp.float32) for _ in range(count))),
    )
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    for grad_ref, grad in zip(grad_refs, grads, strict=True):
        grad_ref[...] = grad


def _dkv_kernel(q_ref, k_ref, v_ref, *refs, degree, time, dq, dv, block_q, block_k):
    count = degree + 1
    coefficients = tuple(ref[...] for ref in refs[:count])
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


def _coefficient_spec():
    return pl.BlockSpec((None,), lambda i, b, h: (h,))


def _columns(coefficients):
    return tuple(coefficients[:, order] for order in range(coefficients.shape[-1]))


def _forward(q, k, v, coefficients, interpret, query_kernel_guard=None):
    batch, time, heads, dq = q.shape
    dv, degree = v.shape[-1], coefficients.shape[-1] - 1
    block_q, block_k = _tile_sizes(dq, dv)
    guard_inputs = () if query_kernel_guard is None else (query_kernel_guard,)
    guard_specs = (() if query_kernel_guard is None else
                   (pl.BlockSpec((None, block_q, None, 2), lambda i, b, h: (b, i, h, 0)),))
    return pl.pallas_call(
        partial(_forward_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k,
                query_guard=query_kernel_guard is not None),
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(),) * (degree + 1), *guard_specs),
        out_specs=_spec(block_q, dv),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        interpret=interpret, name=f"learned_polynomial_{degree}_unnormalized_forward",
    )(q, k, v, *_columns(coefficients), *guard_inputs)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _attention(q, k, v, coefficients, interpret):
    return _forward(q, k, v, coefficients, interpret).astype(v.dtype)


def _attention_fwd(q, k, v, coefficients, interpret):
    return _forward(q, k, v, coefficients, interpret).astype(v.dtype), (q, k, v, coefficients)


def _attention_bwd(interpret, residuals, do):
    q, k, v, coefficients = residuals
    batch, time, heads, dq = q.shape
    dv, degree = v.shape[-1], coefficients.shape[-1] - 1
    count = degree + 1
    block_q, block_k = _tile_sizes(dq, dv)
    query_tiles = pl.cdiv(time, block_q)
    compiler = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    columns = _columns(coefficients)
    grad_q, *partials = pl.pallas_call(
        partial(_dq_coefficient_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype),
                   *(jax.ShapeDtypeStruct((query_tiles, batch, heads), jnp.float32),) * count),
        grid=(query_tiles, batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(),) * count, _spec(block_q, dv)),
        out_specs=(_spec(block_q, dq), *(pl.BlockSpec((None, None, None), lambda i, b, h: (i, b, h)),) * count),
        compiler_params=compiler, interpret=interpret,
        name=f"learned_polynomial_{degree}_unnormalized_dq_coefficients",
    )(q, k, v, *columns, do)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  *(_coefficient_spec(),) * count, _full_spec(time, dv)),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)), compiler_params=compiler, interpret=interpret,
        name=f"learned_polynomial_{degree}_unnormalized_dkv",
    )(q, k, v, *columns, do)
    grad_coefficients = jnp.stack([jnp.sum(part, axis=(0, 1), dtype=jnp.float32) for part in partials], axis=-1)
    return grad_q, grad_k, grad_v, grad_coefficients


_attention.defvjp(_attention_fwd, _attention_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _attention_fp32(q, k, v, coefficients, interpret):
    return _forward(q, k, v, coefficients, interpret)


def _attention_fp32_fwd(q, k, v, coefficients, interpret):
    return _forward(q, k, v, coefficients, interpret), (q, k, v, coefficients)


_attention_fp32.defvjp(_attention_fp32_fwd, _attention_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _attention_guarded_fp32(q, k, v, coefficients, guard, interpret):
    return _forward(q, k, v, coefficients, interpret, query_kernel_guard=guard)


def _attention_guarded_fp32_fwd(q, k, v, coefficients, guard, interpret):
    return (_forward(q, k, v, coefficients, interpret, query_kernel_guard=guard),
            (q, k, v, coefficients, guard))


def _attention_guarded_fp32_bwd(interpret, residuals, do):
    # The extra argument only guards invalid rows; valid output arithmetic and
    # its derivatives remain the basis polynomial. Coefficient learning flows
    # through the caller's FP32 combination of the basis outputs.
    *ordinary, guard = residuals
    return (*_attention_bwd(interpret, tuple(ordinary), do), jnp.zeros_like(guard))


_attention_guarded_fp32.defvjp(_attention_guarded_fp32_fwd, _attention_guarded_fp32_bwd)


def fused_learned_polynomial_attention(q, k, v, coefficients, *, keep_output_float32=False,
                                       query_kernel_guard=None):
    """Causal p_h(QK^T / sqrt(d_k))V with learned FP32 coefficients [heads, m+1], m >= 1.

    Q/K/V use [batch, time, heads, channels], a common FP32/BF16 dtype, and head
    widths up to 128. GPU execution requires an Ampere-or-newer NVIDIA device.
    Nonfinite causal scores or kernel values poison their rows with NaN. The
    blocked implementation remains the scalar-diagnostics reference.
    ``keep_output_float32`` retains the accumulator for later FP32 combinations.
    The default preserves the established output cast to V's dtype.
    An optional [B,T,H,2] ``query_kernel_guard`` checks finiteness of the actual
    query-dependent c0+c1*s kernel while accumulating a linear basis.
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
    if (coefficients.ndim != 2 or coefficients.shape[0] != q.shape[2] or coefficients.shape[1] < 2
            or coefficients.dtype != jnp.float32):
        raise ValueError("Polynomial coefficients must be float32 with shape [heads, m+1], m >= 1")
    if query_kernel_guard is not None:
        if not keep_output_float32 or query_kernel_guard.shape != (*q.shape[:3], 2):
            raise ValueError("Query kernel guard requires FP32 output and shape [batch,time,heads,2]")
        return _attention_guarded_fp32(q, k, v, coefficients, query_kernel_guard.astype(jnp.float32), False)
    operation = _attention_fp32 if keep_output_float32 else _attention
    return operation(q, k, v, coefficients, False)
