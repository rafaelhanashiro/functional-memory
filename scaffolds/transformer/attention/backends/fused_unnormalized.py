"""Fused unnormalized causal attention and its derivative on CUDA.

The kernel k(S) is the exponential (``degree=None``) or its degree-m Taylor
truncation p_m(S) = sum_{n<=m} S^n / n!. The exponential also takes the score
functions of attention/pieces.py: k(S) = f(S) exp(phi(S)), with the transform
phi and prefactor f in softplus and sigmoid (attention/score_transform.py, with
the same stable formulas and derivatives). Each Pallas/Triton program owns a query
tile (forward/dQ) or a key tile (dK/dV). Only causal tiles are visited. Scores
and kernel weights stay in registers/shared memory; neither pass materializes a
sequence-square array. Unshifted kernel values and accumulators are FP32. Mixed
FP32/BF16 products decompose the FP32 operand into three BF16 components;
FP32/FP32 products use TF32x3. Weights are never rounded to a single BF16 value.
For Y = k(S)V, dS = k'(S) * (dY V^T). There is no softmax derivative term,
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

from scaffolds.transformer.attention.backends.fused_taylor import _polynomial, _polynomial_and_derivative


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


def _softplus(scores):
    """max(s, 0) + log1p(exp(-|s|)), as attention/score_transform.py; its derivative is sigmoid(s)."""
    return jnp.maximum(scores, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(scores)))


def _sigmoid_and_slope(scores):
    """sigmoid(s) and its derivative exp(-|s|) / (1 + exp(-|s|))^2, as attention/score_transform.py."""
    tail = jnp.exp(-jnp.abs(scores))
    return jnp.where(scores >= 0.0, 1.0, tail) / (1.0 + tail), tail / ((1.0 + tail) * (1.0 + tail))


def _score_function(scores, transform, prefactor):
    """f(s) exp(phi(s)) and its derivative f'(s) exp(phi(s)) + f(s) exp(phi(s)) phi'(s)."""
    if transform == "softplus":
        values = jnp.exp(_softplus(scores))
        derivative = values * _sigmoid_and_slope(scores)[0]
    elif transform == "sigmoid":
        exponent, slope = _sigmoid_and_slope(scores)
        values = jnp.exp(exponent)
        derivative = values * slope
    else:
        values = jnp.exp(scores)
        derivative = values
    if prefactor == "softplus":
        factor, factor_slope = _softplus(scores), _sigmoid_and_slope(scores)[0]
    elif prefactor == "sigmoid":
        factor, factor_slope = _sigmoid_and_slope(scores)
    else:
        return values, derivative
    return factor * values, factor_slope * values + factor * derivative


def _kernel(scores, degree):
    """``degree`` is the attention degree, or (None, transform, prefactor) for a score function."""
    if isinstance(degree, tuple):
        return _score_function(scores, degree[1], degree[2])[0]
    return jnp.exp(scores) if degree is None else _polynomial(scores, degree)


def _kernel_and_derivative(scores, degree):
    if isinstance(degree, tuple):
        return _score_function(scores, degree[1], degree[2])
    if degree is None:
        values = jnp.exp(scores)
        return values, values
    return _polynomial_and_derivative(scores, degree)


def _kernel_name(degree):
    if isinstance(degree, tuple):
        from scaffolds.transformer.attention.pieces import ScoreFunction, exponential_kernel_name

        return exponential_kernel_name(ScoreFunction(degree[1], degree[2]), normalized=False)
    return "exp_unnormalized" if degree is None else f"taylor_{degree}_unnormalized"


def _load(ref, rows, width, time):
    columns = jnp.arange(ref.shape[-1])
    return plt.load(ref.at[rows, :],
                    mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)


def _forward_kernel(q_ref, k_ref, v_ref, out_ref, *, degree, time, dq, dv, block_q, block_k):
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
        values = _kernel(jnp.where(causal, scores, 0.0), degree)
        invalid_entries = causal & (~jnp.isfinite(scores) | ~jnp.isfinite(values))
        if isinstance(degree, int) and degree % 2 == 0:
            # Even truncations are strictly positive in exact arithmetic.
            invalid_entries |= causal & (values <= 0)
        invalid |= jnp.max(invalid_entries.astype(jnp.int32), axis=1) != 0
        values = jnp.where(causal, values, 0.0)
        output += _dot(values, v)
        return output, invalid

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    output, invalid = lax.fori_loop(0, end, body, initial)
    output = jnp.where(invalid[:, None], jnp.nan, output)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))


def _dq_kernel(q_ref, k_ref, v_ref, do_ref, dq_ref, *, degree, time, dq, dv, block_q, block_k):
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
        _, derivative = _kernel_and_derivative(jnp.where(causal, scores, 0.0), degree)
        dp = _dot(do, v.T)
        ds = jnp.where(causal, dp * derivative, 0.0)
        return grad_q + _dot(ds, k) * (1.0 / math.sqrt(dq))

    grad_q = jnp.zeros(q_ref.shape, jnp.float32)
    # A constant degree-zero kernel has no score derivative.
    if degree != 0:
        end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
        grad_q = lax.fori_loop(0, end, body, grad_q)
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)


def _dkv_kernel(q_ref, k_ref, v_ref, do_ref, dk_ref, dv_ref, *, degree, time, dq, dv, block_q, block_k):
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
        values, derivative = _kernel_and_derivative(jnp.where(causal, scores, 0.0), degree)
        grad_v += _dot(jnp.where(causal, values, 0.0).T, do)
        if degree != 0:
            dp = _dot(do, v.T)
            ds = jnp.where(causal, dp * derivative, 0.0)
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


def _forward(q, k, v, degree):
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv)),
        out_specs=_spec(block_q, dv),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        name=f"{_kernel_name(degree)}_forward",
    )(q, k, v)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _attention(q, k, v, degree):
    output = _forward(q, k, v, degree)
    return output.astype(v.dtype)


def _attention_fwd(q, k, v, degree):
    output = _forward(q, k, v, degree)
    return output.astype(v.dtype), (q, k, v)


def _attention_bwd(degree, residuals, do):
    q, k, v = residuals
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    params = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    grad_q = pl.pallas_call(
        partial(_dq_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  _spec(block_q, dv)),
        out_specs=_spec(block_q, dq), compiler_params=params,
        name=f"{_kernel_name(degree)}_dq",
    )(q, k, v, do)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, degree=degree, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  _full_spec(time, dv)),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)), compiler_params=params,
        name=f"{_kernel_name(degree)}_dkv",
    )(q, k, v, do)
    return grad_q, grad_k, grad_v


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_unnormalized_attention(q, k, v, *, attention_degree: int | None = None, score_transform: str = "identity",
                                 score_prefactor: str = "none"):
    """Causal k(QK^T / sqrt(d_k))V, without a row denominator.

    ``attention_degree=None`` selects the exponential; a nonnegative integer m
    selects its unshifted degree-m Taylor truncation. ``score_transform`` and
    ``score_prefactor`` give the exponential k(S) = f(S) exp(phi(S))
    (attention/pieces.py ``fused_exponential_problem``).

    Supports BF16 and FP32 Q/K/V with one common dtype, positive dimensions, and head widths up to 128.
    GPU execution requires an Ampere-or-newer NVIDIA device (TF32x3 support).
    The blocked implementation remains the scalar-diagnostics reference.
    """
    if attention_degree is not None and (
            not isinstance(attention_degree, int) or isinstance(attention_degree, bool) or attention_degree < 0):
        raise ValueError("attention_degree must be None or a nonnegative integer")
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
    from scaffolds.transformer.attention.pieces import ScoreFunction, fused_exponential_problem

    score_function = ScoreFunction(score_transform, score_prefactor)
    problem = fused_exponential_problem(score_function, attention_degree=attention_degree)
    if problem is not None:
        raise ValueError(f"Fused unnormalized attention cannot evaluate this score function: {problem}")
    if score_function.identity:
        # Every historical recipe: the same static argument, kernels and names as before.
        return _attention(q, k, v, attention_degree)
    return _attention(q, k, v, (None, score_transform, score_prefactor))
