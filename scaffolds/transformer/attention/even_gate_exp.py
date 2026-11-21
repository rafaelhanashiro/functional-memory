"""Degree-16 query-gated even powers and a separately gated exp, in common row units."""
import math

import jax
from jax import lax
import jax.numpy as jnp


ORDERS = tuple(range(2, 17, 2))
DEGREE = ORDERS[-1]
BASIS_COUNT = len(ORDERS) + 1


def row_shift(scores, exp_active):
    """Log upper bound for every ungated basis in a row. Masked scores may be zero."""
    magnitude = jnp.maximum(1.0, jnp.max(jnp.abs(scores), axis=-1))
    exp_max = jnp.where(exp_active, jnp.max(scores, axis=-1), 0.0)
    return lax.stop_gradient(jnp.maximum(exp_max, DEGREE * jnp.log(magnitude)))


def shifted_basis(scores, shift, exp_active, *, slopes=False):
    """Return e^-shift times each basis (and optionally its derivative with shift fixed).

    Set u = s exp(-shift/16). Then |u| <= 1 and s^r exp(-shift) = u^r exp(-(16-r)shift/16).
    Applying the scale in two halves also handles large negative FP32 scores whose reciprocals
    would be subnormal. All exponential arguments are nonpositive on eligible pairs.
    """
    half = jnp.exp(shift * (-0.5 / DEGREE))
    u = (scores * half) * half
    even = []
    derivatives = []
    for r in ORDERS:
        # For r=16 the exponential factor is identically one.
        common = 1.0 if r == DEGREE else jnp.exp(shift * (-(DEGREE - r) / DEGREE))
        even.append((u ** r) * common)
        if slopes:
            derivatives.append((r * (u ** (r - 1)) * common * half) * half)
    # A sigmoid rounded to zero is flat in the model. Its basis and its logit gradient both vanish.
    # Mask before exp: a switched-off basis must neither overflow nor set the units of the polynomial.
    exponential = jnp.exp(jnp.where(exp_active, scores, -jnp.inf) - shift)
    values = (*even, exponential)
    return (values, (*derivatives, exponential)) if slopes else values


def _reference(q, k, v, c, exp_gate, return_diagnostics):
    """Evaluate complete causal rows in small batches; all diagnostics use the same shifted weights."""
    batch, time, heads, width = q.shape
    block = min(32, time)
    positions = jnp.arange(time)

    def rows(block_index):
        query_positions = block * block_index + jnp.arange(block)
        index = jnp.minimum(query_positions, time - 1)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q[:, index], k, preferred_element_type=jnp.float32,
                            precision=lax.Precision.HIGHEST) * (1.0 / math.sqrt(width))
        causal = positions[None, None, None, :] <= index[None, None, :, None]
        scores = jnp.where(causal, scores, 0.0)
        exp_coefficient = exp_gate[:, index].swapaxes(1, 2)
        active = exp_coefficient != 0.0
        shift = row_shift(scores, active)
        bases = shifted_basis(scores, shift[..., None], active[..., None])
        weights = exp_coefficient[..., None] * bases[-1]
        for r, basis in zip(ORDERS, bases[:-1], strict=True):
            weights = weights + c[..., r][:, index].swapaxes(1, 2)[..., None] * basis
        weights = jnp.where(causal, weights, 0.0)
        output = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32),
                            precision=lax.Precision.HIGHEST)
        outputs = output, shift.swapaxes(1, 2)
        if not return_diagnostics:
            return outputs
        actual = query_positions < time
        measured = actual[None, None, :, None] & causal
        weight_sums = weights.sum(axis=-1)
        min_measured = lambda x, mask: jnp.min(jnp.where(mask, x, jnp.inf))
        max_measured = lambda x, mask: jnp.max(jnp.where(mask, x, -jnp.inf))
        stats = {"score_min": min_measured(scores, measured), "score_max": max_measured(scores, measured),
                 "kernel_min": min_measured(weights, measured), "kernel_max": max_measured(weights, measured),
                 "weight_sum_min": min_measured(weight_sums, actual), "weight_sum_max": max_measured(weight_sums, actual),
                 "exp_shift_min": min_measured(shift, actual), "exp_shift_max": max_measured(shift, actual),
                 "invalid_score_count": jnp.sum(measured & ~jnp.isfinite(scores)),
                 "invalid_kernel_count": jnp.sum(measured & ~jnp.isfinite(weights)),
                 "invalid_weight_sum_count": jnp.sum(actual & ~jnp.isfinite(weight_sums)),
                 "invalid_output_count": jnp.sum(actual[None, :, None, None] & ~jnp.isfinite(output)),
                 "causal_entry_count": jnp.sum(measured) * batch * heads,
                 "negative_weight_count": jnp.sum(measured & (weights < 0))}
        return outputs, stats

    result = lax.map(rows, jnp.arange((time + block - 1) // block))
    if return_diagnostics:
        result, stats = result
    def flatten(x):
        return x.swapaxes(0, 1).reshape(batch, -1, *x.shape[3:])[:, :time]
    output = tuple(flatten(x) for x in result)
    if not return_diagnostics:
        return output
    stats = {name: (values.min() if name.endswith("_min") else values.max() if name.endswith("_max")
                    else values.sum()) for name, values in stats.items()}
    stats["min_abs_weight_sum"] = stats["weight_sum_min"]
    stats["negative_kernel_count"] = stats["negative_weight_count"]
    stats["negative_weight_fraction"] = stats["negative_weight_count"] / stats["causal_entry_count"]
    stats["negative_kernel_fraction"] = stats["negative_weight_fraction"]
    # Every summand is nonnegative; zero-weight rows have no cancellation either.
    stats["cancellation_ratio_mean"] = stats["cancellation_ratio_max"] = jnp.float32(1.0)
    from scaffolds.transformer.attention.input_dependent import coefficient_statistics
    stats.update(coefficient_statistics(c))
    return output, lax.stop_gradient(stats)


def even_gate_exp_attention(q, k, v, coefficients, exp_gate, *, backend="blocked", return_diagnostics=False):
    """Return shifted output and a stopped row shift; head RMSNorm compensates its epsilon with that shift.

    exp_gate is the model's sigmoid output. A sigmoid rounded to exactly zero switches the exponential
    off, including its zero derivative through the sigmoid logit; it is not a free linear coefficient at zero.
    """
    if (q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]
            or min(*q.shape, v.shape[-1]) <= 0 or coefficients.shape != (*q.shape[:3], DEGREE + 1)
            or exp_gate.shape != q.shape[:3]):
        raise ValueError("Expected Q/K/V [B,T,H,D], degree-16 coefficients [B,T,H,17] and exp gate [B,T,H]")
    if (q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16)
            or coefficients.dtype != jnp.float32 or exp_gate.dtype != jnp.float32):
        raise ValueError("The gated even polynomial plus exponential requires BF16 or FP32 Q/K/V and FP32 gates")
    if backend == "blocked":
        return _reference(q, k, v, coefficients, exp_gate, return_diagnostics)
    if backend != "fused" or max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("The gated even polynomial plus exponential uses blocked or fused attention (width <= 128)")
    from scaffolds.transformer.attention.backends.fused_even_gate_exp import fused_even_gate_exp
    output = fused_even_gate_exp(q, k, v, coefficients, exp_gate)
    if not return_diagnostics:
        return output
    _, stats = _reference(q, k, v, coefficients, exp_gate, True)
    return output, stats
