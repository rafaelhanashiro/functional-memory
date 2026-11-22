"""Per-query polynomials p_t(z) = sum_r c_tr z^r of degree M, z = s or softplus(s), in common row units e^(-m_t).

m_t = M log max(1, max_i |z_ti|) over query t's causal row bounds every term: with half = e^(-m_t / (2M)) and
u = (z half) half, |u| <= 1 and c_r z^r e^(-m_t) = c_r e^(-(M - r) m_t / M) u^r. Head RMSNorm divides the row factor
out when its epsilon is scaled by e^(-2 m_t) (attention/output_scaling.py); m_t is a constant to differentiation.
The query sum-of-squares generator supplies c (attention/nonnegative_polynomial.py), prefactor included.
"""
import math

import jax
from jax import lax
import jax.numpy as jnp


TRANSFORMS = ("identity", "softplus")


def transformed(scores, transform):
    """z = phi(s) in FP32."""
    if transform == "identity":
        return scores
    if transform == "softplus":
        return jax.nn.softplus(scores)
    raise ValueError(f"The shifted polynomial takes the transforms {TRANSFORMS}, not {transform!r}")


def row_shift(z, degree):
    """M log max(1, max |z|) over the last axis, the log of the row's largest power. Entries of magnitude at most one
    (masked ones, say) leave it unchanged."""
    return lax.stop_gradient(degree * jnp.log(jnp.maximum(1.0, jnp.max(jnp.abs(z), axis=-1))))


def shifted_terms(z, shift, degree):
    """e^-shift z^r for r = 0..M as u^r e^(-(M - r) shift / M); shift broadcasts against z. Applying the scale in two
    halves keeps it normal where its reciprocal would not be."""
    half = jnp.exp(shift * (-0.5 / degree))
    u = (z * half) * half
    return [(u ** r) * (1.0 if r == degree else jnp.exp(shift * (-(degree - r) / degree))) for r in range(degree + 1)]


def _reference(q, k, v, c, transform, return_diagnostics):
    """Evaluate complete causal rows in small batches; every diagnostic uses the same shifted weights."""
    batch, time, heads, width = q.shape
    degree = c.shape[-1] - 1
    block = min(32, time)
    positions = jnp.arange(time)

    def rows(block_index):
        query_positions = block * block_index + jnp.arange(block)
        index = jnp.minimum(query_positions, time - 1)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q[:, index], k, preferred_element_type=jnp.float32,
                            precision=lax.Precision.HIGHEST) * (1.0 / math.sqrt(width))
        causal = positions[None, None, None, :] <= index[None, None, :, None]
        scores = jnp.where(causal, scores, 0.0)
        # A masked score is zero, so its z (0, or log 2 for softplus) is below the shift's floor of one.
        z = transformed(scores, transform)
        shift = row_shift(z, degree)
        coefficients = c[:, index].swapaxes(1, 2)
        terms = shifted_terms(z, shift[..., None], degree)
        weights = sum(coefficients[..., r, None] * term for r, term in enumerate(terms))
        weights = jnp.where(causal, weights, 0.0)
        output = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32), precision=lax.Precision.HIGHEST)
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
                 "polynomial_shift_min": min_measured(shift, actual),
                 "polynomial_shift_max": max_measured(shift, actual),
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
    # A sum of squares, with a positive prefactor if any, has no negative summand to cancel.
    stats["cancellation_ratio_mean"] = stats["cancellation_ratio_max"] = jnp.float32(1.0)
    from scaffolds.transformer.attention.input_dependent import coefficient_statistics
    stats.update(coefficient_statistics(c))
    return output, lax.stop_gradient(stats)


def shifted_polynomial_attention(q, k, v, coefficients, *, transform="identity", backend="blocked",
                                 return_diagnostics=False):
    """Return the shifted output e^(-m) o and the stopped row shifts m [B,T,H], which head RMSNorm takes."""
    if (q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]
            or min(*q.shape, v.shape[-1]) <= 0 or coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3]
            or coefficients.shape[-1] < 2):
        raise ValueError("Expected Q/K/V [B,T,H,D] and query coefficients [B,T,H,M+1] with M >= 1")
    if (q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16)
            or coefficients.dtype != jnp.float32):
        raise ValueError("The shifted polynomial requires BF16 or FP32 Q/K/V and FP32 coefficients")
    if transform not in TRANSFORMS:
        raise ValueError(f"The shifted polynomial takes the transforms {TRANSFORMS}, not {transform!r}")
    if backend == "blocked":
        return _reference(q, k, v, coefficients, transform, return_diagnostics)
    if backend != "fused" or max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("The shifted polynomial uses its blocked reference or fused kernel (width <= 128)")
    from scaffolds.transformer.attention.backends.fused_shifted_polynomial import fused_shifted_polynomial
    output = fused_shifted_polynomial(q, k, v, coefficients, transform=transform)
    if not return_diagnostics:
        return output
    _, stats = _reference(q, k, v, coefficients, transform, True)
    return output, stats
