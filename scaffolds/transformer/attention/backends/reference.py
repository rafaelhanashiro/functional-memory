"""Blocked causal reference evaluation shared by attention mechanisms."""

import math

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.taylor import _taylor_polynomial, _validate_degree
from scaffolds.transformer.attention.learned_quadratic import _quadratic_polynomial
from scaffolds.transformer.attention.nonnegative_polynomial import square_sum_polynomial, validate_factors
from scaffolds.transformer.attention.rational_polynomial import denominator_factors, select_denominator
from scaffolds.transformer.attention.unnormalized import _learned_polynomial

QUERY_BLOCK_SIZE = 128


def blocked_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    attention_degree: int | None = None,
    attention_normalization: bool = True,
    coefficients: jax.Array | None = None,
    quadratic_mode: str = "fixed",
    polynomial_coefficients: jax.Array | None = None,
    polynomial_square_factors: jax.Array | None = None,
    query_block_size: int = QUERY_BLOCK_SIZE,
    return_diagnostics: bool = False,
    **denominators,
) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    """Attend to the full causal prefix using bounded, recomputed score blocks.

    Inputs are post-RoPE [batch, time, heads, channels]. Polynomial and reduction
    arithmetic is FP32, and the result has V's dtype. The same blocked path is
    used for every degree and the exact-exponential control. No scores are
    shifted in the Taylor branch. A checkpointed scan body prevents retaining
    the [batch, heads, query_block, time] intermediates for every block/layer.

    Invalid causal scores, nonfinite kernels, and invalid normalizers poison
    their rows with NaN. Even Taylor and positive learned kernels must stay
    positive; odd Taylor and unrestricted quadratics permit signed/zero kernels
    and negative nonzero normalizers. Stable exact exponentials may underflow to zero.
    Without normalization, the exponential, fixed Taylor, or learned polynomial
    kernel multiplies V directly; its row sum is measured but never divided by.
    Learned polynomial coefficients ([m+1] or [heads, m+1]) permit signed kernels;
    with normalization they divide by the signed row sum like odd Taylor kernels.
    Square factors ([m+1] or [heads, m+1], even m) give q1^2 + q2^2: nonnegative
    kernel values, exact zeros allowed, and only a positive row sum is valid.
    Denominator coefficients ([d] or [heads, d]) with learned polynomial coefficients
    give the rational kernel p_a(s) / sum_j p_b(s_j), p_b(s) = 1 + sum_r b_r s^r: the
    numerator weights and the denominator sum are signed, and only an exactly zero
    or nonfinite denominator sum is invalid. ``denominators`` holds that one array under
    any name of ``rational_polynomial.DENOMINATOR_KINDS``: coefficients or square
    factors, with the constant fixed at 1 or learned. Its ``kernel_*`` diagnostics describe
    the numerator, ``denominator_*`` the denominator values, and the cancellation
    ratio is sum_j |p_b(s_j)| / |sum_j p_b(s_j)|.
    Diagnostics exclude padded queries and masked future entries.
    """
    _validate_degree(attention_degree)
    if not isinstance(attention_normalization, bool):
        raise ValueError("attention_normalization must be a boolean")
    if not attention_normalization and quadratic_mode != "fixed":
        raise ValueError("Unnormalized attention requires the exponential or a fixed Taylor kernel")
    if quadratic_mode not in ("fixed", "positive", "unrestricted"):
        raise ValueError(f"Unknown quadratic mode: {quadratic_mode}")
    if quadratic_mode != "fixed" and (attention_degree != 2 or coefficients is None):
        raise ValueError("Learned quadratic attention requires degree=2 and coefficients")
    if quadratic_mode == "fixed" and coefficients is not None:
        raise ValueError("Fixed attention does not accept learned coefficients")
    if polynomial_coefficients is not None:
        if quadratic_mode != "fixed":
            raise ValueError("Learned polynomial coefficients exclude the learned quadratic modes")
        if attention_degree is None or attention_degree < 1:
            raise ValueError("Learned polynomial coefficients require an integer attention_degree >= 1")
    nonnegative = polynomial_square_factors is not None
    if nonnegative:
        if polynomial_coefficients is not None or quadratic_mode != "fixed" or not attention_normalization:
            raise ValueError("Square factors define the kernel alone, under row normalization")
        validate_factors(polynomial_square_factors, attention_degree, q.shape[2])
    selected = select_denominator(q.shape[2], **denominators)
    rational = selected is not None
    if rational:
        if polynomial_coefficients is None or not attention_normalization:
            raise ValueError("Denominator coefficients require learned numerator coefficients and row normalization")
        denominator_factor_arrays = denominator_factors(selected[0], selected[1], selected[2])
    signed = polynomial_coefficients is not None or quadratic_mode == "unrestricted" or (
        not nonnegative and quadratic_mode == "fixed"
        and attention_degree is not None and attention_degree % 2 == 1
    )
    extended_diagnostics = quadratic_mode != "fixed" or signed or nonnegative
    if not isinstance(query_block_size, int) or isinstance(query_block_size, bool) or query_block_size <= 0:
        raise ValueError("query_block_size must be a positive integer")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch, time, heads, channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    batch, time, heads, qk_dim = q.shape
    if coefficients is not None and coefficients.shape not in ((2,), (heads, 2)):
        raise ValueError("Quadratic coefficients must have shape (2,) or (heads, 2)")
    if polynomial_coefficients is not None and polynomial_coefficients.shape not in (
            (attention_degree + 1,), (heads, attention_degree + 1)):
        raise ValueError("Polynomial coefficients must have shape (m+1,) or (heads, m+1)")
    if min(batch, time, heads, qk_dim, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    # Avoid excessive padding for short CPU/reference checks; production uses 128.
    block_size = min(query_block_size, time)
    block_count = (time + block_size - 1) // block_size
    pad = block_count * block_size - time
    queries = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
    queries = queries.reshape(batch, block_count, block_size, heads, qk_dim)
    queries = jnp.swapaxes(queries, 0, 1)
    starts = jnp.arange(block_count, dtype=jnp.int32) * block_size
    key_positions = jnp.arange(time)

    def attend_block(_, inputs):
        query, start = inputs
        query_positions = start + jnp.arange(block_size)
        causal = key_positions[None, :] <= query_positions[:, None]
        actual_rows = query_positions < time
        measured = causal & actual_rows[:, None]
        # Preserve BF16 input compute with FP32 dot-product accumulation.
        scores = jnp.einsum(
            "bqhd,bkhd->bhqk", query, k,
            preferred_element_type=jnp.float32,
        ) * (1.0 / math.sqrt(qk_dim))
        if not attention_normalization and attention_degree is None:
            # A row shift changes the unnormalized operator. Underflow to zero
            # is allowed; a nonfinite causal exponential poisons the output.
            kernel = jnp.exp(jnp.where(causal, scores, 0.0))
            invalid_kernel = ~jnp.isfinite(kernel)
        elif attention_degree is None:
            # Shifting is an exact identity only for the exponential control.
            maximum = jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1, keepdims=True)
            safe_scores = jnp.where(causal, scores - maximum, 0.0)
            kernel = jnp.exp(safe_scores)
            invalid_kernel = (~jnp.isfinite(kernel)) | (kernel < 0)
        else:
            # Evaluate masked positions at zero, never at masking sentinels.
            safe_scores = jnp.where(causal, scores, 0.0)
            kernel = (_learned_polynomial(safe_scores, polynomial_coefficients) if polynomial_coefficients is not None
                      else square_sum_polynomial(safe_scores, polynomial_square_factors, attention_degree)
                      if nonnegative
                      else _taylor_polynomial(safe_scores, attention_degree) if quadratic_mode == "fixed"
                      else _quadratic_polynomial(safe_scores, coefficients, quadratic_mode))
            invalid_kernel = ~jnp.isfinite(kernel)
            if nonnegative:
                # A sum of squares may be exactly zero at a common real root; never negative.
                invalid_kernel = invalid_kernel | (kernel < 0)
            elif not signed:
                invalid_kernel = invalid_kernel | (kernel <= 0)
        kernel = jnp.where(causal, kernel, 0.0)
        if rational:
            factor_values = [_learned_polynomial(safe_scores, factor) for factor in denominator_factor_arrays]
            denominator = (factor_values[0] if len(factor_values) == 1
                           else jnp.square(factor_values[0]) + jnp.square(factor_values[1]))
            invalid_kernel = invalid_kernel | ~jnp.isfinite(denominator)
            denominator = jnp.where(causal, denominator, 0.0)
            normalizer = jnp.sum(denominator, axis=-1, keepdims=True, dtype=jnp.float32)
        else:
            denominator = kernel
            normalizer = jnp.sum(kernel, axis=-1, keepdims=True, dtype=jnp.float32)
        invalid_score = ~jnp.isfinite(scores)
        invalid_normalizer = ((~jnp.isfinite(normalizer)) | (
            (normalizer == 0) if signed else (normalizer <= 0)
        )) if attention_normalization else jnp.zeros_like(normalizer, dtype=bool)
        invalid_row = (
            jnp.any(causal & (invalid_score | invalid_kernel), axis=-1, keepdims=True)
            | invalid_normalizer
        )
        weights = kernel / normalizer if attention_normalization else kernel
        output = jnp.einsum(
            "bhqk,bkhd->bqhd", weights, v.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        output = jnp.where(jnp.swapaxes(invalid_row, 1, 2), jnp.nan, output)
        output = output.astype(v.dtype)
        if signed:
            output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        if not return_diagnostics:
            return None, output
        diagnostics = {
            "score_min": jnp.min(jnp.where(measured, scores, jnp.inf)),
            "score_max": jnp.max(jnp.where(measured, scores, -jnp.inf)),
            "kernel_min": jnp.min(jnp.where(measured, kernel, jnp.inf)),
            "kernel_max": jnp.max(jnp.where(measured, kernel, -jnp.inf)),
            "normalizer_min": jnp.min(jnp.where(actual_rows[None, None, :, None], normalizer, jnp.inf)),
            "normalizer_max": jnp.max(jnp.where(actual_rows[None, None, :, None], normalizer, -jnp.inf)),
            "invalid_score_count": jnp.sum(measured & invalid_score, dtype=jnp.int32),
            "invalid_kernel_count": jnp.sum(measured & invalid_kernel, dtype=jnp.int32),
            "invalid_normalizer_count": jnp.sum(actual_rows[None, None, :, None] & invalid_normalizer, dtype=jnp.int32),
        }
        if not attention_normalization:
            # This sum is observed, never divided by. It may be zero or infinite
            # while the weighted output is finite, and must not stop training.
            for suffix in ("min", "max"):
                diagnostics[f"weight_sum_{suffix}"] = diagnostics.pop(f"normalizer_{suffix}")
            diagnostics["invalid_weight_sum_count"] = diagnostics.pop("invalid_normalizer_count")
            diagnostics["invalid_output_count"] = jnp.sum(
                actual_rows[None, :, None, None] & ~jnp.isfinite(output), dtype=jnp.int32,
            )
        if rational:
            diagnostics.update({
                "denominator_min": jnp.min(jnp.where(measured, denominator, jnp.inf)),
                "denominator_max": jnp.max(jnp.where(measured, denominator, -jnp.inf)),
            })
        if extended_diagnostics:
            actual = actual_rows[None, None, :, None]
            cancellation = jnp.sum(jnp.abs(denominator), axis=-1, keepdims=True) / jnp.abs(normalizer)
            diagnostics.update({
                ("min_abs_normalizer" if attention_normalization else "min_abs_weight_sum"):
                    jnp.min(jnp.where(actual, jnp.abs(normalizer), jnp.inf)),
                "negative_weight_count": jnp.sum(measured & (weights < 0), dtype=jnp.int32),
                "negative_kernel_count": jnp.sum(measured & (kernel < 0), dtype=jnp.int32),
                "causal_entry_count": jnp.sum(measured, dtype=jnp.int32) * batch * heads,
                "cancellation_ratio_max": jnp.max(jnp.where(actual, cancellation, -jnp.inf)),
                "cancellation_sum": jnp.sum(jnp.where(actual, cancellation, 0.0)),
                "row_count": jnp.sum(actual_rows, dtype=jnp.int32) * batch * heads,
                "invalid_output_count": jnp.sum(actual_rows[None, :, None, None] & ~jnp.isfinite(output), dtype=jnp.int32),
            })
        return None, (output, jax.lax.stop_gradient(diagnostics))

    _, result = jax.lax.scan(
        jax.checkpoint(attend_block, prevent_cse=False), None, (queries, starts),
    )
    if return_diagnostics:
        output_blocks, per_block = result
        diagnostics = {
            name: (jnp.min(values) if name.endswith("_min") or name.startswith("min_abs_") else
                   jnp.max(values) if name.endswith("_max") else jnp.sum(values))
            for name, values in per_block.items()
        }
        if extended_diagnostics:
            diagnostics["negative_weight_fraction"] = diagnostics["negative_weight_count"] / diagnostics["causal_entry_count"]
            diagnostics["negative_kernel_fraction"] = diagnostics["negative_kernel_count"] / diagnostics["causal_entry_count"]
            diagnostics["cancellation_ratio_mean"] = diagnostics.pop("cancellation_sum") / diagnostics["row_count"]
    else:
        output_blocks = result
    output = jnp.swapaxes(output_blocks, 0, 1).reshape(batch, block_count * block_size, heads, v.shape[-1])
    output = output[:, :time]
    return (output, diagnostics) if return_diagnostics else output
