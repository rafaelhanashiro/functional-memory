"""Blocked causal reference evaluation shared by attention mechanisms.

Two plain-JAX executions of one per-block arithmetic (``_attend_block``):

- ``blocked_dot_product_attention``, the historical scan of 128-query blocks against
  every key. Diagnostics, kernel qualification, and every run that recorded the scan use it.
- ``strip_dot_product_attention``, causal 512-query strips against their key prefixes,
  for training with ``model.attention_blocked_implementation: strips``.

``blocked_backend_attention`` selects between them for the mechanisms.
"""

from functools import partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.taylor import _taylor_polynomial, _validate_degree
from scaffolds.transformer.attention.learned_quadratic import _quadratic_polynomial
from scaffolds.transformer.attention.nonnegative_polynomial import square_sum_polynomial, validate_factors
from scaffolds.transformer.attention.rational_polynomial import denominator_factors, select_denominator
from scaffolds.transformer.attention.unnormalized import _learned_polynomial

QUERY_BLOCK_SIZE = 128
STRIP_SIZE = 512
BLOCKED_IMPLEMENTATIONS = ("scan", "strips")


class _Mechanism(NamedTuple):
    """One call's validated kernel settings and parameter arrays, shared by all of its blocks."""

    attention_degree: int | None
    attention_normalization: bool
    quadratic_mode: str
    coefficients: jax.Array | None
    polynomial_coefficients: jax.Array | None
    polynomial_square_factors: jax.Array | None
    denominator_factor_arrays: tuple[jax.Array, ...] | None
    query_conditioned: bool
    key_conditioned: bool
    nonnegative: bool
    rational: bool
    signed: bool
    extended_diagnostics: bool
    endpoint: bool
    decayed: bool
    per_order: bool
    query_decay_mode: str


def _mechanism(q, k, v, compute, block_size, block_name, *, attention_degree, attention_normalization,
               coefficients, quadratic_mode, polynomial_coefficients, polynomial_square_factors,
               query_polynomial_coefficients, key_polynomial_factors, query_log_decay, query_decay_mode,
               query_gate_input, key_gate_input, gate_log_rate, **denominators) -> _Mechanism:
    """Validate one call's arguments (shared by the scan and the strips) and derive its settings."""
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
    query_conditioned = query_polynomial_coefficients is not None
    if query_conditioned and (attention_normalization or attention_degree is None or attention_degree < 1
                              or polynomial_coefficients is not None or coefficients is not None
                              or polynomial_square_factors is not None or denominators):
        raise ValueError("Query coefficients define an unnormalized polynomial of degree >= 1 alone")
    nonnegative = polynomial_square_factors is not None
    if nonnegative:
        if polynomial_coefficients is not None or quadratic_mode != "fixed" or not attention_normalization:
            raise ValueError("Square factors define the kernel alone, under row normalization")
        validate_factors(polynomial_square_factors, attention_degree, q.shape[2])
    selected = select_denominator(q.shape[2], **denominators)
    rational = selected is not None
    denominator_factor_arrays = None
    if rational:
        if polynomial_coefficients is None or not attention_normalization:
            raise ValueError("Denominator coefficients require learned numerator coefficients and row normalization")
        denominator_factor_arrays = denominator_factors(selected[0], selected[1], selected[2], dtype=compute)
    signed = query_conditioned or polynomial_coefficients is not None or quadratic_mode == "unrestricted" or (
        not nonnegative and quadratic_mode == "fixed"
        and attention_degree is not None and attention_degree % 2 == 1
    )
    extended_diagnostics = quadratic_mode != "fixed" or signed or nonnegative
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError(f"{block_name} must be a positive integer")
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
    if query_conditioned and query_polynomial_coefficients.shape != (batch, time, heads, attention_degree + 1):
        raise ValueError("Query coefficients must have shape [batch,time,heads,m+1]")
    key_conditioned = key_polynomial_factors is not None
    if key_conditioned and (not query_conditioned
                            or key_polynomial_factors.shape != query_polynomial_coefficients.shape):
        raise ValueError("Key factors require query coefficients of the same shape [batch,time,heads,m+1]")
    endpoint = any(value is not None for value in (query_gate_input, key_gate_input, gate_log_rate))
    per_order = ((query_log_decay is not None and query_log_decay.ndim == 4)
                 or (query_gate_input is not None and query_gate_input.ndim == 4))
    if per_order and (not query_conditioned or query_decay_mode != "query_distance"):
        raise ValueError("Per-order decay requires query-distance polynomial attention")
    gate_shape = (batch, time, heads, attention_degree + 1) if per_order else (batch, time, heads)
    rate_shape = (heads, attention_degree + 1, 1) if per_order else (heads, 1)
    if endpoint and (any(value is None for value in (query_gate_input, key_gate_input, gate_log_rate))
                     or query_log_decay is not None or query_decay_mode != "query_distance"):
        raise ValueError("Endpoint decay requires all three operands and query-distance mode without log decay")
    if endpoint and (query_gate_input.shape != gate_shape
                     or key_gate_input.shape != gate_shape or gate_log_rate.shape != rate_shape):
        raise ValueError("Endpoint inputs require [batch,time,heads] and log rate [heads,1]")
    decayed = query_log_decay is not None or endpoint
    if query_decay_mode not in ("cumulative", "query_distance"):
        raise ValueError(f"Unknown query decay mode: {query_decay_mode}")
    if query_decay_mode != "cumulative" and not decayed:
        raise ValueError("A query-distance decay mode requires query_log_decay")
    exponential = attention_degree is None and quadratic_mode == "fixed"
    if decayed and (key_conditioned or not (query_conditioned or exponential)
                    or (not endpoint and query_log_decay.shape != gate_shape)):
        raise ValueError("A forget gate's log decay [batch,time,heads] requires query coefficients alone, "
                         "softmax, or the unnormalized exponential")
    if endpoint and not query_conditioned:
        raise ValueError("Endpoint decay requires query coefficients")
    if query_decay_mode != "cumulative" and not query_conditioned:
        raise ValueError("The query-distance decay mode requires query coefficients; softmax and the "
                         "exponential take the cumulative decay")
    if min(batch, time, heads, qk_dim, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    return _Mechanism(
        attention_degree=attention_degree, attention_normalization=attention_normalization,
        quadratic_mode=quadratic_mode, coefficients=coefficients, polynomial_coefficients=polynomial_coefficients,
        polynomial_square_factors=polynomial_square_factors, denominator_factor_arrays=denominator_factor_arrays,
        query_conditioned=query_conditioned, key_conditioned=key_conditioned, nonnegative=nonnegative,
        rational=rational, signed=signed, extended_diagnostics=extended_diagnostics, endpoint=endpoint,
        decayed=decayed, per_order=per_order, query_decay_mode=query_decay_mode,
    )


def _attend_block(mechanism, query, start, rows, *, block_size, time, key_positions, k, v, query_coefficients,
                  key_factors, query_gate_input, key_gate, gate_log_rate, gate_rate, query_log_decay,
                  key_log_decay, compute, output_dtype, return_diagnostics):
    """The query block [start, start + block_size) against the keys at ``key_positions``.

    ``query`` holds the block's queries and ``rows(array)`` takes its rows of a query-side
    [batch, time, ...] array (query coefficients, gate inputs, log decay). ``k``, ``v`` and
    the per-key operands (key factors, key gate [batch, heads, 1, keys], key log decay) cover
    ``key_positions``: every key in the scan, the causal prefix in a strip. Arithmetic is in
    ``compute``, FP32 in every model; the output has ``output_dtype``, V's dtype.
    """
    m = mechanism
    batch, _, heads, qk_dim = query.shape
    query_positions = start + jnp.arange(block_size)
    causal = key_positions[None, :] <= query_positions[:, None]
    actual_rows = query_positions < time
    measured = causal & actual_rows[:, None]
    # Preserve BF16 input compute with FP32 dot-product accumulation.
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk", query, k,
        preferred_element_type=compute,
    ) * (1.0 / math.sqrt(qk_dim))
    if m.decayed and not m.per_order:
        if m.endpoint:
            query_input = rows(query_gate_input)
            query_input = jnp.swapaxes(query_input, 1, 2)[..., None]
            valid_gate_inputs = (jnp.isfinite(query_input) & jnp.isfinite(key_gate)
                                 & jnp.isfinite(gate_log_rate[:, 0])[None, :, None, None])
            query_decay = -gate_rate * jax.nn.softplus(query_input + key_gate)
            query_decay = jnp.where(valid_gate_inputs, query_decay, jnp.nan)
        else:
            query_decay = rows(query_log_decay)
            query_decay = jnp.swapaxes(query_decay, 1, 2)[..., None]
        if m.query_decay_mode == "cumulative":
            # prod_(j=i+1..t) alpha_j = exp(L_t - L_i), never above one for causal pairs.
            gap = jnp.where(causal, query_decay - jnp.swapaxes(key_log_decay, 1, 2)[:, :, None, :], 0.0)
        else:
            # The current query chooses a single rate for its entire prefix. Never
            # exponentiate positive gaps for masked future keys; the diagonal is exp(0).
            distance = jnp.maximum(query_positions[:, None] - key_positions[None, :], 0)
            valid_decay = jnp.isfinite(query_decay) & (query_decay <= 0)
            safe_decay = jnp.where(valid_decay, query_decay, 0.0)
            gap = safe_decay * distance.astype(compute)
    # The exponential kernels see the decayed logits s_ti + L_t - L_i.
    logits = scores + gap if m.decayed and m.attention_degree is None else scores
    if not m.attention_normalization and m.attention_degree is None:
        # A row shift changes the unnormalized operator. Underflow to zero
        # is allowed; a nonfinite causal exponential poisons the output.
        kernel = jnp.exp(jnp.where(causal, logits, 0.0))
        invalid_kernel = ~jnp.isfinite(kernel)
    elif m.attention_degree is None:
        # Shifting is an exact identity only for the exponential control.
        maximum = jnp.max(jnp.where(causal, logits, -jnp.inf), axis=-1, keepdims=True)
        safe_scores = jnp.where(causal, logits - maximum, 0.0)
        kernel = jnp.exp(safe_scores)
        invalid_kernel = (~jnp.isfinite(kernel)) | (kernel < 0)
    else:
        # Evaluate masked positions at zero, never at masking sentinels.
        safe_scores = jnp.where(causal, scores, 0.0)
        if m.query_conditioned:
            block_coefficients = rows(query_coefficients)
            if m.per_order:
                # Slice the gate operands once for every order, so their transposes combine all
                # order gradients locally before updating the full query buffer. Repeating the
                # slice per order produces nested updates that XLA can fuse into a GPU kernel
                # whose device stack overflows.
                order_inputs = rows(query_gate_input if m.endpoint else query_log_decay)
                distance = jnp.maximum(query_positions[:, None] - key_positions[None, :], 0).astype(compute)

            def order_log_alpha(order):
                # Only one [B,H,Q,K] rate block at a time; no full pair-by-order tensor.
                projected = jnp.swapaxes(order_inputs[..., order], 1, 2)[..., None]
                if not m.endpoint:
                    return projected
                key_projected = key_gate[..., order]
                valid = (jnp.isfinite(projected) & jnp.isfinite(key_projected)
                         & jnp.isfinite(gate_log_rate[:, order, 0])[None, :, None, None])
                alpha = -gate_rate[..., order] * jax.nn.softplus(projected + key_projected)
                return jnp.where(valid, alpha, jnp.nan)

            def pair_coefficient(order):
                # [b, h, q, 1] per query; a key factor [b, h, 1, k] scales each key's degree-r term.
                coefficient = jnp.swapaxes(block_coefficients[..., order], 1, 2)[..., None]
                if m.key_conditioned:
                    coefficient = coefficient * jnp.swapaxes(key_factors[..., order], 1, 2)[:, :, None, :]
                if m.per_order:
                    # Each order's own query-distance decay, before the polynomial sum.
                    alpha = order_log_alpha(order)
                    valid = jnp.isfinite(alpha) & (alpha <= 0)
                    factor = jnp.exp(jnp.where(valid, alpha, 0.0) * distance)
                    coefficient = jnp.where(valid, coefficient * factor, jnp.nan)
                return coefficient

            kernel = jnp.broadcast_to(pair_coefficient(m.attention_degree), safe_scores.shape)
            for order in range(m.attention_degree - 1, -1, -1):
                kernel = kernel * safe_scores + pair_coefficient(order)
            if m.decayed and not m.per_order:
                kernel = kernel * jnp.exp(gap)
                if m.query_decay_mode == "query_distance":
                    kernel = jnp.where(valid_decay, kernel, jnp.nan)
        else:
            kernel = (_learned_polynomial(safe_scores, m.polynomial_coefficients, dtype=compute)
                      if m.polynomial_coefficients is not None
                      else square_sum_polynomial(safe_scores, m.polynomial_square_factors, m.attention_degree,
                                                 dtype=compute)
                      if m.nonnegative
                      else _taylor_polynomial(safe_scores, m.attention_degree) if m.quadratic_mode == "fixed"
                      else _quadratic_polynomial(safe_scores, m.coefficients, m.quadratic_mode, dtype=compute))
        invalid_kernel = ~jnp.isfinite(kernel)
        if m.nonnegative:
            # A sum of squares may be exactly zero at a common real root; never negative.
            invalid_kernel = invalid_kernel | (kernel < 0)
        elif not m.signed:
            invalid_kernel = invalid_kernel | (kernel <= 0)
    kernel = jnp.where(causal, kernel, 0.0)
    if m.rational:
        factor_values = [_learned_polynomial(safe_scores, factor, dtype=compute)
                         for factor in m.denominator_factor_arrays]
        denominator = (factor_values[0] if len(factor_values) == 1
                       else jnp.square(factor_values[0]) + jnp.square(factor_values[1]))
        invalid_kernel = invalid_kernel | ~jnp.isfinite(denominator)
        denominator = jnp.where(causal, denominator, 0.0)
        normalizer = jnp.sum(denominator, axis=-1, keepdims=True, dtype=compute)
    else:
        denominator = kernel
        normalizer = jnp.sum(kernel, axis=-1, keepdims=True, dtype=compute)
    invalid_score = ~jnp.isfinite(scores)
    invalid_normalizer = ((~jnp.isfinite(normalizer)) | (
        (normalizer == 0) if m.signed else (normalizer <= 0)
    )) if m.attention_normalization else jnp.zeros_like(normalizer, dtype=bool)
    invalid_row = (
        jnp.any(causal & (invalid_score | invalid_kernel), axis=-1, keepdims=True)
        | invalid_normalizer
    )
    weights = kernel / normalizer if m.attention_normalization else kernel
    output = jnp.einsum(
        "bhqk,bkhd->bqhd", weights, v.astype(compute),
        preferred_element_type=compute,
    )
    output = jnp.where(jnp.swapaxes(invalid_row, 1, 2), jnp.nan, output)
    output = output.astype(output_dtype)
    if m.signed:
        output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
    if not return_diagnostics:
        return output
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
    if m.endpoint and m.per_order:
        summaries = [
            (jnp.sum(jnp.where(measured, order_log_alpha(r), 0.0), axis=(0, 2, 3)),
             jnp.min(jnp.where(measured, order_log_alpha(r), jnp.inf), axis=(0, 2, 3)),
             jnp.max(jnp.where(measured, order_log_alpha(r), -jnp.inf), axis=(0, 2, 3)))
            for r in range(m.attention_degree + 1)]
        diagnostics.update({f"forget_gate_log_alpha_{name}": jnp.stack(
            [summary[index] for summary in summaries], axis=-1)
            for index, name in enumerate(("sum", "min", "max"))})
    elif m.endpoint:
        diagnostics.update({
            "forget_gate_log_alpha_sum": jnp.sum(jnp.where(measured, query_decay, 0.0), axis=(0, 2, 3)),
            "forget_gate_log_alpha_min": jnp.min(jnp.where(measured, query_decay, jnp.inf), axis=(0, 2, 3)),
            "forget_gate_log_alpha_max": jnp.max(jnp.where(measured, query_decay, -jnp.inf), axis=(0, 2, 3)),
        })
    if not m.attention_normalization:
        # This sum is observed, never divided by. It may be zero or infinite
        # while the weighted output is finite, and must not stop training.
        for suffix in ("min", "max"):
            diagnostics[f"weight_sum_{suffix}"] = diagnostics.pop(f"normalizer_{suffix}")
        diagnostics["invalid_weight_sum_count"] = diagnostics.pop("invalid_normalizer_count")
        diagnostics["invalid_output_count"] = jnp.sum(
            actual_rows[None, :, None, None] & ~jnp.isfinite(output), dtype=jnp.int32,
        )
    if m.rational:
        diagnostics.update({
            "denominator_min": jnp.min(jnp.where(measured, denominator, jnp.inf)),
            "denominator_max": jnp.max(jnp.where(measured, denominator, -jnp.inf)),
        })
    if m.extended_diagnostics:
        actual = actual_rows[None, None, :, None]
        cancellation = jnp.sum(jnp.abs(denominator), axis=-1, keepdims=True) / jnp.abs(normalizer)
        diagnostics.update({
            ("min_abs_normalizer" if m.attention_normalization else "min_abs_weight_sum"):
                jnp.min(jnp.where(actual, jnp.abs(normalizer), jnp.inf)),
            "negative_weight_count": jnp.sum(measured & (weights < 0), dtype=jnp.int32),
            "negative_kernel_count": jnp.sum(measured & (kernel < 0), dtype=jnp.int32),
            "causal_entry_count": jnp.sum(measured, dtype=jnp.int32) * batch * heads,
            "cancellation_ratio_max": jnp.max(jnp.where(actual, cancellation, -jnp.inf)),
            "cancellation_sum": jnp.sum(jnp.where(actual, cancellation, 0.0)),
            "row_count": jnp.sum(actual_rows, dtype=jnp.int32) * batch * heads,
            "invalid_output_count": jnp.sum(actual_rows[None, :, None, None] & ~jnp.isfinite(output), dtype=jnp.int32),
        })
    return output, jax.lax.stop_gradient(diagnostics)


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
    query_polynomial_coefficients: jax.Array | None = None,
    key_polynomial_factors: jax.Array | None = None,
    query_log_decay: jax.Array | None = None,
    query_decay_mode: str = "cumulative",
    query_gate_input: jax.Array | None = None,
    key_gate_input: jax.Array | None = None,
    gate_log_rate: jax.Array | None = None,
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
    Query coefficients [batch,time,heads,m+1] instead define one unnormalized
    polynomial of degree m>=1 for each query and head; key factors of the same shape
    multiply each key's degree-r term, giving the pair coefficient c_r(t) f_r(i).
    A forget gate's cumulative log decay L [batch,time,heads] (FP32, nonincreasing in time)
    multiplies each query-coefficient weight by exp(L_t - L_i), evaluated as one difference,
    and adds L_t - L_i to every softmax or unnormalized exponential score (the Forgetting
    Transformer); score diagnostics describe the scores before the decay. With query
    coefficients and ``query_decay_mode="query_distance"``, the same array holds raw FP32
    log alpha_t instead: each polynomial weight is multiplied by exp((t-i)*log alpha_t). Gates
    from intervening tokens are unused. A nonfinite or positive raw log alpha poisons its query.
    Alternatively, endpoint operands query_gate_input/key_gate_input [B,T,H] and gate_log_rate [H,1]
    replace query_log_decay and yield -exp(gate_log_rate) softplus(query_input_t + key_input_i).
    Only token projections are stored; pair rates and their causal diagnostics are computed in blocks.
    Per-order query-distance gates add an R=m+1 axis to those operands (rates [H,R,1]):
    each coefficient term gets its own decay before the polynomial sum. Query-only raw log
    alpha may likewise be [B,T,H,R]. Parameters and diagnostics retain order zero.
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
    return _blocked_scan(
        q, k, v, compute=jnp.float32, attention_degree=attention_degree,
        attention_normalization=attention_normalization, coefficients=coefficients, quadratic_mode=quadratic_mode,
        polynomial_coefficients=polynomial_coefficients, polynomial_square_factors=polynomial_square_factors,
        query_polynomial_coefficients=query_polynomial_coefficients, key_polynomial_factors=key_polynomial_factors,
        query_log_decay=query_log_decay, query_decay_mode=query_decay_mode, query_gate_input=query_gate_input,
        key_gate_input=key_gate_input, gate_log_rate=gate_log_rate, query_block_size=query_block_size,
        return_diagnostics=return_diagnostics, **denominators,
    )


def _blocked_scan(q, k, v, *, compute, attention_degree=None, attention_normalization=True, coefficients=None,
                  quadratic_mode="fixed", polynomial_coefficients=None, polynomial_square_factors=None,
                  query_polynomial_coefficients=None, key_polynomial_factors=None, query_log_decay=None,
                  query_decay_mode="cumulative", query_gate_input=None, key_gate_input=None, gate_log_rate=None,
                  query_block_size=QUERY_BLOCK_SIZE, return_diagnostics=False, **denominators):
    """blocked_dot_product_attention with its arithmetic in ``compute``: FP32 there, FP64 only
    in exact-arithmetic tests of the strips against the scan."""
    m = _mechanism(
        q, k, v, compute, query_block_size, "query_block_size", attention_degree=attention_degree,
        attention_normalization=attention_normalization, coefficients=coefficients, quadratic_mode=quadratic_mode,
        polynomial_coefficients=polynomial_coefficients, polynomial_square_factors=polynomial_square_factors,
        query_polynomial_coefficients=query_polynomial_coefficients, key_polynomial_factors=key_polynomial_factors,
        query_log_decay=query_log_decay, query_decay_mode=query_decay_mode, query_gate_input=query_gate_input,
        key_gate_input=key_gate_input, gate_log_rate=gate_log_rate, **denominators,
    )
    batch, time, heads, qk_dim = q.shape
    # Avoid excessive padding for short CPU/reference checks; production uses 128.
    block_size = min(query_block_size, time)
    block_count = (time + block_size - 1) // block_size
    pad = block_count * block_size - time
    queries = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
    queries = queries.reshape(batch, block_count, block_size, heads, qk_dim)
    queries = jnp.swapaxes(queries, 0, 1)
    padded_coefficients = key_factors = padded_query_gate_input = key_gate = gate_rate = None
    log_decay = padded_log_decay = None
    if m.query_conditioned:
        padded_coefficients = jnp.pad(query_polynomial_coefficients.astype(compute),
                                      ((0, 0), (0, pad), (0, 0), (0, 0)))
        key_factors = key_polynomial_factors.astype(compute) if m.key_conditioned else None
    if m.endpoint:
        padded_query_gate_input = jnp.pad(query_gate_input.astype(compute),
                                          ((0, 0), (0, pad), (0, 0)) + (((0, 0),) if m.per_order else ()),
                                          mode="edge")
        key_gate = jnp.swapaxes(key_gate_input.astype(compute), 1, 2)[:, :, None, :]
        gate_rate = jnp.exp(gate_log_rate[..., 0].astype(compute))[None, :, None, None]
    elif m.decayed:
        log_decay = query_log_decay.astype(compute)
        # Repeat the final gate on padded queries; these rows never enter the output or diagnostics.
        padded_log_decay = jnp.pad(log_decay, ((0, 0), (0, pad), (0, 0))
                                   + (((0, 0),) if m.per_order else ()), mode="edge")
    starts = jnp.arange(block_count, dtype=jnp.int32) * block_size
    key_positions = jnp.arange(time)

    def attend_block(_, inputs):
        query, start = inputs
        return None, _attend_block(
            m, query, start, lambda array: jax.lax.dynamic_slice_in_dim(array, start, block_size, axis=1),
            block_size=block_size, time=time, key_positions=key_positions, k=k, v=v,
            query_coefficients=padded_coefficients, key_factors=key_factors,
            query_gate_input=padded_query_gate_input, key_gate=key_gate, gate_log_rate=gate_log_rate,
            gate_rate=gate_rate, query_log_decay=padded_log_decay, key_log_decay=log_decay,
            compute=compute, output_dtype=v.dtype, return_diagnostics=return_diagnostics,
        )

    _, result = jax.lax.scan(
        jax.checkpoint(attend_block, prevent_cse=False), None, (queries, starts),
    )
    if return_diagnostics:
        output_blocks, per_block = result
        diagnostics = {
            name: (jnp.min(values, axis=0 if name.startswith("forget_gate_") else None)
                   if name.endswith("_min") or name.startswith("min_abs_") else
                   jnp.max(values, axis=0 if name.startswith("forget_gate_") else None)
                   if name.endswith("_max") else
                   jnp.sum(values, axis=0 if name.startswith("forget_gate_") else None))
            for name, values in per_block.items()
        }
        if m.endpoint:
            diagnostics["forget_gate_log_alpha_mean"] = diagnostics.pop("forget_gate_log_alpha_sum") / (batch * time * (time + 1) / 2)
        if m.extended_diagnostics:
            diagnostics["negative_weight_fraction"] = diagnostics["negative_weight_count"] / diagnostics["causal_entry_count"]
            diagnostics["negative_kernel_fraction"] = diagnostics["negative_kernel_count"] / diagnostics["causal_entry_count"]
            diagnostics["cancellation_ratio_mean"] = diagnostics.pop("cancellation_sum") / diagnostics["row_count"]
    else:
        output_blocks = result
    output = jnp.swapaxes(output_blocks, 0, 1).reshape(batch, block_count * block_size, heads, v.shape[-1])
    output = output[:, :time]
    return (output, diagnostics) if return_diagnostics else output


def _compute_dtype(*arrays):
    """FP32 for BF16 and FP32 operands, as the scan; FP64 for FP64 operands (exact-arithmetic tests)."""
    return jnp.promote_types(jnp.result_type(*arrays), jnp.float32)


def _rounded_to(x, dtype):
    """``x`` rounded to ``dtype``'s precision and kept in x's dtype; ``x`` itself otherwise.

    The strips write every rounding to BF16 that may meet an upcast this way, so that it happens.
    With the default xla_allow_excess_precision, XLA GPU removes a conversion pair f32 -> bf16 ->
    f32 when the second consumes the first: a BF16 operand that the model rounded from FP32 math
    (QK RMSNorm, RoPE) would reach the strips' products unrounded, unlike the scan's BF16 loop
    operands, and the forward would differ from the backward's recomputation from the saved BF16
    value. reduce_precision, which XLA keeps, rounds exactly as the conversion when ``dtype``
    shares x's exponent range (BF16 and FP32), subnormals included; it is the identity on values
    already at that precision, and its transpose rounds cotangents the same way. Types with a
    narrower exponent range are left to the plain conversion.
    """
    source, target = jnp.finfo(x.dtype), jnp.finfo(dtype)
    if target.nmant >= source.nmant or target.nexp != source.nexp:
        return x
    return jax.lax.reduce_precision(x, exponent_bits=target.nexp, mantissa_bits=target.nmant)


def _exact_upcast(x, dtype):
    """``x`` in ``dtype``, explicitly kept at its own precision (see _rounded_to)."""
    return _rounded_to(x.astype(dtype), x.dtype)


def strip_dot_product_attention(
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
    query_polynomial_coefficients: jax.Array | None = None,
    key_polynomial_factors: jax.Array | None = None,
    query_log_decay: jax.Array | None = None,
    query_decay_mode: str = "cumulative",
    query_gate_input: jax.Array | None = None,
    key_gate_input: jax.Array | None = None,
    gate_log_rate: jax.Array | None = None,
    strip_size: int = STRIP_SIZE,
    **denominators,
) -> jax.Array:
    """blocked_dot_product_attention's mathematics over causal query strips, for training.

    Every argument, kernel, normalization, and invalid-row rule is the scan's: each strip of
    queries [s, e) (the last may be shorter; one strip when T <= strip_size) evaluates the
    scan's per-block arithmetic, the same code, against the causal key prefix [0, e) alone.
    Each strip runs under ``jax.checkpoint``, so the backward recomputes it and no
    [strip x keys] intermediate outlives it. K, V, and the other per-key operands are
    converted once, before the strips, to the compute dtype (FP32 for BF16 or FP32 inputs,
    FP64 for FP64): autodiff sums their strips' cotangents there and rounds once to the input
    dtype, where the scan sums its blocks' in the input dtype (BF16 in training). Queries are
    converted per strip and saved in their own dtype, since dQ has no sum across strips. Every
    conversion of a BF16 operand, and of the output back to V's dtype, keeps its rounding
    explicitly (_rounded_to), which XLA GPU's excess-precision simplification would otherwise
    remove next to FP32 producers or consumers. The products therefore read the BF16 values,
    exact in FP32 (and TF32), with FP32 accumulation, in the forward and in the recomputation
    alike. Weights stay FP32.

    One intended difference: masked keys are never multiplied, so a nonfinite V entry
    poisons the rows from the start of its strip onward. The scan multiplies every key's V by
    each block's weights, and a masked zero weight times NaN or inf poisons every row.
    Sequence lengths are static. There are no diagnostics: ``blocked_backend_attention``
    takes them from blocked_dot_product_attention (forward only), as the fused kernels do.
    """
    compute = _compute_dtype(q, k, v)
    m = _mechanism(
        q, k, v, compute, strip_size, "strip_size", attention_degree=attention_degree,
        attention_normalization=attention_normalization, coefficients=coefficients, quadratic_mode=quadratic_mode,
        polynomial_coefficients=polynomial_coefficients, polynomial_square_factors=polynomial_square_factors,
        query_polynomial_coefficients=query_polynomial_coefficients, key_polynomial_factors=key_polynomial_factors,
        query_log_decay=query_log_decay, query_decay_mode=query_decay_mode, query_gate_input=query_gate_input,
        key_gate_input=key_gate_input, gate_log_rate=gate_log_rate, **denominators,
    )
    time = q.shape[1]
    # The strips slice these full arrays inside their checkpoints, so the residuals each layer
    # keeps for its backward are these arrays alone, never a copy per strip. Every upcast keeps
    # its operand's rounding explicitly (_exact_upcast). Queries are upcast inside each strip,
    # so the saved queries keep their own dtype.
    operands = {"query": q, "k": _exact_upcast(k, compute), "v": _exact_upcast(v, compute)}
    if m.query_conditioned:
        operands["query_coefficients"] = _exact_upcast(query_polynomial_coefficients, compute)
        if m.key_conditioned:
            operands["key_factors"] = _exact_upcast(key_polynomial_factors, compute)
    if m.endpoint:
        operands["query_gate_input"] = _exact_upcast(query_gate_input, compute)
        operands["key_gate"] = jnp.swapaxes(_exact_upcast(key_gate_input, compute), 1, 2)[:, :, None, :]
        operands["gate_rate"] = jnp.exp(_exact_upcast(gate_log_rate[..., 0], compute))[None, :, None, None]
    elif m.decayed:
        operands["log_decay"] = _exact_upcast(query_log_decay, compute)

    def attend_strip(arrays, *, start, end):
        def prefix(name, axis=1):
            return None if name not in arrays else jax.lax.slice_in_dim(arrays[name], 0, end, axis=axis)

        output = _attend_block(
            m, _exact_upcast(arrays["query"][:, start:end], compute), start, lambda array: array[:, start:end],
            block_size=end - start, time=time, key_positions=jnp.arange(end), k=prefix("k"), v=prefix("v"),
            query_coefficients=arrays.get("query_coefficients"), key_factors=prefix("key_factors"),
            query_gate_input=arrays.get("query_gate_input"), key_gate=prefix("key_gate", axis=3),
            gate_log_rate=gate_log_rate, gate_rate=arrays.get("gate_rate"),
            query_log_decay=arrays.get("log_decay"), key_log_decay=prefix("log_decay"),
            compute=compute, output_dtype=compute, return_diagnostics=False,
        )
        # The block's conversion to V's dtype, rounded explicitly: a consumer that upcasts the
        # output (head RMSNorm) would otherwise let XLA drop it. A signed kernel's row with an
        # entry that overflows V's dtype is then poisoned, as in the block.
        output = _rounded_to(output, v.dtype).astype(v.dtype)
        if m.signed:
            output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        return output

    # prevent_cse stays True: unlike the scan's loop body, unrolled strips would otherwise let
    # XLA merge each recomputation with its forward and keep every strip's scores alive.
    outputs = [jax.checkpoint(partial(attend_strip, start=start, end=min(start + strip_size, time)))(operands)
               for start in range(0, time, strip_size)]
    return jnp.concatenate(outputs, axis=1)


def blocked_backend_attention(q, k, v, *, implementation: str = "scan", return_diagnostics: bool = False,
                              **arguments):
    """Attention on the blocked backend with the recipe's implementation.

    ``"scan"`` is blocked_dot_product_attention, unchanged. ``"strips"`` takes the output from
    strip_dot_product_attention and any diagnostics from blocked_dot_product_attention,
    forward only, as the fused kernels do.
    """
    if implementation not in BLOCKED_IMPLEMENTATIONS:
        raise ValueError(f"Unknown blocked attention implementation: {implementation}")
    if implementation == "scan":
        return blocked_dot_product_attention(q, k, v, return_diagnostics=return_diagnostics, **arguments)
    output = strip_dot_product_attention(q, k, v, **arguments)
    if not return_diagnostics:
        return output
    _, diagnostics = blocked_dot_product_attention(q, k, v, return_diagnostics=True, **arguments)
    return output, diagnostics
