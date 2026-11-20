"""Unnormalized causal k(QK^T / sqrt(d_k)) V: exp, its Taylor truncation, or a learned polynomial."""

from fractions import Fraction
import math

import jax
import jax.numpy as jnp


def initial_coefficients(degree: int, initialization: str = "taylor") -> tuple[float, ...]:
    """Initial c_0..c_m: the Taylor values 1/r!, linear attention (c_1 = 1, every other 0),
    or the truncated Hermite expansion exp(1/2) sum_{n<=m} He_n(s) / n! in monomials."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("Learned polynomial coefficients require an integer degree >= 1")
    if initialization == "taylor":
        return tuple(1.0 / math.factorial(order) for order in range(degree + 1))
    if initialization == "linear":
        return tuple(1.0 if order == 1 else 0.0 for order in range(degree + 1))
    if initialization == "hermite":
        previous, current = [1] + [0] * degree, [0, 1] + [0] * (degree - 1)  # He_0 and He_1, ascending powers
        total = [Fraction(previous[power]) + current[power] for power in range(degree + 1)]
        for order in range(2, degree + 1):
            # He_order = s He_(order-1) - (order-1) He_(order-2)
            previous, current = current, [(current[power - 1] if power else 0) - (order - 1) * previous[power]
                                          for power in range(degree + 1)]
            total = [value + Fraction(term, math.factorial(order)) for value, term in zip(total, current)]
        return tuple(math.exp(0.5) * float(value) for value in total)
    raise ValueError(f"Unknown coefficient initialization: {initialization}")


def parameter_specs(*, degree: int, sharing: str, initialization: str = "taylor"):
    """Learned coefficients c_0..c_m of sum_r c_r s^r (see ``initial_coefficients``)."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (AttentionParameterSpec(
        name="polynomial_coefficients", argument="polynomial_coefficients",
        initial=initial_coefficients(degree, initialization), sharing=sharing, weight_decay=False,
    ),)


def _learned_polynomial(scores: jax.Array, coefficients: jax.Array, *, dtype=jnp.float32) -> jax.Array:
    """Horner evaluation of per-head coefficients [heads, m+1] over [batch, heads, queries, keys] scores.

    Coefficients enter in ``dtype``: FP32 in every model, FP64 only in exact-arithmetic tests."""
    coefficients = jnp.broadcast_to(coefficients, (scores.shape[1], coefficients.shape[-1])).astype(dtype)
    value = jnp.broadcast_to(coefficients[:, -1][None, :, None, None], scores.shape)
    for order in range(coefficients.shape[-1] - 2, -1, -1):
        value = value * scores + coefficients[:, order][None, :, None, None]
    return value


def _validate_coefficients(coefficients, attention_degree, heads):
    if isinstance(attention_degree, bool) or not isinstance(attention_degree, int) or attention_degree < 1:
        raise ValueError("Learned polynomial coefficients require an integer attention_degree >= 1")
    if coefficients.shape not in ((attention_degree + 1,), (heads, attention_degree + 1)):
        raise ValueError("Polynomial coefficients must have shape (m+1,) or (heads, m+1)")


def unnormalized_attention(q, k, v, *, attention_degree=None, backend="blocked", return_diagnostics=False,
                           polynomial_coefficients=None, blocked_implementation="scan", score_transform="identity",
                           score_prefactor="none", exp_shift="none"):
    """``attention_degree=None`` selects unshifted exp; an integer m selects p_m.

    ``polynomial_coefficients`` replaces the Taylor coefficients of p_m by learned
    values, shared ([m+1]) or per head ([heads, m+1]); the kernel may be signed.
    ``score_transform`` and ``score_prefactor`` (attention/pieces.py) give the exponential
    f(s) exp(phi(s)), on the blocked backend or the fused kernel. ``exp_shift="row_max"`` shifts the
    exponential by its causal row maximum m and returns ``(output, m)`` for head RMSNorm
    (attention/output_scaling.py): on the blocked backend, or plain on the fused kernel.
    """
    from scaffolds.transformer.attention.pieces import EXP_SHIFTS, ScoreFunction
    from scaffolds.transformer.attention.backends.reference import (
        blocked_backend_attention, blocked_dot_product_attention,
    )

    # Fixed kernels keep their established reference call; learned ones add their coefficients.
    learned = {}
    if polynomial_coefficients is not None:
        _validate_coefficients(polynomial_coefficients, attention_degree, q.shape[2])
        learned = {"polynomial_coefficients": polynomial_coefficients}
    # Only a non-identity score function is passed on, so every historical call is unchanged.
    functions = {} if ScoreFunction(score_transform, score_prefactor).identity else dict(
        score_transform=score_transform, score_prefactor=score_prefactor)
    if functions and (attention_degree is not None or polynomial_coefficients is not None):
        raise ValueError("Here the transform and prefactor apply to the exponential alone")
    if exp_shift not in EXP_SHIFTS:
        raise ValueError(f"Unknown exponential shift: {exp_shift!r}")
    # Likewise only a shift is passed on.
    shift = {} if exp_shift == "none" else {"exp_shift": exp_shift}
    if shift and (attention_degree is not None or polynomial_coefficients is not None or backend == "mosaic"):
        raise ValueError("The row-maximum shift applies to the exponential, on the blocked backend or fused kernel")
    if backend == "blocked":
        return blocked_backend_attention(
            q, k, v, implementation=blocked_implementation, attention_degree=attention_degree,
            attention_normalization=False, return_diagnostics=return_diagnostics, **learned, **functions, **shift,
        )
    if backend not in ("fused", "mosaic"):
        raise ValueError("Unnormalized attention requires backend='blocked', 'fused' or 'mosaic'")
    if backend == "mosaic" and functions:
        raise ValueError("The Mosaic kernels do not implement the exponential's score functions")
    if backend == "mosaic":
        from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention

        # The query-coefficient kernel with every query's coefficients equal; broadcasting sums
        # the per-query coefficient gradients back to the shared or per-head coefficients.
        if attention_degree is None:
            raise ValueError("Mosaic unnormalized attention requires a polynomial degree")
        coefficients = (jnp.asarray(initial_coefficients(attention_degree), jnp.float32)
                        if polynomial_coefficients is None else polynomial_coefficients.astype(jnp.float32))
        coefficients = jnp.broadcast_to(coefficients, (q.shape[2], attention_degree + 1))
        output = query_polynomial_attention(
            q, k, v, jnp.broadcast_to(coefficients[None, None], (*q.shape[:3], attention_degree + 1)))
    elif polynomial_coefficients is None:
        from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

        output = fused_unnormalized_attention(q, k, v, attention_degree=attention_degree, **functions, **shift)
    else:
        from scaffolds.transformer.attention.backends.fused_learned_polynomial import (
            fused_learned_polynomial_attention,
        )

        # Broadcasting a shared array sums its per-head gradients in the VJP.
        output = fused_learned_polynomial_attention(
            q, k, v, jnp.broadcast_to(polynomial_coefficients, (q.shape[2], attention_degree + 1)))
    if not return_diagnostics:
        return output
    _, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=attention_degree, attention_normalization=False,
        return_diagnostics=True, **learned, **functions, **shift,
    )
    return output, diagnostics
