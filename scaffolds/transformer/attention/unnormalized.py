"""Unnormalized causal k(QK^T / sqrt(d_k)) V: exp, its Taylor truncation, or a learned polynomial."""

import math

import jax
import jax.numpy as jnp


def initial_coefficients(degree: int, initialization: str = "taylor") -> tuple[float, ...]:
    """Initial c_0..c_m: the Taylor values 1/r!, or linear attention (c_1 = 1, every other 0)."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("Learned polynomial coefficients require an integer degree >= 1")
    if initialization == "taylor":
        return tuple(1.0 / math.factorial(order) for order in range(degree + 1))
    if initialization == "linear":
        return tuple(1.0 if order == 1 else 0.0 for order in range(degree + 1))
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
                           polynomial_coefficients=None, blocked_implementation="scan"):
    """``attention_degree=None`` selects unshifted exp; an integer m selects p_m.

    ``polynomial_coefficients`` replaces the Taylor coefficients of p_m by learned
    values, shared ([m+1]) or per head ([heads, m+1]); the kernel may be signed.
    """
    from scaffolds.transformer.attention.backends.reference import (
        blocked_backend_attention, blocked_dot_product_attention,
    )

    # Fixed kernels keep their established reference call; learned ones add their coefficients.
    learned = {}
    if polynomial_coefficients is not None:
        _validate_coefficients(polynomial_coefficients, attention_degree, q.shape[2])
        learned = {"polynomial_coefficients": polynomial_coefficients}
    if backend == "blocked":
        return blocked_backend_attention(
            q, k, v, implementation=blocked_implementation, attention_degree=attention_degree,
            attention_normalization=False, return_diagnostics=return_diagnostics, **learned,
        )
    if backend not in ("fused", "mosaic"):
        raise ValueError("Unnormalized attention requires backend='blocked', 'fused' or 'mosaic'")
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

        output = fused_unnormalized_attention(q, k, v, attention_degree=attention_degree)
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
        return_diagnostics=True, **learned,
    )
    return output, diagnostics
