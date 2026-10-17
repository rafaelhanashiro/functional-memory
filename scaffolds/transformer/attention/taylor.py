"""Fixed, unshifted Taylor truncations of the exponential attention kernel."""

import jax
import jax.numpy as jnp


def _validate_degree(attention_degree: int | None) -> None:
    if attention_degree is not None and (
        isinstance(attention_degree, bool)
        or not isinstance(attention_degree, int)
        or attention_degree < 0
    ):
        raise ValueError("attention_degree must be None or a nonnegative integer")


def _taylor_polynomial(scores: jax.Array, degree: int) -> jax.Array:
    """Nested Horner evaluation without underflowing factorial coefficients."""
    result = jnp.ones_like(scores)
    for ell in range(degree, 0, -1):
        result = 1.0 + scores * (result / ell)
    return result


def taylor_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, attention_degree: int,
    backend: str = "blocked", return_diagnostics: bool = False,
):
    """Normalized causal Taylor attention, including signed odd truncations."""
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    _validate_degree(attention_degree)
    if attention_degree is None:
        raise ValueError("Taylor attention requires an integer degree")
    if backend == "blocked":
        return blocked_dot_product_attention(
            q, k, v, attention_degree=attention_degree,
            return_diagnostics=return_diagnostics,
        )
    if backend == "fused":
        from scaffolds.transformer.attention.backends.fused_taylor import fused_polynomial_attention

        output = fused_polynomial_attention(q, k, v, attention_degree=attention_degree)
    elif backend == "mosaic":
        from scaffolds.transformer.attention.backends.mosaic.api import taylor_attention as mosaic_taylor_attention

        output = mosaic_taylor_attention(q, k, v, attention_degree=attention_degree)
    else:
        raise ValueError("Taylor attention requires backend='blocked', 'fused' or 'mosaic'")
    if not return_diagnostics:
        return output
    # Measure actual layer inputs while retaining the fused output downstream.
    _, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=attention_degree, return_diagnostics=True,
    )
    return output, diagnostics
