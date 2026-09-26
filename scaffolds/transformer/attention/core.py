"""Complete attention operations on post-RoPE Q/K/V arrays.

Mechanisms consume [batch, time, heads, channels] arrays and return an array
with V's shape. They own score/kernel mathematics and backend dispatch; the
Transformer owns projections and model parameters. Dispatch uses static Python
configuration and disappears when JAX traces the training step.
"""

import jax


def attention_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    backend: str = "blocked",
    attention_degree: int | None = None,
    attention_normalization: bool = True,
    quadratic_mode: str = "fixed",
    coefficients: jax.Array | None = None,
    polynomial_coefficients: jax.Array | None = None,
    return_diagnostics: bool = False,
    dropout=None,
    deterministic: bool = True,
    rngs=None,
    weights_dtype=None,
):
    """Apply the mechanism selected by the existing flat model configuration."""
    if not isinstance(attention_normalization, bool):
        raise ValueError("attention_normalization must be a boolean")
    if not attention_normalization:
        from scaffolds.transformer.attention.unnormalized import unnormalized_attention

        if quadratic_mode != "fixed" or coefficients is not None:
            raise ValueError("Unnormalized attention requires the exponential or a fixed Taylor kernel")
        return unnormalized_attention(
            q, k, v, attention_degree=attention_degree, backend=backend,
            return_diagnostics=return_diagnostics, polynomial_coefficients=polynomial_coefficients,
        )
    if polynomial_coefficients is not None:
        from scaffolds.transformer.attention.learned_polynomial import normalized_learned_polynomial_attention

        if quadratic_mode != "fixed" or coefficients is not None:
            raise ValueError("Learned polynomial coefficients exclude the learned quadratic modes")
        return normalized_learned_polynomial_attention(
            q, k, v, attention_degree=attention_degree, polynomial_coefficients=polynomial_coefficients,
            backend=backend, return_diagnostics=return_diagnostics,
        )
    if quadratic_mode != "fixed":
        from scaffolds.transformer.attention.learned_quadratic import learned_quadratic_attention

        if attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        return learned_quadratic_attention(
            q, k, v, coefficients=coefficients, quadratic_mode=quadratic_mode,
            backend=backend, return_diagnostics=return_diagnostics,
        )
    if coefficients is not None:
        raise ValueError("Fixed attention does not accept learned coefficients")
    if attention_degree is not None:
        from scaffolds.transformer.attention.taylor import taylor_attention

        return taylor_attention(
            q, k, v, attention_degree=attention_degree, backend=backend,
            return_diagnostics=return_diagnostics,
        )
    from scaffolds.transformer.attention.softmax import softmax_attention

    return softmax_attention(
        q, k, v, backend=backend, return_diagnostics=return_diagnostics,
        dropout=dropout, deterministic=deterministic, rngs=rngs,
        weights_dtype=weights_dtype,
    )
