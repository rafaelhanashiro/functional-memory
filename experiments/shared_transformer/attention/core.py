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
    quadratic_mode: str = "fixed",
    coefficients: jax.Array | None = None,
    return_diagnostics: bool = False,
    dropout=None,
    deterministic: bool = True,
    rngs=None,
    weights_dtype=None,
):
    """Apply the mechanism selected by the existing flat model configuration."""
    if quadratic_mode != "fixed":
        from experiments.shared_transformer.attention.learned_quadratic import learned_quadratic_attention

        if attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        return learned_quadratic_attention(
            q, k, v, coefficients=coefficients, quadratic_mode=quadratic_mode,
            backend=backend, return_diagnostics=return_diagnostics,
        )
    if coefficients is not None:
        raise ValueError("Fixed attention does not accept learned coefficients")
    if attention_degree is not None:
        from experiments.shared_transformer.attention.taylor import taylor_attention

        return taylor_attention(
            q, k, v, attention_degree=attention_degree, backend=backend,
            return_diagnostics=return_diagnostics,
        )
    from experiments.shared_transformer.attention.softmax import softmax_attention

    return softmax_attention(
        q, k, v, backend=backend, return_diagnostics=return_diagnostics,
        dropout=dropout, deterministic=deterministic, rngs=rngs,
        weights_dtype=weights_dtype,
    )
