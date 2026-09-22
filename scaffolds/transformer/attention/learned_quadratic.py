"""Learned positive and unrestricted quadratic attention mechanisms."""

import jax
import jax.numpy as jnp


def _quadratic_polynomial(scores: jax.Array, coefficients: jax.Array, mode: str) -> jax.Array:
    """Evaluate raw learned parameters over [batch, heads, queries, keys] scores."""
    coefficients = jnp.broadcast_to(coefficients, (scores.shape[1], 2)).astype(jnp.float32)
    first, second = coefficients[:, 0][None, :, None, None], coefficients[:, 1][None, :, None, None]
    if mode == "positive":
        return 0.5 * (1.0 + jnp.square(first * scores + second))
    return 1.0 + scores * (first + scores * second)


def parameter_specs(*, mode: str, sharing: str):
    """Declare parameter identity and ownership without allocating or using RNGs."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    if mode not in ("positive", "unrestricted"):
        raise ValueError(f"Unknown learned quadratic mode: {mode}")
    return (AttentionParameterSpec(
        name="quadratic_coefficients", argument="coefficients",
        initial=(1.0, 1.0 if mode == "positive" else 0.5), sharing=sharing, weight_decay=False,
    ),)


def learned_quadratic_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, coefficients: jax.Array,
    quadratic_mode: str, backend: str = "blocked", return_diagnostics: bool = False,
):
    """Apply a learned pair shared across heads or an independent pair per head."""
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if quadratic_mode not in ("positive", "unrestricted"):
        raise ValueError(f"Unknown learned quadratic mode: {quadratic_mode}")
    if coefficients is None or coefficients.shape not in ((2,), (q.shape[2], 2)):
        raise ValueError("Quadratic coefficients must have shape (2,) or (heads, 2)")
    if backend == "blocked":
        return blocked_dot_product_attention(
            q, k, v, attention_degree=2, coefficients=coefficients,
            quadratic_mode=quadratic_mode, return_diagnostics=return_diagnostics,
        )
    if backend != "fused":
        raise ValueError("Learned quadratic attention requires backend='blocked' or 'fused'")
    from scaffolds.transformer.attention.backends.fused_quadratic import fused_quadratic_attention

    output = fused_quadratic_attention(
        q, k, v, jnp.broadcast_to(coefficients, (q.shape[2], 2)),
        quadratic_mode=quadratic_mode,
    )
    if not return_diagnostics:
        return output
    _, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=2, coefficients=coefficients,
        quadratic_mode=quadratic_mode, return_diagnostics=True,
    )
    return output, diagnostics
