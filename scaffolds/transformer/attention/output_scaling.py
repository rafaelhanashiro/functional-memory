"""Rescaling of attention outputs before the output projection.

Unnormalized attention returns ``o_t = sum_(j<=t) k(s_tj) v_j`` for each head.
Normalized attention equals ``o_t / sum_j k(s_tj)``, so the unnormalized output
is the normalized one times the row's kernel sum. ``inverse_position`` divides an
unnormalized output by the number of visible keys ``t`` instead, which removes the
sum's growth with ``t`` but keeps its dependence on the scores. ``head_rms_norm``
applies an RMSNorm over each head's value channels with a learned gain; it removes
any positive per-row scale without dividing by a kernel sum that a signed kernel can
drive through zero. It also applies to normalized attention: for a positive kernel,
RMSNorm of the normalized and unnormalized outputs differ only through ``eps``.

In FP32 the mean square of a head's output overflows once its L2 norm over the value
channels passes about 1.8e19 (the square root of the FP32 maximum), and ``rsqrt(inf) = 0``
would silently zero that output and its gradient. Such a row is first divided by its
largest magnitude, ``eps`` with it: RMSNorm is scale-invariant, so it is the same function.
A row whose mean square is finite keeps scale one, which leaves its arithmetic and gradient
unchanged bit for bit, and a row with an infinite or NaN entry is NaN, a numerical failure.
"""

import jax
import jax.numpy as jnp

OUTPUT_SCALINGS = ("none", "inverse_position", "head_rms_norm")


def parameter_specs(*, value_head_dim: int):
    """The RMSNorm gain per layer, head, and value channel, initialized at 1, without weight decay."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (AttentionParameterSpec(
        name="output_norm_scale", argument="output_norm_scale",
        initial=(1.0,) * value_head_dim, sharing="per_head", weight_decay=False,
    ),)


def scale_attention_output(y: jax.Array, scaling: str, *, output_norm_scale: jax.Array | None = None,
                           eps: float = 1e-6) -> jax.Array:
    """Rescale ``[batch, queries, heads, value_dim]`` outputs; query ``t`` (0-based) sees ``t + 1`` keys."""
    if scaling not in OUTPUT_SCALINGS:
        raise ValueError(f"Unknown attention output scaling: {scaling}")
    if (scaling == "head_rms_norm") != (output_norm_scale is not None):
        raise ValueError("Head RMSNorm output scaling requires exactly its output_norm_scale parameter")
    if scaling == "none":
        return y
    value = y.astype(jnp.float32)
    if scaling == "inverse_position":
        value = value / jnp.arange(1, y.shape[1] + 1, dtype=jnp.float32)[None, :, None, None]
    else:
        if output_norm_scale.shape != y.shape[-2:]:
            raise ValueError("output_norm_scale must have shape (heads, value_dim)")
        # The divisor is one where the FP32 mean square is finite, and otherwise the row's largest magnitude (module
        # docstring); a constant to differentiation, since RMSNorm's output does not depend on it (up to eps).
        overflow = ~jnp.isfinite(jnp.mean(jnp.square(value), axis=-1, keepdims=True))
        divisor = jax.lax.stop_gradient(jnp.where(overflow, jnp.max(jnp.abs(value), axis=-1, keepdims=True), 1.0))
        value = value / divisor
        value = value * jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + eps / jnp.square(divisor))
        value = value * output_norm_scale.astype(jnp.float32)[None, None]
    return value.astype(y.dtype)
