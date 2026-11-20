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

The unnormalized exponential may arrive shifted by its causal row maximum (``model.attention_exp_shift:
row_max``): ``y_t = e^(-m_t) o_t`` with ``m_t = max_(i<=t)`` of the row's logits, whose terms
``e^(s_ti - m_t)`` never exceed one, so ``y_t`` stays finite where ``e^(s_ti)`` overflows FP32 (s above about
88.7). Head RMSNorm then takes the offset ``m_t`` and scales its epsilon by ``e^(-2 m_t)``:
``y_t / sqrt(mean(y_t^2) + eps e^(-2 m_t))`` equals ``o_t / sqrt(mean(o_t^2) + eps)`` in real arithmetic.
That result does not depend on ``m_t``, so ``m_t`` is a constant to differentiation, in the kernel as here.
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
                           eps: float = 1e-6, output_shift: jax.Array | None = None) -> jax.Array:
    """Rescale ``[batch, queries, heads, value_dim]`` outputs; query ``t`` (0-based) sees ``t + 1`` keys.

    ``output_shift`` (head RMSNorm only) is the row offset ``m`` [batch, queries, heads] of an output that holds
    ``e^(-m)`` times the attention output (module docstring); None, every other recipe, leaves the arithmetic as it
    was."""
    if scaling not in OUTPUT_SCALINGS:
        raise ValueError(f"Unknown attention output scaling: {scaling}")
    if (scaling == "head_rms_norm") != (output_norm_scale is not None):
        raise ValueError("Head RMSNorm output scaling requires exactly its output_norm_scale parameter")
    if output_shift is not None and scaling != "head_rms_norm":
        raise ValueError("Only head RMSNorm takes a row-shifted output: other scalings would change its function")
    if output_shift is not None and output_shift.shape != y.shape[:-1]:
        raise ValueError("output_shift must have shape (batch, queries, heads)")
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
        if output_shift is None:
            value = value * jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True)
                                          + eps / jnp.square(divisor))
        else:
            # eps e^(-2m), the offset a constant to differentiation (module docstring). Past m of about 45 it
            # underflows to zero; a row that is then exactly zero stays zero, as unshifted, with scale and
            # gradient zero instead of 0 * rsqrt(0). A NaN denominator still makes its row NaN.
            shift = jax.lax.stop_gradient(output_shift.astype(jnp.float32))[..., None]
            denominator = (jnp.mean(jnp.square(value), axis=-1, keepdims=True)
                           + eps * jnp.exp(-2.0 * shift) / jnp.square(divisor))
            zero = denominator == 0
            value = value * jnp.where(zero, 0.0, jax.lax.rsqrt(jnp.where(zero, 1.0, denominator)))
        value = value * output_norm_scale.astype(jnp.float32)[None, None]
    return value.astype(y.dtype)
