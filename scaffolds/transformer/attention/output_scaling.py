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

The max-abs prescale (``model.attention_head_rms_prescale: max_abs``; "none" is every historical recipe, whose
arithmetic above is unchanged). Written as above, the FP32 backward of ``o rsqrt(x)``, ``x = mean(o^2) + eps``,
multiplies by ``rsqrt(x) / x = x^(-3/2)``, which is a subnormal number once the row's RMS passes 4.4e12 and
rounds to zero past 1.1e15. That factor scales the term that removes the gradient's component along ``o``. On
GPU, which keeps subnormal numbers, a row's gradient degrades from an RMS of about 3e13 (1e-5 of its norm; 0.2%
at 2e14, 2.6% at 6e14) and from 9e14 keeps the whole component along ``o``, which the exact Jacobian does not
have, until the mean square overflows and the guard above takes over (validation/README.md); XLA's CPU code
flushes subnormal numbers, so there the whole term is lost from 4.4e12. The prescale evaluates the same function
in units of each row's own size. For any constant ``c > 0``,

    o / sqrt(mean(o^2) + eps) = (o / c) / sqrt(mean((o / c)^2) + eps / c^2)

identically in ``o``: both sides are one function of ``o`` whatever ``c`` is, so the right side's gradient with
``c`` held constant is the left side's, and nothing is lost by not differentiating through ``c`` (the total
derivative through it vanishes). Head RMSNorm takes ``c`` per token and head from the row itself, as a constant:

- an unshifted row ``y = o``: ``c = max(max_c |y_c|, 1)``. A row whose largest magnitude is at most one keeps
  ``c = 1``, the plain arithmetic bit for bit, which is in range there (``x`` between ``eps`` and ``1 + eps``;
  dividing a vanishing row by its own size would instead overflow ``eps / c^2`` and zero its gradient). A larger
  row is divided by its largest magnitude: ``mean((y / c)^2)`` lies in ``[1 / channels, 1]``, ``eps / c^2`` is at
  most ``eps`` (zero once ``c^2`` overflows, which is its FP32 value next to a mean square of at least
  ``1 / channels``), and every factor of the forward and the backward is within a few powers of ten of one,
  whatever the row's scale. This also covers the overflowing mean square, so the prescale needs no guard. The
  backward's last step divides by ``c``, so the gradient's elements are of size cotangent / RMS: they are
  exact to rounding while FP32's normal range (down to 1.2e-38) holds them. A GPU keeps smaller ones as
  subnormal numbers, with fewer bits, and XLA's CPU code flushes them to zero, however the function is
  evaluated: on CPU the row gradient is within 1e-7 of FP64 for a quotient of 1e-33 or more, 7e-7 at 1e-34
  and 2% at 1e-37 (its elements under the range), and zero from 1e-39.
- a row-shifted exponential ``y = e^(-m) o``: the size of ``y`` says nothing about that of ``o``, so the row is
  classified by ``log max|o| = m + log max|y|``. Where ``max|o|`` exceeds ``e^-23 sqrt(eps)`` (1e-13 at
  ``eps = 1e-6``), ``c = max|o|``: ``o / c = y / max|y|`` and ``eps / c^2 = eps (e^(-m/2) / max|y| * e^(-m/2))^2``,
  at most ``e^46``, the two half factors keeping the quotient in range at shifts whose ``e^(-m)`` is not. Below
  it the normalization is in its linear range: ``mean(o^2) < e^-46 eps`` is below the rounding of ``eps`` in FP32
  and FP64 alike, so the function is ``o / sqrt(eps)`` and is evaluated as ``y (e^m / sqrt(eps))``. That slope
  is capped at ``e^80.5`` (FP32), which only a row that is exactly zero, or below FP32's normal range, reaches:
  its output stays zero and its gradient finite where the exact slope leaves FP32.

Either way a row with an infinite or NaN entry (or a NaN shift) is NaN, as without the prescale, and the learned
gain multiplies the result as before.
"""

import math

import jax
import jax.numpy as jnp

OUTPUT_SCALINGS = ("none", "inverse_position", "head_rms_norm")
# How head RMSNorm is evaluated (module docstring): "none" (every historical recipe) as written, with a divisor
# only for rows whose FP32 mean square overflows; "max_abs" in units of each row's largest magnitude, the same
# function with an FP32 gradient that is exact to rounding wherever FP32 holds the gradient itself.
HEAD_RMS_PRESCALES = ("none", "max_abs")
# A shifted row whose largest unshifted magnitude is below e^-LINEAR_LOG_RANGE sqrt(eps) is in RMSNorm's linear
# range: mean(o^2) / eps < e^-46 = 1.05e-20, below FP64's rounding of eps (1.1e-16), let alone FP32's.
LINEAR_LOG_RANGE = 23.0
# The linear range's slope e^m / sqrt(eps) is capped this many e-folds below the dtype's largest number (e^80.5 in
# FP32), which leaves the product with a gain and a cotangent of up to e^8.2 = 3,600 finite.
SLOPE_HEADROOM = 8.2


def parameter_specs(*, value_head_dim: int):
    """The RMSNorm gain per layer, head, and value channel, initialized at 1, without weight decay."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (AttentionParameterSpec(
        name="output_norm_scale", argument="output_norm_scale",
        initial=(1.0,) * value_head_dim, sharing="per_head", weight_decay=False,
    ),)


def max_abs_head_rms(value: jax.Array, eps: float, shift: jax.Array | None = None) -> jax.Array:
    """``value`` [..., channels] times ``rsqrt(mean(o^2) + eps)`` per row, ``o = e^shift value`` (``shift``
    [..., 1], a constant to differentiation; None: ``o = value``), evaluated in units of each row's largest
    magnitude (module docstring). It computes in ``value``'s dtype: FP32 in the model, FP64 in its tests."""
    largest = jnp.max(jnp.abs(jax.lax.stop_gradient(value)), axis=-1, keepdims=True)
    if shift is None:
        divisor = jnp.maximum(largest, 1.0)
        scaled = value / divisor
        return scaled * jax.lax.rsqrt(jnp.mean(jnp.square(scaled), axis=-1, keepdims=True)
                                      + eps / jnp.square(divisor))
    if eps <= 0:
        raise ValueError("The max-abs prescale of a row-shifted output requires a positive epsilon")
    shift = jax.lax.stop_gradient(shift.astype(value.dtype))
    half_log_eps = 0.5 * math.log(eps)
    # log(max|o| / sqrt(eps)): -inf for a zero row, and NaN (linear, then NaN below) for a NaN entry or shift.
    normalized = shift + jnp.log(largest) - half_log_eps > -LINEAR_LOG_RANGE
    # Rows outside a branch run it on stand-in constants (divisor and inverse one), which keep that branch and its
    # derivative finite for every finite row, so the selection below passes no NaN into the gradient.
    divisor = jnp.where(normalized, largest, 1.0)
    half = jnp.exp(-0.5 * shift)
    inverse = jnp.where(normalized, half / divisor * half, 1.0)  # 1 / max|o|
    scaled = value / divisor
    scaled = scaled * jax.lax.rsqrt(jnp.mean(jnp.square(scaled), axis=-1, keepdims=True)
                                    + eps * jnp.square(inverse))
    cap = math.log(float(jnp.finfo(value.dtype).max)) - SLOPE_HEADROOM + half_log_eps
    linear = value * (jnp.exp(jnp.minimum(shift, cap)) * eps ** -0.5)
    finite = jnp.all(jnp.isfinite(value), axis=-1, keepdims=True) & ~jnp.isnan(shift)
    return jnp.where(finite, jnp.where(normalized, scaled, linear), jnp.nan)


def scale_attention_output(y: jax.Array, scaling: str, *, output_norm_scale: jax.Array | None = None,
                           eps: float = 1e-6, output_shift: jax.Array | None = None,
                           head_rms_prescale: str = "none") -> jax.Array:
    """Rescale ``[batch, queries, heads, value_dim]`` outputs; query ``t`` (0-based) sees ``t + 1`` keys.

    ``output_shift`` (head RMSNorm only) is the row offset ``m`` [batch, queries, heads] of an output that holds
    ``e^(-m)`` times the attention output (module docstring); None, every other recipe, leaves the arithmetic as it
    was. ``head_rms_prescale="max_abs"`` (head RMSNorm only) evaluates the same function in units of each row's
    largest magnitude; "none", every historical recipe, likewise leaves the arithmetic as it was."""
    if scaling not in OUTPUT_SCALINGS:
        raise ValueError(f"Unknown attention output scaling: {scaling}")
    if (scaling == "head_rms_norm") != (output_norm_scale is not None):
        raise ValueError("Head RMSNorm output scaling requires exactly its output_norm_scale parameter")
    if output_shift is not None and scaling != "head_rms_norm":
        raise ValueError("Only head RMSNorm takes a row-shifted output: other scalings would change its function")
    if output_shift is not None and output_shift.shape != y.shape[:-1]:
        raise ValueError("output_shift must have shape (batch, queries, heads)")
    if head_rms_prescale not in HEAD_RMS_PRESCALES:
        raise ValueError(f"Unknown head RMSNorm prescale: {head_rms_prescale!r}")
    if head_rms_prescale != "none" and scaling != "head_rms_norm":
        raise ValueError("Only head RMSNorm takes a prescale: it is that normalization's own evaluation")
    if scaling == "none":
        return y
    value = y.astype(jnp.float32)
    if scaling == "inverse_position":
        value = value / jnp.arange(1, y.shape[1] + 1, dtype=jnp.float32)[None, :, None, None]
    elif head_rms_prescale == "max_abs":
        if output_norm_scale.shape != y.shape[-2:]:
            raise ValueError("output_norm_scale must have shape (heads, value_dim)")
        shift = None if output_shift is None else output_shift.astype(jnp.float32)[..., None]
        value = max_abs_head_rms(value, eps, shift) * output_norm_scale.astype(jnp.float32)[None, None]
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
