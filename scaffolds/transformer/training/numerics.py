"""Surface invalid training updates through the existing failure channel."""

import jax.numpy as jnp
import optax


def is_odd_taylor(model) -> bool:
    return (model.attention_degree is not None and model.attention_degree % 2 == 1
            and not has_fixed_polynomial(model) and not has_learned_polynomial(model)
            and model.quadratic_mode == "fixed")


def has_fixed_polynomial(model) -> bool:
    return getattr(model, "attention_fixed_coefficients", None) is not None


def has_fixed_square_factors(model) -> bool:
    return getattr(model, "attention_fixed_square_factors", None) is not None


def has_learned_polynomial(model) -> bool:
    return getattr(model, "attention_coefficients", "fixed") == "learned"


def has_sum_of_squares_kernel(model) -> bool:
    return (has_fixed_square_factors(model)
            or (has_learned_polynomial(model)
                and getattr(model, "attention_coefficient_parameterization", "direct") == "sum_of_squares"))


def has_rational_kernel(model) -> bool:
    """Separate learned numerator and denominator polynomials (normalized attention)."""
    return (has_learned_polynomial(model)
            and getattr(model, "attention_coefficient_parameterization", "direct") == "rational")


def has_score_mlp(model) -> bool:
    """A learned, input-dependent score MLP replaces the exponential (attention/score_mlp.py)."""
    return getattr(model, "attention_score_mlp_width", None) is not None


def is_signed_kernel(model) -> bool:
    """Kernels whose weights may be negative or zero by design."""
    return (is_odd_taylor(model) or has_fixed_polynomial(model) or model.quadratic_mode == "unrestricted"
            or (has_learned_polynomial(model) and not has_sum_of_squares_kernel(model)) or has_score_mlp(model))


def requires_update_checks(model) -> bool:
    """Every mechanism can have invalid gradients despite a finite forward loss."""
    return True


def update_numerics(grads, updated_params, loss):
    """Small device-side record that distinguishes forward and update failures."""
    return jnp.stack((loss, optax.tree.norm(grads), optax.tree.norm(updated_params)))


def checked_update_loss(params, grads, updated_params, loss, *, check_all_gradients=False):
    """Reject gradient/parameter breakdown even when the forward loss was finite.

    The training loop already synchronizes/checks the loss every update;
    poisoning that scalar avoids an additional host synchronization. Checking
    the gradient norm also rejects finite gradients whose norm overflows during
    clipping, which would otherwise silently discard their update.

    ``params`` and ``check_all_gradients`` remain accepted for existing callers.
    The flag controls callers' diagnostic return shapes, never these checks.
    """
    finite = jnp.isfinite(optax.tree.norm(grads))
    finite &= jnp.isfinite(optax.tree.norm(updated_params))
    return jnp.where(finite, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))
