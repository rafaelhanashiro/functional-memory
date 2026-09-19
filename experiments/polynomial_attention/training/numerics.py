"""Surface invalid signed-kernel updates through the existing failure channel."""

import jax.numpy as jnp
import optax


def is_odd_taylor(model) -> bool:
    return model.attention_degree is not None and model.attention_degree % 2 == 1


def update_numerics(grads, updated_params, loss):
    """Small device-side record that distinguishes forward and update failures."""
    return jnp.stack((loss, optax.tree.norm(grads), optax.tree.norm(updated_params)))


def checked_update_loss(params, grads, updated_params, loss, *, check_all_gradients=False):
    """Reject gradient/parameter breakdown even when the forward loss was finite.

    A signed attention denominator can make its derivative overflow before its
    forward value does. The training loop already synchronizes/checks the loss
    every update; poisoning that scalar avoids an additional host synchronization.
    Historical fixed even-degree kernels keep their original execution path.
    """
    if not check_all_gradients and "quadratic_coefficients" not in params:
        return loss
    finite = jnp.isfinite(optax.tree.norm(grads))
    if check_all_gradients:
        finite &= jnp.isfinite(optax.tree.norm(updated_params))
    if "quadratic_coefficients" in params:
        finite &= jnp.all(jnp.isfinite(updated_params["quadratic_coefficients"][...]))
    return jnp.where(finite, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))
