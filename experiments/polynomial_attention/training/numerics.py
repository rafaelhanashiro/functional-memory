"""Surface invalid learned-kernel updates through the existing failure channel."""

import jax.numpy as jnp
import optax


def checked_update_loss(params, grads, updated_params, loss):
    """Reject gradient/parameter breakdown even when the forward loss was finite.

    A signed attention denominator can make its derivative overflow before its
    forward value does. The training loop already synchronizes/checks the loss
    every update; poisoning that scalar avoids an additional host synchronization.
    Fixed-kernel runs keep their original execution path.
    """
    if "quadratic_coefficients" not in params:
        return loss
    finite = jnp.isfinite(optax.tree.norm(grads))
    finite &= jnp.all(jnp.isfinite(updated_params["quadratic_coefficients"][...]))
    return jnp.where(finite, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))
