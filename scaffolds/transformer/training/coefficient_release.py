"""Coefficient release: learned polynomial coefficients stay at their initial values, then train.

``optim.coefficient_release_*`` gives each polynomial order r a release count s_r. Every
stored entry of order r (static c_r; for diagonal tanh query coefficients b_r, a_r, and
column r of W) is frozen for optimizer updates with count k < s_r and trains for k >= s_r.
k is the learning-rate schedule's own clock: the update logged as training step k + 1 uses
learning rate schedule(k). "all" sets s_r = coefficient_release_step for every order;
"by_order" sets s_r = coefficient_release_step + floor(r (coefficient_release_end_step -
coefficient_release_step) / m), lowest order first.

Freezing acts twice and depends only on the update count. A frozen entry's gradient is zeroed
before global-norm clipping and AdamW, so it adds nothing to the clip norm and its Adam moments
stay exactly zero; its final update is zeroed after AdamW, so decoupled weight decay cannot
move it either. A released entry's moments therefore start from zero at its release, while bias
correction keeps the global count (its first update is about half size), and it joins the
shared learning-rate schedule at the schedule's current value.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

# The stored coefficient arrays a schedule releases; the last axis of each indexes order 0..m.
STATIC_COEFFICIENTS = ("polynomial_coefficients",)
QUERY_COEFFICIENTS = ("query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel")


def release_counts(optim, degree: int) -> tuple[int, ...] | None:
    """s_0..s_degree, each order's first trained update count, or None without a schedule."""
    schedule = getattr(optim, "coefficient_release_schedule", None)
    if schedule is None:
        return None
    start = optim.coefficient_release_step
    if schedule == "all":
        return (start,) * (degree + 1)
    if schedule == "by_order":
        end = optim.coefficient_release_end_step
        return tuple(start + order * (end - start) // degree for order in range(degree + 1))
    raise ValueError(f"Unknown coefficient release schedule: {schedule!r}")


@dataclass(frozen=True)
class CoefficientRelease:
    """When each polynomial order of the learned coefficients starts to train."""

    schedule: str
    counts: tuple[int, ...]
    """s_0..s_m: order r is frozen for optimizer updates with count k < s_r and trains for k >= s_r."""
    parameters: tuple[str, ...]
    """The coefficient arrays the schedule freezes; every one must be present."""

    def trainable(self, opt_state) -> jax.Array:
        """Boolean ``[m + 1]``: the orders that train in the update ``opt_state`` is about to take."""
        count = schedule_count(opt_state)
        return count >= jnp.asarray(self.counts, dtype=count.dtype)

    def mask(self, tree, trainable: jax.Array, fill: float):
        """``tree`` (gradients or updates) with every frozen coefficient entry set to ``fill``."""
        masked_names = []

        def select(path, value):
            keys = tuple(entry.key for entry in path if isinstance(entry, jax.tree_util.DictKey))
            if not keys or keys[0] not in self.parameters:
                return value
            if value.shape[-1] != len(self.counts):
                raise ValueError(f"{keys[0]} stores {value.shape[-1]} orders, but the release schedule "
                                 f"has {len(self.counts)}")
            masked_names.append(keys[0])
            return jnp.where(trainable, value, jnp.asarray(fill, dtype=value.dtype))

        masked = jax.tree.map_with_path(select, tree)
        if sorted(masked_names) != sorted(self.parameters):
            # A missing or renamed array would otherwise train from the first update.
            raise ValueError(f"The coefficient release schedule freezes {list(self.parameters)}, "
                             f"but the parameters hold {sorted(masked_names)}")
        return masked

    def released_at(self, count: int) -> tuple[int, ...]:
        """The orders whose first trained update has optimizer count ``count``."""
        return tuple(order for order, first in enumerate(self.counts) if first == count)

    def trainable_orders(self, count: int) -> int:
        """How many orders train in the update with optimizer count ``count``."""
        return sum(first <= count for first in self.counts)


def coefficient_release(config) -> CoefficientRelease | None:
    """The recipe's release schedule; None trains every coefficient from the first update."""
    optim = getattr(config, "optim", None)
    counts = release_counts(optim, config.model.attention_degree)
    if counts is None:
        return None
    query = config.model.attention_coefficient_conditioning == "query"
    return CoefficientRelease(optim.coefficient_release_schedule, counts,
                              QUERY_COEFFICIENTS if query else STATIC_COEFFICIENTS)


def schedule_count(opt_state) -> jax.Array:
    """Completed optimizer updates, as the learning-rate schedule's own state counts them."""
    clocks = [node.count for node in jax.tree.leaves(
        opt_state, is_leaf=lambda node: isinstance(node, optax.ScaleByScheduleState))
        if isinstance(node, optax.ScaleByScheduleState)]
    if len(clocks) != 1:
        raise ValueError(f"A coefficient release needs exactly one learning-rate schedule clock, found {len(clocks)}")
    return clocks[0]


@dataclass(frozen=True)
class CoefficientReleaseOptimizer:
    """``inner`` (clipping, then AdamW) with the release masks; its state is exactly ``inner``'s.

    It masks the gradients itself, so it freezes the coefficients for any caller; train steps
    mask them beforehand as well (``mask_frozen_gradients``), which leaves them unchanged here.
    """

    inner: optax.GradientTransformation
    coefficient_release: CoefficientRelease

    def init(self, params):
        return self.inner.init(params)

    def update(self, updates, state, params=None):
        release = self.coefficient_release
        trainable = release.trainable(state)
        updates, state = self.inner.update(release.mask(updates, trainable, 0.0), state, params)
        # -0.0 is the exact additive identity: p + (-0.0) == p bitwise for every p, even -0.0.
        return release.mask(updates, trainable, -0.0), state


def mask_frozen_gradients(state, grads):
    """``grads`` as clipping and AdamW receive them, with frozen coefficient entries zeroed.

    Train steps apply this before the update, its numerical checks, and its telemetry, so the
    reported gradient norm is the one clipping sees. Without a schedule it returns ``grads``.
    """
    release = getattr(state.tx, "coefficient_release", None)
    if release is None:
        return grads
    return release.mask(grads, release.trainable(state.opt_state), 0.0)
