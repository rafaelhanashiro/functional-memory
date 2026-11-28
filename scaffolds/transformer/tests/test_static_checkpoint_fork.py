"""Matched early continuations of a learned static polynomial under its original LR curve."""

from copy import deepcopy
from dataclasses import replace
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from scaffolds.records import _tree_descriptor
from scaffolds.transformer.configs import ContinuationConfig, save_config, validate_config
from scaffolds.transformer.tests.test_coefficient_release import flat, release_config
from scaffolds.transformer.tests.test_continuation import assert_equal_tree
from scaffolds.transformer.training.continuation import initialize_continuation_state
from scaffolds.transformer.training.optim import make_learning_rate_schedule
from scaffolds.transformer.training.state import create_state


def test_static_checkpoint_fork_matches_uninterrupted_clipped_updates(tmp_path):
    prefix = release_config("static", None, total_steps=3, workdir=tmp_path)
    prefix.optim.schedule_steps = 12
    prefix.optim.warmup_steps = 2
    config_path = tmp_path / "config.yaml"
    save_config(prefix, config_path)
    state = create_state(prefix, jax.random.PRNGKey(prefix.train.seed))
    initial = flat(state.params)
    # Nonzero gradients and a dominant coefficient gradient make removing it
    # before clipping observably different for every other parameter.
    grads = jax.tree.map(lambda x: jnp.ones_like(x), state.params)
    grads["polynomial_coefficients"][...] *= 100.0
    for _ in range(3):
        state = state.apply_gradients(grads=grads)
    name = ("polynomial_coefficients",)
    baseline = flat(state.params)
    assert np.any(baseline[name] != initial[name])
    checkpoint = tmp_path / "checkpoint" / "3"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(checkpoint / "default", {"state": state})
    child = deepcopy(prefix)
    child.optim.total_steps = 8
    child.continuation = ContinuationConfig(
        parent_config=str(config_path), parent_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=_tree_descriptor(checkpoint)["sha256"],
        parent_step=3, preserve_all_parameters=True,
    )
    full, full_audit = initialize_continuation_state(child)
    frozen_config = deepcopy(child)
    frozen_config.continuation.freeze_static_coefficients = True
    frozen, frozen_audit = initialize_continuation_state(frozen_config)
    assert full_audit == frozen_audit
    assert full_audit["backbone_optimizer_count"] == 3
    assert full_audit["data_start_sequence"] == 3 * prefix.data.batch_size
    for fork in (full, frozen):
        assert_equal_tree(fork.params, state.params)
        assert_equal_tree(fork.opt_state, state.opt_state)
    ordinary = deepcopy(prefix)
    ordinary.optim = replace(ordinary.optim, total_steps=12, schedule_steps=None)
    counts = jnp.arange(12)
    for arm in (prefix, child, frozen_config):
        np.testing.assert_array_equal(make_learning_rate_schedule(arm)(counts),
                                      make_learning_rate_schedule(ordinary)(counts))

    # Drive identical gradients through the real optimizer. With the full
    # history restored, only the coefficient displacement may differ.
    for clock in range(4, 9):
        state = state.apply_gradients(grads=grads)
        full = full.apply_gradients(grads=grads)
        frozen = frozen.apply_gradients(grads=grads)
        assert int(frozen.step) == int(full.step) == clock
        assert_equal_tree(full.params, state.params)
        assert_equal_tree(full.opt_state, state.opt_state)
        assert_equal_tree(frozen.opt_state, full.opt_state)
        full_params, frozen_params = flat(full.params), flat(frozen.params)
        assert frozen_params[name].tobytes() == baseline[name].tobytes()
        assert np.any(full_params[name] != baseline[name])
        for path in full_params.keys() - {name}:
            np.testing.assert_array_equal(frozen_params[path], full_params[path])
        assert any(np.any(frozen_params[path] != baseline[path]) for path in full_params.keys() - {name})

    changed = deepcopy(child)
    changed.optim.schedule_steps += 1
    with pytest.raises(ValueError, match=r"parent optim.schedule_horizon"):
        initialize_continuation_state(changed)
    changed.optim.schedule_steps = changed.optim.total_steps - 1
    with pytest.raises(ValueError, match=r"schedule_steps must be an integer at least"):
        validate_config(changed)
    changed = deepcopy(frozen_config)
    changed.continuation.preserve_all_parameters = False
    with pytest.raises(ValueError, match="Freezing static coefficients requires"):
        validate_config(changed)
