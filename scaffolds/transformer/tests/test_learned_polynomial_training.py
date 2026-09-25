"""Training, resume, and failure reporting with learned unnormalized polynomial coefficients."""

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.numerics import checked_update_loss, requires_update_checks
from scaffolds.transformer.training.state import create_state


def _learned(config, degree=2, sharing="per_head"):
    config.model.attention_normalization = False
    config.model.attention_backend = "blocked"
    config.model.attention_degree = degree
    config.model.attention_coefficients = "learned"
    config.model.attention_coefficient_sharing = sharing
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    return config


@pytest.mark.parametrize("degree,sharing", [(1, "per_head"), (4, "per_head"), (2, "global")])
def test_learned_training_resume_and_identity(tmp_path, degree, sharing):
    config = _learned(tiny_config(tmp_path / "full"), degree, sharing)
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    initial_audit, resumed_audit = interrupted["checkpoint_audit"], resumed["checkpoint_audit"]
    assert initial_audit["final_state_sha256"] == initial_audit["checkpoint_state_sha256"]
    assert initial_audit["final_state_sha256"] == resumed_audit["start_state_sha256"]
    assert initial_audit["next_update"] == resumed_audit["first_update"]
    assert resumed["restored_step"] == 1
    assert full["attention_kernel"] == resumed["attention_kernel"] == "learned_polynomial_unnormalized"
    assert full["attention_coefficient_sharing"] == sharing
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == config.optim.total_steps
            states.append(state)
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    coefficients = np.asarray(states[0].params["polynomial_coefficients"][...])
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)], np.float32)
    assert coefficients.shape[-1] == degree + 1
    # Every coefficient, including the constant, is trained.
    assert np.all(coefficients.reshape(-1, degree + 1) != taylor)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validations = [record for record in records if record.get("event") == "validation"]
    assert validations and all(record["attention_diagnostics"]["valid"] for record in validations)
    reported = np.asarray(validations[-1]["attention_diagnostics"]["coefficients"]["effective"])
    np.testing.assert_allclose(reported, coefficients, rtol=1e-7)
    json.dumps(records, allow_nan=False)


def test_learned_coefficients_require_every_update_check(tmp_path):
    config = _learned(tiny_config(tmp_path))
    assert requires_update_checks(config.model)
    state = create_state(config, jax.random.PRNGKey(0))
    grads = jax.tree.map(jnp.zeros_like, state.params)
    grads["polynomial_coefficients"][...] = grads["polynomial_coefficients"][...].at[0, 0, 0].set(jnp.inf)
    loss = checked_update_loss(state.params, grads, state.params, jnp.array(2.0), check_all_gradients=True)
    assert np.isnan(loss)


def test_nonfinite_update_is_a_structured_numerical_failure(tmp_path, monkeypatch):
    config = _learned(tiny_config(tmp_path), 3)
    actual_step = loop.accumulated_train_step

    def poisoned_step(state, *args, **kwargs):
        state, loss, numerics = actual_step(state, *args, **kwargs)
        return state, jnp.asarray(jnp.nan, dtype=loss.dtype), numerics

    monkeypatch.setattr(loop, "accumulated_train_step", poisoned_step)
    with pytest.raises(FloatingPointError, match="optimizer update at step 1"):
        loop.run_training(config)
    report = json.loads((tmp_path / "numerical_failure_step_00000001.json").read_text())
    assert report["attention_kernel"] == "learned_polynomial_unnormalized"
    assert report["attention_coefficients"] == "learned"
    assert report["checked_quantities"] == ["forward_loss", "global_gradient_norm", "updated_parameters"]
