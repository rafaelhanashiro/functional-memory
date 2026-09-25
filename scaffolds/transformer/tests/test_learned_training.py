"""Integration checks for learned coefficients, failure reporting, and retention."""

import hashlib
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.numerics import checked_update_loss
from scaffolds.transformer.training.state import create_state


@pytest.mark.parametrize("mode", ["positive", "unrestricted"])
@pytest.mark.parametrize("sharing", ["global", "per_head"])
def test_learned_training_resume_and_identity(tmp_path, mode, sharing):
    config = tiny_config(tmp_path / "full")
    config.model.attention_degree = 2
    config.model.quadratic_mode = mode
    config.model.quadratic_sharing = sharing
    config.optim.learning_rate = 1e-4
    config.optim.min_learning_rate = 1e-5
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    initial_audit, resumed_audit = interrupted["checkpoint_audit"], resumed["checkpoint_audit"]
    assert initial_audit["final_state_sha256"] == initial_audit["checkpoint_state_sha256"]
    assert initial_audit["final_state_sha256"] == resumed_audit["start_state_sha256"]
    assert initial_audit["next_update"] == resumed_audit["first_update"]
    assert resumed_audit["final_state_sha256"] == resumed_audit["checkpoint_state_sha256"]
    assert resumed["restored_step"] == 1
    assert resumed["data_start_sequence"] == config.data.batch_size
    assert full["quadratic_mode"] == resumed["quadratic_mode"] == mode
    assert full["quadratic_sharing"] == resumed["quadratic_sharing"] == sharing
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
    initial = np.array([1.0, 1.0 if mode == "positive" else 0.5], dtype=np.float32)
    assert not np.array_equal(np.asarray(states[0].params["quadratic_coefficients"].value).reshape(-1, 2)[0], initial)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert all(record["quadratic_mode"] == mode and record["quadratic_sharing"] == sharing for record in records)
    assert all(record["attention_diagnostics"]["valid"] for record in records)
    json.dumps(records, allow_nan=False)


def test_finite_forward_loss_does_not_hide_nonfinite_coefficient_gradient(tmp_path):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    config.model.quadratic_mode = "unrestricted"
    state = create_state(config, jax.random.PRNGKey(0))
    grads = jax.tree.map(jnp.zeros_like, state.params)
    grads["quadratic_coefficients"].value = jnp.array([jnp.inf, 0.0])
    loss = checked_update_loss(state.params, grads, state.params, jnp.array(2.0))
    assert np.isnan(loss)


def test_checkpoint_audit_rejects_finite_restoration_corruption(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    config.model.quadratic_mode = "positive"
    actual_restore = loop.restore_latest_checkpoint

    def corrupt_restore(manager, template):
        restored, step = actual_restore(manager, template)
        if step is not None:
            coefficients = restored.params["quadratic_coefficients"]
            coefficients.value = coefficients.value + jnp.array([1e-3, 0.0])
        return restored, step

    monkeypatch.setattr(loop, "restore_latest_checkpoint", corrupt_restore)
    with pytest.raises(RuntimeError, match="differs from its durable checkpoint"):
        loop.run_training(config, stop_after_step=1, audit_checkpoint=True)


def test_run_config_without_quadratic_fields_resumes_unchanged(tmp_path):
    config = tiny_config(tmp_path)
    loop.ensure_run_config(config, resume=True)
    path = tmp_path / "config.json"
    historical = json.loads(path.read_text())
    historical["model"].pop("quadratic_mode")
    historical["model"].pop("quadratic_sharing")
    path.write_text(json.dumps(historical))
    original = path.read_bytes()
    loop.ensure_run_config(config, resume=True)
    assert path.read_bytes() == original
    config.model.attention_degree = 2
    config.model.quadratic_mode = "positive"
    with pytest.raises(ValueError, match="does not match"):
        loop.ensure_run_config(config, resume=True)


def test_finite_large_nll_is_a_structured_numerical_failure(tmp_path):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    config.model.quadratic_mode = "unrestricted"
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = loop.fixed_diagnostic_sample(config)
    with pytest.raises(FloatingPointError, match="Perplexity overflow"):
        loop._perplexity(config, state, 1, 1000.0, sample, provenance)
    report = json.loads((tmp_path / "numerical_failure_step_00000001.json").read_text())
    assert report["loss"] == 1000.0
    assert report["quadratic_mode"] == "unrestricted"


@pytest.mark.parametrize("tied", [False, True])
def test_retention_keeps_best_and_completed_run_is_idempotent(tmp_path, monkeypatch, tied):
    from scaffolds.transformer.retention import finalize_run

    name = "test-positive-global"
    config = tiny_config(tmp_path / "experiments" / "learned_quadratic_attention" / "runs" / name)
    config.model.attention_degree = 2
    config.model.quadratic_mode = "positive"
    config.train.source_commit = "a" * 40
    config.train.workdir.mkdir(parents=True)
    canonical = tmp_path / "experiments" / "learned_quadratic_attention" / "checkpoints" / name
    canonical.mkdir(parents=True)
    (config.train.workdir / "checkpoints").symlink_to(canonical, target_is_directory=True)
    actual_evaluate = loop.evaluate

    def earlier_best(state, *args, **kwargs):
        if tied:
            return 3.0
        return actual_evaluate(state, *args, **kwargs) + (1.0 if int(state.step) == 4 else 0.0)

    monkeypatch.setattr(loop, "evaluate", earlier_best)
    outcome = loop.run_training(config)
    expected_best = 4 if tied else 2
    assert outcome["retained_best_checkpoint_step"] == expected_best
    completion_path = config.train.workdir / "completion.json"
    original = completion_path.read_bytes()
    retention = finalize_run(config)
    assert retention["completion_sha256"] == hashlib.sha256(original).hexdigest()
    assert sorted(path.name for path in canonical.iterdir() if path.name.isdigit()) == [str(expected_best)]
    assert completion_path.read_bytes() == original
    assert loop.run_training(config)["final_step"] == 4
    assert finalize_run(config) == retention
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["retained_terminal_checkpoint_step"] == (4 if tied else None)
    assert metadata["status"] == "complete"
