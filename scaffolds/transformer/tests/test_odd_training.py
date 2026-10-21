"""Training-level invariants for the signed, fixed odd Taylor experiments."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.configs import load_config
from scaffolds.transformer.models import build_model
from scaffolds.transformer.models.registry import model_parameter_count
from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.tests.test_training import diagnostic_layer, tiny_config
from scaffolds.transformer.training import diagnostics, loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.distributed import (
    make_data_parallel_functions, put_replicated, put_train_batches, resolve_batch_layout,
)
from scaffolds.transformer.training.numerics import checked_update_loss
from scaffolds.transformer.training.state import create_state


def odd_config(path, degree):
    config = tiny_config(path)
    config.model.attention_degree = degree
    config.optim.learning_rate = 1e-4
    config.optim.min_learning_rate = 1e-5
    return config


@pytest.mark.parametrize("degree", [1, 3])
def test_odd_backbone_initialization_and_parameter_count_are_unchanged(tmp_path, degree):
    config = odd_config(tmp_path, degree)
    config.train.dtype = "bfloat16"
    odd = create_state(config, jax.random.PRNGKey(0))
    quadratic = create_state(replace(config, model=replace(config.model, attention_degree=2)), jax.random.PRNGKey(0))
    odd_flat = dict(nnx.to_flat_state(odd.params))
    quadratic_flat = dict(nnx.to_flat_state(quadratic.params))
    assert set(odd_flat) == set(quadratic_flat)
    assert "quadratic_coefficients" not in odd.params
    for key in odd_flat:
        assert odd_flat[key][...].dtype == jnp.float32
        np.testing.assert_array_equal(odd_flat[key][...], quadratic_flat[key][...])
    model = nnx.merge(odd.graphdef, odd.params)
    linear = [module for _, module in nnx.iter_graph(model) if isinstance(module, nnx.Linear)]
    assert linear and all(module.dtype == jnp.bfloat16 for module in linear)
    base = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml").model
    production = replace(base, attention_degree=degree)
    shaped = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    assert sum(value.size for value in jax.tree.leaves(shaped)) == 199_906_304 + production.num_layers * production.num_heads
    assert model_parameter_count(production) == 199_906_304 + production.num_layers * production.num_heads


@pytest.mark.parametrize("degree", [1, 3])
def test_odd_packed_training_resume_preserves_state_data_and_rng(tmp_path, degree):
    config = odd_config(tmp_path / "full", degree)
    config.data.source = "packed"
    for split, count in (("train", 16), ("eval", 4)):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5, num_sequences=count, shard_sequences=3)
        writer.write(np.arange(count * 5).reshape(count, 5) % 16)
        setattr(config.data, f"{split}_manifest", str(writer.finish()))
    full = loop.run_training(config, audit_checkpoint=True)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    paused = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    before, after = paused["checkpoint_audit"], resumed["checkpoint_audit"]
    assert before["final_state_sha256"] == before["checkpoint_state_sha256"] == after["start_state_sha256"]
    assert before["next_update"] == after["first_update"]
    assert after["final_state_sha256"] == after["checkpoint_state_sha256"]
    assert after["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    assert resumed["restored_step"] == 1
    assert resumed["data_start_sequence"] == config.data.batch_size
    assert full["attention_degree"] == resumed["attention_degree"] == degree
    assert full["attention_signed"] is resumed["attention_signed"] is True
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == config.optim.total_steps
            states.append(state)
        finally:
            manager.close()
    for first, second in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(first, second)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [2, 4]
    assert all(row["attention_degree"] == degree and row["attention_signed"] for row in records)
    for row in records:
        report = row["attention_diagnostics"]
        assert report["valid"]
        assert report["sample"]["eval_manifest_sha256"]
        assert report["sample"]["prediction_tokens"] == 4
        assert "coefficients" not in report
        for layer in report["layers"]:
            assert "min_abs_normalizer" in layer
            assert "negative_weight_fraction" in layer
            assert "cancellation_ratio_max" in layer
    json.dumps(records, allow_nan=False)


@pytest.mark.parametrize("degree", [1, 3])
def test_odd_signed_diagnostics_accept_negative_values_and_reject_breakdown(tmp_path, monkeypatch, degree):
    config = odd_config(tmp_path, degree)
    sample, provenance = diagnostics.fixed_diagnostic_sample(config)
    state = SimpleNamespace(graphdef=None, params={}, step=3)
    raw = diagnostic_layer(kernel_min=-3.0, normalizer_min=-2.0, min_abs_normalizer=0.5,
                           negative_weight_fraction=0.25, invalid_output_count=0)
    monkeypatch.setattr(diagnostics, "_diagnostic_step", lambda *args: (raw,))
    report = diagnostics.attention_diagnostics(state, config, sample, provenance)
    assert report["valid"] and report["attention_signed"]
    assert report["kernel_values"] == "unshifted_taylor"
    # An individual zero polynomial value is legal; a zero final row sum is not.
    raw["kernel_min"] = 0.0
    assert diagnostics.attention_diagnostics(state, config, sample, provenance)["valid"]
    raw.update(normalizer_min=0.0, min_abs_normalizer=0.0, invalid_normalizer_count=1)
    report = diagnostics.attention_diagnostics(state, config, sample, provenance)
    assert not report["valid"]
    assert any("invalid_normalizer_count" in issue for issue in report["issues"])
    raw.update(normalizer_min=-2.0, min_abs_normalizer=0.5, invalid_normalizer_count=0,
               score_max=np.inf, invalid_score_count=1)
    report = diagnostics.attention_diagnostics(state, config, sample, provenance)
    assert not report["valid"] and report["layers"][0]["score_max"] == "Infinity"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("degree", [1, 3])
def test_odd_finite_forward_cannot_hide_invalid_gradient_or_parameter_update(tmp_path, degree):
    config = odd_config(tmp_path, degree)
    params = create_state(config, jax.random.PRNGKey(0)).params
    grads = jax.tree.map(jnp.zeros_like, params)
    assert float(checked_update_loss(params, grads, params, jnp.array(2.0), check_all_gradients=True)) == 2.0
    broken_grads = jax.tree.map(lambda value: jnp.full_like(value, jnp.inf), grads)
    assert np.isnan(checked_update_loss(params, broken_grads, params, jnp.array(2.0), check_all_gradients=True))
    broken_params = jax.tree.map(lambda value: jnp.full_like(value, jnp.nan), params)
    assert np.isnan(checked_update_loss(params, grads, broken_params, jnp.array(2.0), check_all_gradients=True))


@pytest.mark.parametrize("degree", [1, 3])
def test_odd_finite_loss_invalid_gradient_persists_scientific_failure(tmp_path, monkeypatch, degree):
    config = odd_config(tmp_path, degree)
    config.train.log_every = config.train.eval_every = config.train.ckpt_every = 4

    def finite_loss_invalid_gradient(state, *args):
        return jnp.array(2.0), jax.tree.map(lambda value: jnp.full_like(value, jnp.inf), state.params)

    # Accumulation now traces the differentiation boundary into its scan.
    monkeypatch.setattr(loop, "loss_and_grads", finite_loss_invalid_gradient)
    loop.accumulated_train_step.clear_cache()
    with pytest.raises(FloatingPointError, match="step 1"):
        loop.run_training(config)
    failure = json.loads((tmp_path / "numerical_failure_step_00000001.json").read_text())
    assert failure["attention_degree"] == degree
    assert failure["attention_signed"] and failure["attention_kernel"] == "taylor"
    assert failure["loss"] == "NaN"
    assert {"forward_loss", "global_gradient_norm", "updated_parameters"} <= set(failure["checked_quantities"])
    assert failure["update_numerics"]["forward_loss"] == 2.0
    assert failure["update_numerics"]["global_gradient_norm"] == "Infinity"
    assert failure["training_tokens"] == 16
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "failed" and metadata["step"] == 1
    assert not (tmp_path / "completion.json").exists()
    json.dumps(failure, allow_nan=False)


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("tied", [False, True])
def test_odd_best_only_retention_preserves_terminal_metrics_and_resume_is_idempotent(tmp_path, monkeypatch, degree, tied):
    from scaffolds.transformer.retention import finalize_run

    name = f"test-m{degree:03d}"
    config = odd_config(tmp_path / "experiments" / "odd_taylor_attention" / "runs" / name, degree)
    config.train.source_commit = "a" * 40
    config.train.workdir.mkdir(parents=True)
    canonical = tmp_path / "experiments" / "odd_taylor_attention" / "checkpoints" / name
    canonical.mkdir(parents=True)
    (config.train.workdir / "checkpoints").symlink_to(canonical, target_is_directory=True)
    monkeypatch.setattr(loop, "evaluate", lambda state, *args, **kwargs: 3.0 if tied or int(state.step) == 2 else 4.0)
    outcome = loop.run_training(config)
    expected_best = 4 if tied else 2
    assert outcome["retained_best_checkpoint_step"] == expected_best
    original = (config.train.workdir / "completion.json").read_bytes()
    original_metrics = (config.train.workdir / "metrics.jsonl").read_bytes()
    record = finalize_run(config)
    assert record["completion_sha256"] == hashlib.sha256(original).hexdigest()
    assert sorted(path.name for path in canonical.iterdir() if path.name.isdigit()) == [str(expected_best)]
    assert (config.train.workdir / "completion.json").read_bytes() == original
    assert (config.train.workdir / "metrics.jsonl").read_bytes() == original_metrics
    assert loop.run_training(config)["final_step"] == 4
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["retained_terminal_checkpoint_step"] == (4 if tied else None)
    assert metadata["status"] == "complete"
    assert finalize_run(config) == record


@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two JAX devices")
@pytest.mark.parametrize("degree", [1, 3])
def test_odd_distributed_accumulation_and_numerical_checks_match_single_device(tmp_path, degree):
    config = odd_config(tmp_path, degree)
    state = create_state(config, jax.random.PRNGKey(0))
    batch = np.arange(20, dtype=np.int32).reshape(4, 5) % 16
    expected, expected_loss, expected_numerics = loop.train_step(
        state, jnp.asarray(batch), jax.random.PRNGKey(1), check_all_gradients=True,
    )
    layout = resolve_batch_layout(config)
    distributed = make_data_parallel_functions(state, tuple(jax.local_devices()), check_all_gradients=True)
    replicated = put_replicated(state, distributed.replicated_sharding)
    batches = put_train_batches(
        batch.reshape(layout.grad_accum_steps, layout.global_micro_batch_size, 5),
        distributed.train_batch_sharding,
    )
    actual, actual_loss, actual_numerics = distributed.train_step(replicated, batches, jax.random.PRNGKey(1))
    for first, second in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(expected_loss, actual_loss, rtol=1e-6)
    np.testing.assert_allclose(expected_numerics, actual_numerics, rtol=2e-6)
    assert np.all(np.isfinite(actual_numerics))
