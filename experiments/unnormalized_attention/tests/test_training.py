import json
import math
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.unnormalized_attention.configs import Config, ModelConfig
from experiments.unnormalized_attention.data.packed import FORMAT_NAME, FORMAT_VERSION
from experiments.unnormalized_attention.training import loop
from experiments.unnormalized_attention.training import diagnostics as diagnostic_tools
from experiments.unnormalized_attention.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from experiments.unnormalized_attention.training.distributed import (
    make_data_parallel_functions, put_replicated, put_train_batches, resolve_batch_layout,
)
from experiments.unnormalized_attention.training.state import create_state
from experiments.unnormalized_attention.train import should_resume


def tiny_config(path):
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=16, max_seq_len=4, num_layers=1, hidden_size=8,
                            num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
                            intermediate_size=12, tie_embeddings=False)
    cfg.data.synthetic_vocab_size, cfg.data.sequence_length, cfg.data.batch_size = 16, 5, 4
    cfg.train.micro_batch_size = 2
    cfg.train.workdir = path
    cfg.train.eval_every, cfg.train.eval_batches, cfg.train.ckpt_every, cfg.train.log_every = 2, 1, 2, 1
    cfg.optim.total_steps, cfg.optim.warmup_steps = 4, 0
    # Force clipping to catch clipping-before-accumulation errors.
    cfg.optim.grad_clip_norm = 0.01
    return cfg


def diagnostic_layer(**updates):
    return {
        "score_min": -1.0, "score_max": 1.0,
        "kernel_min": 0.5, "kernel_max": 2.5,
        "weight_sum_min": 0.5, "weight_sum_max": 5.0,
        "invalid_score_count": 0, "invalid_kernel_count": 0,
        "invalid_weight_sum_count": 0, "invalid_output_count": 0, **updates,
    }


def assert_state_close(first, second, *, atol=1e-7):
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=atol)


def test_accumulation_matches_full_batch_before_clipping(tmp_path):
    cfg = tiny_config(tmp_path)
    rng = jax.random.PRNGKey(0)
    state = create_state(cfg, rng)
    batch = jnp.asarray(np.arange(20).reshape(4, 5) % 16)
    full, loss, norm = loop.train_step(state, batch, rng)
    accum, accum_loss, accum_norm = loop.accumulated_train_step(state, batch.reshape(2, 2, 5), jax.random.split(rng, 2))
    assert_state_close(full, accum)
    np.testing.assert_allclose(loss, accum_loss, rtol=1e-6)
    np.testing.assert_allclose(norm, accum_norm, rtol=1e-6)


def test_bfloat16_compute_preserves_float32_parameters(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.train.dtype = "bfloat16"
    state = create_state(cfg, jax.random.PRNGKey(0))
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(state.params))
    from flax import nnx
    model = nnx.merge(state.graphdef, state.params)
    linear_layers = [obj for _, obj in nnx.iter_graph(model) if isinstance(obj, nnx.Linear)]
    assert linear_layers and all(layer.dtype == jnp.bfloat16 for layer in linear_layers)


def test_token_nll_weighting_and_perplexity(monkeypatch):
    batches = [np.full((2, 3), 1, dtype=np.int32), np.full((1, 5), 3, dtype=np.int32)]
    monkeypatch.setattr(loop, "eval_step", lambda state, batch, rng: float(batch[0, 0]))
    nll = loop.evaluate(None, iter(batches), 2, micro_batch_size=1)
    assert nll == 2.0  # Each batch contributes four prediction tokens.
    assert math.exp(nll) != (math.exp(1) + math.exp(3)) / 2
    logits = jnp.zeros((2, 4, 16))
    assert float(loop.cross_entropy_loss(logits, jnp.zeros((2, 4), jnp.int32))) == pytest.approx(math.log(16))


def test_interrupted_packed_training_matches_uninterrupted_and_retains_terminal(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path / "full")
    cfg.data.source = "packed"
    for split, count in (("train", 16), ("eval", 4)):
        directory = tmp_path / split
        directory.mkdir()
        np.save(directory / "shard.npy", (np.arange(count * 5).reshape(count, 5) % 16).astype("uint16"))
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({
            "format": FORMAT_NAME, "version": FORMAT_VERSION, "dtype": "uint16",
            "sequence_length": 5, "num_sequences": count, "num_tokens": count * 5,
            "metadata": {}, "shards": [{"path": "shard.npy", "sequence_start": 0, "num_sequences": count}],
        }))
        setattr(cfg.data, f"{split}_manifest", str(manifest))
    # Make the earlier validation strictly better: both checkpoints must survive.
    actual_evaluate = loop.evaluate
    def staged_loss(state, *args, **kwargs):
        return actual_evaluate(state, *args, **kwargs) + (1 if int(state.step) == 4 else 0)
    monkeypatch.setattr(loop, "evaluate", staged_loss)
    loop.run_training(cfg)
    resumed_cfg = replace(cfg, train=replace(cfg.train, workdir=tmp_path / "resumed"))
    paused = loop.run_training(resumed_cfg, stop_after_step=1)
    assert paused["final_step"] == 1
    resumed = loop.run_training(resumed_cfg)
    assert resumed["restored_step"] == 1 and resumed["data_start_sequence"] == 4
    assert resumed["retained_terminal_checkpoint_step"] == 4
    assert resumed["retained_best_checkpoint_step"] == 2
    states = []
    for current_cfg in (cfg, resumed_cfg):
        manager = make_checkpoint_manager(current_cfg)
        try:
            assert manager.all_steps() == [2, 4]
            state, step = restore_latest_checkpoint(manager, create_state(current_cfg, jax.random.PRNGKey(0)))
            assert step == 4
            states.append(state)
        finally:
            manager.close()
    for uninterrupted, restarted in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(uninterrupted, restarted)
    before = (resumed_cfg.train.workdir / "metrics.jsonl").read_bytes()
    assert loop.run_training(resumed_cfg)["final_step"] == 4
    assert (resumed_cfg.train.workdir / "metrics.jsonl").read_bytes() == before
    records = [json.loads(line) for line in before.splitlines()]
    assert [record["step"] for record in records] == [2, 4]
    completion = json.loads((resumed_cfg.train.workdir / "completion.json").read_text())
    assert completion["training_tokens"] == 64
    assert completion["validation_perplexity"] == math.exp(records[-1]["validation_nll"])
    assert completion["attention_normalization"] is False
    assert completion["attention_kernel"] == "exp_unnormalized"
    assert completion["attention_backend"] == "blocked"
    assert completion["attention_diagnostics"] == records[-1]["attention_diagnostics"]
    for record in records:
        diagnostic = record["attention_diagnostics"]
        assert diagnostic["valid"]
        assert diagnostic["step"] == record["step"]
        assert diagnostic["sample"]["num_sequences"] == 1
        assert diagnostic["sample"]["prediction_tokens"] == 4
        assert diagnostic["sample"]["sequence_index"] == 0
        assert diagnostic["sample"]["eval_manifest_sha256"]
        assert record["validation_sequences"] == 4
        assert record["validation_tokens"] == 16
    assert records[0]["attention_diagnostics"]["sample"] == records[1]["attention_diagnostics"]["sample"]


def test_resume_config_drift_rejected_and_slurm_requeue_resumes(tmp_path):
    cfg = tiny_config(tmp_path)
    loop.ensure_run_config(cfg, resume=True)
    cfg.model.qk_head_dim = 4
    with pytest.raises(ValueError, match="does not match"):
        loop.ensure_run_config(cfg, resume=True)
    assert not should_resume(no_resume=True, slurm_restart_count="0")
    assert should_resume(no_resume=True, slurm_restart_count="1")


def test_attention_diagnostics_detect_nonfinite_and_nonpositive_values(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    sample, provenance = diagnostic_tools.fixed_diagnostic_sample(cfg)
    state = SimpleNamespace(graphdef=None, params={}, step=3)
    raw = diagnostic_layer(score_max=np.nan, kernel_min=-1.0, invalid_score_count=2)
    monkeypatch.setattr(diagnostic_tools, "_diagnostic_step", lambda *args: (raw,))
    report = diagnostic_tools.attention_diagnostics(state, cfg, sample, provenance)
    assert not report["valid"]
    assert report["layers"][0]["score_max"] == "NaN"
    assert report["layers"][0]["invalid_score_count"] == 2
    assert any("kernel_min" in issue for issue in report["issues"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("weight_sum", (0.0, float("inf")))
def test_unnormalized_weight_sum_is_diagnostic_only(tmp_path, monkeypatch, weight_sum):
    cfg = tiny_config(tmp_path)
    state = SimpleNamespace(graphdef=None, params={}, step=0)
    sample, provenance = diagnostic_tools.fixed_diagnostic_sample(cfg)
    raw = diagnostic_layer(kernel_min=0.0, weight_sum_min=weight_sum, weight_sum_max=weight_sum)
    monkeypatch.setattr(diagnostic_tools, "_diagnostic_step", lambda *args: (raw,))
    report = diagnostic_tools.attention_diagnostics(state, cfg, sample, provenance)
    assert report["valid"]
    json.dumps(report, allow_nan=False)
    cfg.model.attention_normalization = True
    assert not diagnostic_tools.attention_diagnostics(state, cfg, sample, provenance)["valid"]


def test_nonfinite_loss_between_log_events_persists_failure(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    cfg.train.log_every, cfg.train.eval_every, cfg.train.ckpt_every = 3, 4, 4
    actual_step = loop.accumulated_train_step

    def nonfinite_second_step(*args, **kwargs):
        state, loss, norm = actual_step(*args, **kwargs)
        return state, jnp.nan if int(state.step) == 2 else loss, norm

    monkeypatch.setattr(loop, "accumulated_train_step", nonfinite_second_step)
    with pytest.raises(FloatingPointError, match="training loss at step 2"):
        loop.run_training(cfg)
    failure = json.loads((tmp_path / "numerical_failure_step_00000002.json").read_text())
    assert failure["step"] == 2 and failure["loss"] == "NaN"
    assert failure["attention_diagnostics"]["sample"]["num_sequences"] == 1
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "failed" and metadata["step"] == 2
    assert not (tmp_path / "completion.json").exists()


def test_nonfinite_gradient_with_finite_loss_persists_failure(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    actual_step = loop.accumulated_train_step

    def nonfinite_gradient(*args, **kwargs):
        state, loss, _ = actual_step(*args, **kwargs)
        return state, loss, jnp.inf

    monkeypatch.setattr(loop, "accumulated_train_step", nonfinite_gradient)
    with pytest.raises(FloatingPointError, match="gradient norm at step 1"):
        loop.run_training(cfg)
    failure = json.loads((tmp_path / "numerical_failure_step_00000001.json").read_text())
    assert math.isfinite(failure["loss"])
    assert failure["gradient_norm"] == "Infinity"
    assert not (tmp_path / "completion.json").exists()


def test_invalid_attention_diagnostics_fail_before_validation_commit(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    original = loop.attention_diagnostics

    def invalid_kernel(*args):
        report = original(*args)
        report["valid"] = False
        report["issues"] = ["layer 0: invalid_kernel_count=1"]
        report["layers"][0]["invalid_kernel_count"] = 1
        return report

    monkeypatch.setattr(loop, "attention_diagnostics", invalid_kernel)
    with pytest.raises(FloatingPointError, match="Invalid attention diagnostics at step 2"):
        loop.run_training(cfg)
    failure = json.loads((tmp_path / "numerical_failure_step_00000002.json").read_text())
    assert not failure["attention_diagnostics"]["valid"]
    assert not (tmp_path / "metrics.jsonl").exists()
    assert not (tmp_path / "completion.json").exists()


@pytest.mark.gpu
@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two JAX devices")
def test_distributed_accumulation_matches_single_device(tmp_path):
    cfg = tiny_config(tmp_path)
    state = create_state(cfg, jax.random.PRNGKey(0))
    batch = np.arange(20, dtype=np.int32).reshape(4, 5) % 16
    expected, expected_loss, expected_norm = loop.train_step(state, jnp.asarray(batch), jax.random.PRNGKey(1))
    layout = resolve_batch_layout(cfg)
    distributed = make_data_parallel_functions(state, tuple(jax.local_devices()))
    replicated = put_replicated(state, distributed.replicated_sharding)
    batches = put_train_batches(batch.reshape(layout.grad_accum_steps, layout.global_micro_batch_size, 5), distributed.train_batch_sharding)
    actual, actual_loss, actual_norm = distributed.train_step(replicated, batches, jax.random.PRNGKey(1))
    assert_state_close(expected, actual)
    np.testing.assert_allclose(expected_loss, actual_loss, rtol=1e-6)
    np.testing.assert_allclose(expected_norm, actual_norm, rtol=1e-6)
    sample, provenance = diagnostic_tools.fixed_diagnostic_sample(cfg)
    report = diagnostic_tools.attention_diagnostics(actual, cfg, sample, provenance)
    assert report["valid"] and report["sample"]["num_sequences"] == 1
