from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity


BASE_CONFIG = Path(__file__).resolve().parents[5] / "results/unnormalized_attention/configs/base.yaml"


def test_baseline_matches_ofm_250m_budget_and_geometry():
    config = load_config(BASE_CONFIG)
    assert config.model.num_layers == 14
    assert config.model.hidden_size == 1024
    assert config.model.num_heads == config.model.num_kv_heads == 16
    assert config.model.qk_head_dim == config.model.value_head_dim == 64
    assert config.model.intermediate_size == 3008
    assert config.model.vocab_size == 32000
    assert config.model.tie_embeddings is False
    assert config.model.dropout == 0
    assert config.model.remat is False
    assert config.model.attention_backend == "auto"
    assert config.model.attention_normalization is False
    assert config.train.seed == config.data.shuffle_seed == 0
    assert config.optim.total_steps == 21935
    assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 11_500_257_280
    assert config.train.eval_batches * config.data.batch_size == 1792
    assert config.data.batch_size == 256 and config.train.micro_batch_size == 64
    assert config.optim.learning_rate == pytest.approx(3e-4)
    assert config.optim.min_learning_rate == pytest.approx(3e-5)
    assert config.optim.warmup_steps == 2000
    assert config.optim.beta1 == .9 and config.optim.beta2 == .95
    assert config.optim.weight_decay == .1 and config.optim.grad_clip_norm == 1.
    assert config.train.source_commit == "auto"


@pytest.mark.parametrize("normalized", (False, True))
def test_250m_parameter_count_is_unchanged(normalized):
    cfg = replace(load_config(BASE_CONFIG).model, attention_normalization=normalized, attention_backend="blocked")
    params = jax.eval_shape(lambda: nnx.state(build_model(cfg, 0), nnx.Param))
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    assert count == model_parameter_count(cfg) == 253_654_016
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))


def test_parameter_initialization_is_independent_of_normalization():
    cfg = ModelConfig(vocab_size=128, max_seq_len=16, num_layers=2, hidden_size=32,
                      num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                      intermediate_size=64, tie_embeddings=False, attention_backend="blocked")
    baseline = build_model(replace(cfg, attention_normalization=True), 0)
    expected, expected_tree = jax.tree.flatten(nnx.state(baseline, nnx.Param))
    for normalized in (False, True):
        model = build_model(replace(cfg, attention_normalization=normalized), 0)
        actual, actual_tree = jax.tree.flatten(nnx.state(model, nnx.Param))
        assert actual_tree == expected_tree
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(observed, reference)


def test_defaults_are_tiny_synthetic_and_round_trip(tmp_path):
    config = Config()
    validate_config(config)
    assert config.model.hidden_size == 128
    assert config.model.num_layers == 2
    assert config.data.source == "synthetic"
    assert config.train.wandb_mode == "disabled"
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert asdict(load_config(path)) == asdict(config)


@pytest.mark.parametrize("body,error", [
    ("model:\n  head_dim: 64\n", "head_dim"),
    ("model:\n  qk_head_dim: 3\n", "even"),
    ("model:\n  qk_head_dim: 0\n", "positive"),
    ("model:\n  num_kv_heads: 2\n", "full MHA"),
    ("model:\n  attention_backend: fast\n", "backend"),
    ("model:\n  dropout: 0.1\n", "dropout=0"),
    ("model:\n  attention_degree: 2\n  attention_normalization: false\n", "exact exponential"),
    ("model:\n  attention_normalization: 0\n", "boolean"),
    ("model:\n  attention_normalization: null\n", "boolean"),
    ('model:\n  attention_normalization: "false"\n', "boolean"),
    ("train:\n  micro_batch_size: 3\n", "divisible"),
    ("unknown: 1\n", "Unknown"),
    ("model: []\n", "mapping"),
    ("[]\n", "mapping"),
])
def test_reject_invalid_or_stale_configs(tmp_path, body, error):
    path = tmp_path / "invalid.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=error):
        load_config(path)


@pytest.mark.parametrize("normalized", (False, True))
def test_fused_backend_round_trip(tmp_path, normalized):
    config = Config()
    config.model.attention_backend = "fused"
    config.model.attention_normalization = normalized
    if normalized:
        config.train.dtype = "bfloat16"
    validate_config(config)
    path = tmp_path / "fused.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.model.attention_backend == "fused"
    assert restored.model.attention_normalization == normalized


def test_fused_softmax_rejects_float32_before_execution():
    config = Config()
    config.model = ModelConfig(attention_backend="fused", attention_normalization=True)
    with pytest.raises(ValueError, match="cuDNN.*train.dtype=bfloat16"):
        validate_config(config)
    config.train.dtype = "bfloat16"
    validate_config(config)


@pytest.mark.parametrize("qk_dim,value_dim,tile", [(64, 64, 64), (32, 65, 32), (128, 64, 32)])
def test_fused_diagnostics_metadata_reports_actual_implementation(qk_dim, value_dim, tile):
    config = Config()
    config.model = ModelConfig(attention_backend="fused", attention_normalization=False,
                               qk_head_dim=qk_dim, value_head_dim=value_dim)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "exp_unnormalized"
    assert identity["attention_implementation"] == "pallas_triton"
    assert identity["query_block_size"] == identity["key_block_size"] == tile
    assert identity["diagnostic_backend"] == "blocked"
    assert identity["diagnostic_query_block_size"] == 128
    assert identity["attention_output_normalization"] is False
    config.model.attention_normalization = True
    identity = attention_identity(config)
    assert identity["attention_implementation"] == "cudnn"
    assert identity["query_block_size"] is identity["key_block_size"] is None


@pytest.mark.parametrize("platform,dtype,width,expected", [
    ("cpu", "float32", 64, "blocked"),
    ("gpu", "float32", 64, "fused"),
    ("gpu", "bfloat16", 64, "fused"),
    ("gpu", "bfloat16", 256, "blocked"),
])
def test_automatic_backend_selects_exponential_implementation(platform, dtype, width, expected):
    from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
    model = ModelConfig(attention_normalization=False, qk_head_dim=width)
    assert resolve_attention_backend(model, platform=platform, dtype=dtype) == expected


def test_unnormalized_auto_choice_is_pinned_and_recorded_on_resume(tmp_path):
    import json
    from scaffolds.transformer.training.loop import ensure_run_config, _write_status
    config = Config()
    config.model.attention_normalization = False
    config.train.workdir = tmp_path
    ensure_run_config(config, resume=True)
    _write_status(config, status="initialized")
    assert config.model.attention_backend == "blocked"
    config.model.attention_backend = "auto"
    ensure_run_config(config, resume=True)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["attention_backend_requested"] == "auto"
    assert metadata["attention_implementation"] == "blocked"
    assert metadata["attention_kernel"] == "exp_unnormalized"
    config.model.attention_normalization = True
    with pytest.raises(ValueError, match="does not match"):
        ensure_run_config(config, resume=True)


def test_historical_config_defaults_compare_without_changing_original():
    from scaffolds.transformer.configs.loader import config_dicts_equal
    old = {"model": {"attention_normalization": False}, "train": {}}
    new = {"model": {"attention_normalization": False, "attention_degree": None,
                     "quadratic_mode": "fixed", "quadratic_sharing": "global"},
           "train": {"checkpoint_dir": None}}
    assert config_dicts_equal(old, new)
    assert old == {"model": {"attention_normalization": False}, "train": {}}
    assert not config_dicts_equal(old, {"model": {}, "train": {}})


def test_pilot_does_not_use_production_checkpoint_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scaffolds.transformer.studies.unnormalized_attention import preflight
    from scaffolds.transformer.training import loop
    config = Config()
    config.train.checkpoint_dir = tmp_path / "production-checkpoints"
    monkeypatch.setattr(preflight, "load_config", lambda path: config)
    captured = {}
    def run(pilot, **kwargs):
        captured.update(workdir=pilot.train.workdir, checkpoints=pilot.train.checkpoint_dir,
                        horizon=pilot.optim.total_steps, **kwargs)
        return captured
    monkeypatch.setattr(loop, "run_training", run)
    args = SimpleNamespace(worker="pilot", config=tmp_path / "config.yaml",
                           directory=tmp_path / "acceptance", stop_after_step=20)
    result = preflight.worker(args, {"source_commit": "frozen"})
    assert result["workdir"] == tmp_path / "acceptance/pilot"
    assert result["checkpoints"] is None
    assert result["horizon"] == config.optim.total_steps
    assert result["stop_after_step"] == 20 and result["resume"] is True
