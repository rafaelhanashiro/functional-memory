from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.key_dimension.configs import Config, ModelConfig, load_config, save_config, validate_config


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def test_baseline_matches_approved_budget_and_geometry():
    config = load_config(BASE_CONFIG)
    assert config.model.num_layers == 10
    assert config.model.hidden_size == 1024
    assert config.model.num_heads == config.model.num_kv_heads == 16
    assert config.model.qk_head_dim == config.model.value_head_dim == 64
    assert config.model.intermediate_size == 3008
    assert config.model.vocab_size == 32000
    assert config.model.tie_embeddings is False
    assert config.model.dropout == 0
    assert config.model.attention_backend == "cudnn"
    assert config.train.seed == config.data.shuffle_seed == 0
    assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10_000_269_312
    assert config.train.eval_batches * config.data.batch_size == 1792
    assert config.data.batch_size // config.train.micro_batch_size == 4
    assert config.optim.learning_rate == pytest.approx(3e-4)
    assert config.optim.min_learning_rate == pytest.approx(3e-5)
    assert config.optim.warmup_steps == 2000
    assert config.train.source_commit == "auto"


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
    ("model:\n  attention_backend: xla\n  dropout: 0.1\n", "dropout=0"),
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


@pytest.mark.parametrize("qk_dim", (-2, 0, 3, 2.5, True))
def test_invalid_qk_dimensions_rejected_at_construction(qk_dim):
    with pytest.raises(ValueError):
        ModelConfig(qk_head_dim=qk_dim)
