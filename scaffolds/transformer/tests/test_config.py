from dataclasses import asdict
from pathlib import Path

import pytest

from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config, validate_config


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
    assert config.model.attention_backend == "auto"
    assert config.model.attention_degree is None
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
    ("model:\n  dropout: 0.1\n", "dropout=0"),
    ("model:\n  attention_degree: -2\n", "nonnegative integer"),
    ("model:\n  attention_degree: true\n", "nonnegative integer"),
    ("model:\n  attention_degree: 2.0\n", "nonnegative integer"),
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


@pytest.mark.parametrize("degree", (1, 3))
def test_fixed_odd_degree_configs_are_valid(tmp_path, degree):
    path = tmp_path / "odd.yaml"
    path.write_text(f"model:\n  attention_degree: {degree}\n")
    config = load_config(path)
    assert config.model.attention_degree == degree
    assert config.model.quadratic_mode == "fixed"


@pytest.mark.parametrize("qk_dim", (-2, 0, 3, 2.5, True))
def test_invalid_qk_dimensions_rejected_at_construction(qk_dim):
    with pytest.raises(ValueError):
        ModelConfig(qk_head_dim=qk_dim)


def test_scientific_recipe_matches_original_softmax_reference():
    import yaml

    original = yaml.safe_load((Path(__file__).resolve().parents[3] / "results/key_dimension/configs/base.yaml").read_text())
    current = yaml.safe_load(BASE_CONFIG.read_text())
    current["model"].pop("attention_degree")
    for config in (original, current):
        config["model"].pop("attention_backend")
        # The same immutable corpus lives at a different absolute path on Kennel.
        for key in ("train_manifest", "eval_manifest"):
            config["data"].pop(key)
        for key in ("workdir", "wandb_project", "wandb_run_name"):
            config["train"].pop(key)
    assert current == original


@pytest.mark.parametrize("degree", (0, 2, 4, 8, 16, None))
def test_degree_round_trip(tmp_path, degree):
    config = Config()
    config.model.attention_degree = degree
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path).model.attention_degree == degree


@pytest.mark.parametrize("degree", (0, 2, 4, 8, 16, None))
def test_fused_backend_is_explicit_and_round_trips(tmp_path, degree):
    config = Config()
    assert config.model.attention_backend == "auto"
    config.model.attention_backend = "fused"
    config.model.attention_degree = degree
    if degree is None:
        config.train.dtype = "bfloat16"
    validate_config(config)
    path = tmp_path / "fused.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.model.attention_backend == "fused"
    assert restored.model.attention_degree == degree


def test_fused_softmax_rejects_float32_training_before_execution():
    config = Config()
    config.model = ModelConfig(attention_backend="fused", attention_degree=None)
    with pytest.raises(ValueError, match="cuDNN.*train.dtype=bfloat16"):
        validate_config(config)
    config.train.dtype = "bfloat16"
    validate_config(config)


@pytest.mark.parametrize("qk_dim,value_dim,tile", [(64, 64, 64), (32, 65, 32), (128, 64, 32)])
def test_fused_diagnostics_metadata_reports_adaptive_tiles(qk_dim, value_dim, tile):
    from scaffolds.transformer.training.diagnostics import attention_identity

    config = Config()
    config.model = ModelConfig(attention_backend="fused", attention_degree=2,
                               qk_head_dim=qk_dim, value_head_dim=value_dim)
    identity = attention_identity(config)
    assert identity["attention_implementation"] == "pallas_triton"
    assert identity["query_block_size"] == identity["key_block_size"] == tile
    assert identity["diagnostic_backend"] == "blocked"
    assert identity["diagnostic_query_block_size"] == 128
    config.model.attention_degree = None
    identity = attention_identity(config)
    assert identity["attention_implementation"] == "cudnn"
    assert identity["query_block_size"] is identity["key_block_size"] is None
