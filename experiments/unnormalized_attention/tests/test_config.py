"""The canonical unnormalized recipe keeps the 200M sweep budget and geometry."""
from dataclasses import asdict, replace
from pathlib import Path
from scaffolds.records import load_study, run_directory

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import nnx

from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity


ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = run_directory(ROOT / "experiments/unnormalized_attention", "200m-exp-seed0-kennel") / "config.yaml"
QK_NORM_CONFIG = run_directory(ROOT / "experiments/unnormalized_attention", "200m-qk-exp-seed0") / "config.yaml"


def test_recipe_matches_200m_sweep_budget_and_geometry():
    config = load_config(BASE_CONFIG)
    assert config.model.num_layers == 10
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
    assert config.optim.total_steps == 19074
    assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10_000_269_312
    assert config.train.eval_batches * config.data.batch_size == 1792
    assert config.data.batch_size == 256 and config.train.micro_batch_size == 64
    assert config.optim.learning_rate == pytest.approx(3e-4)
    assert config.optim.min_learning_rate == pytest.approx(3e-5)
    assert config.optim.warmup_steps == 2000
    assert config.optim.beta1 == .9 and config.optim.beta2 == .95
    assert config.optim.weight_decay == .1 and config.optim.grad_clip_norm == 1.
    assert config.train.source_commit == "auto"

@pytest.mark.parametrize("normalized", (False, True))
def test_200m_parameter_count_is_unchanged(normalized):
    cfg = replace(load_config(BASE_CONFIG).model, attention_normalization=normalized, attention_backend="blocked")
    params = jax.eval_shape(lambda: nnx.state(build_model(cfg, 0), nnx.Param))
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    assert count == model_parameter_count(cfg) == 199_906_304
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))


def test_qk_norm_rerun_changes_only_scores_decay_backend_and_tracking():
    raw = yaml.safe_load(QK_NORM_CONFIG.read_text())
    assert raw["model"]["qk_norm"] is True and raw["model"]["qk_norm_sharing"] == "per_head"
    assert raw["model"]["qk_norm_initial_scale"] is None and raw["model"]["attention_backend"] == "fused"
    assert raw["optim"]["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"
    rerun, base = config_to_dict(load_config(QK_NORM_CONFIG)), config_to_dict(load_config(BASE_CONFIG))
    for recipe in (rerun, base):
        for name in ("qk_norm", "attention_backend"):
            recipe["model"].pop(name)
        for name in ("workdir", "checkpoint_dir", "wandb_project", "wandb_entity", "wandb_run_name",
                     "wandb_group", "wandb_mode"):
            recipe["train"].pop(name)
    assert rerun == base
    assert model_parameter_count(load_config(QK_NORM_CONFIG).model) == 199_906_304 + 160
