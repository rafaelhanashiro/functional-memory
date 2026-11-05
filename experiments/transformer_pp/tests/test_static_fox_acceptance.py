"""Guard the static FoX ablation's recipe, initialization, and parameter decay."""
from copy import deepcopy
from dataclasses import replace

import jax
from flax import nnx
import pytest

from experiments.transformer_pp import static_fox_acceptance as adapter
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256


def _variant():
    config = adapter.control_config()
    config.model = replace(config.model, **adapter.STATIC_SETTINGS)
    return config


def test_recipe_is_static_fox_with_same_training_budget_and_gate_decay():
    config = _variant()
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.train.seed == 42
    assert config.model.attention_degree == 4 and config.model.max_seq_len == 2048
    assert config.model.attention_coefficient_scale == "factorial"
    assert config.model.attention_forget_gate and config.model.positional_encoding == "none"
    assert config.optim.weight_decay == .1
    assert not config.continuation.parent_checkpoint


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "learning_rate", .002), ("optim", "weight_decay", 0.0),
    ("optim", "total_steps", 7571), ("optim", "warmup_steps", 500),
    ("train", "seed", 43), ("train", "eval_every", 1000),
    ("data", "batch_size", 128), ("data", "shuffle_seed", 1),
    ("model", "attention_forget_gate", False), ("model", "positional_encoding", "rope"),
    ("model", "attention_coefficient_scale", "none"),
    ("model", "attention_output_scaling", "none"),
    ("model", "qk_norm_head_scale", False),
))
def test_unrelated_scientific_changes_are_rejected(section, field, value):
    config = _variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_control_cannot_be_submitted_as_static():
    with pytest.raises(ValueError, match="static FoX ablation"):
        adapter.check_recipe(adapter.control_config())


@pytest.fixture
def small_control(monkeypatch):
    control = adapter.control_config()
    control.model = replace(control.model, vocab_size=16, max_seq_len=8, num_layers=2,
                            hidden_size=8, num_heads=2, num_kv_heads=2, qk_head_dim=4,
                            value_head_dim=4, intermediate_size=16)
    control.data.sequence_length, control.data.batch_size = 9, 2
    control.train.micro_batch_size = 2
    monkeypatch.setattr(adapter, "control_config", lambda: deepcopy(control))
    rng = jax.random.PRNGKey(control.train.seed)
    params = nnx.state(build_model(control.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0))), nnx.Param)
    monkeypatch.setattr(adapter, "CONTROL_INITIALIZATION_SHA256", state_sha256(params))
    return control


def test_actual_parameters_and_masks_match_the_control(small_control):
    result = adapter.check_initialization(_variant())
    assert result["shared_initial_parameters_match_control"]
    assert result["gate_initial_parameters_match_control"]
    assert result["shared_decay_masks_match_control"]
    assert result["static_coefficient_shape"] == [2, 2, 5]
    assert result["attention_decay_masks"] == {
        "polynomial_coefficients": False, "forget_gate_kernel": True,
        "forget_gate_log_rate": False, "forget_gate_dt_bias": False,
    }


@pytest.mark.parametrize("name", ["forget_gate_kernel", "polynomial_coefficients"])
def test_corrupted_initialization_fails(small_control, monkeypatch, name):
    from scaffolds.transformer import models

    def corrupted(config, rngs):
        model = build_model(config, rngs)
        if config.attention_coefficient_conditioning == "static":
            getattr(model, name)[...] += 0.1
        return model

    monkeypatch.setattr(models, "build_model", corrupted)
    with pytest.raises(RuntimeError, match="Initialization differs|Static coefficients"):
        adapter.check_initialization(_variant())


def test_adapter_installs_both_guards():
    spec = adapter.experiment_spec(_variant(), entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization
