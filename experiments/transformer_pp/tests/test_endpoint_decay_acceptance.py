"""Endpoint ablations change exactly one gate choice and preserve the matched recipe."""
from copy import deepcopy
import hashlib

import pytest

from experiments.transformer_pp import endpoint_decay_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_test_paths
from scaffolds.transformer.configs import config_to_dict


def variant(conditioning="query_key", activation="softplus"):
    config = adapter.control_config()
    config.model.attention_forget_gate_conditioning = conditioning
    config.model.attention_forget_gate_rate_activation = activation
    return config


@pytest.mark.parametrize("settings", adapter.VARIANTS)
def test_each_approved_variant_preserves_budget_and_recipe(settings):
    config = variant(*settings)
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.train.seed == 42
    assert config.model.max_seq_len == 2048 and config.data.sequence_length == 2049
    assert config.model.attention_degree == 4 and config.model.positional_encoding == "none"
    assert config.model.attention_coefficient_conditioning == "query"
    assert config.model.attention_coefficient_kernel_weight_decay
    assert config.model.qk_norm_head_scale and not config.continuation.parent_checkpoint


def test_control_recipe_is_pinned():
    assert hashlib.sha256(adapter.CONTROL.read_bytes()).hexdigest() == adapter.CONTROL_SHA256
    from experiments.transformer_pp import query_distance_decay_acceptance as previous
    assert adapter.INITIALIZATION_SHA256 == previous.INITIALIZATION_SHA256
    assert adapter.PARAMETER_COUNT == previous.PARAMETER_COUNT


@pytest.mark.parametrize("settings", (("query", "softplus"), ("query_key", "sigmoid")))
def test_combined_or_unchanged_gates_are_not_ablation_runs(settings):
    with pytest.raises(ValueError):
        adapter.check_recipe(variant(*settings))


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "total_steps", 18925),
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", .0),
    ("optim", "warmup_steps", 1000),
    ("data", "batch_size", 128),
    ("train", "seed", 43),
    ("model", "attention_degree", 8),
    ("model", "positional_encoding", "rope"),
    ("model", "qk_norm_head_scale", False),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_amplitude_scale", "none"),
))
def test_unrelated_scientific_changes_are_rejected(section, field, value):
    config = variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("settings", adapter.VARIANTS)
def test_mosaic_kernels_are_an_execution_choice_but_other_backends_are_rejected(settings):
    config = variant(*settings)
    config.model.attention_backend = "mosaic"
    adapter.check_recipe(config)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    gpu = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert "scaffolds/transformer/tests/test_endpoint_decay.py" in gpu
    config.model.attention_backend = "fused"
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_execution_names_may_change():
    config = variant()
    config.train.workdir = "/new/run"
    config.train.checkpoint_dir = "/new/run/checkpoints"
    config.train.source_commit = "new_commit"
    config.train.wandb_run_name = "endpoint"
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("settings", adapter.VARIANTS)
def test_guards_and_endpoint_tests_are_installed_on_cpu_and_gpu(settings):
    config = variant(*settings)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization
    assert "experiments/transformer_pp/tests/test_endpoint_decay_acceptance.py" in spec.cpu_tests
    for cpu in (True, False):
        assert "scaffolds/transformer/tests/test_endpoint_decay.py" in selected_test_paths(
            spec, config_to_dict(config), resolved_backend="blocked", cpu=cpu)


@pytest.mark.parametrize("settings", adapter.VARIANTS)
def test_initialization_restores_full_control_hash_and_rejects_changes(monkeypatch, settings):
    """Use a tiny independent control state to exercise the full fingerprint guard cheaply."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.training.checkpointing import state_sha256

    config = variant(*settings)
    control = adapter.control_config()
    for current in (config, control):
        current.model.hidden_size = 8
        current.model.num_layers = 2
        current.model.num_heads = current.model.num_kv_heads = 2
        current.model.qk_head_dim = current.model.value_head_dim = 4
    monkeypatch.setattr(adapter, "control_config", lambda: deepcopy(control))
    def state(current):
        declared = initial_attention_parameter_values(
            attention_parameter_specs(current.model), num_layers=current.model.num_layers,
            num_heads=current.model.num_heads, key=mechanism_key(jax.random.PRNGKey(current.train.seed)))
        module = nnx.Module()
        for name, value in declared.items():
            setattr(module, name, nnx.Param(value))
        module.backbone = nnx.Param(jnp.arange(17, dtype=jnp.float32))
        return nnx.state(module, nnx.Param)
    expected = state(control)
    monkeypatch.setattr(adapter, "INITIALIZATION_SHA256", state_sha256(expected))
    monkeypatch.setattr(adapter, "PARAMETER_COUNT", sum(leaf.size for leaf in jax.tree.leaves(expected)))
    result = adapter._check_initialized_parameters(state(config), config)
    assert result["unchanged_parameters_match_control"]
    assert result["initial_zero_input_decay_rates_match_control"]
    broken = state(config)
    broken["backbone"] = broken["backbone"].replace(value=jnp.ones(17))
    with pytest.raises(RuntimeError, match="Restored initialization differs"):
        adapter._check_initialized_parameters(broken, config)
    broken = state(config)
    if settings[0] == "query_key":
        broken["forget_gate_key_kernel"] = broken["forget_gate_key_kernel"].replace(
            value=jnp.ones_like(broken["forget_gate_key_kernel"][...]))
        pattern = "key projection must initialize at zero"
    else:
        broken["forget_gate_dt_bias"] = broken["forget_gate_dt_bias"].replace(
            value=jnp.zeros_like(broken["forget_gate_dt_bias"][...]))
        pattern = "initial zero-input rates"
    with pytest.raises(RuntimeError, match=pattern):
        adapter._check_initialized_parameters(broken, config)
