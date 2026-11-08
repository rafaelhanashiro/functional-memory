"""The two degree-dependent gates preserve their controls' full training recipes."""
from copy import deepcopy
import pytest

from experiments.transformer_pp import per_order_decay_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_paths
from scaffolds.transformer.configs import config_to_dict


def variant(conditioning="query"):
    config = adapter.control_config(conditioning)
    config.model.attention_backend = "blocked"
    config.model.attention_forget_gate_per_order = True
    return config


@pytest.mark.parametrize("conditioning", ("query", "query_key"))
def test_approved_recipe_and_required_gpu_checks(conditioning):
    config = variant(conditioning)
    adapter.check_recipe(config)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    for cpu in (True, False):
        assert "scaffolds/transformer/tests/test_per_order_decay.py" in selected_paths(
            spec, config_to_dict(config), resolved_backend="blocked", cpu=cpu)
    assert spec.check_initialization is adapter.check_initialization


@pytest.mark.parametrize("section,key,value", (
    ("model", "attention_forget_gate_per_order", False),
    ("model", "attention_degree", 8), ("model", "positional_encoding", "rope"),
    ("model", "qk_norm_head_scale", False), ("model", "attention_coefficient_kernel_weight_decay", False),
    ("optim", "weight_decay", 0.), ("optim", "total_steps", 3000), ("train", "seed", 43),
))
def test_unrelated_changes_are_rejected(section, key, value):
    config = variant()
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("conditioning", ("query", "query_key"))
def test_initialization_guard_recovers_control_and_rejects_changed_order(monkeypatch, conditioning):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.training.checkpointing import state_sha256
    control, config = adapter.control_config(), variant(conditioning)
    for current in (control, config):
        current.model.hidden_size = 8
        current.model.num_layers = 2
        current.model.num_heads = current.model.num_kv_heads = 2
        current.model.qk_head_dim = current.model.value_head_dim = 4
    monkeypatch.setattr(adapter, "control_config", lambda conditioning="query": deepcopy(control))
    def state(current):
        values = initial_attention_parameter_values(attention_parameter_specs(current.model),
            num_layers=2, num_heads=2, key=mechanism_key(jax.random.PRNGKey(42)))
        module = nnx.Module()
        for name, value in values.items():
            setattr(module, name, nnx.Param(value))
        module.backbone = nnx.Param(jnp.arange(17, dtype=jnp.float32))
        return nnx.state(module, nnx.Param)
    expected = state(control)
    monkeypatch.setattr(adapter, "INITIALIZATION_SHA256", state_sha256(expected))
    monkeypatch.setattr(adapter, "PARAMETER_COUNT", sum(x.size for x in jax.tree.leaves(expected)))
    assert adapter._check_initialized_parameters(state(config), config)["all_order_gates_match_shared_control"]
    broken = state(config)
    value = broken["forget_gate_kernel"][...].at[:, :, 3, :].add(.01)
    broken["forget_gate_kernel"] = broken["forget_gate_kernel"].replace(value=value)
    with pytest.raises(RuntimeError, match="Initial parameter"):
        adapter._check_initialized_parameters(broken, config)
    broken = state(config)
    broken["backbone"] = broken["backbone"].replace(value=jnp.ones(17))
    with pytest.raises(RuntimeError, match="Restored initialization"):
        adapter._check_initialized_parameters(broken, config)
