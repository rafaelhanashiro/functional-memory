"""The degree-dependent gates preserve their controls' full training recipes."""
from copy import deepcopy
import json

import pytest

from experiments.transformer_pp import per_order_decay_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_paths
from scaffolds.transformer.configs import config_to_dict, load_config

APPROVED = (("query", 4), ("query_key", 4), ("query", 8), ("query", 12), ("query_key", 8), ("query_key", 12))
# The per-order query-key runs at degrees eight and twelve and their per-order query twins.
QUERY_KEY_RUNS = ((8, "r227b0ed7c605", "r20fe6fe864bc"), (12, "rc2f9273c49cc", "rfe140c073b2f"))


def variant(conditioning="query", degree=4):
    config = adapter.control_config(conditioning, degree)
    config.model.attention_backend = "mosaic"
    config.model.attention_forget_gate_per_order = True
    config.model.attention_forget_gate_conditioning = conditioning
    return config


def run_config(run_id):
    return load_config(adapter.ROOT / "runs" / run_id / "config.yaml")


@pytest.mark.parametrize("conditioning,degree", APPROVED)
def test_approved_recipe_and_required_gpu_checks(conditioning, degree):
    config = variant(conditioning, degree)
    adapter.check_recipe(config)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    for cpu in (True, False):
        # The whole per-order file: its GPU cases cover both gates at degrees 4, 8 and 12 (with the
        # degree-twelve top orders alone), and the endpoint file the shared Mosaic decay kernels.
        paths = selected_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=cpu)
        assert {"scaffolds/transformer/tests/test_per_order_decay.py",
                "scaffolds/transformer/tests/test_endpoint_decay.py"} <= set(paths)
    assert spec.check_initialization is adapter.check_initialization


@pytest.mark.parametrize("degree,run_id", ((8, "r20fe6fe864bc"), (12, "rfe140c073b2f"),
                                           (8, "r227b0ed7c605"), (12, "rc2f9273c49cc")))
def test_frozen_higher_degree_runs_are_their_shared_gate_controls_with_per_order_gates(degree, run_id):
    from scaffolds.transformer.models import model_parameter_count

    run = run_config(run_id)
    adapter.check_recipe(run)
    control = adapter.control_config("query", degree)
    assert adapter.control_config(run.model.attention_forget_gate_conditioning, degree) == control
    assert run.model.attention_degree == control.model.attention_degree == degree
    assert run.model.attention_backend == control.model.attention_backend == "mosaic"
    assert control.model.attention_forget_gate_mode == "query_distance" and not control.model.attention_forget_gate_per_order
    assert control.model.attention_forget_gate_conditioning == "query"
    assert model_parameter_count(control.model) == adapter.PARAMETER_COUNT[degree]
    orders, model = degree + 1, run.model
    shared_gate, key = ((model.hidden_size + extra) * model.num_layers * model.num_heads for extra in (2, 0))
    endpoint = model.attention_forget_gate_conditioning == "query_key"
    assert model_parameter_count(model) == (adapter.PARAMETER_COUNT[degree] + (orders - 1) * shared_gate
                                            + (orders * key if endpoint else 0))
    for pinned in (4, 8, 12):
        record = json.loads((adapter.ROOT / "runs" / adapter.CONTROLS[pinned, "query"][0] / "record.json").read_text())
        assert record["config"]["sha256"] == adapter.CONTROLS[pinned, "query"][1]


@pytest.mark.parametrize("degree,run_id,twin", QUERY_KEY_RUNS)
def test_query_key_runs_change_only_their_twins_gate_conditioning(degree, run_id, twin):
    """Each run is its per-order query twin with query-key gates, and the three per-order query-key
    runs are one recipe at degrees 4, 8 and 12. Query conditioning therefore gives back the twin,
    which the adapter qualifies as such."""
    def scientific(config, **changes):
        for key, value in changes.items():
            setattr(config.model, key, value)
        recipe = config_to_dict(config)
        for key in ("workdir", "checkpoint_dir", "wandb_run_name"):
            recipe["train"].pop(key)
        return recipe
    assert adapter.CONTROLS[degree, "query_key"] == adapter.CONTROLS[degree, "query"]
    assert run_config(run_id).model.attention_forget_gate_conditioning == "query_key"
    assert scientific(run_config(run_id)) == scientific(run_config(twin), attention_forget_gate_conditioning="query_key")
    for other, run in ((4, "r03be769d2069"), *((m, r) for m, r, _ in QUERY_KEY_RUNS)):
        assert scientific(run_config(run_id), attention_degree=other) == scientific(run_config(run))
    query = run_config(run_id)
    query.model.attention_forget_gate_conditioning = "query"
    adapter.check_recipe(query)
    assert scientific(query) == scientific(run_config(twin))


@pytest.mark.parametrize("section,key,value,reason", (
    ("model", "attention_degree", 6, "No shared-gate control"),
    ("model", "attention_degree", 10, "No shared-gate control"),
    ("model", "attention_forget_gate_conditioning", "none", "No shared-gate control"),
    ("model", "attention_forget_gate_per_order", False, "independent per-order gates"),
    ("model", "positional_encoding", "rope", "model.positional_encoding"),
    ("train", "seed", 43, "train.seed"),
    ("optim", "learning_rate", 2e-3, "optim.learning_rate"),
    ("model", "attention_forget_gate_mode", "cumulative", "model.attention_forget_gate_mode"),
    ("model", "attention_backend", "blocked", "model.attention_backend"),
    ("model", "attention_coefficient_conditioning", "static", "model.attention_coefficient_conditioning"),
    ("model", "attention_coefficient_conditioning", "query_key", "model.attention_coefficient_conditioning"),
))
@pytest.mark.parametrize("degree,run_id,twin", QUERY_KEY_RUNS)
def test_altered_query_key_recipes_are_rejected(monkeypatch, degree, run_id, twin, section, key, value, reason):
    """Rejected in full, and by the control comparison alone where config validation rejects first."""
    config = run_config(run_id)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    with pytest.raises(ValueError, match=reason):
        adapter.check_recipe(config)


@pytest.mark.parametrize("conditioning,degree", (("query_key", 6), ("query_key", 10), ("query", 6), ("query", 10)))
def test_degrees_without_a_shared_gate_control_are_rejected(conditioning, degree):
    config = variant()
    config.model.attention_forget_gate_conditioning = conditioning
    config.model.attention_degree = degree
    with pytest.raises(ValueError, match="No shared-gate control"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("conditioning,degree", APPROVED)
def test_only_a_blocked_control_admits_another_backend(conditioning, degree):
    config = variant(conditioning, degree)
    config.model.attention_backend = "blocked"
    if adapter.control_config(conditioning, degree).model.attention_backend == "blocked":
        adapter.check_recipe(config)
    else:
        with pytest.raises(ValueError, match="attention_backend"):
            adapter.check_recipe(config)


@pytest.mark.parametrize("section,key,value", (
    ("model", "attention_forget_gate_per_order", False),
    ("model", "positional_encoding", "rope"),
    ("model", "qk_norm_head_scale", False), ("model", "attention_coefficient_kernel_weight_decay", False),
    ("optim", "weight_decay", 0.), ("optim", "total_steps", 3000), ("train", "seed", 43),
))
@pytest.mark.parametrize("conditioning,degree", APPROVED)
def test_unrelated_changes_are_rejected(conditioning, degree, section, key, value):
    config = variant(conditioning, degree)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("conditioning,degree", APPROVED)
def test_initialization_guard_recovers_control_and_rejects_changed_order(monkeypatch, conditioning, degree):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.training.checkpointing import state_sha256
    control, config = adapter.control_config("query", degree), variant(conditioning, degree)
    for current in (control, config):
        current.model.hidden_size = 8
        current.model.num_layers = 2
        current.model.num_heads = current.model.num_kv_heads = 2
        current.model.qk_head_dim = current.model.value_head_dim = 4
    monkeypatch.setattr(adapter, "control_config", lambda conditioning="query", degree=4: deepcopy(control))
    def state(current):
        values = initial_attention_parameter_values(attention_parameter_specs(current.model),
            num_layers=2, num_heads=2, key=mechanism_key(jax.random.PRNGKey(42)))
        module = nnx.Module()
        for name, value in values.items():
            setattr(module, name, nnx.Param(value))
        module.backbone = nnx.Param(jnp.arange(17, dtype=jnp.float32))
        return nnx.state(module, nnx.Param)
    expected = state(control)
    monkeypatch.setattr(adapter, "INITIALIZATION_SHA256", {degree: state_sha256(expected)})
    monkeypatch.setattr(adapter, "PARAMETER_COUNT", {degree: sum(x.size for x in jax.tree.leaves(expected))})
    assert adapter._check_initialized_parameters(state(config), config)["all_order_gates_match_shared_control"]
    # Another draw for one order of any gate parameter, including the top order.
    for name, order in (("forget_gate_kernel", 3), ("forget_gate_kernel", degree),
                        ("forget_gate_log_rate", degree), ("forget_gate_dt_bias", degree)):
        broken = state(config)
        value = broken[name][...].at[:, :, order, :].add(.01)
        broken[name] = broken[name].replace(value=value)
        with pytest.raises(RuntimeError, match=f"Initial parameter {name}"):
            adapter._check_initialized_parameters(broken, config)
    if conditioning == "query_key":
        broken = state(config)
        value = broken["forget_gate_key_kernel"][...].at[1, 0, degree, 2].set(1e-3)
        broken["forget_gate_key_kernel"] = broken["forget_gate_key_kernel"].replace(value=value)
        with pytest.raises(RuntimeError, match="key projection"):
            adapter._check_initialized_parameters(broken, config)
    broken = state(config)
    broken["backbone"] = broken["backbone"].replace(value=jnp.ones(17))
    with pytest.raises(RuntimeError, match="Restored initialization"):
        adapter._check_initialized_parameters(broken, config)
