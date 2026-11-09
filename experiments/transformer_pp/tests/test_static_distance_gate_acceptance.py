"""The input-free gate runs keep their static controls' recipes and starts, apart from the gate."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import static_distance_gate_acceptance as adapter
from scaffolds.transformer.acceptance import INPUT_FREE_GATE_TESTS, test_paths as selected_test_paths
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = {(4, False): "r0e76396bdee5", (4, True): "r527a30ff7ed1",
        (8, False): "rc60345dfcff7", (8, True): "re253a5f1b03e"}
CASES = list(RUNS)


def run_config(degree, per_order):
    return load_config(adapter.RUNS / RUNS[degree, per_order] / "config.yaml")


def test_pins_come_from_the_recorded_runs():
    pins = [entry[:2] for entry in (*adapter.CONTROLS.values(), *adapter.SHARED_GATES.values())]
    for run, digest in (*pins, *adapter.QUERY_GATES.values()):
        assert json.loads((adapter.RUNS / run / "record.json").read_text())["config"]["sha256"] == digest, run
    for degree, (run, _, _, count) in adapter.CONTROLS.items():
        assert json.loads((adapter.RUNS / run / "metadata.json").read_text())["parameter_count"] == count
        assert adapter.SHARED_GATES[degree][3] == count + 2 * 15 * 14
    assert [adapter.SHARED_GATES[degree][0] for degree in (4, 8)] == [RUNS[4, False], RUNS[8, False]]


@pytest.mark.parametrize("degree,per_order", CASES)
def test_frozen_runs_are_their_controls_with_the_gate(degree, per_order):
    config = run_config(degree, per_order)
    adapter.check_recipe(config)
    model = config.model
    assert (model.attention_degree, model.attention_forget_gate_per_order) == (degree, per_order)
    assert model.attention_backend == "mosaic" and model.positional_encoding == "none"
    assert model.attention_forget_gate and model.attention_forget_gate_conditioning == "none"
    assert model.attention_forget_gate_mode == "query_distance" and model.attention_coefficient_conditioning == "static"
    assert model.attention_coefficients == "learned" and model.attention_coefficient_sharing == "per_head"
    assert model.attention_coefficient_scale == "factorial" and not model.attention_coefficient_kernel_weight_decay
    assert model.qk_norm_head_scale and model.attention_output_scaling == "head_rms_norm"
    assert config.optim.total_steps == 7570 and config.train.seed == 42 and not config.continuation.parent_checkpoint
    assert config.data.batch_size == 256 and config.data.sequence_length == 2049
    record = json.loads((adapter.RUNS / RUNS[degree, per_order] / "record.json").read_text())
    assert record["home_cluster"] == "kennel" and record["status"] == "planned"


@pytest.mark.parametrize(("section", "field", "value"), (
    ("model", "positional_encoding", "rope"),
    ("model", "attention_backend", "blocked"),
    ("model", "attention_forget_gate_conditioning", "query"),
    ("model", "attention_forget_gate_mode", "cumulative"),
    ("model", "attention_coefficient_scale", "none"),
    ("model", "attention_output_scaling", "none"),
    ("model", "qk_norm_head_scale", False),
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", 0.),
    ("optim", "total_steps", 7571),
    ("data", "batch_size", 128),
    ("train", "seed", 43),
))
@pytest.mark.parametrize("degree,per_order", CASES)
def test_other_scientific_changes_are_rejected(degree, per_order, section, field, value):
    config = run_config(degree, per_order)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_only_degrees_with_a_static_control_are_these_runs():
    config = run_config(4, False)
    config.model.attention_degree = 6
    with pytest.raises(ValueError, match="No static control"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("degree,per_order", CASES)
def test_execution_names_may_change(degree, per_order):
    config = run_config(degree, per_order)
    config.train.workdir = Path("/new/run")
    config.train.checkpoint_dir = Path("/new/run/checkpoints")
    config.train.source_commit = "new_commit"
    config.train.wandb_run_name = "renamed"
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("degree,per_order", CASES)
def test_spec_installs_both_guards_and_the_gate_tests(degree, per_order):
    config = run_config(degree, per_order)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert "experiments/transformer_pp/tests/test_static_distance_gate_acceptance.py" in spec.cpu_tests
    for cpu in (True, False):
        paths = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=cpu)
        assert INPUT_FREE_GATE_TESTS in paths and "scaffolds/transformer/tests/test_endpoint_decay.py" in paths
        assert ("scaffolds/transformer/tests/test_per_order_decay.py" in paths) == per_order


def _shrink(config):
    config.model.hidden_size = 16
    config.model.num_layers = 2
    config.model.num_heads = config.model.num_kv_heads = 2
    config.model.qk_head_dim = config.model.value_head_dim = 8
    config.model.intermediate_size = 32
    config.model.vocab_size = config.data.synthetic_vocab_size = 64
    config.model.attention_backend = "blocked"  # Mosaic needs width 64; parameters do not depend on it
    return config


@pytest.fixture
def small(monkeypatch):
    """Small models of the four runs, with the pins replaced by their small controls' starts."""
    import jax
    from scaffolds.transformer.training.checkpointing import state_sha256

    configs = {case: _shrink(run_config(*case)) for case in CASES}
    controls, shared = {}, {}
    for degree in (4, 8):
        control = adapter._initial_parameters(_shrink(adapter.control_config(degree)))
        gated = adapter._initial_parameters(configs[degree, False])
        controls[degree] = (*adapter.CONTROLS[degree][:2], state_sha256(control),
                            sum(leaf.size for leaf in jax.tree.leaves(control)))
        shared[degree] = (*adapter.SHARED_GATES[degree][:2], state_sha256(gated),
                          sum(leaf.size for leaf in jax.tree.leaves(gated)))
    monkeypatch.setattr(adapter, "CONTROLS", controls)
    monkeypatch.setattr(adapter, "SHARED_GATES", shared)
    return configs


@pytest.mark.parametrize("degree,per_order", CASES)
def test_initialization_recovers_the_controls_and_the_query_gate_draws(small, degree, per_order):
    config = small[degree, per_order]
    facts = adapter._check_initialized_parameters(adapter._initial_parameters(config), config)
    assert facts["parameters_other_than_the_gate_match_static_control"]
    assert facts["gate_parameter_count"] == 2 * 2 * 2 * (degree + 1 if per_order else 1)
    assert facts["gate_draws_match_query_distance_gate_run_id"] == adapter.QUERY_GATES[degree][0]
    assert ("collapsed_parameter_sha256" in facts) == per_order
    if per_order:
        assert facts["collapsed_parameter_sha256"] == adapter.SHARED_GATES[degree][2]


def _changed(params, name, change):
    params[name] = params[name].replace(value=change(params[name][...]))
    return params


@pytest.mark.parametrize(("per_order", "name", "change", "message"), (
    (True, "forget_gate_log_rate", lambda x: x.at[:, :, 2].add(.01), "repeated over the orders"),
    (False, "forget_gate_dt_bias", lambda x: x + .01, "query-distance gate's draw"),
    (True, "forget_gate_dt_bias", lambda x: x + .01, "query-distance gate's draw"),
    (False, "polynomial_coefficients", lambda x: x + .01, "differ from"),
    (True, "output_norm_scale", lambda x: x * 2, "differ"),
))
def test_initialization_check_rejects_other_starts(small, per_order, name, change, message):
    config = small[4, per_order]
    params = _changed(adapter._initial_parameters(config), name, change)
    with pytest.raises(RuntimeError, match=message):
        adapter._check_initialized_parameters(params, config)


def test_check_initialization_runs_the_shared_backbone_check_first(small, monkeypatch):
    config = small[8, True]
    calls = []
    monkeypatch.setattr(adapter, "check_recipe", lambda config: calls.append("recipe"))
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: calls.append("backbone") or {"backbone_checked": True})
    facts = adapter.check_initialization(deepcopy(config))
    assert calls == ["recipe", "backbone"] and facts["backbone_checked"]
    assert facts["static_distance_gate"]["shared_gate_run_id"] == RUNS[8, False]


def test_historical_controls_and_query_gates_are_unchanged():
    """The static controls still run the gate-free kernels without gate parameters; the query-distance
    runs keep their query gates and arrays."""
    from scaffolds.transformer.attention.forget_gate import PARAMETERS
    from scaffolds.transformer.attention.parameters import attention_parameter_specs

    for degree in (4, 8):
        model = adapter.control_config(degree).model
        assert model.mosaic_supported() and not model.mosaic_decay_supported() and not model.attention_forget_gate
        assert not any(spec.name.startswith("forget_gate") for spec in attention_parameter_specs(model))
    for run in ("r3e26412b2811", "r507193d47450", "r9320e5cd3071", "r20fe6fe864bc"):
        model = load_config(adapter.RUNS / run / "config.yaml").model
        assert model.attention_forget_gate_conditioning == "query"
        assert [spec.name for spec in attention_parameter_specs(model)
                if spec.name.startswith("forget_gate")] == list(PARAMETERS)
