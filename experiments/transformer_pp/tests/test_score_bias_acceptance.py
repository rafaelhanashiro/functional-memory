"""The per-degree score-bias runs remain their pinned controls with only the score biases and gradient-event
capture added, and start from the controls' exact initialization. The reference-shape runs train on the blocked
backend's causal strips (their controls trained on the Mosaic kernels) on the GPU types and allocator share
their measured memory allows; the trial at the 12-layer width-640 shape keeps its control's attention settings
and trains on the Mosaic score-bias kernels, on H100 only."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re

import pytest

from experiments.transformer_pp import score_bias_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

# The allocated runs: control, score-bias mode, and label. The reference-shape runs train on the blocked strips,
# the trial on the Mosaic kernels.
STRIPS_RUNS = {
    "rb70f9e57190f": ("query", "static", "query-m8-dk64-w-decay-score-bias-static"),
    "r891870ef80e9": ("query", "query", "query-m8-dk64-w-decay-score-bias-query"),
    "r3b1619f8a901": ("query", "query_key", "query-m8-dk64-w-decay-score-bias-query-key"),
    "ra2edcdc5ab60": ("per_order_query_key_gate", "static",
                      "query-m8-per-order-query-key-gate-nope-wd-score-bias-static"),
    "rbf5d3bd6059c": ("per_order_query_key_gate", "query",
                      "query-m8-per-order-query-key-gate-nope-wd-score-bias-query"),
    "r75dccfcf072e": ("per_order_query_key_gate", "query_key",
                      "query-m8-per-order-query-key-gate-nope-wd-score-bias-query-key"),
}
TRIAL = "r8dc6e147601c"
FROZEN_RUNS = {**STRIPS_RUNS, TRIAL: ("query_l12_d640", "query", "query-m8-w-decay-l12-d640-score-bias-query")}
BACKBONES = tuple(adapter.CONTROLS)
REFERENCE_BACKBONES = ("query", "per_order_query_key_gate")
CAPTURE_FIELDS = tuple(adapter.CAPTURE_POLICY)
# Each control's shape and budget: layers, heads, width; updates, warmup, sequences per update.
SHAPES = {"query": (15, 14, 896, 7570, 455, 256), "per_order_query_key_gate": (15, 14, 896, 7570, 455, 256),
          "query_l12_d640": (12, 10, 640, 7551, 454, 128)}
# What the adapter's refusal of another attention implementation names, per control.
IMPLEMENTATION = {"query": "causal strips", "per_order_query_key_gate": "causal strips",
                  "query_l12_d640": "Mosaic score-bias kernels on H100"}
NO_CONTROL = (r"No score-bias control has this recipe's \['model.num_layers', 'model.hidden_size', "
              r"'model.attention_forget_gate'\]")


def run_config(run_id):
    return load_config(adapter.RUNS / run_id / "config.yaml")


def variant(name="query", mode="query_key"):
    """The control's recipe as its runs change it: score biases, capture, and the runs' attention settings."""
    config = adapter.control_config(name)
    config.model.attention_score_bias = mode
    for key, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, key, value)
    for key, value in adapter.BACKENDS[name].items():
        setattr(config.model, key, value)
    return config


def scientific_differences(config, other):
    """The settings two recipes differ in, apart from execution locations and monitoring labels."""
    first, second = config_to_dict(config), config_to_dict(other)
    for recipe in (first, second):
        for key in adapter.EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    return {f"{section}.{key}" for section in set(first) | set(second)
            for key in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(key) != second.get(section, {}).get(key)}


@pytest.mark.parametrize("name", BACKBONES)
def test_pinned_controls_are_the_recorded_runs(name):
    run, digest, _, count = adapter.CONTROLS[name]
    path = adapter.RUNS / run / "config.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert json.loads((adapter.RUNS / run / "record.json").read_text())["config"]["sha256"] == digest
    assert json.loads((adapter.RUNS / run / "completion.json").read_text())["parameter_count"] == count
    config = adapter.control_config(name)
    model = config.model
    assert model.attention_score_bias == "none" and model.attention_backend == "mosaic" and model.attention_degree == 8
    assert (model.attention_coefficient_conditioning, model.attention_coefficient_activation) == ("query", "tanh")
    assert model.attention_coefficient_kernel_weight_decay and not model.attention_normalization
    if name == "per_order_query_key_gate":
        assert model.attention_forget_gate and model.attention_forget_gate_per_order
        assert (model.attention_forget_gate_mode, model.attention_forget_gate_conditioning,
                model.positional_encoding) == ("query_distance", "query_key", "none")
    else:
        assert not model.attention_forget_gate and model.positional_encoding == "rope"
    assert not config.train.gradient_event_capture
    # The depth, width and forget gate that select the control are the recipe's own.
    assert (model.num_layers, model.hidden_size, model.attention_forget_gate) == adapter.CONTROL_SHAPES[name]
    assert adapter.control_name(config) == name
    layers, heads, width, updates, warmup, batch = SHAPES[name]
    assert (model.num_layers, model.num_heads, model.hidden_size, config.optim.total_steps,
            config.optim.warmup_steps, config.data.batch_size) == (layers, heads, width, updates, warmup, batch)


@pytest.mark.parametrize("run_id", FROZEN_RUNS)
def test_allocated_runs_are_their_controls_with_score_biases_and_capture(run_id):
    from scaffolds.transformer.models import model_parameter_count

    name, mode, label = FROZEN_RUNS[run_id]
    config, control = run_config(run_id), adapter.control_config(name)
    adapter.check_recipe(config)
    assert adapter.control_name(config) == name and config.model.attention_score_bias == mode
    implementation = (config.model.attention_backend, config.model.attention_blocked_implementation)
    assert implementation == (("mosaic", "scan") if run_id == TRIAL else ("blocked", "strips"))
    assert {key: getattr(config.train, key) for key in CAPTURE_FIELDS} == adapter.CAPTURE_POLICY
    # The capture policy's other fields hold their defaults, so only the switch differs in the recipe. The
    # reference-shape runs' attention implementation is the blocked strips, where both of their controls trained
    # on the Mosaic kernels; the trial keeps its control's Mosaic settings.
    moved = (set() if run_id == TRIAL else
             {"model.attention_backend", "model.attention_blocked_implementation"})
    assert scientific_differences(config, control) == {
        "model.attention_score_bias", "train.gradient_event_capture", *moved}
    _, _, _, count = adapter.CONTROLS[name]
    assert model_parameter_count(config.model) == count + adapter.ADDED_PARAMETERS[name][mode]
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["home_cluster"] == "kennel"
    recipe = (adapter.RUNS / run_id / "config.yaml").read_bytes()
    assert record["config"]["sha256"] == hashlib.sha256(recipe).hexdigest()
    layers, heads, width, updates, warmup, batch = SHAPES[name]
    assert (config.model.num_layers, config.model.num_heads, config.model.hidden_size) == (layers, heads, width)
    assert config.optim.total_steps == updates and config.optim.warmup_steps == warmup and config.train.seed == 42
    assert config.data.batch_size == batch and config.data.sequence_length == 2049 and config.model.max_seq_len == 2048


def test_the_trial_adds_its_query_biases_to_the_controls_parameters():
    """b and a [12, 10, 8] and u [12, 10, 8, 640] beside the control's 99,678,056 parameters."""
    from scaffolds.transformer.models import model_parameter_count

    added = 12 * 10 * 8 + 12 * 10 * 8 + 12 * 10 * 8 * 640
    assert added == 616_320 == adapter.ADDED_PARAMETERS["query_l12_d640"]["query"]
    assert adapter.CONTROLS["query_l12_d640"][3] + added == 100_294_376 == model_parameter_count(
        run_config(TRIAL).model)


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", adapter.MODES)
def test_added_arrays_are_the_scaffolds_declared_score_biases(name, mode):
    """The adapter's own expectation (names, shapes, starts) agrees with the declared parameters, the parameter
    formula, and the pinned additions at the controls' shapes."""
    from scaffolds.transformer.attention import score_bias
    from scaffolds.transformer.models import model_parameter_count

    config, control = variant(name, mode), adapter.control_config(name)
    layers, heads, width, *_ = SHAPES[name]
    arrays = adapter.score_bias_arrays(config)
    assert set(arrays) == set(adapter.MODE_ARRAYS[mode]) == set(score_bias.parameter_names(mode))
    specs = {spec.name: spec for spec in score_bias.parameter_specs(hidden_size=width, degree=8, mode=mode)}
    for array, (shape, value) in arrays.items():
        assert shape[:3] == (layers, heads, 8) and shape[3:] in ((), (width,))
        assert set(specs[array].initial) == {value}
        assert specs[array].weight_decay is (array in ("score_bias_query_kernel", "score_bias_key_kernel"))
    added = sum(math.prod(shape) for shape, _ in arrays.values())
    assert added == adapter.ADDED_PARAMETERS[name][mode] == model_parameter_count(config.model) - model_parameter_count(
        control.model)


def test_pinned_additions():
    reference = {"static": 1_680, "query": 1_508_640, "query_key": 3_013_920}
    assert adapter.ADDED_PARAMETERS == {"query": reference, "per_order_query_key_gate": reference,
                                        "query_l12_d640": {"static": 960, "query": 616_320, "query_key": 1_230_720}}


@pytest.mark.parametrize("run_id", ("rb70f9e57190f", "r75dccfcf072e", TRIAL))
def test_the_control_comparison_covers_every_other_field(monkeypatch, run_id):
    """Every field but the execution settings and the score-bias mode, changed alone, is rejected by name, with
    validation bypassed so that the adapter's own checks must reject it: the attention implementation and
    capture policy by their own checks, the depth, width and forget gate by the control selection, everything
    else by the control comparison."""
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    frozen, missed = run_config(run_id), []
    implementation = IMPLEMENTATION[FROZEN_RUNS[run_id][0]]
    for section, values in config_to_dict(frozen).items():
        for key, value in values.items():
            if (section == "train" and key in adapter.EXECUTION_KEYS) or key == "attention_score_bias":
                continue
            config = deepcopy(frozen)
            setattr(getattr(config, section), key, ("changed", value))
            reason = (implementation if key in adapter.STRIPS else
                      "default policy" if key in CAPTURE_FIELDS else f"'{section}.{key}'")
            try:
                adapter.check_recipe(config)
            except ValueError as error:
                if reason not in str(error):
                    missed.append(f"{section}.{key}: {error}")
            else:
                missed.append(f"{section}.{key}: accepted")
    assert not missed


@pytest.mark.parametrize(("name", "field", "value"), (
    ("query_l12_d640", "num_layers", 15), ("query_l12_d640", "hidden_size", 896),
    ("query_l12_d640", "attention_forget_gate", True), ("query", "num_layers", 12), ("query", "hidden_size", 640),
    ("per_order_query_key_gate", "num_layers", 12),
))
def test_a_recipe_matching_no_control_is_refused(monkeypatch, name, field, value):
    """The depth, width and forget gate select the control; with validation bypassed, the adapter's own
    selection refuses a recipe that matches none and names the selecting settings."""
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    config = variant(name, "query")
    setattr(config.model, field, value)
    with pytest.raises(ValueError, match=NO_CONTROL):
        adapter.control_name(config)
    for check in (adapter.check_recipe, adapter.check_initialization):
        with pytest.raises(ValueError, match=NO_CONTROL):
            check(config)


SCIENTIFIC_CHANGES = (
    ("optim", "learning_rate", .002), ("optim", "weight_decay", .2), ("optim", "total_steps", 7571),
    ("train", "seed", 43), ("train", "eval_every", 1000), ("train", "ckpt_every", 1000), ("data", "batch_size", 64),
    ("model", "attention_degree", 4), ("model", "positional_encoding", "path"), ("model", "qk_norm_head_scale", False),
    ("model", "attention_coefficient_kernel_weight_decay", False), ("model", "attention_output_scaling", "none"),
    ("model", "num_layers", 13), ("model", "num_heads", 7),
)
GATE_CHANGES = (("model", "attention_forget_gate_per_order", False),
                ("model", "attention_forget_gate_conditioning", "query"), ("model", "attention_forget_gate", False),
                ("model", "positional_encoding", "rope"))


@pytest.mark.parametrize(("name", "section", "field", "value"), (
    *((name, *change) for name in BACKBONES for change in SCIENTIFIC_CHANGES),
    ("query", "model", "positional_encoding", "none"), ("query_l12_d640", "model", "positional_encoding", "none"),
    *(("per_order_query_key_gate", *change) for change in GATE_CHANGES),
))
def test_other_scientific_changes_are_rejected(name, section, field, value):
    """With the scaffold's validation, whichever check meets the change first."""
    config = variant(name)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", adapter.MODES)
def test_only_execution_locations_and_labels_may_differ(name, mode):
    config = variant(name, mode)
    adapter.check_recipe(config)
    config.train.workdir = Path("/new/execution")
    config.train.checkpoint_dir = Path("/new/execution/checkpoints")
    config.train.source_commit = "new-source-commit"
    config.train.wandb_run_name = "another-label"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/execution/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("backend,implementation", (
    ("blocked", "strips"), ("mosaic", "scan"), ("mosaic", "strips"), ("blocked", "scan"), ("auto", "strips"),
    ("fused", "scan"), ("xla", "scan"), ("cudnn", "scan"), ("manual", "scan"),
))
@pytest.mark.parametrize("name", BACKBONES)
def test_only_the_runs_attention_implementation_is_accepted(name, backend, implementation):
    """The reference-shape runs train on the blocked strips and the 12-layer width-640 runs on their control's
    Mosaic settings. The other Mosaic and blocked settings are valid score-bias recipes, so the adapter refuses
    them itself; validation refuses the other backends, which never run score biases."""
    from scaffolds.transformer.configs.base import validate_config

    config = variant(name)
    config.model.attention_backend = backend
    config.model.attention_blocked_implementation = implementation
    if {"attention_backend": backend, "attention_blocked_implementation": implementation} == adapter.BACKENDS[name]:
        adapter.check_recipe(config)
    elif backend in ("mosaic", "blocked"):
        validate_config(config)
        with pytest.raises(ValueError, match=IMPLEMENTATION[name]):
            adapter.check_recipe(config)
    else:
        for check in (validate_config, adapter.check_recipe):
            with pytest.raises(ValueError):
                check(config)


@pytest.mark.parametrize(("field", "value"), (
    ("gradient_event_capture", False),
    ("gradient_event_factor", 4.0),
    ("gradient_event_min_norm", 0.5),
    ("gradient_event_history", 50),
    ("gradient_event_min_history", 10),
    ("gradient_event_max_states", 5),
))
@pytest.mark.parametrize("name", BACKBONES)
def test_capture_must_be_on_with_the_default_policy(name, field, value):
    config = variant(name)
    setattr(config.train, field, value)
    with pytest.raises(ValueError, match="Gradient-event capture must be on with the default policy"):
        adapter.check_recipe(config)


def test_the_pinned_capture_policy_is_the_scaffold_default_rule():
    from scaffolds.transformer.configs.base import TrainConfig

    defaults = TrainConfig()
    assert adapter.CAPTURE_POLICY == {**{key: getattr(defaults, key) for key in CAPTURE_FIELDS},
                                      "gradient_event_capture": True}


@pytest.mark.parametrize("name", BACKBONES)
def test_the_controls_themselves_are_refused(name):
    with pytest.raises(ValueError, match="requires per-degree score biases"):
        adapter.check_recipe(adapter.control_config(name))
    config = adapter.control_config(name)
    for key, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, key, value)
    with pytest.raises(ValueError, match="requires per-degree score biases"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("name", BACKBONES)
def test_an_edited_control_recipe_is_refused(monkeypatch, tmp_path, name):
    """Any byte change to a pinned control recipe, even a comment, stops the comparison."""
    config, run = variant(name), adapter.CONTROLS[name][0]
    edited = tmp_path / run / "config.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_bytes((adapter.RUNS / run / "config.yaml").read_bytes() + b"# edited\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    for check in (lambda: adapter.control_config(name), lambda: adapter.check_recipe(config)):
        with pytest.raises(ValueError, match=f"pinned control recipe {run} changed"):
            check()


def _tiny(config):
    """The recipe at a small shape, built on the blocked reference."""
    return replace(config, model=replace(config.model, attention_backend="blocked", vocab_size=64, max_seq_len=32,
                                         num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2, qk_head_dim=8,
                                         value_head_dim=8, intermediate_size=64))


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", adapter.MODES)
def test_removing_the_exact_score_biases_recovers_the_controls_initialization(name, mode):
    """At a small shape: the variant's arrays are the control's plus the mode's biases at b = 0, a = 1,
    u = v = 0, and removing them recovers the control's fingerprint; any changed start, missing or extra
    array, or changed backbone parameter is detected."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scaffolds.transformer.training.checkpointing import state_sha256

    config, control = _tiny(variant(name, mode)), _tiny(adapter.control_config(name))
    reference = adapter.initial_parameters(control)
    count = sum(leaf.size for leaf in jax.tree.leaves(reference))
    arrays = adapter.score_bias_arrays(config)
    assert {array: shape for array, (shape, _) in arrays.items()} == {
        array: (2, 2, 8, *((32,) if array.endswith("kernel") else ())) for array in adapter.MODE_ARRAYS[mode]}
    facts = adapter.without_score_biases(adapter.initial_parameters(config), config)
    assert facts == {"parameter_count": count + sum(math.prod(shape) for shape, _ in arrays.values()),
                     "control_parameter_sha256": state_sha256(reference)}

    def changed(array, edit):
        params = adapter.initial_parameters(config)
        params[array] = params[array].replace(value=edit(params[array][...]))
        return params

    for array, (_, value) in arrays.items():
        # The nearest larger FP32 start (a normal number at zero), BF16, a lost order entry, and -0 for 0.
        nearby = np.float32(1e-30) if value == 0 else np.nextafter(np.float32(value), np.float32(2))
        for edit in (lambda x: x.at[1, 0, 7].set(nearby), lambda x: x.astype(jnp.bfloat16),
                     lambda x: x[..., :-1], *((lambda x: -x,) if value == 0 else ())):
            with pytest.raises(RuntimeError, match=f"{array} must start at exactly"):
                adapter.without_score_biases(changed(array, edit), config)
        params = adapter.initial_parameters(config)
        del params[array]
        with pytest.raises(RuntimeError, match="add exactly"):
            adapter.without_score_biases(params, config)
    if mode != "query_key":
        extra = adapter.initial_parameters(_tiny(variant(name, "query_key")))
        with pytest.raises(RuntimeError, match="add exactly"):
            adapter.without_score_biases(extra, config)
    backbone = changed("qk_norm_q_gain", lambda x: x + 1)  # a parameter the control also has
    assert adapter.without_score_biases(backbone, config)["control_parameter_sha256"] != state_sha256(reference)


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mismatch", ("digest", "count", "count_without_biases", "other_control"))
def test_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, name, mismatch):
    _, _, digest, count = adapter.CONTROLS[name]
    added = adapter.ADDED_PARAMETERS[name]["query_key"]
    other = next(adapter.CONTROLS[other] for other in BACKBONES if other != name)
    facts = {"digest": ("0" * 64, count + added), "count": (digest, count + added + 1),
             "count_without_biases": (digest, count), "other_control": (other[2], other[3] + added)}[mismatch]
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: "built")
    monkeypatch.setattr(adapter, "without_score_biases",
                        lambda params, config: {"control_parameter_sha256": facts[0], "parameter_count": facts[1]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match="Initialization without the score biases differs from control"):
        adapter.check_initialization(variant(name, "query_key"))


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", adapter.MODES)
def test_matching_initialization_proceeds_to_backbone_qualification(monkeypatch, name, mode):
    run, _, digest, count = adapter.CONTROLS[name]
    built = []
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: built.append(config) or "built")
    monkeypatch.setattr(adapter, "without_score_biases", lambda params, config: {
        "control_parameter_sha256": digest, "parameter_count": count + adapter.ADDED_PARAMETERS[name][mode]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: {"backbone_checked": config.model.attention_score_bias})
    result = adapter.check_initialization(variant(name, mode))
    assert len(built) == 1 and result["backbone_checked"] == mode
    assert (result["control_run_id"], result["control_parameter_sha256"], result["control_parameter_count"]) == (
        run, digest, count)
    assert result["score_bias_parameter_count"] == adapter.ADDED_PARAMETERS[name][mode]
    assert set(result["score_bias_arrays"]) == set(adapter.MODE_ARRAYS[mode])
    assert result["all_other_initial_parameters_match_control"]


@pytest.mark.parametrize("name", BACKBONES)
def test_initialization_checks_the_recipe_before_building_models(monkeypatch, name):
    """Acceptance's initialization worker calls check_initialization on its own."""
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: pytest.fail("Built a model"))
    with pytest.raises(ValueError, match="requires per-degree score biases"):
        adapter.check_initialization(adapter.control_config(name))
    config = variant(name)
    config.model.attention_backend = {"blocked": "mosaic", "mosaic": "blocked"}[config.model.attention_backend]
    with pytest.raises(ValueError, match=IMPLEMENTATION[name]):
        adapter.check_initialization(config)


@pytest.mark.parametrize("run_id", STRIPS_RUNS)
def test_adapter_installs_its_guards_and_the_score_bias_checks(run_id):
    config = run_config(run_id)
    gated = STRIPS_RUNS[run_id][0] == "per_order_query_key_gate"
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == "blocked"
    assert adapter.RUNS.parent.parents[1] / adapter.OWN_TESTS == Path(__file__).resolve()
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
    # CPU: the score-bias reference and model checks, the strips' own checks, capture, and this adapter,
    # beside the backbone's CPU checks; no Mosaic module.
    assert {adapter.OWN_TESTS, acceptance.SCORE_BIAS_TESTS, acceptance.BLOCKED_STRIPS_TESTS,
            acceptance.GRADIENT_EVENT_TESTS} <= set(cpu)
    assert not {acceptance.SCORE_BIAS_LOWERING_TESTS, acceptance.MOSAIC_SCORE_BIAS_TESTS} & set(cpu)
    assert ({acceptance.PER_ORDER_DECAY_TESTS, acceptance.ENDPOINT_DECAY_TESTS} <= set(cpu)) is gated
    # GPU: the strips with score biases at production precision, and nothing that could skip or refuse there.
    assert gpu == (acceptance.SCORE_BIAS_STRIPS_TESTS,)
    assert acceptance.mosaic_test_options(recipe, "blocked") == ()
    with pytest.raises(ValueError):
        adapter.experiment_spec(adapter.control_config(STRIPS_RUNS[run_id][0]), entry_module=adapter.__name__)


# The GPU modules of the H100 qualification of the score-bias Mosaic kernels for the RoPE recipes (validation
# record, job 242565: none skipped or failed): the query polynomial's Mosaic kernels and their repeatability,
# the score-bias kernels against FP64 and attention_core's dispatch to them, and the tanh query generator's GPU
# cases (test_coefficient_scale.py has none). Their cases skip only without a GPU or on a GPU other than H100
# and B300, so none skips on an H100.
H100_SCORE_BIAS_GPU_MODULES = tuple(f"scaffolds/transformer/tests/test_{name}.py" for name in (
    "mosaic_query", "mosaic_repeatability", "mosaic_score_bias", "score_bias", "input_dependent_coefficients",
    "query_tanh_polynomial", "coefficient_scale"))


def test_the_trial_qualifies_the_mosaic_kernels_it_trains_on():
    config = run_config(TRIAL)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == "mosaic" and spec.runtime_env == {}
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=False)
    # CPU: the score-bias reference and model checks, the score-bias kernels' tiles and shared memory lowered on
    # the host, capture, and this adapter, beside the backbone's CPU checks; nothing of the strips.
    assert {adapter.OWN_TESTS, acceptance.SCORE_BIAS_TESTS, acceptance.SCORE_BIAS_LOWERING_TESTS,
            acceptance.GRADIENT_EVENT_TESTS, acceptance.QUERY_TANH_POLYNOMIAL_TESTS} <= set(cpu)
    assert not {acceptance.MOSAIC_SCORE_BIAS_TESTS, acceptance.BLOCKED_STRIPS_TESTS,
                acceptance.SCORE_BIAS_STRIPS_TESTS, acceptance.PER_ORDER_DECAY_TESTS} & set(cpu)
    # GPU: exactly the qualified modules, at the row-shift kernels' BF16 weights.
    assert gpu == H100_SCORE_BIAS_GPU_MODULES
    assert acceptance.mosaic_test_options(recipe, "mosaic") == ("--mosaic-precision", "bf16")
    with pytest.raises(ValueError):
        adapter.experiment_spec(adapter.control_config("query_l12_d640"), entry_module=adapter.__name__)


@pytest.mark.parametrize("resolved", ("blocked", "fused", "mosaic"))
@pytest.mark.parametrize(("run_id", "required", "device"), (
    ("r75dccfcf072e", "blocked", "NVIDIA B200"), (TRIAL, "mosaic", "NVIDIA H100 80GB HBM3")))
def test_the_pilot_must_train_on_the_runs_backend(monkeypatch, tmp_path, run_id, required, device, resolved):
    """Acceptance's own pilot check, with this adapter's spec: a pilot that resolved any backend but the run's
    (the blocked strips, or the trial's Mosaic kernels) fails the allocation before training; a pilot on the
    run's backend proceeds to its validation checks."""
    from types import SimpleNamespace

    config = run_config(run_id)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    experiment = {"identity": "test", "source_commit": "a" * 40, "config": config_to_dict(config)}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "config.yaml", gpu_count=1,
                           gpu_type=device.split()[1])

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": required, "device_kinds": [device]}
        if name == "initialization":
            return {"checked": True}
        metrics = args.directory / "pilot" / "metrics.jsonl"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text(json.dumps({"step": 20, "validation_nll": 3.0, "attention_diagnostics": {"valid": False}}))
        return {"final_step": 20, "data_parallel_devices": 1, "start_step": 0, "attention_backend": resolved}

    def run(command, **options):
        junit = next(value.split("=", 1)[1] for value in command if value.startswith("--junitxml="))
        Path(junit).write_text('<testsuite><testcase name="check"/></testsuite>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance, "_run_worker", worker)
    monkeypatch.setattr(acceptance.subprocess, "run", run)
    reason = ("attention diagnostics failed" if resolved == required
              else f"requires backend '{required}'; resolved '{resolved}'")
    with pytest.raises(RuntimeError, match=reason):
        acceptance._run(spec, args, experiment)
    assert json.loads((tmp_path / "failure.json").read_text())["stage"] == "pilot"


# Each run's GPU types and allocator share (the validation record's full-recipe pilots at micro-batch 16 for the
# strips; the trial trains on the Mosaic score-bias kernels, which run only on H100).
HARDWARE = {
    "rb70f9e57190f": (("B200", "B300", "H100"), None),
    "r891870ef80e9": (("B200", "B300", "H100"), None),
    "r3b1619f8a901": (("B200", "B300", "H100"), None),
    "ra2edcdc5ab60": (("B200", "B300", "H100"), "0.95"),
    "rbf5d3bd6059c": (("B200", "B300", "H100"), "0.95"),
    "r75dccfcf072e": (("B200", "B300"), "0.95"),
    TRIAL: (("H100",), None),
}


def recipe_path(run_id):
    return str(adapter.RUNS / run_id / "config.yaml")


@pytest.mark.parametrize("run_id", FROZEN_RUNS)
def test_each_run_has_its_qualified_gpu_types_and_allocator_share(run_id):
    """The trial trains on H100 alone, and of the strips runs only the gated query_key recipe (147 GiB) is kept
    off the H100; the gated recipes raise the allocator's share to 0.95, through the spec's runtime environment,
    which the launcher exports and acceptance records."""
    types, fraction = HARDWARE[run_id]
    config = run_config(run_id)
    assert adapter.gpu_types(config) == types
    expected = {} if fraction is None else {"XLA_PYTHON_CLIENT_MEM_FRACTION": fraction}
    assert adapter.runtime_env(config) == expected
    assert adapter.experiment_spec(config, entry_module=adapter.__name__).runtime_env == expected
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in acceptance.RUNTIME_KEYS


@pytest.mark.parametrize("run_id", FROZEN_RUNS)
def test_the_launchers_runtime_query_exports_the_allocator_share(capsys, run_id):
    """The launcher evaluates the adapter's --runtime-env output before acceptance and training; the query names
    no GPU type, so the GPU gate lets it through for every recipe, the gated query_key one included."""
    adapter.main([recipe_path(run_id), "--runtime-env"])
    fraction = HARDWARE[run_id][1]
    expected = "" if fraction is None else f"export XLA_PYTHON_CLIENT_MEM_FRACTION={fraction}\n"
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(("run_id", "gpu_type"), [  # no skips: acceptance's CPU stage runs this module
    (run_id, gpu_type) for run_id in FROZEN_RUNS for gpu_type in ("A100", "A6000", "H200", "H100", "B200", "B300")
    if gpu_type not in HARDWARE[run_id][0]])
def test_other_gpu_types_are_refused_before_acceptance(monkeypatch, run_id, gpu_type):
    allowed = HARDWARE[run_id][0]
    monkeypatch.setattr(adapter, "run_acceptance", lambda *args: pytest.fail("Acceptance started"))
    for value in (gpu_type, gpu_type.lower()):
        for argv in ([recipe_path(run_id), "--gpu-type", value], [recipe_path(run_id), f"--gpu-type={value}"]):
            message = (rf"trains only on {re.escape(', '.join(allowed))} \(.+\), not on {value}\. "
                       rf"Submit it with GPU types {re.escape(','.join(allowed))}\.")
            with pytest.raises(ValueError, match=message) as error:
                adapter.main(argv)
            assert ("147 GiB" in str(error.value)) is (allowed == ("B200", "B300"))
            assert ("Mosaic score-bias kernels" in str(error.value)) is (allowed == ("H100",))


@pytest.mark.parametrize("run_id", FROZEN_RUNS)
def test_qualified_allocations_reach_acceptance(monkeypatch, run_id):
    calls = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, arguments: calls.append(arguments))
    allowed = HARDWARE[run_id][0]
    argvs = [[recipe_path(run_id), "--gpu-type", gpu_type, "--gpu-count", "1"] for gpu_type in allowed]
    argvs.append([recipe_path(run_id), f"--gpu-type={allowed[0].lower()}"])
    # A worker's invocation (acceptance's _run_worker) passes the same gate.
    argvs.append([recipe_path(run_id), "--directory", "/tmp/acceptance", "--gpu-count", "1", "--gpu-type", allowed[0],
                  "--worker", "pilot", "--output", "/tmp/pilot.json", "--stop-after-step", "20"])
    for argv in argvs:
        adapter.main(argv)
    assert calls == argvs


@pytest.mark.parametrize("name", BACKBONES)
def test_pinned_initialization_is_the_control_recipes_own(name):
    """Builds the full control at seed 42 on the host (seconds): the pinned fingerprint and count."""
    _, _, digest, count = adapter.CONTROLS[name]
    assert acceptance.initialization_facts(adapter.control_config(name)) == {
        "parameter_count": count, "parameter_sha256": digest}


@pytest.mark.parametrize("name", REFERENCE_BACKBONES)
def test_full_size_query_key_biases_start_exactly_and_hide_the_control(name):
    """Builds the full query-key variant at seed 42 on the host (about ten seconds): removing its four exact
    score-bias arrays leaves the control's pinned fingerprint."""
    _, _, digest, count = adapter.CONTROLS[name]
    config = variant(name, "query_key")
    assert adapter.without_score_biases(adapter.initial_parameters(config), config) == {
        "parameter_count": count + adapter.ADDED_PARAMETERS[name]["query_key"], "control_parameter_sha256": digest}


def test_the_trials_full_size_query_biases_start_exactly_and_hide_its_control():
    """Builds the trial's own recipe at seed 42 on the host: 100,294,376 parameters, and removing its exact b, a
    [12, 10, 8] and u [12, 10, 8, 640] leaves the pinned fingerprint of r95a5691d293f."""
    config = run_config(TRIAL)
    assert {name: shape for name, (shape, _) in adapter.score_bias_arrays(config).items()} == {
        "score_bias_offset": (12, 10, 8), "score_bias_amplitude": (12, 10, 8),
        "score_bias_query_kernel": (12, 10, 8, 640)}
    assert adapter.without_score_biases(adapter.initial_parameters(config), config) == {
        "parameter_count": 100_294_376,
        "control_parameter_sha256": "72d75524ec107d4acdeb4f77d8cdea838d4c57d54f79aa9c924da4a154b03667"}
