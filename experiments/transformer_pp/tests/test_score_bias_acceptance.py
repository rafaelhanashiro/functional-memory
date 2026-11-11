"""The per-degree score-bias runs remain their pinned controls with only the score biases and gradient-event
capture added, train on the blocked backend's causal strips (the controls trained on the Mosaic kernels),
and start from the controls' exact initialization."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.transformer_pp import score_bias_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

# The allocated runs: control, score-bias mode, and label.
FROZEN_RUNS = {
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
BACKBONES = tuple(adapter.CONTROLS)
CAPTURE_FIELDS = tuple(adapter.CAPTURE_POLICY)


def run_config(run_id):
    return load_config(adapter.RUNS / run_id / "config.yaml")


def variant(name="query", mode="query_key"):
    """The control's recipe as the runs change it: score biases, capture, and the blocked strips."""
    config = adapter.control_config(name)
    config.model.attention_score_bias = mode
    for key, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, key, value)
    for key, value in adapter.BACKEND.items():
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
    model = adapter.control_config(name).model
    assert model.attention_score_bias == "none" and model.attention_backend == "mosaic" and model.attention_degree == 8
    assert (model.attention_coefficient_conditioning, model.attention_coefficient_activation) == ("query", "tanh")
    assert model.attention_coefficient_kernel_weight_decay and not model.attention_normalization
    if name == "query":
        assert not model.attention_forget_gate and model.positional_encoding == "rope"
    else:
        assert model.attention_forget_gate and model.attention_forget_gate_per_order
        assert (model.attention_forget_gate_mode, model.attention_forget_gate_conditioning,
                model.positional_encoding) == ("query_distance", "query_key", "none")
    assert not adapter.control_config(name).train.gradient_event_capture


@pytest.mark.parametrize("run_id", FROZEN_RUNS)
def test_allocated_runs_are_their_controls_with_score_biases_and_capture(run_id):
    from scaffolds.transformer.models import model_parameter_count

    name, mode, label = FROZEN_RUNS[run_id]
    config = run_config(run_id)
    adapter.check_recipe(config)
    assert adapter.control_name(config) == name and config.model.attention_score_bias == mode
    assert (config.model.attention_backend, config.model.attention_blocked_implementation) == ("blocked", "strips")
    assert {key: getattr(config.train, key) for key in CAPTURE_FIELDS} == adapter.CAPTURE_POLICY
    # The capture policy's other fields hold their defaults, so only the switch differs in the recipe; the
    # attention implementation is the blocked strips, where both controls trained on the Mosaic kernels.
    assert scientific_differences(config, adapter.control_config(name)) == {
        "model.attention_score_bias", "train.gradient_event_capture", "model.attention_backend",
        "model.attention_blocked_implementation"}
    _, _, _, count = adapter.CONTROLS[name]
    assert model_parameter_count(config.model) == count + adapter.ADDED_PARAMETERS[mode]
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    recipe = (adapter.RUNS / run_id / "config.yaml").read_bytes()
    assert record["config"]["sha256"] == hashlib.sha256(recipe).hexdigest()
    assert config.optim.total_steps == 7570 and config.optim.warmup_steps == 455 and config.train.seed == 42
    assert config.data.batch_size == 256 and config.data.sequence_length == 2049 and config.model.max_seq_len == 2048


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", adapter.MODES)
def test_added_arrays_are_the_scaffolds_declared_score_biases(name, mode):
    """The adapter's own expectation (names, shapes, starts) agrees with the declared parameters, the parameter
    formula, and the pinned additions at the controls' shape."""
    from scaffolds.transformer.attention import score_bias
    from scaffolds.transformer.models import model_parameter_count

    config, control = variant(name, mode), adapter.control_config(name)
    arrays = adapter.score_bias_arrays(config)
    assert set(arrays) == set(adapter.MODE_ARRAYS[mode]) == set(score_bias.parameter_names(mode))
    specs = {spec.name: spec for spec in score_bias.parameter_specs(hidden_size=896, degree=8, mode=mode)}
    for array, (shape, value) in arrays.items():
        assert shape[:3] == (15, 14, 8) and shape[3:] in ((), (896,))
        assert set(specs[array].initial) == {value}
        assert specs[array].weight_decay is (array in ("score_bias_query_kernel", "score_bias_key_kernel"))
    added = sum(math.prod(shape) for shape, _ in arrays.values())
    assert added == adapter.ADDED_PARAMETERS[mode] == model_parameter_count(config.model) - model_parameter_count(
        control.model)
    assert adapter.ADDED_PARAMETERS == {"static": 1_680, "query": 1_508_640, "query_key": 3_013_920}


@pytest.mark.parametrize("run_id", ("rb70f9e57190f", "r75dccfcf072e"))
def test_the_control_comparison_covers_every_other_field(monkeypatch, run_id):
    """Every field but the execution settings and the score-bias mode, changed alone, is rejected by name, with
    validation bypassed so that the adapter's own checks must reject it: the attention implementation and
    capture policy by their own checks, everything else by the control comparison."""
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    frozen, missed = run_config(run_id), []
    for section, values in config_to_dict(frozen).items():
        for key, value in values.items():
            if (section == "train" and key in adapter.EXECUTION_KEYS) or key == "attention_score_bias":
                continue
            config = deepcopy(frozen)
            setattr(getattr(config, section), key, ("changed", value))
            reason = ("causal strips" if key in adapter.BACKEND else
                      "default policy" if key in CAPTURE_FIELDS else f"'{section}.{key}'")
            try:
                adapter.check_recipe(config)
            except ValueError as error:
                if reason not in str(error):
                    missed.append(f"{section}.{key}: {error}")
            else:
                missed.append(f"{section}.{key}: accepted")
    assert not missed


SCIENTIFIC_CHANGES = (
    ("optim", "learning_rate", .002), ("optim", "weight_decay", .2), ("optim", "total_steps", 7571),
    ("train", "seed", 43), ("train", "eval_every", 1000), ("train", "ckpt_every", 1000), ("data", "batch_size", 128),
    ("model", "attention_degree", 4), ("model", "positional_encoding", "path"), ("model", "qk_norm_head_scale", False),
    ("model", "attention_coefficient_kernel_weight_decay", False), ("model", "attention_output_scaling", "none"),
)
GATE_CHANGES = (("model", "attention_forget_gate_per_order", False),
                ("model", "attention_forget_gate_conditioning", "query"), ("model", "attention_forget_gate", False),
                ("model", "positional_encoding", "rope"))


@pytest.mark.parametrize(("name", "section", "field", "value"), (
    *((name, *change) for name in BACKBONES for change in SCIENTIFIC_CHANGES),
    ("query", "model", "positional_encoding", "none"),
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
    ("mosaic", "strips"), ("mosaic", "scan"), ("blocked", "scan"), ("auto", "strips"), ("fused", "scan"),
    ("xla", "scan"), ("cudnn", "scan"), ("manual", "scan"),
))
@pytest.mark.parametrize("name", BACKBONES)
def test_only_the_blocked_strips_are_accepted(name, backend, implementation):
    """The Mosaic kernels (the controls' backend) and the blocked scan are valid score-bias recipes, so the
    adapter refuses them itself; validation refuses the other backends, which never run score biases."""
    from scaffolds.transformer.configs.base import validate_config

    config = variant(name)
    config.model.attention_backend = backend
    config.model.attention_blocked_implementation = implementation
    if backend in ("mosaic", "blocked"):
        validate_config(config)
        with pytest.raises(ValueError, match="causal strips"):
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
    added = adapter.ADDED_PARAMETERS["query_key"]
    other = adapter.CONTROLS["query" if name != "query" else "per_order_query_key_gate"]
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
        "control_parameter_sha256": digest, "parameter_count": count + adapter.ADDED_PARAMETERS[mode]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: {"backbone_checked": config.model.attention_score_bias})
    result = adapter.check_initialization(variant(name, mode))
    assert len(built) == 1 and result["backbone_checked"] == mode
    assert (result["control_run_id"], result["control_parameter_sha256"], result["control_parameter_count"]) == (
        run, digest, count)
    assert result["score_bias_parameter_count"] == adapter.ADDED_PARAMETERS[mode]
    assert set(result["score_bias_arrays"]) == set(adapter.MODE_ARRAYS[mode])
    assert result["all_other_initial_parameters_match_control"]


@pytest.mark.parametrize("name", BACKBONES)
def test_initialization_checks_the_recipe_before_building_models(monkeypatch, name):
    """Acceptance's initialization worker calls check_initialization on its own."""
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: pytest.fail("Built a model"))
    with pytest.raises(ValueError, match="requires per-degree score biases"):
        adapter.check_initialization(adapter.control_config(name))
    config = variant(name)
    config.model.attention_backend = "mosaic"
    with pytest.raises(ValueError, match="causal strips"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("run_id", FROZEN_RUNS)
def test_adapter_installs_its_guards_and_the_score_bias_checks(run_id):
    config = run_config(run_id)
    gated = FROZEN_RUNS[run_id][0] == "per_order_query_key_gate"
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
        adapter.experiment_spec(adapter.control_config(FROZEN_RUNS[run_id][0]), entry_module=adapter.__name__)


@pytest.mark.parametrize("resolved", ("blocked", "fused", "mosaic"))
def test_the_pilot_must_train_on_the_blocked_strips(monkeypatch, tmp_path, resolved):
    """Acceptance's own pilot check, with this adapter's spec: a pilot that resolved any backend but the
    blocked one fails the allocation before training; a blocked pilot proceeds to its validation checks."""
    from types import SimpleNamespace

    config = run_config("r75dccfcf072e")
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    experiment = {"identity": "test", "source_commit": "a" * 40, "config": config_to_dict(config)}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "config.yaml", gpu_count=1, gpu_type="B200")

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "blocked", "device_kinds": ["NVIDIA B200"]}
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
    reason = ("attention diagnostics failed" if resolved == "blocked"
              else f"requires backend 'blocked'; resolved '{resolved}'")
    with pytest.raises(RuntimeError, match=reason):
        acceptance._run(spec, args, experiment)
    assert json.loads((tmp_path / "failure.json").read_text())["stage"] == "pilot"


@pytest.mark.parametrize("argv", (
    ["config.yaml", "--gpu-type", "A100"], ["config.yaml", "--gpu-type=A6000"], ["config.yaml", "--gpu-type", "H200"],
))
def test_other_gpu_types_are_refused_before_acceptance(monkeypatch, argv):
    monkeypatch.setattr(adapter, "run_acceptance", lambda *args: pytest.fail("Acceptance started"))
    with pytest.raises(ValueError, match="train only on B200, B300, H100"):
        adapter.main(argv)


@pytest.mark.parametrize("argv", (
    ["config.yaml", "--gpu-type", "H100", "--gpu-count", "1"], ["config.yaml", "--gpu-type=h100"],
    ["config.yaml", "--gpu-type", "B200"], ["config.yaml", "--gpu-type=B300"], ["config.yaml", "--runtime-env"],
))
def test_b200_b300_and_h100_allocations_reach_acceptance(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, arguments: calls.append(arguments))
    adapter.main(argv)
    assert calls == [argv]


@pytest.mark.parametrize("name", BACKBONES)
def test_pinned_initialization_is_the_control_recipes_own(name):
    """Builds the full control at seed 42 on the host (seconds): the pinned fingerprint and count."""
    _, _, digest, count = adapter.CONTROLS[name]
    assert acceptance.initialization_facts(adapter.control_config(name)) == {
        "parameter_count": count, "parameter_sha256": digest}


@pytest.mark.parametrize("name", BACKBONES)
def test_full_size_query_key_biases_start_exactly_and_hide_the_control(name):
    """Builds the full query-key variant at seed 42 on the host (about ten seconds): removing its four exact
    score-bias arrays leaves the control's pinned fingerprint."""
    _, _, digest, count = adapter.CONTROLS[name]
    config = variant(name, "query_key")
    assert adapter.without_score_biases(adapter.initial_parameters(config), config) == {
        "parameter_count": count + adapter.ADDED_PARAMETERS["query_key"], "control_parameter_sha256": digest}
