"""Recipe and initialization controls for the softplus-SoS experiments.

Each run is its raw-score SoS control's recipe with the softplus transform, its total degree and input power, and
gradient-event capture; a run at the 12-layer width-640 shape's 1x budget also takes derive_recipe's 1x schedule in
place of its 2x control's. Its initial parameters are the control's."""
from copy import deepcopy
import hashlib
import json

import pytest
import yaml

from experiments.transformer_pp import derive_recipe
from experiments.transformer_pp import softplus_sos_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

KEYS = tuple(adapter.CONTROLS)
REFERENCE_M8, REFERENCE_M12 = (8, 15, 896, 7570), (12, 15, 896, 7570)
L12_D640_1X, L12_D640_2X, L12_D640_4X = (12, 12, 640, 7551), (12, 12, 640, 15103), (12, 12, 640, 30205)
# Each control's heads, warmup, batch and validation cadence (eval_early_every, eval_early_until, eval_every).
SCHEDULES = {REFERENCE_M8: (14, 455, 256, (100, 1500, 500)), REFERENCE_M12: (14, 455, 256, (100, 1500, 500)),
             L12_D640_1X: (10, 454, 128, (100, 1500, 500)), L12_D640_2X: (10, 908, 128, (200, 3000, 1000)),
             L12_D640_4X: (10, 1815, 128, (400, 6000, 2000))}
# The allocated runs: label, input power and control.
RUNS = {
    "r41a4c74c5f58": ("query-sos-core-m8-softplus-rope-w-decay", 0, REFERENCE_M8),
    "r841f35a24241": ("query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay", 1, REFERENCE_M8),
    "rc5aac8f37a2e": ("query-m12-sos-softplus-w-decay", 0, REFERENCE_M12),
    "rf637de9b65b1": ("query-m12-sos-softplus-prefactor-w-decay", 1, REFERENCE_M12),
    "r3b9c8cc6fab8": ("query-m12-sos-softplus-prefactor-w-decay-l12-d640-1x", 1, L12_D640_1X),
    "re7593f1d0a4c": ("query-m12-sos-softplus-prefactor-w-decay-l12-d640-2x", 1, L12_D640_2X),
    "r4589172dcd8b": ("query-m12-sos-softplus-prefactor-w-decay-l12-d640-4x", 1, L12_D640_4X),
    "r5a18841b21b5": ("query-m12-sos-softplus-w-decay-l12-d640-1x", 0, L12_D640_1X),
    "r83fa09df3931": ("query-m12-sos-softplus-w-decay-l12-d640-2x", 0, L12_D640_2X),
    "r0623d3020268": ("query-m12-sos-softplus-w-decay-l12-d640-4x", 0, L12_D640_4X),
}
# The 1x schedule's fields other than the update count, which selects the control.
SCHEDULE_FIELDS = tuple((section, key) for section, values in adapter.L12_D640_1X_SCHEDULE.items() for key in values
                        if (section, key) != ("optim", "total_steps"))


def run_config(run_id):
    return load_config(adapter.RUNS / run_id / "config.yaml")


def variant(power, key=REFERENCE_M8):
    """A control's recipe as its runs change it: softplus, total degree and input power, capture, and at the 12-layer
    width-640 shape's 1x budget derive_recipe's 1x schedule."""
    config = adapter.control_config(key)
    for section, values in adapter.CONTROLS[key].get("schedule", {}).items():
        for name, value in values.items():
            setattr(getattr(config, section), name, value)
    config.model.attention_degree = key[0] + power
    config.model.attention_coefficient_input_power = power
    config.model.attention_score_transform = "softplus"
    for name, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, name, value)
    return config


def scientific_differences(config, other):
    """The settings two recipes differ in, apart from execution locations and monitoring labels."""
    first, second = config_to_dict(config), config_to_dict(other)
    for recipe in (first, second):
        for name in adapter.EXECUTION_KEYS:
            recipe["train"].pop(name, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("key", KEYS)
def test_pinned_controls_are_the_recorded_raw_score_sos_runs(key):
    from scaffolds.transformer.models import model_parameter_count

    control = adapter.CONTROLS[key]
    run = control["run_id"]
    assert hashlib.sha256((adapter.RUNS / run / "config.yaml").read_bytes()).hexdigest() == control["config_sha256"]
    assert json.loads((adapter.RUNS / run / "record.json").read_text())["config"]["sha256"] == control["config_sha256"]
    completion = adapter.RUNS / run / "completion.json"
    if completion.exists():
        assert json.loads(completion.read_text())["parameter_count"] == control["parameter_count"]
    config = adapter.control_config(key)
    model = config.model
    assert model_parameter_count(model) == control["parameter_count"]
    assert model.attention_coefficient_parameterization == "sum_of_squares"
    assert (model.attention_score_transform, model.attention_coefficient_input_power) == ("identity", 0)
    assert (model.attention_coefficient_conditioning, model.attention_coefficient_activation) == ("query", "tanh")
    assert model.attention_coefficient_kernel_weight_decay and model.positional_encoding == "rope"
    assert model.attention_backend == "mosaic" and not model.attention_normalization
    assert not model.attention_forget_gate and model.attention_score_bias == "none"
    assert not config.train.gradient_event_capture
    recipe = adapter.control_recipe(key)
    assert (model.attention_degree, model.num_layers, model.hidden_size, recipe["optim"]["total_steps"]) == key
    heads, warmup, batch, cadence = SCHEDULES[key]
    assert (model.num_heads, recipe["optim"]["warmup_steps"], recipe["data"]["batch_size"]) == (heads, warmup, batch)
    assert tuple(recipe["train"][name] for name in ("eval_early_every", "eval_early_until", "eval_every")) == cadence
    assert recipe["train"]["eval_batches"] * batch == 4864  # the reference's validation sequences at every shape


def test_the_12_layer_width_640_controls_share_one_initialization():
    pins = {(adapter.CONTROLS[key]["parameter_sha256"], adapter.CONTROLS[key]["parameter_count"])
            for key in (L12_D640_1X, L12_D640_2X, L12_D640_4X)}
    assert pins == {("c8a7ed88e12ea4c33806ee8a701e0c75655fc86f176790db6e07b189948e89b9", 99_986_216)}
    assert adapter.CONTROLS[L12_D640_1X]["run_id"] == adapter.CONTROLS[L12_D640_2X]["run_id"] == "r08be7093202a"
    assert adapter.CONTROLS[L12_D640_4X]["run_id"] == "r1f935f18fe21"


def test_the_1x_schedule_is_derive_recipes_for_the_2x_control():
    """derive_recipe.py --budget 1 changes exactly the pinned schedule fields of the 2x control, to the pinned values."""
    partner = yaml.safe_load((adapter.RUNS / "r08be7093202a" / "config.yaml").read_text())
    derived, dropped = derive_recipe.derive(partner, budget=1)
    assert not dropped
    assert (derive_recipe.budget_of(partner), derive_recipe.budget_of(derived)) == (2.0, 1.0)
    assert {key: new for key, (_, new) in derive_recipe._changes(partner, derived).items()} == {
        f"{section}.{name}": value for section, values in adapter.L12_D640_1X_SCHEDULE.items()
        for name, value in values.items()}
    assert adapter.CONTROLS[L12_D640_1X]["schedule"] is adapter.L12_D640_1X_SCHEDULE


@pytest.mark.parametrize("power", (0, 1))
@pytest.mark.parametrize("key", KEYS)
def test_both_recipes_preserve_the_control(power, key):
    config = variant(power, key)
    adapter.check_recipe(config)
    assert adapter.control_key(config) == key
    assert config.model.positional_encoding == "rope"
    assert config.model.attention_coefficient_kernel_weight_decay
    assert config.optim.weight_decay == 0.1
    assert config.model.attention_backend == "mosaic"


@pytest.mark.parametrize("run_id", RUNS)
def test_allocated_runs_are_their_control_with_the_composition(run_id):
    label, power, key = RUNS[run_id]
    config = run_config(run_id)
    adapter.check_recipe(config)
    assert adapter.control_key(config) == key
    model = config.model
    assert (model.attention_score_transform, model.attention_degree, model.attention_coefficient_input_power) == (
        "softplus", key[0] + power, power)
    schedule = {f"{section}.{name}" for section, values in adapter.CONTROLS[key].get("schedule", {}).items()
                for name in values}
    assert scientific_differences(config, adapter.control_config(key)) == {
        "model.attention_score_transform", "train.gradient_event_capture", *schedule,
        *(("model.attention_degree", "model.attention_coefficient_input_power") if power else ())}
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["home_cluster"] == "kennel"  # status is the launcher's, so not checked
    assert record["config"]["sha256"] == hashlib.sha256((adapter.RUNS / run_id / "config.yaml").read_bytes()).hexdigest()
    heads, warmup, batch, cadence = SCHEDULES[key]
    assert (config.optim.total_steps, config.optim.warmup_steps, config.data.batch_size) == (key[3], warmup, batch)
    assert (config.train.eval_early_every, config.train.eval_early_until, config.train.eval_every) == cadence
    assert (config.train.seed, config.data.sequence_length, model.num_heads) == (42, 2049, heads)


@pytest.mark.parametrize("run_id", RUNS)
def test_the_control_comparison_covers_every_other_field(monkeypatch, run_id):
    """Every field but the execution settings, changed alone, is refused by name, with validation bypassed so that
    the adapter's own checks must refuse it: the degree and input power by the core-degree check, the depth, width
    and update count by the control selection, everything else by the control comparison."""
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    frozen, missed = run_config(run_id), []
    for section, values in config_to_dict(frozen).items():
        for key, value in values.items():
            if section == "train" and key in adapter.EXECUTION_KEYS:
                continue
            config = deepcopy(frozen)
            setattr(getattr(config, section), key, ("changed", value))
            reason = ("degree-eight or degree-twelve SoS core" if key in (
                          "attention_degree", "attention_coefficient_input_power") else
                      "No raw-score SoS control has this recipe's" if key in (
                          "num_layers", "hidden_size", "total_steps") else f"'{section}.{key}'")
            try:
                adapter.check_recipe(config)
            except ValueError as error:
                if reason not in str(error):
                    missed.append(f"{section}.{key}: {error}")
            else:
                missed.append(f"{section}.{key}: accepted")
    assert not missed


@pytest.mark.parametrize("section,key,value", (
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("optim", "learning_rate", 0.002),
    ("model", "attention_backend", "blocked"),
    ("data", "batch_size", 64),
    ("train", "gradient_event_capture", False),
    ("model", "attention_forget_gate", True),
    ("model", "positional_encoding", "none"),
    ("train", "eval_batches", 76),
))
@pytest.mark.parametrize("key_", KEYS)
def test_unrelated_recipe_changes_are_refused(section, key, value, key_):
    config = variant(0, key_)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("section,key", SCHEDULE_FIELDS)
def test_a_1x_recipe_takes_the_derived_schedule_not_the_2x_controls(section, key):
    """At the 12-layer width-640 shape's 1x budget each of warmup and the validation cadence must take its derived 1x
    value; the update count, which selects the control, is covered by the selection test."""
    config = variant(1, L12_D640_1X)
    adapter.check_recipe(config)
    setattr(getattr(config, section), key, getattr(getattr(adapter.control_config(L12_D640_2X), section), key))
    with pytest.raises(ValueError, match=f"'{section}.{key}'"):
        adapter.check_recipe(config)


def test_the_core_degree_depth_width_and_updates_select_the_control():
    for key in KEYS:
        assert adapter.control_key(variant(1, key)) == key
    config = variant(1, L12_D640_1X)
    config.optim.total_steps = 15103  # the 2x control's update count with the 1x schedule
    assert adapter.control_key(config) == L12_D640_2X
    with pytest.raises(ValueError, match=r"'optim.warmup_steps'"):
        adapter.check_recipe(config)
    refused = []
    for change in ({"total_steps": 18878},  # 2.5x at the 12-layer width-640 shape
                   {"total_steps": 15140},  # the reference shape's 2x update count
                   {"attention_degree": 9},  # a degree-eight core at the 12-layer width-640 shape
                   {"hidden_size": 896},  # the 12-layer depth at the reference width
                   {"num_layers": 15}):  # the reference depth at width 640
        config = variant(1, L12_D640_2X)
        for name, value in change.items():
            setattr(config.optim if name == "total_steps" else config.model, name, value)
        with pytest.raises(ValueError, match="No raw-score SoS control has this recipe's") as error:
            adapter.control_key(config)
        refused.append(str(error.value))
    assert all("r08be7093202a [12, 12, 640, 15103]" in message for message in refused)
    with pytest.raises(ValueError, match="No raw-score SoS control"):
        adapter.control_config((12, 12, 640, 18878))


@pytest.mark.parametrize("key", KEYS)
def test_prefactor_does_not_silently_reduce_core_degree(key):
    config = variant(1, key)
    config.model.attention_degree = key[0] - 1
    with pytest.raises(ValueError, match="degree-eight or degree-twelve SoS core"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key", KEYS)
def test_an_edited_control_recipe_is_refused(monkeypatch, tmp_path, key):
    """Any byte change to a pinned control recipe, even a comment, stops the comparison."""
    config, run = variant(1, key), adapter.CONTROLS[key]["run_id"]
    edited = tmp_path / run / "config.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_bytes((adapter.RUNS / run / "config.yaml").read_bytes() + b"# edited\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    for check in (lambda: adapter.control_config(key), lambda: adapter.check_recipe(config)):
        with pytest.raises(ValueError, match=f"Pinned SoS control recipe {run} changed"):
            check()


@pytest.mark.parametrize("power", (0, 1))
@pytest.mark.parametrize("key", KEYS)
def test_initialization_must_match_recorded_control(monkeypatch, power, key):
    config = variant(power, key)
    control = adapter.CONTROLS[key]
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {
        "parameter_count": control["parameter_count"], "parameter_sha256": control["parameter_sha256"]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    facts = adapter.check_initialization(config)
    assert facts["all_initial_parameters_match_control"]
    assert facts["control_run_id"] == control["run_id"]
    assert facts["sos_core_degree"] == key[0]
    assert list(facts["control_selection"].values()) == list(key)
    assert facts["derived_1x_schedule"] == (key == L12_D640_1X)
    assert facts["initial_kernel"] == ("z*" if power else "") + f"T_{key[0]}(z)"
    for wrong in ({"parameter_count": control["parameter_count"], "parameter_sha256": "wrong"},
                  {"parameter_count": control["parameter_count"] + 1, "parameter_sha256": control["parameter_sha256"]}):
        monkeypatch.setattr(adapter, "initialization_facts", lambda _, wrong=wrong: wrong)
        with pytest.raises(RuntimeError, match=f"Initial parameters differ .* {control['run_id']}"):
            adapter.check_initialization(config)


@pytest.mark.parametrize("key", KEYS)
def test_acceptance_requires_composed_gradient_tests(key):
    spec = adapter.experiment_spec(variant(1, key), entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic"
    assert "scaffolds/transformer/tests/test_mosaic_score_transform.py" in spec.gpu_tests
    assert "scaffolds/transformer/tests/test_mosaic_softplus_sos.py" in spec.gpu_tests
    assert "scaffolds/transformer/tests/test_softplus_sos.py" in spec.cpu_tests


@pytest.mark.parametrize("gpu_type", ("H100", "B200", "b200"))
def test_h100_and_b200_allocations_start_acceptance(monkeypatch, gpu_type):
    calls = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, argv: calls.append(argv))
    argv = ["experiments/transformer_pp/runs/r3b9c8cc6fab8/config.yaml", "--gpu-type", gpu_type]
    adapter.main(argv)
    assert calls == [argv]


@pytest.mark.parametrize("gpu_type", ("B300", "A100", "A6000"))
def test_other_gpu_types_are_refused_before_acceptance(monkeypatch, gpu_type):
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, argv: pytest.fail("acceptance must not start"))
    with pytest.raises(ValueError, match="H100 or B200 only"):
        adapter.main(["experiments/transformer_pp/runs/r3b9c8cc6fab8/config.yaml", "--gpu-type", gpu_type])


# Prefactor runs: W = f(s) h_t(g(s)) at the degree-eight reference control, input power zero.
PREFACTOR_PAIRS = [(transform, prefactor) for transform in adapter.PREFACTOR_TRANSFORMS
                   for prefactor in adapter.PREFACTORS if (transform, prefactor) != ("softplus", "softplus")]


def prefactor_variant(transform, prefactor):
    config = adapter.control_config(REFERENCE_M8)
    config.model.attention_score_transform = transform
    config.model.attention_score_prefactor = prefactor
    for name, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, name, value)
    return config


@pytest.mark.parametrize("transform,prefactor", PREFACTOR_PAIRS)
def test_prefactor_runs_are_the_control_with_g_f_and_capture(transform, prefactor, monkeypatch):
    config = prefactor_variant(transform, prefactor)
    adapter.check_recipe(config)
    assert adapter.control_key(config) == REFERENCE_M8
    assert scientific_differences(config, adapter.control_config(REFERENCE_M8)) == {
        "model.attention_score_prefactor", "train.gradient_event_capture",
        *(("model.attention_score_transform",) if transform != "identity" else ())}
    control = adapter.CONTROLS[REFERENCE_M8]
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {
        "parameter_count": control["parameter_count"], "parameter_sha256": control["parameter_sha256"]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    facts = adapter.check_initialization(config)
    assert facts["control_run_id"] == "reeefbcc19abf" and facts["input_power"] == 0
    assert facts["score_prefactor"] == prefactor and facts["score_transform"] == transform
    assert facts["initial_kernel"] == f"{prefactor}(s)*T_8({'s' if transform == 'identity' else 'z'})"
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert set(adapter.PREFACTOR_GPU_TESTS) <= set(spec.gpu_tests)
    assert set(adapter.PREFACTOR_CPU_TESTS) <= set(spec.cpu_tests)


def test_prefactor_initialization_is_the_controls_on_the_host():
    """The prefactor adds no parameter: the D recipes' initial arrays are the degree-eight control's, at a small
    backbone built exactly as the full one (the full fingerprint is acceptance's)."""
    from dataclasses import replace

    from scaffolds.transformer.acceptance import initialization_facts

    def small(config):
        return replace(config, model=replace(config.model, attention_backend="blocked", num_layers=1, hidden_size=64,
                                             num_heads=2, num_kv_heads=2, intermediate_size=128, vocab_size=64))

    control = initialization_facts(small(adapter.control_config(REFERENCE_M8)))
    for transform, prefactor in PREFACTOR_PAIRS:
        assert initialization_facts(small(prefactor_variant(transform, prefactor))) == control


@pytest.mark.parametrize("change,reason", (
    ({"attention_score_transform": "softplus", "attention_score_prefactor": "softplus"}, "input power's z h_t"),
    ({"attention_score_transform": "abs", "attention_score_prefactor": "exp"}, "attention_score_transform"),
    ({"attention_score_prefactor": "exp", "attention_degree": 9, "attention_coefficient_input_power": 1,
      "attention_score_transform": "softplus"}, "input power zero"),
))
def test_prefactor_runs_outside_the_composition_are_refused(change, reason, monkeypatch):
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    config = prefactor_variant("identity", "exp")
    for name, value in change.items():
        setattr(config.model, name, value)
    with pytest.raises(ValueError, match=reason):
        adapter.check_recipe(config)


def test_prefactor_runs_take_only_the_degree_eight_reference_control():
    config = adapter.control_config(REFERENCE_M12)
    config.model.attention_score_prefactor = "exp"
    for name, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, name, value)
    with pytest.raises(ValueError, match="degree-eight reference control"):
        adapter.check_recipe(config)


# The allocated prefactor runs: label, transform g and prefactor f.
PREFACTOR_RUNS = {
    "r41662adb22fa": ("query-sos-m8-softplus-prefactor-raw-input-w-decay", "identity", "softplus"),
    "r6525664e6696": ("query-sos-m8-exp-prefactor-raw-input-w-decay", "identity", "exp"),
    "raeaa47865867": ("query-sos-m8-exp-prefactor-softplus-input-w-decay", "softplus", "exp"),
    "r8f56a5b7e525": ("query-sos-m8-sigmoid-prefactor-raw-input-w-decay", "identity", "sigmoid"),
    "r5d578a29f080": ("query-sos-m8-sigmoid-prefactor-softplus-input-w-decay", "softplus", "sigmoid"),
    "re6846615ffe2": ("query-sos-m8-swish-prefactor-raw-input-w-decay", "identity", "swish"),
    "r50c893fe159c": ("query-sos-m8-swish-prefactor-softplus-input-w-decay", "softplus", "swish"),
}


def test_the_allocated_prefactor_runs_cover_every_pair_once():
    assert sorted((g, f) for _, g, f in PREFACTOR_RUNS.values()) == sorted(PREFACTOR_PAIRS)


@pytest.mark.parametrize("run_id", PREFACTOR_RUNS)
def test_allocated_prefactor_runs_are_the_control_with_g_f_and_capture(run_id):
    label, transform, prefactor = PREFACTOR_RUNS[run_id]
    config = run_config(run_id)
    adapter.check_recipe(config)
    assert adapter.control_key(config) == REFERENCE_M8
    model = config.model
    assert (model.attention_score_transform, model.attention_score_prefactor) == (transform, prefactor)
    assert (model.attention_degree, model.attention_coefficient_input_power) == (8, 0)
    assert scientific_differences(config, prefactor_variant(transform, prefactor)) == set()
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256((adapter.RUNS / run_id / "config.yaml").read_bytes()).hexdigest()
