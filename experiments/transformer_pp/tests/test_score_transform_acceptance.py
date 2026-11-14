"""The score-transform runs remain their pinned control with only the transform, a softplus run's constant start,
gradient-event capture and (at the 12-layer width-640 shape's 1x budget) the blocked strips changed, and start from
the control's exact initialization but for b_0 = 0 in a softplus run. The 12-layer width-640 runs at the 1x budget
train on the causal strips, on B200, B300 or H100; the other runs train on the Mosaic transformed query-polynomial
kernels, like their controls on Mosaic, on the GPU types those kernels are qualified on."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import score_transform_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

# The allocated runs: transform, constant start, label, and control.
RUNS = {"r1d342ca6e58c": ("softplus", 0.0, "query-m8-w-decay-l12-d640-softplus-input", "l12_d640"),
        "r17a1290cb4ac": ("abs", None, "query-m8-w-decay-l12-d640-abs-input", "l12_d640"),
        "ra3b54e2489f8": ("softplus", 0.0, "query-m8-dk64-w-decay-softplus-input", "reference"),
        "r9eb845aae5bb": ("softplus", 0.0, "query-m12-dk64-w-decay-softplus-input", "reference_m12"),
        "rc6a7e50d27e1": ("softplus", 0.0, "query-m8-w-decay-l12-d640-softplus-input-2x", "l12_d640_2x"),
        "r53d5bbcac16f": ("softplus", 0.0, "query-m12-w-decay-l12-d640-softplus-input-2x", "l12_d640_2x_m12")}
# Each control's shape and schedule: layers, heads, width, updates, warmup, batch; and its degree.
SHAPES = {"l12_d640": (12, 10, 640, 7551, 454, 128), "reference": (15, 14, 896, 7570, 455, 256),
          "reference_m12": (15, 14, 896, 7570, 455, 256), "l12_d640_2x": (12, 10, 640, 15103, 908, 128),
          "l12_d640_2x_m12": (12, 10, 640, 15103, 908, 128)}
DEGREES = {"l12_d640": 8, "reference": 8, "reference_m12": 12, "l12_d640_2x": 8, "l12_d640_2x_m12": 12}
CONTROLS = tuple(adapter.CONTROLS)
CAPTURE_FIELDS = tuple(adapter.CAPTURE_POLICY)


def run_config(run_id):
    return load_config(adapter.RUNS / run_id / "config.yaml")


def variant(transform="softplus", start="default", control="l12_d640"):
    """A control's recipe as its runs change it: transform, constant start, capture, and its runs' backend."""
    config = adapter.control_config(control)
    config.model.attention_score_transform = transform
    config.model.attention_coefficient_constant_initial = adapter.STARTS.get(transform) if start == "default" else start
    for key, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, key, value)
    for key, value in adapter.BACKENDS[control].items():
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


@pytest.mark.parametrize("control", CONTROLS)
def test_pinned_controls_are_the_recorded_runs(control):
    from scaffolds.transformer.models import model_parameter_count

    run, digest, _, count = adapter.CONTROLS[control]
    path = adapter.RUNS / run / "config.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert json.loads((adapter.RUNS / run / "record.json").read_text())["config"]["sha256"] == digest
    completion = adapter.RUNS / run / "completion.json"
    if completion.exists():  # a control still training has no completion yet
        assert json.loads(completion.read_text())["parameter_count"] == count
    config = adapter.control_config(control)
    model = config.model
    assert model_parameter_count(model) == count
    assert (model.attention_score_transform, model.attention_coefficient_constant_initial) == ("identity", None)
    assert (model.attention_backend, model.attention_blocked_implementation) == ("mosaic", "scan")
    assert model.attention_degree == DEGREES[control] and not model.attention_normalization and model.qk_head_dim == 64
    assert (model.attention_coefficient_conditioning, model.attention_coefficient_activation) == ("query", "tanh")
    assert model.attention_coefficient_kernel_weight_decay and model.positional_encoding == "rope"
    assert not model.attention_forget_gate and model.attention_score_bias == "none"
    layers, heads, width, steps, warmup, batch = SHAPES[control]
    assert (model.num_layers, model.num_heads, model.hidden_size) == (layers, heads, width)
    assert adapter.CONTROL_SHAPES[control] == (layers, width, DEGREES[control], steps)
    assert adapter.control_name(config) == control
    assert (config.optim.total_steps, config.optim.warmup_steps, config.data.batch_size) == (steps, warmup, batch)
    assert not config.train.gradient_event_capture
    assert adapter.control_bias_start(control) == 1.0  # Taylor's 1/0!, stored times 0! under the factorial scale


def test_depth_width_degree_and_updates_select_the_control():
    for control in CONTROLS:
        assert adapter.control_name(variant(control=control)) == control
    config = variant(control="reference")
    config.model.hidden_size = 640  # the reference depth at the 12-layer shape's width
    with pytest.raises(ValueError, match="No score-transform control has this recipe's"):
        adapter.control_name(config)
    with pytest.raises(ValueError, match="No score-transform control 'other'"):
        adapter.control_config("other")


@pytest.mark.parametrize("run_id", RUNS)
def test_allocated_runs_are_the_control_with_the_transform_and_capture(run_id):
    from scaffolds.transformer.models import model_parameter_count

    transform, start, label, control = RUNS[run_id]
    config, reference = run_config(run_id), adapter.control_config(control)
    adapter.check_recipe(config)
    assert adapter.control_name(config) == control
    model = config.model
    assert (model.attention_score_transform, model.attention_coefficient_constant_initial) == (transform, start)
    backend = adapter.BACKENDS[control]
    assert (model.attention_backend, model.attention_blocked_implementation) == (
        backend["attention_backend"], backend["attention_blocked_implementation"])
    assert {key: getattr(config.train, key) for key in CAPTURE_FIELDS} == adapter.CAPTURE_POLICY
    # The capture policy's other fields hold their defaults, so only the switch differs in the recipe; the
    # reference-shape run keeps its control's Mosaic setting.
    moved = {"model.attention_backend", "model.attention_blocked_implementation"} if control == "l12_d640" else set()
    assert scientific_differences(config, reference) == {
        "model.attention_score_transform", "train.gradient_event_capture", *moved,
        *(("model.attention_coefficient_constant_initial",) if start is not None else ())}
    assert model_parameter_count(model) == model_parameter_count(reference.model) == adapter.CONTROLS[control][3]
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["home_cluster"] == "kennel"  # status is the launcher's, so not checked
    assert record["config"]["sha256"] == hashlib.sha256((adapter.RUNS / run_id / "config.yaml").read_bytes()).hexdigest()
    _, _, _, steps, warmup, batch = SHAPES[control]
    assert (config.optim.total_steps, config.optim.warmup_steps, config.train.seed) == (steps, warmup, 42)
    assert (config.data.batch_size, config.data.sequence_length, model.max_seq_len) == (batch, 2049, 2048)


@pytest.mark.parametrize("run_id", RUNS)
def test_the_control_comparison_covers_every_other_field(monkeypatch, run_id):
    """Every field but the execution settings and the transform's, changed alone, is rejected by name, with
    validation bypassed so that the adapter's own checks must reject it: the start by the transform's start, the
    depth, width, degree and update count by the control selection, the attention implementation and capture
    policy by their own checks, everything else by the control comparison."""
    monkeypatch.setattr(adapter, "validate_config", lambda config: None)
    frozen, missed = run_config(run_id), []
    for section, values in config_to_dict(frozen).items():
        for key, value in values.items():
            if (section == "train" and key in adapter.EXECUTION_KEYS) or key == "attention_score_transform":
                continue
            config = deepcopy(frozen)
            setattr(getattr(config, section), key, ("changed", value))
            reason = ("starts its constant coefficient" if key == "attention_coefficient_constant_initial" else
                      "No score-transform control" if key in ("num_layers", "hidden_size", "attention_degree",
                                                              "total_steps") else
                      "the recipe must set" if key in adapter.STRIPS else
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
    ("train", "seed", 43), ("train", "eval_every", 1500), ("train", "ckpt_every", 1000), ("data", "batch_size", 64),
    ("model", "attention_degree", 4), ("model", "positional_encoding", "none"), ("model", "qk_norm_head_scale", False),
    ("model", "attention_coefficient_kernel_weight_decay", False), ("model", "attention_output_scaling", "none"),
    ("model", "num_layers", 13), ("model", "hidden_size", 768), ("model", "num_heads", 16),
    ("model", "attention_coefficient_amplitude_initial", 1.0),
)


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("transform", tuple(adapter.STARTS))
@pytest.mark.parametrize(("section", "field", "value"), SCIENTIFIC_CHANGES)
def test_other_scientific_changes_are_rejected(control, transform, section, field, value):
    """With the scaffold's validation, whichever check meets the change first."""
    config = variant(transform, control=control)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("transform", tuple(adapter.STARTS))
def test_only_execution_locations_and_labels_may_differ(control, transform):
    config = variant(transform, control=control)
    adapter.check_recipe(config)
    config.train.workdir = Path("/new/execution")
    config.train.checkpoint_dir = Path("/new/execution/checkpoints")
    config.train.source_commit = "new-source-commit"
    config.train.wandb_run_name = "another-label"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/execution/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("transform,start", (
    ("softplus", None), ("softplus", -.5), ("softplus", 1.0), ("abs", 0.0), ("abs", 1.0), ("identity", None),
    ("identity", 0.0)))
def test_only_each_transforms_own_start_is_accepted(control, transform, start):
    """softplus starts c_0 at zero and abs at Taylor's; the control itself, with or without a start, is refused."""
    config = variant(transform, start, control=control)
    message = ("requires a score transform" if transform == "identity" else "starts its constant coefficient")
    with pytest.raises(ValueError, match=message):
        adapter.check_recipe(config)


@pytest.mark.parametrize("backend,implementation", (
    ("blocked", "strips"), ("blocked", "scan"), ("mosaic", "scan"), ("mosaic", "strips"), ("auto", "strips"),
    ("fused", "strips")))
@pytest.mark.parametrize("transform", tuple(adapter.STARTS))
@pytest.mark.parametrize("control", CONTROLS)
def test_only_each_controls_backend_is_accepted(control, transform, backend, implementation):
    """The 12-layer width-640 runs train on the blocked strips and the reference-shape runs on the Mosaic
    transformed kernels. The other blocked and Mosaic settings are valid transformed recipes, so the adapter
    refuses them itself; validation refuses the other backends, which never evaluate the transform."""
    from scaffolds.transformer.configs.base import validate_config

    config = variant(transform, control=control)
    config.model.attention_backend = backend
    config.model.attention_blocked_implementation = implementation
    required = adapter.BACKENDS[control]
    if {"attention_backend": backend, "attention_blocked_implementation": implementation} == required:
        adapter.check_recipe(config)
    elif backend in ("blocked", "mosaic"):
        validate_config(config)
        with pytest.raises(ValueError, match=f"train on {adapter.TRAINS_ON[required['attention_backend']]}"):
            adapter.check_recipe(config)
    else:
        for check in (validate_config, adapter.check_recipe):
            with pytest.raises(ValueError, match="score transform"):
                check(config)


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize(("field", "value"), (
    ("gradient_event_capture", False), ("gradient_event_factor", 4.0), ("gradient_event_min_norm", 0.5),
    ("gradient_event_history", 50), ("gradient_event_min_history", 10), ("gradient_event_max_states", 5)))
def test_capture_must_be_on_with_the_default_policy(control, field, value):
    config = variant(control=control)
    setattr(config.train, field, value)
    with pytest.raises(ValueError, match="Gradient-event capture must be on with the default policy"):
        adapter.check_recipe(config)


def test_the_pinned_capture_policy_is_the_scaffold_default_rule():
    from scaffolds.transformer.configs.base import TrainConfig

    defaults = TrainConfig()
    assert adapter.CAPTURE_POLICY == {**{key: getattr(defaults, key) for key in CAPTURE_FIELDS},
                                      "gradient_event_capture": True}


@pytest.mark.parametrize("control", CONTROLS)
def test_an_edited_control_recipe_is_refused(monkeypatch, tmp_path, control):
    """Any byte change to a pinned control recipe, even a comment, stops the comparison."""
    config, run = variant(control=control), adapter.CONTROLS[control][0]
    edited = tmp_path / run / "config.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_bytes((adapter.RUNS / run / "config.yaml").read_bytes() + b"# edited\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    for check in (lambda: adapter.control_config(control), lambda: adapter.check_recipe(config)):
        with pytest.raises(ValueError, match=f"pinned control recipe {run} changed"):
            check()


def _tiny(config):
    """The recipe at a small shape, built on the blocked reference."""
    return replace(config, model=replace(config.model, attention_backend="blocked", vocab_size=64, max_seq_len=32,
                                         num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2, qk_head_dim=8,
                                         value_head_dim=8, intermediate_size=64))


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("transform", tuple(adapter.STARTS))
def test_restoring_the_constant_start_recovers_the_controls_initialization(monkeypatch, transform, control):
    """At a small shape: the variant's arrays are the control's, but b_0 = 0 in a softplus run; restoring the
    control's b_0 recovers the control's fingerprint; any other start of b_0 (a normal number near zero, -0, the
    control's 1, BF16) or a changed backbone parameter is detected."""
    import jax
    import jax.numpy as jnp
    from scaffolds.transformer.training.checkpointing import state_sha256

    config, reference = _tiny(variant(transform, control=control)), _tiny(adapter.control_config(control))
    baseline = adapter.initial_parameters(reference)
    count = sum(leaf.size for leaf in jax.tree.leaves(baseline))
    as_control = lambda params: adapter.as_control(params, config, control)
    assert as_control(adapter.initial_parameters(config)) == {
        "parameter_count": count, "control_parameter_sha256": state_sha256(baseline)}

    def changed(path, edit):
        params = adapter.initial_parameters(config)
        params[path] = params[path].replace(value=edit(params[path][...]))
        return params

    if transform == "softplus":
        for edit in (lambda x: x.at[1, 0, 0].set(1e-30), lambda x: x.at[1, 0, 0].set(-0.), lambda x: x.at[0, 1, 0].set(1.),
                     lambda x: x.astype(jnp.bfloat16)):
            with pytest.raises(RuntimeError, match="order 0 must start at exactly 0.0"):
                as_control(changed(adapter.BIAS, edit))
        # Another order's start is the control's comparison's, not the constant start's.
        params = changed(adapter.BIAS, lambda x: x.at[1, 0, 3].set(2.))
        assert as_control(params)["control_parameter_sha256"] != state_sha256(baseline)
    else:
        params = changed(adapter.BIAS, lambda x: x.at[1, 0, 0].set(0.))
        assert as_control(params)["control_parameter_sha256"] != state_sha256(baseline)
    backbone = changed("qk_norm_q_gain", lambda x: x + 1)  # a parameter the control also has
    assert as_control(backbone)["control_parameter_sha256"] != state_sha256(baseline)
    monkeypatch.setattr(adapter, "control_bias_start", lambda name: 0.5)
    if transform == "softplus":
        assert as_control(adapter.initial_parameters(config))["control_parameter_sha256"] != state_sha256(baseline)


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("mismatch", ("digest", "count"))
def test_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, mismatch, control):
    _, _, digest, count = adapter.CONTROLS[control]
    facts = {"digest": ("0" * 64, count), "count": (digest, count + 1)}[mismatch]
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: "built")
    monkeypatch.setattr(adapter, "as_control",
                        lambda params, config: {"parameter_count": facts[1], "control_parameter_sha256": facts[0]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match=f"Initialization differs from control {adapter.CONTROLS[control][0]}"):
        adapter.check_initialization(variant(control=control))


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("transform", tuple(adapter.STARTS))
def test_matching_initialization_proceeds_to_backbone_qualification(monkeypatch, transform, control):
    run, _, digest, count = adapter.CONTROLS[control]
    built = []
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: built.append(config) or "built")
    monkeypatch.setattr(adapter, "as_control", lambda params, config: {
        "parameter_count": count, "control_parameter_sha256": digest})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: {"backbone_checked": config.model.attention_score_transform})
    result = adapter.check_initialization(variant(transform, control=control))
    assert len(built) == 1 and result["backbone_checked"] == transform
    assert (result["control_run_id"], result["control_parameter_sha256"], result["control_parameter_count"]) == (
        run, digest, count)
    assert result["score_transform"] == transform and result["constant_coefficient_start"] == adapter.STARTS[transform]
    assert result["all_other_initial_parameters_match_control"]


@pytest.mark.parametrize("control", CONTROLS)
def test_initialization_checks_the_recipe_before_building_models(monkeypatch, control):
    """Acceptance's initialization worker calls check_initialization on its own."""
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: pytest.fail("Built a model"))
    with pytest.raises(ValueError, match="requires a score transform"):
        adapter.check_initialization(adapter.control_config(control))
    config = variant(control=control)
    config.model.attention_blocked_implementation = "scan" if control == "l12_d640" else "strips"
    with pytest.raises(ValueError, match="the recipe must set"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("run_id", RUNS)
def test_adapter_installs_its_guards_and_the_transform_checks(run_id):
    config = run_config(run_id)
    control = RUNS[run_id][3]
    backend = adapter.BACKENDS[control]["attention_backend"]
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == backend and spec.runtime_env == {}
    assert adapter.RUNS.parent.parents[1] / adapter.OWN_TESTS == Path(__file__).resolve()
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend=backend, cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend=backend, cpu=False)
    shared = {adapter.OWN_TESTS, acceptance.SCORE_TRANSFORM_TESTS, acceptance.GRADIENT_EVENT_TESTS,
              acceptance.QUERY_TANH_POLYNOMIAL_TESTS}
    never = {acceptance.SCORE_BIAS_TESTS, acceptance.SCORE_BIAS_LOWERING_TESTS, acceptance.MOSAIC_SCORE_BIAS_TESTS,
             acceptance.SCORE_TRANSFORM_STRIPS_TESTS, acceptance.SCORE_BIAS_STRIPS_TESTS}
    if control == "l12_d640":
        # CPU: the transform's references, start, refusals and diagnostics, the strips' own checks, capture, and
        # this adapter, beside the backbone's CPU checks; no Mosaic module and no score-bias module.
        assert shared | {acceptance.BLOCKED_STRIPS_TESTS} <= set(cpu) and not never & set(cpu)
        assert not any("mosaic" in path for path in cpu)
        # GPU: the strips at production precision, with the transform and with score biases, and nothing that
        # could skip or refuse there.
        assert gpu == (acceptance.SCORE_BIAS_STRIPS_TESTS, acceptance.SCORE_TRANSFORM_STRIPS_TESTS)
        assert acceptance.mosaic_test_options(recipe, "blocked") == ()
    else:
        # CPU: the same references beside the transformed kernels' tile mathematics, shared memory and lowering,
        # no strips module. GPU: the transformed kernels against FP64 and the strips, beside the query kernels'
        # own GPU cases (which the control's acceptance ran), at the kernels' BF16 weight precision.
        assert shared | {acceptance.SCORE_TRANSFORM_LOWERING_TESTS} <= set(cpu) and not never & set(cpu)
        assert acceptance.BLOCKED_STRIPS_TESTS not in cpu
        assert acceptance.MOSAIC_SCORE_TRANSFORM_TESTS in gpu and not never & set(gpu)
        assert {"scaffolds/transformer/tests/test_mosaic_query.py",
                "scaffolds/transformer/tests/test_mosaic_repeatability.py"} <= set(gpu)
        assert acceptance.mosaic_test_options(recipe, "mosaic") == ("--mosaic-precision", "bf16")
    with pytest.raises(ValueError):
        adapter.experiment_spec(adapter.control_config(control), entry_module=adapter.__name__)


@pytest.mark.parametrize("resolved", ("blocked", "fused", "mosaic"))
@pytest.mark.parametrize("run_id", RUNS)
def test_the_pilot_must_train_on_its_controls_backend(monkeypatch, tmp_path, run_id, resolved):
    """Acceptance's own pilot check, with this adapter's spec: a pilot that resolved any backend but the run's own
    fails the allocation before training; a pilot on it proceeds to its validation checks."""
    from types import SimpleNamespace

    config = run_config(run_id)
    required = adapter.BACKENDS[RUNS[run_id][3]]["attention_backend"]
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    experiment = {"identity": "test", "source_commit": "a" * 40, "config": config_to_dict(config)}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "config.yaml", gpu_count=1, gpu_type="H100")

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": required, "device_kinds": ["NVIDIA H100 PCIe"]}
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


def recipe_path(run_id):
    return str(adapter.RUNS / run_id / "config.yaml")


@pytest.mark.parametrize("run_id", RUNS)
def test_the_launchers_runtime_query_exports_nothing(capsys, run_id):
    """The launcher evaluates the adapter's --runtime-env output; the default allocator share applies."""
    adapter.main([recipe_path(run_id), "--runtime-env"])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("gpu_type", ("A100", "A6000", "H200", "L40S", "B200", "B300"))
@pytest.mark.parametrize("run_id", RUNS)
def test_other_gpu_types_are_refused_before_acceptance(monkeypatch, run_id, gpu_type):
    control = RUNS[run_id][3]
    allowed = adapter.GPU_TYPES[control]
    if gpu_type in allowed:
        return
    monkeypatch.setattr(adapter, "run_acceptance", lambda *args: pytest.fail("Acceptance started"))
    for value in (gpu_type, gpu_type.lower()):
        for argv in ([recipe_path(run_id), "--gpu-type", value], [recipe_path(run_id), f"--gpu-type={value}"]):
            message = (rf"trains only on {', '.join(allowed)} \({adapter.GPU_REASONS[control]}\), not on {value}\. "
                       rf"Submit it with GPU types {','.join(allowed)}\.")
            with pytest.raises(ValueError, match=message):
                adapter.main(argv)


@pytest.mark.parametrize("run_id", RUNS)
def test_qualified_allocations_reach_acceptance(monkeypatch, run_id):
    calls = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, arguments: calls.append(arguments))
    allowed = adapter.GPU_TYPES[RUNS[run_id][3]]
    argvs = [[recipe_path(run_id), "--gpu-type", gpu_type, "--gpu-count", "1"] for gpu_type in allowed]
    argvs.append([recipe_path(run_id), f"--gpu-type={allowed[-1].lower()}"])
    # A worker's invocation (acceptance's _run_worker) passes the same gate.
    argvs.append([recipe_path(run_id), "--directory", "/tmp/acceptance", "--gpu-count", "1", "--gpu-type", "H100",
                  "--worker", "pilot", "--output", "/tmp/pilot.json", "--stop-after-step", "20"])
    for argv in argvs:
        adapter.main(argv)
    assert calls == argvs


def test_the_gpu_types_are_each_controls_qualified_hardware():
    """The strips' GPU tests ran on B200, B300 and H100; the transformed Mosaic kernels' on H100."""
    assert adapter.GPU_TYPES == {"l12_d640": ("B200", "B300", "H100"), "reference": ("H100",),
                                 "reference_m12": ("H100",), "l12_d640_2x": ("H100",), "l12_d640_2x_m12": ("H100",)}
    assert set(adapter.GPU_TYPES) == set(adapter.GPU_REASONS) == set(adapter.CONTROLS) == set(adapter.BACKENDS)


@pytest.mark.parametrize("control", CONTROLS)
def test_pinned_initialization_is_the_control_recipes_own(control):
    """Builds the full control at seed 42 on the host (seconds): the pinned fingerprint and count."""
    _, _, digest, count = adapter.CONTROLS[control]
    assert acceptance.initialization_facts(adapter.control_config(control)) == {
        "parameter_count": count, "parameter_sha256": digest}


@pytest.mark.parametrize("run_id", RUNS)
def test_full_size_runs_start_as_the_control_but_b0(run_id):
    """Builds each run's own recipe at seed 42 on the host: its control's parameter count, b_0 at exactly 0 in
    every layer and head of a softplus run, and with the control's b_0 the control's pinned fingerprint."""
    import numpy as np

    _, start, _, control = RUNS[run_id]
    config = run_config(run_id)
    params = adapter.initial_parameters(config)
    bias = np.asarray(params[adapter.BIAS][...])
    layers, heads = SHAPES[control][:2]
    assert bias.shape == (layers, heads, DEGREES[control] + 1)
    np.testing.assert_array_equal(bias[..., 1:], 1.0)
    assert (bias[..., 0] == (0.0 if start == 0.0 else 1.0)).all()
    _, _, digest, count = adapter.CONTROLS[control]
    assert adapter.as_control(params, config) == {"parameter_count": count, "control_parameter_sha256": digest}


def test_the_gpu_gate_needs_the_recipe_and_passes_the_runtime_query():
    with pytest.raises(ValueError, match="needs the run's recipe"):
        adapter.require_gpu_type(["--gpu-type", "H100"])
    adapter.require_gpu_type(["--gpu-type", "A100", "--runtime-env"])
    with pytest.raises(ValueError, match="causal strips' GPU tests run there"):
        adapter.require_gpu_type([recipe_path("r1d342ca6e58c"), "--gpu-type", "A100"])
    with pytest.raises(ValueError, match="transformed query-polynomial kernels are qualified there"):
        adapter.require_gpu_type([recipe_path("ra3b54e2489f8"), "--gpu-type", "B300"])
