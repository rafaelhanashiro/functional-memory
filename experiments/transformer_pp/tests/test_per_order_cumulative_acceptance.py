"""Per-order cumulative gates must remain one-change ablations of their pinned shared-gate FoX controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp import per_order_cumulative_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

FROZEN_RUNS = {4: "r319b83140040", 8: "rf8fc9ab2cecc", 12: "r95bfff096cd4"}
# Degrees four and eight train on the blocked backend's causal strips, twelve on the Mosaic kernels.
FROZEN_BACKENDS = {4: "blocked", 8: "blocked", 12: "mosaic"}
DEGREES = tuple(adapter.CONTROLS)
RECORDED_CONTROL_DEGREES = tuple(degree for degree in DEGREES if degree not in adapter.DERIVED_CONTROL_DEGREES)


def variant(degree, backend="blocked"):
    config = adapter.control_config(degree)
    config.model.attention_forget_gate_per_order = True
    config.model.attention_blocked_implementation = "strips"
    config.model.attention_backend = backend
    return config


def recipe_path(degree):
    return str(adapter.ROOT / "runs" / FROZEN_RUNS[degree] / "config.yaml")


def frozen(degree, monkeypatch):
    """A frozen run reads Kennel's manifests. Another cluster's checkout cannot read them, so there the
    pinned digests stand in; the run's own acceptance hashes the files on Kennel."""
    config = load_config(adapter.ROOT / "runs" / FROZEN_RUNS[degree] / "config.yaml")
    pinned = {getattr(config.data, key): digest for key, digest in adapter.MANIFEST_SHA256.items()}
    read = adapter.manifest_sha256
    monkeypatch.setattr(adapter, "manifest_sha256", lambda path: read(path) or pinned[path])
    return config


@pytest.mark.parametrize("degree", tuple(FROZEN_RUNS))
def test_frozen_run_is_its_control_with_per_order_gates(degree, monkeypatch):
    config = frozen(degree, monkeypatch)
    adapter.check_recipe(config)
    model = config.model
    assert model.attention_degree == degree and model.attention_forget_gate and model.attention_forget_gate_per_order
    assert (model.attention_forget_gate_mode, model.attention_forget_gate_form, model.attention_forget_gate_conditioning,
            model.attention_forget_gate_rate_activation) == ("cumulative", "input", "query", "softplus")
    assert (model.attention_backend, model.attention_blocked_implementation, model.positional_encoding) == (
        FROZEN_BACKENDS[degree], "strips", "none")
    assert (model.attention_coefficient_conditioning, model.attention_coefficient_activation) == ("query", "tanh")
    assert model.attention_coefficient_scale == model.attention_coefficient_amplitude_scale == "factorial"
    assert model.attention_coefficient_kernel_weight_decay and not model.attention_normalization
    assert model.attention_output_scaling == "head_rms_norm" and model.qk_norm_head_scale
    assert model.initialization == "transformer_truncated_reference"
    assert config.optim.total_steps == 7570 and config.optim.warmup_steps == 455 and config.optim.weight_decay == .1
    assert config.train.seed == 42 and config.data.batch_size == 256
    assert model.max_seq_len == 2048 and config.data.sequence_length == 2049
    # Every run reads Kennel's copies of the manifests, which every control's data identity names.
    assert all(path.startswith("/data/cl/u/rafah/home/datasets/fineweb_12b/")
               for path in (config.data.train_manifest, config.data.eval_manifest))
    assert not config.continuation.parent_checkpoint


def test_the_degree_twelve_run_is_the_degree_eight_run_on_mosaic_at_degree_twelve():
    """The series differs only in degree and implementation: the Mosaic run copies the degree-eight run's recipe."""
    actual, expected = (config_to_dict(load_config(recipe_path(degree))) for degree in (12, 8))
    expected["model"].update(attention_degree=12, attention_backend="mosaic")
    for recipe in (actual, expected):
        for key in adapter.EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    assert actual == expected


@pytest.mark.parametrize("degree", RECORDED_CONTROL_DEGREES)
def test_pinned_controls_are_the_recorded_shared_gate_runs(degree):
    run, digest = adapter.CONTROLS[degree]
    directory = adapter.ROOT / "runs" / run
    record = json.loads((directory / "record.json").read_text())
    assert record["config"]["sha256"] == digest
    assert json.loads((directory / "completion.json").read_text())["parameter_count"] == adapter.PARAMETER_COUNT[degree]
    data = json.loads((directory / "data_identity.json").read_text())
    assert {key: data[f"{key}_sha256"] for key in adapter.MANIFEST_SHA256} == adapter.MANIFEST_SHA256
    if "initialization" in record:  # Degree four recorded its initialization at launch.
        assert record["initialization"]["parameter_sha256"] == adapter.INITIALIZATION_SHA256[degree]
    control = adapter.control_config(degree)
    assert control.model.attention_degree == degree and control.model.attention_forget_gate
    assert control.model.attention_forget_gate_mode == "cumulative" and not control.model.attention_forget_gate_per_order
    assert control.model.attention_blocked_implementation == "scan"


def test_degree_twelve_reference_changes_only_the_degree_of_the_pinned_eight_recipe():
    assert adapter.CONTROLS[12] == adapter.CONTROLS[8]
    assert adapter.DERIVED_CONTROL_DEGREES == {12: 8}
    actual = config_to_dict(adapter.control_config(12))
    expected = config_to_dict(adapter.control_config(8))
    expected["model"]["attention_degree"] = 12
    assert actual == expected
    adapter.check_recipe(variant(12))


@pytest.mark.parametrize("degree", DEGREES)
def test_pinned_initialization_is_the_control_recipes_own(degree):
    """Builds the full control at seed 42 on the host: a few seconds per degree."""
    assert acceptance.initialization_facts(adapter.control_config(degree)) == {
        "parameter_count": adapter.PARAMETER_COUNT[degree],
        "parameter_sha256": adapter.INITIALIZATION_SHA256[degree]}


def _recorded(degree):
    return yaml.safe_load((adapter.ROOT / "runs" / adapter.CONTROLS[degree][0] / "config.yaml").read_text())


def test_the_retired_key_is_dropped_only_with_its_recorded_value():
    raw = _recorded(4)
    assert raw["model"]["attention_coefficient_key_kernel_weight_decay"] is False
    adapter.control_recipe(deepcopy(raw), context="degree-four control")
    for value in (True, 0, None):
        changed = deepcopy(raw)
        changed["model"]["attention_coefficient_key_kernel_weight_decay"] = value
        with pytest.raises(ValueError, match="Unexpected model.attention_coefficient_key_kernel_weight_decay"):
            adapter.control_recipe(changed, context="degree-four control")


def test_other_unknown_keys_are_refused():
    raw = _recorded(8)
    raw["model"]["attention_unknown_switch"] = False
    with pytest.raises(ValueError, match="Unexpected model.attention_unknown_switch"):
        adapter.control_recipe(raw, context="degree-eight control")


@pytest.mark.parametrize("degree", DEGREES)
def test_only_execution_locations_and_labels_may_differ(degree):
    config = variant(degree)
    adapter.check_recipe(config)
    config.train.workdir = Path("/new/execution")
    config.train.checkpoint_dir = Path("/new/execution/checkpoints")
    config.train.source_commit = "new-source-commit"
    config.train.wandb_run_name = f"query-m{degree}-per-order-gates"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/execution/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("degree", DEGREES)
def test_shared_gates_query_distance_and_the_scan_are_rejected(degree):
    for change, message in ((("attention_forget_gate_per_order", False), "per-order cumulative"),
                            (("attention_forget_gate_mode", "query_distance"), "per-order cumulative"),
                            (("attention_blocked_implementation", "scan"), "causal strips"),
                            (("attention_forget_gate_conditioning", "query_key"),
                             "Per-order cumulative gates take query conditioning")):
        config = variant(degree)
        setattr(config.model, *change)
        with pytest.raises(ValueError, match=message):
            adapter.check_recipe(config)


@pytest.mark.parametrize("degree", DEGREES)
def test_mosaic_differs_from_the_strips_only_in_its_backend(degree):
    """The Mosaic kernels never run the blocked implementation, so either value passes; shared, query-distance
    and query-key gates and unrelated changes are rejected as on the strips."""
    config = variant(degree, "mosaic")
    adapter.check_recipe(config)
    config.model.attention_blocked_implementation = "scan"
    adapter.check_recipe(config)
    for change, message in ((("attention_forget_gate_per_order", False), "per-order cumulative"),
                            (("attention_forget_gate_mode", "query_distance"), "per-order cumulative"),
                            (("attention_forget_gate_conditioning", "query_key"), "Backend 'mosaic'"),
                            (("positional_encoding", "rope"), "Scientific settings differ"),
                            (("attention_coefficient_kernel_weight_decay", False), "Scientific settings differ")):
        config = variant(degree, "mosaic")
        setattr(config.model, *change)
        with pytest.raises(ValueError, match=message):
            adapter.check_recipe(config)


@pytest.mark.parametrize("degree", (2, 6, 10))
def test_degrees_without_a_shared_gate_control_are_rejected(degree):
    config = variant(4)
    config.model.attention_degree = degree
    with pytest.raises(ValueError, match="No cumulative FoX control"):
        adapter.check_recipe(config)


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", 0.),
    ("optim", "total_steps", 3000),
    ("train", "seed", 43),
    ("train", "dtype", "float32"),
    ("data", "batch_size", 128),
    ("data", "shuffle_seed", 1),
    ("model", "positional_encoding", "rope"),
    ("model", "qk_norm_head_scale", False),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_projection_initial_std", .02),
    ("model", "attention_output_scaling", "none"),
    ("model", "initialization", "transformer_reference"),
))
@pytest.mark.parametrize("degree", DEGREES)
def test_unrelated_changes_are_rejected(degree, section, field, value):
    config = variant(degree)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Scientific settings differ"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("degree", DEGREES)
def test_a_changed_control_recipe_is_refused(degree, monkeypatch):
    config = variant(degree)
    run = adapter.CONTROLS[degree][0]
    monkeypatch.setitem(adapter.CONTROLS, degree, (run, "0" * 64))
    with pytest.raises(ValueError, match=f"control recipe {run} changed"):
        adapter.check_recipe(config)


def test_a_copied_manifest_must_hash_like_the_controls(tmp_path, monkeypatch):
    copy = tmp_path / "manifest.json"
    copy.write_text("{}")
    config = variant(4)
    config.data.train_manifest = str(copy)
    with pytest.raises(ValueError, match="not a copy of the control's dataset manifest"):
        adapter.check_recipe(config)
    config.data.train_manifest = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="not a copy of the control's dataset manifest"):
        adapter.check_recipe(config)
    config.data.train_manifest = str(copy)
    monkeypatch.setitem(adapter.MANIFEST_SHA256, "train_manifest", hashlib.sha256(b"{}").hexdigest())
    adapter.check_recipe(config)


def _tiny(config):
    model = config.model
    model.hidden_size, model.num_layers, model.num_heads, model.num_kv_heads = 8, 2, 2, 2
    model.qk_head_dim = model.value_head_dim = 4
    model.intermediate_size, model.vocab_size = 16, 64
    return config


def _initial_parameters(config):
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    rng = jax.random.PRNGKey(config.train.seed)
    return nnx.state(build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0))), nnx.Param)


@pytest.mark.parametrize("degree", DEGREES)
def test_initialization_guard_recovers_the_control_and_rejects_any_change(degree, monkeypatch):
    """Built models at a tiny shape: each order's gate is the shared draw, and the collapsed state is the
    control's own; a changed order, backbone weight, or seed is refused."""
    import jax
    from scaffolds.transformer.training.checkpointing import state_sha256

    control, config = _tiny(adapter.control_config(degree)), _tiny(variant(degree))
    expected = _initial_parameters(control)
    monkeypatch.setitem(adapter.INITIALIZATION_SHA256, degree, state_sha256(expected))
    monkeypatch.setitem(adapter.PARAMETER_COUNT, degree, sum(leaf.size for leaf in jax.tree.leaves(expected)))
    facts = adapter._check_initialized_parameters(_initial_parameters(config), _initial_parameters(control), config)
    assert facts["all_order_gates_match_shared_control"] and facts["unchanged_parameters_match_control"]
    assert facts["recovered_control_parameter_sha256"] == state_sha256(expected)
    assert facts["added_gate_parameter_count"] == degree * (8 + 2) * 2 * 2
    assert facts["parameter_count"] == adapter.PARAMETER_COUNT[degree] + facts["added_gate_parameter_count"]
    if degree == 12:
        assert facts["control_reference_is_derived"] is True
        assert facts["control_recipe_degree"] == 8 and facts["control_reference_degree"] == 12
        assert facts["control_run_id"] == adapter.CONTROLS[8][0]

    broken = _initial_parameters(config)
    gate = broken["forget_gate_kernel"]
    broken["forget_gate_kernel"] = gate.replace(value=gate[...].at[:, :, degree].add(.01))
    with pytest.raises(RuntimeError, match="forget_gate_kernel differs from the shared-gate control in some order"):
        adapter._check_initialized_parameters(broken, _initial_parameters(control), config)

    broken = _initial_parameters(config)
    head = broken["lm_head"]["kernel"]
    broken["lm_head"]["kernel"] = head.replace(value=head[...] + 1)
    with pytest.raises(RuntimeError, match="lm_head/kernel differs from the shared-gate control"):
        adapter._check_initialized_parameters(broken, _initial_parameters(control), config)

    shared = _initial_parameters(control)  # The control itself has no order axis.
    with pytest.raises(RuntimeError, match="forget_gate_dt_bias differs"):
        adapter._check_initialized_parameters(shared, _initial_parameters(control), config)

    reseeded = deepcopy(config)
    reseeded.train.seed = 43
    with pytest.raises(RuntimeError, match="differs from the shared-gate control"):
        adapter._check_initialized_parameters(_initial_parameters(reseeded), _initial_parameters(control), config)

    # Arrays equal to a control that is not the pinned one: the collapsed fingerprint must still match.
    monkeypatch.setitem(adapter.INITIALIZATION_SHA256, degree, "0" * 64)
    with pytest.raises(RuntimeError, match="Collapsed initialization differs"):
        adapter._check_initialized_parameters(_initial_parameters(config), _initial_parameters(control), config)


def test_initialization_checks_the_recipe_before_building_models(monkeypatch):
    monkeypatch.setattr(adapter, "_check_initialized_parameters", lambda *args: pytest.fail("Built a model"))
    with pytest.raises(ValueError, match="per-order cumulative"):
        adapter.check_initialization(adapter.control_config(4))


@pytest.mark.parametrize("degree", DEGREES)
def test_adapter_installs_its_guards_and_the_mechanism_checks(degree):
    config = variant(degree)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == "blocked"
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
    assert "experiments/transformer_pp/tests/test_per_order_cumulative_acceptance.py" in cpu
    assert acceptance.PER_ORDER_CUMULATIVE_TESTS in cpu and acceptance.PER_ORDER_CUMULATIVE_TESTS in gpu
    # The per-order parameters on CPU; that module's GPU cases qualify the Mosaic query-distance kernels.
    assert acceptance.PER_ORDER_DECAY_TESTS in cpu and acceptance.PER_ORDER_DECAY_TESTS not in gpu
    assert acceptance.BLOCKED_STRIPS_TESTS in cpu
    with pytest.raises(ValueError):
        adapter.experiment_spec(adapter.control_config(degree), entry_module=adapter.__name__)


@pytest.mark.parametrize("degree", DEGREES)
def test_mosaic_recipes_qualify_the_kernels_they_train_on(degree):
    config = variant(degree, "mosaic")
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == "mosaic"
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=False)
    assert "experiments/transformer_pp/tests/test_per_order_cumulative_acceptance.py" in cpu
    # The kernels against FP64 and the strips, and the strips' own references, on both stages.
    assert all(acceptance.PER_ORDER_CUMULATIVE_MOSAIC_TESTS in paths and acceptance.PER_ORDER_CUMULATIVE_TESTS in paths
               for paths in (cpu, gpu))
    # The Mosaic query kernels and their repeatability, at the run's BF16 weight precision.
    assert {"scaffolds/transformer/tests/test_mosaic_query.py",
            "scaffolds/transformer/tests/test_mosaic_repeatability.py"} <= set(gpu)
    assert acceptance.mosaic_test_options(recipe, "mosaic") == ("--mosaic-precision", "bf16")


WORKER = ("--directory", "acceptance", "--gpu-count", "1", "--worker", "pilot", "--output", "pilot.json",
          "--stop-after-step", "20")


@pytest.mark.parametrize("gpu_type", ("B300", "b300", "B200", "A100"))
def test_mosaic_recipes_are_refused_off_h100_before_acceptance(monkeypatch, gpu_type):
    monkeypatch.setattr(adapter, "run_acceptance", lambda *args: pytest.fail("Reached acceptance"))
    for argv in ([recipe_path(12), "--gpu-type", gpu_type], [recipe_path(12), *WORKER, "--gpu-type", gpu_type]):
        with pytest.raises(ValueError, match=f"run only on H100, not on {gpu_type}"):
            adapter.main(argv)


def test_qualified_allocations_reach_acceptance(monkeypatch):
    """H100 (acceptance's default) for Mosaic, either GPU type for the strips, and the launcher's runtime query,
    which names no GPU type."""
    reached = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, argv: reached.append(argv))
    argvs = [[recipe_path(12)], [recipe_path(12), "--gpu-type", "H100"],
             [recipe_path(12), *WORKER, "--gpu-type", "H100"], [recipe_path(12), "--runtime-env"],
             *([recipe_path(degree), "--gpu-type", gpu_type] for degree in (4, 8) for gpu_type in ("B300", "H100"))]
    for argv in argvs:
        adapter.main(argv)
    assert reached == argvs
