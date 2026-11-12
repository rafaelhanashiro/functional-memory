"""Query-distance softmax and exp must remain one-change ablations of their pinned no-RoPE FoX controls; a
query-key gate changes the gate's conditioning as well."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.transformer_pp import exponential_query_distance_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

FROZEN_RUNS = {"softmax": "r208110a25271", "exp": "r17444ad0f60e"}
FROZEN_QUERY_KEY_RUNS = {"exp": "rd3501a40e020"}
KERNELS = tuple(adapter.CONTROLS)
CONDITIONINGS = ("query", "query_key")


def variant(kernel, conditioning="query"):
    config = adapter.control_config(kernel)
    config.model.attention_forget_gate_mode = "query_distance"
    config.model.attention_forget_gate_conditioning = conditioning
    config.model.attention_blocked_implementation = "strips"
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


@pytest.mark.parametrize("kernel", KERNELS)
def test_frozen_run_is_its_control_with_query_distance_decay(kernel):
    config = load_config(adapter.RUNS / FROZEN_RUNS[kernel] / "config.yaml")
    adapter.check_recipe(config)
    model = config.model
    assert adapter.kernel(config) == kernel and model.attention_degree is None and model.attention_forget_gate
    assert (model.attention_forget_gate_mode, model.attention_forget_gate_form, model.attention_forget_gate_conditioning,
            model.attention_forget_gate_rate_activation) == ("query_distance", "input", "query", "softplus")
    assert not model.attention_forget_gate_per_order
    assert (model.attention_backend, model.attention_blocked_implementation, model.positional_encoding) == (
        "blocked", "strips", "none")
    assert model.attention_output_scaling == ("none" if kernel == "softmax" else "head_rms_norm")
    assert model.qk_norm and model.qk_norm_type == "rms_norm" and model.qk_norm_sharing == "per_layer"
    assert model.qk_norm_head_scale and model.initialization == "transformer_truncated_reference"
    assert config.optim.total_steps == 7570 and config.optim.warmup_steps == 455 and config.optim.weight_decay == .1
    assert config.train.seed == 42 and config.data.batch_size == 256
    assert model.max_seq_len == 2048 and config.data.sequence_length == 2049
    assert not config.continuation.parent_checkpoint


@pytest.mark.parametrize("kernel", tuple(FROZEN_QUERY_KEY_RUNS))
def test_frozen_query_key_run_is_its_query_distance_counterpart_with_a_key_projection(kernel):
    """The query-key run differs from its kernel's query-distance run only in the gate's conditioning, and from
    the FoX control in that, the decay mode, and the causal strips."""
    config = load_config(adapter.RUNS / FROZEN_QUERY_KEY_RUNS[kernel] / "config.yaml")
    adapter.check_recipe(config)
    model = config.model
    assert adapter.kernel(config) == kernel and adapter.key_conditioned(config)
    assert (model.attention_forget_gate_mode, model.attention_forget_gate_form, model.attention_forget_gate_conditioning,
            model.attention_forget_gate_rate_activation) == ("query_distance", "input", "query_key", "softplus")
    assert not model.attention_forget_gate_per_order and model.attention_degree is None
    assert (model.attention_backend, model.attention_blocked_implementation, model.positional_encoding) == (
        "blocked", "strips", "none")
    counterpart = load_config(adapter.RUNS / FROZEN_RUNS[kernel] / "config.yaml")
    assert scientific_differences(config, counterpart) == {"model.attention_forget_gate_conditioning"}
    assert scientific_differences(config, adapter.control_config(kernel)) == {
        "model.attention_forget_gate_mode", "model.attention_forget_gate_conditioning",
        "model.attention_blocked_implementation"}
    assert not config.continuation.parent_checkpoint


@pytest.mark.parametrize("kernel", KERNELS)
def test_pinned_controls_are_the_recorded_cumulative_fox_runs(kernel):
    run, digest, _, count = adapter.CONTROLS[kernel]
    assert json.loads((adapter.RUNS / run / "record.json").read_text())["config"]["sha256"] == digest
    assert json.loads((adapter.RUNS / run / "completion.json").read_text())["parameter_count"] == count
    control = adapter.control_config(kernel)
    assert adapter.kernel(control) == kernel and control.model.attention_degree is None
    assert control.model.attention_forget_gate and control.model.attention_forget_gate_mode == "cumulative"
    assert control.model.attention_forget_gate_conditioning == "query"
    assert control.model.positional_encoding == "none" and control.model.attention_blocked_implementation == "scan"


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_key_projection_count_is_one_row_per_layer_and_head(kernel):
    model = adapter.control_config(kernel).model
    assert adapter.KEY_PROJECTION_PARAMETERS == model.num_layers * model.num_heads * model.hidden_size == 15 * 14 * 896


@pytest.mark.parametrize("kernel", KERNELS)
def test_pinned_initialization_is_the_control_recipes_own(kernel):
    """Builds the full control at seed 42 on the host: a few seconds per kernel."""
    _, _, digest, count = adapter.CONTROLS[kernel]
    assert acceptance.initialization_facts(adapter.control_config(kernel)) == {
        "parameter_count": count, "parameter_sha256": digest}


@pytest.mark.parametrize("kernel", KERNELS)
def test_query_key_initialization_is_the_control_plus_a_zero_key_projection(kernel):
    """Builds the full query-key variant at seed 42 on the host: its key projection starts at exactly zero, and
    the remaining parameters hash as the control's initialization."""
    config = variant(kernel, "query_key")
    params = adapter.initial_parameters(config)
    assert adapter.KEY_PROJECTION in params
    _, _, digest, count = adapter.CONTROLS[kernel]
    assert adapter.without_key_projection(params, config) == {
        "parameter_count": count + adapter.KEY_PROJECTION_PARAMETERS, "control_parameter_sha256": digest}
    assert adapter.KEY_PROJECTION not in params


def _small_state(*, key=0.0, key_shape=(2, 3, 4), backbone=None):
    """A stand-in state: a backbone array and a key projection [layers, heads, hidden]."""
    import jax.numpy as jnp
    from flax import nnx

    module = nnx.Module()
    module.backbone = nnx.Param(jnp.arange(17, dtype=jnp.float32) if backbone is None else backbone)
    if key_shape is not None:
        module.forget_gate_key_kernel = nnx.Param(jnp.full(key_shape, key, jnp.float32))
    return nnx.state(module, nnx.Param)


SMALL = SimpleNamespace(model=SimpleNamespace(num_layers=2, num_heads=3, hidden_size=4))


def test_removing_the_zero_key_projection_recovers_the_remaining_fingerprint():
    import jax.numpy as jnp
    from scaffolds.transformer.training.checkpointing import state_sha256

    facts = adapter.without_key_projection(_small_state(), SMALL)
    assert facts == {"parameter_count": 17 + 24, "control_parameter_sha256": state_sha256(_small_state(key_shape=None))}
    changed = adapter.without_key_projection(_small_state(backbone=jnp.ones(17, jnp.float32)), SMALL)
    assert changed["control_parameter_sha256"] != facts["control_parameter_sha256"]


@pytest.mark.parametrize("state", (
    pytest.param(dict(key=1e-30), id="nonzero"),
    pytest.param(dict(key=float("nan")), id="nan"),
    pytest.param(dict(key_shape=(2, 3, 5)), id="width"),
    pytest.param(dict(key_shape=(3, 2, 4)), id="layers-heads"),
    pytest.param(dict(key_shape=None), id="missing"),
))
def test_a_key_projection_that_does_not_start_at_zero_is_refused(state):
    with pytest.raises(RuntimeError, match="must start at exactly zero"):
        adapter.without_key_projection(_small_state(**state), SMALL)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_only_execution_locations_and_labels_may_differ(kernel, conditioning):
    config = variant(kernel, conditioning)
    adapter.check_recipe(config)
    config.train.workdir = Path("/new/execution")
    config.train.checkpoint_dir = Path("/new/execution/checkpoints")
    config.train.source_commit = "new-source-commit"
    config.train.wandb_run_name = f"{kernel}-query-distance"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/execution/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_the_scan_is_rejected(kernel, conditioning):
    config = variant(kernel, conditioning)
    config.model.attention_blocked_implementation = "scan"
    with pytest.raises(ValueError, match="causal strips"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_cumulative_control_is_not_the_variant(kernel):
    config = adapter.control_config(kernel)
    config.model.attention_blocked_implementation = "strips"
    with pytest.raises(ValueError, match="requires query-distance decay"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_cumulative_query_key_gate_is_rejected(kernel):
    """The query-key gate exists only in query-distance mode."""
    config = variant(kernel, "query_key")
    config.model.attention_forget_gate_mode = "cumulative"
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_the_polynomial_query_distance_run_is_not_an_exponential_variant():
    config = load_config(adapter.RUNS / "r3e26412b2811/config.yaml")
    config.model.attention_blocked_implementation = "strips"
    with pytest.raises(ValueError, match="requires query-distance decay of softmax"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("run", ("r3e26412b2811", "r03be769d2069"))
def test_polynomial_query_key_gates_are_not_exponential_variants(run):
    """The polynomial query-distance control and the per-order query-key polynomial run, with query-key
    conditioning on the strips, are still polynomials."""
    config = load_config(adapter.RUNS / run / "config.yaml")
    config.model.attention_forget_gate_conditioning = "query_key"
    config.model.attention_backend = "blocked"
    config.model.attention_blocked_implementation = "strips"
    with pytest.raises(ValueError, match="requires query-distance decay of softmax"):
        adapter.check_recipe(config)


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "learning_rate", .002),
    ("optim", "min_learning_rate", .0002),
    ("optim", "weight_decay", .2),
    ("optim", "warmup_steps", 500),
    ("optim", "total_steps", 7571),
    ("train", "seed", 43),
    ("train", "dtype", "float32"),
    ("train", "eval_every", 1000),
    ("data", "batch_size", 128),
    ("data", "shuffle_seed", 1),
    ("data", "train_manifest", "/elsewhere/fineweb_12b/train/manifest.json"),
    ("model", "positional_encoding", "rope"),
    ("model", "num_layers", 12),
    ("model", "qk_norm_head_scale", False),
    ("model", "initialization", "transformer_reference"),
))
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_other_scientific_changes_are_rejected(kernel, conditioning, section, field, value):
    config = variant(kernel, conditioning)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Scientific settings differ"):
        adapter.check_recipe(config)


@pytest.mark.parametrize(("field", "value"), (
    ("attention_forget_gate_rate_activation", "sigmoid"),
    ("attention_forget_gate_per_order", True),
    ("attention_forget_gate_form", "constant"),
    ("attention_forget_gate_form", "sigmoid"),
    ("positional_encoding", "path"),
    ("attention_backend", "mosaic"),
    ("attention_backend", "fused"),
    ("attention_backend", "auto"),
))
@pytest.mark.parametrize("kernel", KERNELS)
def test_other_query_key_gate_changes_are_rejected(kernel, field, value):
    """Only the input-form softplus query-key gate with RoPE-free scores on the blocked strips is this
    experiment: bounded rates, per-order gates, other gate forms, PaTH, and other backends are refused,
    whether by the scaffold's validation or by the comparison with the control."""
    config = variant(kernel, "query_key")
    setattr(config.model, field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_changing_the_normalization_alone_is_rejected(kernel, conditioning):
    """Each kernel is checked against its own control, so softmax and exp cannot trade places."""
    config = variant(kernel, conditioning)
    config.model.attention_normalization = kernel == "exp"
    with pytest.raises(ValueError, match="attention_output_scaling"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_a_changed_control_recipe_is_refused(monkeypatch, conditioning):
    config = variant("exp", conditioning)
    run, _, digest, count = adapter.CONTROLS["exp"]
    monkeypatch.setitem(adapter.CONTROLS, "exp", (run, "0" * 64, digest, count))
    with pytest.raises(ValueError, match=f"control recipe {run} changed"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("mismatch", ("digest", "count", "other_kernel"))
def test_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, kernel, mismatch):
    _, _, digest, count = adapter.CONTROLS[kernel]
    other = adapter.CONTROLS["exp" if kernel == "softmax" else "softmax"]
    facts = {"digest": ("0" * 64, count), "count": (digest, count + 1), "other_kernel": other[2:]}[mismatch]
    monkeypatch.setattr(adapter, "initialization_facts",
                        lambda config: {"parameter_sha256": facts[0], "parameter_count": facts[1]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match="Initialization differs from FoX control"):
        adapter.check_initialization(variant(kernel))


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("mismatch", ("digest", "count", "count_without_key", "other_kernel"))
def test_query_key_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, kernel, mismatch):
    _, _, digest, count = adapter.CONTROLS[kernel]
    added = adapter.KEY_PROJECTION_PARAMETERS
    other = adapter.CONTROLS["exp" if kernel == "softmax" else "softmax"]
    facts = {"digest": ("0" * 64, count + added), "count": (digest, count + added + 1),
             "count_without_key": (digest, count), "other_kernel": (other[2], other[3] + added)}[mismatch]
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: "built")
    monkeypatch.setattr(adapter, "without_key_projection",
                        lambda params, config: {"control_parameter_sha256": facts[0], "parameter_count": facts[1]})
    monkeypatch.setattr(adapter, "initialization_facts",
                        lambda config: pytest.fail("A query-key gate's full fingerprint is not the control's"))
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match="Initialization without the key projection differs from FoX control"):
        adapter.check_initialization(variant(kernel, "query_key"))


@pytest.mark.parametrize("kernel", KERNELS)
def test_initialization_checks_the_recipe_before_building_models(monkeypatch, kernel):
    """Acceptance's initialization worker calls check_initialization on its own: the cumulative control
    itself, and a query-key gate on the scan, are refused before any model is built."""
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: pytest.fail("Built a model"))
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: pytest.fail("Built a model"))
    with pytest.raises(ValueError, match="requires query-distance decay"):
        adapter.check_initialization(adapter.control_config(kernel))
    config = variant(kernel, "query_key")
    config.model.attention_blocked_implementation = "scan"
    with pytest.raises(ValueError, match="causal strips"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("kernel", KERNELS)
def test_matching_initialization_proceeds_to_backbone_qualification(monkeypatch, kernel):
    run, _, digest, count = adapter.CONTROLS[kernel]
    monkeypatch.setattr(adapter, "initialization_facts",
                        lambda config: {"parameter_sha256": digest, "parameter_count": count})
    calls = []

    def backbone(config):
        calls.append(config.model.attention_forget_gate_mode)
        return {"backbone_checked": True}

    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", backbone)
    result = adapter.check_initialization(variant(kernel))
    assert calls == ["query_distance"]
    assert result["backbone_checked"] and result["all_initial_parameters_match_control"]
    assert (result["control_run_id"], result["control_parameter_sha256"]) == (run, digest)


@pytest.mark.parametrize("kernel", KERNELS)
def test_matching_query_key_initialization_proceeds_to_backbone_qualification(monkeypatch, kernel):
    run, _, digest, count = adapter.CONTROLS[kernel]
    built = []
    monkeypatch.setattr(adapter, "initial_parameters", lambda config: built.append(config) or "built")
    monkeypatch.setattr(adapter, "without_key_projection", lambda params, config: {
        "control_parameter_sha256": digest, "parameter_count": count + adapter.KEY_PROJECTION_PARAMETERS})
    calls = []

    def backbone(config):
        calls.append((config.model.attention_forget_gate_mode, config.model.attention_forget_gate_conditioning))
        return {"backbone_checked": True}

    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", backbone)
    result = adapter.check_initialization(variant(kernel, "query_key"))
    assert len(built) == 1 and calls == [("query_distance", "query_key")]
    assert result["backbone_checked"] and result["all_other_initial_parameters_match_control"]
    assert result["key_projection_starts_at_zero"] and "all_initial_parameters_match_control" not in result
    assert result["key_projection_parameter_count"] == 188_160
    assert (result["control_run_id"], result["control_parameter_sha256"], result["control_parameter_count"]) == (
        run, digest, count)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_adapter_installs_its_guards_and_the_mechanism_checks(kernel, conditioning):
    config = variant(kernel, conditioning)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == "blocked"
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
    assert "experiments/transformer_pp/tests/test_exponential_query_distance_acceptance.py" in cpu
    for paths in (cpu, gpu):
        assert acceptance.EXPONENTIAL_QUERY_DISTANCE_TESTS in paths
        assert acceptance.FORGET_GATE_ATTENTION_TESTS in paths
        # The pair rate's own checks, FP64 on CPU and the 2,048-token BF16 strips on the GPU.
        assert (acceptance.EXPONENTIAL_QUERY_KEY_TESTS in paths) is (conditioning == "query_key")
    # The endpoint arithmetic shared with the query polynomials, whose GPU cases are the polynomial kernels.
    assert (acceptance.ENDPOINT_DECAY_TESTS in cpu) is (conditioning == "query_key")
    assert acceptance.ENDPOINT_DECAY_TESTS not in gpu
    assert acceptance.BLOCKED_STRIPS_TESTS in cpu
    with pytest.raises(ValueError):
        adapter.experiment_spec(adapter.control_config(kernel), entry_module=adapter.__name__)
