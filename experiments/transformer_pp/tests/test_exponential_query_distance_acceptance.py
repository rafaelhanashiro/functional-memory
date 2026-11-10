"""Query-distance softmax and exp must remain one-change ablations of their pinned no-RoPE FoX controls."""
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import exponential_query_distance_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

FROZEN_RUNS = {"softmax": "r208110a25271", "exp": "r17444ad0f60e"}
KERNELS = tuple(adapter.CONTROLS)


def variant(kernel):
    config = adapter.control_config(kernel)
    config.model.attention_forget_gate_mode = "query_distance"
    config.model.attention_blocked_implementation = "strips"
    return config


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


@pytest.mark.parametrize("kernel", KERNELS)
def test_pinned_controls_are_the_recorded_cumulative_fox_runs(kernel):
    run, digest, _, count = adapter.CONTROLS[kernel]
    assert json.loads((adapter.RUNS / run / "record.json").read_text())["config"]["sha256"] == digest
    assert json.loads((adapter.RUNS / run / "completion.json").read_text())["parameter_count"] == count
    control = adapter.control_config(kernel)
    assert adapter.kernel(control) == kernel and control.model.attention_degree is None
    assert control.model.attention_forget_gate and control.model.attention_forget_gate_mode == "cumulative"
    assert control.model.positional_encoding == "none" and control.model.attention_blocked_implementation == "scan"


@pytest.mark.parametrize("kernel", KERNELS)
def test_pinned_initialization_is_the_control_recipes_own(kernel):
    """Builds the full control at seed 42 on the host: a few seconds per kernel."""
    _, _, digest, count = adapter.CONTROLS[kernel]
    assert acceptance.initialization_facts(adapter.control_config(kernel)) == {
        "parameter_count": count, "parameter_sha256": digest}


@pytest.mark.parametrize("kernel", KERNELS)
def test_only_execution_locations_and_labels_may_differ(kernel):
    config = variant(kernel)
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
def test_the_scan_is_rejected(kernel):
    config = variant(kernel)
    config.model.attention_blocked_implementation = "scan"
    with pytest.raises(ValueError, match="causal strips"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_cumulative_control_is_not_the_variant(kernel):
    config = adapter.control_config(kernel)
    config.model.attention_blocked_implementation = "strips"
    with pytest.raises(ValueError, match="requires query-distance decay"):
        adapter.check_recipe(config)


def test_the_polynomial_query_distance_run_is_not_an_exponential_variant():
    config = load_config(adapter.RUNS / "r3e26412b2811/config.yaml")
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
def test_other_scientific_changes_are_rejected(kernel, section, field, value):
    config = variant(kernel)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Scientific settings differ"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kernel", KERNELS)
def test_changing_the_normalization_alone_is_rejected(kernel):
    """Each kernel is checked against its own control, so softmax and exp cannot trade places."""
    config = variant(kernel)
    config.model.attention_normalization = kernel == "exp"
    with pytest.raises(ValueError, match="attention_output_scaling"):
        adapter.check_recipe(config)


def test_a_changed_control_recipe_is_refused(monkeypatch):
    config = variant("exp")
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
def test_initialization_checks_the_recipe_before_building_models(monkeypatch, kernel):
    """Acceptance's initialization worker calls check_initialization on its own: the cumulative control
    itself is refused before any model is built."""
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: pytest.fail("Built a model"))
    with pytest.raises(ValueError, match="requires query-distance decay"):
        adapter.check_initialization(adapter.control_config(kernel))


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
def test_adapter_installs_its_guards_and_the_mechanism_checks(kernel):
    config = variant(kernel)
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
    assert acceptance.BLOCKED_STRIPS_TESTS in cpu
    with pytest.raises(ValueError):
        adapter.experiment_spec(adapter.control_config(kernel), entry_module=adapter.__name__)
