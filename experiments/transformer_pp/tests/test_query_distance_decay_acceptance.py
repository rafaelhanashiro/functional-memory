"""The query-distance experiment must remain a one-change ablation of its pinned FoX control."""
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import query_distance_decay_acceptance as adapter
from scaffolds.transformer.configs import load_config


def _variant():
    config = adapter.control_config()
    config.model.attention_forget_gate_mode = "query_distance"
    return config


def test_frozen_run_matches_the_control_and_declared_budget():
    config = load_config(Path(adapter.__file__).parent / "runs/r3e26412b2811/config.yaml")
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.train.seed == 42
    assert config.model.max_seq_len == 2048 and config.data.sequence_length == 2049
    assert not config.continuation.parent_checkpoint
    assert config.model.positional_encoding == "none" and config.model.attention_degree == 4
    assert config.model.attention_coefficient_kernel_weight_decay is True
    assert config.optim.weight_decay == .1
    assert config.model.attention_coefficient_scale == config.model.attention_coefficient_amplitude_scale == "factorial"


def test_pinned_initialization_expectation_comes_from_recorded_control():
    record = json.loads(adapter.CONTROL.with_name("record.json").read_text())
    assert adapter.CONTROL_SHA256 == record["config"]["sha256"]
    assert adapter.INITIALIZATION_SHA256 == record["initialization"]["parameter_sha256"]
    assert adapter.PARAMETER_COUNT == record["initialization"]["parameter_count"]
    assert adapter.control_config().model.attention_forget_gate_mode == "cumulative"


def test_only_execution_paths_and_monitoring_labels_may_differ():
    config = _variant()
    adapter.check_recipe(config)
    config.train.workdir = Path("/new/execution")
    config.train.checkpoint_dir = Path("/new/execution/checkpoints")
    config.train.source_commit = "new-source-commit"
    config.train.wandb_run_name = "query-distance-decay"
    config.train.wandb_group = "transformer_pp"
    config.train.wandb_dir = Path("/new/execution/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "learning_rate", .002),
    ("optim", "min_learning_rate", .0002),
    ("optim", "weight_decay", .2),
    ("optim", "warmup_steps", 500),
    ("optim", "total_steps", 7571),
    ("optim", "query_coefficient_learning_rate_scale", 3.0),
    ("train", "seed", 43),
    ("train", "dtype", "float32"),
    ("train", "eval_every", 1000),
    ("data", "batch_size", 128),
    ("data", "shuffle_seed", 1),
    ("model", "positional_encoding", "rope"),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_scale", "none"),
    ("model", "attention_coefficient_amplitude_scale", "none"),
    ("model", "attention_output_scaling", "none"),
    ("model", "initialization", "transformer_reference"),
    ("model", "qk_norm_head_scale", False),
    ("model", "attention_normalization", True),
))
def test_other_scientific_changes_are_rejected(section, field, value):
    config = _variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_a_consistent_but_different_context_is_rejected():
    config = _variant()
    config.model.max_seq_len, config.data.sequence_length = 4096, 4097
    with pytest.raises(ValueError, match="Scientific settings differ"):
        adapter.check_recipe(config)


def test_cumulative_control_cannot_be_submitted_as_the_new_variant():
    with pytest.raises(ValueError, match="requires query-distance"):
        adapter.check_recipe(adapter.control_config())


@pytest.mark.parametrize(("digest", "count"), (
    ("0" * 64, adapter.PARAMETER_COUNT),
    (adapter.INITIALIZATION_SHA256, adapter.PARAMETER_COUNT + 1),
))
def test_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, digest, count):
    monkeypatch.setattr(adapter, "initialization_facts",
                        lambda config: {"parameter_sha256": digest, "parameter_count": count})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match="Initialization differs from FoX control"):
        adapter.check_initialization(_variant())


def test_matching_initialization_proceeds_to_backbone_qualification(monkeypatch):
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: {
        "parameter_sha256": adapter.INITIALIZATION_SHA256, "parameter_count": adapter.PARAMETER_COUNT})
    calls = []

    def backbone(config):
        calls.append(config.model.attention_forget_gate_mode)
        return {"backbone_checked": True}

    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", backbone)
    result = adapter.check_initialization(_variant())
    assert calls == ["query_distance"]
    assert result["backbone_checked"] and result["all_initial_parameters_match_control"]
    assert result["control_run_id"] == "r4a80e28ce0a7"


def test_experiment_adapter_installs_both_guards():
    spec = adapter.experiment_spec(_variant(), entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization
