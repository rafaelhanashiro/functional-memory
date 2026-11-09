"""The degree-eight query-distance gate keeps its degree-four recipe and FoX's degree-eight initialization."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import query_distance_degree_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_test_paths
from scaffolds.transformer.configs import config_to_dict, load_config

RUN = Path(adapter.__file__).parent / "runs/r9320e5cd3071/config.yaml"


def _variant():
    config = adapter.control_config()
    config.model.attention_degree = 8
    config.model.attention_backend = "mosaic"
    return config


def test_frozen_run_is_the_degree_four_recipe_at_degree_eight_on_mosaic():
    config = load_config(RUN)
    adapter.check_recipe(config)
    assert config.model.attention_degree == 8 and config.model.attention_backend == "mosaic"
    assert config.model.attention_forget_gate_mode == "query_distance" and not config.model.attention_forget_gate_per_order
    assert config.optim.total_steps == 7570 and config.train.seed == 42 and not config.continuation.parent_checkpoint
    assert config.model.max_seq_len == 2048 and config.data.sequence_length == 2049
    assert config.model.positional_encoding == "none" and config.model.attention_coefficient_kernel_weight_decay


def test_pins_come_from_the_recorded_runs():
    for path, digest in ((adapter.CONTROL, adapter.CONTROL_SHA256), (adapter.REFERENCE, adapter.REFERENCE_SHA256)):
        assert json.loads(path.with_name("record.json").read_text())["config"]["sha256"] == digest
    metadata = json.loads(adapter.REFERENCE.with_name("metadata.json").read_text())
    assert adapter.PARAMETER_COUNT == metadata["parameter_count"]
    reference = adapter.reference_config()
    assert (reference.model.attention_degree, reference.model.attention_forget_gate_mode) == (8, "cumulative")


@pytest.mark.parametrize(("field", "value"), (("attention_degree", 4), ("attention_degree", 12),
                                              ("attention_backend", "blocked")))
def test_only_degree_eight_on_the_mosaic_kernels_is_this_run(field, value):
    config = _variant()
    setattr(config.model, field, value)
    with pytest.raises(ValueError, match="degree eight"):
        adapter.check_recipe(config)


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", .2),
    ("optim", "total_steps", 7571),
    ("train", "seed", 43),
    ("data", "batch_size", 128),
    ("model", "positional_encoding", "rope"),
    ("model", "attention_forget_gate_mode", "cumulative"),
    ("model", "attention_forget_gate_per_order", True),
    ("model", "attention_forget_gate_conditioning", "query_key"),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_amplitude_scale", "none"),
    ("model", "qk_norm_head_scale", False),
))
def test_other_scientific_changes_are_rejected(section, field, value):
    config = _variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_execution_names_may_change():
    config = _variant()
    config.train.workdir = Path("/new/run")
    config.train.checkpoint_dir = Path("/new/run/checkpoints")
    config.train.source_commit = "new_commit"
    config.train.wandb_run_name = "query-distance-m8"
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


def test_spec_installs_both_guards_and_the_mosaic_decay_tests():
    config = load_config(RUN)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert "experiments/transformer_pp/tests/test_query_distance_degree_acceptance.py" in spec.cpu_tests
    gpu = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert "scaffolds/transformer/tests/test_endpoint_decay.py" in gpu
    assert "scaffolds/transformer/tests/test_per_order_decay.py" not in gpu


def _shrink(config):
    config.model.hidden_size = 16
    config.model.num_layers = 2
    config.model.num_heads = config.model.num_kv_heads = 2
    config.model.qk_head_dim = config.model.value_head_dim = 8
    config.model.intermediate_size = 32
    config.model.vocab_size = config.data.synthetic_vocab_size = 64
    return config


def test_query_distance_and_cumulative_fox_share_their_initialization(monkeypatch):
    """Real (small) models: the variant and the cumulative reference build identical parameters."""
    from scaffolds.transformer.acceptance import initialization_facts

    reference = _shrink(adapter.reference_config())
    config = _shrink(_variant())
    config.model.attention_backend = "blocked"  # Mosaic needs value width 64; initialization ignores the backend
    monkeypatch.setattr(adapter, "check_recipe", lambda config: None)  # the full-size recipe is checked above
    monkeypatch.setattr(adapter, "reference_config", lambda: deepcopy(reference))
    monkeypatch.setattr(adapter, "PARAMETER_COUNT", initialization_facts(deepcopy(reference))["parameter_count"])
    result = adapter.check_initialization(config)
    assert result["all_initial_parameters_match_reference"] and result["initialization_reference_run_id"] == "r9353f554b917"
    config.train.seed = 43
    with pytest.raises(ValueError, match="same seed"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("facts", (
    ({"parameter_count": 2, "parameter_sha256": "a"}, {"parameter_count": 2, "parameter_sha256": "b"}),
    ({"parameter_count": 2, "parameter_sha256": "a"}, {"parameter_count": 2, "parameter_sha256": "a"}),
))
def test_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, facts):
    calls = iter(facts)
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: next(calls))
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match="differs from FoX r9353f554b917"):
        adapter.check_initialization(_variant())  # the second case matches but has the wrong parameter count


def test_matching_initialization_proceeds_to_backbone_qualification(monkeypatch):
    facts = {"parameter_count": adapter.PARAMETER_COUNT, "parameter_sha256": "d" * 64}
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: dict(facts))
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda config: {"backbone_checked": True})
    result = adapter.check_initialization(_variant())
    assert result["backbone_checked"] and result["initialization_reference_parameter_sha256"] == "d" * 64
