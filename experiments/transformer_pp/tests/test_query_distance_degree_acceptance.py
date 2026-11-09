"""The degree-eight and degree-twelve query-distance gates keep their degree-four recipe and FoX's
initialization at their degree."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import query_distance_degree_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_test_paths
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = {8: Path(adapter.__file__).parent / "runs/r9320e5cd3071/config.yaml",
        12: Path(adapter.__file__).parent / "runs/r47bfe39ce9b4/config.yaml"}
DEGREES = pytest.mark.parametrize("degree", tuple(RUNS))


def _variant(degree=8):
    config = adapter.control_config()
    config.model.attention_degree = degree
    config.model.attention_backend = "mosaic"
    return config


@DEGREES
def test_frozen_runs_are_the_degree_four_recipe_at_their_degree_on_mosaic(degree):
    config = load_config(RUNS[degree])
    adapter.check_recipe(config)
    assert config.model.attention_degree == degree and config.model.attention_backend == "mosaic"
    assert config.model.attention_forget_gate_mode == "query_distance" and not config.model.attention_forget_gate_per_order
    assert config.optim.total_steps == 7570 and config.train.seed == 42 and not config.continuation.parent_checkpoint
    assert config.model.max_seq_len == 2048 and config.data.sequence_length == 2049
    assert config.model.positional_encoding == "none" and config.model.attention_coefficient_kernel_weight_decay


def test_pins_come_from_the_recorded_runs():
    from scaffolds.transformer.models import model_parameter_count

    for path, digest in ((adapter.CONTROL, adapter.CONTROL_SHA256), (adapter.REFERENCE, adapter.REFERENCE_SHA256)):
        assert json.loads(path.with_name("record.json").read_text())["config"]["sha256"] == digest
    metadata = json.loads(adapter.REFERENCE.with_name("metadata.json").read_text())
    assert adapter.PARAMETER_COUNT[8] == metadata["parameter_count"]
    reference = adapter.reference_config()
    assert (reference.model.attention_degree, reference.model.attention_forget_gate_mode) == (8, "cumulative")
    # No FoX run exists at degree twelve: its reference is the same recipe with only the degree changed.
    twelve = adapter.reference_config(12)
    expected = config_to_dict(reference)
    expected["model"]["attention_degree"] = 12
    assert config_to_dict(twelve) == expected
    for model in (twelve.model, load_config(RUNS[12]).model):
        assert model_parameter_count(model) == adapter.PARAMETER_COUNT[12]


@DEGREES
@pytest.mark.parametrize(("field", "value"), (("attention_degree", 4), ("attention_degree", 10),
                                              ("attention_backend", "blocked")))
def test_only_degrees_eight_and_twelve_on_the_mosaic_kernels_are_these_runs(field, value, degree):
    config = _variant(degree)
    setattr(config.model, field, value)
    with pytest.raises(ValueError, match="degree eight or twelve"):
        adapter.check_recipe(config)


@DEGREES
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
def test_other_scientific_changes_are_rejected(section, field, value, degree):
    config = _variant(degree)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@DEGREES
def test_execution_names_may_change(degree):
    config = _variant(degree)
    config.train.workdir = Path("/new/run")
    config.train.checkpoint_dir = Path("/new/run/checkpoints")
    config.train.source_commit = "new_commit"
    config.train.wandb_run_name = f"query-distance-m{degree}"
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@DEGREES
def test_spec_installs_both_guards_and_the_mosaic_decay_tests(degree):
    config = load_config(RUNS[degree])
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


@DEGREES
def test_query_distance_and_cumulative_fox_share_their_initialization(monkeypatch, degree):
    """Real (small) models: the variant and the cumulative reference at its degree build identical parameters."""
    from scaffolds.transformer.acceptance import initialization_facts

    reference = _shrink(adapter.reference_config(degree))
    config = _shrink(_variant(degree))
    config.model.attention_backend = "blocked"  # Mosaic needs value width 64; initialization ignores the backend
    monkeypatch.setattr(adapter, "check_recipe", lambda config: None)  # the full-size recipe is checked above
    monkeypatch.setattr(adapter, "reference_config", lambda requested: deepcopy(reference) if requested == degree
                        else pytest.fail(f"The reference must be built at degree {degree}"))
    monkeypatch.setattr(adapter, "PARAMETER_COUNT", {degree: initialization_facts(deepcopy(reference))["parameter_count"]})
    result = adapter.check_initialization(config)
    assert result["all_initial_parameters_match_reference"] and result["initialization_reference_run_id"] == "r9353f554b917"
    assert result.get("initialization_reference_degree") == (None if degree == 8 else degree)
    config.train.seed = 43
    with pytest.raises(ValueError, match="same seed"):
        adapter.check_initialization(config)


@DEGREES
@pytest.mark.parametrize("facts", (
    ({"parameter_count": 2, "parameter_sha256": "a"}, {"parameter_count": 2, "parameter_sha256": "b"}),
    ({"parameter_count": 2, "parameter_sha256": "a"}, {"parameter_count": 2, "parameter_sha256": "a"}),
))
def test_initialization_mismatch_stops_before_backbone_qualification(monkeypatch, facts, degree):
    calls = iter(facts)
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: next(calls))
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization",
                        lambda config: pytest.fail("A mismatched initialization must stop before the next check"))
    with pytest.raises(RuntimeError, match=f"differs from FoX r9353f554b917 at degree {adapter.DEGREE_NAMES[degree]}"):
        adapter.check_initialization(_variant(degree))  # the second case matches but has the wrong parameter count


@DEGREES
def test_matching_initialization_proceeds_to_backbone_qualification(monkeypatch, degree):
    facts = {"parameter_count": adapter.PARAMETER_COUNT[degree], "parameter_sha256": "d" * 64}
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: dict(facts))
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda config: {"backbone_checked": True})
    result = adapter.check_initialization(_variant(degree))
    assert result["backbone_checked"] and result["initialization_reference_parameter_sha256"] == "d" * 64
    assert result["parameter_count"] == adapter.PARAMETER_COUNT[degree]
