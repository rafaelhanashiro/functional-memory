"""The coefficient-input experiment changes one field of its frozen controls."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import coefficient_l2_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict


CASES = tuple((degree, form) for degree, forms in adapter.CONTROLS.items() for form in forms)


def variant(form, degree):
    config = adapter.control_config(form, degree)
    config.model.attention_coefficient_input_normalization = "l2"
    return config


@pytest.mark.parametrize("degree,form", CASES)
def test_pinned_controls_and_single_scientific_change(degree, form):
    run_id, digest = adapter.CONTROLS[degree][form]
    path = adapter.RUNS / run_id
    assert hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest() == digest
    assert json.loads((path / "record.json").read_text())["config"]["sha256"] == digest
    assert json.loads((path / "completion.json").read_text())["parameter_count"] == adapter.INITIAL_PARAMETERS[degree]["parameter_count"]
    before, config = adapter.control_config(form, degree), variant(form, degree)
    adapter.check_recipe(config)
    first, second = config_to_dict(before), config_to_dict(config)
    differences = {f"{section}.{key}" for section in first for key in first[section]
                   if first[section][key] != second[section][key]}
    assert differences == {"model.attention_coefficient_input_normalization"}
    model = config.model
    assert (model.attention_degree, model.positional_encoding, model.attention_backend) == (degree, "rope", "mosaic")
    assert model.attention_coefficient_kernel_weight_decay and config.optim.weight_decay == .1
    assert (model.num_layers, model.hidden_size, model.num_heads) == (15, 896, 14)
    assert (config.train.seed, config.optim.total_steps, config.data.batch_size) == (42, 7570, 256)


@pytest.mark.parametrize("degree,form", CASES)
@pytest.mark.parametrize(("section", "field", "value"), (
    ("model", "attention_coefficient_input_normalization", "none"),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_amplitude_initial", 1.0),
    ("model", "attention_coefficient_projection_initial_std", .1),
    ("model", "attention_degree", 10),
    ("model", "positional_encoding", "none"),
    ("model", "attention_backend", "blocked"),
    ("model", "qk_norm_head_scale", False),
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", .2),
    ("optim", "total_steps", 7571),
    ("train", "seed", 43),
    ("train", "eval_every", 1000),
    ("data", "batch_size", 128),
))
def test_other_changes_are_rejected(degree, form, section, field, value):
    config = variant(form, degree)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("degree,form", CASES)
def test_execution_paths_and_monitoring_labels_may_change(degree, form):
    config = variant(form, degree)
    config.train.workdir = Path("/new/workdir")
    config.train.checkpoint_dir = Path("/new/checkpoints")
    config.train.source_commit = "another-source"
    config.train.wandb_run_name = "another-run"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("degree,form", CASES)
def test_changed_control_source_is_rejected(degree, form, tmp_path, monkeypatch):
    run_id, _ = adapter.CONTROLS[degree][form]
    target = tmp_path / run_id
    target.mkdir()
    (target / "config.yaml").write_text("changed: true\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="Pinned.*changed"):
        adapter.control_config(form, degree)
    with pytest.raises(ValueError, match="No coefficient-input L2 control"):
        adapter.control_config("other")


@pytest.mark.parametrize("degree,form", CASES)
def test_acceptance_requires_new_feature_checks_and_native_mosaic(degree, form):
    config = variant(form, degree)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    cpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=True)
    gpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert adapter.FEATURE_TESTS in cpu and adapter.OWN_TESTS in cpu
    assert adapter.FEATURE_TESTS in gpu
    assert "scaffolds/transformer/tests/test_mosaic_query.py" in gpu
    assert spec.required_backend == "mosaic" and spec.pilot_steps == 20
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization


@pytest.mark.parametrize("degree,form", CASES)
def test_initialization_check_requires_exact_pinned_fingerprint(degree, form, monkeypatch):
    config = variant(form, degree)
    facts = adapter.INITIAL_PARAMETERS[degree].copy()
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: facts.copy())
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda config: {"backbone_passed": True})
    result = adapter.check_initialization(config)
    assert result["all_initial_parameters_match_control"] and result["backbone_passed"]
    assert result["control_run_id"] == adapter.CONTROLS[degree][form][0]
    facts["parameter_sha256"] = "different"
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("degree,form", CASES)
def test_l2_feature_preserves_every_initial_parameter_on_small_model(degree, form):
    config = adapter.control_config(form, degree)
    config.model = replace(config.model, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=2,
                           qk_head_dim=16, value_head_dim=16, intermediate_size=64,
                           vocab_size=64, max_seq_len=16, attention_backend="blocked")
    changed = deepcopy(config)
    changed.model.attention_coefficient_input_normalization = "l2"
    assert acceptance.initialization_facts(changed) == acceptance.initialization_facts(config)


@pytest.mark.parametrize("form", ("direct", "sum_of_squares"))
def test_raw_input_recipe_changes_only_source_of_completed_l2_control(form):
    before = adapter.control_config(form, 12, "pre_rmsnorm")
    changed = deepcopy(before)
    changed.model.attention_coefficient_input_source = "pre_rmsnorm"
    adapter.check_recipe(changed)
    first, second = config_to_dict(before), config_to_dict(changed)
    differences = {f"{section}.{key}" for section in first for key in first[section]
                   if first[section][key] != second[section][key]}
    assert differences == {"model.attention_coefficient_input_source"}
    rid, digest = adapter.POST_RMSNORM_CONTROLS[12][form]
    assert hashlib.sha256((adapter.RUNS / rid / "config.yaml").read_bytes()).hexdigest() == digest
    assert json.loads((adapter.RUNS / rid / "record.json").read_text())["status"] == "complete"
    spec = adapter.experiment_spec(changed, entry_module=adapter.__name__)
    assert adapter.RAW_FEATURE_TESTS in spec.cpu_tests
    assert adapter.RAW_FEATURE_TESTS in spec.gpu_tests
    assert spec.required_backend == "mosaic"
    for section, key, value in (("model", "attention_coefficient_input_normalization", "none"),
                                ("model", "attention_coefficient_kernel_weight_decay", False),
                                ("optim", "learning_rate", .002),
                                ("train", "seed", 43)):
        bad = deepcopy(changed)
        setattr(getattr(bad, section), key, value)
        with pytest.raises(ValueError):
            adapter.check_recipe(bad)


@pytest.mark.parametrize("form", ("direct", "sum_of_squares"))
def test_raw_input_initialization_uses_matching_post_rmsnorm_control(form, monkeypatch):
    config = adapter.control_config(form, 12, "pre_rmsnorm")
    config.model.attention_coefficient_input_source = "pre_rmsnorm"
    facts = adapter.INITIAL_PARAMETERS[12].copy()
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: facts.copy())
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda config: {"backbone_passed": True})
    result = adapter.check_initialization(config)
    assert result["all_initial_parameters_match_control"] and result["backbone_passed"]
    assert result["control_run_id"] == adapter.POST_RMSNORM_CONTROLS[12][form][0]
    assert result["coefficient_input_source"] == "pre_rmsnorm"
    facts["parameter_sha256"] = "different"
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(config)


def test_raw_input_acceptance_rejects_unpinned_degree_and_control_change(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="No coefficient-input L2 control"):
        adapter.control_config("direct", 8, "pre_rmsnorm")
    with pytest.raises(ValueError, match="Unknown coefficient-input source"):
        adapter.control_config("direct", 12, "unknown")
    rid, _ = adapter.POST_RMSNORM_CONTROLS[12]["direct"]
    target = tmp_path / rid
    target.mkdir()
    (target / "config.yaml").write_text("changed: true\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="Pinned.*changed"):
        adapter.control_config("direct", 12, "pre_rmsnorm")
