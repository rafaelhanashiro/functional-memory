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


FORMS = tuple(adapter.CONTROLS)


def variant(form):
    config = adapter.control_config(form)
    config.model.attention_coefficient_input_normalization = "l2"
    return config


@pytest.mark.parametrize("form", FORMS)
def test_pinned_controls_and_single_scientific_change(form):
    run_id, digest = adapter.CONTROLS[form]
    path = adapter.RUNS / run_id
    assert hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest() == digest
    assert json.loads((path / "record.json").read_text())["config"]["sha256"] == digest
    assert json.loads((path / "completion.json").read_text())["parameter_count"] == adapter.PARAMETER_COUNT
    before, config = adapter.control_config(form), variant(form)
    adapter.check_recipe(config)
    first, second = config_to_dict(before), config_to_dict(config)
    differences = {f"{section}.{key}" for section in first for key in first[section]
                   if first[section][key] != second[section][key]}
    assert differences == {"model.attention_coefficient_input_normalization"}
    model = config.model
    assert (model.attention_degree, model.positional_encoding, model.attention_backend) == (8, "rope", "mosaic")
    assert model.attention_coefficient_kernel_weight_decay and config.optim.weight_decay == .1
    assert (model.num_layers, model.hidden_size, model.num_heads) == (15, 896, 14)
    assert (config.train.seed, config.optim.total_steps, config.data.batch_size) == (42, 7570, 256)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(("section", "field", "value"), (
    ("model", "attention_coefficient_input_normalization", "none"),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_amplitude_initial", 1.0),
    ("model", "attention_coefficient_projection_initial_std", .1),
    ("model", "attention_degree", 12),
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
def test_other_changes_are_rejected(form, section, field, value):
    config = variant(form)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("form", FORMS)
def test_execution_paths_and_monitoring_labels_may_change(form):
    config = variant(form)
    config.train.workdir = Path("/new/workdir")
    config.train.checkpoint_dir = Path("/new/checkpoints")
    config.train.source_commit = "another-source"
    config.train.wandb_run_name = "another-run"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


def test_changed_control_source_is_rejected(tmp_path, monkeypatch):
    run_id, _ = adapter.CONTROLS["direct"]
    target = tmp_path / run_id
    target.mkdir()
    (target / "config.yaml").write_text("changed: true\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="Pinned.*changed"):
        adapter.control_config("direct")
    with pytest.raises(ValueError, match="No coefficient-input L2 control"):
        adapter.control_config("other")


@pytest.mark.parametrize("form", FORMS)
def test_acceptance_requires_new_feature_checks_and_native_mosaic(form):
    config = variant(form)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    cpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=True)
    gpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert adapter.FEATURE_TESTS in cpu and adapter.OWN_TESTS in cpu
    assert adapter.FEATURE_TESTS in gpu
    assert "scaffolds/transformer/tests/test_mosaic_query.py" in gpu
    assert spec.required_backend == "mosaic" and spec.pilot_steps == 20
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization


@pytest.mark.parametrize("form", FORMS)
def test_initialization_check_requires_exact_pinned_fingerprint(form, monkeypatch):
    config = variant(form)
    facts = {"parameter_count": adapter.PARAMETER_COUNT, "parameter_sha256": adapter.PARAMETER_SHA}
    monkeypatch.setattr(adapter, "initialization_facts", lambda config: facts.copy())
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda config: {"backbone_passed": True})
    result = adapter.check_initialization(config)
    assert result["all_initial_parameters_match_control"] and result["backbone_passed"]
    assert result["control_run_id"] == adapter.CONTROLS[form][0]
    facts["parameter_sha256"] = "different"
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("form", FORMS)
def test_l2_feature_preserves_every_initial_parameter_on_small_model(form):
    config = adapter.control_config(form)
    config.model = replace(config.model, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=2,
                           qk_head_dim=16, value_head_dim=16, intermediate_size=64,
                           vocab_size=64, max_seq_len=16, attention_backend="blocked")
    changed = deepcopy(config)
    changed.model.attention_coefficient_input_normalization = "l2"
    assert acceptance.initialization_facts(changed) == acceptance.initialization_facts(config)
