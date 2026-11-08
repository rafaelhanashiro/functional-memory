"""Coefficient-gate runs change only the gate (and degree 10) and start exactly at the ungated recipe."""
import hashlib

import pytest

from experiments.transformer_pp import coefficient_gate_acceptance as adapter
from scaffolds.transformer.acceptance import COEFFICIENT_GATE_TESTS, test_paths as selected_test_paths
from scaffolds.transformer.configs import config_to_dict


def variant(degree=8):
    config = adapter.control_config()
    config.model.attention_coefficient_gate = "sigmoid"
    config.model.attention_degree = degree
    return config


def test_control_recipe_is_pinned():
    assert hashlib.sha256(adapter.CONTROL.read_bytes()).hexdigest() == adapter.CONTROL_SHA256


@pytest.mark.parametrize("degree", adapter.DEGREES)
def test_each_gated_degree_preserves_the_control_budget_and_recipe(degree):
    config = variant(degree)
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.optim.warmup_steps == 455 and config.train.seed == 42
    assert config.data.batch_size == 256 and config.data.sequence_length == 2049
    assert config.model.qk_head_dim == 64 and config.model.positional_encoding == "rope"
    assert config.model.attention_backend == "mosaic" and config.model.attention_output_scaling == "head_rms_norm"
    assert config.model.attention_coefficient_kernel_weight_decay and config.model.qk_norm_head_scale
    assert config.model.attention_coefficient_amplitude_initial == 0 and not config.model.attention_forget_gate


@pytest.mark.parametrize("degree", (4, 12))
def test_other_degrees_are_not_gate_runs(degree):
    with pytest.raises(ValueError):
        adapter.check_recipe(variant(degree))


def test_the_ungated_recipe_is_not_a_gate_run():
    config = variant()
    config.model.attention_coefficient_gate = "none"
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize(("section", "field", "value"), (
    ("optim", "total_steps", 18925),
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", .0),
    ("data", "batch_size", 128),
    ("train", "seed", 43),
    ("model", "qk_norm_head_scale", False),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_projection_initial_std", .02),
    ("model", "initialization", "transformer_reference"),
))
def test_unrelated_scientific_changes_are_rejected(section, field, value):
    config = variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_execution_names_may_change():
    config = variant()
    config.train.workdir = "/new/run"
    config.train.checkpoint_dir = "/new/run/checkpoints"
    config.train.wandb_run_name = "query-m8-w-decay-order-gates"
    adapter.check_recipe(config)


def test_a_copied_manifest_must_hash_like_the_control(tmp_path, monkeypatch):
    copy = tmp_path / "manifest.json"
    copy.write_text("{}")
    config = variant()
    config.data.train_manifest = str(copy)
    with pytest.raises(ValueError, match="dataset manifest"):
        adapter.check_recipe(config)
    monkeypatch.setitem(adapter.MANIFEST_SHA256, "train_manifest", hashlib.sha256(b"{}").hexdigest())
    adapter.check_recipe(config)


def test_acceptance_adds_the_gate_and_adapter_checks():
    config = variant(10)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    cpu = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=True)
    gpu = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert COEFFICIENT_GATE_TESTS in cpu
    assert "experiments/transformer_pp/tests/test_coefficient_gate_acceptance.py" in cpu
    assert {path for path in gpu if path.startswith(f"{COEFFICIENT_GATE_TESTS}::")} >= {
        f"{COEFFICIENT_GATE_TESTS}::test_gated_coefficients_through_mosaic_match_fp64[{degree}]"
        for degree in adapter.DEGREES}
    assert spec.check_initialization is adapter.check_initialization and spec.check_config is adapter.check_recipe


@pytest.mark.parametrize("degree", adapter.DEGREES)
def test_initialization_check_finds_exact_ungated_parity(degree, monkeypatch):
    from scaffolds.transformer.tests.test_coefficient_gates import _config

    monkeypatch.setattr(adapter, "check_recipe", lambda config: None)  # a small backbone, not the control
    parity = adapter.check_initialization(_config(degree))["ungated_parity"]
    assert parity["gate_parameters_exact_zero"] and parity["shared_parameters_match_ungated_recipe"]
    assert parity["gate_parameter_count"] == 2 * 2 * 17 * (degree + 1)
