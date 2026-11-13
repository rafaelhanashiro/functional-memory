"""Recipe and initialization controls for the two softplus-SoS experiments."""
from dataclasses import replace

import pytest

from experiments.transformer_pp import softplus_sos_acceptance as adapter


def variant(power):
    config = adapter.control_config()
    config.model.attention_degree = 8 + power
    config.model.attention_coefficient_input_power = power
    config.model.attention_score_transform = "softplus"
    for key, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, key, value)
    return config


@pytest.mark.parametrize("power", (0, 1))
def test_both_recipes_preserve_the_control(power):
    config = variant(power)
    adapter.check_recipe(config)
    assert config.model.positional_encoding == "rope"
    assert config.model.attention_coefficient_kernel_weight_decay
    assert config.optim.weight_decay == 0.1
    assert config.model.attention_backend == "mosaic"


@pytest.mark.parametrize("section,key,value", (
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("optim", "learning_rate", 0.002),
    ("model", "attention_backend", "blocked"),
    ("data", "batch_size", 128),
    ("train", "gradient_event_capture", False),
))
def test_unrelated_recipe_changes_are_refused(section, key, value):
    config = variant(0)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_prefactor_does_not_silently_reduce_core_degree():
    config = variant(1)
    config.model.attention_degree = 7
    with pytest.raises(ValueError, match="degree-eight SoS core"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("power", (0, 1))
def test_initialization_must_match_recorded_control(monkeypatch, power):
    config = variant(power)
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {
        "parameter_count": adapter.PARAMETER_COUNT, "parameter_sha256": adapter.PARAMETER_SHA})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    assert adapter.check_initialization(config)["all_initial_parameters_match_control"]
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {
        "parameter_count": adapter.PARAMETER_COUNT, "parameter_sha256": "wrong"})
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(config)


def test_acceptance_requires_composed_gradient_tests():
    spec = adapter.experiment_spec(variant(1), entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic"
    assert "scaffolds/transformer/tests/test_mosaic_softplus_sos.py" in spec.gpu_tests
    assert "scaffolds/transformer/tests/test_softplus_sos.py" in spec.cpu_tests
