"""Recipe and initialization controls for the softplus-SoS experiments."""

import pytest

from experiments.transformer_pp import softplus_sos_acceptance as adapter


def variant(power, core_degree=8):
    config = adapter.control_config(core_degree)
    config.model.attention_degree = core_degree + power
    config.model.attention_coefficient_input_power = power
    config.model.attention_score_transform = "softplus"
    for key, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, key, value)
    return config


@pytest.mark.parametrize("power", (0, 1))
@pytest.mark.parametrize("core_degree", (8, 12))
def test_both_recipes_preserve_the_control(power, core_degree):
    config = variant(power, core_degree)
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
    ("model", "attention_forget_gate", True),
    ("model", "positional_encoding", "none"),
))
@pytest.mark.parametrize("core_degree", (8, 12))
def test_unrelated_recipe_changes_are_refused(section, key, value, core_degree):
    config = variant(0, core_degree)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("core_degree", (8, 12))
def test_prefactor_does_not_silently_reduce_core_degree(core_degree):
    config = variant(1, core_degree)
    config.model.attention_degree = core_degree - 1
    with pytest.raises(ValueError, match="degree-eight or degree-twelve SoS core"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("power", (0, 1))
@pytest.mark.parametrize("core_degree", (8, 12))
def test_initialization_must_match_recorded_control(monkeypatch, power, core_degree):
    config = variant(power, core_degree)
    control = adapter.CONTROLS[core_degree]
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {
        "parameter_count": control["parameter_count"], "parameter_sha256": control["parameter_sha256"]})
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    facts = adapter.check_initialization(config)
    assert facts["all_initial_parameters_match_control"]
    assert facts["control_run_id"] == control["run_id"]
    assert facts["sos_core_degree"] == core_degree
    assert facts["initial_kernel"] == ("z*" if power else "") + f"T_{core_degree}(z)"
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {
        "parameter_count": control["parameter_count"], "parameter_sha256": "wrong"})
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(config)


@pytest.mark.parametrize("core_degree", (8, 12))
def test_acceptance_requires_composed_gradient_tests(core_degree):
    spec = adapter.experiment_spec(variant(1, core_degree), entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic"
    assert "scaffolds/transformer/tests/test_mosaic_score_transform.py" in spec.gpu_tests
    assert "scaffolds/transformer/tests/test_mosaic_softplus_sos.py" in spec.gpu_tests
    assert "scaffolds/transformer/tests/test_softplus_sos.py" in spec.cpu_tests
