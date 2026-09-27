"""Recipe validation and compatibility for immutable polynomial coefficients."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from scaffolds.transformer.attention.parameters import attention_parameter_count, attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal


COEFFICIENTS = (0.625, 0.5, 0.25, 1.0 / 6.0, 1.0 / 24.0)


@pytest.mark.parametrize("normalized", (False, True))
def test_fixed_polynomial_yaml_and_json_roundtrip(tmp_path, normalized):
    config = Config(model=ModelConfig(
        attention_degree=4, attention_normalization=normalized,
        attention_fixed_coefficients=list(COEFFICIENTS), qk_norm=False,
    ))
    assert config.model.attention_fixed_coefficients == COEFFICIENTS
    assert isinstance(config.model.attention_fixed_coefficients, tuple)
    encoded = config_to_dict(config)
    assert encoded["model"]["attention_fixed_coefficients"] == list(COEFFICIENTS)
    yaml_path, json_path = tmp_path / "recipe.yaml", tmp_path / "recipe.json"
    save_config(config, yaml_path)
    json_path.write_text(json.dumps(encoded, allow_nan=False))
    for path in (yaml_path, json_path):
        restored = load_config(path)
        assert restored.model.attention_fixed_coefficients == COEFFICIENTS
        assert restored.model.attention_coefficients == "fixed"
        assert config_to_dict(restored) == encoded


def test_legacy_absence_equals_none_but_not_explicit_coefficients():
    current = config_to_dict(Config(model=ModelConfig(attention_degree=4, qk_norm=False)))
    legacy = deepcopy(current)
    del legacy["model"]["attention_fixed_coefficients"]
    before = deepcopy(legacy)
    assert config_dicts_equal(legacy, current)
    assert legacy == before  # Comparison must not rewrite retained recipes.
    explicit = deepcopy(current)
    explicit["model"]["attention_fixed_coefficients"] = list(COEFFICIENTS)
    assert not config_dicts_equal(legacy, explicit)


def test_fixed_polynomial_config_equality_normalizes_container_but_checks_every_value():
    first = config_to_dict(Config(model=ModelConfig(attention_degree=4, attention_fixed_coefficients=COEFFICIENTS)))
    second = deepcopy(first)
    second["model"]["attention_fixed_coefficients"] = COEFFICIENTS
    assert config_dicts_equal(first, second)
    second["model"]["attention_fixed_coefficients"] = (*COEFFICIENTS[:-1], COEFFICIENTS[-1] + 1e-12)
    assert not config_dicts_equal(first, second)


@pytest.mark.parametrize("coefficients", (
    [], [1.0], [1.0, 1.0, 0.5, 0.0], "1,1,0.5", {0: 1.0, 1: 1.0, 2: 0.5},
    [1.0, True, 0.5], [1.0, "1", 0.5], [1.0, None, 0.5],
    [1.0, float("nan"), 0.5], [1.0, float("inf"), 0.5], [1.0, float("-inf"), 0.5],
))
def test_malformed_fixed_coefficients_are_rejected(coefficients):
    with pytest.raises(ValueError, match="attention_fixed_coefficients"):
        ModelConfig(attention_degree=2, attention_fixed_coefficients=coefficients)


@pytest.mark.parametrize("override", (
    {"attention_degree": None}, {"attention_degree": 0}, {"attention_degree": -1},
    {"attention_degree": True}, {"attention_degree": 2.0},
    {"attention_coefficients": "learned"}, {"attention_coefficient_parameterization": "sum_of_squares"},
    {"quadratic_mode": "positive"}, {"quadratic_mode": "unrestricted"},
    {"attention_mechanism": "gla", "attention_degree": None, "qk_norm": False},
    {"attention_backend": "manual"}, {"attention_backend": "xla"}, {"attention_backend": "cudnn"},
))
def test_conflicting_fixed_coefficient_settings_are_rejected(override):
    arguments = {"attention_degree": 2, "attention_fixed_coefficients": (0.5, 1.0, 0.5), **override}
    with pytest.raises(ValueError):
        ModelConfig(**arguments)


def test_mutating_a_recipe_cannot_bypass_fixed_coefficient_validation():
    config = Config(model=ModelConfig(attention_degree=2, attention_fixed_coefficients=(0.5, 1.0, 0.5)))
    config.model.attention_coefficients = "learned"
    with pytest.raises(ValueError, match="fixed/direct"):
        validate_config(config)


@pytest.mark.parametrize("sharing", ("global", "per_head"))
@pytest.mark.parametrize("qk_norm", (False, True))
def test_fixed_coefficients_never_declare_trainable_coefficient_parameters(sharing, qk_norm):
    model = ModelConfig(attention_degree=4, attention_fixed_coefficients=COEFFICIENTS,
                        attention_coefficient_sharing=sharing, qk_norm=qk_norm)
    control = replace(model, attention_fixed_coefficients=None)
    specs = attention_parameter_specs(model)
    assert specs == attention_parameter_specs(control)
    assert attention_parameter_count(model) == attention_parameter_count(control)
    assert all(spec.name not in ("polynomial_coefficients", "polynomial_square_factors", "quadratic_coefficients")
               for spec in specs)
