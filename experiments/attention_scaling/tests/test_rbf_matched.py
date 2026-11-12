"""Reject scientific recipe drift before spending a production allocation."""
import pytest

from experiments.attention_scaling.rbf_matched import check_config, recipe, reference_config, MODEL_CHANGES
from scaffolds.transformer.configs import config_to_dict, validate_config


@pytest.mark.parametrize("mode", ("fixed", "static", "query"))
def test_only_rbf_fields_differ_from_executed_reference(mode):
    config = recipe(mode)
    validate_config(config)
    check_config(config)
    actual, reference = config_to_dict(config), config_to_dict(reference_config())
    assert actual["data"] == reference["data"]
    assert actual["optim"] == reference["optim"]
    assert actual["train"] == reference["train"]
    changed = {name for name in actual["model"] if actual["model"][name] != reference["model"][name]}
    assert changed <= MODEL_CHANGES
    assert config.optim.total_steps == 7570 and config.optim.warmup_steps == 455
    assert config.train.seed == 42 and config.train.eval_batches == 19
    assert config.model.num_layers == 15 and config.model.hidden_size == 896


@pytest.mark.parametrize(("section", "name", "value"), [
    ("model", "num_layers", 10), ("model", "qk_norm_type", "cosine_log_scale"),
    ("model", "initialization", "transformer_reference"), ("model", "rms_norm_fp32", False),
    ("optim", "learning_rate", 0.0003), ("optim", "warmup_steps", 160),
    ("optim", "total_steps", 7683), ("train", "seed", 0), ("train", "eval_batches", 7),
    ("model", "attention_coefficient_scale", "none"),
])
def test_preflight_rejects_recipe_drift(section, name, value):
    config = recipe("query")
    setattr(getattr(config, section), name, value)
    with pytest.raises(ValueError, match=f"{section}.{name}"):
        check_config(config)


def test_query_generator_matches_rafaels_degree_eight_parameters_and_random_draw():
    import jax
    import numpy as np
    from dataclasses import replace
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    ours = recipe("query").model
    # The executed degree-eight reference used this factorial-scaled generator.
    # Keep the audit independent of the reference study's directory and run retention.
    reference = replace(
        reference_config().model, attention_basis=None, attention_degree=8,
        attention_backend="mosaic", attention_coefficients="learned",
        attention_coefficient_initialization="taylor", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_projection_initial_std=1 / (896 ** 0.5),
    )
    names = ("attention_coefficient_activation", "attention_coefficient_conditioning",
             "attention_coefficient_sharing", "attention_coefficient_amplitude_initial",
             "attention_coefficient_projection_initial_std", "attention_coefficient_tanh_bias")
    assert all(getattr(ours, name) == getattr(reference, name) for name in names)
    assert reference.attention_coefficient_scale == reference.attention_coefficient_amplitude_scale == "factorial"
    key = mechanism_key(jax.random.PRNGKey(42))
    a = initial_attention_parameter_values(attention_parameter_specs(ours), num_layers=15, num_heads=14, key=key)
    b = initial_attention_parameter_values(attention_parameter_specs(reference), num_layers=15, num_heads=14, key=key)
    for suffix in ("bias", "amplitude", "kernel"):
        np.testing.assert_array_equal(a[f"query_basis_{suffix}"], b[f"query_polynomial_{suffix}"])


@pytest.mark.parametrize("mode", ("static", "query"))
def test_recorded_coefficient_formula_uses_fitted_weights(mode):
    from scaffolds.transformer.training.diagnostics import attention_identity
    facts = attention_identity(recipe(mode))
    assert facts["attention_coefficient_scale"] == "projection"
    assert "frozen fitted basis weights" in facts["attention_coefficient_scale_formula"]
    assert "s_j *" in facts["attention_formula"]
    if mode == "query":
        assert "s_j * b_j + s_j * a_j * tanh(x_t W_j)" in facts["attention_formula"]
