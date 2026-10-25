"""Factorial scaling of the input-independent polynomial coefficients (static c_r, query bias b_r)."""

from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.regularization import coefficient_penalty_sum
from scaffolds.transformer.training.state import create_state

TOKENS = jnp.arange(26).reshape(2, 13) % 16


def _model(degree=3, conditioning="static", **updates):
    settings = dict(vocab_size=16, max_seq_len=13, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                    qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=degree,
                    attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_backend="blocked",
                    initialization="transformer_reference", qk_norm=True, attention_output_scaling="head_rms_norm")
    if conditioning != "static":
        settings.update(attention_coefficient_conditioning=conditioning, attention_coefficient_activation="tanh",
                        attention_coefficient_amplitude_initial=0.0,
                        attention_coefficient_projection_initial_std=0.3)
    return ModelConfig(**{**settings, **updates})


def _factorials(degree):
    return np.asarray([math.factorial(order) for order in range(degree + 1)], np.float32)


def _stored(conditioning):
    return "polynomial_coefficients" if conditioning == "static" else "query_polynomial_bias"


@pytest.mark.parametrize("conditioning", ["static", "query", "query_key"])
@pytest.mark.parametrize("degree", [2, 8, 12])
def test_stored_coefficients_start_at_one_and_represent_the_unscaled_model(conditioning, degree):
    plain = _model(degree, conditioning)
    scaled = replace(plain, attention_coefficient_scale="factorial")
    plain_model, scaled_model = build_model(plain, 13), build_model(scaled, 13)
    np.testing.assert_array_equal(getattr(scaled_model, _stored(conditioning))[...], 1.0)
    assert model_parameter_count(scaled) == model_parameter_count(plain)
    np.testing.assert_allclose(scaled_model(TOKENS), plain_model(TOKENS), rtol=2e-5, atol=2e-6)
    # Any stored value represents its value divided by r!.
    values = jnp.asarray(np.random.default_rng(degree).normal(size=getattr(plain_model, _stored(conditioning)).shape)
                         / _factorials(degree), jnp.float32)
    getattr(plain_model, _stored(conditioning))[...] = values
    getattr(scaled_model, _stored(conditioning))[...] = values * _factorials(degree)
    if conditioning != "static":
        for model in (plain_model, scaled_model):
            model.query_polynomial_amplitude[...] = 0.1
            if conditioning == "query_key":
                model.key_polynomial_amplitude[...] = 0.1
    np.testing.assert_allclose(scaled_model(TOKENS), plain_model(TOKENS), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("conditioning", ["static", "query"])
def test_stored_gradient_is_the_coefficient_gradient_divided_by_r_factorial(conditioning):
    degree = 4
    plain, scaled = _model(degree, conditioning), _model(degree, conditioning, attention_coefficient_scale="factorial")
    gradients = []
    for config in (plain, scaled):
        model = build_model(config, 13)
        if conditioning == "query":
            model.query_polynomial_amplitude[...] = 0.1
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss(params):
            return jnp.mean(nnx.merge(graphdef, params, rest)(TOKENS) ** 2)

        gradients.append(np.asarray(jax.grad(loss)(params)[_stored(conditioning)][...]))
    np.testing.assert_allclose(gradients[1], gradients[0] / _factorials(degree), rtol=2e-4, atol=1e-9)
    assert np.abs(gradients[0][..., -1]).max() > 0


@pytest.mark.parametrize("conditioning", ["query", "query_key"])
@pytest.mark.parametrize("degree", [8, 12])
def test_entire_coefficient_scaling_preserves_values_and_all_parameter_gradients(conditioning, degree):
    """(b + a*tanh(Wx))/r! scales the whole coefficient, including its key interaction.

    Compare two storage parameterizations of identical nonconstant coefficients;
    this catches scaling just b or a, or applying a second factorial to the key factor.
    """
    plain = _model(degree, conditioning)
    both = replace(plain, attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial")
    plain_model, scaled_model = build_model(plain, 13), build_model(both, 13)
    amplitude = jnp.asarray(np.random.default_rng(1).normal(size=plain_model.query_polynomial_amplitude.shape) * 0.05,
                            jnp.float32)
    plain_model.query_polynomial_amplitude[...] = amplitude / _factorials(degree)
    scaled_model.query_polynomial_amplitude[...] = amplitude
    if conditioning == "query_key":
        # The key factor is dimensionless; it is multiplied by the already scaled
        # query coefficient, and starts at one when its amplitude starts at zero.
        for model in (plain_model, scaled_model):
            model.key_polynomial_amplitude[...] = 0.15
    np.testing.assert_allclose(scaled_model(TOKENS), plain_model(TOKENS), rtol=2e-4, atol=2e-5)
    gradients = []
    for model in (plain_model, scaled_model):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss(params):
            return jnp.mean(nnx.merge(graphdef, params, rest)(TOKENS) ** 2)

        gradients.append(jax.grad(loss)(params))
    for name in ("query_polynomial_bias", "query_polynomial_amplitude"):
        expected = np.asarray(gradients[0][name][...]) / _factorials(degree)
        np.testing.assert_allclose(gradients[1][name][...], expected, rtol=3e-4, atol=1e-10)
        assert np.abs(expected[..., -1]).max() > 0
    unchanged = ["query_polynomial_kernel"]
    if conditioning == "query_key":
        unchanged += ["key_polynomial_kernel", "key_polynomial_amplitude"]
    for name in unchanged:
        np.testing.assert_allclose(gradients[1][name][...], gradients[0][name][...], rtol=3e-4, atol=1e-10)


@pytest.mark.parametrize("updates", [
    {"attention_coefficient_scale": "linear"},
    {"attention_coefficient_scale": "factorial", "attention_coefficients": "fixed"},
    {"attention_coefficient_scale": "factorial", "attention_normalization": True},
    {"attention_coefficient_scale": "factorial", "attention_coefficient_parameterization": "sum_of_squares",
     "attention_normalization": True, "attention_degree": 2},
])
def test_rejects_coefficient_scale_outside_learned_direct_unnormalized(updates):
    with pytest.raises(ValueError):
        _model(**updates)


def test_legacy_recipes_omit_the_field_and_the_identity_names_it():
    config = Config(model=_model(3))
    current, legacy = config_to_dict(config), config_to_dict(config)
    del legacy["model"]["attention_coefficient_scale"]
    assert config_dicts_equal(current, legacy)
    scaled = Config(model=replace(config.model, attention_coefficient_scale="factorial"))
    assert not config_dicts_equal(current, config_to_dict(scaled))
    assert "attention_coefficient_scale" not in attention_identity(config)
    identity = attention_identity(scaled)
    assert identity["attention_coefficient_scale"] == "factorial"
    assert identity["attention_coefficient_count"] == attention_identity(config)["attention_coefficient_count"]


def _small_config(model):
    config = Config(model=model)
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 14, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 3, 0
    return config


def test_diagnostics_report_represented_static_coefficients_and_query_bias():
    static = _small_config(_model(3, attention_coefficient_scale="factorial"))
    state = create_state(static, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(static)
    report = attention_diagnostics(state, static, sample, provenance)
    assert report["valid"], report["issues"]
    np.testing.assert_allclose(report["coefficients"]["raw"], np.ones((2, 2, 4)))
    np.testing.assert_allclose(report["coefficients"]["effective"], np.broadcast_to(1 / _factorials(3), (2, 2, 4)))
    query = _small_config(_model(3, "query", attention_coefficient_scale="factorial"))
    state = create_state(query, jax.random.PRNGKey(0))
    report = attention_diagnostics(state, query, sample, provenance)
    assert report["valid"], report["issues"]
    coefficients = report["coefficients"]
    np.testing.assert_allclose(coefficients["projection_bias"], np.ones((2, 2, 4)))
    assert coefficients["projection_bias_scale"] == "1/r!"
    np.testing.assert_allclose(coefficients["effective_projection_bias"], np.broadcast_to(1 / _factorials(3), (2, 2, 4)))
    # a = 0: the actual coefficients equal the represented bias.
    np.testing.assert_allclose(coefficients["effective_summary"]["mean"], coefficients["effective_projection_bias"],
                               rtol=1e-6)


def test_acceptance_runs_these_checks_for_scaled_recipes():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (False, True):
        scaled = config_to_dict(Config(model=_model(3, attention_coefficient_scale="factorial")))
        plain = config_to_dict(Config(model=_model(3)))
        assert acceptance.COEFFICIENT_SCALE_TESTS in acceptance.test_paths(spec, scaled, resolved_backend="mosaic", cpu=cpu)
        assert acceptance.COEFFICIENT_SCALE_TESTS not in acceptance.test_paths(spec, plain, resolved_backend="mosaic",
                                                                              cpu=cpu)


def test_penalty_and_acceptance_controls_respect_the_scale():
    model = _model(3, attention_coefficient_scale="factorial")
    with pytest.raises(ValueError, match="unscaled"):
        coefficient_penalty_sum({}, model)
    for conditioning in ("static", "query"):
        config = Config(model=_model(3, conditioning, attention_coefficient_scale="factorial"))
        facts = acceptance.matches_softmax_backbone_initialization(config)
        assert facts["mechanism_parameters"][_stored(conditioning)]["initial"] == [1.0] * 4
