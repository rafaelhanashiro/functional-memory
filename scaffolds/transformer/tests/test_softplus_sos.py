"""Query SoS and z*SoS: actual coefficients, independent factor algebra, model integration and recipes."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients, taylor_square_factors
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity
from scaffolds.transformer.training.numerics import is_signed_kernel
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state


def _config(power=0, **changes):
    model = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                 qk_head_dim=8, value_head_dim=8, intermediate_size=32, positional_encoding="rope",
                 attention_backend="blocked", attention_blocked_implementation="strips",
                 attention_degree=8 + power, attention_normalization=False,
                 attention_coefficients="learned", attention_coefficient_sharing="per_head",
                 attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                 attention_coefficient_parameterization="sum_of_squares", attention_coefficient_input_power=power,
                 attention_score_transform="softplus", attention_coefficient_amplitude_initial=0.0,
                 attention_coefficient_projection_initial_std=.25, attention_coefficient_kernel_weight_decay=True,
                 attention_coefficient_amplitude_scale="none", attention_coefficient_scale="none",
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True, rms_norm_fp32=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.batch_size = config.train.micro_batch_size = 2
    config.data.sequence_length, config.data.synthetic_vocab_size = 9, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    if config.model.attention_backend == "mosaic":
        config.train.dtype = "bfloat16"
    validate_config(config)
    return config


def _direct_squares(generated, score, core_degree, power):
    """Independent value evaluation: two explicit factors, never their convolution."""
    z = jnp.logaddexp(jnp.asarray(0., score.dtype), score)
    factors = jnp.asarray(taylor_square_factors(core_degree), generated.dtype) * (1 + generated) / 2
    half = core_degree // 2
    first = sum(factors[..., r] * z ** r for r in range(half + 1))
    second = sum(factors[..., half + 1 + r] * z ** r for r in range(half))
    return z ** power * (first ** 2 + second ** 2)


def _expanded(generated, score, core_degree, power):
    c = query_square_coefficients(generated, core_degree + power, input_power=power)
    z = jnp.logaddexp(0., score)
    return sum(c[..., r] * z ** r for r in range(core_degree + power + 1))


@pytest.mark.parametrize("core_degree", (2, 6, 8, 12))
@pytest.mark.parametrize("power", (0, 1))
def test_taylor_start_and_structural_zero(core_degree, power):
    generated = jnp.ones((2, 3, core_degree + 1), jnp.float32)
    coefficients = query_square_coefficients(generated, core_degree + power, input_power=power)
    expected = [0.] * power + [1 / math.factorial(r) for r in range(core_degree + 1)]
    assert coefficients.dtype == jnp.float32
    np.testing.assert_allclose(coefficients, np.broadcast_to(expected, coefficients.shape), rtol=3e-6, atol=0)
    if power:
        zero_gradient = jax.grad(lambda g: query_square_coefficients(g, core_degree + 1, input_power=1)[..., 0].sum())
        np.testing.assert_array_equal(zero_gradient(generated), 0.)


@pytest.mark.parametrize("core_degree", (6, 8))
@pytest.mark.parametrize("power", (0, 1))
def test_expansion_values_and_both_gradients_match_independent_squares(core_degree, power):
    rng = np.random.default_rng(53 + core_degree)
    generated = jnp.asarray(rng.uniform(-1.5, 3., (6, core_degree + 1)), jnp.float32)
    score = jnp.asarray([-20., -3., 0., .5, 3., 8.], jnp.float32)
    actual, pullback = jax.vjp(lambda g, s: _expanded(g, s, core_degree, power), generated, score)
    expected, reference_pullback = jax.vjp(lambda g, s: _direct_squares(g, s, core_degree, power), generated, score)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)
    for got, want in zip(pullback(jnp.ones_like(actual)), reference_pullback(jnp.ones_like(expected))):
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-7)
    assert np.all(np.asarray(expected) >= 0)


def test_negative_score_baseline_and_zero_tail_are_distinct():
    g = jnp.ones(9, jnp.float32)
    s = jnp.array(-20., jnp.float32)
    assert float(_expanded(g, s, 8, 0)) == pytest.approx(1., rel=1e-6)
    assert float(_expanded(g, s, 8, 1) / jnp.exp(s)) == pytest.approx(1., rel=1e-6)


def test_prefactor_preserves_every_initial_parameter_and_generator_count():
    baseline, prefactor = _config(), _config(1)
    assert model_parameter_count(baseline.model) == model_parameter_count(prefactor.model)
    params = []
    for config in (baseline, prefactor):
        model = build_model(config.model, 42)
        params.append({path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))})
    assert params[0].keys() == params[1].keys()
    for path in params[0]:
        np.testing.assert_array_equal(params[0][path], params[1][path])
    assert params[0][("query_polynomial_bias",)].shape == (2, 2, 9)
    np.testing.assert_array_equal(params[0][("query_polynomial_bias",)], 1.)
    np.testing.assert_array_equal(params[0][("query_polynomial_amplitude",)], 0.)


@pytest.mark.parametrize("power", (0, 1))
def test_w_decays_but_factor_amplitude_and_bias_do_not(power):
    specs = attention_parameter_specs(_config(power).model)
    selected = {spec.name: spec for spec in specs if spec.name.startswith("query_polynomial_")}
    assert set(selected) == {"query_polynomial_kernel", "query_polynomial_amplitude", "query_polynomial_bias"}
    params = {name: jnp.ones((2, 3)) for name in selected}
    mask = attention_weight_decay_mask(params, excluded_names={s.name for s in specs if not s.weight_decay})
    assert mask == {"query_polynomial_kernel": True, "query_polynomial_amplitude": False, "query_polynomial_bias": False}


@pytest.mark.parametrize("power", (0, 1))
def test_training_and_diagnostics_use_actual_expanded_coefficients(power):
    config = _config(power)
    state = create_state(config, jax.random.PRNGKey(1))
    batch = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    after, *_ = loop.train_step(state, batch, jax.random.PRNGKey(0))
    assert np.any(np.asarray(after.params["query_polynomial_bias"][...]) != np.asarray(state.params["query_polynomial_bias"][...]))
    report = attention_diagnostics(after, config, np.asarray(batch[:1]), {"selection": "test"})
    assert report["valid"], report["issues"]
    coefficients = report["coefficients"]
    assert np.asarray(coefficients["effective_summary"]["mean"]).shape == (2, 2, 9 + power)
    assert np.asarray(coefficients["projection_bias"]).shape == (2, 2, 9)
    assert coefficients["core_degree"] == 8
    if power:
        for statistic in ("mean", "std", "min", "max"):
            np.testing.assert_array_equal(np.asarray(coefficients["effective_summary"][statistic])[..., 0], 0.)
    identity = attention_identity(config)
    assert identity["attention_coefficient_count"] == 9 * 18 * 2 * 2
    assert identity["attention_coefficient_core_degree"] == 8
    assert identity["attention_coefficient_input_power"] == power
    assert "F_j(x_t) = f_j (1 + g_j(x_t))/2" in identity["attention_formula"]
    assert "then per-head RMSNorm" in identity["attention_formula"]
    assert is_signed_kernel(config.model)  # Expanded FP32 polynomial can round slightly negative near roots.


@pytest.mark.parametrize("power", (0, 1))
def test_mosaic_recipe_support_uses_total_degree(power):
    config = _config(power, hidden_size=128, qk_head_dim=64, value_head_dim=64, max_seq_len=2048,
                     attention_backend="mosaic")
    assert config.model.mosaic_score_transform_supported()
    assert config.model.attention_degree == 8 + power


@pytest.mark.parametrize("changes", (
    dict(attention_coefficient_input_power=True), dict(attention_coefficient_input_power=2),
    dict(attention_coefficient_input_power=1),  # total 8 would leave odd core 7
    dict(attention_degree=9),
    dict(attention_coefficient_scale="factorial"), dict(attention_coefficient_amplitude_scale="factorial"),
    dict(attention_coefficient_constant_initial=0.), dict(attention_coefficient_conditioning="query_key"),
    dict(attention_coefficient_tanh_bias=True), dict(attention_score_transform="abs"),
    dict(attention_degree=9, attention_coefficient_input_power=1, attention_score_transform="identity"),
    dict(attention_degree=9, attention_coefficient_input_power=1, attention_coefficient_parameterization="direct"),
))
def test_incompatible_recipes_rejected(changes):
    with pytest.raises(ValueError):
        _config(**changes)


def test_existing_query_sos_without_transform_is_preserved():
    config = _config(attention_score_transform="identity")
    assert config.model.attention_coefficient_input_power == 0
    config.model = replace(config.model, hidden_size=128, qk_head_dim=64, value_head_dim=64, max_seq_len=2048,
                           attention_backend="mosaic")
    assert config.model.mosaic_supported()


def test_factor_indices_cannot_use_polynomial_order_release():
    config = _config(attention_score_transform="identity")
    config.optim.coefficient_release_schedule = "all"
    config.optim.coefficient_release_step = 1
    with pytest.raises(ValueError, match="coefficient_release_schedule"):
        validate_config(config)
