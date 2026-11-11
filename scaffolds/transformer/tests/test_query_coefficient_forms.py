"""Hermite starts, sum-of-squares query coefficients, and static coefficients under the per-order query-key gate."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import hermite_e
import pytest

from scaffolds.transformer.attention.nonnegative_polynomial import (
    effective_coefficients_numpy, query_square_coefficients, taylor_square_factors,
)
from scaffolds.transformer.attention.unnormalized import initial_coefficients
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.numerics import is_signed_kernel
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity
from scaffolds.transformer.training.state import create_state

TOKENS = jnp.arange(16).reshape(2, 8) % 32
SQUARES = dict(attention_coefficient_parameterization="sum_of_squares", attention_coefficient_scale="none",
               attention_coefficient_amplitude_scale="none")
GATE = dict(positional_encoding="none", attention_forget_gate=True, attention_forget_gate_mode="query_distance",
            attention_forget_gate_conditioning="query_key", attention_forget_gate_per_order=True)
STATIC = dict(attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
              attention_coefficient_kernel_weight_decay=False, attention_coefficient_amplitude_initial=1.0,
              attention_coefficient_projection_initial_std=0.0, attention_coefficient_amplitude_scale="none")


def _config(**changes):
    """A small recipe of the runs' kind: tanh query coefficients stored factorial-scaled, a = 0, random W."""
    model = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                 qk_head_dim=8, value_head_dim=8, intermediate_size=32, positional_encoding="rope",
                 attention_backend="blocked", attention_degree=4, attention_normalization=False,
                 attention_coefficients="learned", attention_coefficient_sharing="per_head",
                 attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                 attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=.25,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_coefficient_kernel_weight_decay=True, attention_output_scaling="head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    if config.model.attention_backend == "mosaic":
        config.train.dtype = "bfloat16"
    validate_config(config)
    return config


def _params(model):
    return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


@pytest.mark.parametrize("degree,expected", ((2, (.5, 1, .5)), (4, (5 / 8, .5, .25, 1 / 6, 1 / 24))))
def test_hermite_start_in_closed_form(degree, expected):
    assert initial_coefficients(degree, "hermite") == pytest.approx(tuple(math.exp(.5) * value for value in expected), rel=1e-14)


@pytest.mark.parametrize("degree", (1, 2, 5, 8, 12))
def test_hermite_start_is_the_truncated_expansion(degree):
    points = np.linspace(-6, 6, 25)
    weights = [1 / math.factorial(order) for order in range(degree + 1)]
    np.testing.assert_allclose(np.polyval(initial_coefficients(degree, "hermite")[::-1], points),
                               math.exp(.5) * hermite_e.hermeval(points, weights), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("degree", (2, 4, 8, 12))
def test_square_coefficients_start_at_taylor(degree):
    actual = np.asarray(query_square_coefficients(jnp.ones((2, 3, degree + 1), jnp.float32), degree))
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, np.broadcast_to([1 / math.factorial(r) for r in range(degree + 1)], actual.shape),
                               rtol=3e-6, atol=0)


@pytest.mark.parametrize("degree", (2, 4, 8, 12))
def test_square_coefficients_expand_the_generated_factors(degree):
    generated = np.random.default_rng(degree).uniform(-1, 3, size=(3, degree + 1)).astype(np.float32)
    factors = np.asarray(taylor_square_factors(degree)) * (1.0 + generated.astype(np.float64)) / 2.0
    np.testing.assert_allclose(query_square_coefficients(jnp.asarray(generated), degree),
                               effective_coefficients_numpy(factors), rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("degree", (4, 8))
def test_a_uniform_step_in_g_moves_every_coefficient_by_that_relative_step(degree):
    """The direct form's first-order effect: g -> 1 + e multiplies c by (1 + e/2)^2 = 1 + e + O(e^2)."""
    start = jnp.ones(degree + 1, jnp.float32)
    value, derivative = jax.jvp(lambda g: query_square_coefficients(g, degree), (start,), (start,))
    np.testing.assert_allclose(derivative, value, rtol=3e-6, atol=0)


def test_a_sum_of_squares_model_starts_at_the_direct_taylor_model():
    square, direct = _config(**SQUARES), _config()
    assert model_parameter_count(square.model) == model_parameter_count(direct.model)
    first, second = build_model(square.model, 42), build_model(direct.model, 42)
    params = _params(first)
    np.testing.assert_array_equal(params[("query_polynomial_bias",)], 1.0)
    np.testing.assert_array_equal(params[("query_polynomial_amplitude",)], 0.0)
    np.testing.assert_allclose(first(TOKENS), second(TOKENS), rtol=2e-3, atol=2e-3)


def test_the_hermite_model_stores_factorial_scaled_starts_and_changes_only_the_bias():
    hermite, taylor = _config(attention_degree=8, attention_coefficient_initialization="hermite"), _config(attention_degree=8)
    first, second = _params(build_model(hermite.model, 42)), _params(build_model(taylor.model, 42))
    stored = [value * math.factorial(order) for order, value in enumerate(initial_coefficients(8, "hermite"))]
    np.testing.assert_allclose(first[("query_polynomial_bias",)], np.broadcast_to(stored, (2, 2, 9)), rtol=1e-6)
    assert first.keys() == second.keys()
    for path in first:
        if path != ("query_polynomial_bias",):
            np.testing.assert_array_equal(first[path], second[path])


def test_static_coefficients_under_the_query_key_gate_start_as_query_coefficients():
    static, query = _config(**GATE, **STATIC), _config(**GATE)
    first, second = build_model(static.model, 42), build_model(query.model, 42)
    params, other = _params(first), _params(second)
    assert ("polynomial_coefficients",) in params and ("query_polynomial_kernel",) not in params
    for path, value in params.items():
        if path[0].startswith("forget_gate_"):
            assert value.shape == (2, 2, 5, *other[path].shape[3:])
            np.testing.assert_array_equal(value, other[path])
    np.testing.assert_allclose(first(TOKENS), second(TOKENS), rtol=1e-5, atol=1e-5)


def test_static_coefficients_receive_gradients_through_the_gate():
    model = build_model(_config(**GATE, **STATIC).model, 42)
    def loss(state):
        merged = nnx.merge(graph, state, rest)
        return jnp.mean(merged(TOKENS) ** 2)
    graph, state, rest = nnx.split(model, nnx.Param, ...)
    grads = {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(jax.grad(loss)(state))}
    for name in ("polynomial_coefficients", "forget_gate_kernel", "forget_gate_key_kernel", "forget_gate_log_rate"):
        assert np.any(grads[(name,)] != 0), name


@pytest.mark.parametrize("changes", (
    dict(GATE, **STATIC, attention_forget_gate_per_order=False),
    dict(attention_coefficient_initialization="hermite", **STATIC),
    dict(SQUARES, attention_degree=3), dict(SQUARES, attention_coefficient_scale="factorial"),
    dict(SQUARES, attention_coefficient_amplitude_scale="factorial"),
    dict(SQUARES, attention_coefficient_initialization="hermite"),
    dict(SQUARES, attention_coefficient_tanh_bias=True),
    dict(attention_coefficient_initialization="hermite", attention_coefficient_conditioning="query_key"),
))
def test_unqualified_recipes_are_rejected(changes):
    with pytest.raises(ValueError):
        _config(**changes)


@pytest.mark.parametrize("changes,decay", ((SQUARES, False), (dict(attention_coefficient_initialization="hermite"), False),
                                          (dict(GATE, **STATIC), True)))
def test_the_mosaic_kernels_accept_the_forms(changes, decay):
    config = _config(hidden_size=128, qk_head_dim=64, value_head_dim=64, max_seq_len=2048, attention_backend="mosaic",
                     **changes)
    assert config.model.mosaic_decay_supported() if decay else config.model.mosaic_supported()


@pytest.mark.parametrize("changes", (dict(GATE, **STATIC), SQUARES, dict(attention_coefficient_initialization="hermite")),
                         ids=("static_gate", "sum_of_squares", "hermite"))
def test_training_diagnostics_and_identity_run(changes):
    config = _config(**changes)
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    state = create_state(config, jax.random.PRNGKey(1))
    batch = jnp.asarray(np.random.default_rng(0).integers(0, 32, (2, 9)), jnp.int32)
    after, *_ = loop.train_step(state, batch, jax.random.PRNGKey(0))
    name = "polynomial_coefficients" if changes.get("attention_coefficient_conditioning") == "static" else "query_polynomial_bias"
    assert np.any(np.asarray(after.params[name][...]) != np.asarray(state.params[name][...]))
    sample = np.asarray(TOKENS[:1, :1].repeat(9, axis=1) + jnp.arange(9), np.int32) % 32
    diagnostics = attention_diagnostics(after, config, sample, {"selection": "test"})
    assert diagnostics["valid"], diagnostics["issues"]
    identity = attention_identity(config)
    if changes.get("attention_forget_gate"):
        assert identity["attention_forget_gate_conditioning"] == "query_key" and identity["attention_forget_gate_per_order"]
        assert identity["forget_gate_parameter_count"] == (2 * 16 + 2) * 2 * 2 * 5
        assert "c_r s_ti^r exp(-(t-i) lambda_tir)" in identity["attention_formula"]
    else:
        assert identity["attention_coefficient_conditioning"] == "query"


def test_expanded_sum_of_squares_kernels_may_round_below_zero():
    """The kernels evaluate the expanded c_r(x_t), so values near a double root of q1^2 + q2^2 round to
    either sign; the normalized static form evaluates q1^2 + q2^2 itself and stays nonnegative."""
    assert is_signed_kernel(_config(**SQUARES).model)
    assert not is_signed_kernel(_config(attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                                        attention_coefficient_kernel_weight_decay=False,
                                        attention_coefficient_amplitude_initial=1.0,
                                        attention_coefficient_projection_initial_std=0.0,
                                        attention_normalization=True, attention_output_scaling="none",
                                        attention_coefficient_parameterization="sum_of_squares",
                                        attention_coefficient_scale="none",
                                        attention_coefficient_amplitude_scale="none").model)
