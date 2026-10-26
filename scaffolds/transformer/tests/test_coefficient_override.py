"""Free query coefficients preserve checkpoint forwards and isolate adaptation."""

from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.models import build_model


def _config(**updates):
    settings = dict(
        vocab_size=16, max_seq_len=7, num_layers=2, hidden_size=8, num_heads=2,
        num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
        attention_degree=8, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
        attention_coefficient_activation="tanh", attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_tanh_bias=True,
        attention_coefficient_projection_initial_std=0.05, attention_backend="blocked",
        initialization="transformer_reference", qk_norm=True,
        attention_output_scaling="head_rms_norm",
    )
    return ModelConfig(**{**settings, **updates})


def _tokens():
    return jnp.arange(14).reshape(2, 7) % 16


def _static_config(**updates):
    return _config(**{
        "attention_coefficient_conditioning": "static",
        "attention_coefficient_activation": "identity",
        "attention_coefficient_amplitude_scale": "none",
        "attention_coefficient_projection_initial_std": 0,
        "attention_coefficient_tanh_bias": False,
        "qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
        "rms_norm_fp32": True, "initialization": "transformer_truncated_reference",
        **updates,
    })


def _loss(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1).mean()


def test_capture_replays_the_original_forward_and_can_be_jitted():
    model = build_model(_config(), 7)
    tokens = _tokens()
    expected = model(tokens)
    captured_logits, coefficients = model(tokens, return_coefficients=True)
    assert coefficients.shape == (2, 2, 7, 2, 9)
    assert coefficients.dtype == jnp.float32
    np.testing.assert_array_equal(captured_logits, expected)
    replay, captured_again = model(tokens, query_coefficients=coefficients, return_coefficients=True)
    np.testing.assert_array_equal(replay, expected)
    np.testing.assert_array_equal(captured_again, coefficients)

    graphdef, state = nnx.split(model)

    @jax.jit
    def replay_jit(state, coefficients):
        return nnx.merge(graphdef, state)(tokens, query_coefficients=coefficients)

    @jax.jit
    def capture_jit(state):
        return nnx.merge(graphdef, state)(tokens, return_coefficients=True)

    compiled_logits, compiled_coefficients = capture_jit(state)
    np.testing.assert_allclose(replay_jit(state, compiled_coefficients), compiled_logits, rtol=2e-5, atol=2e-6)


def test_effective_coefficients_bypass_the_generator_and_are_independent_per_example():
    model = build_model(_config(), 7)
    tokens = _tokens()
    expected, coefficients = model(tokens, return_coefficients=True)
    model.query_polynomial_bias[...] += 2
    model.query_polynomial_amplitude[...] *= -3
    model.query_polynomial_kernel[...] += 0.5
    model.query_polynomial_tanh_bias[...] += 1
    np.testing.assert_array_equal(model(tokens, query_coefficients=coefficients), expected)
    assert not np.allclose(model(tokens), expected)

    changed = coefficients.at[:, 0, :, :, 1].add(0.2)
    actual = model(tokens, query_coefficients=changed)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert not np.allclose(actual[0], expected[0])


def test_only_coefficient_optimization_changes_loss_and_preserves_every_model_parameter():
    model = build_model(_config(), 7)
    tokens = _tokens()
    _, coefficients = model(tokens, return_coefficients=True)
    before = jax.tree.map(lambda x: np.array(x, copy=True), nnx.state(model, nnx.Param))
    graphdef, state = nnx.split(model)

    def objective(coefficients):
        logits = nnx.merge(graphdef, state)(tokens, query_coefficients=coefficients)
        return _loss(logits, (tokens + 1) % 16)

    loss, gradients = jax.jit(jax.value_and_grad(objective))(coefficients)
    assert np.isfinite(gradients).all()
    assert all(float(jnp.linalg.norm(layer)) > 0 for layer in gradients)
    direction = gradients / jnp.linalg.norm(gradients)
    # A change in an early layer propagates through the frozen later layers.
    early_direction = direction.at[1:].set(0)
    eps = 2e-3
    finite_difference = (objective(coefficients + eps * early_direction)
                         - objective(coefficients - eps * early_direction)) / (2 * eps)
    expected = jnp.sum(gradients * early_direction)
    np.testing.assert_allclose(finite_difference, expected, rtol=0.03, atol=2e-5)
    assert float(objective(coefficients - 1e-3 * direction)) < float(loss)
    after = nnx.state(model, nnx.Param)
    for original, current in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(current, original)


def test_shape_errors_and_ambiguous_returns_are_rejected():
    model = build_model(_config(), 7)
    tokens = _tokens()
    _, coefficients = model(tokens, return_coefficients=True)
    with pytest.raises(ValueError, match="must have shape"):
        model(tokens, query_coefficients=coefficients[:, :1])
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(tokens, return_coefficients=True, return_diagnostics=True)
    attention = model.layers_0.attention
    x = jnp.ones((2, 7, 8))
    with pytest.raises(ValueError, match="must have shape"):
        attention(x, query_coefficients=coefficients[0, :1])
    with pytest.raises(ValueError, match="mutually exclusive"):
        attention(x, return_coefficients=True, return_diagnostics=True)


def test_coefficient_access_rejects_query_key_conditioning():
    model = build_model(replace(_config(), attention_coefficient_conditioning="query_key"), 7)
    with pytest.raises(ValueError, match="static or query"):
        model(_tokens(), return_coefficients=True)
    with pytest.raises(ValueError, match="static or query"):
        model(_tokens(), query_coefficients=jnp.zeros((2, 2, 7, 2, 9)))


def test_static_capture_preserves_forward_and_replays_effective_coefficients():
    config = _static_config(num_layers=3, num_heads=14, num_kv_heads=14,
                            hidden_size=28, intermediate_size=40)
    model = build_model(config, 7)
    # Include signed, non-Taylor learned values so duplicate factorial scaling or
    # accidental tanh application cannot pass merely by reproducing initialization.
    stored = jnp.linspace(-0.3, 1.7, model.polynomial_coefficients.size).reshape(
        model.polynomial_coefficients.shape)
    model.polynomial_coefficients[...] = stored
    tokens = _tokens()
    expected = model(tokens)
    captured, coefficients = model(tokens, return_coefficients=True)
    np.testing.assert_array_equal(captured, expected)
    represented = stored / jnp.asarray([math.factorial(r) for r in range(9)], jnp.float32)
    np.testing.assert_array_equal(coefficients, jnp.broadcast_to(
        represented[:, None, None, :, :], (3, 2, 7, 14, 9)))
    np.testing.assert_allclose(model(tokens, query_coefficients=coefficients),
                               expected, rtol=2e-5, atol=2e-6)
    graph, state = nnx.split(model)
    compiled = jax.jit(lambda state, table: nnx.merge(graph, state)(tokens, query_coefficients=table))
    np.testing.assert_allclose(compiled(state, coefficients), expected, rtol=2e-5, atol=2e-6)
    model.polynomial_coefficients[...] += 0.5
    np.testing.assert_allclose(model(tokens, query_coefficients=coefficients),
                               expected, rtol=2e-5, atol=2e-6)
    assert not np.allclose(model(tokens), expected)


def test_static_override_has_independent_query_gradients_and_frozen_backbone():
    model = build_model(_static_config(), 7)
    tokens = _tokens()
    expected, coefficients = model(tokens, return_coefficients=True)
    before = jax.tree.map(lambda x: np.array(x, copy=True), nnx.state(model, nnx.Param))
    graph, state = nnx.split(model)

    def objective(table):
        return _loss(nnx.merge(graph, state)(tokens, query_coefficients=table), (tokens + 1) % 16)

    loss, gradient = jax.jit(jax.value_and_grad(objective))(coefficients)
    assert np.isfinite(gradient).all()
    assert all(float(jnp.linalg.norm(layer)) > 0 for layer in gradient)
    direction = gradient / jnp.linalg.norm(gradient)
    eps = 2e-3
    early = direction.at[1:].set(0)
    finite_difference = (objective(coefficients + eps * early)
                         - objective(coefficients - eps * early)) / (2 * eps)
    np.testing.assert_allclose(finite_difference, jnp.sum(gradient * early), rtol=0.03, atol=2e-5)
    assert float(objective(coefficients - 1e-3 * direction)) < float(loss)
    changed = coefficients.at[:, 0, -1, :, 1].add(0.2)
    actual = model(tokens, query_coefficients=changed)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual[0, :-1], expected[0, :-1], rtol=2e-5, atol=2e-6)
    assert not np.allclose(actual[0, -1], expected[0, -1])
    for original, current in zip(jax.tree.leaves(before),
                                 jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True):
        np.testing.assert_array_equal(current, original)


def test_default_static_model_diagnostics_remain_available_with_remat():
    config = _config(attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                     attention_coefficient_amplitude_scale="none", attention_coefficient_projection_initial_std=0,
                     attention_coefficient_tanh_bias=False)
    model = build_model(config, 7)
    rematerialized = build_model(replace(config, remat=True), 7)
    expected = model(_tokens(), return_diagnostics=True)
    actual = rematerialized(_tokens(), return_diagnostics=True)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("fixed_coefficients", [None, (1., .8, .4, -.1, .03)])
def test_fixed_coefficients_capture_preserves_head_scales_and_allows_free_adaptation(fixed_coefficients):
    config = _static_config(attention_degree=4, attention_coefficients="fixed",
                            attention_coefficient_scale="none", qk_norm_head_scale=True,
                            attention_fixed_coefficients=fixed_coefficients)
    model = build_model(config, 7)
    model.qk_norm_head_log_scale[...] = jnp.asarray([[[.3], [-.2]], [[.1], [.4]]])
    tokens = _tokens()
    before = jax.tree.map(lambda x: np.array(x, copy=True), nnx.state(model, nnx.Param))
    expected = model(tokens)
    captured, coefficients = model(tokens, return_coefficients=True)
    np.testing.assert_array_equal(captured, expected)
    represented = jnp.asarray(fixed_coefficients if fixed_coefficients is not None
                              else [1 / math.factorial(order) for order in range(5)], jnp.float32)
    np.testing.assert_array_equal(coefficients, jnp.broadcast_to(represented, (2, 2, 7, 2, 5)))
    np.testing.assert_allclose(model(tokens, query_coefficients=coefficients), expected,
                               rtol=2e-5, atol=2e-6)
    graph, state = nnx.split(model)
    objective = lambda table: _loss(nnx.merge(graph, state)(tokens, query_coefficients=table),
                                    (tokens + 1) % 16)
    loss, gradient = jax.jit(jax.value_and_grad(objective))(coefficients)
    assert np.isfinite(gradient).all() and float(jnp.linalg.norm(gradient)) > 0
    assert float(objective(coefficients - .001 * gradient / jnp.linalg.norm(gradient))) < float(loss)
    for original, current in zip(jax.tree.leaves(before),
                                 jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True):
        np.testing.assert_array_equal(current, original)


def test_coefficient_access_rejects_rematerialized_model():
    model = build_model(_config(remat=True), 7)
    with pytest.raises(ValueError, match="model.remat == False"):
        model(_tokens(), return_coefficients=True)
    with pytest.raises(ValueError, match="model.remat == False"):
        model(_tokens(), query_coefficients=jnp.zeros((2, 2, 7, 2, 9)))
