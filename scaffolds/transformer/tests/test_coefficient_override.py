"""Free query coefficients preserve checkpoint forwards and isolate adaptation."""

from dataclasses import replace

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


@pytest.mark.parametrize("conditioning", ("static", "query_key"))
def test_coefficient_access_rejects_other_conditioning(conditioning):
    config = _config() if conditioning == "query_key" else _config(
        attention_coefficient_conditioning="static",
        attention_coefficient_activation="identity", attention_coefficient_amplitude_scale="none",
        attention_coefficient_projection_initial_std=0, attention_coefficient_tanh_bias=False)
    model = build_model(replace(config, attention_coefficient_conditioning=conditioning), 7)
    with pytest.raises(ValueError, match="query-only"):
        model(_tokens(), return_coefficients=True)
    with pytest.raises(ValueError, match="query-only"):
        model(_tokens(), query_coefficients=jnp.zeros((2, 2, 7, 2, 9)))


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


def test_coefficient_access_rejects_rematerialized_model():
    model = build_model(_config(remat=True), 7)
    with pytest.raises(ValueError, match="model.remat == False"):
        model(_tokens(), return_coefficients=True)
    with pytest.raises(ValueError, match="model.remat == False"):
        model(_tokens(), query_coefficients=jnp.zeros((2, 2, 7, 2, 9)))
