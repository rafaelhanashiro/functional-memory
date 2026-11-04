"""The per-token query scale gate: closed at start, bounded, a shifted head scale on constant
inputs, trainable, counted, and reported."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.attention.qk_norm import scale_rms_queries, token_log_scale
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_identity, qk_norm_token_scale_report
from scaffolds.transformer.training.optim import split_continuation_parameters
from scaffolds.transformer.training.state import create_state


def _config(token_scale="cosine", **changes):
    """Degree-four unnormalized attention with the canonical QK RMSNorm and head scale."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=4,
                 attention_backend="blocked", attention_normalization=False,
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True, qk_norm_token_scale=token_scale,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _batch(config, *, constant=None):
    shape = (2, config.data.sequence_length)
    if constant is not None:
        return jnp.full(shape, constant, jnp.int32)
    return jnp.asarray(np.random.default_rng(0).integers(0, config.model.vocab_size, shape), jnp.int32)


def _with(params, name, value):
    """A copy of the parameter tree with one named array replaced."""
    def replace(path, leaf):
        keys = tuple(entry.key for entry in path if isinstance(entry, jax.tree_util.DictKey))
        return jnp.asarray(value, leaf.dtype) if keys and keys[0] == name else leaf

    return jax.tree.map_with_path(replace, params)


def test_token_log_scale_matches_numpy_and_is_bounded():
    rng = np.random.default_rng(1)
    x, kernel, amplitude = rng.normal(size=(2, 5, 8)), rng.normal(size=(3, 8)), rng.normal(size=(3, 1))
    eps = 1e-6
    unit = lambda v: v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps)
    expected = amplitude[:, 0] * np.einsum("btd,hd->bth", unit(x), unit(kernel))
    arrays = tuple(jnp.asarray(value, jnp.float32) for value in (x, kernel, amplitude))
    actual = np.asarray(token_log_scale(*arrays, eps=eps))
    assert actual.dtype == np.float32 and actual.shape == (2, 5, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert np.all(np.abs(actual) <= np.abs(amplitude[:, 0]) + 1e-6)
    closed = token_log_scale(arrays[0], arrays[1], jnp.zeros((3, 1), jnp.float32), eps=eps)
    np.testing.assert_array_equal(np.asarray(closed), np.zeros((2, 5, 3), np.float32))
    with pytest.raises(ValueError):
        token_log_scale(arrays[0], arrays[1], jnp.zeros((3,), jnp.float32), eps=eps)


def test_scale_rms_queries_adds_the_token_log_scale():
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.normal(size=(2, 5, 3, 4)), jnp.float32)
    gamma = jnp.asarray(rng.normal(size=(3, 1)), jnp.float32)
    delta = jnp.asarray(rng.normal(size=(2, 5, 3)), jnp.float32)
    static = np.asarray(q) * np.exp(np.asarray(gamma)[:, 0])[None, None, :, None]
    np.testing.assert_allclose(np.asarray(scale_rms_queries(q, gamma)), static, rtol=1e-6, atol=1e-6)
    gated = np.asarray(q) * np.exp(np.asarray(gamma)[:, 0][None, None, :] + np.asarray(delta))[..., None]
    np.testing.assert_allclose(np.asarray(scale_rms_queries(q, gamma, delta)), gated, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        scale_rms_queries(q, gamma, delta[:, :, :2])


def test_acceptance_softmax_comparison_drops_the_token_scale():
    """The acceptance's derived softmax controls must validate without the gate."""
    from scaffolds.transformer import acceptance

    config = _config()
    facts = acceptance.matches_softmax_backbone_initialization(config)  # raises on any mismatch
    assert "qk_norm_token_scale_amplitude" in facts["mechanism_parameters"]


def test_token_scale_requires_head_scale_and_a_known_form():
    with pytest.raises(ValueError, match="head scaling"):
        _config(qk_norm_head_scale=False)
    with pytest.raises(ValueError, match="Unknown QK token scale"):
        _config("sigmoid")


def test_specs_parameter_count_and_identity():
    config = _config()
    specs = {spec.name: spec for spec in attention_parameter_specs(config.model)}
    assert specs["qk_norm_token_scale_kernel"].initial_std == pytest.approx(1 / math.sqrt(8))
    assert specs["qk_norm_token_scale_amplitude"].initial == (0.0,)
    assert not specs["qk_norm_token_scale_kernel"].weight_decay
    assert not specs["qk_norm_token_scale_amplitude"].weight_decay
    state = create_state(config, jax.random.PRNGKey(0))
    amplitude = np.asarray(state.params["qk_norm_token_scale_amplitude"][...])
    kernel = np.asarray(state.params["qk_norm_token_scale_kernel"][...])
    assert amplitude.shape == (2, 2, 1) and np.all(amplitude == 0)
    assert kernel.shape == (2, 2, 8) and np.all(np.isfinite(kernel)) and np.all(np.any(kernel != 0, axis=-1))
    count = sum(leaf.size for leaf in jax.tree.leaves(state.params))
    assert count == model_parameter_count(config.model)
    assert count - model_parameter_count(_config("none").model) == 2 * 2 * (8 + 1)
    identity = attention_identity(config)
    assert identity["qk_norm_token_scale"] == "cosine"
    assert identity["attention_score_formula"].startswith("exp(a_h cos(w_h, x_t)) * exp(gamma_h) * ")
    assert "qk_norm_token_scale" not in attention_identity(_config("none"))


def test_closed_gate_reproduces_the_head_scaled_model_exactly():
    gated_config, plain_config = _config(), _config("none")
    gated = create_state(gated_config, jax.random.PRNGKey(3))
    plain = create_state(plain_config, jax.random.PRNGKey(3))
    gate, backbone = split_continuation_parameters(gated.params)
    assert set(gate) == {"qk_norm_token_scale_kernel", "qk_norm_token_scale_amplitude"}
    assert jax.tree.structure(backbone) == jax.tree.structure(plain.params)
    for actual, expected in zip(jax.tree.leaves(backbone), jax.tree.leaves(plain.params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    batch = _batch(gated_config)
    inputs = batch[:, :-1]
    gated_logits = nnx.merge(gated.graphdef, gated.params)(inputs, deterministic=True)
    plain_logits = nnx.merge(plain.graphdef, plain.params)(inputs, deterministic=True)
    np.testing.assert_array_equal(np.asarray(gated_logits), np.asarray(plain_logits))
    _, gated_grads = loop.loss_and_grads(gated, batch, jax.random.PRNGKey(0))
    _, plain_grads = loop.loss_and_grads(plain, batch, jax.random.PRNGKey(0))
    gate_grads, backbone_grads = split_continuation_parameters(gated_grads)
    for actual, expected in zip(jax.tree.leaves(backbone_grads), jax.tree.leaves(plain_grads), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    # A closed gate still receives amplitude gradients; its direction learns once a_h moves.
    assert np.any(np.asarray(gate_grads["qk_norm_token_scale_amplitude"][...]) != 0)
    assert np.all(np.asarray(gate_grads["qk_norm_token_scale_kernel"][...]) == 0)


def test_open_gate_equals_a_shifted_head_scale_when_its_cosine_is_constant():
    config = _config(num_layers=1)
    state = create_state(config, jax.random.PRNGKey(4))
    # Two token types give two distinct attention inputs x_A, x_B in the only layer. A direction
    # orthogonal to u(x_A) - u(x_B) sees the same cosine at every position, so the open gate is
    # exactly one log-scale shift per head, while the keys still differ and scale matters.
    model = nnx.merge(state.graphdef, state.params)
    tokens = jnp.asarray([3, 9], jnp.int32)
    unit = lambda v: v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + config.model.qk_norm_eps)
    x = unit(np.asarray(model.layers_0.attention_norm(model.token_embedding(tokens)), np.float64))
    difference = x[0] - x[1]
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(1, 2, 8))
    kernel -= (kernel @ difference)[..., None] * difference / (difference @ difference)
    amplitude = np.asarray([[[0.7], [-0.4]]], np.float32)
    params = _with(_with(state.params, "qk_norm_token_scale_kernel", kernel),
                   "qk_norm_token_scale_amplitude", amplitude)
    inputs = jnp.asarray(tokens)[jnp.asarray(rng.integers(0, 2, (2, config.model.max_seq_len)))]
    logits, diagnostics = nnx.merge(state.graphdef, params)(inputs, deterministic=True, return_diagnostics=True)
    delta = np.asarray(diagnostics[0]["token_scale_log_delta_mean"])
    np.testing.assert_allclose(np.asarray(diagnostics[0]["token_scale_log_delta_std"]), 0, atol=1e-6)
    assert np.all(delta != 0) and np.all(np.abs(delta) <= np.abs(amplitude[0, :, 0]) + 1e-6)
    plain = create_state(_config("none", num_layers=1), jax.random.PRNGKey(4))
    gamma = np.asarray(plain.params["qk_norm_head_log_scale"][...])
    shifted = nnx.merge(plain.graphdef, _with(plain.params, "qk_norm_head_log_scale", gamma + delta[None, :, None]))
    np.testing.assert_allclose(np.asarray(logits), np.asarray(shifted(inputs, deterministic=True)),
                               rtol=1e-4, atol=1e-5)
    closed = nnx.merge(plain.graphdef, plain.params)(inputs, deterministic=True)
    assert not np.allclose(np.asarray(logits), np.asarray(closed), atol=1e-4)


def test_training_opens_the_gate_then_its_direction():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(5))
    batch = _batch(config)
    after, *_ = loop.train_step(state, batch, jax.random.PRNGKey(0))
    assert np.any(np.asarray(after.params["qk_norm_token_scale_amplitude"][...]) != 0)
    np.testing.assert_array_equal(np.asarray(after.params["qk_norm_token_scale_kernel"][...]),
                                  np.asarray(state.params["qk_norm_token_scale_kernel"][...]))
    twice, *_ = loop.train_step(after, batch, jax.random.PRNGKey(1))
    assert np.any(np.asarray(twice.params["qk_norm_token_scale_kernel"][...])
                  != np.asarray(after.params["qk_norm_token_scale_kernel"][...]))


def test_token_scale_report_summarizes_amplitudes_and_the_sample():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(6))
    raw_layers = [{f"token_scale_log_delta_{name}": np.full(2, 0.25) for name in ("mean", "std", "min", "max")}
                  for _ in range(config.model.num_layers)]
    report = qk_norm_token_scale_report(state.params, raw_layers)
    assert report["finite"] and report["effective_coefficient_names"] == ["a_h"]
    assert np.asarray(report["effective"]).shape == (2, 2, 1)
    assert np.asarray(report["kernel_l2"]).shape == (2, 2) and np.all(np.asarray(report["kernel_l2"]) > 0)
    assert np.asarray(report["log_delta_sample"]["max"]).shape == (2, 2)
