"""Per-head score scales retain RMS channel gains and their initial function."""

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import rms_score_operands, scale_rms_queries
from scaffolds.transformer.configs import load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.state import create_state


def dense_config(path, *, degree=4, head_scale=True, readout_scale=1 / 3):
    config = tiny_config(path)
    config.model = replace(
        config.model, attention_backend="blocked", attention_degree=degree,
        attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
        attention_coefficient_activation="tanh", attention_coefficient_readout="dense",
        attention_coefficient_feature_count=9, attention_coefficient_readout_scale=readout_scale,
        attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=.2,
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
        attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
        qk_norm_sharing="per_layer", qk_norm_head_scale=head_scale,
        initialization="transformer_truncated_reference")
    validate_config(config)
    return config


def test_scores_and_log_scale_gradient_match_independent_reference():
    rng = np.random.default_rng(381)
    q, k = (jnp.asarray(rng.normal(size=(2, 7, 3, 8)), jnp.float32) for _ in range(2))
    q_gain, k_gain = (jnp.asarray(rng.uniform(.5, 1.5, size=8), jnp.float32) for _ in range(2))
    gamma = jnp.log(jnp.asarray([[.6], [1.0], [1.8]], jnp.float32))
    cotangent = jnp.asarray(rng.normal(size=(2, 3, 7, 7)), jnp.float32)

    def scores(log_scale):
        qn, kn = rms_score_operands(q, k, q_gain, k_gain, eps=1e-6)
        qn = scale_rms_queries(qn, log_scale)
        qn, kn = (apply_rope(v, jnp.arange(7)) for v in (qn, kn))
        return jnp.einsum("bthd,bshd->bhts", qn, kn) / math.sqrt(8)

    with jax.enable_x64():
        q64, k64 = (v.astype(jnp.float64) for v in (q, k))
        q64 = q64 / jnp.sqrt(jnp.mean(q64 ** 2, axis=-1, keepdims=True) + 1e-6) * q_gain
        k64 = k64 / jnp.sqrt(jnp.mean(k64 ** 2, axis=-1, keepdims=True) + 1e-6) * k_gain
        q64, k64 = (apply_rope(v, jnp.arange(7)) for v in (q64, k64))
        expected = jnp.einsum("bthd,bshd->bhts", q64, k64) / math.sqrt(8)
        expected = expected * jnp.exp(gamma.astype(jnp.float64)).reshape(1, 3, 1, 1)
    actual = jax.jit(scores)(gamma)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    gradient = jax.grad(lambda g: jnp.sum(scores(g) * cotangent))(gamma)
    expected_gradient = np.sum(np.asarray(expected) * np.asarray(cotangent), axis=(0, 2, 3))[:, None]
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-6)
    assert np.all(np.abs(np.asarray(gradient)) > 0)
    changed = jax.jit(scores)(gamma.at[1, 0].add(math.log(2)))
    np.testing.assert_array_equal(changed[:, [0, 2]], actual[:, [0, 2]])
    np.testing.assert_allclose(changed[:, 1], actual[:, 1] * 2, rtol=2e-6, atol=2e-6)


def test_head_scale_is_applied_once_after_rms_and_before_rope(monkeypatch):
    attention = multihead.MultiHeadAttention(
        8, 2, 2, 4, 4, qk_norm=True, qk_norm_type="rms_norm", qk_norm_head_scale=True,
        rngs=nnx.Rngs(9))
    x = jax.random.normal(jax.random.key(4), (1, 7, 8))
    parameters = {"qk_norm_q_gain": jnp.asarray([.3, .8, 1.1, 1.9]),
                  "qk_norm_k_gain": jnp.asarray([1.8, 1.1, .8, .3]),
                  "qk_norm_head_log_scale": jnp.log(jnp.asarray([[.5], [2.0]]))}
    captured = {}

    def capture(q, k, v, **_kwargs):
        captured.update(q=q, k=k)
        return v

    monkeypatch.setattr(multihead, "attention_core", capture)
    attention(x, attention_parameters=parameters)
    q, k = attention.q_proj(x).reshape(1, 7, 2, 4), attention.k_proj(x).reshape(1, 7, 2, 4)
    qn, kn = rms_score_operands(q, k, parameters["qk_norm_q_gain"], parameters["qk_norm_k_gain"], eps=1e-6)
    expected_q = apply_rope(qn * jnp.asarray([.5, 2.0])[None, None, :, None], jnp.arange(7))
    np.testing.assert_allclose(captured["q"], expected_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(captured["k"], apply_rope(kn, jnp.arange(7)))
    with pytest.raises(ValueError, match="qk_norm_head_log_scale"):
        attention(x, attention_parameters={name: value for name, value in parameters.items()
                                           if name != "qk_norm_head_log_scale"})


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_zero_log_scale_leaves_normalized_queries_array_exact(dtype):
    q = jnp.asarray(np.random.default_rng(31).normal(size=(2, 5, 3, 8)), dtype)
    actual = jax.jit(scale_rms_queries)(q, jnp.zeros((3, 1), jnp.float32))
    np.testing.assert_array_equal(actual, q)
    assert actual.dtype == dtype
    with pytest.raises(ValueError, match="log scales"):
        scale_rms_queries(q, jnp.zeros((1,)))


@pytest.mark.parametrize("degree", (4, 8))
def test_full_model_initial_parity_and_separate_head_parameters(tmp_path, degree):
    config = dense_config(tmp_path, degree=degree)
    baseline_config = replace(config.model, qk_norm_head_scale=False, attention_coefficient_readout_scale=1.0)
    model, baseline = build_model(config.model, 17), build_model(baseline_config, 17)
    assert model.qk_norm_head_log_scale.shape == (config.model.num_layers, config.model.num_heads, 1)
    np.testing.assert_array_equal(model.qk_norm_head_log_scale[...], 0)
    baseline_params, params = nnx.state(baseline, nnx.Param), nnx.state(model, nnx.Param)
    for name, value in baseline_params.items():
        for a, b in zip(jax.tree.leaves(value), jax.tree.leaves(params[name]), strict=True):
            np.testing.assert_array_equal(a, b)
    assert model_parameter_count(config.model) - model_parameter_count(baseline_config) == 2
    tokens = jnp.arange(16).reshape(4, 4) % 16
    np.testing.assert_array_equal(model(tokens), baseline(tokens))
    # Taylor's input-independent bias is untouched by the dense readout scale.
    np.testing.assert_array_equal(model.query_polynomial_bias[...], 1)
    np.testing.assert_array_equal(model.query_polynomial_mixing[...], 0)


def test_head_scales_are_fp32_not_decayed_and_train(tmp_path):
    config = dense_config(tmp_path)
    state = create_state(config, jax.random.key(13))
    spec = next(spec for spec in attention_parameter_specs(config.model)
                if spec.name == "qk_norm_head_log_scale")
    assert spec.sharing == "per_head" and not spec.weight_decay
    state.params[spec.name][...] = jnp.full(state.params[spec.name].shape, .25)
    assert state.params[spec.name].dtype == jnp.float32
    tx = make_optimizer(config)
    zero_updates, _ = tx.update(jax.tree.map(jnp.zeros_like, state.params), tx.init(state.params), state.params)
    updated = optax.apply_updates(state.params, zero_updates)
    np.testing.assert_array_equal(updated[spec.name][...], state.params[spec.name][...])
    batch = jnp.arange(20).reshape(4, 5) % 16
    after, loss, _ = loop.train_step(state, batch, jax.random.key(2), check_all_gradients=True)
    assert np.isfinite(float(loss))
    assert np.any(np.asarray(after.params[spec.name][...]) != np.asarray(state.params[spec.name][...]))


@pytest.mark.parametrize("changes", [
    {"qk_norm_head_scale": 1}, {"qk_norm": False}, {"attention_degree": 0},
    {"qk_norm_type": "cosine_log_scale", "qk_norm_sharing": "per_head"},
])
def test_invalid_head_scale_configs_are_rejected(tmp_path, changes):
    config = tiny_config(tmp_path)
    with pytest.raises(ValueError, match="head scaling|head_scale"):
        replace(config.model, **{"qk_norm": True, "qk_norm_type": "rms_norm",
                                "qk_norm_sharing": "per_layer", "qk_norm_head_scale": True, **changes})


def test_old_recipes_keep_disabled_head_scaling_and_unit_readout(tmp_path):
    config = dense_config(tmp_path, head_scale=False, readout_scale=1.0)
    current = config_to_dict(config)
    old = json.loads(json.dumps(current))
    old["model"].pop("qk_norm_head_scale")
    old["model"].pop("attention_coefficient_readout_scale")
    assert config_dicts_equal(old, current)
    assert not config_dicts_equal(old, config_to_dict(dense_config(tmp_path)))
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    restored = load_config(path)
    assert not restored.model.qk_norm_head_scale
    assert restored.model.attention_coefficient_readout_scale == 1.0
    assert "qk_norm_head_log_scale" not in nnx.state(build_model(restored.model, 2), nnx.Param)


