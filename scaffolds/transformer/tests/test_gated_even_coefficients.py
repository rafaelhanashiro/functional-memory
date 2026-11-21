"""Compact positive sigmoid coefficients on unit-L2 raw token activations."""
from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.input_dependent import normalize_coefficient_input, project_coefficients
from scaffolds.transformer.attention.parameters import attention_parameter_specs, parameters_for_layer
from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.diagnostics import attention_identity, coefficient_input_report
from scaffolds.transformer.training.numerics import is_signed_kernel

ORDERS = (2, 4, 6, 8)


def _model(initialization):
    return ModelConfig(vocab_size=32, max_seq_len=8, num_layers=1, hidden_size=16, num_heads=2, num_kv_heads=2,
                       qk_head_dim=4, value_head_dim=4, intermediate_size=32, attention_degree=8,
                       attention_normalization=False, attention_backend="blocked", attention_coefficients="learned",
                       attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                       attention_coefficient_activation="sigmoid", attention_coefficient_orders=ORDERS,
                       attention_coefficient_sigmoid_bias_initial=initialization,
                       attention_coefficient_input_source="pre_rmsnorm", attention_coefficient_input_normalization="l2",
                       attention_coefficient_projection_initial_std=.25, attention_coefficient_kernel_weight_decay=True,
                       qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                       attention_output_scaling="head_rms_norm", attention_head_rms_prescale="max_abs",
                       initialization="transformer_truncated_reference")


@pytest.mark.parametrize("initialization", ("zero", "logit_taylor"))
def test_full_head_uses_compact_gate_bias_and_raw_l2_input(initialization):
    config = _model(initialization)
    model = build_model(config, 41)
    other = build_model(replace(config, attention_coefficient_sigmoid_bias_initial=(
        "zero" if initialization == "logit_taylor" else "logit_taylor")), 41)
    np.testing.assert_array_equal(model.query_polynomial_kernel[...], other.query_polynomial_kernel[...])
    decay = {s.name: s.weight_decay for s in attention_parameter_specs(config)}
    assert decay["query_polynomial_kernel"] and not decay["query_polynomial_bias"]
    assert not is_signed_kernel(config)
    params = parameters_for_layer(model, model.attention_parameter_specs, 0)
    kernel = np.asarray(params["query_polynomial_kernel"], np.float64)
    bias = np.asarray(params["query_polynomial_bias"], np.float64)
    assert kernel.shape == (2, 16, 4)
    assert .1 < kernel.std() < .4
    expected = [0 if initialization == "zero" else -math.log(math.factorial(r)-1) for r in ORDERS]
    np.testing.assert_allclose(bias, np.tile(expected, (2, 1)), atol=1e-6)
    raw = jnp.asarray(np.random.default_rng(8).normal(size=(2, 8, 16)), jnp.float32)
    attn = model.layers_0.attention
    x = model.layers_0.attention_norm(raw)
    _, cs = attn(x, attention_parameters=params, coefficient_source_input=raw, return_coefficients=True)
    unit = np.asarray(raw, np.float64)
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    z = np.einsum("btd,hdr->bthr", unit, kernel) + bias
    np.testing.assert_allclose(np.asarray(cs)[..., ORDERS], 1 / (1 + np.exp(-z)), rtol=2e-6)
    np.testing.assert_array_equal(np.asarray(cs)[..., [0, 1, 3, 5, 7]], 0)
    _, diagnostic = attn(x, attention_parameters=params, coefficient_source_input=raw, return_diagnostics=True)
    report = coefficient_input_report([diagnostic], config)
    assert report["coordinate_labels"] == ["r2", "r4", "r6", "r8"]
    assert report["finite"] and report["saturation_sigmoid_tail_thresholds"] == [.01, .99]
    identity = attention_identity(Config(model=config))
    assert identity["attention_coefficient_count"] == 2 * 4 * 17
    assert "sigmoid(x_t W_r + b_r)" in identity["attention_formula"]


def test_gate_and_input_gradients_match_independent_fp64_reference():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 5, 8)).astype(np.float32)
    w = rng.normal(size=(2, 8, 4)).astype(np.float32)
    b = rng.normal(size=(2, 4)).astype(np.float32)
    scores = rng.uniform(-2, 2, size=(2, 5, 2)).astype(np.float32)
    def production(x, w, b):
        coefficients = project_coefficients(normalize_coefficient_input(x, "l2"), w, None, activation="sigmoid",
                                            sigmoid_logit_bias=b, orders=ORDERS)
        return (coefficients * scores[..., None] ** jnp.arange(9)).sum()

    def oracle(x, w, b):
        unit = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        logits = jnp.einsum("btd,hdr->bthr", unit, w) + b
        return (jax.nn.sigmoid(logits) * jnp.asarray(scores, jnp.float64)[..., None] ** jnp.array(ORDERS)).sum()

    with jax.enable_x64():
        expected = jax.grad(oracle, (0, 1, 2))(*(jnp.asarray(v, jnp.float64) for v in (x, w, b)))
    for actual, wanted in zip(jax.grad(production, (0, 1, 2))(x, w, b), expected, strict=True):
        assert np.linalg.norm(wanted) > 0
        np.testing.assert_allclose(actual, wanted, atol=2e-5, rtol=3e-5)
