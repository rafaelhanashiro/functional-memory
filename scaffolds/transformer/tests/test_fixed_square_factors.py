"""Immutable sum-of-squares polynomial recipes and near-root behavior."""

from dataclasses import replace
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import matches_softmax_initialization
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.nonnegative_polynomial import effective_coefficients_numpy
from scaffolds.transformer.attention.parameters import attention_parameter_count, attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity


FACTOR = math.sqrt(math.exp(0.5) / 2)
FACTORS = (FACTOR, FACTOR, 0.0)


def _config(**overrides):
    settings = dict(attention_degree=2, attention_backend="blocked", qk_norm=False,
                    attention_coefficient_parameterization="sum_of_squares",
                    attention_fixed_square_factors=FACTORS)
    return ModelConfig(**(settings | overrides))


def _near_root_inputs(dtype=jnp.bfloat16):
    # Every nonzero input is exactly representable in BF16. The FP32 score is
    # -0.9998779296875, near the observed pilot score but not at the true root.
    q = jnp.zeros((1, 1, 1, 64), dtype).at[0, 0, 0, 0].set(1).at[0, 0, 0, 1].set(1 / 1024)
    k = jnp.zeros_like(q).at[0, 0, 0, 0].set(-8).at[0, 0, 0, 1].set(1)
    v = jnp.asarray([[[[3.0, -2.0]]]], dtype)
    return q, k, v


def _attention(q, k, v, *, backend="blocked"):
    return attention_core(q, k, v, attention_degree=2, backend=backend,
                          polynomial_square_factors=jnp.asarray(FACTORS, jnp.float32))


def test_factors_are_same_hermite_polynomial_without_expansion():
    np.testing.assert_allclose(effective_coefficients_numpy(FACTORS),
                               math.exp(0.5) * np.asarray([0.5, 1.0, 0.5]), rtol=0, atol=3e-16)
    assert np.float32(FACTOR * np.float32(-0.9998779296875) + FACTOR) != 0


def test_fixed_square_recipe_roundtrip_and_resume_identity(tmp_path):
    config = Config(model=_config())
    encoded = config_to_dict(config)
    assert encoded["model"]["attention_fixed_square_factors"] == list(FACTORS)
    path = tmp_path / "recipe.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.model.attention_fixed_square_factors == FACTORS
    assert config_to_dict(restored) == encoded
    legacy = json.loads(json.dumps(encoded))
    del legacy["model"]["attention_fixed_square_factors"]
    assert not config_dicts_equal(legacy, encoded)
    changed = config_to_dict(config)
    changed["model"]["attention_fixed_square_factors"][0] *= 1.0001
    assert not config_dicts_equal(changed, encoded)


@pytest.mark.parametrize("overrides", (
    {"attention_degree": 1}, {"attention_degree": 3}, {"attention_degree": None},
    {"attention_normalization": False}, {"attention_coefficients": "learned"},
    {"attention_coefficient_parameterization": "direct"},
    {"attention_fixed_coefficients": (0.5, 1.0, 0.5)},
    {"quadratic_mode": "positive"},
))
def test_invalid_fixed_square_combinations_are_rejected(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


@pytest.mark.parametrize("factors", (
    [], [1], [1, 1], [1, 1, 0, 0], "1,1,0", [1, True, 0],
    [1, float("nan"), 0], [1, float("inf"), 0],
))
def test_invalid_fixed_square_factors_are_rejected(factors):
    with pytest.raises(ValueError, match="attention_fixed_square_factors"):
        ModelConfig(attention_degree=2, attention_coefficient_parameterization="sum_of_squares",
                    attention_fixed_square_factors=factors)


def test_mutated_config_cannot_bypass_fixed_square_validation():
    config = Config(model=_config())
    config.model.attention_normalization = False
    with pytest.raises(ValueError, match="Fixed square factors"):
        validate_config(config)


def test_fixed_factors_keep_softmax_backbone_and_have_no_trainable_factors():
    config = Config(model=_config())
    control = replace(config.model, attention_fixed_square_factors=None,
                      attention_coefficient_parameterization="direct")
    assert attention_parameter_specs(config.model) == attention_parameter_specs(control) == ()
    assert attention_parameter_count(config.model) == 0
    assert model_parameter_count(config.model) == model_parameter_count(control)
    assert matches_softmax_initialization(config)["softmax_counterpart_parameter_sha256"]
    model = build_model(config.model, 7)
    assert not any("polynomial" in str(path) for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param)))
    modules = [module for _, module in nnx.iter_graph(model) if isinstance(module, MultiHeadAttention)]
    assert modules and all(module.attention_fixed_square_factors == FACTORS for module in modules)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "fixed_polynomial_sum_of_squares"
    assert identity["attention_output_normalization"] == "positive_row_sum"
    assert identity["attention_coefficient_count"] == 0
    assert identity["attention_fixed_square_factors"] == list(FACTORS)


def test_multihead_routes_fixed_factors_to_existing_square_kernel(monkeypatch):
    from scaffolds.transformer.attention import multihead

    original = multihead.attention_core
    calls = []

    def capture(q, k, v, **kwargs):
        np.testing.assert_array_equal(kwargs["polynomial_square_factors"], np.asarray(FACTORS, np.float32))
        assert kwargs["attention_degree"] == 2
        assert kwargs["attention_normalization"] is True
        calls.append(True)
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(multihead, "attention_core", capture)
    module = MultiHeadAttention(8, 1, 1, 64, 4, attention_degree=2, backend="blocked",
                                attention_fixed_square_factors=FACTORS,
                                rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    output = module(jnp.ones((1, 2, 8)))
    assert output.shape == (1, 2, 8)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert calls == [True]


def test_near_root_one_key_row_is_finite_with_finite_backward():
    q, k, v = _near_root_inputs()
    score = jnp.sum(q.astype(jnp.float32) * k.astype(jnp.float32)) / 8
    assert float(score) == -0.9998779296875
    output, pullback = jax.vjp(_attention, q, k, v)
    np.testing.assert_array_equal(output, v)
    gradients = pullback(jnp.ones_like(output))
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
    # The normalized one-key result is independent of the score. BF16 output
    # rounding permits a tiny residual in the quotient's reverse calculation.
    for gradient in gradients[:2]:
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-2)
    np.testing.assert_array_equal(gradients[2], jnp.ones_like(v))


def test_exact_root_still_has_invalid_denominator():
    q, k, v = _near_root_inputs()
    k = k.at[0, 0, 0, 1].set(0)
    output = _attention(q, k, v)
    assert bool(jnp.all(jnp.isnan(output)))
