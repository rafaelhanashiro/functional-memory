"""Unit cosine after learned RMSNorm gains and positional rotation."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import unit_cosine_score_operands
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training.diagnostics import qk_norm_identity


def test_model_scores_match_rotated_cosines_with_channel_gains(monkeypatch):
    module = multihead.MultiHeadAttention(
        8, 2, 2, 4, 4, qk_norm=True, qk_norm_type="rms_norm_cosine", rngs=nnx.Rngs(91))
    x = jax.random.normal(jax.random.key(14), (1, 7, 8))
    parameters = {"qk_norm_q_gain": jnp.asarray([0.4, 1.5, 1.1, 0.8]),
                  "qk_norm_k_gain": jnp.asarray([1.5, 0.7, 1.8, 0.6])}
    captured = {}

    def capture(q, k, v, **kwargs):
        captured.update(q=q, k=k)
        return v

    monkeypatch.setattr(multihead, "attention_core", capture)
    module(x, attention_parameters=parameters)

    def reference_direction(projected, gain):
        projected = projected.reshape(1, 7, 2, 4)
        rotated = apply_rope(projected * gain, jnp.arange(7))
        return rotated / jnp.linalg.norm(rotated, axis=-1, keepdims=True)

    actual = jnp.einsum("bthd,bshd->bhts", captured["q"], captured["k"]) / math.sqrt(4)
    expected = jnp.einsum("bthd,bshd->bhts",
                          reference_direction(module.q_proj(x), parameters["qk_norm_q_gain"]),
                          reference_direction(module.k_proj(x), parameters["qk_norm_k_gain"]))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-7)
    assert jnp.max(jnp.abs(actual)) <= 1.0 + 1e-6


def test_parallel_antiparallel_orthogonal_and_zero_have_finite_backward():
    source = jnp.array([1., 2., 3., 4.])
    queries = jnp.stack((source, 3 * source, -source, jnp.array([-2., 1., 0., 0.]), jnp.zeros(4)))[:, None, None, :]
    keys = jnp.broadcast_to(source, queries.shape)

    def score(q, k):
        q, k = unit_cosine_score_operands(q, k, eps=1e-6)
        return jnp.sum(q * k, axis=-1) / math.sqrt(4)

    np.testing.assert_allclose(score(queries, keys).ravel(), [1, 1, -1, 0, 0], atol=2e-7)
    dq, dk = jax.grad(lambda q, k: jnp.sum(score(q, k)), argnums=(0, 1))(queries, keys)
    assert jnp.isfinite(dq).all() and jnp.isfinite(dk).all()


def test_model_owns_only_channel_gains_and_reports_score(tmp_path):
    config = replace(tiny_config(tmp_path).model, qk_norm=True, qk_norm_type="rms_norm_cosine",
                     qk_norm_sharing="per_layer", qk_norm_head_scale=False)
    names = {spec.name for spec in attention_parameter_specs(config)}
    assert names == {"qk_norm_q_gain", "qk_norm_k_gain"}
    identity = qk_norm_identity(config)
    assert identity["attention_score_scale"] == "unit_cosine"
    assert identity["qk_norm_parameter_count"] == 2 * config.num_layers * config.qk_head_dim
    assert not identity["qk_norm_head_scale"]
    with pytest.raises(ValueError, match="head scaling"):
        replace(config, qk_norm_head_scale=True)
