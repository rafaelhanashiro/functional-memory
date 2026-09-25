"""Softmax attention backends (manual, XLA, cuDNN) agree across key widths."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.softmax import padded_dot_product_attention
from scaffolds.transformer.attention.positional import apply_rope


KEY_DIMENSIONS = (2, 4, 8, 16, 32, 64, 128)
ATTENTION_DIMENSIONS = tuple(dict.fromkeys((*KEY_DIMENSIONS, 256)))


def _unpad_reference(q, k, v):
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("bhts,bshd->bthd", probabilities, v)


@pytest.mark.parametrize("qk_dim", ATTENTION_DIMENSIONS)
def test_padded_xla_matches_unpadded_output_and_qkv_gradients(qk_dim):
    keys = jax.random.split(jax.random.key(7), 3)
    q, k = [jax.random.normal(key, (2, 9, 2, qk_dim)) for key in keys[:2]]
    v = jax.random.normal(keys[2], (2, 9, 2, 64))

    def fused(q, k, v):
        return padded_dot_product_attention(q, k, v, backend="xla")

    expected, actual = _unpad_reference(q, k, v), fused(q, k, v)
    assert actual.shape == (2, 9, 2, 64)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    reference_grads = jax.grad(lambda q, k, v: jnp.square(_unpad_reference(q, k, v)).mean(), argnums=(0, 1, 2))(q, k, v)
    actual_grads = jax.grad(lambda q, k, v: jnp.square(fused(q, k, v)).mean(), argnums=(0, 1, 2))(q, k, v)
    for actual_grad, expected_grad in zip(actual_grads, reference_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-7, rtol=3e-5)


def _layer(qk_dim, backend, *, hidden_size=32, heads=4, dtype=jnp.float32):
    layer = MultiHeadAttention(hidden_size, heads, heads, qk_dim, 64, backend=backend, rngs=nnx.Rngs(8))
    layer.dtype = dtype
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(layer, name).dtype = dtype
    return layer


def _model_gradients(layer, x):
    graph, params = nnx.split(layer, nnx.Param)

    def loss(params, x):
        result = nnx.merge(graph, params)(x)
        return jnp.square(result.astype(jnp.float32)).mean()

    return jax.grad(loss, argnums=(0, 1))(params, x)


@pytest.mark.parametrize("qk_dim", ATTENTION_DIMENSIONS)
def test_xla_projection_shapes_causality_and_model_gradient_parity(qk_dim):
    manual, fused = _layer(qk_dim, "manual"), _layer(qk_dim, "xla")
    x = jax.random.normal(jax.random.key(9), (2, 11, 32))
    assert fused.q_proj.kernel.shape == (32, 4 * qk_dim)
    assert fused.k_proj.kernel.shape == (32, 4 * qk_dim)
    assert fused.v_proj.kernel.shape == (32, 4 * 64)
    assert fused.o_proj.kernel.shape == (4 * 64, 32)
    expected, actual = manual(x), fused(x)
    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    actual_grads, expected_grads = _model_gradients(fused, x), _model_gradients(manual, x)
    for actual_grad, expected_grad in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=4e-7, rtol=4e-4)
    changed = x.at[:, 7:, :].set(-3 * x[:, 7:, :])
    np.testing.assert_allclose(fused(changed)[:, :7], actual[:, :7], atol=1e-6, rtol=1e-6)


class _OFMReferenceAttention(nnx.Module):
    """Standalone equal-width full-MHA subset of OFM 8977f529e83e.

    This intentionally reconstructs the original allocation order and direct
    public SDPA call, without using the new attention implementation.
    """

    def __init__(self, *, rngs):
        self.q_proj = nnx.Linear(256, 256, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(256, 256, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(256, 256, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(0.0)
        self.o_proj = nnx.Linear(256, 256, use_bias=False, rngs=rngs)

    def __call__(self, x):
        batch, time, _ = x.shape
        q, k, v = [projection(x).reshape(batch, time, 4, 64) for projection in (self.q_proj, self.k_proj, self.v_proj)]
        q, k = [apply_rope(operand, jnp.arange(time), 10000.0) for operand in (q, k)]
        output = jax.nn.dot_product_attention(q, k, v, scale=1.0 / math.sqrt(64), is_causal=True, implementation="xla")
        return self.o_proj(output.reshape(batch, time, 256))


def test_dk64_baseline_preserves_ofm_initialization_outputs_and_gradients():
    baseline = _OFMReferenceAttention(rngs=nnx.Rngs(8))
    actual = _layer(64, "xla", hidden_size=256)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        np.testing.assert_array_equal(getattr(actual, name).kernel[...], getattr(baseline, name).kernel[...])
        assert getattr(actual, name).kernel.dtype == jnp.float32
    x = jax.random.normal(jax.random.key(10), (1, 5, 256))
    np.testing.assert_array_equal(actual(x), baseline(x))
    for observed, expected in zip(jax.tree.leaves(_model_gradients(actual, x)), jax.tree.leaves(_model_gradients(baseline, x)), strict=True):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("qk_dim", ATTENTION_DIMENSIONS)
def test_cudnn_matches_manual_outputs_input_and_parameter_gradients(qk_dim):
    if jax.default_backend() != "gpu":
        pytest.skip("cuDNN parity requires a GPU")
    manual = _layer(qk_dim, "manual", hidden_size=64, dtype=jnp.bfloat16)
    fused = _layer(qk_dim, "cudnn", hidden_size=64, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(11), (2, 128, 64)).astype(jnp.bfloat16)
    expected, actual = manual(x), fused(x)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=4e-2, rtol=4e-2)
    for observed, reference in zip(jax.tree.leaves(_model_gradients(fused, x)), jax.tree.leaves(_model_gradients(manual, x)), strict=True):
        np.testing.assert_allclose(observed.astype(jnp.float32), reference.astype(jnp.float32), atol=5e-4, rtol=6e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("qk_dim", ATTENTION_DIMENSIONS)
def test_cudnn_production_context_forward_and_backward_are_finite(qk_dim):
    if jax.default_backend() != "gpu":
        pytest.skip("cuDNN production-context check requires a GPU")
    layer = _layer(qk_dim, "cudnn", hidden_size=1024, heads=16, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(12), (1, 2048, 1024)).astype(jnp.bfloat16)
    output = layer(x)
    gradient = jax.grad(lambda candidate: jnp.square(layer(candidate).astype(jnp.float32)).mean())(x)
    assert output.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient.astype(jnp.float32))) > 0
