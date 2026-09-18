import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from experiments.polynomial_attention.configs import ATTENTION_DEGREES
from experiments.polynomial_attention.models.components.attention import (
    QUERY_BLOCK_SIZE,
    _taylor_polynomial,
    blocked_dot_product_attention,
    padded_dot_product_attention,
)


def _dense_reference(q, k, v, degree):
    """Independent direct power sum, evaluated in FP64 by reference tests."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    if degree is None:
        kernel = jnp.exp(scores - jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1, keepdims=True))
    else:
        kernel = sum(scores ** order / float(math.factorial(order)) for order in range(degree + 1))
        kernel = jnp.broadcast_to(kernel, scores.shape)
    kernel = jnp.where(causal, kernel, 0)
    weights = kernel / kernel.sum(axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


def _inputs(time=13, dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(7), 3)
    q, k = [0.7 * jax.random.normal(key, (2, time, 2, 8)) for key in keys[:2]]
    v = jax.random.normal(keys[2], (2, time, 2, 5))
    return tuple(x.astype(dtype) for x in (q, k, v))


@pytest.mark.parametrize("degree", ATTENTION_DEGREES)
def test_blocked_outputs_and_qkv_gradients_match_dense_fp64(degree):
    q, k, v = _inputs()
    cotangent = jax.random.normal(jax.random.key(9), v.shape)

    def blocked(q, k, v):
        return blocked_dot_product_attention(q, k, v, attention_degree=degree, query_block_size=4)

    actual = blocked(q, k, v)
    actual_grads = jax.grad(lambda q, k, v: jnp.sum(blocked(q, k, v) * cotangent), argnums=(0, 1, 2))(q, k, v)
    with jax.enable_x64():
        q64, k64, v64 = (x.astype(jnp.float64) for x in (q, k, v))
        expected = _dense_reference(q64, k64, v64, degree)
        expected_grads = jax.grad(
            lambda q, k, v: jnp.sum(_dense_reference(q, k, v, degree) * cotangent.astype(jnp.float64)),
            argnums=(0, 1, 2),
        )(q64, k64, v64)
    np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=3e-5)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(observed, reference, atol=2e-6, rtol=5e-5)


@pytest.mark.parametrize("degree", ATTENTION_DEGREES)
def test_default_block_boundaries_causality_and_diagnostics(degree):
    q, k, v = _inputs(time=QUERY_BLOCK_SIZE + 3)
    output, diagnostics = blocked_dot_product_attention(q, k, v, attention_degree=degree, return_diagnostics=True)
    changed = (q.at[:, 127:].multiply(-4), k.at[:, 127:].multiply(3), v.at[:, 127:].add(2))
    modified = blocked_dot_product_attention(*changed, attention_degree=degree)
    np.testing.assert_array_equal(modified[:, :127], output[:, :127])
    np.testing.assert_allclose(output, _dense_reference(q, k, v, degree), atol=2e-6, rtol=6e-5)
    assert all(value.shape == () for value in diagnostics.values())
    assert diagnostics["normalizer_min"] > 0
    assert diagnostics["kernel_min"] > 0
    assert all(int(value) == 0 for name, value in diagnostics.items() if name.endswith("_count"))


def test_degree_zero_is_prefix_average_and_qk_gradients_are_zero():
    q, k, v = _inputs()
    expected = jnp.cumsum(v, axis=1) / jnp.arange(1, v.shape[1] + 1)[None, :, None, None]
    actual, diagnostics = blocked_dot_product_attention(q, k, v, attention_degree=0, query_block_size=4, return_diagnostics=True)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    grads = jax.grad(lambda q, k, v: jnp.square(blocked_dot_product_attention(q, k, v, attention_degree=0)).sum(), argnums=(0, 1, 2))(q, k, v)
    np.testing.assert_array_equal(grads[0], jnp.zeros_like(q))
    np.testing.assert_array_equal(grads[1], jnp.zeros_like(k))
    assert float(jnp.linalg.norm(grads[2])) > 0
    assert float(diagnostics["kernel_min"]) == float(diagnostics["kernel_max"]) == 1
    assert float(diagnostics["normalizer_min"]) == 1
    assert float(diagnostics["normalizer_max"]) == v.shape[1]


def test_taylor_uses_unshifted_negative_scores():
    q = jnp.ones((1, 2, 1, 1))
    k = jnp.array([-5.0, 0.0]).reshape(1, 2, 1, 1)
    v = jnp.array([1.0, 0.0]).reshape(1, 2, 1, 1)
    actual, diag = blocked_dot_product_attention(q, k, v, attention_degree=2, return_diagnostics=True)
    np.testing.assert_allclose(actual[0, 1, 0, 0], 8.5 / 9.5, atol=1e-7)
    assert float(diag["score_min"]) == -5
    assert float(diag["kernel_max"]) == 8.5


def test_masked_positions_are_not_polynomial_sentinels_or_diagnostic_entries():
    q = jnp.array([1.0, 0.0]).reshape(1, 2, 1, 1)
    k = jnp.array([0.0, 1e30]).reshape(1, 2, 1, 1)
    v = jnp.array([3.0, 7.0]).reshape(1, 2, 1, 1)
    result, diag = blocked_dot_product_attention(q, k, v, attention_degree=16, query_block_size=1, return_diagnostics=True)
    np.testing.assert_array_equal(result.reshape(-1), [3, 5])
    assert float(diag["score_max"]) == 0
    assert int(diag["invalid_kernel_count"]) == 0


def test_overflow_is_reported_and_poisoned_without_clipping():
    q = jnp.full((1, 2, 1, 1), 1e10)
    output, diag = blocked_dot_product_attention(q, q, jnp.ones_like(q), attention_degree=16, return_diagnostics=True)
    assert bool(jnp.all(jnp.isnan(output)))
    assert int(diag["invalid_kernel_count"]) == 3
    assert int(diag["invalid_normalizer_count"]) == 2


def test_exact_exponential_underflow_is_allowed():
    q = jnp.ones((1, 2, 1, 1))
    k = jnp.array([-1000.0, 1000.0]).reshape(q.shape)
    output, diag = blocked_dot_product_attention(q, k, q, return_diagnostics=True)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert float(diag["kernel_min"]) == 0
    assert int(diag["invalid_kernel_count"]) == 0


@pytest.mark.parametrize("degree", (-2, 1, 3, 2.0, True, "2"))
def test_invalid_degrees_are_rejected(degree):
    with pytest.raises(ValueError, match="nonnegative even"):
        blocked_dot_product_attention(*_inputs(), attention_degree=degree)


def test_exact_control_matches_original_sdpa_outputs_and_gradients():
    q, k, v = _inputs(time=17)
    expected = padded_dot_product_attention(q, k, v, backend="xla")
    actual = blocked_dot_product_attention(q, k, v)
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=3e-5)
    reference = jax.grad(lambda q, k, v: jnp.square(padded_dot_product_attention(q, k, v)).sum(), argnums=(0, 1, 2))(q, k, v)
    observed = jax.grad(lambda q, k, v: jnp.square(blocked_dot_product_attention(q, k, v)).sum(), argnums=(0, 1, 2))(q, k, v)
    for x, y in zip(observed, reference, strict=True):
        np.testing.assert_allclose(x, y, atol=3e-6, rtol=4e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", ATTENTION_DEGREES)
def test_gpu_production_context_bf16_forward_backward(degree):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU")
    keys = jax.random.split(jax.random.key(12), 3)
    operands = tuple(jax.random.normal(key, (1, 2048, 4, 64)).astype(jnp.bfloat16) for key in keys)

    def loss(q, k, v):
        output = blocked_dot_product_attention(q, k, v, attention_degree=degree)
        return jnp.square(output.astype(jnp.float32)).mean()

    value, grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(*operands)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in grads)
    assert all(x.dtype == jnp.bfloat16 for x in grads)
    assert float(jnp.linalg.norm(grads[2].astype(jnp.float32))) > 0
    if degree == 0:
        assert not bool(jnp.any(grads[0]))
        assert not bool(jnp.any(grads[1]))


@pytest.mark.gpu
def test_gpu_exact_control_matches_original_cudnn():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU")
    keys = jax.random.split(jax.random.key(13), 3)
    q, k, v = tuple(jax.random.normal(key, (2, 256, 4, 64)).astype(jnp.bfloat16) for key in keys)
    reference = lambda q, k, v: padded_dot_product_attention(q, k, v, backend="cudnn")
    actual = blocked_dot_product_attention(q, k, v)
    expected = reference(q, k, v)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=2e-2, rtol=3e-2)
    reference_grads = jax.grad(lambda q, k, v: jnp.square(reference(q, k, v).astype(jnp.float32)).mean(), argnums=(0, 1, 2))(q, k, v)
    actual_grads = jax.grad(lambda q, k, v: jnp.square(blocked_dot_product_attention(q, k, v).astype(jnp.float32)).mean(), argnums=(0, 1, 2))(q, k, v)
    for observed, expected in zip(actual_grads, reference_grads, strict=True):
        np.testing.assert_allclose(observed.astype(jnp.float32), expected.astype(jnp.float32), atol=2e-6, rtol=6e-2)


@pytest.mark.parametrize("degree", (2, 4, 8, 16))
def test_negative_score_cancellation_stress_against_fp64(degree):
    scores = np.concatenate([np.linspace(-20, 20, 161), [-5.2, -5.19, -5.04]]).astype(np.float32)
    scores64 = scores.astype(np.float64)
    expected_kernel = sum(scores64 ** order / float(math.factorial(order)) for order in range(degree + 1))
    actual_kernel = jax.jit(lambda scores: _taylor_polynomial(scores, degree))(jnp.asarray(scores))
    assert bool(jnp.all(jnp.isfinite(actual_kernel) & (actual_kernel > 0)))
    np.testing.assert_allclose(actual_kernel, expected_kernel, rtol=1e-3, atol=1e-6)
    q = jnp.ones((1, len(scores), 1, 1))
    k = jnp.asarray(scores).reshape(q.shape)
    v = jnp.sin(jnp.arange(len(scores), dtype=jnp.float32)).reshape(q.shape)
    actual = blocked_dot_product_attention(q, k, v, attention_degree=degree)
    with jax.enable_x64():
        expected = _dense_reference(q.astype(jnp.float64), k.astype(jnp.float64), v.astype(jnp.float64), degree)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-6)


def test_large_degree_recurrence_avoids_factorial_coefficient_overflow():
    actual = _taylor_polynomial(jnp.array([0.0, 1.0]), 172)
    np.testing.assert_allclose(actual, [1.0, math.e], rtol=2e-7, atol=1e-7)
