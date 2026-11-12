"""FoX decay multiplies the polynomial after PaTH transports the query and key."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.path import path_attention


def _inputs(*, time=12, degree=4, seed=0):
    rng = np.random.default_rng(seed)
    shape = (1, time, 2, 4)
    q, k, w = (rng.normal(size=shape).astype(np.float32) for _ in range(3))
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    v = rng.normal(size=(*shape[:3], 3)).astype(np.float32)
    beta = rng.uniform(0.2, 1.8, size=shape[:3]).astype(np.float32)
    c = rng.normal(scale=0.15, size=(*shape[:3], degree + 1)).astype(np.float32)
    c += np.asarray([1 / math.factorial(r) for r in range(degree + 1)], np.float32)
    log_decay = -np.cumsum(rng.uniform(0.01, 0.5, size=shape[:3]), axis=1).astype(np.float32)
    return tuple(jnp.asarray(x) for x in (q, k, v, w, beta, c, log_decay))


def _dense_definition(q, k, v, w, beta, c, log_decay):
    """Independent FP64 dense products, without chunk transforms or Horner evaluation."""
    q, k, v, w, beta, c, log_decay = (
        np.asarray(x, np.float64) for x in (q, k, v, w, beta, c, log_decay))
    batch, time, heads, dim = q.shape
    output = np.zeros(v.shape, np.float64)
    weights = []
    row_sums = []
    for b in range(batch):
        for h in range(heads):
            for t in range(time):
                transport = np.eye(dim)
                row_sum = 0.0
                for s in range(t, -1, -1):
                    score = q[b, t, h] @ transport @ k[b, s, h] / math.sqrt(dim)
                    polynomial = sum(c[b, t, h, r] * score ** r for r in range(c.shape[-1]))
                    weight = np.exp(log_decay[b, t, h] - log_decay[b, s, h]) * polynomial
                    output[b, t, h] += weight * v[b, s, h]
                    weights.append(weight)
                    row_sum += weight
                    transport = transport @ (np.eye(dim) - beta[b, s, h] * np.outer(w[b, s, h], w[b, s, h]))
                row_sums.append(row_sum)
    return output, {"kernel_min": min(weights), "kernel_max": max(weights),
                    "weight_sum_min": min(row_sums), "weight_sum_max": max(row_sums)}


def _sequential_definition(q, k, v, w, beta, c, log_decay):
    """Differentiable per-query transport by the defining Householder products."""
    batch, time, heads, dim = q.shape
    rows = []
    for t in range(time):
        transported_query = q[:, t]
        row = jnp.zeros((batch, heads, v.shape[-1]), jnp.float32)
        for s in range(t, -1, -1):
            score = jnp.sum(transported_query * k[:, s], axis=-1) / math.sqrt(dim)
            polynomial = sum(c[:, t, :, r] * score ** r for r in range(c.shape[-1]))
            weight = jnp.exp(log_decay[:, t] - log_decay[:, s]) * polynomial
            row += weight[..., None] * v[:, s]
            transported_query -= (beta[:, s] * jnp.sum(transported_query * w[:, s], axis=-1))[..., None] * w[:, s]
        rows.append(row)
    return jnp.stack(rows, axis=1)


def _attend(*operands, chunk=4, return_diagnostics=False):
    return path_attention(*operands[:5], coefficients=operands[5], log_decay=operands[6],
                          kernel="query_polynomial", chunk_size=chunk,
                          precision=jax.lax.Precision.HIGHEST, return_diagnostics=return_diagnostics)


def _assert_close(actual, expected, *, tolerance=3e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("chunk", (1, 3, 12))
def test_gated_chunked_attention_and_diagnostics_match_dense_definition(chunk):
    operands = _inputs()
    actual, diagnostics = _attend(*operands, chunk=chunk, return_diagnostics=True)
    expected, expected_diagnostics = _dense_definition(*operands)
    _assert_close(actual, expected)
    for name, value in expected_diagnostics.items():
        _assert_close(diagnostics[name], value)
    assert int(diagnostics["invalid_row_count"]) == int(diagnostics["invalid_output_count"]) == 0


def test_every_operand_gradient_matches_sequential_definition():
    operands = _inputs(time=6, degree=3, seed=2)
    cotangent = jnp.asarray(np.random.default_rng(3).normal(size=operands[2].shape), jnp.float32)
    with jax.default_matmul_precision("highest"):
        actual, pull = jax.vjp(lambda *x: _attend(*x, chunk=2), *operands)
        expected, pull_expected = jax.vjp(_sequential_definition, *operands)
        actual_grads, expected_grads = pull(cotangent), pull_expected(cotangent)
    _assert_close(actual, expected)
    for name, actual_grad, expected_grad in zip(
            ("q", "k", "v", "w", "beta", "coefficients", "log_decay"), actual_grads, expected_grads, strict=True):
        assert float(jnp.linalg.norm(expected_grad)) > 0, name
        _assert_close(actual_grad, expected_grad, tolerance=8e-5)


def test_zero_decay_recovers_ungated_path_and_preserves_existing_gradients():
    operands = _inputs(time=8, degree=2, seed=4)
    zero = jnp.zeros_like(operands[6])
    gated = lambda *x: _attend(*x, zero)
    ungated = lambda *x: path_attention(*x[:5], coefficients=x[5], kernel="query_polynomial",
                                       chunk_size=4, precision=jax.lax.Precision.HIGHEST)
    actual, pull = jax.vjp(gated, *operands[:6])
    expected, pull_expected = jax.vjp(ungated, *operands[:6])
    np.testing.assert_array_equal(actual, expected)
    for actual_grad, expected_grad in zip(pull(jnp.ones_like(actual)), pull_expected(jnp.ones_like(expected)), strict=True):
        np.testing.assert_array_equal(actual_grad, expected_grad)


def test_identity_transport_recovers_existing_fox_attention_and_gradients():
    q, k, v, w, beta, c, log_decay = _inputs(time=8, degree=3, seed=5)
    operands = (q, k, v, c, log_decay)
    gated_path = lambda q, k, v, c, decay: _attend(q, k, v, w, jnp.zeros_like(beta), c, decay)
    fox = lambda q, k, v, c, decay: blocked_dot_product_attention(
        q, k, v, attention_degree=3, attention_normalization=False,
        query_polynomial_coefficients=c, query_log_decay=decay, query_block_size=4)
    with jax.default_matmul_precision("highest"):
        actual, pull = jax.vjp(gated_path, *operands)
        expected, pull_expected = jax.vjp(fox, *operands)
        grads, expected_grads = pull(jnp.ones_like(actual)), pull_expected(jnp.ones_like(expected))
    _assert_close(actual, expected)
    for actual_grad, expected_grad in zip(grads, expected_grads, strict=True):
        _assert_close(actual_grad, expected_grad)


def test_outputs_and_all_operand_gradients_are_causal():
    operands = _inputs(time=8, degree=2, seed=6)
    prefix = 3  # Splits a chunk as well as leaving later chunks.
    changed = [x.at[:, prefix:].add(0.7) for x in operands]
    changed[3] /= jnp.linalg.norm(changed[3], axis=-1, keepdims=True)
    changed[4] = jnp.clip(changed[4], 0.1, 1.9)
    changed[6] = operands[6].at[:, prefix:].add(-5.0)
    output, pull = jax.vjp(_attend, *operands)
    perturbed, pull_perturbed = jax.vjp(_attend, *changed)
    _assert_close(output[:, :prefix], perturbed[:, :prefix])
    cotangent = jnp.ones_like(output).at[:, prefix:].set(0.0)
    for first, second in zip(pull(cotangent), pull_perturbed(cotangent), strict=True):
        np.testing.assert_array_equal(first[:, prefix:], 0.0)
        np.testing.assert_array_equal(second[:, prefix:], 0.0)
        _assert_close(first[:, :prefix], second[:, :prefix])


def test_large_cumulative_decay_masks_future_overflow_in_forward_and_backward():
    operands = list(_inputs(time=8, degree=3, seed=7))
    operands[6] = jnp.broadcast_to(-1000.0 * jnp.arange(8, dtype=jnp.float32)[None, :, None], operands[6].shape)
    output, pull = jax.vjp(_attend, *operands)
    q, k, v, _, _, c, _ = operands
    score = jnp.sum(q * k, axis=-1) / math.sqrt(q.shape[-1])
    self_weight = sum(c[..., r] * score ** r for r in range(c.shape[-1]))
    _assert_close(output, self_weight[..., None] * v)
    for gradient in pull(jnp.ones_like(output)):
        assert np.isfinite(gradient).all()


def test_only_cumulative_decay_differences_matter():
    operands = _inputs(time=8, seed=8)
    shifted = (*operands[:6], operands[6] + jnp.asarray([[[3.0, -5.0]]], jnp.float32))
    _assert_close(_attend(*operands), _attend(*shifted))


@pytest.mark.parametrize("shape", ((1, 8), (1, 8, 2, 1), (1, 7, 2)))
def test_rejects_wrong_cumulative_decay_shape(shape):
    operands = _inputs(time=8)
    with pytest.raises(ValueError, match="cumulative log decay"):
        _attend(*operands[:6], jnp.zeros(shape))

