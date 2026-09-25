"""Mathematical and numerical contracts for the blocked unnormalized reference."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.backends.reference import QUERY_BLOCK_SIZE, blocked_dot_product_attention
from scaffolds.transformer.attention.softmax import padded_dot_product_attention


blocked_dot_product_attention = partial(blocked_dot_product_attention, attention_normalization=False)
TAYLOR_DEGREES = (0, 1, 2, 3, 4)


def _dense_reference(q, k, v, attention_normalization=False, attention_degree=None):
    """Independent dense expression; reference callers execute this in FP64.

    Taylor kernels use the explicit power sum, not the implementations' Horner form.
    """
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    if attention_degree is not None:
        if attention_normalization:
            raise ValueError("The dense reference covers unnormalized Taylor kernels only")
        kernel = sum(scores ** n / math.factorial(n) for n in range(attention_degree + 1))
        return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, kernel, 0.0), v)
    if attention_normalization:
        scores -= jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1, keepdims=True)
    weights = jnp.where(causal, jnp.exp(scores), 0.0)
    if attention_normalization:
        weights /= weights.sum(axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


def _inputs(time=13, dtype=jnp.float32):
    rng = np.random.default_rng(7)
    q, k = [jnp.asarray(0.7 * rng.normal(size=(2, time, 2, 8)), dtype=dtype) for _ in range(2)]
    v = jnp.asarray(rng.normal(size=(2, time, 2, 5)), dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("normalized", [False, True])
def test_blocked_outputs_and_qkv_gradients_match_dense_fp64(normalized):
    q, k, v = _inputs()
    cotangent = jnp.asarray(np.random.default_rng(9).normal(size=v.shape), dtype=v.dtype)

    def blocked(q, k, v):
        return blocked_dot_product_attention(q, k, v, attention_normalization=normalized, query_block_size=4)

    with jax.default_matmul_precision("highest"):
        actual = blocked(q, k, v)
        actual_grads = jax.grad(lambda q, k, v: jnp.sum(blocked(q, k, v) * cotangent), argnums=(0, 1, 2))(q, k, v)
        with jax.enable_x64():
            q64, k64, v64 = (x.astype(jnp.float64) for x in (q, k, v))
            expected = _dense_reference(q64, k64, v64, normalized)
            expected_grads = jax.grad(
                lambda q, k, v: jnp.sum(_dense_reference(q, k, v, normalized) * cotangent.astype(jnp.float64)),
                argnums=(0, 1, 2),
            )(q64, k64, v64)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(observed, reference, atol=8e-6, rtol=5e-5)


@pytest.mark.parametrize("normalized", [False, True])
def test_default_block_boundaries_causality_and_diagnostics(normalized):
    q, k, v = _inputs(time=QUERY_BLOCK_SIZE + 3)
    output, diag = blocked_dot_product_attention(q, k, v, attention_normalization=normalized, return_diagnostics=True)
    changed = (q.at[:, 127:].multiply(-2), k.at[:, 127:].multiply(2), v.at[:, 127:].add(2))
    modified = blocked_dot_product_attention(*changed, attention_normalization=normalized)
    np.testing.assert_array_equal(modified[:, :127], output[:, :127])
    assert all(value.shape == () for value in diag.values())
    assert float(diag["normalizer_min" if normalized else "weight_sum_min"]) > 0
    assert float(diag["kernel_min"]) > 0
    assert all(int(value) == 0 for name, value in diag.items() if name.endswith("_count"))


def test_zero_scores_are_prefix_sum_not_prefix_mean():
    q, k, v = _inputs()
    actual, diag = blocked_dot_product_attention(
        jnp.zeros_like(q), k, v, query_block_size=4, return_diagnostics=True,
    )
    np.testing.assert_allclose(actual, jnp.cumsum(v, axis=1), atol=2e-6, rtol=2e-6)
    assert float(diag["kernel_min"]) == float(diag["kernel_max"]) == 1
    assert float(diag["weight_sum_min"]) == 1
    assert float(diag["weight_sum_max"]) == v.shape[1]


def test_row_score_offset_multiplies_output_instead_of_cancelling():
    # q=1 turns a constant key shift into exactly the same shift in every score.
    q = jnp.ones((1, 5, 1, 1))
    k = jnp.linspace(-1, 1, 5).reshape(q.shape)
    v = jnp.arange(1, 6, dtype=jnp.float32).reshape(q.shape)
    original = blocked_dot_product_attention(q, k, v)
    shifted = blocked_dot_product_attention(q, k + 2, v)
    np.testing.assert_allclose(shifted, math.exp(2) * original, rtol=2e-6)
    normalized = blocked_dot_product_attention(q, k, v, attention_normalization=True)
    shifted_normalized = blocked_dot_product_attention(q, k + 2, v, attention_normalization=True)
    np.testing.assert_allclose(shifted_normalized, normalized, rtol=2e-6)


def test_masked_future_overflow_is_not_an_exponential_or_diagnostic_entry():
    q = jnp.array([1.0, 0.0]).reshape(1, 2, 1, 1)
    k = jnp.array([0.0, 1e30]).reshape(q.shape)
    v = jnp.array([3.0, 7.0]).reshape(q.shape)
    output, diag = blocked_dot_product_attention(q, k, v, query_block_size=1, return_diagnostics=True)
    np.testing.assert_array_equal(output.reshape(-1), [3, 10])
    assert float(diag["score_max"]) == 0
    assert int(diag["invalid_kernel_count"]) == 0


@pytest.mark.parametrize("bad_score", [100.0, np.inf, -np.inf, np.nan])
def test_overflow_or_nonfinite_scores_are_visible_without_clipping(bad_score):
    q = jnp.ones((1, 3, 1, 1))
    k = jnp.array([0.0, 0.0, bad_score]).reshape(q.shape)
    v = jnp.array([3.0, 7.0, 11.0]).reshape(q.shape)
    output, diag = blocked_dot_product_attention(q, k, v, return_diagnostics=True)
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [3, 10])
    assert bool(jnp.all(jnp.isnan(output[:, 2:])))
    assert int(diag["invalid_output_count"]) == 1
    if np.isfinite(bad_score):
        assert int(diag["invalid_kernel_count"]) == 1
    else:
        assert int(diag["invalid_score_count"]) == 1
    assert int(diag["invalid_weight_sum_count"]) == 0


def test_output_overflow_is_reported_even_when_each_weight_is_finite():
    q = jnp.ones((1, 2, 1, 1))
    _, diag = blocked_dot_product_attention(q, q * 80, q * 1e10, return_diagnostics=True)
    assert int(diag["invalid_kernel_count"]) == 0
    assert int(diag["invalid_output_count"]) == 2


def test_exp_underflow_and_zero_weight_sum_are_allowed():
    q = jnp.ones((1, 2, 1, 1))
    output, diag = blocked_dot_product_attention(q, q * -1000, q, return_diagnostics=True)
    np.testing.assert_array_equal(output, jnp.zeros_like(output))
    assert float(diag["weight_sum_min"]) == float(diag["weight_sum_max"]) == 0
    assert float(diag["kernel_min"]) == 0
    assert all(int(value) == 0 for name, value in diag.items() if name.endswith("_count"))


def test_weight_sum_overflow_does_not_create_a_denominator():
    q = jnp.ones((1, 4, 1, 1))
    output, diag = blocked_dot_product_attention(q, q * 88, q * 1e-20, return_diagnostics=True)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.isinf(diag["weight_sum_max"]))
    assert int(diag["invalid_weight_sum_count"]) == 0


@pytest.mark.parametrize("normalization", [None, 0, 1, "false"])
def test_normalization_is_explicit_boolean(normalization):
    with pytest.raises(ValueError, match="boolean"):
        blocked_dot_product_attention(*_inputs(), attention_normalization=normalization)


def test_normalized_control_matches_original_sdpa_outputs_and_gradients():
    q, k, v = _inputs(time=17)
    control = lambda q, k, v: blocked_dot_product_attention(q, k, v, attention_normalization=True)
    original = lambda q, k, v: padded_dot_product_attention(q, k, v, backend="xla")
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(control(q, k, v), original(q, k, v), atol=5e-7, rtol=3e-5)
        reference = jax.grad(lambda q, k, v: jnp.square(original(q, k, v)).sum(), argnums=(0, 1, 2))(q, k, v)
        observed = jax.grad(lambda q, k, v: jnp.square(control(q, k, v)).sum(), argnums=(0, 1, 2))(q, k, v)
    for actual, expected in zip(observed, reference, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=4e-5)


def test_mha_preserves_parameter_initialization_across_attention_modes():
    settings = dict(hidden_size=16, num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=6)
    layers = [MultiHeadAttention(**settings, attention_normalization=normalized, rngs=nnx.Rngs(19))
              for normalized in (False, True)]
    leaves = [jax.tree.flatten(nnx.state(layer, nnx.Param)) for layer in layers]
    assert leaves[0][1] == leaves[1][1]
    for actual, expected in zip(leaves[0][0], leaves[1][0], strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["manual", "auto", "xla", "cudnn"])
def test_unnormalized_mode_cannot_silently_dispatch_to_softmax(backend):
    with pytest.raises(ValueError, match="Unnormalized attention requires"):
        MultiHeadAttention(16, 2, 2, 4, 4, backend=backend, attention_normalization=False, rngs=nnx.Rngs(19))


@pytest.mark.parametrize("normalized", [False, True])
def test_model_fused_dispatch_retains_real_output_when_collecting_diagnostics(monkeypatch, normalized):
    from flax import nnx
    from scaffolds.transformer.attention import multihead as attention_module
    from scaffolds.transformer.attention.backends import fused_unnormalized as fused_module
    from scaffolds.transformer.attention.backends import reference as reference_module
    from scaffolds.transformer.attention import softmax as softmax_module

    settings = dict(hidden_size=16, num_heads=2, num_kv_heads=2, qk_head_dim=4,
                    value_head_dim=6, attention_normalization=normalized)
    layer = attention_module.MultiHeadAttention(**settings, backend="fused", rngs=nnx.Rngs(19))
    calls = []
    captured_operands = []

    def fast(q, k, v, **kwargs):
        calls.append(("fast", kwargs))
        captured_operands.append((q, k, v))
        return jnp.full_like(v, 3)

    def diagnostics(q, k, v, **kwargs):
        calls.append(("diagnostics", kwargs))
        for actual, expected in zip((q, k, v), captured_operands[-1], strict=True):
            np.testing.assert_array_equal(actual, expected)
        return jnp.full_like(v, -11), {"weight_sum_min": jnp.asarray(7.0)}

    monkeypatch.setattr(fused_module, "fused_unnormalized_attention", fast)
    monkeypatch.setattr(softmax_module, "padded_dot_product_attention", fast)
    monkeypatch.setattr(reference_module, "blocked_dot_product_attention", diagnostics)
    x = jnp.arange(3 * 16, dtype=jnp.float32).reshape(1, 3, 16) / 48
    expected = layer.o_proj(jnp.full((1, 3, 12), 3, dtype=jnp.float32))
    plain = layer(x)
    output, observed_diagnostics = layer(x, return_diagnostics=True)
    assert [kind for kind, _ in calls] == ["fast", "fast", "diagnostics"]
    expected_fast_kwargs = {"backend": "cudnn"} if normalized else {"attention_degree": None}
    assert calls[0][1] == calls[1][1] == expected_fast_kwargs
    assert calls[2][1] == ({"return_diagnostics": True} if normalized else
                           {"attention_degree": None, "attention_normalization": False,
                            "return_diagnostics": True})
    np.testing.assert_array_equal(plain, expected)
    np.testing.assert_array_equal(output, expected)
    assert float(observed_diagnostics["weight_sum_min"]) == 7.0


def _broad_inputs(time=13):
    # Scores with standard deviation near 2 make odd truncations change sign.
    q, k, v = _inputs(time=time)
    return 2 * q, 2 * k, v


@pytest.mark.parametrize("degree", TAYLOR_DEGREES)
def test_unnormalized_taylor_outputs_and_qkv_gradients_match_dense_fp64(degree):
    q, k, v = _broad_inputs()
    cotangent = jnp.asarray(np.random.default_rng(9).normal(size=v.shape), dtype=v.dtype)

    def blocked(q, k, v):
        return blocked_dot_product_attention(q, k, v, attention_degree=degree, query_block_size=4)

    def dense(q, k, v):
        return _dense_reference(q, k, v, attention_degree=degree)

    with jax.default_matmul_precision("highest"):
        actual = blocked(q, k, v)
        actual_grads = jax.grad(lambda q, k, v: jnp.sum(blocked(q, k, v) * cotangent), argnums=(0, 1, 2))(q, k, v)
        with jax.enable_x64():
            q64, k64, v64 = (x.astype(jnp.float64) for x in (q, k, v))
            expected = dense(q64, k64, v64)
            expected_grads = jax.grad(
                lambda q, k, v: jnp.sum(dense(q, k, v) * cotangent.astype(jnp.float64)), argnums=(0, 1, 2),
            )(q64, k64, v64)
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, atol=3e-6 * scale, rtol=3e-5)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        grad_scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(observed, reference, atol=8e-6 * max(grad_scale, 1.0), rtol=5e-5)
    if degree == 0:
        np.testing.assert_array_equal(actual_grads[0], jnp.zeros_like(q))
        np.testing.assert_array_equal(actual_grads[1], jnp.zeros_like(k))


@pytest.mark.parametrize("degree", TAYLOR_DEGREES)
def test_unnormalized_taylor_zero_scores_are_prefix_sum(degree):
    q, k, v = _inputs()
    actual, diag = blocked_dot_product_attention(
        jnp.zeros_like(q), k, v, attention_degree=degree, query_block_size=4, return_diagnostics=True,
    )
    np.testing.assert_allclose(actual, jnp.cumsum(v, axis=1), atol=2e-6, rtol=2e-6)
    assert float(diag["kernel_min"]) == float(diag["kernel_max"]) == 1
    assert float(diag["weight_sum_min"]) == 1
    assert float(diag["weight_sum_max"]) == v.shape[1]


@pytest.mark.parametrize("degree,expected", [(1, [-6.0, 4.0]), (3, [-6.0, -6.0 + 5.0 * 8.0 / 3.0])])
def test_odd_unnormalized_taylor_keeps_signed_weights_and_zero_sums(degree, expected):
    # p_1(-3) = p_3(-3) = -2; with degree one, the second row's weights sum to zero.
    q = jnp.ones((1, 2, 1, 1))
    k = jnp.array([-3.0, 1.0]).reshape(q.shape)
    v = jnp.array([3.0, 5.0]).reshape(q.shape)
    output, diag = blocked_dot_product_attention(
        q, k, v, attention_degree=degree, query_block_size=1, return_diagnostics=True,
    )
    np.testing.assert_allclose(output.reshape(-1), expected, rtol=1e-6)
    assert float(diag["kernel_min"]) == -2
    assert int(diag["negative_kernel_count"]) == 2
    assert all(int(value) == 0 for name, value in diag.items() if name.startswith("invalid_"))
    if degree == 1:
        assert float(diag["min_abs_weight_sum"]) == 0
        assert bool(jnp.isinf(diag["cancellation_ratio_max"]))
    assert "min_abs_normalizer" not in diag


@pytest.mark.parametrize("degree", (2, 4))
def test_even_unnormalized_taylor_kernels_stay_positive(degree):
    q, k, v = _broad_inputs(time=QUERY_BLOCK_SIZE + 3)
    _, diag = blocked_dot_product_attention(q, k, v, attention_degree=degree, return_diagnostics=True)
    assert float(diag["score_min"]) < -2
    assert float(diag["kernel_min"]) > 0
    assert all(int(value) == 0 for name, value in diag.items() if name.endswith("_count"))


def test_unnormalized_taylor_score_offset_does_not_cancel():
    q = jnp.ones((1, 5, 1, 1))
    k = jnp.linspace(-1, 1, 5).reshape(q.shape)
    v = jnp.arange(1, 6, dtype=jnp.float32).reshape(q.shape)
    original = blocked_dot_product_attention(q, k, v, attention_degree=2)
    shifted = blocked_dot_product_attention(q, k + 2, v, attention_degree=2)
    expected = _dense_reference(q, k + 2, v, attention_degree=2)
    np.testing.assert_allclose(shifted, expected, rtol=2e-6)
    assert not np.allclose(shifted, original)


def test_mha_initialization_is_independent_of_the_unnormalized_kernel():
    settings = dict(hidden_size=16, num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=6)
    layers = [MultiHeadAttention(**settings, attention_normalization=False, attention_degree=degree,
                                 rngs=nnx.Rngs(19)) for degree in (None, *TAYLOR_DEGREES)]
    reference, reference_tree = jax.tree.flatten(nnx.state(layers[0], nnx.Param))
    for layer in layers[1:]:
        leaves, tree = jax.tree.flatten(nnx.state(layer, nnx.Param))
        assert tree == reference_tree
        for actual, expected in zip(leaves, reference, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_unnormalized_attention_rejects_learned_quadratic_kernels():
    with pytest.raises(ValueError, match="fixed exp/Taylor kernel"):
        MultiHeadAttention(16, 2, 2, 4, 4, attention_degree=2, quadratic_mode="positive",
                           attention_normalization=False, rngs=nnx.Rngs(19))
