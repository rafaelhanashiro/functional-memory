"""GPU correctness gates for the custom polynomial-attention forward and VJP.

Use ``pytest -m gpu tests/test_fused_attention.py`` inside a GPU allocation.
The independent dense reference evaluates a direct power sum in FP64; the
blocked reference uses the original training arithmetic, including BF16 casts.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.polynomial_attention.models.components.attention import (
    blocked_dot_product_attention,
)
from experiments.polynomial_attention.models.components.fused_attention import (
    fused_polynomial_attention,
)
from experiments.polynomial_attention.tests.test_attention import _dense_reference


DEGREES = (0, 2, 4, 8, 16)
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation; the compiled fused kernel is under test")


def _inputs(shape, *, dtype=jnp.float32, seed=91):
    """shape is (batch, time, heads, qk channels, value channels)."""
    batch, time, heads, qk_dim, value_dim = shape
    rng = np.random.default_rng(seed)
    qk_shape = (batch, time, heads, qk_dim)
    value_shape = (batch, time, heads, value_dim)
    q, k = (jnp.asarray(0.7 * rng.standard_normal(qk_shape), dtype=dtype) for _ in range(2))
    v = jnp.asarray(rng.standard_normal(value_shape), dtype=dtype)
    cotangent = jnp.asarray(rng.standard_normal(value_shape), dtype=dtype)
    return (q, k, v), cotangent


def _forward_and_vjp(attend, operands, cotangent):
    output, pullback = jax.vjp(attend, *operands)
    return output, pullback(cotangent)


def _compiled_forward_and_vjp(attend):
    return jax.jit(partial(_forward_and_vjp, attend))


def _fp64_forward_and_vjp(operands, cotangent, degree):
    # Keep the context active through execution: deferred lowering must also
    # retain FP64 rather than silently truncating the independent reference.
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        operands64 = tuple(x.astype(jnp.float64) for x in operands)
        attend = partial(_dense_reference, degree=degree)
        output, grads = _compiled_forward_and_vjp(attend)(operands64, cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), (output, grads))


def _assert_relative(observed, expected, *, rtol, name):
    """Relative L2 per batch/head plus a max-error check, without an atol floor.

    An absolute tolerance on a mean-loss VJP can accept entirely missing q/k
    gradients. These tests instead normalize by the reference's actual norm,
    including the deliberately tiny-cotangent cases. Exact zeros stay exact.
    """
    actual = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    assert actual.shape == reference.shape, name
    assert np.isfinite(actual).all(), f"{name}: nonfinite actual values"
    assert np.isfinite(reference).all(), f"{name}: nonfinite reference values"
    for batch in range(reference.shape[0]):
        for head in range(reference.shape[2]):
            got = actual[batch, :, head, :]
            want = reference[batch, :, head, :]
            norm = np.linalg.norm(want)
            label = f"{name}, batch={batch}, head={head}"
            if norm == 0:
                np.testing.assert_array_equal(got, want, err_msg=label)
                continue
            error = np.linalg.norm(got - want) / norm
            assert error <= rtol, f"{label}: relative L2 error {error:.6g} exceeds {rtol}"
            # L2 alone can hide an isolated bad row in a long context. Three
            # times the norm tolerance bounds the worst element as well.
            max_error = np.max(np.abs(got - want))
            max_limit = 3 * rtol * np.max(np.abs(want))
            assert max_error <= max_limit, f"{label}: max error {max_error:.6g} exceeds {max_limit:.6g}"


def _assert_result(actual, expected, *, output_rtol, gradient_rtols, label):
    _assert_relative(actual[0], expected[0], rtol=output_rtol, name=f"{label} output")
    for index, (observed, reference, tolerance) in enumerate(zip(actual[1], expected[1], gradient_rtols, strict=True)):
        _assert_relative(observed, reference, rtol=tolerance, name=f"{label} d{'qkv'[index]}")


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("shape", [(2, 17, 3, 7, 5), (1, 131, 2, 33, 65)])
def test_fp32_outputs_and_vjp_match_blocked_and_independent_fp64(degree, shape):
    # Odd widths force internal padding; 131 crosses common 32/64/128 tiles.
    # The reference scaling uses the original qk width, never the padded width.
    operands, cotangent = _inputs(shape)
    fused = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))
    # Reference blocks of 32 avoid an XLA layout-normalization bug triggered
    # by larger blocks with odd value widths. Both paths still cross tiles.
    blocked = _compiled_forward_and_vjp(partial(
        blocked_dot_product_attention, attention_degree=degree, query_block_size=32,
    ))
    # Default NVIDIA FP32 matmul may use one TF32 multiply, whose input
    # rounding is less accurate than the fused kernel. Request full reference
    # precision explicitly rather than loosening the correctness tolerance.
    with jax.default_matmul_precision("highest"):
        for scale in (1.0, 1e-5):
            scaled_cotangent = cotangent * scale
            actual = fused(operands, scaled_cotangent)
            assert actual[0].dtype == jnp.float32
            assert all(grad.dtype == jnp.float32 for grad in actual[1])
            reference = blocked(operands, scaled_cotangent)
            dense64 = _fp64_forward_and_vjp(operands, scaled_cotangent, degree)
            for label, expected in (("blocked", reference), ("dense FP64", dense64)):
                _assert_result(actual, expected, output_rtol=3e-5, gradient_rtols=(1e-4,) * 3,
                               label=f"{label}, degree={degree}, cotangent scale={scale}")


@pytest.mark.parametrize("degree", DEGREES)
def test_bf16_production_context_outputs_and_vjp_match_blocked(degree):
    # Sample batch/head axes to keep the correctness gate inexpensive, while
    # retaining the production context length and qk/value channel widths.
    operands, cotangent = _inputs((1, 2048, 4, 64, 64), dtype=jnp.bfloat16, seed=92)
    fused = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))
    blocked = _compiled_forward_and_vjp(partial(blocked_dot_product_attention, attention_degree=degree))
    actual = fused(operands, cotangent)
    reference = blocked(operands, cotangent)
    assert actual[0].dtype == jnp.bfloat16
    assert all(grad.dtype == jnp.bfloat16 for grad in actual[1])
    # Output casts can differ by one BF16 ULP. Both backward paths retain
    # FP32 score cotangents, but differ in reduction order and final BF16
    # gradient rounding. Norm bounds allow these differences without accepting
    # zero gradients.
    _assert_result(actual, reference, output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015),
                   label=f"BF16 production context, degree={degree}")

    # Independently anchor BF16 arithmetic to FP64 on a causal prefix, including
    # q/k/v VJPs. These inputs retain exactly the original BF16-rounded values.
    prefix = tuple(x[:, :129, :1] for x in operands)
    prefix_cotangent = cotangent[:, :129, :1]
    prefix_actual = fused(prefix, prefix_cotangent)
    dense64 = _fp64_forward_and_vjp(prefix, prefix_cotangent, degree)
    _assert_result(prefix_actual, dense64, output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015),
                   label=f"BF16 prefix vs dense FP64, degree={degree}")
    _assert_relative(actual[0][:, :129, :1], prefix_actual[0], rtol=0.006, name="causal prefix across shapes")


@pytest.mark.parametrize("degree", DEGREES)
def test_causal_outputs_and_vjp_ignore_future_tokens(degree):
    operands, cotangent = _inputs((1, 131, 2, 33, 5), dtype=jnp.bfloat16, seed=93)
    prefix = 65
    fused = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))
    restricted_cotangent = cotangent.at[:, prefix:].set(0)
    output, grads = fused(operands, restricted_cotangent)
    q, k, v = operands
    changed = (q.at[:, prefix:].multiply(-2), k.at[:, prefix:].multiply(3), v.at[:, prefix:].add(2))
    changed_output, _ = fused(changed, restricted_cotangent)
    np.testing.assert_array_equal(changed_output[:, :prefix], output[:, :prefix])
    for name, grad in zip("qkv", grads, strict=True):
        assert bool(jnp.all(jnp.isfinite(grad))), name
        np.testing.assert_array_equal(grad[:, prefix:], jnp.zeros_like(grad[:, prefix:]), err_msg=f"future d{name}")


def test_degree_zero_is_prefix_mean_with_exact_zero_qk_vjp():
    operands, cotangent = _inputs((2, 131, 3, 7, 5))
    output, grads = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=0))(operands, cotangent)
    expected = jnp.cumsum(operands[2], axis=1) / jnp.arange(1, 132)[None, :, None, None]
    _assert_relative(output, expected, rtol=2e-6, name="degree-zero prefix mean")
    np.testing.assert_array_equal(grads[0], jnp.zeros_like(operands[0]))
    np.testing.assert_array_equal(grads[1], jnp.zeros_like(operands[1]))
    assert np.linalg.norm(np.asarray(grads[2])) > 0


@pytest.mark.parametrize("degree", (2, 4, 8, 16))
def test_negative_score_cancellation_outputs_and_vjp_match_fp64(degree):
    scores = np.concatenate(([-5.2, -5.19, -5.04], np.linspace(-20, 20, 161))).astype(np.float32)
    q = jnp.ones((1, len(scores), 1, 1), dtype=jnp.float32)
    k = jnp.asarray(scores).reshape(q.shape)
    positions = jnp.arange(len(scores), dtype=jnp.float32)
    v = jnp.stack((jnp.sin(positions), jnp.cos(positions)), axis=-1)[None, :, None, :]
    cotangent = jnp.stack((jnp.cos(positions * 0.3), jnp.sin(positions * 0.7)), axis=-1)[None, :, None, :]
    operands = (q, k, v)
    actual = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))(operands, cotangent)
    dense64 = _fp64_forward_and_vjp(operands, cotangent, degree)
    # Horner cancellation around negative scores is intrinsically less accurate
    # than the well-conditioned random cases; this matches the existing kernel
    # stress tolerance while checking its separately derived backward as well.
    _assert_result(actual, dense64, output_rtol=1e-3, gradient_rtols=(3e-3,) * 3,
                   label=f"negative-score cancellation, degree={degree}")


def test_negative_scores_use_unshifted_taylor_kernel():
    q = jnp.ones((1, 2, 1, 1), dtype=jnp.float32)
    k = jnp.asarray([-5.0, 0.0], dtype=jnp.float32).reshape(q.shape)
    v = jnp.asarray([1.0, 0.0], dtype=jnp.float32).reshape(q.shape)
    output = jax.jit(partial(fused_polynomial_attention, attention_degree=2))(q, k, v)
    np.testing.assert_allclose(output[0, 1, 0, 0], 8.5 / 9.5, atol=2e-7, rtol=2e-7)


def test_masked_future_overflow_does_not_poison_valid_rows():
    q = jnp.asarray([1.0, 0.0], dtype=jnp.float32).reshape(1, 2, 1, 1)
    k = jnp.asarray([0.0, 1e30], dtype=jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0], dtype=jnp.float32).reshape(q.shape)
    output = jax.jit(partial(fused_polynomial_attention, attention_degree=16))(q, k, v)
    np.testing.assert_array_equal(output.reshape(-1), [3.0, 5.0])


@pytest.mark.parametrize("degree,bad_score", [(16, 1e30), (16, np.inf), (16, np.nan), (0, np.inf), (0, np.nan)])
def test_invalid_causal_scores_or_kernel_poison_only_affected_rows(degree, bad_score):
    q = jnp.ones((1, 3, 1, 1), dtype=jnp.float32)
    k = jnp.asarray([0.0, 0.0, bad_score], dtype=jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0, 11.0], dtype=jnp.float32).reshape(q.shape)
    output = jax.jit(partial(fused_polynomial_attention, attention_degree=degree))(q, k, v)
    reference = blocked_dot_product_attention(q, k, v, attention_degree=degree)
    np.testing.assert_array_equal(jnp.isnan(output), jnp.isnan(reference))
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [3.0, 5.0])
    assert bool(jnp.all(jnp.isnan(output[:, 2:])))


@pytest.mark.parametrize("degree,score_scale", [(2, 1e10), (4, 1e5), (8, 1e3), (16, 1e2)])
def test_finite_large_kernel_and_values_do_not_overflow_numerator(degree, score_scale):
    # Each kernel and its row sum are finite, and the correct convex combination
    # of V is finite. Forming sum(kernel * V) before dividing would overflow FP32.
    # A one-dimensional query only rescales all scores; at large positive
    # scores p_m(s) is nearly a monomial, so that radial derivative nearly
    # cancels under normalization. A second key component gives a substantial
    # transverse query derivative without changing the score magnitudes, so
    # strict relative VJP checks measure arithmetic accuracy rather than that
    # ill-conditioned cancellation.
    q = jnp.tile(jnp.asarray([2.0 ** 0.5, 0.0], dtype=jnp.float32), (1, 3, 1, 1))
    k = (jnp.asarray([[0.5, -0.25], [0.75, 0.5], [1.0, -0.75]], dtype=jnp.float32)
         * score_scale)[None, :, None, :]
    v = (jnp.asarray([[1.0, -2.0], [-3.0, 2.0], [2.0, 1.0]], dtype=jnp.float32) * 1e25)[None, :, None, :]
    cotangent = jnp.asarray([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.7]], dtype=jnp.float32)[None, :, None, :]
    with jax.default_matmul_precision("highest"):
        actual = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))(
            (q, k, v), cotangent,
        )
        blocked = jax.jit(partial(blocked_dot_product_attention, attention_degree=degree))(q, k, v)
    dense64 = _fp64_forward_and_vjp((q, k, v), cotangent, degree)
    reference_dq = dense64[1][0]
    assert np.linalg.norm(reference_dq[..., 1]) >= 0.1 * np.linalg.norm(reference_dq)
    _assert_relative(actual[0], blocked, rtol=3e-5, name=f"large finite kernel/V vs blocked, degree={degree}")
    # The backward must form derivative/kernel_sum before multiplying by dP;
    # dP*derivative can overflow even when the correct q/k gradients are finite.
    _assert_result(actual, dense64, output_rtol=3e-5, gradient_rtols=(1e-4,) * 3,
                   label=f"large finite kernel/V vs FP64, degree={degree}")


@pytest.mark.parametrize("degree", (None, 0, 16))
def test_model_fused_dispatch_retains_real_output_when_collecting_diagnostics(monkeypatch, degree):
    from flax import nnx
    from experiments.polynomial_attention.models.components import attention as attention_module
    from experiments.polynomial_attention.models.components import fused_attention as fused_module

    settings = dict(hidden_size=16, num_heads=2, num_kv_heads=2, qk_head_dim=4,
                    value_head_dim=6, attention_degree=degree)
    layer = attention_module.MultiHeadAttention(**settings, backend="fused", rngs=nnx.Rngs(19))
    reference = attention_module.MultiHeadAttention(**settings, backend="blocked", rngs=nnx.Rngs(19))
    actual_params, actual_tree = jax.tree.flatten(nnx.state(layer, nnx.Param))
    expected_params, expected_tree = jax.tree.flatten(nnx.state(reference, nnx.Param))
    assert actual_tree == expected_tree
    for actual, expected in zip(actual_params, expected_params, strict=True):
        np.testing.assert_array_equal(actual, expected)

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
        # A distinctive reference output must never become the actual layer
        # output: later layers need diagnostics on the real fused trajectory.
        return jnp.full_like(v, -11), {"normalizer_min": jnp.asarray(7.0)}

    monkeypatch.setattr(fused_module, "fused_polynomial_attention", fast)
    monkeypatch.setattr(attention_module, "padded_dot_product_attention", fast)
    monkeypatch.setattr(attention_module, "blocked_dot_product_attention", diagnostics)
    x = jnp.arange(3 * 16, dtype=jnp.float32).reshape(1, 3, 16) / 48
    expected = layer.o_proj(jnp.full((1, 3, 12), 3, dtype=jnp.float32))
    plain = layer(x)
    assert len(calls) == 1
    output, observed_diagnostics = layer(x, return_diagnostics=True)
    assert [kind for kind, _ in calls] == ["fast", "fast", "diagnostics"]
    expected_fast_kwargs = {"backend": "cudnn"} if degree is None else {"attention_degree": degree}
    assert calls[0][1] == calls[1][1] == expected_fast_kwargs
    assert calls[2][1] == {"attention_degree": degree, "return_diagnostics": True}
    np.testing.assert_array_equal(plain, expected)
    np.testing.assert_array_equal(output, expected)
    assert float(observed_diagnostics["normalizer_min"]) == 7.0
