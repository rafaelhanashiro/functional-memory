"""GPU gates for the fused unnormalized kernel with learned per-head polynomial coefficients."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.fused_learned_polynomial import fused_learned_polynomial_attention
from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.tests.test_learned_polynomial import _dense_reference


pytestmark = pytest.mark.gpu
DEGREES = (1, 2, 3, 4)


@pytest.fixture(autouse=True)
def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation; the compiled fused kernel is under test")


def _inputs(shape, degree, *, dtype=jnp.float32, seed=301, score_scale=1.0, taylor=False):
    batch, time, heads, qk_dim, value_dim = shape
    rng = np.random.default_rng(seed + degree)
    q, k = (jnp.asarray(score_scale * 0.7 * rng.standard_normal((batch, time, heads, qk_dim)), dtype=dtype)
            for _ in range(2))
    v, cotangent = (jnp.asarray(rng.standard_normal((batch, time, heads, value_dim)), dtype=dtype) for _ in range(2))
    factorials = np.array([math.factorial(r) for r in range(degree + 1)])
    # Distinct signed per-head coefficients expose shared loads and wrong reductions.
    coefficients = 1 / factorials if taylor else rng.normal(size=(heads, degree + 1)) / factorials
    coefficients = jnp.asarray(np.broadcast_to(coefficients, (heads, degree + 1)), jnp.float32)
    return (q, k, v, coefficients), cotangent


def _blocked(q, k, v, coefficients, *, query_block_size=128):
    return blocked_dot_product_attention(
        q, k, v, attention_degree=coefficients.shape[-1] - 1, attention_normalization=False,
        polynomial_coefficients=coefficients, query_block_size=query_block_size)


def _vjp(attend, operands, cotangent):
    output, pullback = jax.vjp(attend, *operands)
    return output, pullback(cotangent)


def _compiled(attend):
    return jax.jit(partial(_vjp, attend))


def _fp64(operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _compiled(_dense_reference)(tuple(x.astype(jnp.float64) for x in operands),
                                             cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), result)


def _assert_relative(observed, expected, *, rtol, name):
    """Per-head relative L2 and max errors, so a small or missing gradient cannot pass."""
    actual, reference = (np.asarray(x, dtype=np.float64) for x in (observed, expected))
    assert actual.shape == reference.shape, name
    assert np.isfinite(actual).all(), f"{name}: nonfinite actual values"
    assert np.isfinite(reference).all(), f"{name}: nonfinite reference values"
    views = ([(f"head={h}", actual[h], reference[h]) for h in range(reference.shape[0])] if reference.ndim == 2 else
             [(f"batch={b}, head={h}", actual[b, :, h], reference[b, :, h])
              for b in range(reference.shape[0]) for h in range(reference.shape[2])])
    for label, got, want in views:
        norm = np.linalg.norm(want)
        if norm == 0:
            np.testing.assert_array_equal(got, want, err_msg=f"{name}, {label}")
            continue
        error = np.linalg.norm(got - want) / norm
        assert error <= rtol, f"{name}, {label}: relative L2 error {error:.6g} exceeds {rtol}"
        assert np.max(np.abs(got - want)) <= 3 * rtol * np.max(np.abs(want)), f"{name}, {label}: max error"


def _assert_result(actual, expected, *, output_rtol, gradient_rtols, label):
    _assert_relative(actual[0], expected[0], rtol=output_rtol, name=f"{label} output")
    for name, observed, reference, tolerance in zip("qkvc", actual[1], expected[1], gradient_rtols, strict=True):
        _assert_relative(observed, reference, rtol=tolerance, name=f"{label} d{name}")


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("shape", [(2, 17, 3, 7, 5), (1, 131, 2, 33, 65)])
def test_fp32_outputs_and_all_gradients_match_blocked_and_fp64(shape, degree):
    # Doubled scores and signed coefficients change the kernel's sign inside every tile.
    operands, cotangent = _inputs(shape, degree, score_scale=2.0)
    fused = _compiled(fused_learned_polynomial_attention)
    blocked = _compiled(partial(_blocked, query_block_size=32))
    with jax.default_matmul_precision("highest"):
        actual = fused(operands, cotangent)
        assert actual[0].dtype == jnp.float32
        assert actual[1][3].dtype == jnp.float32 and actual[1][3].shape == operands[3].shape
        for label, expected in (("blocked", blocked(operands, cotangent)), ("dense FP64", _fp64(operands, cotangent))):
            _assert_result(actual, expected, output_rtol=3e-5, gradient_rtols=(1e-4,) * 4,
                           label=f"{label}, degree={degree}")


@pytest.mark.parametrize("degree", DEGREES)
def test_bf16_production_context_matches_blocked_and_fp64_prefix(degree):
    operands, cotangent = _inputs((1, 2048, 4, 64, 64), degree, dtype=jnp.bfloat16, seed=92)
    fused = _compiled(fused_learned_polynomial_attention)
    actual = fused(operands, cotangent)
    assert actual[0].dtype == jnp.bfloat16
    assert all(grad.dtype == jnp.bfloat16 for grad in actual[1][:3])
    tolerances = dict(output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015, 0.01))
    _assert_result(actual, _compiled(_blocked)(operands, cotangent), **tolerances,
                   label=f"BF16 production context, degree={degree}")
    prefix = (*(x[:, :129, :1] for x in operands[:3]), operands[3][:1])
    prefix_actual = fused(prefix, cotangent[:, :129, :1])
    _assert_result(prefix_actual, _fp64(prefix, cotangent[:, :129, :1]), **tolerances,
                   label=f"BF16 prefix vs dense FP64, degree={degree}")


@pytest.mark.parametrize("degree", DEGREES)
def test_taylor_coefficients_reproduce_the_fixed_unnormalized_kernel(degree):
    operands, cotangent = _inputs((1, 2048, 4, 64, 64), degree, dtype=jnp.bfloat16, seed=94, taylor=True)
    learned = _compiled(fused_learned_polynomial_attention)(operands, cotangent)
    fixed = _compiled(partial(fused_unnormalized_attention, attention_degree=degree))(operands[:3], cotangent)
    _assert_relative(learned[0], fixed[0], rtol=0.006, name="output")
    for name, observed, reference in zip("qkv", learned[1][:3], fixed[1], strict=True):
        _assert_relative(observed, reference, rtol=0.015, name=f"d{name}")


@pytest.mark.parametrize("degree", (1, 4))
def test_causal_outputs_and_all_gradients_ignore_future_tokens(degree):
    (q, k, v, coefficients), cotangent = _inputs((1, 131, 2, 33, 5), degree, dtype=jnp.bfloat16, seed=93)
    prefix = 65
    fused = _compiled(fused_learned_polynomial_attention)
    restricted = cotangent.at[:, prefix:].set(0)
    output, grads = fused((q, k, v, coefficients), restricted)
    changed = (q.at[:, prefix:].multiply(-2), k.at[:, prefix:].multiply(3), v.at[:, prefix:].add(2), coefficients)
    changed_output, changed_grads = fused(changed, restricted)
    np.testing.assert_array_equal(changed_output[:, :prefix], output[:, :prefix])
    np.testing.assert_array_equal(changed_grads[3], grads[3])
    for name, grad in zip("qkv", grads[:3], strict=True):
        assert bool(jnp.all(jnp.isfinite(grad))), name
        np.testing.assert_array_equal(grad[:, prefix:], jnp.zeros_like(grad[:, prefix:]), err_msg=f"future d{name}")


@pytest.mark.parametrize("degree", (1, 3))
def test_zero_scores_give_constant_prefix_sums_and_only_constant_gradient(degree):
    (q, k, v, coefficients), cotangent = _inputs((2, 131, 3, 7, 5), degree)
    output, grads = _compiled(fused_learned_polynomial_attention)((jnp.zeros_like(q), k, v, coefficients), cotangent)
    constant = coefficients[:, 0][None, None, :, None]
    _assert_relative(output, constant * jnp.cumsum(v, axis=1), rtol=2e-6, name="constant prefix sum")
    reverse = jnp.flip(jnp.cumsum(jnp.flip(cotangent, axis=1), axis=1), axis=1)
    _assert_relative(grads[2], constant * reverse, rtol=2e-6, name="dV")
    np.testing.assert_array_equal(grads[1], jnp.zeros_like(k))
    expected_constant = jnp.einsum("bthd,bthd->h", reverse, v)
    _assert_relative(grads[3][:, :1], expected_constant[:, None], rtol=2e-6, name="dc0")
    np.testing.assert_array_equal(grads[3][:, 1:], jnp.zeros_like(grads[3][:, 1:]))


def test_signed_weights_and_negative_constant_stay_finite():
    q = jnp.ones((1, 2, 1, 1), jnp.float32)
    k = jnp.asarray([-3.0, 1.0], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 5.0], jnp.float32).reshape(q.shape)
    coefficients = jnp.asarray([[-1.0, 2.0]], jnp.float32)  # p(s) = -1 + 2s
    output = jax.jit(fused_learned_polynomial_attention)(q, k, v, coefficients)
    np.testing.assert_allclose(np.asarray(output).reshape(-1), [-21.0, -16.0], rtol=1e-6)
    np.testing.assert_allclose(output, _blocked(q, k, v, coefficients), rtol=1e-6)


@pytest.mark.parametrize("bad_score", [np.inf, -np.inf, np.nan, 1e30])
def test_invalid_scores_or_kernel_values_poison_only_affected_rows(bad_score):
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    k = jnp.asarray([0.0, 0.0, bad_score], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0, 11.0], jnp.float32).reshape(q.shape)
    coefficients = jnp.asarray([[1.0, 1.0, 0.5]], jnp.float32)
    output = jax.jit(fused_learned_polynomial_attention)(q, k, v, coefficients)
    reference = _blocked(q, k, v, coefficients)
    np.testing.assert_array_equal(jnp.isnan(output), jnp.isnan(reference))
    np.testing.assert_array_equal(np.asarray(output[:, :2]).reshape(-1), [3.0, 10.0])
    assert bool(jnp.all(jnp.isnan(output[:, 2:])))


def test_bf16_scores_at_production_magnitudes_match_fp64():
    # Fixed-coefficient unnormalized runs reached |scores| near 50 at degree 4.
    operands, cotangent = _inputs((1, 131, 2, 64, 17), 4, dtype=jnp.bfloat16, seed=184, score_scale=5.0)
    scores = jnp.einsum("bthd,bshd->bhts", *(x.astype(jnp.float32) for x in operands[:2])) / 8
    assert float(jnp.max(jnp.abs(scores))) > 40
    actual = _compiled(fused_learned_polynomial_attention)(operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent), output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015, 0.01),
                   label="BF16 large scores vs FP64")
