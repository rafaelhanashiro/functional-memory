"""GPU correctness gates for unnormalized static-coefficient attention on Mosaic GPU (H100 and B300).

Fixed Taylor and learned (shared or per-head) coefficients run on the query-coefficient
kernel with every query's coefficients equal. Use ``pytest -m gpu tests/test_mosaic_unnormalized.py``
inside an H100 or B300 allocation. Outputs and the q/k/v/coefficient VJPs are compared with an
independent NumPy FP64 reference and with the Pallas-Triton kernels.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.unnormalized import unnormalized_attention
from scaffolds.transformer.tests.test_fused_attention import _assert_relative
from scaffolds.transformer.tests.test_mosaic_query import COEFFICIENT_TOLERANCE, QKV_TOLERANCES, _numpy_fp64

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _inputs(degree, width, coefficients, *, time=512, heads=2, seed=99):
    rng = np.random.default_rng(seed + degree)
    q, k = (jnp.asarray(0.7 * rng.standard_normal((1, time, heads, width)), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((1, time, heads, 64)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.standard_normal(v.shape), jnp.bfloat16)
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
    if coefficients == "per_head":
        c = jnp.asarray(taylor * (1 + 0.5 * np.tanh(rng.standard_normal((heads, degree + 1)))), jnp.float32)
    elif coefficients == "shared":
        c = jnp.asarray(taylor * (1 + 0.5 * np.tanh(rng.standard_normal(degree + 1))), jnp.float32)
    else:
        c = None
    return (q, k, v), c, cotangent


def _attend(backend, degree, operands, c):
    if c is None:
        return jax.jit(lambda q, k, v: unnormalized_attention(q, k, v, attention_degree=degree, backend=backend))
    return jax.jit(lambda q, k, v, c: unnormalized_attention(q, k, v, attention_degree=degree, backend=backend,
                                                              polynomial_coefficients=c))


def _forward_and_vjp(backend, degree, operands, c, cotangent):
    args = operands if c is None else (*operands, c)
    output, pullback = jax.vjp(_attend(backend, degree, operands, c), *args)
    return output, pullback(cotangent)


def _fp64(operands, c, cotangent, degree):
    """FP64 reference: the per-query reference with equal coefficients, its dc summed back."""
    batch, time, heads = operands[0].shape[:3]
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)], np.float32)
    static = taylor if c is None else np.asarray(c)
    per_query = jnp.asarray(np.broadcast_to(np.broadcast_to(static, (heads, degree + 1)), (batch, time, heads, degree + 1)))
    out, (dq, dk, dv, dc) = _numpy_fp64((*operands, per_query), cotangent)
    if c is None:
        return out, (dq, dk, dv)
    dc = dc.sum(axis=(0, 1))
    return out, (dq, dk, dv, dc if c.ndim == 2 else dc.sum(axis=0))


def _assert_all(actual, expected, label):
    _assert_relative(actual[0], expected[0], rtol=QKV_TOLERANCES[0], name=f"{label} output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], QKV_TOLERANCES[1:]):
        _assert_relative(got, want, rtol=tolerance, name=f"{label} d{name}")
    if len(expected[1]) == 4:
        got, want = (np.asarray(x, np.float64).reshape(-1, np.shape(x)[-1]) for x in (actual[1][3], expected[1][3]))
        for order in range(want.shape[-1]):  # every coefficient order must train
            error = np.linalg.norm(got[:, order] - want[:, order]) / np.linalg.norm(want[:, order])
            assert error <= COEFFICIENT_TOLERANCE, f"{label} dc order {order}: relative error {error:.3g}"


@pytest.mark.parametrize("coefficients", ["fixed", "per_head", "shared"])
@pytest.mark.parametrize("width", (16, 64))
@pytest.mark.parametrize("degree", (1, 2, 3, 4, 8, 12))
def test_outputs_and_vjp_match_fp64_and_triton(degree, width, coefficients):
    operands, c, cotangent = _inputs(degree, width, coefficients)
    mosaic = _forward_and_vjp("mosaic", degree, operands, c, cotangent)
    assert mosaic[0].dtype == jnp.bfloat16
    label = f"degree={degree}, width={width}, {coefficients}"
    _assert_all(mosaic, _fp64(operands, c, cotangent, degree), f"Mosaic vs FP64, {label}")
    _assert_all(mosaic, _forward_and_vjp("fused", degree, operands, c, cotangent), f"Mosaic vs Triton, {label}")


@pytest.mark.parametrize("degree", (3, 8, 12))
@pytest.mark.parametrize("coefficients", ["fixed", "per_head"])
def test_production_context_matches_triton(coefficients, degree):
    operands, c, cotangent = _inputs(degree, 64, coefficients, time=2048, heads=4)
    mosaic = _forward_and_vjp("mosaic", degree, operands, c, cotangent)
    _assert_all(mosaic, _forward_and_vjp("fused", degree, operands, c, cotangent),
                f"production context, degree={degree}, {coefficients}")


def test_model_dispatch_uses_the_kernel_and_reports_diagnostics():
    from scaffolds.transformer.attention.core import attention_core

    (q, k, v), c, _ = _inputs(2, 64, "per_head")
    output, diagnostics = attention_core(q, k, v, backend="mosaic", attention_degree=2, attention_normalization=False,
                                         polynomial_coefficients=c, return_diagnostics=True)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(
        unnormalized_attention(q, k, v, attention_degree=2, backend="mosaic", polynomial_coefficients=c)))
    assert diagnostics and all(bool(jnp.all(jnp.isfinite(value))) for value in diagnostics.values())


def test_exponential_kernel_is_rejected():
    (q, k, v), _, _ = _inputs(1, 64, "fixed")
    with pytest.raises(ValueError):
        unnormalized_attention(q, k, v, attention_degree=None, backend="mosaic")
