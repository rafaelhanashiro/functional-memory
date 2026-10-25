"""GPU correctness gates for Mosaic GPU query-conditioned polynomial attention (H100 and B300).

Use ``pytest -m gpu tests/test_mosaic_query.py`` inside an H100 or B300 allocation. Outputs
and the q/k/v/coefficient VJPs, at both weight precisions, are compared with an independent
NumPy FP64 reference and with the Pallas-Triton kernel (``fused_query_polynomial``).
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.fused_query_polynomial import fused_query_polynomial_attention
from scaffolds.transformer.attention.backends.mosaic.api import (
    MAX_QUERY_DEGREE, WIDTHS, query_degree_supported, query_polynomial_attention,
)
from scaffolds.transformer.attention.backends.mosaic.common import PRECISIONS
from scaffolds.transformer.tests.test_fused_attention import _assert_relative

DEGREES = (1, 2, 3, 4, 8, 12)
QKV_TOLERANCES = (0.006, 0.025, 0.025, 0.015)  # output, dq, dk, dv: the BF16 bounds of test_fused_attention
COEFFICIENT_TOLERANCE = 0.01
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _inputs(degree, width, *, time=512, heads=2, seed=97):
    rng = np.random.default_rng(seed + degree)
    q, k = (jnp.asarray(0.7 * rng.standard_normal((1, time, heads, width)), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((1, time, heads, 64)), jnp.bfloat16)
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
    coefficients = jnp.asarray(taylor * (1 + 0.5 * np.tanh(rng.standard_normal((1, time, heads, degree + 1)))), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal(v.shape), jnp.bfloat16)
    return (q, k, v, coefficients), cotangent


def _numpy_fp64(operands, cotangent):
    q, k, v = (np.asarray(x.astype(jnp.float32), np.float64) for x in operands[:3])
    c = np.asarray(operands[3], np.float64)
    do = np.asarray(cotangent.astype(jnp.float32), np.float64).transpose(0, 2, 1, 3)  # b h t d
    time, width, degree = q.shape[1], q.shape[-1], c.shape[-1] - 1
    s = np.einsum("bthd,bshd->bhts", q, k) / math.sqrt(width)
    causal = np.tril(np.ones((time, time), bool))
    ch = c.transpose(0, 2, 1, 3)[:, :, :, None, :]  # b h t 1 r
    powers = np.stack([s ** r for r in range(degree + 1)], -1)
    weights = np.where(causal, (ch * powers).sum(-1), 0.0)
    slope = np.where(causal, sum(r * ch[..., r] * s ** (r - 1) for r in range(1, degree + 1)), 0.0)
    out = np.einsum("bhts,bshd->bthd", weights, v)
    dp = np.where(causal, np.einsum("bhtd,bshd->bhts", do, v), 0.0)
    ds = dp * slope
    grads = (np.einsum("bhts,bshd->bthd", ds, k) / math.sqrt(width),
             np.einsum("bhts,bthd->bshd", ds, q) / math.sqrt(width),
             np.einsum("bhts,bhtd->bshd", weights, do),
             np.einsum("bhts,bhtsr->bthr", dp, powers))
    return out, grads


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _assert_all(actual, expected, label):
    _assert_relative(actual[0], expected[0], rtol=QKV_TOLERANCES[0], name=f"{label} output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], QKV_TOLERANCES[1:]):
        _assert_relative(got, want, rtol=tolerance, name=f"{label} d{name}")
    got, want = (np.asarray(x, np.float64) for x in (actual[1][3], expected[1][3]))
    for order in range(want.shape[-1]):  # every coefficient order must train
        _assert_relative(got[..., order:order + 1], want[..., order:order + 1], rtol=COEFFICIENT_TOLERANCE,
                         name=f"{label} dc order {order}")


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("degree", DEGREES)
def test_outputs_and_vjp_match_fp64_and_triton(degree, width, precision):
    operands, cotangent = _inputs(degree, width)
    mosaic = jax.jit(partial(_vjp, partial(query_polynomial_attention, precision=precision)))(operands, cotangent)
    assert mosaic[0].dtype == jnp.bfloat16 and mosaic[1][3].dtype == jnp.float32
    label = f"degree={degree}, width={width}, {precision}"
    _assert_all(mosaic, _numpy_fp64(operands, cotangent), f"Mosaic vs FP64, {label}")
    triton = jax.jit(partial(_vjp, fused_query_polynomial_attention))(operands, cotangent)
    _assert_all(mosaic, triton, f"Mosaic vs Triton, {label}")


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("degree", (1, 3))
def test_production_context_matches_triton(degree, precision):
    operands, cotangent = _inputs(degree, 64, time=2048, heads=4)
    mosaic = jax.jit(partial(_vjp, partial(query_polynomial_attention, precision=precision)))(operands, cotangent)
    triton = jax.jit(partial(_vjp, fused_query_polynomial_attention))(operands, cotangent)
    _assert_all(mosaic, triton, f"production context, degree={degree}, {precision}")


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("width", (64, 128))
def test_largest_degree_fits_the_production_context(width, precision):
    # B300's dK/dV keeps every query's m+1 coefficients in shared memory: the largest case.
    degree = max(m for m in range(1, MAX_QUERY_DEGREE + 1) if query_degree_supported(m, width, 2048))
    operands, cotangent = _inputs(degree, width, time=2048, heads=4)
    mosaic = jax.jit(partial(_vjp, partial(query_polynomial_attention, precision=precision)))(operands, cotangent)
    triton = jax.jit(partial(_vjp, fused_query_polynomial_attention))(operands, cotangent)
    _assert_all(mosaic, triton, f"production context, degree={degree}, width={width}, {precision}")


@pytest.mark.parametrize("precision", PRECISIONS)
def test_outputs_and_vjp_ignore_future_tokens(precision):
    operands, cotangent = _inputs(2, 64)
    prefix = 300
    attend = jax.jit(partial(_vjp, partial(query_polynomial_attention, precision=precision)))
    restricted = cotangent.at[:, prefix:].set(0)
    output, grads = attend(operands, restricted)
    changed = tuple(x.at[:, prefix:].add(2 if index < 3 else 0.5) for index, x in enumerate(operands))
    changed_output, changed_grads = attend(changed, restricted)
    np.testing.assert_array_equal(np.asarray(changed_output[:, :prefix]), np.asarray(output[:, :prefix]))
    for name, grad, changed_grad in zip(("q", "k", "v", "c"), grads, changed_grads):
        np.testing.assert_array_equal(np.asarray(changed_grad), np.asarray(grad), err_msg=name)
        np.testing.assert_array_equal(np.asarray(grad[:, prefix:]), 0, err_msg=name)


def test_future_nonfinite_keys_and_coefficients_do_not_poison_the_prefix():
    (q, k, v, c), _ = _inputs(2, 64)
    prefix = 300
    output = query_polynomial_attention(q, k, v, c)
    poisoned = query_polynomial_attention(q, k.at[:, prefix:].set(jnp.inf), v, c.at[:, prefix:].set(jnp.nan))
    np.testing.assert_array_equal(np.asarray(poisoned[:, :prefix]), np.asarray(output[:, :prefix]))
    assert bool(jnp.isnan(poisoned[:, prefix:]).all())


def test_nonfinite_weight_poisons_its_row_even_with_zero_values():
    (q, k, v, c), _ = _inputs(2, 64)
    v = jnp.zeros_like(v)
    c = c.at[0, 100, 1, 2].set(jnp.inf)
    output = np.asarray(query_polynomial_attention(q, k, v, c).astype(jnp.float32))
    assert np.isnan(output[0, 100, 1]).all()
    finite = np.ones(output.shape[:3], bool)
    finite[0, 100, 1] = False
    assert np.isfinite(output[finite]).all()


@pytest.mark.parametrize("change", ["coefficient dtype", "degree too high", "coefficient shape", "width", "time", "precision"])
def test_unsupported_inputs_raise(change):
    (q, k, v, c), _ = _inputs(2, 64, time=512, heads=1)
    if change == "coefficient dtype":
        c = c.astype(jnp.bfloat16)
    elif change == "degree too high":
        c = jnp.zeros((*c.shape[:3], MAX_QUERY_DEGREE + 2), jnp.float32)
    elif change == "coefficient shape":
        c = c[:, :-1]
    elif change == "width":
        q = k = jnp.zeros((1, 512, 1, 48), jnp.bfloat16)
    elif change == "time":
        q, k, v, c = q[:, :200], k[:, :200], v[:, :200], c[:, :200]
    with pytest.raises(ValueError):
        query_polynomial_attention(q, k, v, c, precision="fp8" if change == "precision" else "bf16")


def test_model_dispatch_uses_the_kernel_and_reports_diagnostics():
    from scaffolds.transformer.attention.core import attention_core

    (q, k, v, coefficients), _ = _inputs(3, 64)
    output, diagnostics = attention_core(
        q, k, v, backend="mosaic", attention_degree=3, attention_normalization=False,
        query_polynomial_coefficients=coefficients, attention_coefficient_activation="tanh", return_diagnostics=True)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(query_polynomial_attention(q, k, v, coefficients)))
    assert diagnostics and all(bool(jnp.all(jnp.isfinite(value))) for value in diagnostics.values())
