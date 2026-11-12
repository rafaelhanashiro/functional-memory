"""GPU correctness gates for the Mosaic transformed query-polynomial kernels (H100 and B300).

``mosaic.api.transformed_query_polynomial_attention`` evaluates the tanh query polynomial at z = phi(s), phi =
softplus (ln(1 + e^s)) or abs, inside the query polynomial's causal tiles, with the chain rule through phi in both
gradient passes. Use ``pytest -m gpu tests/test_mosaic_score_transform.py`` inside an H100 or B300 allocation with
``JAX_PLATFORMS=cuda,cpu`` (acceptance's GPU stage), which gives the references the host CPU. Outputs and the
q/k/v/coefficient VJPs, at both weight precisions, are compared with an independent NumPy FP64 reference written
here (every pair, explicit phi(s)^r, phi' = the logistic function or sign(s)) under the query kernels' bounds
(``test_mosaic_query``): relative L2 0.006 (output), 0.025 (dQ, dK), 0.015 (dV), and 0.01 for the FP32
coefficient gradient order by order, so that no order's error hides behind the others'. Scores are unit-scale or
wide (N(0, 9): a sixth below -3, where softplus is close to e^s, and some beyond +-12), and coefficients signed,
N(0, 1)/r!. At the production context (2,048 tokens, d_K = 64, degree eight) the production dispatch,
``attention_core`` on backend mosaic, is also compared with the blocked backend's causal strips at production
precision, the computation the 12-layer width-640 transform runs trained on (``test_score_transform_strips``
qualifies them within 1.2 times BF16's rounding floor of FP64), under the same bounds.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api
from scaffolds.transformer.attention.backends.mosaic.common import PRECISIONS
from scaffolds.transformer.tests.test_fused_attention import _assert_relative

pytestmark = pytest.mark.gpu
TRANSFORMS = ("softplus", "abs")
DEGREES = (1, 2, 4, 8, 12)
SPREADS = {"unit": 1.0, "wide": 3.0}
QKV_TOLERANCES = (0.006, 0.025, 0.025, 0.015)  # output, dq, dk, dv: the query kernels' BF16 bounds
COEFFICIENT_TOLERANCE = 0.01


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _inputs(degree, width, spread, *, time=512, heads=2, seed=31):
    """BF16 q, k, v with s = q.k / sqrt(d_K) ~ N(0, spread^2), signed FP32 coefficients N(0, 1)/r!, and a BF16
    cotangent."""
    rng = np.random.default_rng(seed + 13 * degree + width)
    q = jnp.asarray(SPREADS[spread] * rng.standard_normal((1, time, heads, width)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((1, time, heads, width)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((1, time, heads, 64)), jnp.bfloat16)
    factorials = np.array([math.factorial(r) for r in range(degree + 1)])
    coefficients = jnp.asarray(rng.standard_normal((1, time, heads, degree + 1)) / factorials, jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal(v.shape), jnp.bfloat16)
    return (q, k, v, coefficients), cotangent


def _phi(transform, s):
    if transform == "softplus":
        return np.logaddexp(0., s), np.exp(-np.logaddexp(0., -s))
    return np.abs(s), np.sign(s)


def _numpy_fp64(transform, operands, cotangent):
    """Output and VJP of sum_i sum_r c_r(t) phi(s_ti)^r v_i, every power explicit, order by order."""
    q, k, v = (np.asarray(x.astype(jnp.float32), np.float64) for x in operands[:3])
    c = np.asarray(operands[3], np.float64).transpose(0, 2, 1, 3)  # b h t r
    do = np.asarray(cotangent.astype(jnp.float32), np.float64)
    time, width, degree = q.shape[1], q.shape[-1], c.shape[-1] - 1
    causal = np.tril(np.ones((time, time), bool))
    s = np.einsum("bthd,bshd->bhts", q, k, optimize=True) / math.sqrt(width)
    z, slope = _phi(transform, s)
    dp = np.where(causal, np.einsum("bthd,bshd->bhts", do, v, optimize=True), 0.)
    weights, derivative, dc = np.zeros_like(s), np.zeros_like(s), []
    for r in range(degree + 1):
        power = np.where(causal, z ** r, 0.)
        weights += c[..., r, None] * power
        if r:
            derivative += r * c[..., r, None] * np.where(causal, z ** (r - 1), 0.)
        dc.append((dp * power).sum(-1))
    ds = dp * derivative * slope
    output = np.einsum("bhts,bshd->bthd", weights, v, optimize=True)
    grads = (np.einsum("bhts,bshd->bthd", ds, k, optimize=True) / math.sqrt(width),
             np.einsum("bhts,bthd->bshd", ds, q, optimize=True) / math.sqrt(width),
             np.einsum("bhts,bthd->bshd", weights, do, optimize=True),
             np.stack(dc, -1).transpose(0, 2, 1, 3))
    return output, grads


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _mosaic(transform, precision=None):
    return partial(api.transformed_query_polynomial_attention, transform=transform, precision=precision)


def _assert_all(actual, expected, label):
    _assert_relative(actual[0], expected[0], rtol=QKV_TOLERANCES[0], name=f"{label} output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], QKV_TOLERANCES[1:]):
        _assert_relative(got, want, rtol=tolerance, name=f"{label} d{name}")
    got, want = (np.asarray(x, np.float64) for x in (actual[1][3], expected[1][3]))
    for order in range(want.shape[-1]):  # every coefficient order must train
        _assert_relative(got[..., order:order + 1], want[..., order:order + 1], rtol=COEFFICIENT_TOLERANCE,
                         name=f"{label} dc order {order}")


def _errors(actual, expected):
    """Relative L2 of output, dQ, dK, dV and the largest order's of the coefficient gradient, for the record."""
    relative = lambda a, b: float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64))
                                  / np.linalg.norm(np.asarray(b, np.float64)))
    got, want = np.asarray(actual[1][3], np.float64), np.asarray(expected[1][3], np.float64)
    return {"output": relative(actual[0], expected[0]),
            **{f"d{name}": relative(a, b) for name, a, b in zip("qkv", actual[1][:3], expected[1][:3])},
            "dc": max(relative(got[..., r], want[..., r]) for r in range(want.shape[-1]))}


@pytest.mark.parametrize("spread", tuple(SPREADS))
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_outputs_and_vjp_match_fp64(transform, degree, precision, spread):
    operands, cotangent = _inputs(degree, 64, spread)
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, precision)))(operands, cotangent)
    assert mosaic[0].dtype == jnp.bfloat16 and mosaic[1][3].dtype == jnp.float32
    expected = _numpy_fp64(transform, operands, cotangent)
    label = f"{transform}, degree={degree}, {precision}, {spread} scores"
    _assert_all(mosaic, expected, label)
    print(f"\n{label}: " + " ".join(f"{k}={v:.2e}" for k, v in _errors(mosaic, expected).items()))


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("width", (8, 16, 32, 128))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_every_key_width_matches_fp64(transform, width, precision):
    """Every Q/K width the query kernels take (8 zero-padded to 16, narrower swizzles), at degree eight."""
    operands, cotangent = _inputs(8, width, "unit")
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, precision)))(operands, cotangent)
    _assert_all(mosaic, _numpy_fp64(transform, operands, cotangent), f"{transform}, width={width}, {precision}")


def _strips(transform):
    from scaffolds.transformer.attention.core import attention_core

    def attend(q, k, v, c):
        return attention_core(q, k, v, backend="blocked", blocked_implementation="strips", attention_degree=8,
                              attention_normalization=False, query_polynomial_coefficients=c,
                              attention_coefficient_activation="tanh", score_transform=transform)
    return attend


def _production(transform):
    from scaffolds.transformer.attention.core import attention_core

    def attend(q, k, v, c):
        return attention_core(q, k, v, backend="mosaic", attention_degree=8, attention_normalization=False,
                              query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                              score_transform=transform)
    return attend


@pytest.mark.parametrize("spread", tuple(SPREADS))
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_production_context_matches_fp64_and_the_strips(transform, precision, spread, monkeypatch):
    """2,048 tokens, d_K = d_V = 64, degree eight, four heads: the production dispatch (attention_core on backend
    mosaic, which must reach the transformed kernels) against host FP64 and against the causal strips at production
    precision (BF16 Q, K, V, FP32 coefficients), both under the query kernels' bounds: the strips lie within 1.2
    BF16 rounding floors of FP64, far inside them, so each bound also holds between the two implementations."""
    calls = []
    original = api.transformed_query_polynomial_attention

    def counted(*args, **kwargs):
        calls.append(kwargs["transform"])
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    operands, cotangent = _inputs(8, 64, spread, time=2048, heads=4)
    production = jax.jit(partial(_vjp, _production(transform)))(operands, cotangent)
    assert calls == [transform], "attention_core must reach the Mosaic transformed kernels"
    if precision == "bf16":  # the production dispatch uses the kernels' default precision
        direct = jax.jit(partial(_vjp, _mosaic(transform)))(operands, cotangent)
        for got, want in zip(jax.tree.leaves(production), jax.tree.leaves(direct), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, precision)))(operands, cotangent)
    label = f"{transform}, production context, {precision}, {spread} scores"
    expected = _numpy_fp64(transform, operands, cotangent)
    _assert_all(mosaic, expected, f"{label} vs FP64")
    strips = jax.jit(partial(_vjp, _strips(transform)))(operands, cotangent)
    _assert_all(mosaic, strips, f"{label} vs strips")
    print(f"\n{label}: vs FP64 " + " ".join(f"{k}={v:.2e}" for k, v in _errors(mosaic, expected).items())
          + " | vs strips " + " ".join(f"{k}={v:.2e}" for k, v in _errors(mosaic, strips).items())
          + " | strips vs FP64 " + " ".join(f"{k}={v:.2e}" for k, v in _errors(strips, expected).items()))


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("width", (64, 128))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_largest_degree_fits_the_production_context(transform, width, precision):
    """B300's dK/dV keeps every query's m+1 coefficients in shared memory: the largest supported degree."""
    degree = max(m for m in range(1, api.MAX_QUERY_DEGREE + 1)
                 if api.query_degree_supported(m, width, 2048, transform=transform))
    operands, cotangent = _inputs(degree, width, "unit", time=2048, heads=2)
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, precision)))(operands, cotangent)
    _assert_all(mosaic, _numpy_fp64(transform, operands, cotangent),
                f"{transform}, production context, degree={degree}, width={width}, {precision}")


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_outputs_and_vjp_ignore_future_tokens(transform, precision):
    operands, cotangent = _inputs(2, 64, "wide")
    prefix = 300
    attend = jax.jit(partial(_vjp, _mosaic(transform, precision)))
    restricted = cotangent.at[:, prefix:].set(0)
    output, grads = attend(operands, restricted)
    changed = tuple(x.at[:, prefix:].add(2 if index < 3 else 0.5) for index, x in enumerate(operands))
    changed_output, changed_grads = attend(changed, restricted)
    np.testing.assert_array_equal(np.asarray(changed_output[:, :prefix]), np.asarray(output[:, :prefix]))
    for name, grad, changed_grad in zip(("q", "k", "v", "c"), grads, changed_grads):
        np.testing.assert_array_equal(np.asarray(changed_grad), np.asarray(grad), err_msg=name)
        np.testing.assert_array_equal(np.asarray(grad[:, prefix:]), 0, err_msg=name)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_future_nonfinite_keys_and_coefficients_do_not_poison_the_prefix(transform):
    (q, k, v, c), _ = _inputs(2, 64, "unit")
    prefix = 300
    attend = _mosaic(transform)
    output = attend(q, k, v, c)
    poisoned = attend(q, k.at[:, prefix:].set(jnp.inf), v, c.at[:, prefix:].set(jnp.nan))
    np.testing.assert_array_equal(np.asarray(poisoned[:, :prefix]), np.asarray(output[:, :prefix]))
    assert bool(jnp.isnan(poisoned[:, prefix:]).all())


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_nonfinite_weight_poisons_its_row_even_with_zero_values(transform):
    (q, k, v, c), _ = _inputs(2, 64, "unit")
    v = jnp.zeros_like(v)
    c = c.at[0, 100, 1, 2].set(jnp.inf)
    output = np.asarray(_mosaic(transform)(q, k, v, c).astype(jnp.float32))
    assert np.isnan(output[0, 100, 1]).all()
    finite = np.ones(output.shape[:3], bool)
    finite[0, 100, 1] = False
    assert np.isfinite(output[finite]).all()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_extreme_and_zero_scores(transform):
    """Scores far below zero (softplus's weights tend to c_0 there and its slope underflows to zero), far above it
    (z^8 near 1e13, still finite in FP32), and exactly zero (abs's derivative zero, softplus's one half), against
    FP64 at degree eight."""
    (q, k, v, c), cotangent = _inputs(8, 64, "unit")
    rng = np.random.default_rng(5)
    scale = jnp.asarray(rng.choice([0., 1., 12., -12.], size=(1, q.shape[1], q.shape[2], 1)), jnp.bfloat16)
    q = q * scale  # rows of exactly zero scores, and rows of scores of standard deviation 12
    operands = (q, k, v, c)
    mosaic = jax.jit(partial(_vjp, _mosaic(transform)))(operands, cotangent)
    expected = _numpy_fp64(transform, operands, cotangent)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(mosaic))
    _assert_all(mosaic, expected, f"{transform}, extreme and zero scores")


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_repeated_calls_are_bitwise_identical(transform):
    """Forward and every gradient at 2 x 2,048 tokens and 8 heads of width 64, degree eight, five calls."""
    operands, cotangent = _inputs(8, 64, "wide", time=2048, heads=8)
    operands = tuple(jnp.concatenate([x, x[:, ::-1]], axis=0) for x in operands)
    cotangent = jnp.concatenate([cotangent, cotangent[:, ::-1]], axis=0)
    attend = jax.jit(partial(_vjp, _mosaic(transform)))
    first = [np.asarray(x) for x in jax.tree.leaves(attend(operands, cotangent))]
    for _ in range(4):
        for got, want in zip(jax.tree.leaves(attend(operands, cotangent)), first, strict=True):
            np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_model_dispatch_uses_the_kernel_and_reports_transform_diagnostics(transform):
    from scaffolds.transformer.attention.core import attention_core

    (q, k, v, coefficients), _ = _inputs(3, 64, "unit")
    output, diagnostics = attention_core(
        q, k, v, backend="mosaic", attention_degree=3, attention_normalization=False,
        query_polynomial_coefficients=coefficients, attention_coefficient_activation="tanh", score_transform=transform,
        return_diagnostics=True)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(_mosaic(transform)(q, k, v, coefficients)))
    assert {"score_transform_pair_count", "score_transform_input_sum", "score_transform_decreasing_count"} <= set(
        diagnostics)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in diagnostics.values())
