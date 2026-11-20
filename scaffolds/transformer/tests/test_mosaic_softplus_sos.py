"""GPU qualification of query SoS at softplus scores, with and without a z prefactor.

The independent FP64 reference evaluates the two square factors directly, without
expanding polynomial coefficients. Its analytic VJP covers Q/K/V and the entire
query coefficient generator (input x, projection W, amplitude a and bias b).
Production uses the expanded coefficients and Mosaic's transformed polynomial
kernels. The variants have degree-eight or degree-twelve SoS cores; the prefactor makes total
degree nine or thirteen and contributes its product-rule term to the score gradient. The
input-power runs also run at Q/K widths 4, 8 and 16 (zero-padded to 16). The general score
functions W = f(s) h_t(g(s)) (a prefactor f of the raw score: softplus, sigmoid, exp or swish;
g the identity or softplus) are checked through the same direct-squares reference.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api
from scaffolds.transformer.attention.backends.mosaic.common import PRECISIONS
from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors
from scaffolds.transformer.tests.test_fused_attention import _assert_relative

pytestmark = pytest.mark.gpu
CORE_DEGREES = (8, 12)
QKV_TOLERANCES = (0.006, 0.025, 0.025, 0.015)
GENERATOR_TOLERANCE = 0.02


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu(monkeypatch):
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100, B200 or B300 allocation")
    # Qualification can exercise an architecture before its production admission.
    monkeypatch.setattr(api, "TRANSFORM_ARCHITECTURES", ("hopper", "blackwell"))
    monkeypatch.setattr(api, "TRANSFORM_COMPUTE_CAPABILITIES", ("9.0", "10.0", "10.3"))


def _inputs(*, core_degree=8, time=512, spread=1.0, seed=172, width=64):
    rng = np.random.default_rng(seed)
    shape = (1, time, 1, 64)
    q = jnp.asarray(spread * rng.standard_normal((1, time, 1, width)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((1, time, 1, width)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    x = jnp.asarray(rng.standard_normal((1, time, 16)), jnp.float32)
    w = jnp.asarray(0.25 * rng.standard_normal((1, 16, core_degree + 1)), jnp.float32)
    a = jnp.asarray(0.4 + 0.1 * rng.standard_normal((1, core_degree + 1)), jnp.float32)
    b = jnp.asarray(1.0 + 0.3 * rng.standard_normal((1, core_degree + 1)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    return (q, k, v, x, w, a, b), cotangent


def _generated(x, w, a, b):
    return b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, w))


def _attend(input_power, precision=None, *, core_degree=8, production=False, transform="softplus",
            prefactor="none"):
    from scaffolds.transformer.attention.core import attention_core
    from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients

    def attend(q, k, v, x, w, a, b):
        degree = core_degree + input_power
        coefficients = query_square_coefficients(_generated(x, w, a, b), degree, input_power=input_power)
        if production:
            return attention_core(q, k, v, backend="mosaic", attention_degree=degree,
                                  attention_normalization=False, query_polynomial_coefficients=coefficients,
                                  attention_coefficient_activation="tanh", score_transform=transform,
                                  score_prefactor=prefactor)
        return api.transformed_query_polynomial_attention(q, k, v, coefficients, transform=transform,
                                                         prefactor=prefactor, precision=precision)
    return attend


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _score_function(name, scores):
    """(value, derivative) in FP64 of g or f: plain formulas, explicit derivatives."""
    sigma = np.exp(-np.logaddexp(0.0, -scores))
    return {"identity": (scores, np.ones_like(scores)), "none": (np.ones_like(scores), np.zeros_like(scores)),
            "softplus": (np.logaddexp(0.0, scores), sigma), "sigmoid": (sigma, sigma * (1 - sigma)),
            "exp": (np.exp(scores), np.exp(scores)), "swish": (scores * sigma, sigma + scores * sigma * (1 - sigma))}[name]


def _numpy_fp64(input_power, operands, cotangent, *, transform="softplus", prefactor="none"):
    """Explicit square factors and their analytic VJP; never uses expanded c_r. The weight is
    f(s) z^power h(z), z = g(s) (``transform``), f the ``prefactor`` (one for "none")."""
    q, k, v, x, w, a, b = (np.asarray(value.astype(jnp.float32), np.float64) for value in operands)
    do = np.asarray(cotangent.astype(jnp.float32), np.float64)
    time, width = q.shape[1], q.shape[-1]
    core_degree = a.shape[-1] - 1
    tanh = np.tanh(np.einsum("btd,hdr->bthr", x, w, optimize=True))
    generated = b + a * tanh
    # The execution stores these fixed factors in FP32; use those same constants.
    fixed = np.asarray(taylor_square_factors(core_degree), np.float32).astype(np.float64)
    factors = (fixed * (1 + generated) / 2).transpose(0, 2, 1, 3)
    scores = np.einsum("bthd,bshd->bhts", q, k, optimize=True) / math.sqrt(width)
    z, sigmoid = _score_function(transform, scores)  # z = g(s) and g'(s): the logistic for softplus
    f, f_slope = _score_function(prefactor, scores)
    split = core_degree // 2 + 1
    parts = (factors[..., :split], factors[..., split:])
    values, slopes = [], []
    for part in parts:
        values.append(sum(part[..., order, None] * z ** order for order in range(part.shape[-1])))
        slopes.append(sum(order * part[..., order, None] * z ** (order - 1)
                          for order in range(1, part.shape[-1])))
    square_sum = values[0] ** 2 + values[1] ** 2
    prefactor = z if input_power else np.ones_like(z)
    derivative = prefactor * 2 * (values[0] * slopes[0] + values[1] * slopes[1])
    if input_power:
        derivative += square_sum
    causal = np.tril(np.ones((time, time), bool))
    weights = np.where(causal, f * prefactor * square_sum, 0.0)
    dp = np.where(causal, np.einsum("bthd,bshd->bhts", do, v, optimize=True), 0.0)
    ds = dp * (f * derivative * sigmoid + f_slope * prefactor * square_sum)
    dp = dp * f  # the factors' cotangent carries the prefactor
    output = np.einsum("bhts,bshd->bthd", weights, v, optimize=True)
    dq = np.einsum("bhts,bshd->bthd", ds, k, optimize=True) / math.sqrt(width)
    dk = np.einsum("bhts,bthd->bshd", ds, q, optimize=True) / math.sqrt(width)
    dv = np.einsum("bhts,bthd->bshd", weights, do, optimize=True)
    df = np.stack([np.sum(dp * prefactor * 2 * value * z ** order, axis=-1)
                   for part, value in zip(parts, values, strict=True)
                   for order in range(part.shape[-1])], axis=-1).transpose(0, 2, 1, 3)
    dg = df * fixed / 2
    dpre = dg * a * (1 - tanh ** 2)
    dx = np.einsum("bthr,hdr->btd", dpre, w, optimize=True)
    dw = np.einsum("btd,bthr->hdr", x, dpre, optimize=True)
    da = np.sum(dg * tanh, axis=(0, 1))
    db = np.sum(dg, axis=(0, 1))
    return output, (dq, dk, dv, dx, dw, da, db)


def _assert_all(actual, expected, label):
    _assert_relative(actual[0], expected[0], rtol=QKV_TOLERANCES[0], name=f"{label}: output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], QKV_TOLERANCES[1:]):
        _assert_relative(got, want, rtol=tolerance, name=f"{label}: d{name}")
    _assert_relative(actual[1][3][:, :, None, :], expected[1][3][:, :, None, :],
                     rtol=GENERATOR_TOLERANCE, name=f"{label}: dx")
    # Each generated square-factor coordinate must train, including small factors.
    for name, got, want in zip(("W", "a", "b"), actual[1][4:], expected[1][4:]):
        for factor in range(want.shape[-1]):
            # The shared assertion expects [batch, tokens, heads, channels].
            view = (lambda value: value[..., factor].T[None, :, :, None]) if name == "W" else (
                lambda value: value[..., factor][None, None, :, None])
            _assert_relative(view(got), view(want), rtol=GENERATOR_TOLERANCE,
                             name=f"{label}: d{name}, factor {factor}")


@pytest.mark.parametrize("core_degree", CORE_DEGREES)
@pytest.mark.parametrize("input_power", (0, 1))
@pytest.mark.parametrize("precision", PRECISIONS)
def test_composed_forward_and_generator_vjp_match_direct_squares(core_degree, input_power, precision):
    operands, cotangent = _inputs(core_degree=core_degree, spread=3.0)
    actual = jax.jit(partial(_vjp, _attend(input_power, precision, core_degree=core_degree)))(operands, cotangent)
    assert actual[0].dtype == jnp.bfloat16
    assert all(gradient.dtype == jnp.float32 for gradient in actual[1][3:])
    _assert_all(actual, _numpy_fp64(input_power, operands, cotangent), f"power={input_power}, {precision}")


@pytest.mark.parametrize("core_degree", CORE_DEGREES)
@pytest.mark.parametrize("input_power", (0, 1))
@pytest.mark.parametrize("spread", (1.0, 3.0))
def test_production_context_dispatch_and_vjp(core_degree, input_power, spread, monkeypatch):
    """2,048 tokens and d_K=d_V=64, including total degrees nine/thirteen and all generator gradients."""
    calls = []
    original = api.transformed_query_polynomial_attention

    def counted(q, k, v, coefficients, **kwargs):
        calls.append((coefficients.shape[-1] - 1, kwargs["transform"]))
        return original(q, k, v, coefficients, **kwargs)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    operands, cotangent = _inputs(core_degree=core_degree, time=2048, spread=spread)
    actual = jax.jit(partial(_vjp, _attend(input_power, core_degree=core_degree, production=True)))(operands, cotangent)
    assert calls == [(core_degree + input_power, "softplus")]
    _assert_all(actual, _numpy_fp64(input_power, operands, cotangent),
                f"production, power={input_power}, spread={spread}")


@pytest.mark.parametrize("core_degree", CORE_DEGREES)
@pytest.mark.parametrize("input_power", (0, 1))
def test_causal_prefix_and_generator_gradients_ignore_future_tokens(core_degree, input_power):
    operands, cotangent = _inputs(core_degree=core_degree)
    prefix = 300
    restricted = cotangent.at[:, prefix:].set(0)
    attend = jax.jit(partial(_vjp, _attend(input_power, core_degree=core_degree)))
    output, gradients = attend(operands, restricted)
    changed = tuple(value.at[:, prefix:].add(2) if index < 4 else value
                    for index, value in enumerate(operands))
    changed_output, changed_gradients = attend(changed, restricted)
    np.testing.assert_array_equal(np.asarray(output[:, :prefix]), np.asarray(changed_output[:, :prefix]))
    for index, (gradient, changed_gradient) in enumerate(zip(gradients, changed_gradients, strict=True)):
        np.testing.assert_array_equal(np.asarray(gradient), np.asarray(changed_gradient), err_msg=f"gradient {index}")
        if index < 4:
            np.testing.assert_array_equal(np.asarray(gradient[:, prefix:]), 0)


@pytest.mark.parametrize("core_degree", CORE_DEGREES)
@pytest.mark.parametrize("input_power", (0, 1))
def test_finite_negative_tail_and_zero_square_factors(core_degree, input_power):
    """The prefactor removes the negative-score baseline; an identically zero SoS stays zero."""
    operands, _ = _inputs(core_degree=core_degree)
    q, k, v, x, w, a, b = operands
    q = jnp.zeros_like(q).at[..., 0].set(-32)
    k = jnp.zeros_like(k).at[..., 0].set(8)  # every raw score is -32
    v = jnp.ones_like(v)
    x, w, a, b = jnp.zeros_like(x), jnp.zeros_like(w), jnp.zeros_like(a), jnp.ones_like(b)
    attend = jax.jit(_attend(input_power, core_degree=core_degree))
    output = np.asarray(attend(q, k, v, x, w, a, b).astype(jnp.float32))
    assert np.isfinite(output).all()
    if input_power:
        # At this magnitude the finite-precision softplus may round to exactly zero.
        assert np.max(np.abs(output)) < 1e-6
    else:
        counts = np.arange(1, q.shape[1] + 1)[None, :, None, None]
        np.testing.assert_allclose(output, np.broadcast_to(counts, output.shape), rtol=0.006, atol=0.01)
    zero = np.asarray(attend(q, k, v, x, w, a, -b).astype(jnp.float32))
    np.testing.assert_array_equal(zero, 0)  # g=-1 makes both factors exactly zero


@pytest.mark.parametrize("core_degree", CORE_DEGREES)
@pytest.mark.parametrize("input_power", (0, 1))
def test_overflowing_negative_score_remains_a_numerical_failure(core_degree, input_power):
    """A structural constant zero must not hide a nonfinite causal score."""
    q, k, v, x, w, a, b = _inputs(core_degree=core_degree)[0]
    q, k, v = jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v)
    q = q.at[0, 100, 0, 0].set(2.0 ** 100)
    k = k.at[0, 37, 0, 0].set(-2.0 ** 100)
    output = np.asarray(_attend(input_power, core_degree=core_degree)(q, k, v, x, w, a, b).astype(jnp.float32))
    assert np.isnan(output[0, 100, 0]).all()
    assert np.isfinite(np.delete(output, 100, axis=1)).all()


@pytest.mark.parametrize("width", (4, 8, 16))
def test_input_power_at_narrow_key_widths(width, monkeypatch):
    """The softplus-SoS control's z h_t(z) (core eight, total nine) at Q/K widths 4, 8 and 16, zero-padded to the
    tensor cores' 16 with the unpadded width's scale, through the production dispatch at 2,048 tokens, against the
    direct-squares FP64 reference with every generator gradient."""
    calls = []
    original = api.transformed_query_polynomial_attention

    def counted(q, k, v, coefficients, **kwargs):
        calls.append((q.shape[-1], coefficients.shape[-1] - 1))
        return original(q, k, v, coefficients, **kwargs)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    operands, cotangent = _inputs(time=2048, width=width, seed=181)
    actual = jax.jit(partial(_vjp, _attend(1, production=True)))(operands, cotangent)
    assert calls == [(width, 9)]
    _assert_all(actual, _numpy_fp64(1, operands, cotangent), f"input power, width={width}")


SCORE_FUNCTIONS = [(transform, prefactor) for transform in ("identity", "softplus")
                   for prefactor in ("softplus", "sigmoid", "exp", "swish")]


@pytest.mark.parametrize("transform,prefactor", SCORE_FUNCTIONS, ids=[f"{t}-{f}" for t, f in SCORE_FUNCTIONS])
def test_prefactor_times_sos_at_the_production_context(transform, prefactor, monkeypatch):
    """W = f(s) h_t(g(s)) with a degree-eight query SoS core (input power zero), the D-family runs' composition:
    the production dispatch at 2,048 tokens must reach the prefactor kernels, and its output and the Q/K/V and
    generator (x, W, a, b) gradients match the direct-squares FP64 reference with the prefactor's product rule."""
    calls = []
    original = api.transformed_query_polynomial_attention

    def counted(q, k, v, coefficients, **kwargs):
        calls.append((kwargs["transform"], kwargs.get("prefactor", "none")))
        return original(q, k, v, coefficients, **kwargs)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    operands, cotangent = _inputs(time=2048, seed=191)
    attend = _attend(0, production=True, transform=transform, prefactor=prefactor)
    actual = jax.jit(partial(_vjp, attend))(operands, cotangent)
    assert calls == [(transform, prefactor)]
    _assert_all(actual, _numpy_fp64(0, operands, cotangent, transform=transform, prefactor=prefactor),
                f"{prefactor} x h({transform})")


@pytest.mark.parametrize("precision", PRECISIONS)
def test_softplus_prefactor_equals_the_input_power_route(precision):
    """softplus(s) h_t(softplus(s)) (core eight, input power zero, the general prefactor) and the control's
    z h_t(z) (total degree nine, the input power's structural zero) are one function: both match the direct
    squares, and each other within the bounds."""
    operands, cotangent = _inputs(time=2048, spread=1.0, seed=193)
    general = jax.jit(partial(_vjp, _attend(0, precision, prefactor="softplus")))(operands, cotangent)
    power = jax.jit(partial(_vjp, _attend(1, precision)))(operands, cotangent)
    expected = _numpy_fp64(1, operands, cotangent)
    _assert_all(general, expected, f"general prefactor, {precision}")
    _assert_all(power, expected, f"input power, {precision}")
    _assert_all(general, power, f"general prefactor vs input power, {precision}")
