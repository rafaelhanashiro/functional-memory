"""GPU correctness gates for the Mosaic score-function families (H100 and B200; B300 for its optional check).

``mosaic.api.transformed_query_polynomial_attention`` with a prefactor evaluates W = f(s) p(phi(s)) inside the query
polynomial's causal tiles: f = softplus, sigmoid, exp or swish of the raw score, phi the identity or softplus
(``common.score_function_family``: query_times_<f> and query_softplus_times_<f>), and the swish transform without a
prefactor (query_swish). Use ``pytest -m gpu tests/test_mosaic_score_prefactor.py`` inside a GPU allocation with
``JAX_PLATFORMS=cuda,cpu`` (acceptance's GPU stage), which gives the references the host CPU. Outputs and the
q/k/v/coefficient VJPs, at both weight precisions, are compared with an independent NumPy FP64 reference written here
(every pair, explicit f(s) phi(s)^r and the product rule f'(s) p(z) + f(s) p'(z) phi'(s)) under the query kernels'
bounds (``test_mosaic_query``): relative L2 0.006 (output), 0.025 (dQ, dK), 0.015 (dV), and 0.01 for the FP32
coefficient gradient order by order, each with its largest element within three times the bound. Scores are
unit-scale or wide (N(0, 9)), coefficients signed N(0, 1)/r!. At the production context (2,048 tokens, d_K = d_V =
64, degree eight) the production dispatch, ``attention_core`` on backend mosaic, must reach these kernels and is
compared with host FP64 and with the blocked backend's causal strips at production precision. The prefactor
f = phi = softplus of a degree-eight polynomial is also compared with the degree-nine input-power kernel z p(z) that
the softplus-SoS control trained on (a structural zero prepended to the coefficients). A causal score that
overflows makes its row NaN; future tokens never reach a prefix; repeated calls are bitwise identical.
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
FAMILIES = (("identity", "softplus"), ("identity", "sigmoid"), ("identity", "exp"), ("identity", "swish"),
            ("softplus", "softplus"), ("softplus", "sigmoid"), ("softplus", "exp"), ("softplus", "swish"),
            ("swish", "none"))
IDS = [f"{transform}-{prefactor}" for transform, prefactor in FAMILIES]
SPREADS = {"unit": 1.0, "wide": 3.0}
QKV_TOLERANCES = (0.006, 0.025, 0.025, 0.015)  # output, dq, dk, dv: the query kernels' BF16 bounds
COEFFICIENT_TOLERANCE = 0.01


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu(monkeypatch):
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100, B200 or B300 allocation")
    # These cases qualify the kernels, so they run them on either architecture, admitted or not yet.
    monkeypatch.setattr(api, "TRANSFORM_ARCHITECTURES", ("hopper", "blackwell"))
    monkeypatch.setattr(api, "TRANSFORM_COMPUTE_CAPABILITIES", ("9.0", "10.0", "10.3"))


def _inputs(degree, width, spread, *, time=512, heads=2, seed=37):
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


def _function(name, s):
    """(value, derivative) in FP64 of a score function or prefactor: plain formulas, explicit derivatives."""
    sigma = np.exp(-np.logaddexp(0., -s))
    return {"identity": (s, np.ones_like(s)), "none": (np.ones_like(s), np.zeros_like(s)),
            "softplus": (np.logaddexp(0., s), sigma), "sigmoid": (sigma, sigma * (1 - sigma)),
            "exp": (np.exp(s), np.exp(s)), "swish": (s * sigma, sigma + s * sigma * (1 - sigma))}[name]


def _numpy_fp64(transform, prefactor, operands, cotangent):
    """Output and VJP of sum_i f(s_ti) sum_r c_r(t) phi(s_ti)^r v_i, every power explicit, order by order."""
    q, k, v = (np.asarray(x.astype(jnp.float32), np.float64) for x in operands[:3])
    c = np.asarray(operands[3], np.float64).transpose(0, 2, 1, 3)  # b h t r
    do = np.asarray(cotangent.astype(jnp.float32), np.float64)
    time, width, degree = q.shape[1], q.shape[-1], c.shape[-1] - 1
    causal = np.tril(np.ones((time, time), bool))
    s = np.einsum("bthd,bshd->bhts", q, k, optimize=True) / math.sqrt(width)
    z, slope = _function(transform, s)
    f, f_slope = _function(prefactor, s)
    dp = np.where(causal, np.einsum("bthd,bshd->bhts", do, v, optimize=True), 0.)
    polynomial, derivative, dc = np.zeros_like(s), np.zeros_like(s), []
    for r in range(degree + 1):
        power = np.where(causal, z ** r, 0.)
        polynomial += c[..., r, None] * power
        if r:
            derivative += r * c[..., r, None] * np.where(causal, z ** (r - 1), 0.)
        dc.append((dp * f * power).sum(-1))
    weights = np.where(causal, f * polynomial, 0.)
    ds = np.where(causal, dp * (f * slope * derivative + f_slope * polynomial), 0.)
    output = np.einsum("bhts,bshd->bthd", weights, v, optimize=True)
    grads = (np.einsum("bhts,bshd->bthd", ds, k, optimize=True) / math.sqrt(width),
             np.einsum("bhts,bthd->bshd", ds, q, optimize=True) / math.sqrt(width),
             np.einsum("bhts,bthd->bshd", weights, do, optimize=True),
             np.stack(dc, -1).transpose(0, 2, 1, 3))
    return output, grads


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _mosaic(transform, prefactor, precision=None):
    return partial(api.transformed_query_polynomial_attention, transform=transform, prefactor=prefactor,
                   precision=precision)


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


CASES = [pytest.param(transform, prefactor, degree, spread, id=f"{transform}-{prefactor}-m{degree}-{spread}")
         for transform, prefactor in FAMILIES for degree, spread in ((1, "unit"), (4, "unit"), (8, "unit"),
                                                                       (8, "wide"))]


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("transform,prefactor,degree,spread", CASES)
def test_outputs_and_vjp_match_fp64(transform, prefactor, degree, spread, precision):
    operands, cotangent = _inputs(degree, 64, spread)
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, prefactor, precision)))(operands, cotangent)
    assert mosaic[0].dtype == jnp.bfloat16 and mosaic[1][3].dtype == jnp.float32
    expected = _numpy_fp64(transform, prefactor, operands, cotangent)
    label = f"{transform} x {prefactor}, degree={degree}, {precision}, {spread} scores"
    _assert_all(mosaic, expected, label)
    print(f"\n{label}: " + " ".join(f"{k}={v:.2e}" for k, v in _errors(mosaic, expected).items()))


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("width", (4, 16))
@pytest.mark.parametrize("transform,prefactor", (("identity", "exp"), ("softplus", "softplus"), ("swish", "none")))
def test_narrow_key_widths_match_fp64(transform, prefactor, width, precision):
    """Q/K widths 4 and 16 (4 zero-padded to the tensor cores' 16; the score scale is the unpadded width's)."""
    operands, cotangent = _inputs(8, width, "unit")
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, prefactor, precision)))(operands, cotangent)
    _assert_all(mosaic, _numpy_fp64(transform, prefactor, operands, cotangent),
                f"{transform} x {prefactor}, width={width}, {precision}")


def _strips(transform, prefactor, degree=8):
    from scaffolds.transformer.attention.core import attention_core

    def attend(q, k, v, c):
        return attention_core(q, k, v, backend="blocked", blocked_implementation="strips", attention_degree=degree,
                              attention_normalization=False, query_polynomial_coefficients=c,
                              attention_coefficient_activation="tanh", score_transform=transform,
                              score_prefactor=prefactor)
    return attend


def _production(transform, prefactor, degree=8):
    from scaffolds.transformer.attention.core import attention_core

    def attend(q, k, v, c):
        return attention_core(q, k, v, backend="mosaic", attention_degree=degree, attention_normalization=False,
                              query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                              score_transform=transform, score_prefactor=prefactor)
    return attend


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("transform,prefactor", FAMILIES, ids=IDS)
def test_production_context_matches_fp64_and_the_strips(transform, prefactor, precision, monkeypatch):
    """2,048 tokens, d_K = d_V = 64, degree eight, four heads: the production dispatch (attention_core on backend
    mosaic, which must reach these kernels with the transform and prefactor) against host FP64 and against the
    causal strips at production precision (BF16 Q, K, V, FP32 coefficients), both under the query kernels' bounds."""
    calls = []
    original = api.transformed_query_polynomial_attention

    def counted(*args, **kwargs):
        calls.append((kwargs["transform"], kwargs.get("prefactor", "none")))
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    operands, cotangent = _inputs(8, 64, "unit", time=2048, heads=4)
    production = jax.jit(partial(_vjp, _production(transform, prefactor)))(operands, cotangent)
    assert calls == [(transform, prefactor)], "attention_core must reach the Mosaic transformed kernels"
    if precision == "bf16":  # the production dispatch uses the kernels' default precision
        direct = jax.jit(partial(_vjp, _mosaic(transform, prefactor)))(operands, cotangent)
        for got, want in zip(jax.tree.leaves(production), jax.tree.leaves(direct), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, prefactor, precision)))(operands, cotangent)
    label = f"{transform} x {prefactor}, production context, {precision}"
    expected = _numpy_fp64(transform, prefactor, operands, cotangent)
    _assert_all(mosaic, expected, f"{label} vs FP64")
    strips = jax.jit(partial(_vjp, _strips(transform, prefactor)))(operands, cotangent)
    _assert_all(mosaic, strips, f"{label} vs strips")
    print(f"\n{label}: vs FP64 " + " ".join(f"{k}={v:.2e}" for k, v in _errors(mosaic, expected).items())
          + " | vs strips " + " ".join(f"{k}={v:.2e}" for k, v in _errors(mosaic, strips).items())
          + " | strips vs FP64 " + " ".join(f"{k}={v:.2e}" for k, v in _errors(strips, expected).items()))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_softplus_prefactor_matches_the_input_power_kernel(precision):
    """softplus(s) p(softplus(s)) of degree eight (query_softplus_times_softplus) against the softplus-SoS control's
    route, the degree-nine query_softplus kernel with a structural zero prepended to the coefficients, at the
    production context: equal in exact arithmetic, they differ only by FP32 rounding, far inside the bounds, and
    both match FP64."""
    operands, cotangent = _inputs(8, 64, "unit", time=2048, heads=4, seed=71)
    q, k, v, c = operands
    shifted = jnp.concatenate([jnp.zeros_like(c[..., :1]), c], axis=-1)
    prefactor = jax.jit(partial(_vjp, _mosaic("softplus", "softplus", precision)))(operands, cotangent)
    power = jax.jit(partial(_vjp, partial(api.transformed_query_polynomial_attention, transform="softplus",
                                          precision=precision)))((q, k, v, shifted), cotangent)
    power = (power[0], (*power[1][:3], power[1][3][..., 1:]))  # the structural zero's gradient is not a parameter's
    expected = _numpy_fp64("softplus", "softplus", operands, cotangent)
    _assert_all(prefactor, expected, f"softplus prefactor, {precision} vs FP64")
    _assert_all(power, expected, f"input power, {precision} vs FP64")
    _assert_all(prefactor, power, f"softplus prefactor vs input power, {precision}")
    print(f"\nprefactor vs input power, {precision}: "
          + " ".join(f"{k}={v:.2e}" for k, v in _errors(prefactor, power).items()))


@pytest.mark.parametrize("transform,prefactor", FAMILIES, ids=IDS)
def test_outputs_and_vjp_ignore_future_tokens(transform, prefactor):
    operands, cotangent = _inputs(2, 64, "wide")
    prefix = 300
    attend = jax.jit(partial(_vjp, _mosaic(transform, prefactor)))
    restricted = cotangent.at[:, prefix:].set(0)
    output, grads = attend(operands, restricted)
    changed = tuple(x.at[:, prefix:].add(2 if index < 3 else 0.5) for index, x in enumerate(operands))
    changed_output, changed_grads = attend(changed, restricted)
    np.testing.assert_array_equal(np.asarray(changed_output[:, :prefix]), np.asarray(output[:, :prefix]))
    for name, grad, changed_grad in zip(("q", "k", "v", "c"), grads, changed_grads):
        np.testing.assert_array_equal(np.asarray(changed_grad), np.asarray(grad), err_msg=name)
        np.testing.assert_array_equal(np.asarray(grad[:, prefix:]), 0, err_msg=name)


@pytest.mark.parametrize("transform,prefactor", FAMILIES, ids=IDS)
def test_an_overflowing_score_poisons_its_row_as_in_the_strips(transform, prefactor):
    """A causal pair whose FP32 score overflows to -inf (q_t = 2^100 e_j and k_i = -2^100 e_j, in a channel no other
    token of the head uses) makes its row NaN, as the blocked reference does, even for exp, whose e^-inf = 0 meets the
    nonfinite polynomial: one such pair in each of three rows; every other row is finite and bitwise the run's
    without the overflowing entries, and the causal strips make the same rows NaN."""
    (q, k, v, c), _ = _inputs(8, 64, "unit")
    pairs = ((100, 100, 0, 0), (300, 37, 1, 1), (511, 450, 0, 2))  # query, key, head, channel
    for _, _, head, channel in pairs:
        q, k = q.at[:, :, head, channel].set(0), k.at[:, :, head, channel].set(0)
    poisoned_q, poisoned_k = q, k
    for query, key, head, channel in pairs:
        poisoned_q = poisoned_q.at[0, query, head, channel].set(2.0 ** 100)
        poisoned_k = poisoned_k.at[0, key, head, channel].set(-(2.0 ** 100))
    attend = jax.jit(_mosaic(transform, prefactor))
    clean = np.asarray(attend(q, k, v, c).astype(jnp.float32))
    poisoned = np.asarray(attend(poisoned_q, poisoned_k, v, c).astype(jnp.float32))
    expected = np.zeros(poisoned.shape[:3], bool)
    for query, _, head, _ in pairs:
        expected[0, query, head] = True
    np.testing.assert_array_equal(np.isnan(poisoned).all(-1), expected)
    assert np.isfinite(clean).all() and np.isfinite(poisoned[~expected]).all()
    np.testing.assert_array_equal(poisoned[~expected], clean[~expected])
    strips = np.asarray(jax.jit(_strips(transform, prefactor))(poisoned_q, poisoned_k, v, c).astype(jnp.float32))
    np.testing.assert_array_equal(np.isnan(strips).all(-1), expected)


@pytest.mark.parametrize("transform,prefactor", FAMILIES, ids=IDS)
def test_extreme_and_zero_scores(transform, prefactor):
    """Rows of exactly zero scores (every f finite there; swish zero), and rows of scores of standard deviation 12
    (beyond 50 at the largest, where exp's prefactor exceeds e^50, still finite in FP32), against FP64 at degree
    eight."""
    (q, k, v, c), cotangent = _inputs(8, 64, "unit")
    rng = np.random.default_rng(5)
    scale = jnp.asarray(rng.choice([0., 1., 12., -12.], size=(1, q.shape[1], q.shape[2], 1)), jnp.bfloat16)
    operands = (q * scale, k, v, c)
    mosaic = jax.jit(partial(_vjp, _mosaic(transform, prefactor)))(operands, cotangent)
    expected = _numpy_fp64(transform, prefactor, operands, cotangent)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(mosaic))
    _assert_all(mosaic, expected, f"{transform} x {prefactor}, extreme and zero scores")


@pytest.mark.parametrize("transform,prefactor", FAMILIES, ids=IDS)
def test_repeated_calls_are_bitwise_identical(transform, prefactor):
    """Forward and every gradient at 2 x 2,048 tokens and 8 heads of width 64, degree eight, five calls."""
    operands, cotangent = _inputs(8, 64, "wide", time=2048, heads=8)
    operands = tuple(jnp.concatenate([x, x[:, ::-1]], axis=0) for x in operands)
    cotangent = jnp.concatenate([cotangent, cotangent[:, ::-1]], axis=0)
    attend = jax.jit(partial(_vjp, _mosaic(transform, prefactor)))
    first = [np.asarray(x) for x in jax.tree.leaves(attend(operands, cotangent))]
    for _ in range(4):
        for got, want in zip(jax.tree.leaves(attend(operands, cotangent)), first, strict=True):
            np.testing.assert_array_equal(np.asarray(got), want)
