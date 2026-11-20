"""CPU checks of the score prefactor on polynomials and of the swish score function (attention/pieces.py).

A polynomial's weight is W = f(s) p(phi(s)) (attention/score_transform.py): the prefactor f (softplus, sigmoid, exp
or swish of the raw score) multiplies the polynomial p of the transformed score z = phi(s). The Mosaic transformed
query-polynomial kernels evaluate it in their causal tiles for the families ``common.score_function_family`` names
(a prefactor with the identity or softplus transform, and the swish transform without one); the blocked backend
evaluates it as its reference composition. Checked here, on a CPU host: swish's and exp's values and derivatives
against FP64 autodiff of their plain formulas; the family registry, counts, flags and support predicate; every new
family's Hopper kernels lowering in exactly the shared memory ``hopper_smem_bytes`` counts (one byte less fails),
and the Blackwell kernels and both entry points lowering forward and with every gradient at 2,048 tokens and Q/K
widths 4 to 64; the tile mathematics against explicit FP64 f(s) p(phi(s)) and its product rule, on unit, wide,
zero and extreme scores with masked entries holding +-1e15; a nonfinite causal score making its row NaN; the
prefactor f = phi = softplus on a degree-m polynomial equal to the input power z p(z) of degree m + 1, on the
blocked backend and through two models; dispatch from attention_core and the model to the kernels; admission on
H100 and B200 alone, at the GPU-qualified degrees up to eight; the recipes' validation and refusals; and their
provenance and diagnostics labels. The kernels' numerics on GPUs are qualified by
``tests/test_mosaic_score_prefactor.py``.
"""
from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import score_transform
from scaffolds.transformer.attention.backends.mosaic import api, blackwell, common, hopper

PREFACTOR_FAMILIES = {(transform, prefactor): common.score_function_family(transform, prefactor)
                      for transform in ("identity", "softplus") for prefactor in common.SCORE_PREFACTORS}
NEW_FAMILIES = {**PREFACTOR_FAMILIES, ("swish", "none"): common.QUERY_SWISH}
FAMILY_IDS = [f"{transform}-{prefactor}" for transform, prefactor in NEW_FAMILIES]


def _lowers_for_cuda(function, *operands):
    """Trace and lower ``function`` for cuda on any host, building every Mosaic GPU kernel."""
    jax.jit(lambda *arguments: function(*arguments)).trace(*operands).lower(lowering_platforms=("cuda",))


# Values and derivatives of the new score functions.

def _logistic(s):
    return np.exp(-np.logaddexp(0., -s))


def _function(name, s):
    """(value, derivative) in FP64 of a score function or prefactor: plain formulas, explicit derivatives."""
    sigma = _logistic(s)
    return {"identity": (s, np.ones_like(s)), "none": (np.ones_like(s), np.zeros_like(s)),
            "softplus": (np.logaddexp(0., s), sigma), "sigmoid": (sigma, sigma * (1 - sigma)),
            "abs": (np.abs(s), np.sign(s)), "exp": (np.exp(s), np.exp(s)),
            "swish": (s * sigma, sigma + s * sigma * (1 - sigma))}[name]


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64), ids=("fp32", "fp64"))
def test_swish_and_exp_match_float64_autodiff(dtype):
    """swish(s) = s sigmoid(s) and its JVP sigmoid(s) + s sigmoid'(s) against JAX autodiff of the plain formula in
    FP64: zero and 1/2 at s = 0, the minimum -0.2785 at -1.2785, tending to s and 1 as s -> +inf and to 0 as
    s -> -inf; NaN, value and derivative, for every nonfinite score. exp is e^s with derivative e^s."""
    with jax.enable_x64(dtype == jnp.float64):
        special = jnp.asarray([0., -0., 1e-30, -1e-30, 1., -1., -1.2784645, 4.2, -4.2, 30., -30., 88., -88., 104., -104.,
                               jnp.inf, -jnp.inf, jnp.nan], dtype)
        values = np.asarray(score_transform.swish(special), np.float64)
        slopes = np.asarray(jax.vmap(jax.grad(score_transform.swish))(special), np.float64)
        exp_values = np.asarray(score_transform.prefactor(special, "exp"), np.float64)
        exp_slopes = np.asarray(jax.vmap(jax.grad(lambda x: score_transform.prefactor(x, "exp")))(special), np.float64)
        assert np.asarray(score_transform.apply(special, "swish")).tolist() == pytest.approx(
            np.asarray(score_transform.prefactor(special, "swish")).tolist(), nan_ok=True)
    with jax.enable_x64():
        plain = jnp.asarray(special, jnp.float64)
        reference = jax.vmap(lambda x: x * jax.nn.sigmoid(x))
        want_values = np.asarray(reference(plain))
        want_slopes = np.asarray(jax.vmap(jax.grad(lambda x: x * jax.nn.sigmoid(x)))(plain))
    s = np.asarray(special, np.float64)
    finite = np.isfinite(s)
    rtol = 1e-6 if dtype == jnp.float32 else 1e-13
    # In FP32, e^-|s| is subnormal below s = -87.3 and may flush to zero, and with it s e^-|s| (below 1e-35).
    tiny = 1e-35 if dtype == jnp.float32 else np.finfo(np.float64).tiny
    np.testing.assert_allclose(values[finite], want_values[finite], rtol=rtol, atol=tiny)
    # The slope's two terms cancel near the minimum: an absolute error of the larger term's rounding there.
    np.testing.assert_allclose(slopes[finite], want_slopes[finite], rtol=rtol,
                               atol=1e-6 if dtype == jnp.float32 else 1e-15)
    assert values[0] == 0. and slopes[0] == .5 and slopes[1] == .5
    assert abs(values[6] + 0.2784645) < 1e-6 and abs(slopes[6]) < 1e-6
    assert np.isnan(values[~finite]).all() and np.isnan(slopes[~finite]).all()
    # In the scores' dtype: FP32 overflows above 88.7 and may flush its subnormal results to zero.
    want_exp = np.exp(s[finite]).astype(np.dtype(dtype)).astype(np.float64)
    np.testing.assert_allclose(exp_values[finite], want_exp, rtol=rtol, atol=tiny)
    np.testing.assert_array_equal(exp_slopes, exp_values)
    assert exp_values[-3] == np.inf and exp_values[-2] == 0. and np.isnan(exp_values[-1])


def test_registries_name_every_score_function():
    assert score_transform.TRANSFORMS == ("identity", "softplus", "abs", "sigmoid", "swish")
    assert score_transform.PREFACTORS == ("none", "softplus", "sigmoid", "exp", "swish")
    assert set(score_transform.FORMULAS) == set(score_transform.TRANSFORMS) - {"identity"}
    assert set(score_transform.PREFACTOR_FORMULAS) == set(score_transform.PREFACTORS) - {"none"}
    assert score_transform.SIGNED_FUNCTIONS == ("swish",)
    assert api.TRANSFORM_FAMILY == {"softplus": common.QUERY_SOFTPLUS, "abs": common.QUERY_ABS,
                                    "sigmoid": common.QUERY_SIGMOID, "swish": common.QUERY_SWISH}


# The kernel families.

def test_families_count_like_the_query_family():
    """Every transformed family holds the query family's vectors, sums, keys and heads at every degree, needs no
    Pair, and compiles with approximate exp2/log2 when phi or f evaluates a transcendental (all but abs)."""
    assert common.score_function_family("identity") == common.QUERY
    assert common.score_function_family("identity", "exp") == "query_times_exp"
    assert common.score_function_family("softplus", "sigmoid") == "query_softplus_times_sigmoid"
    for transform in ("abs", "sigmoid", "swish", "square"):
        assert common.score_function_family(transform, "exp") is None
    assert common.score_function_family("identity", "relu") is None
    assert set(common.TRANSFORMED_QUERY_FAMILIES) == {
        *PREFACTOR_FAMILIES.values(), common.QUERY_SOFTPLUS, common.QUERY_ABS, common.QUERY_SIGMOID, common.QUERY_SWISH}
    assert set(common.PREFACTOR_FAMILIES) == set(PREFACTOR_FAMILIES.values()) and len(common.PREFACTOR_FAMILIES) == 8
    for family in common.TRANSFORMED_QUERY_FAMILIES:
        for degree in range(1, 14):
            assert common.vector_counts(family, degree) == common.vector_counts(common.QUERY, degree)
            assert common.key_counts(family, degree) == (0, 0) and common.head_count(family, degree) == 0
            assert common.hopper_smem_bytes(family, degree, 64, 64) == common.hopper_smem_bytes(
                common.QUERY, degree, 64, 64)
        assert not common.has_pair(family) and not common.accumulates(family)
        assert api.default_precision(family, 8) == common.BF16
        assert common.approx_math(family) == (family != common.QUERY_ABS)
    assert not common.approx_math(common.QUERY)


def test_support_is_the_transform_support():
    """A prefactor takes its transform's support (the query family's shared memory at every width, Q/K width 4
    included) at the degrees its GPU qualification covers, up to eight (``MAX_PREFACTOR_QUERY_DEGREE``), not the
    transform's limit (softplus 13 at width 64 and 512 or 2,048 tokens); no family takes a gate, and transforms
    that take no prefactor are refused."""
    assert api.WIDTHS == (4, 8, 16, 32, 64, 128)
    assert api.MAX_PREFACTOR_QUERY_DEGREE == 8
    for (transform, prefactor), family in NEW_FAMILIES.items():
        qualified = api.MAX_PREFACTOR_QUERY_DEGREE if prefactor != "none" else 99
        for width in api.WIDTHS:
            for time in (256, 512, 2048, 4096):
                for degree in range(0, 15):
                    assert api.query_degree_supported(degree, width, time, transform=transform, prefactor=prefactor) == (
                        degree <= qualified and api.query_degree_supported(degree, width, time, transform=transform)), (
                        family, degree, width)
        assert api.query_degree_supported(8, 64, 2048, transform=transform, prefactor=prefactor)
        nine = api.query_degree_supported(9, 64, 2048, transform=transform, prefactor=prefactor)
        assert nine == (prefactor == "none")
        assert not api.query_degree_supported(8, 64, 2048, transform=transform, prefactor=prefactor, gate="query")
    for transform in ("abs", "sigmoid", "swish"):
        assert not api.query_degree_supported(8, 64, 2048, transform=transform, prefactor="exp")
    assert not api.query_degree_supported(8, 64, 2048, prefactor="relu")
    assert api.query_degree_supported(9, 4, 2048, transform="softplus")


@pytest.mark.parametrize("degree", (1, 8, 9, 12))
@pytest.mark.parametrize("transform,prefactor", list(NEW_FAMILIES), ids=FAMILY_IDS)
def test_hopper_kernels_lower_in_exactly_their_counted_shared_memory(transform, prefactor, degree):
    """H100's forward, dQ and dK/dV lower at 2,048 tokens and d_K = 64 in exactly the bytes hopper_smem_bytes
    counts (the query family's), and fail one byte below."""
    from jax.experimental.mosaic.gpu.core import artificial_shared_memory_limit

    family = NEW_FAMILIES[transform, prefactor]
    q = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
    vectors = jax.ShapeDtypeStruct((degree + 1, 1, 2, 2048), jnp.float32)
    options = dict(family=family, degree=degree, scale=.125, precision="bf16")
    kernels = {"forward": (lambda q, k, v, x: hopper.forward(q, k, v, x, **options), (q, q, q)),
               "dq": (lambda q, k, v, do, x: hopper.dq(q, k, v, do, x, **options), (q, q, q, q)),
               "dkv": (lambda q, k, v, do, x: hopper.dkv(q, k, v, do, x, **options), (q, q, q, q))}
    sizes = common.hopper_smem_bytes(family, degree, 64, 64)
    for name, (kernel, arrays) in kernels.items():
        with artificial_shared_memory_limit(sizes[name]):
            _lowers_for_cuda(kernel, *arrays, vectors)
        with artificial_shared_memory_limit(sizes[name] - 1), pytest.raises(ValueError, match="shared memory"):
            _lowers_for_cuda(kernel, *arrays, vectors)


@pytest.mark.parametrize("transform,prefactor", list(NEW_FAMILIES), ids=FAMILY_IDS)
def test_blackwell_kernels_lower_with_the_query_rings(transform, prefactor):
    """B200's forward, dQ, dK and dV kernels lower at 2,048 tokens and degrees eight and nine, with Q/K widths 4, 8
    and 16 zero-padded to the tensor cores' 16, and 64 (three stages)."""
    family = NEW_FAMILIES[transform, prefactor]
    for degree, width in ((8, 64), (9, 64), (9, 16), (8, 16)):
        q = jax.ShapeDtypeStruct((1, 2048, 2, width), jnp.bfloat16)
        v = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
        vectors = jax.ShapeDtypeStruct((degree + 1, 1, 2, 2048), jnp.float32)
        options = dict(family=family, degree=degree, scale=.125, precision="bf16")
        assert common.blackwell_dkv_stages(max(width, api.MIN_WIDTH), 64, degree + 1, 2048) == 3
        _lowers_for_cuda(lambda q, k, v, x: blackwell.forward(q, k, v, x, **options), q, q, v, vectors)
        _lowers_for_cuda(lambda q, k, v, do, x: blackwell.dq(q, k, v, do, x, **options, stages=3), q, q, v, v, vectors)
        for which in ("dk", "dv"):
            _lowers_for_cuda(lambda q, k, v, do, x: blackwell.dkv(q, k, v, do, x, which=which, stages=3, **options),
                             q, q, v, v, vectors)


def _admit_all(monkeypatch, arch):
    monkeypatch.setattr(api, "_kernels", lambda: (arch, {"hopper": hopper, "blackwell": blackwell}[arch]))
    monkeypatch.setattr(api, "_device_compute_capability", lambda: "9.0" if arch == "hopper" else "10.0")


@pytest.mark.parametrize("arch", ("hopper", "blackwell"))
@pytest.mark.parametrize("transform,prefactor", list(NEW_FAMILIES), ids=FAMILY_IDS)
def test_entry_point_lowers_with_every_gradient(monkeypatch, transform, prefactor, arch):
    """The production entry point lowers forward and with every gradient at 2,048 tokens: degree eight at Q/K
    widths 4, 8, 16 (zero-padded to 16) and 64, and degree nine at widths 4 and 64, which recipes take only
    without a prefactor (a prefactor's recipes stop at its GPU-qualified degree eight)."""
    _admit_all(monkeypatch, arch)

    def attend(q, k, v, c):
        return api.transformed_query_polynomial_attention(q, k, v, c, transform=transform, prefactor=prefactor)

    for degree, width in ((8, 4), (8, 8), (8, 16), (8, 64), (9, 4), (9, 64)):
        assert api.query_degree_supported(degree, width, 2048, transform=transform, prefactor=prefactor) == (
            prefactor == "none" or degree <= api.MAX_PREFACTOR_QUERY_DEGREE)
        q = jax.ShapeDtypeStruct((1, 2048, 2, width), jnp.bfloat16)
        v = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
        c = jax.ShapeDtypeStruct((1, 2048, 2, degree + 1), jnp.float32)
        _lowers_for_cuda(attend, q, q, v, c)
        _lowers_for_cuda(lambda *a: jax.vjp(attend, *a)[1](jnp.ones(a[2].shape, a[2].dtype)), q, q, v, c)


@pytest.mark.parametrize("capability", ("9.0", "10.0", "10.3"))
@pytest.mark.parametrize("transform,prefactor", list(PREFACTOR_FAMILIES)[:2] + [("softplus", "swish")])
def test_entry_point_admits_h100_and_b200_only(monkeypatch, transform, prefactor, capability):
    """B200 admission does not admit the B300 that shares its kernels: refused before tracing."""
    q = k = v = jax.ShapeDtypeStruct((1, 256, 2, 64), jnp.bfloat16)
    c = jax.ShapeDtypeStruct((1, 256, 2, 9), jnp.float32)
    attend = lambda *a: api.transformed_query_polynomial_attention(*a, transform=transform, prefactor=prefactor)
    monkeypatch.setattr(api, "_device_compute_capability", lambda: capability)
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper) if capability == "9.0" else ("blackwell", blackwell))
    if capability in ("9.0", "10.0"):
        assert jax.eval_shape(attend, q, k, v, c).shape == v.shape
    else:
        with pytest.raises(ValueError, match=r"admits H100 \(sm90\) and B200 \(sm100\) only"):
            jax.eval_shape(attend, q, k, v, c)


@pytest.mark.parametrize("transform,prefactor", (("identity", "none"), ("abs", "exp"), ("sigmoid", "softplus"),
                                                 ("swish", "swish"), ("identity", "relu"), ("relu", "none")))
def test_entry_point_refuses_score_functions_without_a_family(monkeypatch, transform, prefactor):
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    q = k = v = jnp.zeros((1, 256, 2, 64), jnp.bfloat16)
    c = jnp.zeros((1, 256, 2, 5), jnp.float32)
    with pytest.raises(ValueError, match="transform|kernel family"):
        jax.eval_shape(lambda *a: api.transformed_query_polynomial_attention(
            *a, transform=transform, prefactor=prefactor), q, k, v, c)


# Tile mathematics: the functions every kernel calls on its tiles, on one whole causal matrix as a tile crossing
# the diagonal, with the kernels' masking (masked scores and dP enter as zero; masked W and dS leave as zero).

def _tile_terms(family, degree, s, dp, coefficients):
    time = s.shape[0]
    causal = jnp.asarray(np.arange(time)[:, None] >= np.arange(time)[None, :])
    rows = [jnp.broadcast_to(x[:, None], s.shape) for x in coefficients]
    columns = [jnp.broadcast_to(x[None, :], s.shape) for x in coefficients]
    s, dp = jnp.where(causal, s, 0.), jnp.where(causal, dp, 0.)
    w = jnp.where(causal, common.forward_weights(family, degree, s, rows), 0.)
    ds, sums = common.dq_terms(family, degree, s, dp, rows)
    wt = jnp.where(causal.T, common.value_weights(family, degree, s.T, columns), 0.)
    dst, key_sums = common.dk_terms(family, degree, s.T, dp.T, columns)
    assert key_sums == []
    return w, jnp.where(causal, ds, 0.), jnp.stack(sums), wt, jnp.where(causal.T, dst, 0.)


def _tile_reference(transform, prefactor, s, dp, c):
    """Explicit W = f(s) sum_r c_r z^r, dS = G (f(s) phi'(s) sum_r r c_r z^(r-1) + f'(s) sum_r c_r z^r) and
    dc_r = sum_i G f(s) z^r over causal pairs, z = phi(s)."""
    time, orders = s.shape[0], c.shape[0]
    causal = np.arange(time)[:, None] >= np.arange(time)[None, :]
    z, slope = _function(transform, s)
    f, f_slope = _function(prefactor, s)
    g = np.where(causal, dp, 0.)
    polynomial = sum(c[r][:, None] * z ** r for r in range(orders))
    derivative = sum(r * c[r][:, None] * z ** (r - 1) for r in range(1, orders))
    sums = np.stack([(g * np.where(causal, f * z ** r, 0.)).sum(1) for r in range(orders)])
    weights = np.where(causal, f * polynomial, 0.)
    ds = np.where(causal, g * (f * slope * derivative + f_slope * polynomial), 0.)
    return weights, ds, sums, weights.T, ds.T


def _scores(case, time, rng):
    if case == "unit":
        return rng.standard_normal((time, time))
    if case == "wide":
        return 6. * rng.standard_normal((time, time))  # beyond +-20: e^-|s| below 2^-6 and near underflow
    if case == "exact zeros":
        qk = rng.integers(-1, 2, size=(2, time, 16)).astype(np.float64)
        return qk[0] @ qk[1].T / 4
    # Extremes: far below zero (softplus near e^s, then underflowing; e^s underflowing) and far above it.
    return rng.choice([-120., -87., -40., -9., -4.2, -4.1, -1.28, -1e-3, 0., 1e-3, 4.2, 9., 40.], size=(time, time))


TILE_CASES = [pytest.param(transform, prefactor, degree, case,
                           id=f"{transform}-{prefactor}-m{degree}-{case.replace(' ', '-')}")
              for transform, prefactor in NEW_FAMILIES for degree in (1, 4, 8, 9) for case in ("unit", "wide")] + [
    pytest.param(transform, prefactor, 8, case, id=f"{transform}-{prefactor}-m8-{case.replace(' ', '-')}")
    for transform, prefactor in NEW_FAMILIES for case in ("exact zeros", "extremes")]


@pytest.mark.parametrize("transform,prefactor,degree,case", TILE_CASES)
def test_tile_mathematics_match_explicit_float64(transform, prefactor, degree, case):
    """The forward weights, dS and every dQ sum of a tile crossing the diagonal, and the transposed dK/dV tile's
    W and dS, against explicit f(s) phi(s)^r and the product rule over all pairs; masked entries hold scores of
    +-1e15, which the kernels replace by zero. Exp, sigmoid and swish agree to rounding; softplus, as transform or
    prefactor, to its own approximation: below e^-|s| = 2^-6 the cubic series of ln(1 + y), relative error < 1e-6
    (the GPU's ex2/lg2.approx add about 2^-22 there)."""
    family = NEW_FAMILIES[transform, prefactor]
    rng = np.random.default_rng(degree + 17 * len(case) + 3 * len(prefactor))
    time, orders = 37, degree + 1
    s = _scores(case, time, rng)
    causal = np.tril(np.ones((time, time), bool))
    s_masked = np.where(causal, s, rng.choice([-1e15, 1e15], size=s.shape))
    dp = rng.standard_normal((time, time))
    c = rng.standard_normal((orders, time)) / np.array([math.factorial(r) for r in range(orders)])[:, None]
    if case == "exact zeros":
        assert (s[causal] == 0).mean() > .05
    with jax.enable_x64():
        got = [np.asarray(x) for x in _tile_terms(family, degree, jnp.asarray(s_masked), jnp.asarray(dp),
                                                   jnp.asarray(c))]
    want = _tile_reference(transform, prefactor, s, dp, c)
    tolerance = 2e-6 if "softplus" in (transform, prefactor) else 1e-11
    for name, x, y in zip(("W", "dS", "dQ sums", "W^T", "dS^T"), got, want, strict=True):
        assert x.shape == y.shape and np.isfinite(x).all(), name
        assert np.max(np.abs(x - y)) <= tolerance * np.max(np.abs(y)), (name, np.max(np.abs(x - y)) / np.max(np.abs(y)))
    for order in range(orders):  # every dc_r on its own scale
        scale = np.max(np.abs(want[2][order]))
        assert np.max(np.abs(got[2][order] - want[2][order])) <= tolerance * scale + 1e-300, order


@pytest.mark.parametrize("score", (-np.inf, np.inf, np.nan), ids=("-inf", "+inf", "nan"))
@pytest.mark.parametrize("transform,prefactor", list(NEW_FAMILIES), ids=FAMILY_IDS)
def test_a_nonfinite_causal_score_makes_its_row_nan(transform, prefactor, score):
    """A causal score of -inf (an FP32 dot product that overflowed), +inf or NaN makes its row NaN in the tile
    functions (weights, then the per-row statistic and finish), even for exp at -inf, whose zero meets the
    nonfinite polynomial; the pair's score gradient and its row's coefficient sums of orders >= 1 are nonfinite;
    every other row stays finite, and a masked entry holding the same score changes nothing. The blocked scan and
    strips, given q.k overflowing at the same pair, make the same row NaN."""
    from scaffolds.transformer.attention.core import attention_core

    family = NEW_FAMILIES[transform, prefactor]
    rng = np.random.default_rng(3)
    time, degree, row, column = 8, 2, 5, 2
    causal = np.tril(np.ones((time, time), bool))
    s = rng.standard_normal((time, time))
    s[row, column] = s[1, 3] = score  # (1, 3) is a masked entry
    tile = jnp.asarray(np.where(causal, s, 0.), jnp.float32)
    dp = jnp.asarray(np.where(causal, rng.standard_normal((time, time)), 0.), jnp.float32)
    c = rng.standard_normal((degree + 1, time))
    rows = [jnp.broadcast_to(jnp.asarray(x, jnp.float32)[:, None], (time, time)) for x in c]
    weights = jnp.where(causal, common.forward_weights(family, degree, tile, rows), 0.)
    statistic = common.forward_statistic(family, weights).sum(axis=1)
    output = np.asarray(common.forward_finish(family, degree, jnp.ones((time, 4), jnp.float32), statistic)[0])
    np.testing.assert_array_equal(np.isnan(output).all(axis=1), np.arange(time) == row)
    ds, sums = common.dq_terms(family, degree, tile, dp, rows)
    ds, sums = np.asarray(ds), np.asarray(jnp.stack(sums))
    others = np.ones((time, time), bool)
    others[row, column] = False
    assert not np.isfinite(ds[row, column]) and np.isfinite(ds[others]).all()
    assert not np.isfinite(sums[1:, row]).any() and np.isfinite(np.delete(sums, row, axis=1)).all()
    if np.isnan(score):
        return
    q = np.concatenate([np.zeros((1, time, 1, 1)), rng.standard_normal((1, time, 1, 7))], axis=-1)
    k = np.concatenate([np.zeros((1, time, 1, 1)), rng.standard_normal((1, time, 1, 7))], axis=-1)
    q[0, row, 0, 0], k[0, column, 0, 0] = 2. ** 100, np.sign(score) * 2. ** 100
    v = rng.standard_normal((1, time, 1, 4))
    for implementation in ("scan", "strips"):
        reference = attention_core(
            *(jnp.asarray(x, jnp.float32) for x in (q, k, v)), backend="blocked", blocked_implementation=implementation,
            attention_degree=degree, attention_normalization=False,
            query_polynomial_coefficients=jnp.asarray(c.T[None, :, None, :], jnp.float32),
            attention_coefficient_activation="tanh", score_transform=transform, score_prefactor=prefactor)
        np.testing.assert_array_equal(np.isnan(np.asarray(reference[0, :, 0])).all(axis=-1), np.arange(time) == row)


# The prefactor f = phi = softplus on a degree-m polynomial is the input power z p(z) of degree m + 1.

def _coefficient_operands(degree=8, time=48, seed=4):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(0.7 * rng.standard_normal((2, time, 2, 16)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((2, time, 2, 8)), jnp.float32)
    factorials = np.array([math.factorial(r) for r in range(degree + 1)])
    c = jnp.asarray(rng.standard_normal((2, time, 2, degree + 1)) / factorials, jnp.float32)
    return q, k, v, c


@pytest.mark.parametrize("implementation", ("scan", "strips"))
def test_softplus_prefactor_equals_the_input_power_on_the_blocked_backend(implementation):
    """softplus(s) p(softplus(s)) of degree m equals the degree-(m+1) polynomial [0, c_0..c_m] at softplus(s), the
    input power's structural zero, forward and in every gradient, to FP64 rounding."""
    from scaffolds.transformer.attention.core import attention_core

    with jax.enable_x64():
        q, k, v, c = (x.astype(jnp.float64) for x in _coefficient_operands())
        cotangent = jnp.asarray(np.random.default_rng(9).standard_normal(v.shape), jnp.float64)
        options = dict(backend="blocked", blocked_implementation=implementation, attention_normalization=False,
                       attention_coefficient_activation="tanh", score_transform="softplus")

        def prefactor(q, k, v, c):
            return attention_core(q, k, v, attention_degree=8, query_polynomial_coefficients=c,
                                  score_prefactor="softplus", **options)

        def power(q, k, v, c):
            shifted = jnp.concatenate([jnp.zeros_like(c[..., :1]), c], axis=-1)
            return attention_core(q, k, v, attention_degree=9, query_polynomial_coefficients=shifted, **options)

        results = [jax.vjp(function, q, k, v, c) for function in (prefactor, power)]
        (first, pullback), (second, power_pullback) = results
        # The scan evaluates its blocks in FP32; the strips keep the operands' FP64.
        tolerance = 1e-12 if implementation == "strips" else 2e-6
        for got, want in zip((first, *pullback(cotangent)), (second, *power_pullback(cotangent)), strict=True):
            got, want = np.asarray(got), np.asarray(want)
            assert np.max(np.abs(got - want)) <= tolerance * np.max(np.abs(want))


def _sos_model(**changes):
    from scaffolds.transformer.configs import ModelConfig

    settings = dict(
        vocab_size=32, max_seq_len=128, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2, qk_head_dim=16,
        value_head_dim=16, intermediate_size=64, attention_backend="blocked", attention_blocked_implementation="strips",
        attention_degree=8, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
        attention_coefficient_activation="tanh", attention_coefficient_parameterization="sum_of_squares",
        attention_coefficient_amplitude_initial=0.5, attention_coefficient_projection_initial_std=0.2,
        attention_coefficient_kernel_weight_decay=True, attention_output_scaling="head_rms_norm", qk_norm=True,
        qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        initialization="transformer_truncated_reference")
    settings.update(changes)
    return ModelConfig(**settings)


def test_softplus_prefactor_sos_model_equals_the_input_power_sos_model():
    """The D-family form softplus(s) h_t(softplus(s)) (core degree eight, input power 0, transform and prefactor
    softplus) and the control's z h_t(z) (degree nine, input power one) have the same parameters and, with the same
    values, the same loss and gradients on the blocked backend, to FP32 rounding."""
    from scaffolds.transformer.configs import Config, validate_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.state import create_state
    import types

    configs = []
    for model in (_sos_model(attention_score_transform="softplus", attention_score_prefactor="softplus"),
                  _sos_model(attention_degree=9, attention_coefficient_input_power=1,
                             attention_score_transform="softplus")):
        config = Config(model=model)
        config.data.sequence_length, config.data.synthetic_vocab_size = 129, 32
        config.optim.total_steps, config.optim.warmup_steps = 4, 0
        validate_config(config)
        configs.append(config)
    states = [create_state(config, jax.random.PRNGKey(0)) for config in configs]
    leaves = [jax.tree.leaves(state.params) for state in states]
    assert [x.shape for x in leaves[0]] == [x.shape for x in leaves[1]]
    for a, b in zip(*leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    batch = jnp.asarray(np.random.default_rng(1).integers(0, 32, (2, 129)), jnp.int32)
    results = [loop.loss_and_grads(types.SimpleNamespace(graphdef=state.graphdef, params=state.params), batch,
                                   jax.random.PRNGKey(2)) for state in states]
    losses = [float(result[0]) for result in results]
    assert np.isfinite(losses).all() and abs(losses[0] - losses[1]) <= 2e-6 * abs(losses[1])
    for a, b in zip(*(jax.tree.leaves(result[1]) for result in results), strict=True):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        assert np.linalg.norm(a - b) <= 1e-4 * np.linalg.norm(b) + 1e-12


# Dispatch and admission.

def _bf16_operands(degree=4, time=256, width=64):
    rng = np.random.default_rng(11)
    q, k = (jnp.asarray(rng.standard_normal((1, time, 2, width)), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((1, time, 2, 64)), jnp.bfloat16)
    c = jnp.asarray(rng.standard_normal((1, time, 2, degree + 1)), jnp.float32)
    return q, k, v, c


@pytest.mark.parametrize("transform,prefactor", list(PREFACTOR_FAMILIES), ids=FAMILY_IDS[:8])
def test_attention_core_reaches_the_prefactor_kernels(transform, prefactor, monkeypatch):
    """attention_core on backend mosaic calls the transformed entry point once with the transform and the
    prefactor, never the plain query kernels; the diagnostics come from the blocked scan of the same weights."""
    from scaffolds.transformer.attention.core import attention_core

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    calls = []
    transformed = api.transformed_query_polynomial_attention

    def counted(*args, **kwargs):
        calls.append((kwargs["transform"], kwargs.get("prefactor", "none")))
        return transformed(*args, **kwargs)

    def plain(*args, **kwargs):
        raise AssertionError("the plain query kernels were called")

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    monkeypatch.setattr(api, "query_polynomial_attention", plain)
    q, k, v, c = _bf16_operands()
    output, diagnostics = jax.eval_shape(lambda: attention_core(
        q, k, v, backend="mosaic", attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
        attention_coefficient_activation="tanh", score_transform=transform, score_prefactor=prefactor,
        return_diagnostics=True))
    assert calls == [(transform, prefactor)] and output.shape == v.shape and output.dtype == jnp.bfloat16
    assert "query_coefficient_mean" in diagnostics


def test_prefactor_refusals_in_attention_core(monkeypatch):
    """No other kernel may ignore the prefactor: decay, key factors, score biases and the fused backend refuse it."""
    from scaffolds.transformer.attention.core import attention_core

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    q, k, v, c = _bf16_operands()
    base = dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
                attention_coefficient_activation="tanh", score_prefactor="exp")
    log_decay = jnp.zeros(q.shape[:3], jnp.float32)
    for extra in (dict(backend="mosaic", query_log_decay=log_decay, query_decay_mode="query_distance"),
                  dict(backend="mosaic", key_polynomial_factors=c),
                  dict(backend="mosaic", score_bias_row_shift=jnp.zeros((*q.shape[:3], 4), jnp.float32)),
                  dict(backend="fused")):
        with pytest.raises(ValueError):
            jax.eval_shape(lambda: attention_core(q, k, v, **base, **extra))


def _mosaic_recipe(run, **model):
    from pathlib import Path

    from scaffolds.transformer.configs import load_config

    root = Path(__file__).resolve().parents[3] / "experiments" / "transformer_pp" / "runs"
    config = load_config(root / run / "config.yaml")
    config.model = replace(config.model, **model)
    return config


# The pinned partners of the planned runs, with the fields each run changes.
PLANNED = {
    "A1 swish pointwise": ("r266c5dd1f935", dict(attention_score_transform="swish")),
    "B1 SoS softplus input power d_K 4": ("r841f35a24241", dict(qk_head_dim=4)),
    "B4 exp d_K 4": ("r6a76e7fbde45", dict(qk_head_dim=4)),
    "C1 static softplus": ("r9c69dd7cfc8d", dict(attention_score_transform="softplus")),
    "C2 static softplus prefactor": ("r9c69dd7cfc8d", dict(attention_score_transform="softplus",
                                                           attention_score_prefactor="softplus")),
    "C3 query softplus prefactor": ("ra3b54e2489f8", dict(attention_score_prefactor="softplus")),
    **{f"D {transform} {prefactor}": ("reeefbcc19abf", dict(attention_score_transform=transform,
                                                            attention_score_prefactor=prefactor))
       for transform in ("identity", "softplus") for prefactor in common.SCORE_PREFACTORS},
}


@pytest.mark.parametrize("name", list(PLANNED))
def test_planned_recipes_validate_and_resolve_on_h100_and_b200_only(name, monkeypatch):
    from scaffolds.transformer.attention.backends import selection
    from scaffolds.transformer.configs import validate_config

    run, changes = PLANNED[name]
    config = _mosaic_recipe(run, **changes)
    validate_config(config)
    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")

    def resolve(capabilities):
        monkeypatch.setattr(selection, "_cuda_compute_capabilities", lambda: capabilities)
        return selection.resolve_attention_backend(config.model, dtype="bfloat16", sequence_length=2048)

    if config.model.attention_backend == "fused":  # the exponential at narrow keys
        assert resolve(("9.0",)) == "fused"
        return
    assert resolve(("9.0",)) == resolve(("10.0",)) == "mosaic"
    with pytest.raises(ValueError, match=r"admits H100 \(sm90\) and B200 \(sm100\) only"):
        resolve(("10.3",))


REFUSED = {
    "abs transform with a prefactor": ("ra3b54e2489f8", dict(attention_score_transform="abs",
                                                             attention_score_prefactor="exp")),
    "swish transform of a learned polynomial": ("ra3b54e2489f8", dict(attention_score_transform="swish")),
    "sigmoid transform of a learned polynomial": ("ra3b54e2489f8", dict(attention_score_transform="sigmoid")),
    "swish pointwise with a prefactor": ("r266c5dd1f935", dict(attention_score_transform="swish",
                                                               attention_score_prefactor="exp")),
    "static SoS with a prefactor": ("r9c69dd7cfc8d", dict(attention_coefficient_parameterization="sum_of_squares",
                                                          attention_coefficient_scale="none",
                                                          attention_score_prefactor="softplus")),
    "fixed Taylor with a prefactor on mosaic": ("r9c69dd7cfc8d", dict(attention_coefficients="fixed",
                                                                      attention_coefficient_scale="none",
                                                                      attention_score_prefactor="softplus")),
    "polynomial prefactor on fused": ("ra3b54e2489f8", dict(attention_backend="fused",
                                                            attention_score_prefactor="softplus")),
    "polynomial prefactor with a forget gate on mosaic": ("ra3b54e2489f8", dict(
        attention_score_prefactor="exp", attention_forget_gate=True, attention_forget_gate_mode="query_distance")),
    "exp prefactor of the fused exponential": ("r6a76e7fbde45", dict(attention_score_prefactor="exp")),
    "swish transform of the fused exponential": ("r6a76e7fbde45", dict(attention_score_transform="swish")),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_unsupported_recipes_are_refused(name):
    from scaffolds.transformer.configs import validate_config

    run, changes = REFUSED[name]
    with pytest.raises(ValueError):
        validate_config(_mosaic_recipe(run, **changes))


@pytest.mark.parametrize("degree", (9, 12, 13))
@pytest.mark.parametrize("transform,prefactor", (("softplus", "softplus"), ("identity", "exp"), ("softplus", "swish")))
def test_mosaic_refuses_a_prefactor_above_its_qualified_degree(transform, prefactor, degree):
    """The GPU suites qualify the prefactor families at degrees 1, 4 and 8, so Mosaic takes a prefactor recipe up to
    degree eight; above it the blocked backend still takes the recipe, and Mosaic the transform alone."""
    from scaffolds.transformer.configs import validate_config

    changes = dict(attention_score_transform=transform, attention_degree=degree)
    with pytest.raises(ValueError, match="GPU-qualified at degrees up to 8"):
        validate_config(_mosaic_recipe("ra3b54e2489f8", **changes, attention_score_prefactor=prefactor))
    validate_config(_mosaic_recipe("ra3b54e2489f8", **changes, attention_score_prefactor=prefactor,
                                   attention_backend="blocked"))
    if transform != "identity":
        validate_config(_mosaic_recipe("ra3b54e2489f8", **changes))


def test_polynomial_prefactor_refuses_unqualified_schedules():
    from scaffolds.transformer.configs import validate_config

    config = _mosaic_recipe("ra3b54e2489f8", attention_score_prefactor="softplus")
    config.optim.query_coefficient_learning_rate_scale = 2.0
    with pytest.raises(ValueError, match="polynomial prefactor"):
        validate_config(config)


@pytest.mark.parametrize("name", ["A1 swish pointwise", "C2 static softplus prefactor", "C3 query softplus prefactor",
                                  "D identity exp", "D softplus swish"])
def test_provenance_records_the_prefactor_and_its_kernels(name):
    """The attention identity records the prefactor, its definition in the formula before the output-scaling
    clause, the Mosaic implementation named after it, and signed weights for swish."""
    from scaffolds.transformer.attention.score_transform import PREFACTOR_FORMULAS
    from scaffolds.transformer.training.diagnostics import attention_identity

    run, changes = PLANNED[name]
    config = _mosaic_recipe(run, **changes)
    identity = attention_identity(config)
    prefactor, transform = config.model.attention_score_prefactor, config.model.attention_score_transform
    if prefactor == "none":
        assert identity["attention_kernel"] == "swish_unnormalized" and identity["attention_signed"] is True
        assert identity["attention_implementation"] == "pallas_mosaic_gpu_query_polynomial_swish_scores"
        assert identity["attention_formula"].startswith("sum_(i<=t) swish(s_ti) v_i")
        return
    assert identity["attention_score_prefactor"] == prefactor and identity["attention_score_transform"] == transform
    assert PREFACTOR_FORMULAS[prefactor] in identity["attention_formula"]
    assert identity["attention_formula"].index(PREFACTOR_FORMULAS[prefactor]) < identity["attention_formula"].index(
        "RMSNorm over value channels")
    scores = "" if transform == "identity" else f"_{transform}_scores"
    assert identity["attention_implementation"] == f"pallas_mosaic_gpu_query_polynomial{scores}_times_{prefactor}"
    assert identity["attention_signed"] is True


DIAGNOSTIC_LABELS = {
    "C1 static softplus": "learned_polynomial_of_softplus_scores",
    "C2 static softplus prefactor": "learned_polynomial_of_softplus_scores_times_softplus_of_raw_scores",
    "C3 query softplus prefactor": "query_polynomial_of_softplus_scores_times_softplus_of_raw_scores",
    "D identity exp": "unshifted_query_polynomial_times_exp_of_raw_scores",
    "D softplus sigmoid": "query_polynomial_of_softplus_scores_times_sigmoid_of_raw_scores",
}


@pytest.mark.parametrize("name", list(DIAGNOSTIC_LABELS))
def test_diagnostics_name_the_transform_and_the_prefactor(name, monkeypatch):
    """The planned recipes' diagnostics at a two-layer shape on their Mosaic route (the transformed kernels replaced
    by the blocked scan of the same weights): kernel_values names the polynomial's transform and its prefactor of the
    raw score, and the score-transform report states W = f(s) p(z) with f's definition, its decreasing measure being
    the polynomial factor's dp/dz."""
    from scaffolds.transformer.attention.backends import selection
    from scaffolds.transformer.attention.backends.reference import blocked_backend_attention
    from scaffolds.transformer.attention.score_transform import FORMULAS, PREFACTOR_FORMULAS
    from scaffolds.transformer.configs import validate_config
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
    from scaffolds.transformer.training.state import create_state

    def scan(q, k, v, c, *, transform, prefactor="none", precision=None):
        functions = {key: value for key, value in (("score_transform", transform), ("score_prefactor", prefactor))
                     if value not in ("identity", "none")}
        return blocked_backend_attention(q, k, v, implementation="scan", attention_degree=c.shape[-1] - 1,
                                         attention_normalization=False, query_polynomial_coefficients=c, **functions)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", scan)
    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capabilities", lambda: ("9.0",))
    run, changes = PLANNED[name]
    config = _mosaic_recipe(run, **changes, vocab_size=64, max_seq_len=128, num_layers=2, hidden_size=32,
                            num_heads=2, num_kv_heads=2, intermediate_size=64)
    config.data.source, config.data.sequence_length, config.data.synthetic_vocab_size = "synthetic", 129, 64
    config.data.batch_size = config.train.micro_batch_size = 2
    validate_config(config)
    state = create_state(config, jax.random.PRNGKey(0))
    report = attention_diagnostics(state, config, *fixed_diagnostic_sample(config))
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == DIAGNOSTIC_LABELS[name]
    section = report["coefficients"].get("score_transform")
    if config.model.attention_coefficient_conditioning == "static":
        assert section is None  # static coefficients report their values, not the transformed inputs
        return
    transform, prefactor = config.model.attention_score_transform, config.model.attention_score_prefactor
    if transform == "identity":
        assert section is None
        return
    assert section["formula"] == (f"W(t,i) = f(s_ti) p(z_ti), p(z_ti) = sum_r c_r(x_t) z_ti^r, {FORMULAS[transform]}, "
                                  f"{PREFACTOR_FORMULAS[prefactor]}")
    assert section["decreasing_formula"].startswith("dp/dz = sum_(r=1..m) r c_r(x_t) z_ti^(r-1) < 0")
    assert "without the prefactor f" in section["decreasing_formula"]


def test_model_layers_reach_the_prefactor_kernels(monkeypatch):
    """A two-layer SoS model with the exp prefactor on backend mosaic (resolved as on an H100) reaches the
    transformed kernels with its prefactor in every layer of its loss and gradient program."""
    from scaffolds.transformer.attention.backends import selection
    from scaffolds.transformer.configs import Config, validate_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.state import create_state
    import types

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capabilities", lambda: ("9.0",))
    calls = []
    transformed = api.transformed_query_polynomial_attention
    monkeypatch.setattr(api, "transformed_query_polynomial_attention",
                        lambda *a, **kw: calls.append((kw["transform"], kw.get("prefactor"))) or transformed(*a, **kw))
    model = _sos_model(attention_backend="mosaic", attention_blocked_implementation="scan", qk_head_dim=64,
                       value_head_dim=64, attention_score_prefactor="exp")
    config = Config(model=model)
    config.data.sequence_length, config.data.synthetic_vocab_size = 129, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.train.dtype = "bfloat16"
    validate_config(config)
    state = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.random.default_rng(1).integers(0, 32, (2, 129)), jnp.int32)

    def loss_and_grads(params):
        return loop.loss_and_grads(types.SimpleNamespace(graphdef=state.graphdef, params=params), batch,
                                   jax.random.PRNGKey(2))

    jax.make_jaxpr(loss_and_grads)(state.params)
    assert calls == [("identity", "exp")] * 2
