"""CPU checks of the Mosaic transformed query-polynomial kernels (``mosaic.api.transformed_query_polynomial_attention``).

The query polynomial evaluated at z = phi(s), phi = softplus or abs, runs on the query polynomial's own Hopper and
Blackwell kernels with phi evaluated in their causal tiles (``common.transformed_score``). Checked here, on a CPU
host: the families' counts and flags (those of the query family; approximate math for softplus alone); the
support predicate, which is the query family's; every Hopper kernel lowers for cuda in exactly the shared memory
``hopper_smem_bytes`` counts, the query family's, and one byte less fails; the Blackwell kernels and both
architectures' entry points lower forward and with every gradient; the entry point refuses the identity, unknown
transforms and the operands the query kernels refuse; and the tile mathematics of ``common`` (the functions the
kernels call on each tile) on whole causal matrices in FP64, against explicit all-pairs references, with the
kernels' masking of a tile crossing the diagonal: very negative and very positive scores, exact zeros, masked
entries holding huge scores, and special values; a nonfinite causal score (-inf from an overflowed dot product,
+inf, NaN) makes its row NaN, as the blocked reference does; and backend selection and the entry point admit the
transformed kernels on H100 and B200, while refusing B300. The kernels' numerics on GPUs are qualified by
``tests/test_mosaic_score_transform.py``.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api, blackwell, common, hopper

TRANSFORMS = ("softplus", "abs")
FAMILIES = {"softplus": common.QUERY_SOFTPLUS, "abs": common.QUERY_ABS, "sigmoid": common.QUERY_SIGMOID,
            "swish": common.QUERY_SWISH}


def _lowers_for_cuda(function, *operands):
    """Trace and lower ``function`` for cuda on any host, building every Mosaic GPU kernel (through a fresh
    wrapper: JAX would reuse an earlier lowering of the same function)."""
    jax.jit(lambda *arguments: function(*arguments)).trace(*operands).lower(lowering_platforms=("cuda",))


def test_families_count_like_the_query_family():
    """Vectors, sums, keys and heads are the query family's for every degree; no Pair, no accumulating sums, no
    gate; approximate exp2/log2 only where a transcendental is evaluated (softplus)."""
    assert api.TRANSFORM_FAMILY == FAMILIES
    # Without a prefactor; prefactor families (tests/test_score_prefactor.py) count alike.
    assert set(common.TRANSFORMED_QUERY_FAMILIES) == set(FAMILIES.values()) | set(common.PREFACTOR_FAMILIES)
    for family in FAMILIES.values():
        for degree in range(1, 13):
            assert common.vector_counts(family, degree) == common.vector_counts(common.QUERY, degree)
            assert common.key_counts(family, degree) == (0, 0) and common.head_count(family, degree) == 0
            assert common.hopper_smem_bytes(family, degree, 64, 64) == common.hopper_smem_bytes(
                common.QUERY, degree, 64, 64)
        assert not common.has_pair(family) and not common.accumulates(family)
        assert family not in (*common.DECAY_FAMILIES, *common.PER_ORDER_FAMILIES, *common.SCORE_BIAS_FAMILIES)
        assert api.default_precision(family, 8) == common.BF16
    assert common.approx_math(common.QUERY_SOFTPLUS) and not common.approx_math(common.QUERY_ABS)
    assert common.approx_math(common.QUERY_SIGMOID) and common.approx_math(common.QUERY_SWISH)
    assert not common.approx_math(common.QUERY)


def test_support_is_query_family_plus_scoped_softplus_degree13():
    for transform in TRANSFORMS:
        for width in api.WIDTHS:
            for time in (256, 512, 1024, 2048, 4096):
                for degree in range(0, 15):
                    extension = transform == "softplus" and degree == 13 and width == 64 and time in (512, 2048)
                    assert api.query_degree_supported(degree, width, time, transform=transform) == (
                        api.query_degree_supported(degree, width, time) or extension)
        assert not api.query_degree_supported(8, 64, 2048, transform=transform, gate="query")
    assert not api.query_degree_supported(8, 64, 2048, transform="square")
    assert api.MAX_QUERY_DEGREE == 12
    assert [m for m in range(1, 15) if api.query_degree_supported(m, 64, 2048, transform="softplus")] == list(
        range(1, 14))


@pytest.mark.parametrize("transform,degree", [(t, d) for t in TRANSFORMS for d in (4, 8, 12)] + [("softplus", 13)])
def test_hopper_kernels_lower_in_exactly_their_counted_shared_memory(transform, degree):
    """H100's forward, dQ and dK/dV lower at 2,048 tokens and d_K = 64 in exactly the bytes hopper_smem_bytes
    counts (the query family's), and fail one byte below."""
    from jax.experimental.mosaic.gpu.core import artificial_shared_memory_limit

    family = FAMILIES[transform]
    q = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
    vectors = jax.ShapeDtypeStruct((degree + 1, 1, 2, 2048), jnp.float32)
    options = dict(family=family, degree=degree, scale=.125, precision="bf16")
    kernels = {"forward": (lambda q, k, v, x: hopper.forward(q, k, v, x, **options), (q, q, q)),
               "dq": (lambda q, k, v, do, x: hopper.dq(q, k, v, do, x, **options), (q, q, q, q)),
               "dkv": (lambda q, k, v, do, x: hopper.dkv(q, k, v, do, x, **options), (q, q, q, q))}
    sizes = common.hopper_smem_bytes(family, degree, 64, 64)
    assert sizes == common.hopper_smem_bytes(common.QUERY, degree, 64, 64)
    for name, (kernel, arrays) in kernels.items():
        with artificial_shared_memory_limit(sizes[name]):
            _lowers_for_cuda(kernel, *arrays, vectors)
        with artificial_shared_memory_limit(sizes[name] - 1), pytest.raises(ValueError, match="shared memory"):
            _lowers_for_cuda(kernel, *arrays, vectors)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_blackwell_kernels_lower_with_the_query_rings(transform):
    """B300's forward, dQ, dK and dV kernels lower at 2,048 tokens, d_K = 64 and degree eight with the stages the
    query family's backward uses there (three), and at degree twelve, the largest the query family fits."""
    family = FAMILIES[transform]
    for degree in ((8, 12, 13) if transform == "softplus" else (8, 12)):
        q = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
        vectors = jax.ShapeDtypeStruct((degree + 1, 1, 2, 2048), jnp.float32)
        options = dict(family=family, degree=degree, scale=.125, precision="bf16")
        assert common.blackwell_dkv_stages(64, 64, degree + 1, 2048) == 3
        _lowers_for_cuda(lambda q, k, v, x: blackwell.forward(q, k, v, x, **options), q, q, q, vectors)
        _lowers_for_cuda(lambda q, k, v, do, x: blackwell.dq(q, k, v, do, x, **options, stages=3), q, q, q, q, vectors)
        for which in ("dk", "dv"):
            _lowers_for_cuda(lambda q, k, v, do, x: blackwell.dkv(q, k, v, do, x, which=which, stages=3, **options),
                             q, q, q, q, vectors)


@pytest.mark.parametrize("arch", ("hopper", "blackwell"))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_entry_point_lowers_with_every_gradient(monkeypatch, transform, arch):
    """The production entry point lowers forward and with every gradient at 2,048 tokens: degree eight at Q/K
    widths 8 (zero-padded to 16), 64 and 128, and degree twelve at width 64 (B300 admitted here, as the GPU
    module admits it to qualify its kernels)."""
    monkeypatch.setattr(api, "_kernels", lambda: (arch, {"hopper": hopper, "blackwell": blackwell}[arch]))
    monkeypatch.setattr(api, "TRANSFORM_ARCHITECTURES", ("hopper", "blackwell"))
    monkeypatch.setattr(api, "_device_compute_capability", lambda: "9.0" if arch == "hopper" else "10.3")
    monkeypatch.setattr(api, "TRANSFORM_COMPUTE_CAPABILITIES", ("9.0", "10.0", "10.3"))

    def attend(q, k, v, c):
        return api.transformed_query_polynomial_attention(q, k, v, c, transform=transform)

    cases = ((8, 8), (8, 64), (8, 128), (12, 64)) + (((13, 64),) if transform == "softplus" else ())
    for degree, width in cases:
        assert api.query_degree_supported(degree, width, 2048, transform=transform)
        q = jax.ShapeDtypeStruct((1, 2048, 2, width), jnp.bfloat16)
        v = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
        c = jax.ShapeDtypeStruct((1, 2048, 2, degree + 1), jnp.float32)
        _lowers_for_cuda(attend, q, q, v, c)
        _lowers_for_cuda(lambda *a: jax.vjp(attend, *a)[1](jnp.ones(a[2].shape, a[2].dtype)), q, q, v, c)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("capability", ("9.0", "10.0", "10.3"))
def test_entry_point_admits_h100_and_b200_only(monkeypatch, transform, capability):
    """B200 admission does not admit the B300 sharing its kernels, including restored models without selection."""
    q = k = v = jax.ShapeDtypeStruct((1, 256, 2, 64), jnp.bfloat16)
    c = jax.ShapeDtypeStruct((1, 256, 2, 9), jnp.float32)
    attend = lambda *a: api.transformed_query_polynomial_attention(*a, transform=transform)
    assert api.TRANSFORM_ARCHITECTURES == ("hopper", "blackwell")
    assert api.TRANSFORM_COMPUTE_CAPABILITIES == ("9.0", "10.0")
    monkeypatch.setattr(api, "_device_compute_capability", lambda: capability)
    if capability == "9.0":
        monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    else:
        monkeypatch.setattr(api, "_kernels", lambda: ("blackwell", blackwell))
    if capability in ("9.0", "10.0"):
        assert jax.eval_shape(attend, q, k, v, c).shape == v.shape
    else:
        with pytest.raises(ValueError, match=r"admits H100 \(sm90\) and B200 \(sm100\) only"):
            jax.eval_shape(attend, q, k, v, c)


@pytest.mark.parametrize("transform,width,time", (
    ("softplus", 64, 256), ("softplus", 64, 1024), ("softplus", 64, 4096),
    ("softplus", 32, 2048), ("softplus", 128, 2048), ("abs", 64, 2048),
))
def test_degree13_entry_point_rejects_unqualified_shapes(monkeypatch, transform, width, time):
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    q = jax.ShapeDtypeStruct((1, time, 1, width), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, time, 1, 64), jnp.bfloat16)
    c = jax.ShapeDtypeStruct((1, time, 1, 14), jnp.float32)
    with pytest.raises(ValueError, match="1 <= m <= 12"):
        jax.eval_shape(lambda *a: api.transformed_query_polynomial_attention(*a, transform=transform), q, q, v, c)


def test_degree13_raw_entry_point_remains_rejected(monkeypatch):
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    q = jax.ShapeDtypeStruct((1, 2048, 1, 64), jnp.bfloat16)
    c = jax.ShapeDtypeStruct((1, 2048, 1, 14), jnp.float32)
    with pytest.raises(ValueError, match="1 <= m <= 12"):
        jax.eval_shape(api.query_polynomial_attention, q, q, q, c)


@pytest.mark.parametrize("change", ("identity", "unknown", "coefficient dtype", "degree too high", "coefficient shape",
                                    "width", "value width", "time", "precision"))
def test_invalid_operands_raise_before_tracing(monkeypatch, change):
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    shape = (1, 256, 2)
    q = k = v = jnp.zeros((*shape, 64), jnp.bfloat16)
    c = jnp.zeros((*shape, 5), jnp.float32)
    keywords = dict(transform="softplus")
    if change == "identity":
        keywords["transform"] = "identity"
    elif change == "unknown":
        keywords["transform"] = "relu"
    elif change == "coefficient dtype":
        c = c.astype(jnp.bfloat16)
    elif change == "degree too high":
        c = jnp.zeros((*shape, api.MAX_QUERY_DEGREE + 2), jnp.float32)
    elif change == "coefficient shape":
        c = c[:, :-1]
    elif change == "width":
        q = k = jnp.zeros((*shape, 48), jnp.bfloat16)
    elif change == "value width":
        v = jnp.zeros((*shape, 32), jnp.bfloat16)
    elif change == "time":
        q, k, v, c = (x[:, :200] for x in (q, k, v, c))
    elif change == "precision":
        keywords["precision"] = "fp8"
    with pytest.raises(ValueError):
        jax.eval_shape(lambda *a: api.transformed_query_polynomial_attention(*a, **keywords), q, k, v, c)


# Tile mathematics: the functions every kernel calls on its tiles, on one whole causal matrix as a tile crossing
# the diagonal, with the kernels' masking (masked scores and dP enter as zero; masked W and dS leave as zero).

def _tile_terms(family, degree, s, dp, coefficients):
    """(W, dS, dQ sums, W^T, dS^T) from ``common``: the forward and dQ tile with queries in rows, and the
    transposed dK/dV tile with keys in rows and each query's coefficients broadcast along its column."""
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


def _phi(transform, s):
    """phi(s) and phi'(s) in FP64: logaddexp and the logistic function, or |s| and sign(s) (0 at s = 0)."""
    if transform == "softplus":
        return np.logaddexp(0., s), np.exp(-np.logaddexp(0., -s))
    return np.abs(s), np.sign(s)


def _tile_reference(transform, s, dp, c):
    """Explicit W = sum_r c_r z^r, dS = G phi'(s) sum_r r c_r z^(r-1) and dc_r = sum_i G z^r over causal pairs."""
    time, orders = s.shape[0], c.shape[0]
    causal = np.arange(time)[:, None] >= np.arange(time)[None, :]
    z, slope = _phi(transform, s)
    g = np.where(causal, dp, 0.)
    weights = sum(c[r][:, None] * z ** r for r in range(orders))
    derivative = sum(r * c[r][:, None] * z ** (r - 1) for r in range(1, orders))
    sums = np.stack([(g * np.where(causal, z ** r, 0.)).sum(1) for r in range(orders)])
    weights, ds = np.where(causal, weights, 0.), np.where(causal, g * derivative * slope, 0.)
    return weights, ds, sums, weights.T, ds.T


def _scores(case, time, rng):
    if case == "unit":
        return rng.standard_normal((time, time))
    if case == "wide":
        return 8. * rng.standard_normal((time, time))  # beyond +-20: e^-|s| below 2^-6 and near underflow
    if case == "exact zeros":
        qk = rng.integers(-1, 2, size=(2, time, 16)).astype(np.float64)
        return qk[0] @ qk[1].T / 4
    # Extremes: scores far below zero (softplus near e^s, then underflowing to zero) and far above it.
    return rng.choice([-120., -87., -40., -9., -4.2, -4.1, -1e-3, 0., 1e-3, 4.2, 9., 40.], size=(time, time))


TILE_CASES = [pytest.param(transform, degree, case, id=f"{transform}-m{degree}-{case.replace(' ', '-')}")
              for transform in TRANSFORMS for degree in (1, 4, 8, 12) for case in ("unit", "wide")] + [
    pytest.param(transform, 8, case, id=f"{transform}-m8-{case.replace(' ', '-')}")
    for transform in TRANSFORMS for case in ("exact zeros", "extremes")]


@pytest.mark.parametrize("transform,degree,case", TILE_CASES)
def test_tile_mathematics_match_explicit_float64(transform, degree, case):
    """The forward weights, dS and every dQ sum of a tile crossing the diagonal, and the transposed dK/dV tile's
    W and dS, against explicit phi(s)^r over all pairs; masked entries hold scores of +-1e15, which the kernels
    replace by zero. Abs agrees to rounding. Softplus agrees to its own approximation: below e^-|s| = 2^-6 the
    cubic series of ln(1 + y), relative error < 1e-6 (the GPU's ex2/lg2.approx add about 2^-22 there)."""
    family = FAMILIES[transform]
    rng = np.random.default_rng(degree + 17 * len(case))
    time, orders = 37, degree + 1
    s = _scores(case, time, rng)
    causal = np.tril(np.ones((time, time), bool))
    s_masked = np.where(causal, s, rng.choice([-1e15, 1e15], size=s.shape))
    dp = rng.standard_normal((time, time))
    c = rng.standard_normal((orders, time)) / np.array([math.factorial(r) for r in range(orders)])[:, None]
    if case == "exact zeros":
        assert (s[causal] == 0).mean() > .05
    with jax.enable_x64():
        got = [np.asarray(x) for x in _tile_terms(family, degree, jnp.asarray(s_masked), jnp.asarray(dp), jnp.asarray(c))]
    want = _tile_reference(transform, s, dp, c)
    tolerance = 2e-6 if transform == "softplus" else 1e-12
    for name, x, y in zip(("W", "dS", "dQ sums", "W^T", "dS^T"), got, want, strict=True):
        assert x.shape == y.shape and np.isfinite(x).all(), name
        assert np.max(np.abs(x - y)) <= tolerance * np.max(np.abs(y)), (name, np.max(np.abs(x - y)) / np.max(np.abs(y)))
    # Every dQ sum (dc_r) on its own scale, so that no order's error hides behind a larger order's.
    for order in range(orders):
        scale = np.max(np.abs(want[2][order]))
        assert np.max(np.abs(got[2][order] - want[2][order])) <= tolerance * scale + 1e-300, order


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64), ids=("fp32", "fp64"))
def test_transform_values_and_derivatives_at_special_scores(dtype):
    """softplus(s) and sigmoid(s), |s| and sign(s): exactly 1/2 at s = 0 for softplus within one rounding, and
    abs's derivative exactly 0 there (attention/score_transform.py's convention); limits at +inf; zero once
    e^s underflows; NaN for NaN, and softplus's input NaN at -inf (not its limit 0, a finite weight)."""
    with jax.enable_x64(dtype == jnp.float64):
        special = jnp.asarray([0., -0., 1e-30, -1e-30, 1., -1., 4.158, -4.158, 30., -30., 88., -88., 104., -104., 1e30,
                               -1e30, jnp.inf, -jnp.inf, jnp.nan], dtype)
        z, slope = (np.asarray(x, np.float64) for x in common.transformed_score(common.QUERY_SOFTPLUS, special,
                                                                               slope=True))
        z_abs, sign = (np.asarray(x, np.float64) for x in common.transformed_score(common.QUERY_ABS, special,
                                                                                  slope=True))
    s = np.asarray(special, np.float64)
    finite = np.isfinite(s)
    unit = np.finfo(np.dtype(dtype)).eps
    tiny = np.finfo(np.dtype(dtype)).tiny  # values below the normal range may flush to zero
    want_z, want_slope = _phi("softplus", s[finite])
    # Rounding |s| log2(e) (relative error about |s| eps in e^-|s|) and 1 + y (up to eps / y relative in
    # ln(1 + y) just above the series threshold y = 2^-6) bounds the error: within 1e-5 in FP32 and 1e-13 in FP64
    # below |s| = 105, where none of these points falls under the threshold.
    rtol = 1e-5 if dtype == jnp.float32 else 1e-13
    np.testing.assert_allclose(z[finite], want_z, rtol=rtol, atol=tiny)
    np.testing.assert_allclose(slope[finite], want_slope, rtol=max(rtol, 1e-12), atol=tiny)
    assert abs(slope[0] - .5) <= unit and abs(slope[1] - .5) <= unit and z[0] == z[1]
    assert z[-3] == np.inf and slope[-3] == 1. and np.isnan(z[-2]) and slope[-2] == 0.
    assert np.isnan(z[-1]) and np.isnan(slope[-1])
    assert (z[finite] >= 0).all() and (slope[finite] >= 0).all() and (slope[finite] <= 1).all()
    # Far below zero e^s (and so both values) underflows to zero rather than going negative or NaN.
    assert z[15] == 0. and slope[15] == 0.  # s = -1e30
    np.testing.assert_array_equal(z_abs[:-1], np.abs(s[:-1]))
    np.testing.assert_array_equal(sign[:-1], np.sign(s[:-1]))
    assert sign[0] == 0. and sign[1] == 0. and np.isnan(z_abs[-1])


@pytest.mark.parametrize("score", (-np.inf, np.inf, np.nan), ids=("-inf", "+inf", "nan"))
@pytest.mark.parametrize("transform", ("identity", *TRANSFORMS))
def test_a_nonfinite_causal_score_makes_its_row_nan_as_in_the_blocked_reference(transform, score):
    """A causal score of -inf (an FP32 dot product that overflowed), +inf or NaN makes its row NaN in the tile
    functions every kernel calls (weights, then the query family's per-row statistic and finish), for the query
    family and both transforms, and the pair's score gradient and its row's coefficient sums of orders >= 1 are
    nonfinite; every other row stays finite, and a masked entry holding the same score (the kernels replace it by
    zero) changes nothing. The blocked scan and strips, given q.k overflowing to -inf or +inf at the same pair,
    make the same row NaN."""
    from scaffolds.transformer.attention.core import attention_core

    family = {"identity": common.QUERY, **FAMILIES}[transform]
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
    assert np.isfinite(np.delete(output, row, axis=0)).all()
    ds, sums = common.dq_terms(family, degree, tile, dp, rows)
    ds, sums = np.asarray(ds), np.asarray(jnp.stack(sums))
    others = np.ones((time, time), bool)
    others[row, column] = False
    assert not np.isfinite(ds[row, column]) and np.isfinite(ds[others]).all()
    assert not np.isfinite(sums[1:, row]).any() and np.isfinite(np.delete(sums, row, axis=1)).all()
    if np.isnan(score):
        return
    # One channel that only this query and key use: q.k = +-2^200 overflows FP32 at this pair alone.
    q = np.concatenate([np.zeros((1, time, 1, 1)), rng.standard_normal((1, time, 1, 7))], axis=-1)
    k = np.concatenate([np.zeros((1, time, 1, 1)), rng.standard_normal((1, time, 1, 7))], axis=-1)
    q[0, row, 0, 0], k[0, column, 0, 0] = 2. ** 100, np.sign(score) * 2. ** 100
    v = rng.standard_normal((1, time, 1, 4))
    coefficients = c.T[None, :, None, :]
    for implementation in ("scan", "strips"):
        reference = attention_core(
            *(jnp.asarray(x, jnp.float32) for x in (q, k, v)), backend="blocked", blocked_implementation=implementation,
            attention_degree=degree, attention_normalization=False,
            query_polynomial_coefficients=jnp.asarray(coefficients, jnp.float32),
            attention_coefficient_activation="tanh", score_transform=transform)
        np.testing.assert_array_equal(np.isnan(np.asarray(reference[0, :, 0])).all(axis=-1), np.arange(time) == row)


# Dispatch: the model, attention_core and query_polynomial_attention reach the transformed kernels on backend
# mosaic, and the identity's programs are exactly the untransformed ones. Traced on the host with H100's kernels.

def _bf16_operands(degree=4, time=256):
    rng = np.random.default_rng(11)
    q, k, v = (jnp.asarray(rng.standard_normal((1, time, 2, 64)), jnp.bfloat16) for _ in range(3))
    c = jnp.asarray(rng.standard_normal((1, time, 2, degree + 1)), jnp.float32)
    return q, k, v, c


def _jaxpr(function, *arguments):
    import re

    text = str(jax.make_jaxpr(function)(*arguments))
    return re.sub(r"func_src_info='[^']*'", "", re.sub(r"0x[0-9a-f]+", "0x?", text))


def test_identity_on_mosaic_traces_the_untransformed_programs(monkeypatch):
    """An explicit identity transform traces the same programs as no transform (forward, VJP and diagnostics
    through attention_core and query_polynomial_attention on backend mosaic); a transform traces others."""
    from scaffolds.transformer.attention.core import attention_core
    from scaffolds.transformer.attention.input_dependent import query_polynomial_attention

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    q, k, v, c = _bf16_operands()
    common_arguments = dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
                            attention_coefficient_activation="tanh")
    functions = {
        "core": lambda **extra: attention_core(q, k, v, backend="mosaic", **common_arguments, **extra),
        "core diagnostics": lambda **extra: attention_core(q, k, v, backend="mosaic", return_diagnostics=True,
                                                           **common_arguments, **extra),
        "query": lambda **extra: query_polynomial_attention(q, k, v, c, backend="mosaic", **extra),
        "core vjp": lambda **extra: jax.vjp(lambda q: attention_core(q, k, v, backend="mosaic", **common_arguments,
                                                                     **extra), q)[1](v),
    }
    for name, function in functions.items():
        assert _jaxpr(lambda: function()) == _jaxpr(lambda: function(score_transform="identity")), name
        assert _jaxpr(lambda: function()) != _jaxpr(lambda: function(score_transform="softplus")), name


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_mosaic_dispatch_reaches_the_transformed_kernels(transform, monkeypatch):
    """attention_core on backend mosaic calls the transformed entry point once with the transform, never the
    untransformed one, and its diagnostics, from the blocked scan of the same kernel, carry the transform's
    measurements."""
    from scaffolds.transformer.attention.core import attention_core

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    calls = []
    transformed = api.transformed_query_polynomial_attention

    def counted(*args, **kwargs):
        calls.append(kwargs["transform"])
        return transformed(*args, **kwargs)

    def untransformed(*args, **kwargs):
        raise AssertionError("the untransformed kernels were called")

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", counted)
    monkeypatch.setattr(api, "query_polynomial_attention", untransformed)
    q, k, v, c = _bf16_operands()
    output, diagnostics = jax.eval_shape(lambda: attention_core(
        q, k, v, backend="mosaic", attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
        attention_coefficient_activation="tanh", score_transform=transform, return_diagnostics=True))
    assert calls == [transform] and output.shape == v.shape and output.dtype == jnp.bfloat16
    assert {"score_transform_pair_count", "score_transform_raw_sum", "score_transform_input_sum",
            "score_transform_decreasing_count", "query_coefficient_mean"} <= set(diagnostics)


def _mosaic_model(transform):
    """A two-layer recipe of the tanh query polynomial at degree eight with the transform, on backend mosaic."""
    from scaffolds.transformer.configs import ModelConfig

    return ModelConfig(
        vocab_size=32, max_seq_len=128, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2, qk_head_dim=64,
        value_head_dim=64, intermediate_size=64, attention_backend="mosaic", attention_degree=8,
        attention_normalization=False, attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_amplitude_initial=0., attention_coefficient_projection_initial_std=.2,
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
        attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
        qk_norm_head_scale=True, initialization="transformer_truncated_reference",
        attention_score_transform=transform,
        attention_coefficient_constant_initial=0. if transform == "softplus" else None)


@pytest.mark.parametrize("capabilities", (("9.0",), ("10.0",), ("10.3",), ("10.0", "10.3"),
                                         ("9.0", "10.3")))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_backend_selection_admits_h100_and_b200_only(transform, capabilities, monkeypatch):
    """Require admitted devices throughout the allocation; generic Blackwell support cannot admit a B300."""
    from dataclasses import replace

    from scaffolds.transformer.attention.backends import selection

    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capabilities", lambda: capabilities)
    model = _mosaic_model(transform)

    def resolve(model):
        return selection.resolve_attention_backend(model, dtype="bfloat16", sequence_length=256)

    if all(item in ("9.0", "10.0") for item in capabilities):
        assert resolve(model) == "mosaic"
    else:
        with pytest.raises(ValueError, match=r"admits H100 \(sm90\) and B200 \(sm100\) only"):
            resolve(model)
    assert resolve(replace(model, attention_score_transform="identity")) == "mosaic"
    assert resolve(replace(model, attention_backend="blocked", attention_blocked_implementation="strips")) == "blocked"


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_model_layers_train_on_the_transformed_kernels(transform, monkeypatch):
    """A two-layer model on backend mosaic (resolved as on an H100) reaches the transformed kernels in every layer
    of its loss and gradient program, and its attention identity names them."""
    from scaffolds.transformer.attention.backends import selection
    from scaffolds.transformer.configs import Config, validate_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.diagnostics import attention_identity
    from scaffolds.transformer.training.state import create_state
    import types

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capabilities", lambda: ("9.0",))
    calls = []
    transformed = api.transformed_query_polynomial_attention
    monkeypatch.setattr(api, "transformed_query_polynomial_attention",
                        lambda *a, **kw: calls.append(kw["transform"]) or transformed(*a, **kw))
    model = _mosaic_model(transform)
    config = Config(model=model)
    config.data.sequence_length, config.data.synthetic_vocab_size = 129, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.train.dtype = "bfloat16"
    validate_config(config)
    assert attention_identity(config)["attention_implementation"] == (
        f"pallas_mosaic_gpu_query_polynomial_{transform}_scores")
    state = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.random.default_rng(1).integers(0, 32, (2, 129)), jnp.int32)

    def loss_and_grads(params):
        return loop.loss_and_grads(types.SimpleNamespace(graphdef=state.graphdef, params=params), batch,
                                   jax.random.PRNGKey(2))

    jax.make_jaxpr(loss_and_grads)(state.params)
    assert calls == [transform] * 2  # one trace, two layers
