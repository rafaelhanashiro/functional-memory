"""The Mosaic row-power kernels: the softplus query polynomial in units of its row's largest power.

On CPU, the family's tile arithmetic (mosaic/common.py) against FP64 at score scales 0.5 to 400. On an H100 or B200,
the kernels through ``shifted_polynomial_attention`` against dense FP64 at degrees 9 and 17 (Q/K width 4), with and
without scores whose 17th powers overflow FP32; the pass that finds each row's largest score; and the kernels against
the fused Triton row-power kernel at the training length.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api, common
from scaffolds.transformer.attention.output_scaling import max_abs_head_rms
from scaffolds.transformer.attention.shifted_polynomial import shifted_polynomial_attention
from scaffolds.transformer.tests.test_shifted_polynomial_attention import _oracle

DEGREES = (9, 17)
FAMILY = common.QUERY_SOFTPLUS_ROW_POWER


def _gpu(monkeypatch):
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0"):
        pytest.skip("Requires an H100 or B200 allocation")


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("scale", [0.5, 4.0, 40.0, 400.0])
def test_tile_arithmetic_matches_fp64(scale, degree):
    """Weights, score gradients and coefficient sums of one tile, from the units the entry point forms, against
    e^(-m) p(z), its slope and sum_i G z^r e^(-m) / e^(-(m - r) m / M) in FP64."""
    rng = np.random.default_rng(int(10 * scale) + degree)
    s = (scale * rng.normal(size=(8, 64))).astype(np.float32)
    dp = rng.normal(size=s.shape).astype(np.float32)
    c = (rng.uniform(0.1, 0.9, size=(8, degree + 1)) / np.array([math.factorial(r) for r in range(degree + 1)]))
    c[:, 1:degree:2] *= -0.1
    s64 = s.astype(np.float64)
    z = np.logaddexp(s64, 0.0)
    shift = degree * np.log(np.maximum(1.0, z.max(axis=1, keepdims=True)))
    units = np.exp(shift * (np.arange(degree + 1) - degree) / degree)
    half = np.exp(shift * (-0.5 / degree))
    vectors = [jnp.broadcast_to(jnp.asarray(c[:, r:r + 1] * units[:, r:r + 1], jnp.float32), s.shape)
               for r in range(degree + 1)] + [jnp.broadcast_to(jnp.asarray(half, jnp.float32), s.shape)]
    weights = np.asarray(common.forward_weights(FAMILY, degree, jnp.asarray(s), vectors), np.float64)
    ds, sums = common.dq_terms(FAMILY, degree, jnp.asarray(s), jnp.asarray(dp), vectors)
    _, dk_sums = common.dk_terms(FAMILY, degree, jnp.asarray(s), jnp.asarray(dp), vectors)

    unit = np.exp(-shift)
    terms = [c[:, r:r + 1] * z ** r * unit for r in range(degree + 1)]
    want = sum(terms)
    sigmoid = 1.0 / (1.0 + np.exp(-s64))
    slope = sum(r * c[:, r:r + 1] * z ** (r - 1) for r in range(1, degree + 1)) * unit * sigmoid
    np.testing.assert_allclose(weights, want, rtol=0, atol=2e-6 * np.max(np.abs(sum(np.abs(x) for x in terms))))
    slope_scale = np.max(np.abs(dp * sum(r * np.abs(c[:, r:r + 1]) * z ** (r - 1) for r in range(1, degree + 1))
                                * unit * sigmoid))
    np.testing.assert_allclose(np.asarray(ds, np.float64), dp * slope, rtol=0, atol=4e-6 * max(slope_scale, 1e-30))
    assert dk_sums == []
    u = z * half * half
    for r, got in enumerate(sums):
        moment = np.sum(dp * u ** r, axis=1)
        np.testing.assert_allclose(np.asarray(got, np.float64), moment, rtol=0,
                                   atol=4e-6 * np.max(np.sum(np.abs(dp) * u ** r, axis=1)))


def test_the_family_holds_its_coefficients_and_units():
    assert common.vector_counts(FAMILY, 17) == (19, 19, 18)
    assert common.approx_math(FAMILY)
    assert api.row_power_degree_supported(17, 4, 2048) and not api.row_power_degree_supported(18, 4, 2048)
    assert api.row_power_degree_supported(13, 64, 2048) and not api.row_power_degree_supported(16, 64, 2048)
    assert not api.row_power_degree_supported(9, 4, 2048, transform="abs")


def _inputs(time, stress, degree, width=4, heads=1, seed=0):
    """BF16 Q/K/V at value width 64 and FP32 coefficients as test_shifted_polynomial_attention's, the stress case's
    keys at +-1000 and 1100 (scores to 1100, whose 17th powers overflow FP32)."""
    rng = np.random.default_rng(124 + degree + seed)
    q = rng.normal(size=(1, time, heads, width)) / 2
    k = rng.normal(size=(1, time, heads, width)) / 2
    v = rng.normal(size=(1, time, heads, 64)) / 2
    if stress:
        q[...] = 0.0
        q[..., 0] = 2.0
        k[0, 0, :, :] = 0.0
        k[0, 0, :, 0] = -1000.0
        k[0, time // 2, :, :] = 0.0
        k[0, time // 2, :, 0] = 1000.0
        k[0, -1, :, :] = 0.0
        k[0, -1, :, 0] = 1100.0
    c = (rng.uniform(0.1, 0.9, size=(1, time, heads, degree + 1))
         / np.array([math.factorial(r) for r in range(degree + 1)])).astype(np.float32)
    c[..., 1:degree:2] *= -0.1
    cot = rng.normal(size=v.shape).astype(np.float32)
    return (jnp.asarray(q, jnp.bfloat16), jnp.asarray(k, jnp.bfloat16), jnp.asarray(v, jnp.bfloat16),
            jnp.asarray(c)), jnp.asarray(cot)


def _normalized(backend):
    def normalized(q, k, v, c):
        result, shift = shifted_polynomial_attention(q, k, v, c, transform="softplus", backend=backend)
        return max_abs_head_rms(result, 1e-6, shift[..., None])
    return normalized


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("stress", [False, True])
def test_kernels_match_fp64(monkeypatch, stress, degree):
    """512 tokens (two Blackwell CTAs of 256 queries; four H100 CTAs), with the stress keys' rows dominated by one
    key. BF16 weights and score gradients enter the tensor cores, as the transformed kernels' do: the output within
    BF16 rounding of FP64's, each gradient within 2.5% of its largest entry."""
    _gpu(monkeypatch)
    args, cot = _inputs(512, stress, degree)
    with jax.enable_x64(True):
        oracle, pull = jax.vjp(lambda *a: _oracle(*a, "softplus"), *(jnp.asarray(x, jnp.float64) for x in args))
        oracle_grads = pull(jnp.asarray(cot, jnp.float64))
    y, pull = jax.vjp(jax.jit(_normalized("mosaic")), *args)
    grads = pull(cot)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in (y, *grads))
    np.testing.assert_allclose(np.asarray(y, np.float64), np.asarray(oracle), rtol=0,
                               atol=0.02 * float(jnp.max(jnp.abs(oracle))))
    for name, actual, expected in zip("qkvc", grads, oracle_grads, strict=True):
        scale = float(jnp.max(jnp.abs(expected)))
        error = float(jnp.max(jnp.abs(jnp.asarray(actual, jnp.float64) - expected)))
        assert error <= 0.025 * max(scale, 1e-30), (name, error, scale)


@pytest.mark.gpu
def test_the_units_pass_finds_each_rows_largest_causal_score(monkeypatch):
    _gpu(monkeypatch)
    from scaffolds.transformer.attention.backends.fused_shifted_polynomial import causal_score_max

    (q, k, _, _), _ = _inputs(512, False, 9, heads=3)
    top = causal_score_max(q, k)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(q.shape[-1])
    causal = jnp.arange(512)[:, None] >= jnp.arange(512)[None, :]
    want = jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1).transpose(0, 2, 1)
    np.testing.assert_allclose(top, want, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_kernels_match_the_fused_row_power_kernel(monkeypatch, degree):
    """2,048 tokens and 2 heads at score scales to about 20, as QK RMSNorm's learned scale reaches: the Mosaic and
    fused Triton row-power kernels agree in the shift exactly and in the normalized output and every gradient to
    BF16 weight rounding."""
    _gpu(monkeypatch)
    (q, k, v, c), cot = _inputs(2048, False, degree, heads=2, seed=1)
    q = (6.0 * q.astype(jnp.float32)).astype(jnp.bfloat16)
    _, mosaic_shift = shifted_polynomial_attention(q, k, v, c, transform="softplus", backend="mosaic")
    _, fused_shift = shifted_polynomial_attention(q, k, v, c, transform="softplus", backend="fused")
    np.testing.assert_allclose(mosaic_shift, fused_shift, rtol=1e-6, atol=1e-6)
    results = []
    for backend in ("mosaic", "fused"):
        y, pull = jax.vjp(jax.jit(_normalized(backend)), q, k, v, c)
        results.append((y, *pull(cot)))
    for name, actual, expected in zip(("y", "q", "k", "v", "c"), *results, strict=True):
        actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        scale = float(np.max(np.abs(expected)))
        assert np.max(np.abs(actual - expected)) <= 0.02 * scale, (name, np.max(np.abs(actual - expected)), scale)
