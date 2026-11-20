"""Native Mosaic forward/VJP qualification for softplus with zero constants.

Direct query/static use r=1..8. Query SoS uses U1..U4,V1..V3, with no
prefactor. The independent NumPy FP64 oracle evaluates explicit factors and
positive powers; static gradients explicitly sum over queries and batches.
Static direct coefficients with a learnable constant c_0 (no structural zero),
alone and times the softplus prefactor, take the same broadcast route.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api
from scaffolds.transformer.tests.test_fused_attention import _assert_relative
from scaffolds.transformer.tests.test_softplus_zero_constant import FAMILIES, _attend, _inputs, _numpy_fp64

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_admitted_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0"):
        pytest.skip("Requires the admitted H100/B200 transformed Mosaic backend")


def _assert_result(family, actual, expected):
    _assert_relative(actual[0], expected[0], rtol=.006, name=family + " output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], (.025, .025, .015), strict=True):
        _assert_relative(got, want, rtol=tolerance, name=family + " d" + name)
    if family == "static":
        for r in range(8):
            got, want = actual[1][3][..., r], expected[1][3][..., r]
            assert np.linalg.norm(want) > 0.
            _assert_relative(got[None, None, :, None], want[None, None, :, None],
                             rtol=.02, name=f"static dc{r+1}, summed over all queries")
        return
    _assert_relative(actual[1][3][:, :, None, :], expected[1][3][:, :, None, :],
                     rtol=.025, name=family + " dx")
    for name, got, want in zip(("W", "a", "b"), actual[1][4:], expected[1][4:], strict=True):
        assert got.dtype == jnp.float32
        for coordinate in range(want.shape[-1]):
            view = (lambda arr: arr[..., coordinate].T[None, :, :, None]) if name == "W" else (
                lambda arr: arr[..., coordinate][None, None, :, None])
            assert np.linalg.norm(want[..., coordinate]) > 0.
            _assert_relative(view(got), view(want), rtol=.025,
                             name=f"{family} d{name} retained coordinate {coordinate}")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("spread", (1., 3.), ids=("unit", "wide"))
def test_mosaic_compact_values_and_all_vjps_match_independent_fp64(family, spread):
    operands, cotangent = _inputs(family, time=512, width=64, dtype=jnp.bfloat16, spread=spread)
    def compute(*args):
        output, pullback = jax.vjp(_attend(family, "mosaic"), *args)
        return output, pullback(cotangent)
    actual = jax.jit(compute)(*operands)
    _assert_result(family, actual, _numpy_fp64(family, operands, cotangent))


@pytest.mark.parametrize("family", FAMILIES)
def test_mosaic_production_context_and_native_dispatch(family, monkeypatch):
    calls = []
    original = api.transformed_query_polynomial_attention
    def capture(q, k, v, coefficients, **kwargs):
        calls.append((kwargs["transform"], coefficients.shape))
        return original(q, k, v, coefficients, **kwargs)
    monkeypatch.setattr(api, "transformed_query_polynomial_attention", capture)
    operands, cotangent = _inputs(family, time=2048, width=64, dtype=jnp.bfloat16, seed=487)
    def compute(*args):
        output, pullback = jax.vjp(_attend(family, "mosaic"), *args)
        return output, pullback(cotangent)
    actual = jax.jit(compute)(*operands)
    assert calls == [("softplus", (1, 2048, 2, 9))], "Must reach native transformed Mosaic, with total degree eight"
    _assert_result(family, actual, _numpy_fp64(family, operands, cotangent))


def _static_fp64(prefactor, operands, cotangent):
    """Dense f(s) sum_r c_r z^r, z = softplus(s), with static per-head c (c_0 free) and its analytic VJP; the
    coefficient gradient sums over every query and batch, as the broadcast's does."""
    import math

    q, k, v, c = (np.asarray(value.astype(jnp.float32), np.float64) for value in operands)
    do = np.asarray(cotangent.astype(jnp.float32), np.float64)
    scores = np.einsum("bthd,bshd->bhts", q, k, optimize=True) / math.sqrt(q.shape[-1])
    z, sigmoid = np.logaddexp(0., scores), np.exp(-np.logaddexp(0., -scores))
    f, f_slope = (z, sigmoid) if prefactor == "softplus" else (np.ones_like(z), np.zeros_like(z))
    causal = np.tril(np.ones(scores.shape[-2:], bool))
    dp = np.where(causal, np.einsum("bthd,bshd->bhts", do, v, optimize=True), 0.)
    coefficients = c[None, :, None, None, :]  # [1, heads, 1, 1, orders]
    powers = np.stack([z ** r for r in range(c.shape[-1])], -1)
    polynomial = np.sum(coefficients * powers, -1)
    derivative = np.sum(coefficients[..., 1:] * np.arange(1, c.shape[-1]) * powers[..., :-1], -1)
    weights = np.where(causal, f * polynomial, 0.)
    ds = np.where(causal, dp * (f * derivative * sigmoid + f_slope * polynomial), 0.)
    output = np.einsum("bhts,bshd->bthd", weights, v, optimize=True)
    dq = np.einsum("bhts,bshd->bthd", ds, k, optimize=True) / math.sqrt(q.shape[-1])
    dk = np.einsum("bhts,bthd->bshd", ds, q, optimize=True) / math.sqrt(q.shape[-1])
    dv = np.einsum("bhts,bthd->bshd", weights, do, optimize=True)
    dc = np.sum((dp * f)[..., None] * np.where(causal[..., None], powers, 0.), axis=(0, 2, 3))
    return output, (dq, dk, dv, dc)


@pytest.mark.parametrize("prefactor", ("none", "softplus"))
@pytest.mark.parametrize("time", (512, 2048))
def test_static_broadcast_with_a_learnable_constant(prefactor, time, monkeypatch):
    """Learned static direct coefficients c_0..c_8 with c_0 != 0 (no structural zero), broadcast to every query on
    the transformed kernels, at softplus scores, alone and times the softplus prefactor: the production dispatch
    reaches the kernels, and the output, Q/K/V gradients and each order's coefficient gradient, summed over every
    query, match FP64."""
    from scaffolds.transformer.attention.core import attention_core

    calls = []
    original = api.transformed_query_polynomial_attention

    def capture(q, k, v, coefficients, **kwargs):
        calls.append((kwargs["transform"], kwargs.get("prefactor", "none"), coefficients.shape))
        return original(q, k, v, coefficients, **kwargs)

    monkeypatch.setattr(api, "transformed_query_polynomial_attention", capture)
    rng = np.random.default_rng(491 + time)
    q, k, v, cotangent = (jnp.asarray(rng.normal(size=(1, time, 2, 64)), jnp.bfloat16) for _ in range(4))
    factorials = np.array([1., 1., 2., 6., 24., 120., 720., 5040., 40320.])
    c = jnp.asarray(np.concatenate([rng.uniform(.5, 1.5, (2, 1)), rng.uniform(-.5, 1.5, (2, 8))], -1) / factorials,
                    jnp.float32)

    def attend(q, k, v, c):
        coefficients = jnp.broadcast_to(c[None, None], (1, time, 2, 9))
        return attention_core(q, k, v, backend="mosaic", attention_degree=8, attention_normalization=False,
                              query_polynomial_coefficients=coefficients, attention_coefficient_activation="tanh",
                              score_transform="softplus", score_prefactor=prefactor)

    def compute(*operands):
        output, pullback = jax.vjp(attend, *operands)
        return output, pullback(cotangent)

    actual = jax.jit(compute)(q, k, v, c)
    assert calls == [("softplus", prefactor, (1, time, 2, 9))]
    expected = _static_fp64(prefactor, (q, k, v, c), cotangent)
    _assert_relative(actual[0], expected[0], rtol=.006, name=f"static c0, {prefactor}: output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], (.025, .025, .015), strict=True):
        _assert_relative(got, want, rtol=tolerance, name=f"static c0, {prefactor}: d{name}")
    for r in range(9):
        got, want = np.asarray(actual[1][3][..., r]), expected[1][3][..., r]
        assert np.linalg.norm(want) > 0.
        _assert_relative(got[None, None, :, None], want[None, None, :, None], rtol=.02,
                         name=f"static c0, {prefactor}: dc{r}, summed over all queries")
