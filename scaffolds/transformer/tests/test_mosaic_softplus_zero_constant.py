"""Native Mosaic forward/VJP qualification for softplus with zero constants.

Direct query/static use r=1..8. Query SoS uses U1..U4,V1..V3, with no
prefactor. The independent NumPy FP64 oracle evaluates explicit factors and
positive powers; static gradients explicitly sum over queries and batches.
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
