"""Static SoS Mosaic qualification against host FP64 factors and their analytic VJP."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api
from scaffolds.transformer.attention.backends.mosaic.common import PRECISIONS
from scaffolds.transformer.attention.nonnegative_polynomial import static_square_coefficients, static_square_factors
from scaffolds.transformer.tests.test_fused_attention import _assert_relative
from scaffolds.transformer.tests.test_static_sos import STATIC_CASES, _inputs, _numpy_fp64, _operation

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_qualified_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0"):
        pytest.skip("Static transformed SoS qualification requires H100 or B200")


def _direct_kernel(core_degree, transform, power, precision):
    def attend(q, k, v, b):
        factors = static_square_factors(b, core_degree, parameterization="relative_taylor")
        coefficients = static_square_coefficients(factors, core_degree + power, input_power=power)
        coefficients = jnp.broadcast_to(coefficients, (*q.shape[:3], core_degree + power + 1))
        if transform == "identity":
            return api.query_polynomial_attention(q, k, v, coefficients, precision=precision)
        return api.transformed_query_polynomial_attention(q, k, v, coefficients, transform=transform,
                                                         precision=precision)
    return attend


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _assert_all(actual, expected):
    _assert_relative(actual[0], expected[0], rtol=.006, name="static SoS output")
    for name, got, want, tolerance in zip("qkv", actual[1][:3], expected[1][:3], (.025, .025, .015), strict=True):
        _assert_relative(got, want, rtol=tolerance, name=f"static SoS d{name}")
    # Every factor's static derivative sums over all batches, queries, and keys.
    # Check coordinates independently so tiny high-order factors are not hidden by the low orders.
    for index in range(expected[1][3].shape[-1]):
        view = lambda x: np.asarray(x)[..., index].reshape(1, 1, -1, 1)
        _assert_relative(view(actual[1][3]), view(expected[1][3]), rtol=.025,
                         name=f"static SoS db factor {index}")


@pytest.mark.parametrize("core_degree,transform,power", STATIC_CASES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_static_shared_factor_gradients_match_explicit_fp64(core_degree, transform, power, precision):
    operands, cotangent = _inputs(core_degree, time=512, batch=2, heads=2, width=64, dtype=jnp.bfloat16)
    actual = jax.jit(partial(_vjp, _direct_kernel(core_degree, transform, power, precision)))(operands, cotangent)
    assert actual[0].dtype == jnp.bfloat16
    assert actual[1][3].dtype == jnp.float32
    _assert_all(actual, _numpy_fp64(core_degree, transform, power, operands, cotangent))


@pytest.mark.parametrize("core_degree,transform,power", STATIC_CASES)
def test_production_static_dispatch_at_2048_tokens(core_degree, transform, power, monkeypatch):
    calls = []
    name = "query_polynomial_attention" if transform == "identity" else "transformed_query_polynomial_attention"
    original = getattr(api, name)
    def counted(*args, **kwargs):
        calls.append(args[3].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, name, counted)
    operands, cotangent = _inputs(core_degree, time=2048, batch=1, heads=2, width=64, dtype=jnp.bfloat16)
    actual = jax.jit(partial(_vjp, _operation(core_degree, transform, power, "mosaic")))(operands, cotangent)
    assert calls == [(1, 2048, 2, core_degree + power + 1)]
    _assert_all(actual, _numpy_fp64(core_degree, transform, power, operands, cotangent))


@pytest.mark.parametrize("core_degree", (4, 8, 12), ids=("m4", "m8", "m12"))
def test_global_factor_gradient_sums_heads_batches_and_tokens(core_degree):
    operands, cotangent = _inputs(core_degree, time=512, batch=2, heads=2, width=64, dtype=jnp.bfloat16)
    operands = (*operands[:3], operands[3][0])
    actual = jax.jit(partial(_vjp, _operation(core_degree, "softplus", 1, "mosaic")))(operands, cotangent)
    _assert_all(actual, _numpy_fp64(core_degree, "softplus", 1, operands, cotangent))


@pytest.mark.parametrize("core_degree", (4, 8, 12), ids=("m4", "m8", "m12"))
def test_static_prefactor_causal_prefix_and_gradients_ignore_future_tokens(core_degree):
    operands, cotangent = _inputs(core_degree, time=512, batch=1, heads=2, width=64, dtype=jnp.bfloat16)
    prefix = 300
    cotangent = cotangent.at[:, prefix:].set(0)
    function = jax.jit(partial(_vjp, _operation(core_degree, "softplus", 1, "mosaic")))
    output, gradients = function(operands, cotangent)
    changed = tuple(value.at[:, prefix:].add(2) if index < 3 else value for index, value in enumerate(operands))
    other_output, other_gradients = function(changed, cotangent)
    np.testing.assert_array_equal(output[:, :prefix], other_output[:, :prefix])
    for index, (left, right) in enumerate(zip(gradients, other_gradients, strict=True)):
        np.testing.assert_array_equal(left, right)
        if index < 3:
            np.testing.assert_array_equal(left[:, prefix:], 0.)
