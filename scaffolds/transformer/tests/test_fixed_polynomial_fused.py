"""Actual compiled CUDA gates for fixed polynomials, including signed kernels."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.tests.test_fixed_polynomial import (
    DEGREES, _assert_result, _attention, _evaluate, _fp64, _operands,
)


pytestmark = pytest.mark.gpu
PREFACTOR = math.exp(0.5)


@pytest.fixture(autouse=True)
def _requires_compiled_gpu():
    if jax.default_backend() == "gpu":
        return
    # An allocated qualification job must fail if CUDA is misconfigured; it
    # cannot report a successful test command whose GPU gates all skipped.
    if any(os.environ.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
                                           "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")):
        pytest.fail("GPU allocation is present but JAX did not initialize a GPU backend")
    pytest.skip("Requires a GPU allocation to compile the production Triton kernels")


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("normalized", (False, True))
@pytest.mark.parametrize("shape", ((2, 17, 3, 7, 5), (1, 131, 2, 33, 65)))
def test_compiled_fp32_outputs_and_qkv_gradients_match_fp64(degree, normalized, shape):
    operands, cotangent = _operands(degree, shape=shape)
    with jax.default_matmul_precision("highest"):
        actual = _evaluate(_attention(degree, normalized, prefactor=PREFACTOR, backend="fused"), operands, cotangent)
    assert actual[0].dtype == jnp.float32
    # Use the independent Hermite recurrence oracle. The blocked reference's
    # GPU compilation hits an XLA layout error for the odd-width stress shape.
    _assert_result(actual, _fp64(operands, cotangent, degree, normalized, PREFACTOR))


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("normalized", (False, True))
def test_compiled_bf16_full_context_outputs_and_qkv_gradients(degree, normalized):
    operands, cotangent = _operands(degree, shape=(1, 2048, 2, 64, 64), dtype=jnp.bfloat16, seed=917)
    attend = _attention(degree, normalized, prefactor=PREFACTOR, backend="fused")
    actual = _evaluate(attend, operands, cotangent)
    blocked = _evaluate(_attention(degree, normalized, prefactor=PREFACTOR), operands, cotangent)
    tolerances = dict(output_tolerance=0.006, gradient_tolerances=(0.025, 0.025, 0.015))
    _assert_result(actual, blocked, **tolerances)
    assert actual[0].dtype == jnp.bfloat16
    assert all(value.dtype == jnp.bfloat16 for value in actual[1])
    # The independent oracle sees precisely the BF16-rounded production inputs.
    prefix = tuple(value[:, :129, :1] for value in operands)
    prefix_cotangent = cotangent[:, :129, :1]
    _assert_result(_evaluate(attend, prefix, prefix_cotangent),
                   _fp64(prefix, prefix_cotangent, degree, normalized, PREFACTOR), **tolerances)


@pytest.mark.parametrize("normalized", (False, True))
def test_compiled_zero_quadratic_row_preserves_failure_policy(normalized):
    q = jnp.asarray([0., 1., 0.], jnp.float32).reshape(1, 3, 1, 1)
    k = jnp.asarray([-1., -1., 0.], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3., 7., 11.], jnp.float32).reshape(q.shape)
    actual = jax.jit(_attention(2, normalized, prefactor=PREFACTOR, backend="fused"))(q, k, v)
    expected = _attention(2, normalized, prefactor=PREFACTOR)(q, k, v)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, equal_nan=True)
    assert bool(jnp.isnan(actual[0, 1, 0, 0])) == normalized
