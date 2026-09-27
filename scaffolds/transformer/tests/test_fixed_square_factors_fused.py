"""Compiled fused fixed-factor regression at the degree-two Hermite root."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.tests.test_fixed_square_factors import _attention, _near_root_inputs


pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_compiled_gpu():
    if jax.default_backend() == "gpu":
        return
    if any(os.environ.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
                                           "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")):
        pytest.fail("GPU allocation is present but JAX did not initialize a GPU backend")
    pytest.skip("Requires a GPU allocation to compile the production Triton kernel")


def test_one_key_near_root_has_finite_output_and_qkv_backward():
    q, k, v = _near_root_inputs()

    @jax.jit
    def value_and_backward(q, k, v):
        output, pullback = jax.vjp(lambda q, k, v: _attention(q, k, v, backend="fused"), q, k, v)
        return output, pullback(jnp.ones_like(output))

    output, gradients = value_and_backward(q, k, v)
    np.testing.assert_array_equal(output, v)
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
    for gradient in gradients[:2]:
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-2)
    np.testing.assert_array_equal(gradients[2], jnp.ones_like(v))


def test_one_key_exact_root_still_fails_explicitly():
    q, k, v = _near_root_inputs()
    k = k.at[0, 0, 0, 1].set(0)
    output = jax.jit(lambda q, k, v: _attention(q, k, v, backend="fused"))(q, k, v)
    assert bool(jnp.all(jnp.isnan(output)))
