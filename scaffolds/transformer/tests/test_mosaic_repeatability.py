"""Repeatability stress test for the Mosaic GPU attention kernels (H100 and B300).

The kernels use no atomics and a fixed accumulation order, so repeated executions must be
bitwise identical. A synchronization race (a tile read before its load completes, or a
buffer overwritten before a tensor-core read) would show up as run-to-run differences.
Use ``pytest -m gpu tests/test_mosaic_repeatability.py`` inside an H100 or B300 allocation.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention, taylor_attention

REPEATS = 20
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _operands(width=64, seed=98):
    # Several CTAs per SM at the production context: 8 x 2048 tokens x 16 heads.
    keys = jax.random.split(jax.random.key(seed), 4)
    q, k = (jax.random.normal(key, (8, 2048, 16, width), jnp.float32) for key in keys[:2])
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * 8 * math.sqrt(width)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v, cotangent = (jax.random.normal(key, (8, 2048, 16, 64), jnp.float32) for key in keys[2:])
    return tuple(x.astype(jnp.bfloat16) for x in (q, k, v)), cotangent.astype(jnp.bfloat16)


def _repeat(function, operands, cotangent):
    run = jax.jit(lambda *xs: jax.vjp(function, *xs)[0:1] + (jax.vjp(function, *xs)[1](cotangent),))
    reference = jax.tree.map(np.asarray, run(*operands))
    for attempt in range(REPEATS):
        result = jax.tree.map(np.asarray, run(*operands))
        for index, (got, want) in enumerate(zip(jax.tree.leaves(result), jax.tree.leaves(reference))):
            assert np.array_equal(got, want, equal_nan=True), f"attempt {attempt}: leaf {index} differs"


@pytest.mark.parametrize("degree", (2, 3))
def test_taylor_is_bitwise_repeatable(degree):
    operands, cotangent = _operands()
    _repeat(partial(taylor_attention, attention_degree=degree), operands, cotangent)


def test_query_polynomial_is_bitwise_repeatable():
    operands, cotangent = _operands()
    coefficients = jnp.tile(jnp.array([1.0, 1.0, 0.5], jnp.float32), (*operands[0].shape[:3], 1))
    _repeat(query_polynomial_attention, (*operands, coefficients), cotangent)
