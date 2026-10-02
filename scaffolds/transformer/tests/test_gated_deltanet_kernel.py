"""GPU qualification of FLA's chunked gated delta rule launched from JAX."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.gated_deltanet import GatedDeltaNet, gated_delta_rule_reference

pytestmark = pytest.mark.gpu


def _require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")


def _operands(key_dim, dtype, *, batch=2, length=256, heads=4, value_dim=64, seed=0):
    rng = np.random.default_rng(seed)
    q, k = (rng.standard_normal((batch, length, heads, key_dim)) for _ in range(2))
    q, k = (z / np.linalg.norm(z, axis=-1, keepdims=True) for z in (q, k))
    v = rng.standard_normal((batch, length, heads, value_dim))
    rate = np.exp(rng.uniform(0, math.log(16), (1, 1, heads)))
    g = -rate * np.log1p(np.exp(rng.standard_normal((batch, length, heads)) * 0.5 - 4))
    beta = 1 / (1 + np.exp(-rng.standard_normal((batch, length, heads))))
    cotangent = rng.standard_normal((batch, length, heads, value_dim))
    cast = lambda z, d=dtype: jnp.asarray(z, d)
    return (cast(q), cast(k), cast(v), cast(g, jnp.float32), cast(beta)), cast(cotangent)


@pytest.mark.parametrize("key_dim", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-3), (jnp.bfloat16, 2e-2)])
def test_fused_output_and_all_gradients_match_the_recurrence(key_dim, dtype, tolerance):
    _require_gpu()
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_gated_delta_rule

    args, cotangent = _operands(key_dim, dtype)
    scale = key_dim ** -0.5

    def loss(function):
        return lambda *a: jnp.sum(function(*a, scale).astype(jnp.float32) * cotangent.astype(jnp.float32))

    fused = fla_gated_delta_rule(*args, scale)
    fused_grads = jax.grad(loss(fla_gated_delta_rule), argnums=range(5))(*args)
    reference_args = [a.astype(jnp.float32) for a in args]
    with jax.default_matmul_precision("highest"):
        reference = gated_delta_rule_reference(*reference_args, scale)
        reference_grads = jax.grad(loss(gated_delta_rule_reference), argnums=range(5))(*reference_args)
    for name, ours, theirs in (("o", fused, reference), *zip(("dq", "dk", "dv", "dg", "dbeta"), fused_grads,
                                                               reference_grads)):
        ours = jnp.asarray(ours, jnp.float32)
        assert bool(jnp.all(jnp.isfinite(ours))), name
        error = float(jnp.linalg.norm(ours - theirs) / jnp.linalg.norm(theirs))
        assert error < tolerance, (name, error)


def test_fused_layer_matches_the_reference_layer_with_shared_weights():
    _require_gpu()
    blocked = GatedDeltaNet(128, 4, 32, 64, backend="blocked", rngs=nnx.Rngs(5))
    fused = GatedDeltaNet(128, 4, 32, 64, backend="fused", rngs=nnx.Rngs(5))
    x = jnp.asarray(np.random.default_rng(6).standard_normal((2, 128, 128)), jnp.float32)

    def loss(module, x):
        return jnp.sum(module(x) ** 2)

    with jax.default_matmul_precision("highest"):
        expected, expected_grads = jax.value_and_grad(loss)(blocked, x)
    actual, actual_grads = jax.value_and_grad(loss)(fused, x)
    assert abs(float(actual) / float(expected) - 1) < 5e-3
    for (path, left), (_, right) in zip(nnx.to_flat_state(nnx.state(actual_grads, nnx.Param)),
                                        nnx.to_flat_state(nnx.state(expected_grads, nnx.Param)), strict=True):
        left, right = np.asarray(left[...], np.float64), np.asarray(right[...], np.float64)
        assert np.linalg.norm(left - right) <= 1e-2 * np.linalg.norm(right) + 1e-6, path


def test_device_class_uses_fla_hopper_and_blackwell_branches():
    _require_gpu()
    import triton
    from packaging.version import Version

    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import _device_class, _kernels

    arch, shared = _device_class()
    assert arch in ("hopper", "blackwell") and shared >= 232448
    if arch == "hopper":
        # FLA #640: gated dqkwg is wrong on Hopper with Triton in [3.4.0, 3.7.1).
        assert Version(triton.__version__) >= Version("3.7.1")
    tuned, tile, wy_tile = _kernels()
    assert (tile, wy_tile) == (128, 64)
    fwd_h_warps = {config.num_warps for config in tuned["fwd_h"].configs}
    assert fwd_h_warps == ({2} if arch == "blackwell" else {2, 4})
