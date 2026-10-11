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


def _independent_reference(q, k, v, g, beta, erase, scale):
    return gated_delta_rule_reference(q, k, v, g, beta, scale, erase=erase)


def _check_independent_output_and_gradients(args, cotangent, tolerance):
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_independent_gated_delta_rule

    scale = args[0].shape[-1] ** -0.5

    def loss(function):
        return lambda *a: jnp.sum(function(*a, scale).astype(jnp.float32) * cotangent.astype(jnp.float32))

    fused = fla_independent_gated_delta_rule(*args, scale)
    fused_grads = jax.grad(loss(fla_independent_gated_delta_rule), argnums=range(6))(*args)
    reference_args = [a.astype(jnp.float32) for a in args]
    with jax.default_matmul_precision("highest"):
        reference = _independent_reference(*reference_args, scale)
        reference_grads = jax.grad(loss(_independent_reference), argnums=range(6))(*reference_args)
    pairs = (("output", fused, reference), *zip(("dq", "dk", "dv", "dg", "dbeta", "derase"),
                                              fused_grads, reference_grads, strict=True))
    for name, actual, expected in pairs:
        actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        assert np.all(np.isfinite(actual)), name
        error, reference_norm = np.linalg.norm(actual - expected), np.linalg.norm(expected)
        assert error <= tolerance * reference_norm + 2e-5, (name, error, reference_norm)
    return fused, fused_grads


@pytest.mark.parametrize("key_dim,length", [(16, 256), (64, 256), (128, 256), (64, 2048)])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-3), (jnp.bfloat16, 2e-2)])
def test_independent_fused_output_and_all_gradients_match_recurrence(key_dim, length, dtype, tolerance):
    _require_gpu()
    args, cotangent = _operands(key_dim, dtype, batch=1, length=length, seed=41)
    # Different gates, so a silently tied implementation cannot pass.
    erase = jax.nn.sigmoid(jnp.asarray(np.random.default_rng(42).standard_normal(args[4].shape),
                                      jnp.float32)).astype(dtype)
    _check_independent_output_and_gradients((*args, erase), cotangent, tolerance)


@pytest.mark.parametrize("erase_value", [0.0, 1e-6])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-3), (jnp.bfloat16, 2e-2)])
def test_independent_zero_or_tiny_erase_preserves_nonzero_writes(erase_value, dtype, tolerance):
    _require_gpu()
    args, cotangent = _operands(64, dtype, batch=1, seed=43)
    erase = jnp.full_like(args[4], erase_value)
    output, gradients = _check_independent_output_and_gradients((*args, erase), cotangent, tolerance)
    assert float(jnp.linalg.norm(output.astype(jnp.float32))) > 0
    assert float(jnp.linalg.norm(gradients[5].astype(jnp.float32))) > 0
    if erase_value == 0:
        # Independent additive-memory oracle; no delta-rule reference involved.
        q, k, v, g, beta = (a.astype(jnp.float32) for a in args)
        cumulative_decay = jnp.cumsum(g, axis=1)
        decay = jnp.exp(cumulative_decay[:, :, None, :] - cumulative_decay[:, None, :, :])
        causal = jnp.tril(jnp.ones((q.shape[1], q.shape[1]), dtype=jnp.bool_))
        decay = jnp.where(causal[None, :, :, None], decay, 0)
        scores = jnp.einsum("bthk,bshk->btsh", q, k) * q.shape[-1] ** -0.5
        expected = jnp.einsum("btsh,bsh,bshv->bthv", scores * decay, beta, v)
        error = float(jnp.linalg.norm(output.astype(jnp.float32) - expected))
        assert error <= tolerance * float(jnp.linalg.norm(expected)) + 2e-5


def _repeated_key_fixture(dtype, direction="axis", *, tied=False):
    args, cotangent = _operands(64, dtype, batch=1, heads=2, length=128, seed=44)
    _, _, v, g, beta = args
    if direction == "axis":
        key = jnp.zeros_like(args[0]).at[..., 0].set(1)
    else:
        assert direction == "dense"
        # Each coordinate and the unit norm are exact even in BF16. This exercises
        # value/key reductions beyond the coordinate-axis special case.
        key = jnp.broadcast_to(jnp.where(jnp.arange(64) % 2, -0.125, 0.125), args[0].shape).astype(dtype)
    if tied:
        beta = jnp.ones_like(beta)
    return (key, key, v, jnp.zeros_like(g), beta, jnp.ones_like(beta)), cotangent


def _repeated_key_oracle(args, cotangent):
    """FP64 closed form for constant unit q=k, zero decay and complete erasure.

    The raw q and k arguments are differentiated independently; no derivative
    through key normalization is included. With a_t=beta_t*v_t, the previous
    association a_{t-1} disappears from the output but contributes to dk/derase.
    """
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    c = np.asarray(cotangent, np.float64)
    np.testing.assert_array_equal(q, k)
    np.testing.assert_array_equal(k, np.broadcast_to(k[:, :1], k.shape))
    np.testing.assert_array_equal(np.sum(k * k, axis=-1), np.ones_like(g))
    np.testing.assert_array_equal(g, np.zeros_like(g))
    np.testing.assert_array_equal(erase, np.ones_like(erase))
    scale = k.shape[-1] ** -0.5
    association = beta[..., None] * v
    previous = np.concatenate((np.zeros_like(association[:, :1]), association[:, :-1]), axis=1)
    dot = lambda a: np.sum(c * a, axis=-1)
    expected = {
        "output": scale * association,
        "dq": scale * q * dot(association)[..., None],
        "dk": scale * k * dot(association - 2 * previous)[..., None],
        "dv": scale * beta[..., None] * c,
        "dg": np.zeros_like(g),
        "dbeta": scale * dot(v),
        "derase": -scale * dot(previous),
    }
    # dg is identically zero, so relative error is undefined. Budget FP32
    # accumulation roundoff against the uncancelled state/cotangent products.
    # Use the standard gamma_n summation factor with one value reduction and two
    # chunk-length reductions as a numerical error budget, not a proven bound
    # for every operation in the composed kernel. It depends on dimensions/input
    # scale, not the observed residual or BF16 epsilon: accumulations must be FP32.
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import CHUNK_SIZE

    unit_roundoff = np.finfo(np.float32).eps / 2
    reductions = v.shape[-1] + 2 * CHUNK_SIZE
    gamma = reductions * unit_roundoff / (1 - reductions * unit_roundoff)
    cancellation_scale = scale * np.sum(np.abs(c * previous), axis=-1)
    zero_decay_bound = gamma * np.linalg.norm(cancellation_scale)
    return expected, zero_decay_bound


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-5), (jnp.bfloat16, 5e-3)])
@pytest.mark.parametrize("direction", ["axis", "dense"])
@pytest.mark.parametrize("tied", [False, True], ids=["independent", "tied"])
def test_repeated_key_erases_old_values_with_analytic_gradients(dtype, tolerance, direction, tied):
    _require_gpu()
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
        fla_gated_delta_rule,
        fla_independent_gated_delta_rule,
    )

    args, cotangent = _repeated_key_fixture(dtype, direction, tied=tied)
    expected, zero_decay_bound = _repeated_key_oracle(args, cotangent)
    if tied:
        function, args = fla_gated_delta_rule, args[:5]
        # The original gate controls both writing and erasure.
        expected["dbeta"] += expected.pop("derase")
    else:
        function = fla_independent_gated_delta_rule
    scale = args[0].shape[-1] ** -0.5
    output = function(*args, scale)
    gradients = jax.grad(lambda *a: jnp.sum(function(*a, scale).astype(jnp.float32)
                                           * cotangent.astype(jnp.float32)), argnums=range(len(args)))(*args)
    assert output.dtype == args[2].dtype
    assert tuple(gradient.dtype for gradient in gradients) == tuple(arg.dtype for arg in args)
    failures = []
    for (name, reference), actual in zip(expected.items(), (output, *gradients), strict=True):
        actual = np.asarray(actual, np.float64)
        error = np.linalg.norm(actual - reference)
        bound = zero_decay_bound if name == "dg" else tolerance * np.linalg.norm(reference)
        if not np.all(np.isfinite(actual)) or error > bound:
            failures.append((name, float(error), float(bound)))
    # Report every failed quantity; a dk failure must not hide derase/dg errors.
    assert not failures, failures


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-3), (jnp.bfloat16, 2e-2)])
def test_independent_tied_gates_recover_gdn_including_gate_chain_gradient(dtype, tolerance):
    _require_gpu()
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
        fla_gated_delta_rule,
        fla_independent_gated_delta_rule,
    )

    args, cotangent = _operands(64, dtype, batch=1, seed=45)
    scale = 64 ** -0.5

    def split_with_tied_gates(q, k, v, g, beta):
        return fla_independent_gated_delta_rule(q, k, v, g, beta, beta, scale)

    def original(q, k, v, g, beta):
        return fla_gated_delta_rule(q, k, v, g, beta, scale)

    def loss(function):
        return lambda *a: jnp.sum(function(*a).astype(jnp.float32) * cotangent.astype(jnp.float32))

    new_grads = jax.grad(loss(split_with_tied_gates), argnums=range(5))(*args)
    original_grads = jax.grad(loss(original), argnums=range(5))(*args)
    pairs = (("output", split_with_tied_gates(*args), original(*args)),
             *zip(("dq", "dk", "dv", "dg", "dbeta_plus_derase"), new_grads, original_grads, strict=True))
    for name, actual, expected in pairs:
        actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        assert np.all(np.isfinite(actual)), name
        error, reference_norm = np.linalg.norm(actual - expected), np.linalg.norm(expected)
        assert error <= tolerance * reference_norm + 2e-5, (name, error, reference_norm)
