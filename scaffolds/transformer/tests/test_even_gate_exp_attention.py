"""Independent dense FP64 checks of shifted gated-exp plus even powers, including their gradients."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.fused_even_gate_exp import fused_even_gate_exp
from scaffolds.transformer.attention.even_gate_exp import ORDERS, even_gate_exp_attention
from scaffolds.transformer.attention.output_scaling import max_abs_head_rms


def _oracle(q, k, v, c, g):
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    scores = jnp.where(causal, scores, 0)
    weights = g.transpose(0, 2, 1)[..., None] * jnp.exp(scores)
    for order in ORDERS:
        weights += c[..., order].transpose(0, 2, 1)[..., None] * scores ** order
    output = jnp.einsum("bhqk,bkhd->bqhd", jnp.where(causal, weights, 0), v)
    return output / jnp.sqrt(jnp.mean(jnp.square(output), axis=-1, keepdims=True) + 1e-6)


def _inputs(time, stress):
    rng = np.random.default_rng(124)
    q, k, v = (rng.normal(size=(1, time, 1, d)).astype(np.float32) / 2 for d in (4, 4, 8))
    if stress:
        # exp(110) and (-1000)^16 both overflow in FP32. The last key is inaccessible to earlier rows.
        q[:, :, :, :] = np.array([2., 0., 0., 0.])
        k[0, 0, 0] = [-1000., 0., 0., 0.]
        k[0, time // 2, 0] = [100., 0., 0., 0.]
        k[0, -1, 0] = [110., 0., 0., 0.]
    c = np.zeros((*q.shape[:-1], 17), np.float32)
    c[..., 2::2] = rng.uniform(0.1, 0.9, size=c[..., 2::2].shape)
    g = rng.uniform(0.1, 0.9, size=q.shape[:-1]).astype(np.float32)
    cot = rng.normal(size=v.shape).astype(np.float32)
    return (q, k, v, c, g), cot


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("backend", ["blocked", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_shifted_composition_and_all_gradients(backend, stress):
    # 65 includes masked tail rows/columns and changes the running max on a later causal tile.
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("The fused Triton program requires a GPU")
    args, cot = _inputs(65, stress)
    with jax.enable_x64(True):
        oracle, pull = jax.vjp(_oracle, *(jnp.asarray(x, jnp.float64) for x in args))
        oracle_grads = pull(jnp.asarray(cot, jnp.float64))

    def normalized(*args):
        if backend == "pallas":
            result, shift = fused_even_gate_exp(*args)
        else:
            result, shift = even_gate_exp_attention(*args, backend="blocked")
        return max_abs_head_rms(result, 1e-6, shift[..., None])

    y, pull = jax.vjp(jax.jit(normalized), *map(jnp.asarray, args))
    grads = pull(jnp.asarray(cot))
    np.testing.assert_allclose(y, oracle, rtol=2e-4, atol=1e-5)
    for actual, expected in zip(grads, oracle_grads, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-4)


def test_blocked_diagnostics_describe_the_causal_shifted_weights():
    args, _ = _inputs(35, False)
    (y, shift), d = even_gate_exp_attention(*map(jnp.asarray, args), return_diagnostics=True)
    assert y.shape == args[2].shape and shift.shape == y.shape[:3]
    assert int(d["causal_entry_count"]) == 35 * 36 // 2
    assert int(d["invalid_kernel_count"]) == int(d["negative_weight_count"]) == 0
    assert 0 < d["kernel_min"] <= d["kernel_max"]
    np.testing.assert_allclose(d["exp_shift_max"], shift.max())


@pytest.mark.parametrize("backend", ["blocked", pytest.param("fused", marks=pytest.mark.gpu)])
def test_flat_exp_gate_preserves_polynomial_at_large_positive_scores(backend):
    args, cot = _inputs(5, False)
    q, k, v, c = args[:4]
    q[...] = np.array([2., 0., 0., 0.])
    k[0, :, 0, 0] = [990., 999., 1000., -800., 900.]

    def head(logits):
        gate = jax.nn.sigmoid(logits)
        y, scale = even_gate_exp_attention(*map(jnp.asarray, (q, k, v, c)), gate, backend=backend)
        return max_abs_head_rms(y, 1e-6, scale[..., None])

    logits = jnp.full(q.shape[:-1], -1000., jnp.float32)
    y, pull = jax.vjp(jax.jit(head), logits)
    np.testing.assert_array_equal(pull(jnp.asarray(cot))[0], 0.0)
    # The exponential alone would overflow even FP64; a numerically flat gate leaves the even polynomial.
    with jax.enable_x64(True):
        q64, k64, v64, c64 = (jnp.asarray(x, jnp.float64) for x in (q, k, v, c))
        scores = jnp.einsum("bqhd,bkhd->bhqk", q64, k64) / math.sqrt(q.shape[-1])
        polynomial = sum(c64[..., r].swapaxes(1, 2)[..., None] * scores**r for r in ORDERS)
        raw = jnp.einsum("bhqk,bkhd->bqhd", jnp.tril(polynomial), v64)
        reference = raw / jnp.sqrt(jnp.mean(raw**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(y, reference, rtol=2e-4, atol=1e-5)
