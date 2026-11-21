"""Independent dense FP64 checks of shifted gated-exp plus even powers, including their gradients."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.fused_even_gate_exp import fused_even_gate_exp
from scaffolds.transformer.attention.even_gate_exp import DEGREES, even_gate_exp_attention, even_orders
from scaffolds.transformer.attention.output_scaling import max_abs_head_rms


def _oracle(q, k, v, c, g):
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    scores = jnp.where(causal, scores, 0)
    weights = g.transpose(0, 2, 1)[..., None] * jnp.exp(scores)
    for order in even_orders(c.shape[-1] - 1):
        weights += c[..., order].transpose(0, 2, 1)[..., None] * scores ** order
    output = jnp.einsum("bhqk,bkhd->bqhd", jnp.where(causal, weights, 0), v)
    return output / jnp.sqrt(jnp.mean(jnp.square(output), axis=-1, keepdims=True) + 1e-6)


def _inputs(time, stress, degree=16):
    rng = np.random.default_rng(124)
    q, k, v = (rng.normal(size=(1, time, 1, d)).astype(np.float32) / 2 for d in (4, 4, 8))
    if stress:
        # exp(110) overflows in FP32, and so does (-1000)^16. The last key is inaccessible to earlier rows.
        q[:, :, :, :] = np.array([2., 0., 0., 0.])
        k[0, 0, 0] = [-1000., 0., 0., 0.]
        k[0, time // 2, 0] = [100., 0., 0., 0.]
        k[0, -1, 0] = [110., 0., 0., 0.]
    c = np.zeros((*q.shape[:-1], degree + 1), np.float32)
    c[..., 2::2] = rng.uniform(0.1, 0.9, size=c[..., 2::2].shape)
    g = rng.uniform(0.1, 0.9, size=q.shape[:-1]).astype(np.float32)
    cot = rng.normal(size=v.shape).astype(np.float32)
    return (q, k, v, c, g), cot


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("backend", ["blocked", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_shifted_composition_and_all_gradients(backend, stress, degree):
    # 65 includes masked tail rows/columns and changes the running max on a later causal tile.
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("The fused Triton program requires a GPU")
    args, cot = _inputs(65, stress, degree)
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


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ["blocked", pytest.param("fused", marks=pytest.mark.gpu)])
def test_flat_exp_gate_preserves_polynomial_at_large_positive_scores(backend, degree):
    args, cot = _inputs(5, False, degree)
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
        polynomial = sum(c64[..., r].swapaxes(1, 2)[..., None] * scores**r for r in even_orders(degree))
        raw = jnp.einsum("bhqk,bkhd->bqhd", jnp.tril(polynomial), v64)
        reference = raw / jnp.sqrt(jnp.mean(raw**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(y, reference, rtol=2e-4, atol=1e-5)



@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("scale", [0.5, 4.0, 40.0, 150.0])
def test_fused_horner_arithmetic_matches_the_shifted_basis(scale, degree):
    """The fused programs' per-tile arithmetic (fused_even_gate_exp: Horner in u^2, slopes and gate moments, run here
    over whole rows on CPU because Triton programs need a GPU) against FP64, as close as even_gate_exp.shifted_basis."""
    from scaffolds.transformer.attention.backends import fused_even_gate_exp as fused
    from scaffolds.transformer.attention.even_gate_exp import row_shift, shifted_basis

    rng = np.random.default_rng(int(scale * 10))
    rows, keys = 16, 256
    orders = even_orders(degree)
    s = jnp.asarray(rng.normal(size=(rows, keys)) * scale, jnp.float32)
    coefficients = tuple(jnp.asarray(rng.uniform(0.05, 0.95, size=rows) / math.factorial(r), jnp.float32)
                         for r in orders)
    exp_gate = jnp.asarray(rng.uniform(0.05, 0.95, size=rows), jnp.float32).at[:3].set(0.0)
    gates = (*coefficients, exp_gate)
    dp = jnp.asarray(rng.normal(size=(rows, keys)), jnp.float32)
    shift = row_shift(s, exp_gate != 0.0, degree)

    half, units, horner = fused._row_units(shift, gates)
    polynomial, w, slope = fused._polynomial(s, half, horner, slope=True)
    exponential = fused._exponential(s, shift, exp_gate)
    product, moments = dp, []
    for _ in orders:
        product = product * w
        moments.append(jnp.sum(product, axis=1))
    fused_values = (polynomial + exp_gate[:, None] * exponential, slope + exp_gate[:, None] * exponential,
                    [m * unit for m, unit in zip(moments, units)] + [jnp.sum(dp * exponential, axis=1)])
    bases, derivatives = shifted_basis(s, shift[:, None], (exp_gate != 0.0)[:, None], slopes=True, degree=degree)
    basis_values = (sum(b * g[:, None] for b, g in zip(bases, gates)), sum(d * g[:, None] for d, g in zip(derivatives, gates)),
                    [jnp.sum(dp * b, axis=1) for b in bases])

    with jax.enable_x64(True):
        s64, shift64 = jnp.asarray(s, jnp.float64), jnp.asarray(shift, jnp.float64)[:, None]
        on = (exp_gate != 0.0)[:, None]
        e = jnp.where(on, jnp.exp(s64 - shift64), 0.0)
        g64 = [jnp.asarray(g, jnp.float64)[:, None] for g in gates]
        terms = [g * s64 ** r * jnp.exp(-shift64) for g, r in zip(g64, orders)] + [g64[-1] * e]
        slope_terms = [g * r * s64 ** (r - 1) * jnp.exp(-shift64) for g, r in zip(g64, orders)] + [g64[-1] * e]
        bases64 = [s64 ** r * jnp.exp(-shift64) for r in orders] + [e]
        # Signed sums lose FP32 resolution relative to their terms: compare at the scale of the terms' magnitudes.
        exact = [(np.asarray(sum(x)), np.asarray(sum(jnp.abs(t) for t in x))) for x in (terms, slope_terms)]
        exact_grads = [(np.asarray(jnp.sum(jnp.asarray(dp, jnp.float64) * b, axis=1)),
                        np.asarray(jnp.sum(jnp.abs(b), axis=1))) for b in bases64]
    # Products below FP32's normal range keep only subnormal resolution, up to one smallest normal number each.
    floor = keys * np.finfo(np.float32).tiny
    for values in (fused_values, basis_values):
        for got, (want, size) in zip(values[:2], exact):
            np.testing.assert_array_less(np.abs(np.asarray(got, np.float64) - want), 4e-6 * size + floor)
        for got, (want, size) in zip(values[2], exact_grads):
            np.testing.assert_array_less(np.abs(np.asarray(got, np.float64) - want), 1e-5 * size + floor)
