"""FoX–PaTH fused polynomial attention, including the forget-gate gradient.

The reference differentiates the chunked JAX definition in FP32. CPU cases run
the actual Triton launches in the interpreter; GPU cases qualify BF16 operands
at the production head width and across the 512-token backward split boundary.
The gate input is the cumulative log decay L, not the per-token log alpha.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import triton_port
from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention
from scaffolds.transformer.attention.path import path_attention


NAMES = ("output", "dq", "dk", "dv", "dw", "dbeta", "dc", "dL")
HOPPER = triton_port.Device(9, 0, triton_port.SHARED_MEMORY_TIERS["hopper"])
# Existing PaTH's BF16/FP16 intermediate tolerances, with dL following the
# output/dv tolerance because it differentiates the weighted polynomial itself.
GPU_TOLERANCE = {
    "output": 1e-2, "dq": 2e-2, "dk": 2e-2, "dv": 1e-2,
    "dw": 2.5e-2, "dbeta": 2.5e-2, "dc": 5e-3, "dL": 1e-2,
}


def _inputs(*, time=256, heads=2, dim=16, degree=4, seed=0, dtype=jnp.float32,
            decay=0.02):
    rng = np.random.default_rng(seed)
    shape = (1, time, heads, dim)
    q, k = (rng.normal(size=shape) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) * 4 for x in (q, k))
    v = rng.normal(size=shape)
    w = rng.normal(size=shape)
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    beta = 2 / (1 + np.exp(-rng.normal(size=shape[:3])))
    taylor = 1 / np.array([math.factorial(r) for r in range(degree + 1)])
    c = taylor + 0.3 * rng.normal(size=(*shape[:3], degree + 1))
    log_alpha = jnp.asarray(-decay * (0.5 + rng.random(shape[:3])), jnp.float32)
    prefix = jnp.cumsum(log_alpha, axis=1)
    operands = [jnp.asarray(x, dtype) for x in (q, k, v)]
    operands.extend(jnp.asarray(x, jnp.float32) for x in (w, beta, c))
    operands.append(prefix)
    return operands, jnp.asarray(rng.normal(size=shape), dtype), log_alpha


def _attend(implementation, *, cumulative=True):
    def attend(q, k, v, w, beta, c, decay):
        prefix = decay if cumulative else jnp.cumsum(decay, axis=1)
        return implementation(q, k, v, w, beta, kernel="query_polynomial",
                              coefficients=c, log_decay=prefix)
    return attend


def _value_and_vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return (output, *pullback(cotangent.astype(output.dtype)))


def _reference(operands, cotangent, *, cumulative=True):
    with jax.default_matmul_precision("highest"):
        return _value_and_vjp(_attend(path_attention, cumulative=cumulative),
                              [x.astype(jnp.float32) for x in operands],
                              cotangent.astype(jnp.float32))


def _interpreted(operands, cotangent, *, cumulative=True):
    with triton_port.interpreter(HOPPER):
        return jax.block_until_ready(_value_and_vjp(
            _attend(fused_path_attention, cumulative=cumulative), operands, cotangent))


def _compiled(operands, cotangent, *, cumulative=True):
    return jax.block_until_ready(jax.jit(lambda *xs: _value_and_vjp(
        _attend(fused_path_attention, cumulative=cumulative), xs, cotangent))(*operands))


def _relative(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))


def _check(actual, expected, tolerance):
    assert len(actual) == len(expected) == len(NAMES)
    for name, got, want in zip(NAMES, actual, expected, strict=True):
        assert np.isfinite(np.asarray(got)).all(), name
        limit = tolerance[name] if isinstance(tolerance, dict) else tolerance
        error = _relative(got, want)
        assert error < limit, (name, error, limit)


@pytest.mark.parametrize("time,degree", [(128, 2), (640, 4), (1152, 4)])
def test_interpreted_forget_gate_outputs_and_all_seven_gradients(time, degree):
    operands, cotangent, _ = _inputs(time=time, degree=degree, seed=time)
    _check(_interpreted(operands, cotangent), _reference(operands, cotangent), 1e-3)


def test_interpreted_forget_gate_chain_rule_reaches_per_token_log_alpha():
    operands, cotangent, log_alpha = _inputs(time=640, seed=4)
    raw_operands = [*operands[:-1], log_alpha]
    actual = _interpreted(raw_operands, cotangent, cumulative=False)
    expected = _reference(raw_operands, cotangent, cumulative=False)
    _check(actual, expected, 1e-3)
    # alpha_0 appears in every prefix and cancels from every causal difference.
    gate_gradient = np.asarray(actual[-1])
    assert np.max(np.abs(gate_gradient[:, 0])) < 2e-5 * max(1., np.max(np.abs(gate_gradient)))


def test_interpreted_zero_forget_gate_recovers_ungated_path():
    operands, cotangent, _ = _inputs(time=640, seed=5)
    operands[-1] = jnp.zeros_like(operands[-1])
    gated = _interpreted(operands, cotangent)
    with triton_port.interpreter(HOPPER):
        ungated = jax.block_until_ready(_value_and_vjp(
            lambda q, k, v, w, beta, c: fused_path_attention(
                q, k, v, w, beta, kernel="query_polynomial", coefficients=c),
            operands[:-1], cotangent))
    for name, got, want in zip(NAMES[:-1], gated[:-1], ungated, strict=True):
        np.testing.assert_array_equal(got, want, err_msg=name)
    # The gate derivative need not vanish at an open gate, but adding a constant
    # to every prefix must leave attention unchanged for each batch/head.
    dL = np.asarray(gated[-1], np.float64)
    assert np.max(np.abs(dL.sum(axis=1))) < 2e-5 * max(1., np.abs(dL).sum())


def test_interpreted_strong_decay_is_finite_and_retains_the_diagonal():
    operands, cotangent, _ = _inputs(time=256, seed=6, decay=1000)
    actual = _interpreted(operands, cotangent)
    q, k, v, _, _, c, _ = operands
    score = jnp.sum(q * k, axis=-1) / math.sqrt(q.shape[-1])
    weight = sum(c[..., r] * score ** r for r in range(c.shape[-1]))
    np.testing.assert_allclose(actual[0], weight[..., None] * v, rtol=2e-5, atol=2e-5)
    for name, value in zip(NAMES, actual, strict=True):
        assert np.isfinite(np.asarray(value)).all(), name
    # Every off-diagonal gate underflows to zero. The self-score has no
    # Householder transport and L_t-L_t has exactly zero derivative.
    for name, value in (("dw", actual[4]), ("dbeta", actual[5]), ("dL", actual[7])):
        np.testing.assert_allclose(value, 0., atol=2e-5, rtol=0, err_msg=name)


def test_interpreted_gate_outputs_and_gradients_are_causal():
    operands, cotangent, _ = _inputs(time=640, seed=7)
    prefix = 300
    cotangent = cotangent.at[:, prefix:].set(0)
    changed = [x.at[:, prefix:].add(0.7) for x in operands]
    w = changed[3][:, prefix:]
    changed[3] = changed[3].at[:, prefix:].set(w / jnp.linalg.norm(w, axis=-1, keepdims=True))
    changed[4] = changed[4].at[:, prefix:].set(jnp.clip(changed[4][:, prefix:], 0.01, 1.99))
    changed[-1] = operands[-1].at[:, prefix:].add(-50.)
    first = _interpreted(operands, cotangent)
    second = _interpreted(changed, cotangent)
    np.testing.assert_array_equal(first[0][:, :prefix], second[0][:, :prefix])
    for name, gradient in zip(NAMES[1:], second[1:], strict=True):
        np.testing.assert_array_equal(np.asarray(gradient)[:, prefix:], 0., err_msg=name)


def test_interpreted_per_head_prefix_offsets_leave_attention_unchanged():
    operands, cotangent, _ = _inputs(time=256, seed=8)
    shifted = [*operands[:-1], operands[-1] + jnp.asarray([2., -4.], jnp.float32)]
    _check(_interpreted(shifted, cotangent), _interpreted(operands, cotangent), 1e-3)


def test_forget_gate_rejects_wrong_shape_and_dtype():
    """Softmax and the exponential take the same gate (test_path_attention_fused.py)."""
    operands, _, _ = _inputs(time=128, seed=9)
    with triton_port.interpreter(HOPPER):
        for wrong in (operands[-1][..., None], operands[-1].astype(jnp.bfloat16)):
            with pytest.raises(ValueError):
                fused_path_attention(*operands[:5], kernel="query_polynomial", coefficients=operands[5],
                                     log_decay=wrong)


def _require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires an H100 or B300 allocation")


@pytest.mark.gpu
@pytest.mark.parametrize("time", (128, 640, 1024, 1152, 2048))
def test_gpu_degree_four_gate_matches_reference_across_chunk_and_split_boundaries(time):
    _require_gpu()
    operands, cotangent, _ = _inputs(time=time, dim=64, degree=4, seed=time, dtype=jnp.bfloat16)
    expected = jax.jit(lambda *xs: _reference(xs, cotangent))(*operands)
    _check(_compiled(operands, cotangent), expected, GPU_TOLERANCE)


@pytest.mark.gpu
def test_gpu_gate_chain_rule_matches_reference_at_production_context():
    _require_gpu()
    operands, cotangent, log_alpha = _inputs(time=2048, dim=64, seed=10, dtype=jnp.bfloat16)
    operands[-1] = log_alpha
    expected = jax.jit(lambda *xs: _reference(xs, cotangent, cumulative=False))(*operands)
    _check(_compiled(operands, cotangent, cumulative=False), expected, GPU_TOLERANCE)


@pytest.mark.gpu
@pytest.mark.parametrize("decay", (0., 1000.))
def test_gpu_open_and_strong_gates_remain_finite(decay):
    _require_gpu()
    operands, cotangent, _ = _inputs(time=1152, dim=64, seed=11, dtype=jnp.bfloat16, decay=decay)
    actual = _compiled(operands, cotangent)
    for name, value in zip(NAMES, actual, strict=True):
        assert np.isfinite(np.asarray(value)).all(), name
    if decay == 0:
        expected = jax.jit(lambda *xs: _reference(xs, cotangent))(*operands)
        _check(actual, expected, GPU_TOLERANCE)
    else:
        q, k, v, _, _, c, _ = [x.astype(jnp.float32) for x in operands]
        score = jnp.sum(q * k, axis=-1) / math.sqrt(q.shape[-1])
        weight = sum(c[..., r] * score ** r for r in range(c.shape[-1]))
        assert _relative(actual[0], weight[..., None] * v) < GPU_TOLERANCE["output"]
        for name, value in (("dw", actual[4]), ("dbeta", actual[5]), ("dL", actual[7])):
            np.testing.assert_allclose(value, 0., atol=2e-5, rtol=0, err_msg=name)
