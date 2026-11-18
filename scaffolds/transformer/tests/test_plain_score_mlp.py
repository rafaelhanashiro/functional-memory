"""Conventional score MLP: independent score/activation definition and its full derivatives.

The host FP64 oracle writes out the one-hidden-layer MLP, including its static readout,
width scaling and output bias. Production-length GPU cases exercise both requested widths.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import score_mlp
from scaffolds.transformer.attention.backends import fused_score_mlp as fused
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.parameters import initial_attention_parameter_values

NAMES = ("q", "k", "v", "x", "a", "U", "b", "w", "b_out")


def _inputs(width, *, time=19, batch=2, heads=2, channels=6, value=3, hidden=7, dtype=jnp.float32):
    rng = np.random.default_rng(109 + width)
    shapes = ((batch, time, heads, channels), (batch, time, heads, channels),
              (batch, time, heads, value), (batch, time, hidden), (heads, width),
              (heads, hidden, width), (heads, width), (heads, width), (heads, 1))
    arrays = [rng.normal(size=shape) for shape in shapes]
    arrays[5] /= math.sqrt(hidden)
    arrays[6] *= .5
    arrays[8] += .2
    operands = tuple(jnp.asarray(array, dtype if index < 3 else jnp.float32)
                     for index, array in enumerate(arrays))
    return operands, jnp.asarray(rng.normal(size=shapes[2]), dtype)


def _production(q, k, v, x, a, u, b, w, bias, *, backend="blocked", implementation="scan",
                return_diagnostics=False):
    parameters = {score_mlp.AMPLITUDE_BIAS: a, score_mlp.OFFSET_KERNEL: u,
                  score_mlp.OFFSET_BIAS: b, score_mlp.SLOPE: w, score_mlp.OUTPUT_BIAS: bias}
    amplitude, offset, slope, output_bias = score_mlp.project_parameters(x, parameters, "conventional")
    return score_mlp.score_mlp_attention(q, k, v, amplitude, offset, slope, output_bias=output_bias,
                                        backend=backend, blocked_implementation=implementation,
                                        return_diagnostics=return_diagnostics)


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _dense(q, k, v, x, a, u, b, w, bias):
    """Independent JAX expression for checking the analytic host oracle on a tiny case."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    offset = jnp.einsum("btd,hdj->bhtj", x, u) + b[None, :, None]
    pre = w[None, :, None, None] * scores[..., None] + offset[..., None, :]
    weights = bias[None, :, None] + jnp.sum(a[None, :, None, None] * pre / (1 + jnp.exp(-pre)), axis=-1) / math.sqrt(a.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, weights, 0.0), v)


def _oracle(operands, cotangent):
    """Host FP64 output and analytic derivatives, one hidden unit at a time, no production helpers."""
    q, k, v, x, a, u, b, w, bias = (np.asarray(array, np.float64) for array in operands)
    qh, kh, vh, gh = (np.asarray(array, np.float64).transpose(0, 2, 1, 3)
                       for array in (q, k, v, cotangent))
    scores = qh @ kh.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    causal = np.tril(np.ones(scores.shape[-2:], dtype=bool))
    dkernel = np.where(causal, gh @ vh.swapaxes(-1, -2), 0.0)
    offset = np.einsum("btd,hdj->bhtj", x, u) + b[None, :, None]
    kernel = np.broadcast_to(bias[None, :, None], scores.shape).copy()
    dscore = np.zeros_like(scores)
    da, db, dw, doffset = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w), np.zeros_like(offset)
    scale = 1 / math.sqrt(a.shape[-1])
    for j in range(a.shape[-1]):
        pre = w[None, :, None, None, j] * scores + offset[..., j, None]
        sigmoid = 1 / (1 + np.exp(-pre))
        value = pre * sigmoid
        derivative = sigmoid * (1 + pre * (1 - sigmoid))
        kernel += a[None, :, None, None, j] * scale * value
        dpre = dkernel * a[None, :, None, None, j] * scale * derivative
        dscore += w[None, :, None, None, j] * dpre
        da[:, j] = np.sum(dkernel * value * scale, axis=(0, 2, 3))
        doffset[..., j] = np.sum(dpre, axis=-1)
        db[:, j] = np.sum(doffset[..., j], axis=(0, 2))
        dw[:, j] = np.sum(dpre * scores, axis=(0, 2, 3))
    weights = np.where(causal, kernel, 0.0)
    output = weights @ vh
    dx = np.einsum("bhtj,hdj->btd", doffset, u)
    du = np.einsum("btd,bhtj->hdj", x, doffset)
    dbias = np.sum(dkernel, axis=(0, 2, 3))[:, None]
    gradients = ((dscore @ kh / math.sqrt(q.shape[-1])).transpose(0, 2, 1, 3),
                 (dscore.swapaxes(-1, -2) @ qh / math.sqrt(q.shape[-1])).transpose(0, 2, 1, 3),
                 (weights.swapaxes(-1, -2) @ gh).transpose(0, 2, 1, 3), dx, da, du, db, dw, dbias)
    return output.transpose(0, 2, 1, 3), gradients


def _compare(actual, expected, tolerance):
    for name, observed, wanted in zip(("output", *NAMES), (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert np.isfinite(observed).all(), name
        error = np.linalg.norm(observed - wanted) / max(np.linalg.norm(wanted), 1e-12)
        limit = tolerance[name] if isinstance(tolerance, dict) else tolerance
        assert error < limit, f"{name}: relative error {error} >= {limit}"


def _interpret(monkeypatch):
    """Exact FP32 CPU emulation of the same fused Pallas programs used on the GPU."""
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))
    def load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)
    monkeypatch.setattr(fused, "_load", load)
    entry = fused.fused_score_mlp_attention
    monkeypatch.setattr(fused, "fused_score_mlp_attention", lambda *args, **kwargs: entry(*args, **kwargs, interpret=True))


def test_host_oracle_matches_independent_float64_autodiff():
    operands, cotangent = _inputs(4, time=5, batch=1)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(_dense, tuple(jnp.asarray(a, jnp.float64) for a in operands),
                        jnp.asarray(cotangent, jnp.float64))
    _compare(_oracle(operands, cotangent), expected, 1e-11)


@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("backend", ("scan", "strips", "fused_interpreted"))
def test_conventional_outputs_and_all_gradients_match_float64(monkeypatch, width, backend):
    if backend == "fused_interpreted":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(width, time=37, batch=1)
    function = lambda *x: _production(*x, backend="fused" if backend == "fused_interpreted" else "blocked",
                                      implementation="strips" if backend == "strips" else "scan")
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(function, x, cotangent))(*operands)
    _compare(actual, _oracle(operands, cotangent), 3e-5)


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_conventional_causality_includes_activation_conditioning_and_bias(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(4, time=37, batch=1)
    prefix = 19
    cotangent = cotangent.at[:, prefix:].set(0)
    function = lambda *x: _production(*x, backend=backend)
    actual = _vjp(function, operands, cotangent)
    altered = tuple(array.at[:, prefix:].add(5) if index < 4 else array for index, array in enumerate(operands))
    changed = _vjp(function, altered, cotangent)
    np.testing.assert_array_equal(actual[0][:, :prefix], changed[0][:, :prefix])
    for index, (left, right) in enumerate(zip(actual[1], changed[1], strict=True)):
        np.testing.assert_array_equal(left, right, err_msg=NAMES[index])
        if index < 4:
            np.testing.assert_array_equal(left[:, prefix:], jnp.zeros_like(left[:, prefix:]))


def test_static_readout_and_output_bias_are_in_attention_weights_and_diagnostics():
    operands, _ = _inputs(4, time=7, batch=1)
    q, k, v, x, a, u, b, w, bias = operands
    a = jnp.zeros_like(a)
    bias = jnp.asarray([[2.0], [-3.0]], jnp.float32)
    output, diagnostics = _production(q, k, v, x, a, u, b, w, bias, return_diagnostics=True)
    np.testing.assert_allclose(output, jnp.cumsum(v, axis=1) * bias[None, None], rtol=2e-6, atol=2e-6)
    assert float(diagnostics["kernel_min"]) == -3
    assert float(diagnostics["kernel_max"]) == 2
    assert float(diagnostics["negative_kernel_fraction"]) == .5
    parameters = {score_mlp.AMPLITUDE_BIAS: operands[4], score_mlp.OFFSET_KERNEL: u,
                  score_mlp.OFFSET_BIAS: b, score_mlp.SLOPE: w, score_mlp.OUTPUT_BIAS: bias}
    original = score_mlp.project_parameters(x, parameters, "conventional")
    changed = score_mlp.project_parameters(x + 1, parameters, "conventional")
    np.testing.assert_array_equal(original[0], changed[0])
    assert not np.array_equal(original[1], changed[1])


@pytest.mark.parametrize("width", (4, 8))
def test_conventional_initialization_has_distinct_live_units_and_standard_weight_decay(width):
    specs = score_mlp.conventional_parameter_specs(hidden_size=128, width=width)
    values = initial_attention_parameter_values(specs, num_layers=4, num_heads=8, key=jax.random.key(11))
    assert set(values) == set(score_mlp.CONVENTIONAL_PARAMETERS)
    assert score_mlp.AMPLITUDE_KERNEL not in values
    for name in (score_mlp.AMPLITUDE_BIAS, score_mlp.OFFSET_KERNEL, score_mlp.SLOPE):
        assert np.isfinite(values[name]).all() and np.std(values[name]) > 0
        assert not np.array_equal(values[name][0], values[name][1])
    assert .75 < np.std(values[score_mlp.OFFSET_KERNEL]) * math.sqrt(128) < 1.25
    for name in (score_mlp.OFFSET_BIAS, score_mlp.OUTPUT_BIAS):
        np.testing.assert_array_equal(values[name], jnp.zeros_like(values[name]))
    expected_decay = {score_mlp.AMPLITUDE_BIAS, score_mlp.OFFSET_KERNEL, score_mlp.SLOPE}
    assert {spec.name for spec in specs if spec.weight_decay} == expected_decay


@pytest.mark.parametrize("form,decay", (("exp_residual", False), ("kernel", True)))
def test_output_bias_rejects_incompatible_form_or_gate(form, decay):
    operands, _ = _inputs(4, time=7, batch=1)
    q, k, v, x, a, u, b, w, bias = operands
    amplitude = jnp.broadcast_to(a, (*q.shape[:3], a.shape[-1]))
    offset = jnp.einsum("btd,hdj->bthj", x, u) + b
    with pytest.raises(ValueError, match="output bias"):
        blocked_dot_product_attention(q, k, v, attention_normalization=False, score_mlp_amplitude=amplitude,
                                      score_mlp_offset=offset, score_mlp_slope=w, score_mlp_form=form,
                                      score_mlp_output_bias=bias,
                                      query_log_decay=jnp.zeros(q.shape[:3]) if decay else None)


@pytest.mark.gpu
@pytest.mark.parametrize("width", (4, 8))
def test_gpu_conventional_production_length_outputs_and_all_gradients(width):
    """T=2048, dk=dv64, BF16 QKV; FP32 activations/parameters and a nonzero output bias."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    operands, cotangent = _inputs(width, time=2048, batch=1, heads=2, channels=64, value=64,
                                  hidden=896, dtype=jnp.bfloat16)
    actual = jax.jit(lambda *x: _vjp(lambda *y: _production(*y, backend="fused"), x, cotangent))(*operands)
    # A BF16 output/operand-gradient cast has an approximately .2% relative floor;
    # the value gradient also sums the independently computed constant-bias path.
    limits = {name: (1e-2 if name in ("output", "q", "k", "v") else 3e-4)
              for name in ("output", *NAMES)}
    _compare(actual, _oracle(operands, cotangent), limits)
