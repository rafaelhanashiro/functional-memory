"""Per-degree score biases: explicit shifted powers and every gradient against an independent FP64
oracle, matched starts against the control recipes, weight decay, dispatch, refusals, and diagnostics."""
from dataclasses import replace
from functools import partial
import math
from pathlib import Path
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import score_bias
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.mosaic import api as mosaic
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.input_dependent import query_polynomial_attention
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.parameters import attention_parameter_count, attention_parameter_specs
from scaffolds.transformer.attention.score_bias import AMPLITUDE, KEY_KERNEL, OFFSET, QUERY_KERNEL
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import telemetry
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import run_tags

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments" / "transformer_pp" / "runs"
# The controls: the query polynomial with RoPE, and the per-order query-key gate without positional encoding.
CONTROLS = {"r25fffc16c2f3": 200_150_806, "r227b0ed7c605": 203_541_466}
MODES = ("static", "query", "query_key")
# Added parameters at 15 layers, 14 heads, width 896, degree 8.
ADDED = {"static": 1_680, "query": 1_508_640, "query_key": 3_013_920}
SHIFT_NAMES = {"row": ("score_bias_row_shift",),
               "pair": ("score_bias_query_input", "score_bias_key_input", "score_bias_offset", "score_bias_amplitude")}
GATE_NAMES = ("query_gate_input", "key_gate_input", "gate_log_rate")


def _config(mode="query", *, gated=False, **changes):
    model = dict(vocab_size=32, max_seq_len=16, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                 qk_head_dim=8, value_head_dim=8, intermediate_size=32, attention_backend="blocked",
                 attention_degree=4, attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.,
                 attention_coefficient_projection_initial_std=16 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_coefficient_kernel_weight_decay=True, attention_output_scaling="head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 rms_norm_fp32=True, initialization="transformer_truncated_reference", attention_score_bias=mode)
    if gated:
        model.update(attention_forget_gate=True, attention_forget_gate_mode="query_distance",
                     attention_forget_gate_conditioning="query_key", attention_forget_gate_per_order=True,
                     positional_encoding="none")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = 17, 32
    config.data.batch_size = config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-3
    config.optim.weight_decay = 0.1
    return config


def _params(model):
    return {path: value[...] for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _operands(kind, *, time=13, degree=4, gated=False, seed=5):
    """FP64 operands with unit-scale scores exact in any precision (integer Q/K of width 4, so s = q.k / 2 is
    a multiple of 1/2), signed shifts, and shifted scores exactly zero at pairs of the first and last orders:
    a row shift set to -s(t, t//2), or pair inputs of zero (tanh(0) = 0) with offsets 0 and -1/2."""
    rng = np.random.default_rng(seed)
    batch, heads = 2, 2
    q, k = (rng.integers(-1, 2, size=(batch, time, heads, 4)).astype(np.float64) for _ in range(2))
    v = rng.normal(size=(batch, time, heads, 3))
    c = rng.normal(size=(batch, time, heads, degree + 1)) / np.array([math.factorial(r) for r in range(degree + 1)])
    scores = np.einsum("bthd,bihd->bthi", q, k) / 2
    if kind == "row":
        beta = rng.normal(size=(batch, time, heads, degree))
        for order in {1, degree}:
            for t in range(time):
                beta[:, t, :, order - 1] = -scores[:, t, :, t // 2]
        shift = (beta,)
    else:
        p, key = rng.normal(size=(batch, time, heads, degree)), rng.normal(size=(batch, time, heads, degree)) * .7
        offset, amplitude = rng.normal(size=(heads, degree)) * .7, rng.normal(size=(heads, degree))
        for order, value in ((1, 0.), (degree, -.5)):
            p[..., order - 1] = key[..., order - 1] = 0.
            offset[:, order - 1] = value
        shift = (p, key, offset, amplitude)
    gate = ()
    if gated:
        gate = (rng.uniform(-3, 0, size=(batch, time, heads, degree + 1)),
                rng.normal(size=(batch, time, heads, degree + 1)) * .5, rng.uniform(-1, 1, size=(heads, degree + 1, 1)))
    return (q, k, v, c, *shift, *gate)


def _split(kind, rest):
    count = len(SHIFT_NAMES[kind])
    return rest[:count], rest[count:]


def _dense(kind, gated, q, k, v, c, *rest):
    """Independent oracle: every pair at once, explicit powers, W = c_0 G_0 + sum_r c_r G_r (s + beta_r)^r."""
    shift, gate = _split(kind, rest)
    time = q.shape[1]
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    if kind == "row":
        beta = jnp.swapaxes(shift[0], 1, 2)[:, :, :, None, :]
    else:
        p, key, offset, amplitude = shift
        inputs = jnp.swapaxes(p, 1, 2)[:, :, :, None, :] + jnp.swapaxes(key, 1, 2)[:, :, None, :, :]
        beta = offset[None, :, None, None, :] + amplitude[None, :, None, None, :] * jnp.tanh(inputs)
    distance = jnp.arange(time)[:, None] - jnp.arange(time)[None, :]
    coefficient = jnp.swapaxes(c, 1, 2)[:, :, :, None, :]
    if gated:
        u, w, rate = gate
        inputs = jnp.swapaxes(u, 1, 2)[:, :, :, None, :] + jnp.swapaxes(w, 1, 2)[:, :, None, :, :]
        rates = jnp.exp(rate[..., 0])[None, :, None, None, :] * jnp.logaddexp(0., inputs)
        coefficient = coefficient * jnp.exp(-jnp.maximum(distance, 0)[..., None] * rates)
    weights = coefficient[..., 0] + sum(coefficient[..., r] * (scores + beta[..., r - 1]) ** r
                                        for r in range(1, c.shape[-1]))
    weights = jnp.where(distance >= 0, weights, 0.)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


def _shifted_scores(kind, arrays):
    """z_r = s + beta_r on causal pairs [pairs, m], in NumPy."""
    q, k, _, c, *rest = arrays
    shift, _ = _split(kind, rest)
    scores = np.einsum("bthd,bihd->bhti", q, k) / 2
    if kind == "row":
        beta = np.swapaxes(shift[0], 1, 2)[:, :, :, None, :]
    else:
        p, key, offset, amplitude = shift
        beta = offset[None, :, None, None] + amplitude[None, :, None, None] * np.tanh(
            np.swapaxes(p, 1, 2)[:, :, :, None] + np.swapaxes(key, 1, 2)[:, :, None])
    causal = np.tril(np.ones((q.shape[1],) * 2, bool))
    return (scores[..., None] + beta)[:, :, causal]


def _reference(kind, gated, q, k, v, c, *rest, implementation="scan", compute=jnp.float32, block=4,
               diagnostics=False):
    shift, gate = _split(kind, rest)
    arguments = dict(attention_degree=c.shape[-1] - 1, attention_normalization=False,
                     query_polynomial_coefficients=c, **dict(zip(SHIFT_NAMES[kind], shift)))
    if gated:
        arguments.update(dict(zip(GATE_NAMES, gate)), query_decay_mode="query_distance")
    if implementation == "strips":
        return reference.strip_dot_product_attention(q, k, v, strip_size=block, **arguments)
    return reference._blocked_scan(q, k, v, compute=compute, query_block_size=block,
                                   return_diagnostics=diagnostics, **arguments)


def _assert_close(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert actual.shape == expected.shape and np.isfinite(actual).all(), label
    scale = max(np.abs(expected).max(), 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale, err_msg=label)


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("kind", ("row", "pair"))
@pytest.mark.parametrize("time,degree", ((1, 1), (13, 4), (11, 6)))
def test_reference_matches_float64_oracle_with_negative_and_zero_shifted_scores(kind, gated, time, degree):
    arrays = _operands(kind, time=time, degree=degree, gated=gated)
    shifted = _shifted_scores(kind, arrays)
    if time > 1:
        assert (shifted == 0).any() and (shifted < 0).any() and (shifted > 0).any()
    cotangent = np.random.default_rng(7).normal(size=arrays[2].shape)
    labels = ["output", *(f"d/d{name}" for name in ("q", "k", "v", "c", *SHIFT_NAMES[kind],
                                                       *(GATE_NAMES if gated else ())))]
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        wide = [jnp.asarray(array, jnp.float64) for array in arrays]
        expected = jax.tree.map(np.asarray, _vjp(partial(_dense, kind, gated), wide, jnp.asarray(cotangent)))
        expected = [expected[0], *expected[1]]
        for implementation in ("scan", "strips"):
            exact = jax.jit(lambda *a: _vjp(partial(_reference, kind, gated, implementation=implementation,
                                                    compute=jnp.float64), a, jnp.asarray(cotangent)))(*wide)
            for got, want, label in zip([exact[0], *exact[1]], expected, labels, strict=True):
                _assert_close(got, want, 1e-11, f"{implementation} FP64 {label}")
    with jax.default_matmul_precision("highest"):
        narrow = [jnp.asarray(array, jnp.float32) for array in arrays]
        actual = jax.jit(lambda *a: _vjp(partial(_reference, kind, gated), a,
                                         jnp.asarray(cotangent, jnp.float32)))(*narrow)
    for got, want, label in zip([actual[0], *actual[1]], expected, labels, strict=True):
        _assert_close(got, want, 3e-5, f"FP32 {label}")


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("mode", MODES)
def test_model_operands_chain_rule_reaches_every_bias_parameter(mode, gated):
    rng = np.random.default_rng(11)
    batch, time, heads, hidden, degree = 2, 9, 2, 6, 4
    names = score_bias.parameter_names(mode)
    values = {OFFSET: rng.normal(size=(heads, degree)) * .5, AMPLITUDE: rng.normal(size=(heads, degree)),
              QUERY_KERNEL: rng.normal(size=(heads, degree, hidden)) * .4,
              KEY_KERNEL: rng.normal(size=(heads, degree, hidden)) * .4}
    q, k = (rng.normal(size=(batch, time, heads, 4)) * .7 for _ in range(2))
    v = rng.normal(size=(batch, time, heads, 3))
    c = rng.normal(size=(batch, time, heads, degree + 1)) / np.array([math.factorial(r) for r in range(degree + 1)])
    gate = (rng.uniform(-3, 0, size=(batch, time, heads, degree + 1)),
            rng.normal(size=(batch, time, heads, degree + 1)) * .5,
            rng.uniform(-1, 1, size=(heads, degree + 1, 1))) if gated else ()
    arrays = (rng.normal(size=(batch, time, hidden)), *(values[name] for name in names), q, k, v, c, *gate)
    kind = "pair" if mode == "query_key" else "row"

    def production(x, *rest):
        parameters, (q, k, v, c, *gate) = dict(zip(names, rest[:len(names)])), rest[len(names):]
        operands = score_bias.attention_operands(mode, x, parameters)
        if kind == "row":
            assert set(operands) == {"score_bias_row_shift"}
        return _reference(kind, gated, q, k, v, c, *(operands[name] for name in SHIFT_NAMES[kind]), *gate)

    def oracle(x, *rest):
        parameters, (q, k, v, c, *gate) = dict(zip(names, rest[:len(names)])), rest[len(names):]
        offset = parameters[OFFSET]
        if mode == "static":
            shift = (jnp.broadcast_to(offset, (batch, time, *offset.shape)),)
        else:
            projected = jnp.einsum("btd,hrd->bthr", x, parameters[QUERY_KERNEL])
            shift = ((offset + parameters[AMPLITUDE] * jnp.tanh(projected),) if mode == "query" else
                     (projected, jnp.einsum("btd,hrd->bthr", x, parameters[KEY_KERNEL]), offset,
                      parameters[AMPLITUDE]))
        return _dense(kind, gated, q, k, v, c, *shift, *gate)

    cotangent = np.random.default_rng(3).normal(size=v.shape)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = jax.tree.map(np.asarray, _vjp(oracle, [jnp.asarray(a, jnp.float64) for a in arrays],
                                                 jnp.asarray(cotangent)))
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(production, a, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(a, jnp.float32) for a in arrays))
    labels = ["output", "d/dx", *(f"d/d{name}" for name in names), "d/dq", "d/dk", "d/dv", "d/dc",
              *(f"d/d{name}" for name in (GATE_NAMES if gated else ()))]
    for got, want, label in zip([actual[0], *actual[1]], [expected[0], *expected[1]], labels, strict=True):
        _assert_close(got, want, 3e-5, label)
    for name, gradient in zip(names, actual[1][1:]):
        assert np.linalg.norm(gradient) > 0, name


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
@pytest.mark.parametrize("operand", ("score_bias_row_shift", "score_bias_query_input", "score_bias_key_input"))
def test_invalid_shift_inputs_poison_only_their_causal_rows(operand, bad, gated):
    kind = "row" if operand == "score_bias_row_shift" else "pair"
    arrays = [jnp.asarray(array, jnp.float32) for array in _operands(kind, time=9, gated=gated)]
    position = 4 + SHIFT_NAMES[kind].index(operand)
    token = 5
    arrays[position] = arrays[position].at[0, token, 1, 2].set(bad)
    output = _reference(kind, gated, *arrays)
    key_side = operand == "score_bias_key_input"
    expected = np.zeros(output.shape[:3], bool)
    expected[0, token:, 1] = True
    if not key_side:
        expected[0, token + 1:, 1] = False
    np.testing.assert_array_equal(~np.isfinite(np.asarray(output)).all(axis=-1), expected)
    # Masked entries carry zero shift: with no cotangent on the poisoned rows, every per-token gradient of
    # the tokens the invalid entry never reaches (earlier tokens for a key input, later ones otherwise) is
    # finite, so no NaN crosses the causal mask.
    cotangent = jnp.broadcast_to(jnp.where(jnp.asarray(expected)[..., None], 0., 1.), output.shape).astype(jnp.float32)
    _, gradients = _vjp(partial(_reference, kind, gated), arrays, cotangent)
    untouched = slice(0, token) if key_side else slice(token + 1, None)
    for index, gradient in enumerate(gradients):
        if gradient.ndim == 4:  # per-token operands [batch, time, heads, channels]
            assert np.isfinite(np.asarray(gradient)[:, untouched]).all(), index


def test_zero_shift_reproduces_the_unshifted_kernels():
    for gated in (False, True):
        q, k, v, c, *rest = (jnp.asarray(array, jnp.float32) for array in _operands("pair", gated=gated))
        gate = dict(zip(GATE_NAMES, rest[4:]), query_decay_mode="query_distance") if gated else {}
        common = dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
                      query_block_size=4, **gate)
        control = reference.blocked_dot_product_attention(q, k, v, **common)
        zero_row = reference.blocked_dot_product_attention(q, k, v, **common,
                                                           score_bias_row_shift=jnp.zeros((*q.shape[:3], 4)))
        zero_pair = reference.blocked_dot_product_attention(
            q, k, v, **common, score_bias_query_input=rest[0], score_bias_key_input=rest[1],
            score_bias_offset=jnp.zeros((2, 4)), score_bias_amplitude=jnp.zeros((2, 4)))
        for shifted in (zero_row, zero_pair):
            np.testing.assert_allclose(shifted, control, rtol=1e-5, atol=1e-5)


def test_dispatch_reaches_the_mosaic_score_bias_entry_points(monkeypatch):
    calls = []

    def record(name):
        def entry(q, k, v, coefficients, *args, **kwargs):
            calls.append((name, args, kwargs))
            return v
        return entry

    monkeypatch.setattr(mosaic, "score_bias_query_polynomial_attention", record("query"))
    monkeypatch.setattr(mosaic, "score_bias_per_order_endpoint_attention", record("per_order_endpoint"))
    q, k, v, c, beta, u, w, rate = (jnp.asarray(a, jnp.float32) for a in _operands("row", time=9, gated=True))
    query_polynomial_attention(q, k, v, c, backend="mosaic", score_bias_row_shift=beta)
    query_polynomial_attention(q, k, v, c, backend="mosaic", decay_mode="query_distance", query_gate_input=u,
                               key_gate_input=w, gate_log_rate=rate, score_bias_row_shift=beta)
    p, key, offset, amplitude = (jnp.asarray(a, jnp.float32) for a in _operands("pair", time=9)[4:])
    query_polynomial_attention(q, k, v, c, backend="mosaic", score_bias_query_input=p, score_bias_key_input=key,
                               score_bias_offset=offset, score_bias_amplitude=amplitude)
    assert [call[0] for call in calls] == ["query", "per_order_endpoint", "query"]
    assert set(calls[0][2]) == {"row_shift"} and calls[0][2]["row_shift"] is beta
    assert set(calls[1][2]) == {"row_shift", *GATE_NAMES}
    assert set(calls[2][2]) == {"query_shift_input", "key_shift_input", "shift_offset", "shift_amplitude"}
    # The diagnostics' reference receives the same operands (the shifted kernel, not the unshifted one).
    _, stats = query_polynomial_attention(q, k, v, c, backend="mosaic", score_bias_row_shift=beta,
                                          return_diagnostics=True)
    assert "score_bias_z_sum" in stats


def test_every_other_kernel_and_backend_refuses_score_bias_operands():
    q, k, v, c, beta = (jnp.asarray(a, jnp.float32) for a in _operands("row", time=5))
    row = {"score_bias_row_shift": beta}
    refusals = (
        dict(attention_degree=4, attention_normalization=False, polynomial_coefficients=c[0, 0, 0]),
        dict(attention_degree=None, attention_normalization=True),
        dict(attention_degree=None, attention_normalization=False),
        dict(attention_degree=4, attention_normalization=True),
        dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
             key_polynomial_factors=jnp.ones_like(c), attention_coefficient_activation="tanh"),
        dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
             attention_coefficient_activation="sigmoid"),
    )
    for arguments in refusals:
        with pytest.raises(ValueError, match="score bias"):
            attention_core(q, k, v, **arguments, **row)
    for backend in ("fused", "manual"):
        with pytest.raises(ValueError, match="score bias"):
            query_polynomial_attention(q, k, v, c, backend=backend, **row)
    shared_gate = dict(decay_mode="query_distance", query_gate_input=q[..., 0], key_gate_input=k[..., 0],
                       gate_log_rate=jnp.zeros((2, 1)))
    for decay in (shared_gate, dict(decay_mode="query_distance", log_decay=-jnp.ones(q.shape[:3]))):
        with pytest.raises(ValueError, match="score bias"):
            query_polynomial_attention(q, k, v, c, **decay, **row)
    with pytest.raises(ValueError, match="row shift or the query-key pair"):
        query_polynomial_attention(q, k, v, c, score_bias_row_shift=beta, score_bias_offset=jnp.zeros((2, 4)))
    with pytest.raises(ValueError, match="shape"):
        query_polynomial_attention(q, k, v, c, score_bias_row_shift=beta[..., :3])


@pytest.mark.parametrize("run", CONTROLS)
def test_full_size_parameter_counts(run):
    control = load_config(RUNS / run / "config.yaml")
    assert model_parameter_count(control.model) == CONTROLS[run]
    for mode, added in ADDED.items():
        biased = replace(control.model, attention_score_bias=mode)
        assert model_parameter_count(biased) == CONTROLS[run] + added
        assert attention_parameter_count(biased) - attention_parameter_count(control.model) == added
        names = [spec.name for spec in attention_parameter_specs(biased)]
        assert names[-len(score_bias.parameter_names(mode)) - 1:-1] == list(score_bias.parameter_names(mode))


def _control(run, **changes):
    """A control recipe at a small shape (its every other setting kept), on the blocked reference."""
    config = load_config(RUNS / run / "config.yaml")
    return replace(config, model=replace(config.model, vocab_size=64, max_seq_len=16, num_layers=2, hidden_size=32,
                                         num_heads=2, num_kv_heads=2, intermediate_size=64,
                                         attention_backend="blocked", **changes))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("run", CONTROLS)
def test_start_keeps_every_control_array_and_draw(run, mode):
    control = _control(run)
    biased = replace(control, model=replace(control.model, attention_score_bias=mode))
    key = jax.random.PRNGKey(control.train.seed)

    def build(config):
        return build_model(config.model, nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 0)))

    baseline_model, model = build(control), build(biased)
    baseline, actual = _params(baseline_model), _params(model)
    names = score_bias.parameter_names(mode)
    assert set(actual) - set(baseline) == {(name,) for name in names} and set(baseline) <= set(actual)
    for path, value in baseline.items():
        np.testing.assert_array_equal(actual[path], value, err_msg=str(path))
    shapes = {OFFSET: (2, 2, 8), AMPLITUDE: (2, 2, 8), QUERY_KERNEL: (2, 2, 8, 32), KEY_KERNEL: (2, 2, 8, 32)}
    starts = {OFFSET: 0., AMPLITUDE: 1., QUERY_KERNEL: 0., KEY_KERNEL: 0.}
    for name in names:
        assert actual[(name,)].dtype == jnp.float32
        np.testing.assert_array_equal(actual[(name,)], np.full(shapes[name], starts[name], np.float32))
    # Without the new arrays the state hashes as the control's; the acceptance backbone check passes.
    params = nnx.state(model, nnx.Param)
    for name in names:
        del params[name]
    assert state_sha256(params) == state_sha256(nnx.state(baseline_model, nnx.Param))
    facts = acceptance.matches_softmax_backbone_initialization(biased)
    assert set(names) <= set(facts["mechanism_parameters"])
    # Zero shifts give the control's function, up to the rounding of explicit powers against Horner's rule.
    tokens = jnp.asarray(np.random.default_rng(3).integers(0, 64, size=(2, 16)), jnp.int32)
    np.testing.assert_allclose(model(tokens), baseline_model(tokens), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("mode", MODES)
def test_weight_decay_reaches_only_the_projections(mode, gated):
    config = _config(mode, gated=gated)
    model = build_model(config.model, 42)
    names = score_bias.parameter_names(mode)
    for name in names:
        current = getattr(model, name)
        current[...] = jnp.full(current[...].shape, .25)
    params = nnx.state(model, nnx.Param)
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded))}
    for name in names:
        assert mask[(name,)] == (name in (QUERY_KERNEL, KEY_KERNEL)), name
    assert mask[("query_polynomial_kernel",)] and not mask[("query_polynomial_bias",)]
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    values = {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(updates)}
    for name in names:
        decayed = -1e-3 * .1 * .25 if name in (QUERY_KERNEL, KEY_KERNEL) else 0.
        np.testing.assert_allclose(values[(name,)], decayed, rtol=2e-6, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("changes", (
    {"attention_score_bias": "pair"},
    {"attention_score_bias": None},
    {"attention_coefficient_activation": "sigmoid", "attention_coefficient_amplitude_initial": 1.,
     "attention_coefficient_scale": "none"},
    {"attention_coefficient_activation": "affine_sigmoid"},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_readout": "dense"},
    {"attention_coefficient_orders": (0, 2, 4)},
    {"attention_coefficient_gate": "sigmoid"},
    {"positional_encoding": "path"},
    {"attention_degree": 13},
    {"attention_backend": "auto"},
    {"attention_backend": "fused"},
    {"attention_backend": "mosaic", "qk_head_dim": 12},
    {"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance"},
    {"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance",
     "attention_forget_gate_conditioning": "query_key"},
    {"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance",
     "attention_forget_gate_per_order": True},
    {"attention_forget_gate": True, "attention_forget_gate_per_order": True},
))
def test_unqualified_recipes_are_rejected(changes):
    with pytest.raises(ValueError, match="score bias"):
        _config(**changes)


def test_rejected_training_options_and_accepted_backbones():
    for gated in (False, True):
        for mode in MODES:
            validate_config(_config(mode, gated=gated))
            mosaic_config = _config(mode, gated=gated, attention_backend="mosaic", qk_head_dim=64, value_head_dim=64)
            mosaic_config.train.dtype = "bfloat16"
            validate_config(mosaic_config)
    config = _config()
    for optim in (dict(coefficient_release_schedule="all", coefficient_release_step=1),
                  dict(query_coefficient_learning_rate_scale=2.0)):
        with pytest.raises(ValueError, match="score bias"):
            validate_config(replace(config, optim=replace(config.optim, **optim)))
    continuation = replace(config.continuation, parent_config="parent.yaml", parent_config_sha256="0" * 64,
                           parent_checkpoint="parent/7", parent_checkpoint_sha256="0" * 64, parent_step=1)
    with pytest.raises(ValueError, match="score bias"):
        validate_config(replace(config, continuation=continuation))
    with pytest.raises(ValueError, match="Unknown attention score bias"):
        MultiHeadAttention(16, 2, 2, 8, 8, backend="blocked", attention_degree=4, attention_normalization=False,
                           attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                           attention_score_bias="pair", rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="score bias"):
        MultiHeadAttention(16, 2, 2, 8, 8, backend="blocked", attention_degree=4, attention_normalization=False,
                           attention_coefficient_conditioning="query", attention_coefficient_activation="sigmoid",
                           attention_score_bias="query", rngs=nnx.Rngs(0))
    model = build_model(_config().model, 0)
    tokens = jnp.zeros((1, 8), jnp.int32)
    with pytest.raises(ValueError, match="score bias"):
        model(tokens, return_coefficients=True)
    # Historical recipes, which lack the field, keep comparing equal.
    current = config_to_dict(_config("none"))
    legacy = {**current, "model": {key: value for key, value in current["model"].items()
                                   if key != "attention_score_bias"}}
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(config_to_dict(_config("query")), legacy)


def test_backend_resolution_never_falls_back():
    for request in ("auto", "fused", "xla"):
        with pytest.raises(ValueError, match="score bias"):
            resolve_attention_backend(SimpleNamespace(attention_backend=request, attention_score_bias="query"),
                                      platform="gpu")
    mosaic_config = _config(attention_backend="mosaic", qk_head_dim=64, value_head_dim=64).model
    assert resolve_attention_backend(mosaic_config, dtype="bfloat16", platform="gpu",
                                     sequence_length=128) == "mosaic"
    assert resolve_attention_backend(_config().model, platform="cpu") == "blocked"
    assert not mosaic_config.mosaic_supported() and not mosaic_config.mosaic_decay_supported()
    assert mosaic_config.mosaic_score_bias_supported()


@pytest.mark.parametrize("backend", ("blocked", "mosaic"))
def test_records_create_allocates_score_biases_only_on_mosaic(tmp_path, monkeypatch, capsys, backend):
    from scaffolds.records.__main__ import main

    recipe = yaml.safe_load((RUNS / "r25fffc16c2f3" / "config.yaml").read_text())
    recipe["model"].update(attention_score_bias="query", attention_backend=backend)
    source = tmp_path / "recipe.yaml"
    source.write_text(yaml.safe_dump(recipe))
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source)])
    if backend == "blocked":
        with pytest.raises(SystemExit) as failure:
            main()
        assert failure.value.code == 2 and "Mosaic GPU kernels" in capsys.readouterr().err
        assert not study.exists()
    else:
        main()
        run = Path(capsys.readouterr().out.strip())
        model = yaml.safe_load((run / "config.yaml").read_text())["model"]
        assert model["attention_score_bias"] == "query" and model["attention_backend"] == "mosaic"


def test_identity_tags_and_telemetry_groups():
    for gated in (False, True):
        for mode in MODES:
            config = _config(mode, gated=gated, attention_backend="mosaic", qk_head_dim=64, value_head_dim=64)
            identity = attention_identity(config)
            names = score_bias.parameter_names(mode)
            assert identity["attention_score_bias"] == mode
            assert identity["score_bias_parameter_count"] == sum(
                int(np.prod(value.shape)) for path, value in _params(build_model(
                    replace(config.model, attention_backend="blocked"), 0)).items() if path[0] in names)
            assert identity["score_bias_weight_decay"] == {name: name in (QUERY_KERNEL, KEY_KERNEL) for name in names}
            formula = identity["attention_formula"]
            assert "(s_ti + beta_r(t,i))^r" in formula and "s_ti^r" not in formula.replace("(s_ti + beta_r(t,i))^r", "")
            assert formula.endswith(", then per-head RMSNorm over value channels with a learned gain")
            assert "positive biased score does not imply a positive weight" in formula
            assert formula.count("each weight is multiplied") == 0
            shift = "_pair_score_bias" if mode == "query_key" else "_row_score_bias"
            assert identity["attention_implementation"] == (
                ("pallas_mosaic_gpu_query_polynomial_per_order_endpoint_decay" if gated
                 else "pallas_mosaic_gpu_query_polynomial") + shift)
            assert f"score_bias_{mode}" in run_tags(config, {})
    for name in score_bias.ALL_PARAMETERS:
        assert telemetry.parameter_group((jax.tree_util.DictKey(name),)) == "score_bias"


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("mode", MODES)
def test_diagnostics_report_every_score_bias_measurement(mode, gated):
    config = _config(mode, gated=gated)
    state = create_state(config, jax.random.PRNGKey(0))
    rng = np.random.default_rng(4)
    for name, scale, offset in ((OFFSET, .5, -.3), (AMPLITUDE, .5, 1.), (QUERY_KERNEL, .5, 0.), (KEY_KERNEL, .5, 0.)):
        if name in state.params:
            state.params[name][...] = jnp.asarray(rng.normal(size=state.params[name].shape) * scale + offset,
                                                  jnp.float32)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == "query_polynomial_with_per_degree_score_biases"
    assert report["attention_score_bias"] == mode
    biases = report["coefficients"]["score_biases"]
    layers, heads, degree, time = 2, 2, 4, 16
    pairs = time * (time + 1) // 2
    assert biases["finite"] and biases["mode"] == mode and biases["orders"] == [1, 2, 3, 4]
    assert set(biases["parameters"]) == ({"offset_b"} | ({"amplitude_a", "query_kernel_l2"} if mode != "static" else set())
                                         | ({"key_kernel_l2"} if mode == "query_key" else set()))
    assert "positive weight" in biases["note"]
    shifted = biases["shifted_score"]
    assert shifted["causal_pair_count"] == pairs
    for name in ("mean", "std", "min", "max", "negative_fraction", "near_zero_fraction"):
        values = np.asarray(shifted[name], np.float64)
        assert values.shape == (layers, heads, degree) and np.isfinite(values).all(), name
    assert (np.asarray(shifted["min"]) <= np.asarray(shifted["mean"])).all()
    assert 0 < np.mean(shifted["negative_fraction"]) < 1
    histogram = np.asarray(biases["shifted_score_histogram"]["counts"])
    assert histogram.shape == (layers, degree, len(score_bias.HISTOGRAM_EDGES) + 1)
    np.testing.assert_array_equal(histogram.sum(axis=-1), heads * pairs)
    if mode == "static":
        assert biases["beta"]["domain"].startswith("parameters")
    else:
        beta = {name: np.asarray(biases["beta"][name], np.float64) for name in ("mean", "std", "min", "max")}
        assert all(value.shape == (layers, heads, degree) and np.isfinite(value).all() for value in beta.values())
        assert (beta["min"] <= beta["max"]).all() and (beta["std"] > 0).all()
    if mode == "query":
        # Recompute layer 0's shifts from its attention input: b + a tanh(x u) over the sample's tokens.
        model = nnx.merge(state.graphdef, state.params)
        x = np.asarray(model.layers_0.attention_norm(model.token_embedding(jnp.asarray(sample[:, :-1]))), np.float64)
        offset, amplitude, kernel = (np.asarray(state.params[name][...], np.float64)[0]
                                     for name in (OFFSET, AMPLITUDE, QUERY_KERNEL))
        expected = offset + amplitude * np.tanh(np.einsum("btd,hrd->bthr", x, kernel))
        np.testing.assert_allclose(biases["beta"]["mean"][0], expected.mean(axis=(0, 1)), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(biases["beta"]["max"][0], expected.max(axis=(0, 1)), rtol=1e-5, atol=1e-6)
    terms = biases["order_terms"]
    assert terms["orders"] == list(range(degree + 1))
    for name in ("abs_mean", "rms", "row_share"):
        values = np.asarray(terms[name], np.float64)
        assert values.shape == (layers, heads, degree + 1) and np.isfinite(values).all(), name
    np.testing.assert_allclose(np.sum(terms["row_share"], axis=-1), 1, rtol=1e-5)
    cancellation = biases["cancellation"]
    for name in ("pair_mean", "pair_max", "value_mean", "value_max"):
        values = np.asarray(cancellation[name], np.float64)
        assert values.shape == (layers, heads) and np.isfinite(values).all() and (values >= 1 - 1e-5).all(), name
    assert (np.asarray(cancellation["pair_max"]) >= np.asarray(cancellation["pair_mean"])).all()
