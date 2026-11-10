"""Query-distance forget gates on softmax and the unnormalized exponential.

Every score gains (t - i) log alpha_t, the query's own Gated DeltaNet rate times the key distance:
ALiBi with a learned, input-dependent slope per query token. Softmax normalizes the decayed logits
(its row maximum is taken over them); the exponential exponentiates them. A nonfinite or positive
raw log alpha_t poisons its query's row, as for the query polynomials' query-distance gate. The
mechanism takes the input-form softplus gate with query conditioning, RoPE or no positional
encoding, on the blocked backend (scan or causal strips); its query-key conditioned form, a rate per
query-key pair, is checked in test_exponential_query_key.py.
"""
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import (
    blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import alibi_log_rate, constant_log_decay, log_decay
from scaffolds.transformer.attention.multihead import MultiHeadAttention, attention_from_config
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, validate_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample,
)
from scaffolds.transformer.training.state import create_state

KERNELS = ("softmax", "exp")
IMPLEMENTATIONS = ("scan", "strips")
GATE = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")
# (time, scan block, strip): ragged 8-query blocks and 16-query strips of 37 tokens, one block and
# one strip longer than the sequence, and a single query.
SHAPES = ((37, 8, 16), (5, 8, 8), (1, 8, 8))


def _config(kernel="softmax", **changes):
    """Tiny query-distance recipes as allocated: no positional encoding, canonical QK RMSNorm,
    causal strips, and per-head output RMSNorm for exp."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=4, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
                 attention_blocked_implementation="strips", attention_normalization=kernel == "softmax",
                 attention_forget_gate=True, attention_forget_gate_mode="query_distance",
                 positional_encoding="none",
                 attention_output_scaling="none" if kernel == "softmax" else "head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _dense(q, k, v, log_alpha, *, normalized):
    """All pairs with explicit distances: softmax(s + B) V, or exp(s + B) V, B_ti = (t - i) log alpha_t."""
    time = q.shape[1]
    distance = jnp.arange(time)[:, None] - jnp.arange(time)[None, :]
    causal = distance >= 0
    logits = (jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
              + jnp.swapaxes(log_alpha, 1, 2)[..., None] * jnp.where(causal, distance, 0))
    if normalized:
        weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1)
    else:
        weights = jnp.where(causal, jnp.exp(jnp.where(causal, logits, 0.0)), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def _numpy_vjp(q, k, v, log_alpha, cotangent, *, normalized):
    """Host FP64 forward and analytic VJP (dQ, dK, dV, d log alpha), one head at a time, with no
    JAX compilation: the oracle of the GPU cases."""
    q, k, v, log_alpha, cotangent = (np.asarray(x, np.float64) for x in (q, k, v, log_alpha, cotangent))
    batch, time, heads, width = q.shape
    output, dq, dk, dv = np.zeros_like(v), np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    dlog = np.zeros_like(log_alpha)
    distance = np.arange(time)[:, None] - np.arange(time)[None, :]
    causal = distance >= 0
    distance = np.where(causal, distance, 0)
    for b in range(batch):
        for h in range(heads):
            logits = q[b, :, h] @ k[b, :, h].T / math.sqrt(width) + log_alpha[b, :, h, None] * distance
            if normalized:
                logits = np.where(causal, logits, -np.inf)
                weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
                weights /= weights.sum(axis=-1, keepdims=True)
            else:
                weights = np.where(causal, np.exp(np.where(causal, logits, 0.0)), 0.0)
            output[b, :, h] = weights @ v[b, :, h]
            dweights = cotangent[b, :, h] @ v[b, :, h].T
            dlogits = weights * (dweights - np.sum(dweights * weights, axis=-1, keepdims=True) if normalized
                                 else dweights)
            dq[b, :, h] = dlogits @ k[b, :, h] / math.sqrt(width)
            dk[b, :, h] = dlogits.T @ q[b, :, h] / math.sqrt(width)
            dv[b, :, h] = weights.T @ cotangent[b, :, h]
            dlog[b, :, h] = np.sum(dlogits * distance, axis=-1)
    return output, (dq, dk, dv, dlog)


def _inputs(time, *, batch=2, heads=3, width=8, seed=0):
    """Q, K, V, raw log alpha with half-lives from under one token to about a thousand, and a cotangent."""
    rng = np.random.default_rng(seed + 100 * time)
    q, k, v, cotangent = (rng.standard_normal((batch, time, heads, width)) for _ in range(4))
    log_alpha = -np.exp(rng.uniform(-7.0, 0.5, (batch, time, heads)))
    return (q, k, v, log_alpha), cotangent


def _production(implementation, *, normalized, block=8, strip=16, compute=None):
    """The blocked backend's query-distance call: the public scan or strips, or the scan's own code in
    ``compute`` (FP64 for exact-arithmetic checks); the strips compute in the inputs' dtype."""
    options = dict(attention_normalization=normalized, query_decay_mode="query_distance")
    if implementation == "strips":
        attend = partial(strip_dot_product_attention, strip_size=strip, **options)
    elif compute is not None:
        attend = partial(reference._blocked_scan, compute=compute, query_block_size=block, **options)
    else:
        attend = partial(blocked_dot_product_attention, query_block_size=block, **options)
    return lambda q, k, v, log_alpha: attend(q, k, v, query_log_decay=log_alpha)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _compare(actual, expected, tolerance, names=("output", "q", "k", "v", "log_alpha"), zeros=()):
    """Relative L2 errors below ``tolerance``; the gradients named in ``zeros`` must be exactly zero."""
    for name, observed, wanted in zip(names, (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert observed.shape == wanted.shape and np.isfinite(observed).all(), name
        if name in zeros:
            np.testing.assert_array_equal(wanted, 0, err_msg=name)
            np.testing.assert_array_equal(observed, 0, err_msg=name)
            continue
        assert np.linalg.norm(wanted) > 0, name
        error = _relative_l2(observed, wanted)
        assert error < tolerance, f"{name}: relative L2 error {error:.3e} >= {tolerance}"


def _flat(state):
    return {path: np.asarray(value.get_value()) for path, value in nnx.to_flat_state(state)}


# --- The operation against FP64 all-pairs references ------------------------------------------------

@pytest.mark.parametrize("time", (1, 5, 19))
@pytest.mark.parametrize("kernel", KERNELS)
def test_numpy_oracle_matches_float64_autodiff(kernel, time):
    """The GPU cases' host oracle: output and analytic dQ, dK, dV, d log alpha against FP64 autodiff."""
    arrays, cotangent = _inputs(time)
    normalized = kernel == "softmax"
    observed = _numpy_vjp(*arrays, cotangent, normalized=normalized)
    with jax.enable_x64():
        expected = _vjp(partial(_dense, normalized=normalized), tuple(jnp.asarray(x) for x in arrays),
                        jnp.asarray(cotangent))
    for got, want in zip(jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-11, atol=1e-11)
    if time == 1:
        # Distance zero: the decay and its derivative are exact.
        np.testing.assert_array_equal(observed[1][3], np.zeros_like(arrays[3]))


@pytest.mark.parametrize("time,block,strip", SHAPES)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_float64_blocks_and_strips_match_the_all_pairs_reference(kernel, implementation, time, block, strip):
    """In exact arithmetic (the scan's code and the strips in FP64), the output and the gradients of
    Q, K, V, and the raw log alpha equal the explicit all-pairs reference to rounding."""
    arrays, cotangent = _inputs(time)
    normalized = kernel == "softmax"
    with jax.enable_x64():
        operands = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        expected = _vjp(partial(_dense, normalized=normalized), operands, jnp.asarray(cotangent))
        production = _production(implementation, normalized=normalized, block=block, strip=strip,
                                 compute=jnp.float64)
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent)))(*operands)
    for name, got, want in zip(("output", "q", "k", "v", "log_alpha"), jax.tree.leaves(actual),
                               jax.tree.leaves(expected), strict=True):
        got, want = np.asarray(got), np.asarray(want)
        assert got.dtype == np.float64 and got.shape == want.shape, name
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * max(np.abs(want).max(), 1.0), err_msg=name)


@pytest.mark.parametrize("time,block,strip", SHAPES)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_float32_blocks_and_strips_match_float64(kernel, implementation, time, block, strip):
    """The public FP32 scan and strips: output and every operand gradient within 3e-5 of FP64."""
    arrays, cotangent = _inputs(time)
    normalized = kernel == "softmax"
    with jax.enable_x64():
        expected = _vjp(partial(_dense, normalized=normalized), tuple(jnp.asarray(x, jnp.float64) for x in arrays),
                        jnp.asarray(cotangent))
        expected = jax.tree.map(np.asarray, expected)
    production = _production(implementation, normalized=normalized, block=block, strip=strip)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(x, jnp.float32) for x in arrays))
    # A single query keeps its own value with decay one: the gate cannot affect it, and its softmax
    # weight is one whatever its score.
    zeros = () if time > 1 else ("log_alpha", "q", "k") if normalized else ("log_alpha",)
    _compare(actual, expected, 3e-5, zeros=zeros)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_gate_projection_chain_rule_matches_float64(kernel, implementation):
    """log alpha_t = -exp(A_log) softplus(x_t w + dt_bias) from forget_gate.log_decay, raw (no cumulative
    sum), through either implementation: the output and the gradients of Q, K, V, x, and every gate
    parameter against an independent FP64 formula."""
    rng = np.random.default_rng(43)
    (q, k, v, _), cotangent = _inputs(23, heads=2)
    x = rng.standard_normal((2, 23, 7))
    kernel_w = rng.uniform(-1, 1, (2, 7)) / math.sqrt(7)
    log_rate, dt_bias = np.log([[0.7], [3.0]]), np.asarray([[-2.5], [-4.0]])
    operands = (q, k, v, x, kernel_w, log_rate, dt_bias)
    normalized = kernel == "softmax"

    def oracle(q, k, v, x, w, a, b):
        log_alpha = -jnp.exp(a[:, 0]) * jnp.logaddexp(0.0, jnp.einsum("btd,hd->bth", x, w) + b[:, 0])
        return _dense(q, k, v, log_alpha, normalized=normalized)

    attend = _production(implementation, normalized=normalized)

    def production(q, k, v, x, w, a, b):
        return attend(q, k, v, log_decay(x, w, a, b))

    with jax.enable_x64():
        expected = _vjp(oracle, tuple(jnp.asarray(value, jnp.float64) for value in operands), jnp.asarray(cotangent))
        expected = jax.tree.map(np.asarray, expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(production, a, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(value, jnp.float32) for value in operands))
    _compare(actual, expected, 4e-5, names=("output", "q", "k", "v", "x", "kernel", "log_rate", "dt_bias"))
    for name, gradient in zip(("kernel", "log_rate", "dt_bias"), actual[1][4:]):
        assert np.all(np.linalg.norm(np.asarray(gradient), axis=-1) > 0), name  # every head


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_attention_module_with_its_gate_parameters_matches_float64(kernel, implementation, monkeypatch):
    """The attention module (projections, no positional encoding, raw log alpha routed to the blocked
    backend, output projection) against FP64 built from its weights: the output and the gradients of
    x and of every gate parameter, through the recipe's implementation (counted) in 8-query blocks or
    strips."""
    calls = []

    def counted(original, name, **options):
        def attend(*args, **kwargs):
            calls.append(name)
            return original(*args, **options, **kwargs)
        return attend

    monkeypatch.setattr(reference, "strip_dot_product_attention",
                        counted(reference.strip_dot_product_attention, "strips", strip_size=8))
    monkeypatch.setattr(reference, "blocked_dot_product_attention",
                        counted(reference.blocked_dot_product_attention, "scan", query_block_size=8))
    batch, time, hidden, heads, width = 2, 37, 12, 3, 4
    normalized = kernel == "softmax"
    module = MultiHeadAttention(hidden, heads, heads, width, width, backend="blocked",
                                attention_normalization=normalized, positional_encoding="none",
                                attention_forget_gate=True, attention_forget_gate_mode="query_distance",
                                attention_blocked_implementation=implementation, rngs=nnx.Rngs(0))
    rng = np.random.default_rng(5)
    x = rng.standard_normal((batch, time, hidden))
    gate = (rng.uniform(-1, 1, (heads, hidden)) / math.sqrt(hidden),
            np.log(rng.uniform(0.5, 4.0, (heads, 1))), rng.uniform(-4.0, -2.0, (heads, 1)))
    cotangent = rng.standard_normal((batch, time, hidden))

    def production(x, w, a, b):
        return module(x, attention_parameters=dict(zip(GATE, (w, a, b))))

    with jax.default_matmul_precision("highest"):
        actual = _vjp(production, tuple(jnp.asarray(value, jnp.float32) for value in (x, *gate)),
                      jnp.asarray(cotangent, jnp.float32))
    assert calls and set(calls) == {implementation}
    weights = {name: np.asarray(getattr(module, f"{name}_proj").kernel[...], np.float64) for name in "qkvo"}

    def oracle(x, w, a, b):
        q, k, v = ((x @ weights[name]).reshape(batch, time, heads, width) for name in "qkv")
        log_alpha = -jnp.exp(a[:, 0]) * jnp.logaddexp(0.0, jnp.einsum("btd,hd->bth", x, w) + b[:, 0])
        attention = _dense(q, k, v, log_alpha, normalized=normalized)
        return attention.reshape(batch, time, heads * width) @ weights["o"]

    with jax.enable_x64():
        expected = _vjp(oracle, tuple(jnp.asarray(value, jnp.float64) for value in (x, *gate)),
                        jnp.asarray(cotangent))
    _compare(actual, jax.tree.map(np.asarray, expected), 4e-5, names=("output", "x", *GATE))


# --- Invalid gates, extreme gates, and the query-local rule ------------------------------------------

@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf"), 0.01))
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_invalid_gate_poisons_only_its_query(kernel, implementation, bad):
    """A nonfinite or positive raw log alpha_t poisons row t alone; every other row is unchanged bitwise."""
    arrays, _ = _inputs(37)
    q, k, v, log_alpha = (jnp.asarray(x, jnp.float32) for x in arrays)
    attend = jax.jit(_production(implementation, normalized=kernel == "softmax"))
    clean = np.asarray(attend(q, k, v, log_alpha))
    output = np.asarray(attend(q, k, v, log_alpha.at[0, 20, 1].set(bad)))
    invalid = ~np.isfinite(output).all(axis=-1)
    expected = np.zeros(invalid.shape, bool)
    expected[0, 20, 1] = True
    np.testing.assert_array_equal(invalid, expected)
    np.testing.assert_array_equal(output[~expected], clean[~expected])


@pytest.mark.parametrize("value", (0.0, -1000.0, -1e30))
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_open_and_closed_gates_stay_finite(kernel, implementation, value):
    """An open gate (log alpha = 0) is the ungated kernel; closed gates leave each query its own key,
    with finite outputs and gradients and no overflow of (t - i) log alpha_t."""
    arrays, cotangent = _inputs(37)
    q, k, v, _ = (jnp.asarray(x, jnp.float32) for x in arrays)
    log_alpha = jnp.full(arrays[3].shape, value, jnp.float32)
    normalized = kernel == "softmax"
    production = _production(implementation, normalized=normalized)
    with jax.default_matmul_precision("highest"):
        output, gradients = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            q, k, v, log_alpha)
        plain = blocked_dot_product_attention(q, k, v, attention_normalization=normalized, query_block_size=8)
    for array in (output, *gradients):
        assert np.isfinite(np.asarray(array)).all()
    if value == 0:
        np.testing.assert_allclose(output, plain, rtol=2e-6, atol=2e-6)
    else:
        # Every earlier key underflows: softmax returns v_t, the exponential exp(s_tt) v_t.
        own = jnp.exp(jnp.einsum("bthd,bthd->bth", q, k) / math.sqrt(q.shape[-1]))[..., None]
        np.testing.assert_allclose(output, v if normalized else own * v, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(gradients[3], np.zeros_like(log_alpha))


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_varying_gate_is_query_local_unlike_the_cumulative_decay(kernel):
    """Changing log alpha_j changes row j alone, where the cumulative decay changes every later row;
    a row's output depends on its own gate only."""
    arrays, _ = _inputs(19)
    q, k, v, log_alpha = (jnp.asarray(x, jnp.float32) for x in arrays)
    normalized = kernel == "softmax"
    distance = _production("scan", normalized=normalized, block=4)
    cumulative = lambda a: blocked_dot_product_attention(q, k, v, attention_normalization=normalized,
                                                         query_log_decay=jnp.cumsum(a, axis=1), query_block_size=4)
    changed = log_alpha.at[0, 6, 0].add(-0.5)
    rows = np.abs(np.asarray(distance(q, k, v, changed) - distance(q, k, v, log_alpha))).max(axis=-1) > 0
    expected = np.zeros(rows.shape, bool)
    expected[0, 6, 0] = True
    np.testing.assert_array_equal(rows, expected)
    later = np.abs(np.asarray(cumulative(changed) - cumulative(log_alpha))).max(axis=-1)[0, :, 0] > 0
    np.testing.assert_array_equal(later, np.arange(19) >= 6)
    gradient = jax.grad(lambda a: jnp.sum(distance(q, k, v, a)[0, 12, 0]))(log_alpha)
    assert gradient[0, 12, 0] != 0
    gradient = np.asarray(gradient).copy()
    gradient[0, 12, 0] = 0
    np.testing.assert_array_equal(gradient, 0)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_a_constant_rate_is_the_cumulative_alibi_decay(kernel, implementation):
    """With one log alpha per head at ALiBi's slopes, -exp(A_log) at every token (constant_log_decay's
    formula, in the operands' dtype), the query distance gives the cumulative constant decay: the
    output and the gradients of Q, K, V, and the log rate agree with the cumulative mode to rounding
    in FP32, and in exact arithmetic."""
    arrays, cotangent = _inputs(37)
    q, k, v, _ = arrays
    normalized = kernel == "softmax"
    rate = np.asarray(alibi_log_rate(q.shape[2]), np.float64)
    raw32 = jnp.broadcast_to(-jnp.exp(jnp.asarray(rate[:, 0], jnp.float32)), q.shape[:3])
    np.testing.assert_array_equal(raw32, constant_log_decay(jnp.zeros(q.shape[:2] + (1,)), alibi_log_rate(q.shape[2])))

    def run(mode, compute):
        attend = (partial(strip_dot_product_attention, strip_size=16) if implementation == "strips" else
                  partial(reference._blocked_scan, compute=compute, query_block_size=8))

        def function(q, k, v, log_rate):
            raw = jnp.broadcast_to(-jnp.exp(log_rate[:, 0]), q.shape[:3])
            decay = jnp.cumsum(raw, axis=1) if mode == "cumulative" else raw
            return attend(q, k, v, attention_normalization=normalized, query_log_decay=decay,
                          query_decay_mode=mode)

        dtype = jnp.float64 if compute == jnp.float64 else jnp.float32
        operands = tuple(jnp.asarray(x, dtype) for x in (q, k, v, rate))
        return jax.tree.map(np.asarray, jax.jit(lambda *x: _vjp(function, x, jnp.asarray(cotangent, dtype)))(
            *operands))

    with jax.default_matmul_precision("highest"):
        distance, cumulative = run("query_distance", jnp.float32), run("cumulative", jnp.float32)
    _compare(distance, cumulative, 2e-6, names=("output", "q", "k", "v", "log_rate"))
    with jax.enable_x64():
        distance, cumulative = run("query_distance", jnp.float64), run("cumulative", jnp.float64)
    _compare(distance, cumulative, 1e-12, names=("output", "q", "k", "v", "log_rate"))


# --- Recipes, parameters, and models ------------------------------------------------------------------

@pytest.mark.parametrize("kernel", KERNELS)
def test_accepted_recipes_and_the_routing(kernel):
    """RoPE or no positional encoding, the scan or the strips, on the blocked backend; the module
    routes the raw log alpha (not its cumulative sum) with the query-distance mode."""
    for encoding in ("none", "rope"):
        for implementation in IMPLEMENTATIONS:
            config = _config(kernel, positional_encoding=encoding, attention_blocked_implementation=implementation)
            module = attention_from_config(config.model, rngs=nnx.Rngs(0))
            assert module.attention_forget_gate_mode == "query_distance"


PATH = {"positional_encoding": "path", "qk_head_dim": 16, "value_head_dim": 16,
        "attention_blocked_implementation": "scan"}
INVALID = (
    ({"attention_forget_gate_rate_activation": "sigmoid"}, "softplus gate with query conditioning"),
    ({"attention_forget_gate_per_order": True}, "Per-order gates"),
    ({"attention_forget_gate_form": "constant"}, "constant forget gate form"),
    ({"attention_forget_gate_form": "sigmoid"}, "sigmoid forget gate form"),
    (PATH, "query-distance decay mode requires RoPE"),
    ({**PATH, "attention_backend": "fused"}, "query-distance decay mode requires RoPE"),
    ({"attention_degree": 2}, "forget gate"),
    ({"attention_backend": "fused"}, "forget gate"),
    ({"attention_backend": "mosaic"}, None),
    ({"attention_backend": "auto"}, "forget gate"),
    ({"attention_backend": "cudnn"}, None),
    ({"attention_forget_gate": False}, "requires model.attention_forget_gate"),
)


@pytest.mark.parametrize("change,message", INVALID,
                         ids=[",".join(f"{k}={v}" for k, v in c.items()) for c, _ in INVALID])
@pytest.mark.parametrize("kernel", KERNELS)
def test_unqualified_recipes_are_rejected(kernel, change, message):
    """Bounded rates, per-order gates, other gate forms, PaTH, Taylor kernels, and other backends; the
    module rejects them too when configuration validation is bypassed."""
    with pytest.raises(ValueError, match=message):
        _config(kernel, **change)
    model = _config(kernel).model
    for name, value in change.items():
        setattr(model, name, value)  # bypass config validation
    if change == {"attention_forget_gate_per_order": True}:
        # The module learns per-order gates from their parameter shapes; the operation refuses them.
        q = jnp.zeros((1, 4, 2, 4))
        with pytest.raises(ValueError, match="Per-order decay requires query-coefficient polynomial attention"):
            attention_core(q, q, q, attention_normalization=kernel == "softmax",
                           query_log_decay=jnp.zeros((1, 4, 2, 3)), query_decay_mode="query_distance")
        return
    with pytest.raises(ValueError):
        attention_from_config(model, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_operation_rejects_other_backends_and_malformed_decay(kernel):
    """Endpoint operands are test_exponential_query_key.py's."""
    q = jnp.zeros((1, 4, 2, 4))
    normalized = kernel == "softmax"
    for backend in ("fused", "mosaic", "cudnn"):
        with pytest.raises(ValueError, match="blocked backend"):
            attention_core(q, q, q, backend=backend, attention_normalization=normalized,
                           query_log_decay=jnp.zeros((1, 4, 2)), query_decay_mode="query_distance")
    with pytest.raises(ValueError, match="requires query_log_decay"):
        blocked_dot_product_attention(q, q, q, attention_normalization=normalized, query_decay_mode="query_distance")
    with pytest.raises(ValueError, match="log decay"):
        blocked_dot_product_attention(q, q, q, attention_normalization=normalized,
                                      query_log_decay=jnp.zeros((1, 4, 3)), query_decay_mode="query_distance")
    with pytest.raises(ValueError, match="requires query coefficients alone, softmax, or the unnormalized"):
        blocked_dot_product_attention(q, q, q, attention_degree=2, query_log_decay=jnp.zeros((1, 4, 2)),
                                      query_decay_mode="query_distance")


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_parameters_are_cumulative_fox_s(kernel):
    """The same parameter names, shapes, declared initializations, and initial values as cumulative FoX,
    and the same count in the Transformer++ shape (15 layers, width 896, 14 heads)."""
    distance, cumulative = _config(kernel), _config(kernel, attention_forget_gate_mode="cumulative")
    assert attention_parameter_specs(distance.model) == attention_parameter_specs(cumulative.model)
    flat = {mode: _flat(create_state(config, jax.random.PRNGKey(7)).params)
            for mode, config in (("distance", distance), ("cumulative", cumulative))}
    assert flat["distance"].keys() == flat["cumulative"].keys()
    assert {("forget_gate_kernel",), ("forget_gate_log_rate",), ("forget_gate_dt_bias",)} <= flat["distance"].keys()
    for path, value in flat["cumulative"].items():
        np.testing.assert_array_equal(flat["distance"][path], value, err_msg=str(path))
    shape = dict(vocab_size=32000, max_seq_len=2048, num_layers=15, hidden_size=896, num_heads=14, num_kv_heads=14,
                 qk_head_dim=64, value_head_dim=64, intermediate_size=2304)
    production = {mode: model_parameter_count(_config(kernel, **shape, attention_forget_gate_mode=mode).model)
                  for mode in ("query_distance", "cumulative")}
    ungated = model_parameter_count(_config(kernel, **shape, attention_forget_gate=False,
                                            attention_forget_gate_mode="cumulative").model)
    assert production["query_distance"] == production["cumulative"] == ungated + (896 + 2) * 15 * 14


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_token_independent_gate_gives_the_cumulative_model(kernel):
    """With w_alpha = 0 every token's log alpha is -exp(A_log) softplus(dt_bias): the query-distance
    model equals cumulative FoX with the same parameters, its logits and the gradient of every parameter
    but w_alpha (whose token sensitivities differ by design) agreeing to rounding."""
    configs = {mode: _config(kernel, attention_forget_gate_mode=mode) for mode in ("query_distance", "cumulative")}
    batch = jnp.asarray(np.random.default_rng(2).integers(0, 16, (2, 9)), jnp.int32)
    logits, gradients = {}, {}
    for mode, config in configs.items():
        state = create_state(config, jax.random.PRNGKey(3))
        state.params["forget_gate_kernel"][...] = jnp.zeros_like(state.params["forget_gate_kernel"][...])
        logits[mode] = np.asarray(nnx.merge(state.graphdef, state.params)(batch[:, :-1], deterministic=True))
        gradients[mode] = _flat(loop.loss_and_grads(state, batch, jax.random.PRNGKey(0))[1])
    np.testing.assert_allclose(logits["query_distance"], logits["cumulative"], rtol=1e-5, atol=1e-6)
    assert gradients["query_distance"].keys() == gradients["cumulative"].keys()
    for path, expected in gradients["cumulative"].items():
        if path == ("forget_gate_kernel",):
            continue
        observed = gradients["query_distance"][path]
        assert np.linalg.norm(observed - expected) <= 2e-5 * np.linalg.norm(expected) + 1e-9, path
    for name in ("forget_gate_log_rate", "forget_gate_dt_bias"):
        assert np.all(np.abs(gradients["query_distance"][(name,)]) > 0), name


@pytest.mark.parametrize("kernel", KERNELS)
def test_training_updates_every_gate_parameter(kernel):
    config = _config(kernel)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(18).reshape(2, 9) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    second, _, _ = step(first, jax.random.PRNGKey(2))
    assert np.isfinite(loss)
    gradients = _flat(loop.loss_and_grads(initial, batch, jax.random.PRNGKey(0))[1])
    for name in GATE:
        # Every layer and head of every gate parameter receives a gradient and moves.
        gradient = gradients[(name,)].reshape(*gradients[(name,)].shape[:2], -1)
        assert np.all(np.isfinite(gradient)) and np.all(np.linalg.norm(gradient, axis=-1) > 0), name
        assert np.linalg.norm(np.asarray(second.params[name][...] - initial.params[name][...])) > 0, name


@pytest.mark.parametrize("kernel", KERNELS)
def test_identity_and_diagnostics(kernel):
    config = _config(kernel)
    identity = attention_identity(config)
    assert identity["attention_forget_gate"] is True and identity["attention_forget_gate_mode"] == "query_distance"
    assert identity["attention_kernel"] == ("softmax" if kernel == "softmax" else "exp_unnormalized")
    assert identity["attention_formula"].startswith(
        ("causal_softmax" if kernel == "softmax" else "causal_exp")
        + "(QK^T / sqrt(d_k) + D) V, D_ti = (t - i) log alpha_t, log alpha_t = -exp(A_log) softplus(")
    assert "sum_(j=i+1..t)" not in identity["attention_formula"]
    assert identity["forget_gate_parameter_count"] == (8 + 2) * 2 * 2
    assert identity["attention_implementation"] == "blocked_causal_strips"
    cumulative = attention_identity(_config(kernel, attention_forget_gate_mode="cumulative"))
    assert "sum_(j=i+1..t) log alpha_j" in cumulative["attention_formula"]
    assert "forget_gate" in tracking.run_tags(config, {})
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    gate = report["coefficients"]["forget_gate"]
    assert gate["finite"] and np.asarray(gate["rate_exp_A_log"]).shape == (2, 2)
    mean = np.asarray(gate["log_alpha_sample"]["mean"])
    assert mean.shape == (2, 2) and np.all(np.asarray(gate["log_alpha_sample"]["max"]) <= 0)
    np.testing.assert_allclose(np.asarray(gate["mean_half_life_tokens"]), np.log(0.5) / mean, rtol=1e-12)


@pytest.mark.parametrize("kernel", KERNELS)
def test_acceptance_selects_these_checks_on_cpu_and_gpu(kernel):
    """Query-distance recipes run this module on CPU and its production-context cases on the GPU, with
    the forget-gate and causal-strips checks; cumulative FoX recipes do not select it."""
    config = _config(kernel)
    spec = acceptance.attention_spec(config, entry_module="tests")
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
    for paths in (cpu, gpu):
        assert acceptance.EXPONENTIAL_QUERY_DISTANCE_TESTS in paths
        assert acceptance.FORGET_GATE_ATTENTION_TESTS in paths
        assert acceptance.QUERY_DISTANCE_DECAY_TESTS not in paths
    assert acceptance.BLOCKED_STRIPS_TESTS in cpu
    cumulative = config_to_dict(_config(kernel, attention_forget_gate_mode="cumulative"))
    for flag in (True, False):
        paths = acceptance.test_paths(spec, cumulative, resolved_backend="blocked", cpu=flag)
        assert acceptance.EXPONENTIAL_QUERY_DISTANCE_TESTS not in paths
    gpu_cases = [name for name, test in globals().items() if name.startswith("test_")
                 and any(mark.name == "gpu" for mark in getattr(test, "pytestmark", ()))]
    assert sorted(gpu_cases) == ["test_gpu_strips_query_distance_matches_float64_at_the_production_context"]


@pytest.mark.parametrize("kernel", KERNELS)
def test_acceptance_initialization_is_the_softmax_backbone(kernel):
    facts = acceptance.matches_softmax_backbone_initialization(_config(kernel))
    assert set(facts["mechanism_parameters"]) >= set(GATE)
    assert facts["mechanism_parameters"]["forget_gate_kernel"]["weight_decay"] is True
    assert facts["mechanism_parameters"]["forget_gate_log_rate"]["weight_decay"] is False


# --- GPU qualification at the production context ------------------------------------------------------

def _production_context(kernel, *, time=2048, heads=14, width=64, seed=23):
    """Observed BF16 output and dQ, dK, dV and FP32 d log alpha through the strips, the FP64 oracle of
    the same inputs, and each BF16 quantity's rounding floor."""
    rng = np.random.default_rng(seed)
    q, k, v, cotangent = (jnp.asarray(rng.standard_normal((1, time, heads, width)), jnp.bfloat16) for _ in range(4))
    # Per-head rates from 3e-4 (half-life ~2,300 tokens) to 2 (under a token), varying by token.
    rates = np.exp(rng.uniform(math.log(3e-4), math.log(2.0), heads))
    log_alpha = jnp.asarray(-rates * np.exp(rng.uniform(-0.5, 0.5, (1, time, heads))), jnp.float32)
    normalized = kernel == "softmax"
    attend = partial(attention_core, backend="blocked", attention_normalization=normalized,
                     query_decay_mode="query_distance", blocked_implementation="strips")

    def value_and_gradients(q, k, v, log_alpha, cotangent):
        output, pullback = jax.vjp(lambda q, k, v, a: attend(q, k, v, query_log_decay=a), q, k, v, log_alpha)
        return (output, *pullback(cotangent))

    observed = jax.jit(value_and_gradients)(q, k, v, log_alpha, cotangent)
    output, gradients = _numpy_vjp(*(x.astype(jnp.float32) for x in (q, k, v)), log_alpha,
                                   cotangent.astype(jnp.float32), normalized=normalized)
    expected = (output, *gradients)
    floors = [_relative_l2(np.asarray(jnp.asarray(want, jnp.bfloat16).astype(jnp.float32)), want)
              for want in expected[:4]]
    return observed, expected, floors


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", KERNELS)
def test_gpu_strips_query_distance_matches_float64_at_the_production_context(kernel):
    """BF16 Q/K/V at the production context, head count, and head width through the causal strips, with
    raw log alpha whose half-lives range from under one token to the context: the output and dQ, dK, dV
    each within 1.2x its BF16 rounding floor (the host FP64 oracle rounded to BF16), and the FP32 log-decay
    gradient within 1e-4, against host FP64 of the same inputs."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    observed, expected, floors = _production_context(kernel)
    for name, got, want, floor in zip(("output", "dq", "dk", "dv"), observed[:4], expected[:4], floors, strict=True):
        assert got.dtype == jnp.bfloat16, name
        got = np.asarray(got.astype(jnp.float32), np.float64)
        assert np.isfinite(got).all(), name
        error = _relative_l2(got, want)
        assert error < 1.2 * floor, f"{name}: relative L2 error {error:.3e}, BF16 floor {floor:.3e}"
    gradient = observed[4]
    assert gradient.dtype == jnp.float32 and np.isfinite(np.asarray(gradient)).all()
    error = _relative_l2(gradient, expected[4])
    assert error < 1e-4, f"d log alpha: relative L2 error {error:.3e}"
