"""Query-key (endpoint) query-distance gates on softmax and the unnormalized exponential.

Each pair's rate reads both of its tokens, log alpha_ti = -exp(A_h) softplus(wq_h.x_t + wk_h.x_i + b_h)
(Gated DeltaNet's gate with a key projection inside the softplus), and every score gains
(t - i) log alpha_ti. Softmax normalizes the decayed logits (its row maximum is taken over them); the
exponential exponentiates them. The attention core receives the token projections u_t = wq.x_t + b and
v_i = wk.x_i and the log rates A [H, 1]; pair rates are evaluated inside the score blocks. A nonfinite
projection or log rate poisons the rows of the causal pairs it enters. w_key starts at zero, so a model
starts exactly as its query-gate counterpart (test_exponential_query_distance.py). The mechanism takes
the input-form softplus gate, RoPE or no positional encoding, on the blocked backend (scan or strips).
"""
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import forget_gate
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import (
    blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import alibi_log_rate, constant_log_decay
from scaffolds.transformer.attention.multihead import MultiHeadAttention, attention_from_config
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_endpoint_decay import _config as _polynomial_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample,
)
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state

KERNELS = ("softmax", "exp")
IMPLEMENTATIONS = ("scan", "strips")
GATE = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias", "forget_gate_key_kernel")
OPERANDS = ("output", "q", "k", "v", "query_input", "key_input", "log_rate")
GATE_OPERANDS = ("query_input", "key_input", "log_rate")
# (time, scan block, strip): ragged 8-query blocks and 16-query strips of 37 tokens, one block and
# one strip longer than the sequence, and a single query.
SHAPES = ((37, 8, 16), (5, 8, 8), (1, 8, 8))
# softplus(SOFTPLUS_ONE) = 1: a constant pair input whose rate is exp(A_h) itself.
SOFTPLUS_ONE = math.log(math.expm1(1.0))


def _config(kernel="softmax", **changes):
    """Tiny query-key recipes as allocated: no positional encoding, canonical QK RMSNorm, causal
    strips, and per-head output RMSNorm for exp."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=4, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
                 attention_blocked_implementation="strips", attention_normalization=kernel == "softmax",
                 attention_forget_gate=True, attention_forget_gate_mode="query_distance",
                 attention_forget_gate_conditioning="query_key", positional_encoding="none",
                 attention_output_scaling="none" if kernel == "softmax" else "head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _log_alpha(query, key, log_rate):
    """Every pair's log alpha_ti = -exp(A_h) softplus(u_t + v_i), [batch, heads, query, key]."""
    pair = jnp.swapaxes(query, 1, 2)[..., :, None] + jnp.swapaxes(key, 1, 2)[..., None, :]
    return -jnp.exp(log_rate[:, 0])[None, :, None, None] * jnp.logaddexp(0.0, pair)


def _dense(q, k, v, query, key, log_rate, *, normalized):
    """All pairs with explicit distances: softmax(s + B) V, or exp(s + B) V, B_ti = (t - i) log alpha_ti."""
    time = q.shape[1]
    distance = jnp.arange(time)[:, None] - jnp.arange(time)[None, :]
    causal = distance >= 0
    logits = (jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
              + _log_alpha(query, key, log_rate) * jnp.where(causal, distance, 0))
    if normalized:
        weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1)
    else:
        weights = jnp.where(causal, jnp.exp(jnp.where(causal, logits, 0.0)), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def _numpy_vjp(q, k, v, query, key, log_rate, cotangent, *, normalized):
    """Host FP64 forward and analytic VJP (dQ, dK, dV, du, dv, dA), one head at a time, with no JAX
    compilation: the oracle of the production-context cases."""
    q, k, v, query, key, log_rate, cotangent = (np.asarray(x, np.float64)
                                                for x in (q, k, v, query, key, log_rate, cotangent))
    batch, time, heads, width = q.shape
    output, dq, dk, dv = np.zeros_like(v), np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    dquery, dkey, drate = np.zeros_like(query), np.zeros_like(key), np.zeros_like(log_rate)
    distance = np.arange(time)[:, None] - np.arange(time)[None, :]
    causal = distance >= 0
    distance = np.where(causal, distance, 0)
    for b in range(batch):
        for h in range(heads):
            pair = query[b, :, h, None] + key[b, None, :, h]
            rate = math.exp(log_rate[h, 0])
            decay = -rate * np.logaddexp(0.0, pair) * distance
            logits = q[b, :, h] @ k[b, :, h].T / math.sqrt(width) + decay
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
            # d decay / d pair = -(t - i) exp(A) sigmoid(pair); d decay / dA = decay.
            dpair = -dlogits * distance * rate * np.exp(-np.logaddexp(0.0, -pair))
            dquery[b, :, h] = dpair.sum(axis=-1)
            dkey[b, :, h] = dpair.sum(axis=0)
            drate[h, 0] += np.sum(dlogits * decay)
    return output, (dq, dk, dv, dquery, dkey, drate)


def _inputs(time, *, batch=2, heads=3, width=8, seed=0):
    """Q, K, V, gate inputs, per-head log rates, and a cotangent: pair rates from about 1e-4 (half-lives
    in the thousands of tokens) to about 4 (under a token), varying with both endpoints."""
    rng = np.random.default_rng(seed + 100 * time)
    q, k, v, cotangent = (rng.standard_normal((batch, time, heads, width)) for _ in range(4))
    query = rng.normal(-1.0, 1.0, (batch, time, heads))
    key = rng.normal(0.0, 0.7, (batch, time, heads))
    log_rate = np.log(np.geomspace(0.003, 2.0, heads)).reshape(heads, 1)
    return (q, k, v, query, key, log_rate), cotangent


def _production(implementation, *, normalized, block=8, strip=16, compute=None):
    """The blocked backend's endpoint call: the public scan or strips, or the scan's own code in
    ``compute`` (FP64 for exact-arithmetic checks); the strips compute in the inputs' dtype."""
    options = dict(attention_normalization=normalized, query_decay_mode="query_distance")
    if implementation == "strips":
        attend = partial(strip_dot_product_attention, strip_size=strip, **options)
    elif compute is not None:
        attend = partial(reference._blocked_scan, compute=compute, query_block_size=block, **options)
    else:
        attend = partial(blocked_dot_product_attention, query_block_size=block, **options)
    return lambda q, k, v, query, key, log_rate: attend(q, k, v, query_gate_input=query, key_gate_input=key,
                                                        gate_log_rate=log_rate)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _compare(actual, expected, tolerance, names=OPERANDS, zeros=()):
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
    """The production-context oracle: output and analytic dQ, dK, dV, du, dv, dA against FP64 autodiff."""
    arrays, cotangent = _inputs(time)
    normalized = kernel == "softmax"
    observed = _numpy_vjp(*arrays, cotangent, normalized=normalized)
    with jax.enable_x64():
        expected = _vjp(partial(_dense, normalized=normalized), tuple(jnp.asarray(x) for x in arrays),
                        jnp.asarray(cotangent))
    for name, got, want in zip(OPERANDS, jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True):
        want = np.asarray(want)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * max(np.abs(want).max(), 1.0), err_msg=name)
    if time == 1:
        # Distance zero: the decay and the gate derivatives are exact.
        for gradient in observed[1][3:]:
            np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


@pytest.mark.parametrize("time,block,strip", SHAPES)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_float64_blocks_and_strips_match_the_all_pairs_reference(kernel, implementation, time, block, strip):
    """In exact arithmetic (the scan's code and the strips in FP64), the output and the gradients of Q, K,
    V, both gate inputs, and the log rate equal the explicit all-pairs reference to rounding."""
    arrays, cotangent = _inputs(time)
    normalized = kernel == "softmax"
    with jax.enable_x64():
        operands = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        expected = _vjp(partial(_dense, normalized=normalized), operands, jnp.asarray(cotangent))
        production = _production(implementation, normalized=normalized, block=block, strip=strip,
                                 compute=jnp.float64)
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent)))(*operands)
    for name, got, want in zip(OPERANDS, jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
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
    zeros = () if time > 1 else (*GATE_OPERANDS, "q", "k") if normalized else GATE_OPERANDS
    _compare(actual, expected, 3e-5, zeros=zeros)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_gate_projection_chain_rule_matches_float64(kernel, implementation):
    """u_t = x_t w_query + dt_bias and v_i = x_i w_key from forget_gate.project_gate_inputs, through either
    implementation: the output and the gradients of Q, K, V, x, and every gate parameter (w_query, w_key,
    A_log, dt_bias) against an independent FP64 formula; every head of every gate parameter learns."""
    rng = np.random.default_rng(43)
    (q, k, v, *_), cotangent = _inputs(23, heads=2)
    x = rng.standard_normal((2, 23, 7))
    query_w, key_w = (rng.uniform(-1, 1, (2, 7)) / math.sqrt(7) for _ in range(2))
    log_rate, dt_bias = np.log([[0.7], [3.0]]), np.asarray([[-2.5], [-4.0]])
    operands = (q, k, v, x, query_w, key_w, log_rate, dt_bias)
    normalized = kernel == "softmax"

    def oracle(q, k, v, x, wq, wk, a, b):
        query = jnp.einsum("btd,hd->bth", x, wq) + b[:, 0]
        return _dense(q, k, v, query, jnp.einsum("btd,hd->bth", x, wk), a, normalized=normalized)

    attend = _production(implementation, normalized=normalized)

    def production(q, k, v, x, wq, wk, a, b):
        return attend(q, k, v, *forget_gate.project_gate_inputs(x, wq, b, wk), a)

    with jax.enable_x64():
        expected = _vjp(oracle, tuple(jnp.asarray(value, jnp.float64) for value in operands), jnp.asarray(cotangent))
        expected = jax.tree.map(np.asarray, expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(production, a, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(value, jnp.float32) for value in operands))
    names = ("output", "q", "k", "v", "x", "query_kernel", "key_kernel", "log_rate", "dt_bias")
    _compare(actual, expected, 4e-5, names=names)
    for name, gradient in zip(names[5:], actual[1][4:]):
        assert np.all(np.linalg.norm(np.asarray(gradient), axis=-1) > 0), name  # every head


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_attention_module_with_its_gate_parameters_matches_float64(kernel, implementation, monkeypatch):
    """The attention module (projections, no positional encoding, the gate's token projections routed to
    the blocked backend, output projection) against FP64 built from its weights: the output and the
    gradients of x and of every gate parameter, through the recipe's implementation (counted) in 8-query
    blocks or strips."""
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
                                attention_forget_gate_conditioning="query_key",
                                attention_blocked_implementation=implementation, rngs=nnx.Rngs(0))
    rng = np.random.default_rng(5)
    x = rng.standard_normal((batch, time, hidden))
    gate = (rng.uniform(-1, 1, (heads, hidden)) / math.sqrt(hidden), np.log(rng.uniform(0.5, 4.0, (heads, 1))),
            rng.uniform(-4.0, -2.0, (heads, 1)), rng.uniform(-1, 1, (heads, hidden)) / math.sqrt(hidden))
    cotangent = rng.standard_normal((batch, time, hidden))

    def production(x, *gate):
        return module(x, attention_parameters=dict(zip(GATE, gate)))

    with jax.default_matmul_precision("highest"):
        actual = _vjp(production, tuple(jnp.asarray(value, jnp.float32) for value in (x, *gate)),
                      jnp.asarray(cotangent, jnp.float32))
    assert calls and set(calls) == {implementation}
    weights = {name: np.asarray(getattr(module, f"{name}_proj").kernel[...], np.float64) for name in "qkvo"}

    def oracle(x, w, a, b, w_key):
        q, k, v = ((x @ weights[name]).reshape(batch, time, heads, width) for name in "qkv")
        query = jnp.einsum("btd,hd->bth", x, w) + b[:, 0]
        attention = _dense(q, k, v, query, jnp.einsum("btd,hd->bth", x, w_key), a, normalized=normalized)
        return attention.reshape(batch, time, heads * width) @ weights["o"]

    with jax.enable_x64():
        expected = _vjp(oracle, tuple(jnp.asarray(value, jnp.float64) for value in (x, *gate)),
                        jnp.asarray(cotangent))
    _compare(actual, jax.tree.map(np.asarray, expected), 4e-5, names=("output", "x", *GATE))


# --- Endpoints, invalid gates, and extreme gates -----------------------------------------------------

def test_the_pair_rate_reads_only_its_endpoints_and_keeps_the_unit_diagonal():
    """With zero scores and V the identity, the exponential returns each causal weight exp((t-i) log alpha_ti):
    exactly one on the diagonal and zero above it. The weight from key 0 to query 8 ignores every intervening
    token's gate inputs, and its gradient reaches u_8 and v_0 alone."""
    q = k = jnp.zeros((1, 9, 1, 2))
    v = jnp.eye(9)[None, :, None, :]
    query = jnp.linspace(-4.0, -1.0, 9)[None, :, None]
    key = jnp.linspace(-1.0, 1.0, 9)[None, :, None]
    rate = jnp.array([[0.3]])
    attend = _production("scan", normalized=False, block=4)
    output = attend(q, k, v, query, key, rate)[0, :, 0, :]
    np.testing.assert_allclose(output, _dense(q, k, v, query, key, rate, normalized=False)[0, :, 0],
                               rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(np.diag(output), np.ones(9))
    np.testing.assert_array_equal(np.triu(output, 1), np.zeros((9, 9)))
    changed = attend(q, k, v, query.at[0, 3, 0].set(50.0), key.at[0, 3, 0].set(50.0), rate)
    np.testing.assert_array_equal(changed[0, 8, 0, 0], output[8, 0])
    dquery, dkey = jax.grad(lambda a, b: attend(q, k, v, a, b, rate)[0, 8, 0, 0], argnums=(0, 1))(query, key)
    assert dquery[0, 8, 0] < 0 and dkey[0, 0, 0] < 0
    np.testing.assert_array_equal(dquery[0, :8, 0], np.zeros(8))
    np.testing.assert_array_equal(dkey[0, 1:, 0], np.zeros(8))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_a_query_input_moves_its_row_and_a_key_input_the_later_rows(kernel, implementation):
    """Changing u_j changes row j alone; changing v_j changes the rows t > j alone (its own pair has distance
    zero), unlike the cumulative decay, where a gate changes every later row."""
    arrays, _ = _inputs(19)
    q, k, v, query, key, rate = (jnp.asarray(x, jnp.float32) for x in arrays)
    attend = jax.jit(_production(implementation, normalized=kernel == "softmax", block=4, strip=8))
    clean = np.asarray(attend(q, k, v, query, key, rate))

    def moved(output):
        return np.abs(np.asarray(output) - clean).max(axis=-1) > 0

    expected = np.zeros(clean.shape[:3], bool)
    expected[0, 6, 1] = True
    np.testing.assert_array_equal(moved(attend(q, k, v, query.at[0, 6, 1].add(-0.5), key, rate)), expected)
    expected[0, 6, 1], expected[0, 7:, 1] = False, True
    np.testing.assert_array_equal(moved(attend(q, k, v, query, key.at[0, 6, 1].add(-0.5), rate)), expected)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("operand", ("query", "key", "rate"))
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_invalid_gate_values_poison_only_the_rows_of_their_pairs(kernel, implementation, operand, bad):
    """A nonfinite query input u_t poisons row t; a nonfinite key input v_i the rows t >= i, its own
    diagonal pair included; a nonfinite log rate every row of its head. Every other row is unchanged
    bitwise."""
    arrays, _ = _inputs(37)
    q, k, v, query, key, rate = (jnp.asarray(x, jnp.float32) for x in arrays)
    attend = jax.jit(_production(implementation, normalized=kernel == "softmax"))
    clean = np.asarray(attend(q, k, v, query, key, rate))
    expected = np.zeros(clean.shape[:3], bool)
    if operand == "query":
        query = query.at[0, 20, 1].set(bad)
        expected[0, 20, 1] = True
    elif operand == "key":
        key = key.at[0, 20, 1].set(bad)
        expected[0, 20:, 1] = True
    else:
        rate = rate.at[1, 0].set(bad)
        expected[:, :, 1] = True
    output = np.asarray(attend(q, k, v, query, key, rate))
    np.testing.assert_array_equal(~np.isfinite(output).all(axis=-1), expected)
    np.testing.assert_array_equal(output[~expected], clean[~expected])


@pytest.mark.parametrize("value", (-1000.0, 1e6, 1e20))
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_open_and_closed_gates_stay_finite(kernel, implementation, value):
    """An open gate (softplus underflows to zero) is the ungated kernel; closed gates leave each query its
    own key, with no overflow of (t - i) log alpha_ti. Outputs and gradients stay finite, and every gate
    gradient is exactly zero."""
    arrays, cotangent = _inputs(37)
    q, k, v, query, key, rate = (jnp.asarray(x, jnp.float32) for x in arrays)
    query, key = jnp.full_like(query, value), jnp.zeros_like(key)
    normalized = kernel == "softmax"
    production = _production(implementation, normalized=normalized)
    with jax.default_matmul_precision("highest"):
        output, gradients = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            q, k, v, query, key, rate)
        plain = blocked_dot_product_attention(q, k, v, attention_normalization=normalized, query_block_size=8)
    for array in (output, *gradients):
        assert np.isfinite(np.asarray(array)).all()
    for gradient in gradients[3:]:
        np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
    if value < 0:
        np.testing.assert_allclose(output, plain, rtol=2e-6, atol=2e-6)
    else:
        # Every earlier key underflows: softmax returns v_t, the exponential exp(s_tt) v_t.
        own = jnp.exp(jnp.einsum("bthd,bthd->bth", q, k) / math.sqrt(q.shape[-1]))[..., None]
        np.testing.assert_allclose(output, v if normalized else own * v, rtol=2e-6, atol=2e-6)


# Raw scores far outside FP32's exp range (exp(88.8) overflows, exp(-104) underflows) where neither kernel
# exponentiates them: one head and three tokens, Q and K of width one so that s_ti = q_t k_i, given as
# (q, k, query input u, key input v, log rate A).
EXTREME_SCORES = {
    # Key 1 lies in query 0's future with raw score 200; every causal score is zero.
    "masked_future_key": ((20, 0, 0), (0, 10, 0), (-1.0, 0.5, 0.2), (0.3, -0.4, 0.1), math.log(0.5)),
    # The same masked score beside query 0's own score of -100.
    "masked_future_key_low_own_score": ((20, 0, 0), (-5, 10, 0), (-1.0, 0.5, 0.2), (0.3, -0.4, 0.1),
                                        math.log(0.5)),
    # Query 2's raw score to key 0 is 200, but its closed gate (rate 1,000) decays it to -1,800; its own
    # score is -100.
    "decayed_away_key": ((0, 0, 20), (10, 0, -5), (0, 0, 1000), (0, 0, 0), 0.0),
    # Every score 100 under an open gate: softmax alone, whose row shift keeps it finite.
    "large_scores_open_gate": ((10, 10, 10), (10, 10, 10), (-1000, -1000, -1000), (0, 0, 0), 0.0),
}


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel,scenario", [(kernel, scenario) for kernel in KERNELS for scenario in EXTREME_SCORES
                                             if kernel == "softmax" or scenario != "large_scores_open_gate"])
def test_extreme_raw_scores_never_reach_the_exponential(kernel, scenario, implementation):
    """Masked future keys never reach the exponential, and softmax shifts each row by its maximum over the
    decayed causal logits alone: with raw scores far outside FP32's exp range, through two-query blocks and
    strips, the output and the gradients of Q, K, V, both gate inputs, and the log rate are finite and equal
    the FP64 all-pairs reference to rounding. A masked logit of 200 that reached exp would be infinite, and
    its zero cotangent would turn every gradient but dV into NaN; a maximum over raw or masked scores would
    shift every causal weight of a row to zero, and no shift would overflow scores of 100."""
    q, k, query, key, log_rate = EXTREME_SCORES[scenario]
    rng = np.random.default_rng(7)
    v, cotangent = (rng.standard_normal((1, 3, 1, 2)) for _ in range(2))
    arrays = (np.reshape(q, (1, 3, 1, 1)), np.reshape(k, (1, 3, 1, 1)), v, np.reshape(query, (1, 3, 1)),
              np.reshape(key, (1, 3, 1)), np.full((1, 1), log_rate))
    normalized = kernel == "softmax"
    with jax.enable_x64():
        expected = _vjp(partial(_dense, normalized=normalized), tuple(jnp.asarray(x, jnp.float64) for x in arrays),
                        jnp.asarray(cotangent))
        expected = jax.tree.map(np.asarray, expected)
    production = _production(implementation, normalized=normalized, block=2, strip=2)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(x, jnp.float32) for x in arrays))
    for name, got, want in zip(OPERANDS, (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True):
        got = np.asarray(got, np.float64)
        assert np.isfinite(got).all(), name
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * max(np.abs(want).max(), 1.0), err_msg=name)
    if normalized and scenario == "decayed_away_key":
        # Query 2's decayed logits are -1,800, -1,000, and its own -100: the row is v_2 exactly.
        np.testing.assert_array_equal(actual[0][0, 2], jnp.asarray(v[0, 2], jnp.float32))


@pytest.mark.parametrize("kernel", KERNELS)
def test_diagnostics_measure_every_causal_pair(kernel):
    """The scan's gate statistics are those of log alpha_ti over every real causal pair (diagonal
    included, padded queries and masked keys excluded), per head."""
    arrays, _ = _inputs(19)
    operands = [jnp.asarray(x, jnp.float32) for x in arrays]
    _, diagnostics = blocked_dot_product_attention(
        *operands[:3], attention_normalization=kernel == "softmax", query_decay_mode="query_distance",
        query_gate_input=operands[3], key_gate_input=operands[4], gate_log_rate=operands[5],
        query_block_size=8, return_diagnostics=True)
    query, key, log_rate = arrays[3:]
    log_alpha = -np.exp(log_rate[:, 0]) * np.logaddexp(0.0, query[:, :, None, :] + key[:, None, :, :])
    values = log_alpha[:, np.arange(19)[:, None] >= np.arange(19)[None, :], :]
    for name, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        np.testing.assert_allclose(diagnostics[f"forget_gate_log_alpha_{name}"], function(values, axis=(0, 1)),
                                   rtol=3e-6, atol=1e-8)
    assert diagnostics["invalid_kernel_count"] == 0 and diagnostics["invalid_score_count"] == 0


# --- Equivalences ---------------------------------------------------------------------------------------

def _query_distance(implementation, *, normalized, compute=None):
    """The query-distance gate's call with log alpha_t = -exp(A) softplus(u_t), forget_gate.log_decay's
    arithmetic, as a function of (q, k, v, u, A)."""
    options = dict(attention_normalization=normalized, query_decay_mode="query_distance")
    attend = (partial(strip_dot_product_attention, strip_size=16, **options) if implementation == "strips" else
              partial(reference._blocked_scan, compute=compute, query_block_size=8, **options) if compute else
              partial(blocked_dot_product_attention, query_block_size=8, **options))
    return lambda q, k, v, query, log_rate: attend(
        q, k, v, query_log_decay=-jnp.exp(log_rate[:, 0]) * jax.nn.softplus(query))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_a_zero_key_input_is_the_query_distance_gate(kernel, implementation):
    """With v_i = 0 the pair rate is the query's own: the output and the gradients of Q, K, V, u, and A equal
    the query-distance gate's in exact arithmetic (FP64 within 1e-12) and to FP32 rounding (2e-6; XLA may
    fuse the two programs' decayed logits differently, so they need not agree bitwise)."""
    arrays, cotangent = _inputs(37)
    q, k, v, query, _, log_rate = arrays
    normalized = kernel == "softmax"

    def run(dtype, compute=None):
        endpoint = _production(implementation, normalized=normalized, compute=compute)
        distance = _query_distance(implementation, normalized=normalized, compute=compute)
        operands = tuple(jnp.asarray(x, dtype) for x in (q, k, v, query, log_rate))
        zero_key = jnp.zeros(query.shape, dtype)
        g = jnp.asarray(cotangent, dtype)
        first = jax.jit(lambda *x: _vjp(lambda q, k, v, u, a: endpoint(q, k, v, u, zero_key, a), x, g))(*operands)
        second = jax.jit(lambda *x: _vjp(distance, x, g))(*operands)
        return jax.tree.map(np.asarray, first), jax.tree.map(np.asarray, second)

    names = ("output", "q", "k", "v", "query_input", "log_rate")
    with jax.default_matmul_precision("highest"):
        endpoint, distance = run(jnp.float32)
    _compare(endpoint, distance, 2e-6, names=names)
    with jax.enable_x64():
        endpoint, distance = run(jnp.float64, compute=jnp.float64)
    _compare(endpoint, distance, 1e-12, names=names)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_constant_rates_are_the_cumulative_alibi_decay(kernel, implementation):
    """With every pair input at softplus^-1(1) and A at ALiBi's log slopes, each pair's rate is m_h: the
    output and the gradients of Q, K, V, and the log rate equal the cumulative constant decay's
    (constant_log_decay's ALiBi), within 5e-6 in FP32 and 1e-12 in exact arithmetic."""
    arrays, cotangent = _inputs(37)
    q, k, v = arrays[:3]
    normalized = kernel == "softmax"
    rate = np.asarray(alibi_log_rate(q.shape[2]), np.float64)
    query, key = np.full(q.shape[:3], SOFTPLUS_ONE - 0.25), np.full(q.shape[:3], 0.25)

    def run(mode, dtype):
        compute = jnp.float64 if dtype == jnp.float64 else None
        if mode == "endpoint":
            attend = _production(implementation, normalized=normalized, compute=compute)
            gates = tuple(jnp.asarray(x, dtype) for x in (query, key))

            def function(q, k, v, log_rate):
                return attend(q, k, v, *gates, log_rate)
        else:
            attend = (partial(strip_dot_product_attention, strip_size=16) if implementation == "strips" else
                      partial(reference._blocked_scan, compute=compute or jnp.float32, query_block_size=8))

            def function(q, k, v, log_rate):
                decay = jnp.cumsum(jnp.broadcast_to(-jnp.exp(log_rate[:, 0]), q.shape[:3]), axis=1)
                return attend(q, k, v, attention_normalization=normalized, query_log_decay=decay)

        operands = tuple(jnp.asarray(x, dtype) for x in (q, k, v, rate))
        return jax.tree.map(np.asarray, jax.jit(lambda *x: _vjp(function, x, jnp.asarray(cotangent, dtype)))(
            *operands))

    names = ("output", "q", "k", "v", "log_rate")
    with jax.default_matmul_precision("highest"):
        _compare(run("endpoint", jnp.float32), run("cumulative", jnp.float32), 5e-6, names=names)
    with jax.enable_x64():
        _compare(run("endpoint", jnp.float64), run("cumulative", jnp.float64), 1e-12, names=names)
    # The cumulative ALiBi decay here is the constant form's: L_t = sum_(j<=t) -m_h.
    np.testing.assert_array_equal(
        jnp.broadcast_to(-jnp.exp(jnp.asarray(rate[:, 0], jnp.float32)), q.shape[:3]),
        constant_log_decay(jnp.zeros(q.shape[:2] + (1,)), alibi_log_rate(q.shape[2])))


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_model_starts_exactly_as_its_query_gate_counterpart(kernel):
    """At initialization (w_key = 0) the model's parameters are the query-gate model's plus the zero key
    projection, its logits equal that model's bitwise, and every shared parameter's gradient agrees to
    rounding; every layer and head of the key projection already receives a gradient."""
    configs = {conditioning: _config(kernel, attention_forget_gate_conditioning=conditioning)
               for conditioning in ("query", "query_key")}
    batch = jnp.asarray(np.random.default_rng(2).integers(0, 16, (2, 9)), jnp.int32)
    parameters, logits, gradients = {}, {}, {}
    for conditioning, config in configs.items():
        state = create_state(config, jax.random.PRNGKey(3))
        parameters[conditioning] = _flat(state.params)
        logits[conditioning] = np.asarray(nnx.merge(state.graphdef, state.params)(batch[:, :-1], deterministic=True))
        gradients[conditioning] = _flat(loop.loss_and_grads(state, batch, jax.random.PRNGKey(0))[1])
    key = ("forget_gate_key_kernel",)
    assert parameters["query_key"].keys() - parameters["query"].keys() == {key}
    for path, value in parameters["query"].items():
        np.testing.assert_array_equal(parameters["query_key"][path], value, err_msg=str(path))
    np.testing.assert_array_equal(parameters["query_key"][key], np.zeros((2, 2, 8)))
    np.testing.assert_array_equal(logits["query_key"], logits["query"])
    for path, expected in gradients["query"].items():
        observed = gradients["query_key"][path]
        assert np.linalg.norm(observed - expected) <= 2e-5 * np.linalg.norm(expected) + 1e-9, path
    assert np.all(np.linalg.norm(gradients["query_key"][key], axis=-1) > 0)


# --- Recipes, parameters, and models ------------------------------------------------------------------

@pytest.mark.parametrize("kernel", KERNELS)
def test_accepted_recipes_and_the_routing(kernel):
    """RoPE or no positional encoding, the scan or the strips, on the blocked backend; the module passes the
    token projections (no log decay) with the query-distance mode."""
    for encoding in ("none", "rope"):
        for implementation in IMPLEMENTATIONS:
            config = _config(kernel, positional_encoding=encoding, attention_blocked_implementation=implementation)
            module = attention_from_config(config.model, rngs=nnx.Rngs(0))
            assert (module.attention_forget_gate_mode, module.attention_forget_gate_conditioning) == (
                "query_distance", "query_key")


PATH = {"positional_encoding": "path", "qk_head_dim": 16, "value_head_dim": 16,
        "attention_blocked_implementation": "scan"}
INVALID = (
    ({"attention_forget_gate_mode": "cumulative"}, "Endpoint gate variants require input-form query-distance"),
    ({"attention_forget_gate_rate_activation": "sigmoid"}, "Query-key decay is qualified only with softplus"),
    ({"attention_forget_gate_per_order": True}, "Per-order gates"),
    ({"attention_forget_gate_form": "constant"}, "constant forget gate form"),
    ({"attention_forget_gate_form": "sigmoid"}, "sigmoid forget gate form"),
    (PATH, "query-distance decay mode requires RoPE"),
    ({"attention_backend": "fused"}, "Endpoint gate variants"),
    ({"attention_backend": "auto"}, "Endpoint gate variants"),
    ({"attention_backend": "mosaic"}, None),
    ({"attention_backend": "cudnn"}, None),
    ({"attention_forget_gate": False, "attention_forget_gate_mode": "cumulative"}, "Endpoint gate variants"),
)


@pytest.mark.parametrize("change,message", INVALID,
                         ids=[",".join(f"{k}={v}" for k, v in c.items()) for c, _ in INVALID])
@pytest.mark.parametrize("kernel", KERNELS)
def test_unqualified_recipes_are_rejected(kernel, change, message):
    """The cumulative mode, bounded rates, per-order gates, other gate forms, PaTH, Taylor kernels, other
    backends, and conditioning without a gate; the module rejects them too when configuration validation
    is bypassed (per-order gates, which it learns from parameter shapes, at the operation)."""
    with pytest.raises(ValueError, match=message):
        _config(kernel, **change)
    model = _config(kernel).model
    for name, value in change.items():
        setattr(model, name, value)  # bypass config validation
    if change == {"attention_forget_gate_per_order": True}:
        q, gate = jnp.zeros((1, 4, 2, 4)), jnp.zeros((1, 4, 2, 3))
        with pytest.raises(ValueError, match="Per-order decay requires query-coefficient polynomial attention"):
            attention_core(q, q, q, attention_normalization=kernel == "softmax", query_gate_input=gate,
                           key_gate_input=gate, gate_log_rate=jnp.zeros((2, 3, 1)), query_decay_mode="query_distance")
        return
    with pytest.raises(ValueError):
        attention_from_config(model, rngs=nnx.Rngs(0))


def test_a_degree_composes_with_the_gate_only_unnormalized():
    """Normalized Taylor attention takes no gate; unnormalized fixed Taylor coefficients take this one on the
    blocked backend, per query (attention/pieces.py), in the configuration and in the module alike."""
    with pytest.raises(ValueError, match="forget gate applies to softmax"):
        _config("softmax", attention_degree=2)
    model = _config("softmax").model
    model.attention_degree = 2  # bypass config validation
    with pytest.raises(ValueError):
        attention_from_config(model, rngs=nnx.Rngs(0))
    composed = _config("exp", attention_degree=2).model
    attention_from_config(composed, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_operation_validates_the_endpoint_operands(kernel):
    q = jnp.zeros((1, 4, 2, 4))
    normalized = kernel == "softmax"
    gate = dict(query_gate_input=jnp.zeros((1, 4, 2)), key_gate_input=jnp.zeros((1, 4, 2)),
                gate_log_rate=jnp.zeros((2, 1)))
    for attend in (attention_core, blocked_dot_product_attention, strip_dot_product_attention):
        assert attend(q, q, q, attention_normalization=normalized, query_decay_mode="query_distance",
                      **gate).shape == q.shape
        for missing in gate:
            with pytest.raises(ValueError, match="Endpoint decay requires all three operands"):
                attend(q, q, q, attention_normalization=normalized, query_decay_mode="query_distance",
                       **{name: value for name, value in gate.items() if name != missing})
        with pytest.raises(ValueError, match="Endpoint decay requires all three operands"):
            attend(q, q, q, attention_normalization=normalized, **gate)  # the cumulative mode
        with pytest.raises(ValueError, match="Endpoint decay requires all three operands"):
            attend(q, q, q, attention_normalization=normalized, query_decay_mode="query_distance",
                   query_log_decay=jnp.zeros((1, 4, 2)), **gate)
    for name, shape in (("query_gate_input", (1, 4, 3)), ("key_gate_input", (1, 3, 2)), ("gate_log_rate", (2,))):
        with pytest.raises(ValueError, match="Endpoint inputs require"):
            blocked_dot_product_attention(q, q, q, attention_normalization=normalized,
                                          query_decay_mode="query_distance", **{**gate, name: jnp.zeros(shape)})
    for backend in ("fused", "mosaic", "cudnn"):
        with pytest.raises(ValueError, match="blocked backend"):
            attention_core(q, q, q, backend=backend, attention_normalization=normalized,
                           query_decay_mode="query_distance", **gate)
    with pytest.raises(ValueError, match="requires query coefficients alone, softmax, or the unnormalized"):
        blocked_dot_product_attention(q, q, q, attention_degree=2, query_decay_mode="query_distance", **gate)
    with pytest.raises(ValueError, match="forget gate requires query coefficients, softmax, or the unnormalized"):
        attention_core(q, q, q, attention_degree=2, query_decay_mode="query_distance", **gate)


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_key_projection_is_the_only_added_parameter(kernel):
    """Query-key conditioning adds forget_gate_key_kernel [layers, heads, hidden], zero at initialization,
    weight-decayed, and drawn from no random stream, to the query gate's parameters, which keep their names,
    shapes, declared initializations, and values. In the Transformer++ shape (15 layers, width 896, 14 heads)
    it adds 15 x 14 x 896 = 188,160 parameters."""
    endpoint, query = _config(kernel), _config(kernel, attention_forget_gate_conditioning="query")
    specs = attention_parameter_specs(endpoint.model)
    added = [spec for spec in specs if spec.name == "forget_gate_key_kernel"]
    assert tuple(spec for spec in specs if spec.name != "forget_gate_key_kernel") == attention_parameter_specs(
        query.model)
    assert len(added) == 1 and added[0].initial == (0.0,) * 8 and added[0].initial_shape == (8,)
    assert added[0].sharing == "per_head" and added[0].weight_decay
    assert not added[0].initializer and not added[0].initial_std
    shape = dict(vocab_size=32000, max_seq_len=2048, num_layers=15, hidden_size=896, num_heads=14, num_kv_heads=14,
                 qk_head_dim=64, value_head_dim=64, intermediate_size=2304)
    counts = {conditioning: model_parameter_count(_config(
        kernel, **shape, attention_forget_gate_conditioning=conditioning).model)
        for conditioning in ("query", "query_key")}
    ungated = model_parameter_count(_config(kernel, **shape, attention_forget_gate=False,
                                            attention_forget_gate_mode="cumulative",
                                            attention_forget_gate_conditioning="query").model)
    assert counts["query_key"] - counts["query"] == 15 * 14 * 896 == 188_160
    assert counts["query_key"] == ungated + (2 * 896 + 2) * 15 * 14


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_key_projection_decays_and_the_rates_do_not(kernel):
    """AdamW's mask: both gate projections decay; A_log and dt_bias do not. A nonzero key projection
    moves by the decoupled weight decay alone under a zero gradient."""
    config = _config(kernel)
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-3
    config.optim.weight_decay = 0.1
    model = build_model(config.model, 42)
    model.forget_gate_key_kernel[...] = jnp.full((2, 2, 8), 0.25)
    params = nnx.state(model, nnx.Param)
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded))}
    assert mask[("forget_gate_key_kernel",)] and mask[("forget_gate_kernel",)]
    assert not mask[("forget_gate_log_rate",)] and not mask[("forget_gate_dt_bias",)]
    from scaffolds.transformer.training.optim import make_optimizer

    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    values = {path: value[...] for path, value in nnx.to_flat_state(updates)}
    np.testing.assert_allclose(values[("forget_gate_key_kernel",)], -1e-3 * 0.1 * 0.25, rtol=2e-6, atol=1e-10)


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
    assert identity["attention_forget_gate_conditioning"] == "query_key"
    assert identity["attention_kernel"] == ("softmax" if kernel == "softmax" else "exp_unnormalized")
    assert identity["attention_formula"].startswith(
        ("causal_softmax" if kernel == "softmax" else "causal_exp")
        + "(QK^T / sqrt(d_k) + D) V, D_ti = (t - i) log alpha_ti, log alpha_ti = -exp(A_log) softplus("
          "x_t w_query + x_i w_key + dt_bias) per layer/head, w_key initialized zero")
    assert "each weight is multiplied" not in identity["attention_formula"]
    assert identity["attention_formula"].endswith("with a learned gain") is (kernel == "exp")
    assert identity["forget_gate_parameter_count"] == (2 * 8 + 2) * 2 * 2
    assert identity["attention_implementation"] == "blocked_causal_strips"
    query = attention_identity(_config(kernel, attention_forget_gate_conditioning="query"))
    assert "attention_forget_gate_conditioning" not in query
    assert query["forget_gate_parameter_count"] == (8 + 2) * 2 * 2
    assert "forget_gate" in tracking.run_tags(config, {})
    state = create_state(config, jax.random.PRNGKey(0))
    key_kernel = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 8)) / math.sqrt(8)
    state.params["forget_gate_key_kernel"][...] = key_kernel
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    gate = report["coefficients"]["forget_gate"]
    assert gate["finite"]
    assert gate["formula"] == "log alpha_ti = -exp(A_log) softplus(x_t w_query + x_i w_key + dt_bias)"
    assert gate["sample_domain"] == "all causal query-key pairs in the diagnostic sample, including diagonal"
    np.testing.assert_allclose(np.asarray(gate["key_kernel_l2"]), np.linalg.norm(key_kernel, axis=-1), rtol=1e-6)
    mean = np.asarray(gate["log_alpha_sample"]["mean"])
    assert mean.shape == (2, 2) and np.all(np.asarray(gate["log_alpha_sample"]["max"]) <= 0)
    np.testing.assert_allclose(np.asarray(gate["mean_half_life_tokens"]), np.log(0.5) / mean, rtol=1e-12)


@pytest.mark.parametrize("kernel", KERNELS)
def test_acceptance_selects_these_checks_on_cpu_and_gpu(kernel):
    """Query-key recipes run this module, the query-distance and forget-gate modules, on CPU and on the
    GPU; the endpoint arithmetic they share with the query polynomials on CPU only, whose GPU cases qualify
    those kernels. Query-only recipes and the query polynomials' query-key recipes do not select it."""
    config = _config(kernel)
    spec = acceptance.attention_spec(config, entry_module="tests")
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
    for paths in (cpu, gpu):
        assert acceptance.EXPONENTIAL_QUERY_KEY_TESTS in paths
        assert acceptance.EXPONENTIAL_QUERY_DISTANCE_TESTS in paths
        assert acceptance.FORGET_GATE_ATTENTION_TESTS in paths
        assert acceptance.QUERY_DISTANCE_DECAY_TESTS not in paths
    assert acceptance.ENDPOINT_DECAY_TESTS in cpu and acceptance.ENDPOINT_DECAY_TESTS not in gpu
    assert acceptance.BLOCKED_STRIPS_TESTS in cpu
    others = (config_to_dict(_config(kernel, attention_forget_gate_conditioning="query")),
              config_to_dict(_polynomial_config(attention_forget_gate_conditioning="query_key")))
    for flag in (True, False):
        for other in others:
            assert acceptance.EXPONENTIAL_QUERY_KEY_TESTS not in acceptance.test_paths(
                spec, other, resolved_backend="blocked", cpu=flag)
        assert acceptance.ENDPOINT_DECAY_TESTS in acceptance.test_paths(
            spec, others[1], resolved_backend="blocked", cpu=flag)
    gpu_cases = [name for name, test in globals().items() if name.startswith("test_")
                 and any(mark.name == "gpu" for mark in getattr(test, "pytestmark", ()))]
    assert sorted(gpu_cases) == ["test_gpu_strips_query_key_matches_float64_at_the_production_context"]


@pytest.mark.parametrize("kernel", KERNELS)
def test_acceptance_initialization_is_the_softmax_backbone(kernel):
    facts = acceptance.matches_softmax_backbone_initialization(_config(kernel))
    assert set(facts["mechanism_parameters"]) >= set(GATE)
    key = facts["mechanism_parameters"]["forget_gate_key_kernel"]
    assert key["weight_decay"] is True and key["initial"] == [0.0] * 8 and "initializer" not in key
    assert facts["mechanism_parameters"]["forget_gate_kernel"]["weight_decay"] is True
    assert facts["mechanism_parameters"]["forget_gate_log_rate"]["weight_decay"] is False


# --- GPU qualification at the production context ------------------------------------------------------

def _production_context(kernel, *, time=2048, heads=14, width=64, hidden=896, seed=23):
    """Observed BF16 output and dQ, dK, dV and FP32 gradients of w_query, w_key, A_log, and dt_bias through
    the gate projections (forget_gate.project_gate_inputs) and the strips, the host FP64 oracle of the same
    inputs (its projections and their chain rule in FP64), and each BF16 quantity's rounding floor."""
    rng = np.random.default_rng(seed)
    q, k, v, cotangent = (jnp.asarray(rng.standard_normal((1, time, heads, width)), jnp.bfloat16) for _ in range(4))
    x = rng.standard_normal((1, time, hidden)).astype(np.float32)
    bound = 1 / math.sqrt(hidden)
    query_w, key_w = (rng.uniform(-bound, bound, (heads, hidden)).astype(np.float32) for _ in range(2))
    # softplus(dt_bias) = 1/2 and exp(A_log) = 2 m_h: per-head rates m_h from 3e-4 (half-life ~2,300 tokens)
    # to 2 (under a token) at zero projections, each pair's moved by both endpoints (about 0.3x to 3x).
    dt_bias = np.full((heads, 1), math.log(math.expm1(0.5)), np.float32)
    log_rate = np.log(2 * np.geomspace(3e-4, 2.0, heads)).reshape(heads, 1).astype(np.float32)
    normalized = kernel == "softmax"
    attend = partial(attention_core, backend="blocked", attention_normalization=normalized,
                     query_decay_mode="query_distance", blocked_implementation="strips")

    def value_and_gradients(q, k, v, query_w, key_w, log_rate, dt_bias, cotangent):
        def function(q, k, v, query_w, key_w, log_rate, dt_bias):
            query, key = forget_gate.project_gate_inputs(jnp.asarray(x), query_w, dt_bias, key_w)
            return attend(q, k, v, query_gate_input=query, key_gate_input=key, gate_log_rate=log_rate)

        output, pullback = jax.vjp(function, q, k, v, query_w, key_w, log_rate, dt_bias)
        return (output, *pullback(cotangent))

    observed = jax.jit(value_and_gradients)(q, k, v, query_w, key_w, log_rate, dt_bias, cotangent)
    x64 = x.astype(np.float64)
    query = np.einsum("btd,hd->bth", x64, query_w.astype(np.float64)) + dt_bias[:, 0]
    key = np.einsum("btd,hd->bth", x64, key_w.astype(np.float64))
    output, (dq, dk, dv, dquery, dkey, drate) = _numpy_vjp(
        *(a.astype(jnp.float32) for a in (q, k, v)), query, key, log_rate, cotangent.astype(jnp.float32),
        normalized=normalized)
    expected = (output, dq, dk, dv, np.einsum("bth,btd->hd", dquery, x64), np.einsum("bth,btd->hd", dkey, x64),
                drate, dquery.sum(axis=(0, 1))[:, None])
    floors = [_relative_l2(np.asarray(jnp.asarray(want, jnp.bfloat16).astype(jnp.float32)), want)
              for want in expected[:4]]
    return observed, expected, floors


def _check_production_context(observed, expected, floors):
    for name, got, want, floor in zip(("output", "dq", "dk", "dv"), observed[:4], expected[:4], floors, strict=True):
        assert got.dtype == jnp.bfloat16, name
        got = np.asarray(got.astype(jnp.float32), np.float64)
        assert np.isfinite(got).all(), name
        error = _relative_l2(got, want)
        assert error < 1.2 * floor, f"{name}: relative L2 error {error:.3e}, BF16 floor {floor:.3e}"
    for name, got, want in zip(("query_kernel", "key_kernel", "log_rate", "dt_bias"), observed[4:], expected[4:],
                               strict=True):
        assert got.dtype == jnp.float32 and np.isfinite(np.asarray(got)).all(), name
        error = _relative_l2(got, want)
        assert error < 1e-4, f"{name}: relative L2 error {error:.3e}"


@pytest.mark.parametrize("kernel", KERNELS)
def test_production_context_check_holds_on_cpu_at_a_reduced_size(kernel):
    """The GPU case's own computation and bounds, two 512-query strips at reduced width, on CPU."""
    _check_production_context(*_production_context(kernel, time=640, heads=3, width=16, hidden=24))


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", KERNELS)
def test_gpu_strips_query_key_matches_float64_at_the_production_context(kernel):
    """BF16 Q/K/V at the production context, head count, and head width through the Transformer++-width gate
    projections and the causal strips, with pair half-lives from under one token to the context: the output
    and dQ, dK, dV each within 1.2x its BF16 rounding floor (the host FP64 oracle rounded to BF16), and the
    FP32 gradients of w_query, w_key, A_log, and dt_bias within 1e-4, against host FP64 of the same inputs."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    _check_production_context(*_production_context(kernel))
