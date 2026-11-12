"""Per-order cumulative forget gates for nonlinear query-coefficient polynomials.

Every polynomial order r, including zero, has its own Gated DeltaNet gate per layer and head,
log alpha_tr = -exp(A_log_r) softplus(x_t w_alpha_r + dt_bias_r), and its own cumulative sum
L_tr = sum_(j<=t) log alpha_jr, so that W_ti = sum_r c_r(x_t) s_ti^r exp(L_tr - L_ir). The gates
start as copies of the shared gate's draw: the model starts as cumulative FoX. The blocked backend
(scan or causal strips) evaluates one order's [B,H,Q,K] factor at a time; Mosaic has no
cumulative kernels. A nonfinite log alpha poisons the rows from its token on, as for shared FoX.
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
from scaffolds.transformer.attention.forget_gate import log_decay
from scaffolds.transformer.attention.multihead import attention_from_config
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample,
)
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state

DEGREES = (4, 8, 12)
IMPLEMENTATIONS = ("scan", "strips")
GATE = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")
# (time, scan block, strip): ragged 8-query blocks and 16-query strips of 37 tokens, one block and
# one strip longer than the sequence, and a single query.
SHAPES = ((37, 8, 16), (5, 8, 8), (1, 8, 8))
# Operands of the operation, in order; the coefficients and log decay carry the order axis last.
NAMES = ("output", "q", "k", "v", "coefficients", "log_decay")


def _config(degree=4, *, dtype="float32", **changes):
    """Tiny per-order cumulative recipes as allocated: factorial-scaled tanh query coefficients with
    W decay, no positional encoding, canonical QK RMSNorm, per-head output RMSNorm, causal strips;
    ``dtype`` is train.dtype (Mosaic recipes require BF16)."""
    model = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                 qk_head_dim=8, value_head_dim=8, intermediate_size=32, positional_encoding="none",
                 attention_backend="blocked", attention_blocked_implementation="strips", attention_degree=degree,
                 attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.0,
                 attention_coefficient_projection_initial_std=16 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_coefficient_kernel_weight_decay=True, attention_forget_gate=True,
                 attention_forget_gate_mode="cumulative", attention_forget_gate_per_order=True,
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True, initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = config.model.max_seq_len + 1, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    config.optim.weight_decay = .1
    config.train.dtype = dtype
    validate_config(config)
    return config


def _dense(q, k, v, c, log_decay_sums):
    """All pairs with explicit powers: sum_r c_tr s_ti^r exp(L_tr - L_ir) for i <= t."""
    time = q.shape[1]
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.arange(time)[:, None] >= jnp.arange(time)[None, :]
    sums = jnp.transpose(log_decay_sums, (0, 2, 1, 3))  # [b, h, t, R]
    coefficients = jnp.transpose(c, (0, 2, 1, 3))
    weights = sum(coefficients[..., r, None] * scores ** r
                  * jnp.exp(jnp.where(causal, sums[..., :, None, r] - sums[..., None, :, r], 0.0))
                  for r in range(c.shape[-1]))
    return jnp.einsum("bhti,bihd->bthd", jnp.where(causal, weights, 0.0), v)


def _numpy_vjp(q, k, v, c, log_decay_sums, cotangent):
    """Host FP64 forward and analytic VJP (dQ, dK, dV, dc, dL), one head at a time, with no JAX
    compilation: the oracle of the GPU cases."""
    q, k, v, c, sums, cotangent = (np.asarray(x, np.float64) for x in (q, k, v, c, log_decay_sums, cotangent))
    batch, time, heads, width = q.shape
    orders = c.shape[-1]
    output, dq, dk, dv = np.zeros_like(v), np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    dc, dsums = np.zeros_like(c), np.zeros_like(sums)
    causal = np.arange(time)[:, None] >= np.arange(time)[None, :]
    for b in range(batch):
        for h in range(heads):
            scores = q[b, :, h] @ k[b, :, h].T / math.sqrt(width)
            factors = [np.where(causal, np.exp(np.where(causal, sums[b, :, h, r, None] - sums[b, None, :, h, r], 0.0)),
                                0.0) for r in range(orders)]
            powers = [np.ones_like(scores)]
            for _ in range(orders - 1):
                powers.append(powers[-1] * scores)
            weights = sum(c[b, :, h, r, None] * powers[r] * factors[r] for r in range(orders))
            output[b, :, h] = weights @ v[b, :, h]
            dweights = np.where(causal, cotangent[b, :, h] @ v[b, :, h].T, 0.0)
            dv[b, :, h] = weights.T @ cotangent[b, :, h]
            dscores = np.zeros_like(scores)
            for r in range(orders):
                term = dweights * powers[r] * factors[r]
                dc[b, :, h, r] = term.sum(axis=-1)
                if r:
                    dscores += dweights * c[b, :, h, r, None] * r * powers[r - 1] * factors[r]
                # L_tr enters every pair (t, i) with sign +1 as the query and -1 as the key.
                pair = c[b, :, h, r, None] * term
                dsums[b, :, h, r] = pair.sum(axis=-1) - pair.sum(axis=0)
            dq[b, :, h] = dscores @ k[b, :, h] / math.sqrt(width)
            dk[b, :, h] = dscores.T @ q[b, :, h] / math.sqrt(width)
    return output, (dq, dk, dv, dc, dsums)


def _inputs(time, degree, *, batch=2, heads=3, width=8, seed=0):
    """Unit-scale scores (as QK RMSNorm gives), coefficients near Taylor, and per-order cumulative sums
    of log alpha whose half-lives range from under one token to about a thousand."""
    rng = np.random.default_rng(seed + 100 * time + degree)
    q, k, v, cotangent = (rng.standard_normal((batch, time, heads, width)) for _ in range(4))
    taylor = np.asarray([1 / math.factorial(r) for r in range(degree + 1)])
    c = taylor * (1 + 0.3 * np.tanh(rng.standard_normal((batch, time, heads, degree + 1))))
    sums = np.cumsum(-np.exp(rng.uniform(-7.0, 0.5, (batch, time, heads, degree + 1))), axis=1)
    return (q, k, v, c, sums), cotangent


def _production(implementation, degree, *, block=8, strip=16, compute=None, mode="cumulative"):
    """The blocked backend's per-order call: the public scan or strips, or the scan's own code in
    ``compute`` (FP64 for exact-arithmetic checks); the strips compute in the inputs' dtype."""
    options = dict(attention_degree=degree, attention_normalization=False, query_decay_mode=mode)
    if implementation == "strips":
        attend = partial(strip_dot_product_attention, strip_size=strip, **options)
    elif compute is not None:
        attend = partial(reference._blocked_scan, compute=compute, query_block_size=block, **options)
    else:
        attend = partial(blocked_dot_product_attention, query_block_size=block, **options)
    return lambda q, k, v, c, decay: attend(q, k, v, query_polynomial_coefficients=c, query_log_decay=decay)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _compare(actual, expected, tolerance, names=NAMES, order_axes=None, zeros=()):
    """Relative L2 errors below ``tolerance``, per order along ``order_axes[name]`` (the last axis for the
    coefficients and log decay by default) so that one order's error cannot hide behind the others'
    norms; the gradients named in ``zeros`` must be exactly zero."""
    order_axes = {"coefficients": -1, "log_decay": -1} if order_axes is None else order_axes
    for name, observed, wanted in zip(names, (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert observed.shape == wanted.shape and np.isfinite(observed).all(), name
        if name in zeros:
            np.testing.assert_array_equal(wanted, 0, err_msg=name)
            np.testing.assert_array_equal(observed, 0, err_msg=name)
            continue
        axis = order_axes.get(name)
        parts = ([(observed, wanted)] if axis is None else
                 [(np.take(observed, r, axis), np.take(wanted, r, axis)) for r in range(wanted.shape[axis])])
        for order, (part, reference_part) in enumerate(parts):
            assert np.linalg.norm(reference_part) > 0, (name, order)
            error = _relative_l2(part, reference_part)
            assert error < tolerance, f"{name} order {order}: relative L2 error {error:.3e} >= {tolerance}"


def _flat(state):
    return {path: np.asarray(value.get_value()) for path, value in nnx.to_flat_state(state)}


# --- The operation against FP64 all-pairs references ------------------------------------------------

@pytest.mark.parametrize("time", (1, 5, 19))
@pytest.mark.parametrize("degree", DEGREES)
def test_numpy_oracle_matches_float64_autodiff(degree, time):
    """The GPU cases' host oracle: output and analytic dQ, dK, dV, dc, dL against FP64 autodiff."""
    arrays, cotangent = _inputs(time, degree)
    observed = _numpy_vjp(*arrays, cotangent)
    with jax.enable_x64():
        expected = _vjp(_dense, tuple(jnp.asarray(x) for x in arrays), jnp.asarray(cotangent))
    for got, want in zip(jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-10, atol=1e-10 * max(np.abs(want).max(), 1.0))
    if time == 1:
        np.testing.assert_array_equal(observed[1][4], np.zeros_like(arrays[4]))  # L_t - L_t = 0


@pytest.mark.parametrize("time,block,strip", SHAPES)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("degree", DEGREES)
def test_float64_blocks_and_strips_match_the_all_pairs_reference(degree, implementation, time, block, strip):
    """In exact arithmetic (the scan's code and the strips in FP64), the output and the gradients of
    Q, K, V, the coefficients, and the per-order cumulative sums equal the all-pairs reference."""
    arrays, cotangent = _inputs(time, degree)
    with jax.enable_x64():
        operands = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        expected = jax.tree.map(np.asarray, _vjp(_dense, operands, jnp.asarray(cotangent)))
        production = _production(implementation, degree, block=block, strip=strip, compute=jnp.float64)
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent)))(*operands)
    for name, got, want in zip(NAMES, jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        got = np.asarray(got)
        assert got.dtype == np.float64 and got.shape == want.shape, name
        for order in range(want.shape[-1] if name in ("coefficients", "log_decay") else 1):
            part, wanted = ((got[..., order], want[..., order]) if name in ("coefficients", "log_decay")
                            else (got, want))
            np.testing.assert_allclose(part, wanted, rtol=1e-10, atol=1e-10 * max(np.abs(wanted).max(), 1.0),
                                       err_msg=f"{name} order {order}")


@pytest.mark.parametrize("time,block,strip", SHAPES)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("degree", DEGREES)
def test_float32_blocks_and_strips_match_float64(degree, implementation, time, block, strip):
    """The public FP32 scan and strips: output and every operand gradient within 3e-5 of FP64, the
    coefficient and log-decay gradients order by order."""
    arrays, cotangent = _inputs(time, degree)
    with jax.enable_x64():
        expected = jax.tree.map(np.asarray, _vjp(_dense, tuple(jnp.asarray(x, jnp.float64) for x in arrays),
                                                 jnp.asarray(cotangent)))
    production = _production(implementation, degree, block=block, strip=strip)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(x, jnp.float32) for x in arrays))
    # A single query keeps its own value with decay one: no gate can affect it.
    _compare(actual, expected, 3e-5, zeros=() if time > 1 else ("log_decay",))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("degree", DEGREES)
def test_gate_projection_chain_rule_matches_float64(degree, implementation):
    """The module's route, forget_gate.log_decay with per-order parameters [H,R,...] and the cumulative
    sum over time, through either implementation: the output and the gradients of Q, K, V, the
    coefficients, x, and every gate parameter (order by order) against an independent FP64 formula."""
    rng = np.random.default_rng(degree)
    (q, k, v, c, _), cotangent = _inputs(23, degree, heads=2)
    orders = degree + 1
    x = rng.standard_normal((2, 23, 7))
    kernel = rng.uniform(-1, 1, (2, orders, 7)) / math.sqrt(7)
    log_rate = np.log(rng.uniform(0.2, 4.0, (2, orders, 1)))
    dt_bias = rng.uniform(-5.0, -2.0, (2, orders, 1))
    operands = (q, k, v, c, x, kernel, log_rate, dt_bias)

    def oracle(q, k, v, c, x, w, a, b):
        log_alpha = -jnp.exp(a[..., 0]) * jnp.logaddexp(0.0, jnp.einsum("btd,hrd->bthr", x, w) + b[..., 0])
        return _dense(q, k, v, c, jnp.cumsum(log_alpha, axis=1))

    attend = _production(implementation, degree)

    def production(q, k, v, c, x, w, a, b):
        return attend(q, k, v, c, jnp.cumsum(log_decay(x, w, a, b), axis=1))

    with jax.enable_x64():
        expected = jax.tree.map(np.asarray, _vjp(oracle, tuple(jnp.asarray(value, jnp.float64) for value in operands),
                                                 jnp.asarray(cotangent)))
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(production, a, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(value, jnp.float32) for value in operands))
    _compare(actual, expected, 5e-5, names=("output", "q", "k", "v", "coefficients", "x", *GATE),
             order_axes={"coefficients": -1, **{name: 1 for name in GATE}})


@pytest.mark.parametrize("degree", DEGREES)
def test_bfloat16_strips_stay_near_float32(degree):
    """BF16 Q/K/V through the strips (the training dtype) stay within 2^-7 of the FP32 strips."""
    arrays, cotangent = _inputs(37, degree)
    production = _production("strips", degree)
    operands = [jnp.asarray(x, jnp.bfloat16 if i < 3 else jnp.float32) for i, x in enumerate(arrays)]
    low = jax.jit(production)(*operands)
    assert low.dtype == jnp.bfloat16
    with jax.default_matmul_precision("highest"):
        high = jax.jit(production)(*(x.astype(jnp.float32) for x in operands))
    assert _relative_l2(low.astype(jnp.float32), high) < 2 ** -7


def _value_shapes(jaxpr):
    """The shape of every value that a jaxpr and its sub-jaxprs (scan bodies, checkpoints) compute."""
    from jax.extend.core import subjaxprs

    shapes = {tuple(getattr(variable.aval, "shape", ())) for equation in jaxpr.eqns for variable in equation.outvars}
    for inner in subjaxprs(jaxpr):
        shapes |= _value_shapes(inner)
    return shapes


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("degree", DEGREES)
def test_one_order_factor_at_a_time(degree, implementation):
    """The forward and the VJP hold one order's [B,H,Q,K] decay factor at a time: no value of the scan
    (16-query blocks) or the strips (32-query strips) of 64 tokens carries the order axis with both a
    query axis and a key axis, which would multiply each block's working set by m + 1."""
    batch, time, heads, width, orders = 2, 64, 3, 4, degree + 1
    q, c = jnp.zeros((batch, time, heads, width)), jnp.zeros((batch, time, heads, orders))
    attend = _production(implementation, degree, block=16, strip=32)
    shapes = _value_shapes(jax.make_jaxpr(lambda *x: _vjp(attend, x, q))(q, q, q, c, c).jaxpr)
    # Query blocks or strips and key spans are the only axes of 16 or more here.
    assert not [shape for shape in shapes if orders in shape and sum(size >= 16 for size in shape) >= 2]
    # The walk reaches the blocks: one order's factor [B,H,Q,K] and every key's sums [B,H,1,K,R].
    assert (batch, heads, 16 if implementation == "scan" else 32, time) in shapes
    assert (batch, heads, 1, time, orders) in shapes


# --- Equivalences -------------------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("degree", DEGREES)
def test_duplicated_orders_are_shared_cumulative_fox(degree, implementation):
    """Every order a copy of one gate: the per-order call equals the shared cumulative FoX call, and the
    shared log decay's gradient is the sum of its copies' (FP32 to rounding, FP64 exactly). Through the
    gate projection, each shared parameter's gradient is the sum over its order copies."""
    (q, k, v, c, sums), cotangent = _inputs(37, degree)
    orders = degree + 1

    def run(dtype, compute):
        shared_sums = jnp.asarray(sums[..., 0], dtype)
        per_order = _production(implementation, degree, compute=compute)
        shared = (partial(strip_dot_product_attention, strip_size=16) if implementation == "strips" else
                  partial(reference._blocked_scan, compute=compute, query_block_size=8))
        operands = tuple(jnp.asarray(x, dtype) for x in (q, k, v, c))
        g = jnp.asarray(cotangent, dtype)

        def copied(q, k, v, c, L):
            return per_order(q, k, v, c, jnp.repeat(L[..., None], orders, -1))

        copies = jax.jit(lambda *x: _vjp(copied, x, g))(*operands, shared_sums)
        one = jax.jit(lambda *x: _vjp(lambda q, k, v, c, L: shared(
            q, k, v, attention_degree=degree, attention_normalization=False, query_polynomial_coefficients=c,
            query_log_decay=L), x, g))(*operands, shared_sums)
        return jax.tree.map(np.asarray, copies), jax.tree.map(np.asarray, one)

    with jax.default_matmul_precision("highest"):
        copies, one = run(jnp.float32, jnp.float32)
    _compare(copies, one, 2e-6, order_axes={"coefficients": -1})
    with jax.enable_x64():
        copies, one = run(jnp.float64, jnp.float64)
    _compare(copies, one, 1e-12, order_axes={"coefficients": -1})

    # Through the projection: duplicated parameters [H,R,...] against the shared ones [H,...].
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((2, 37, 5)), jnp.float32)
    shared_parameters = (jnp.asarray(rng.uniform(-1, 1, (3, 5)) / math.sqrt(5), jnp.float32),
                         jnp.asarray(np.log(rng.uniform(0.2, 4.0, (3, 1))), jnp.float32),
                         jnp.asarray(rng.uniform(-5.0, -2.0, (3, 1)), jnp.float32))
    duplicated = tuple(jnp.repeat(p[:, None], orders, axis=1) for p in shared_parameters)
    operands = tuple(jnp.asarray(value, jnp.float32) for value in (q, k, v, c))
    g = jnp.asarray(cotangent, jnp.float32)
    attend = _production(implementation, degree)
    shared_attend = (partial(strip_dot_product_attention, strip_size=16) if implementation == "strips" else
                     partial(blocked_dot_product_attention, query_block_size=8))
    with jax.default_matmul_precision("highest"):
        per_order = _vjp(lambda w, a, b: attend(*operands, jnp.cumsum(log_decay(x, w, a, b), axis=1)), duplicated, g)
        shared = _vjp(lambda w, a, b: shared_attend(
            *operands[:3], attention_degree=degree, attention_normalization=False,
            query_polynomial_coefficients=operands[3], query_log_decay=jnp.cumsum(log_decay(x, w, a, b), axis=1)),
            shared_parameters, g)
    assert _relative_l2(per_order[0], shared[0]) < 2e-6
    for name, copy_gradient, gradient in zip(GATE, per_order[1], shared[1], strict=True):
        assert _relative_l2(np.asarray(copy_gradient).sum(axis=1), gradient) < 2e-5, name


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("degree", DEGREES)
def test_constant_gates_are_per_order_query_distance_gates(degree, implementation):
    """With one log alpha per head and order at every token, each order's cumulative decay is
    exp((t - i) log alpha_r): the per-order cumulative call equals the per-order query-distance call
    with the same constants, the output and the gradients of Q, K, V, the coefficients, and the
    constants agreeing to rounding in FP32 and in exact arithmetic."""
    (q, k, v, c, _), cotangent = _inputs(37, degree)
    rates = -np.exp(np.random.default_rng(3).uniform(-6.0, 0.5, (3, degree + 1)))

    def run(dtype, compute):
        cumulative = _production(implementation, degree, compute=compute)
        distance = _production(implementation, degree, compute=compute, mode="query_distance")
        operands = tuple(jnp.asarray(x, dtype) for x in (q, k, v, c, rates))
        g = jnp.asarray(cotangent, dtype)

        def tokens(rate):
            return jnp.broadcast_to(rate, q.shape[:3] + (degree + 1,))

        first = jax.jit(lambda *x: _vjp(lambda q, k, v, c, a: cumulative(q, k, v, c, jnp.cumsum(tokens(a), axis=1)),
                                        x, g))(*operands)
        second = jax.jit(lambda *x: _vjp(lambda q, k, v, c, a: distance(q, k, v, c, tokens(a)), x, g))(*operands)
        return jax.tree.map(np.asarray, first), jax.tree.map(np.asarray, second)

    names = ("output", "q", "k", "v", "coefficients", "rates")
    with jax.default_matmul_precision("highest"):
        cumulative, distance = run(jnp.float32, jnp.float32)
    _compare(cumulative, distance, 2e-5, names=names, order_axes={"coefficients": -1, "rates": -1})
    with jax.enable_x64():
        cumulative, distance = run(jnp.float64, jnp.float64)
    _compare(cumulative, distance, 1e-11, names=names, order_axes={"coefficients": -1, "rates": -1})


# --- Invalid values, causality, and order isolation ---------------------------------------------------

# Raw log alpha of the five orders at degree four, the same at every token: open gates (exp(0) = 1),
# closed gates (exp(-100) is subnormal in FP32; -1e30 underflows at once), and the two mixed by order.
GATE_STATES = {"open": (0.0,) * 5, "closed": (-100.0,) * 5, "closed_at_once": (-1e30,) * 5,
               "mixed": (0.0, -1e30, 0.0, -100.0, -1e30)}


@pytest.mark.parametrize("state", list(GATE_STATES))
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_open_and_closed_gates_stay_finite(implementation, state):
    """Each order's L_tr - L_ir is masked before exp: a closed gate's huge positive gaps at future keys
    never overflow, so the output and the gradients of Q, K, V, the coefficients, and the raw log alpha
    (through the cumulative sum) stay finite and match FP64. An open order is its ungated term over the
    causal prefix; a closed order keeps each query's own key, c_tr s_tt^r v_t, and its log alpha has a
    zero gradient."""
    (q, k, v, c, _), cotangent = _inputs(37, 4)
    values = np.asarray(GATE_STATES[state])
    closed = values < 0
    raw = np.broadcast_to(values, c.shape)
    attend = _production(implementation, 4)
    with jax.enable_x64():
        expected = jax.tree.map(np.asarray, _vjp(lambda q, k, v, c, a: _dense(q, k, v, c, jnp.cumsum(a, axis=1)),
                                                 tuple(jnp.asarray(x, jnp.float64) for x in (q, k, v, c, raw)),
                                                 jnp.asarray(cotangent)))
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(lambda q, k, v, c, a: attend(q, k, v, c, jnp.cumsum(a, axis=1)),
                                         x, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(x, jnp.float32) for x in (q, k, v, c, raw)))
    for name, array in zip(NAMES, (actual[0], *actual[1]), strict=True):
        assert np.isfinite(np.asarray(array)).all(), name
    _compare((actual[0], actual[1][:4]), (expected[0], expected[1][:4]), 3e-5, names=NAMES[:5])
    log_alpha, wanted = np.asarray(actual[1][4]), expected[1][4]
    np.testing.assert_array_equal(wanted[..., closed], 0)
    np.testing.assert_array_equal(log_alpha[..., closed], 0)
    for order in np.flatnonzero(~closed):
        assert _relative_l2(log_alpha[..., order], wanted[..., order]) < 3e-5, order
    # The limits themselves: the causal prefix for open orders, the diagonal alone for closed ones.
    scores = np.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    masks = np.where(closed[:, None, None], np.eye(37), np.tril(np.ones((37, 37))))
    weights = sum(np.transpose(c, (0, 2, 1, 3))[..., r, None] * scores ** r * masks[r] for r in range(5))
    limit = np.einsum("bhti,bihd->bthd", weights, v)
    np.testing.assert_allclose(actual[0], limit, rtol=2e-5, atol=2e-5 * np.abs(limit).max())


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_nonfinite_log_alpha_poisons_from_its_token_on(implementation, bad):
    """A nonfinite log alpha of one order at token 20 makes that order's cumulative sum nonfinite from
    token 20 on: those rows of its batch and head are poisoned, and every other row is unchanged."""
    (q, k, v, c, sums), _ = _inputs(37, 4)
    raw = np.diff(sums, axis=1, prepend=0.0)
    operands = [jnp.asarray(x, jnp.float32) for x in (q, k, v, c)]
    attend = jax.jit(lambda *x: _production(implementation, 4)(*x[:4], jnp.cumsum(x[4], axis=1)))
    clean = np.asarray(attend(*operands, jnp.asarray(raw, jnp.float32)))
    output = np.asarray(attend(*operands, jnp.asarray(raw, jnp.float32).at[0, 20, 1, 2].set(bad)))
    invalid = ~np.isfinite(output).all(axis=-1)
    expected = np.zeros(invalid.shape, bool)
    expected[0, 20:, 1] = True
    np.testing.assert_array_equal(invalid, expected)
    np.testing.assert_array_equal(output[~expected], clean[~expected])
    # The gate projection turns any nonfinite input or parameter of an order into NaN.
    x = jnp.ones((1, 4, 3))
    alpha = log_decay(x.at[0, 2, 1].set(bad), jnp.ones((2, 5, 3)), jnp.zeros((2, 5, 1)), jnp.zeros((2, 5, 1)))
    assert np.isnan(alpha[0, 2]).all() and np.isfinite(np.delete(np.asarray(alpha), 2, axis=1)).all()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_gate_reaches_later_rows_through_its_own_order_only(implementation):
    """Changing order two's log alpha at token 6 changes rows 6 onward and no earlier row; with that
    order's coefficient zero for every query, the output is unchanged bitwise."""
    (q, k, v, c, sums), _ = _inputs(19, 4)
    raw = np.diff(sums, axis=1, prepend=0.0)
    attend = jax.jit(lambda q, k, v, c, a: _production(implementation, 4, block=4, strip=8)(
        q, k, v, c, jnp.cumsum(a, axis=1)))
    q, k, v, c, raw = (jnp.asarray(x, jnp.float32) for x in (q, k, v, c, raw))
    changed = raw.at[0, 6, 0, 2].add(-0.5)
    rows = np.abs(np.asarray(attend(q, k, v, c, changed) - attend(q, k, v, c, raw))).max(axis=-1)[0, :, 0] > 0
    np.testing.assert_array_equal(rows, np.arange(19) >= 6)
    without = c.at[..., 2].set(0.0)
    np.testing.assert_array_equal(attend(q, k, v, without, changed), attend(q, k, v, without, raw))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_degree_zero_decays_with_its_own_sum_and_the_diagonal_is_exact(implementation):
    """Zero scores leave only order zero: with V the identity the output is the decay matrix
    exp(L_t0 - L_i0) below the diagonal, exactly one on it, and zero above."""
    (_, _, _, c, sums), _ = _inputs(9, 4)
    q = k = jnp.zeros((2, 9, 3, 8), jnp.float32)
    v = jnp.broadcast_to(jnp.eye(9, dtype=jnp.float32)[None, :, None, :], (2, 9, 3, 9))
    c = jnp.zeros(c.shape, jnp.float32).at[..., 0].set(1.0)
    sums = jnp.asarray(sums, jnp.float32)
    weights = np.asarray(_production(implementation, 4, block=4, strip=8)(q, k, v, c, sums))[0, :, 0]
    np.testing.assert_array_equal(np.diag(weights), np.ones(9))
    np.testing.assert_array_equal(np.triu(weights, 1), np.zeros((9, 9)))
    order_zero = np.asarray(sums)[0, :, 0, 0]
    expected = np.tril(np.exp(order_zero[:, None] - order_zero[None, :]))
    np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=0)
    assert 0 < weights[8, 0] < 1


# --- Recipes, parameters, and models ------------------------------------------------------------------

@pytest.mark.parametrize("degree", DEGREES)
def test_accepted_recipes(degree):
    """Query-conditioned nonlinear coefficients (tanh, sigmoid, affine sigmoid, pure tanh; diagonal or
    dense readout) with RoPE or no positional encoding, the scan or the strips, on the blocked backend."""
    for encoding in ("none", "rope"):
        for implementation in IMPLEMENTATIONS:
            config = _config(degree, positional_encoding=encoding, attention_blocked_implementation=implementation)
            assert config.model.attention_forget_gate_per_order
            attention_from_config(config.model, rngs=nnx.Rngs(0))
    for changes in ({"attention_coefficient_activation": "sigmoid", "attention_coefficient_amplitude_initial": 1.0,
                     "attention_coefficient_scale": "none"},
                    {"attention_coefficient_activation": "pure_tanh", "attention_coefficient_amplitude_initial": 1.0,
                     "attention_coefficient_scale": "none"},
                    {"attention_coefficient_activation": "affine_sigmoid"},
                    {"attention_coefficient_readout": "dense", "attention_coefficient_readout_scale": 5 ** -.5}):
        _config(degree, **changes)


PATH = {"positional_encoding": "path", "qk_head_dim": 16, "value_head_dim": 16,
        "attention_blocked_implementation": "scan"}
INVALID = (
    ({"attention_backend": "auto"}, "Per-order gates"),
    ({"attention_backend": "fused"}, "Per-order gates"),
    ({"attention_forget_gate_conditioning": "query_key"}, "Per-order cumulative gates take query conditioning"),
    ({"attention_forget_gate_rate_activation": "sigmoid"}, "Per-order gates"),
    ({"attention_forget_gate_form": "constant"}, "Per-order gates"),
    ({"attention_forget_gate_form": "sigmoid"}, "Per-order gates"),
    (PATH, "Per-order gates"),
    ({"attention_forget_gate": False}, "Per-order gates"),
    ({"attention_coefficient_conditioning": "static", "attention_coefficient_activation": "identity",
      "attention_coefficient_amplitude_scale": "none", "attention_coefficient_projection_initial_std": 0.0,
      "attention_coefficient_amplitude_initial": 1.0, "attention_coefficient_kernel_weight_decay": False},
     "Per-order gates"),
    ({"attention_degree": None, "attention_coefficients": "fixed", "attention_coefficient_conditioning": "static",
      "attention_coefficient_activation": "identity", "attention_coefficient_amplitude_scale": "none",
      "attention_coefficient_scale": "none", "attention_coefficient_projection_initial_std": 0.0,
      "attention_coefficient_amplitude_initial": 1.0, "attention_coefficient_kernel_weight_decay": False},
     "Per-order gates"),
)


@pytest.mark.parametrize("change,message", INVALID,
                         ids=[",".join(f"{k}={v}" for k, v in c.items())[:60] for c, _ in INVALID])
def test_unqualified_recipes_are_rejected(change, message):
    """Auto and fused, endpoint or bounded rates, other gate forms, PaTH, no gate, static coefficients,
    and softmax or the exponential (Mosaic below)."""
    with pytest.raises(ValueError, match=message):
        _config(4, **change)


# The Transformer++ head widths and context, where the Mosaic decay kernels run per-order gates.
MOSAIC_SHAPE = dict(qk_head_dim=64, value_head_dim=64, max_seq_len=2048)


@pytest.mark.parametrize("degree", DEGREES)
def test_mosaic_refuses_cumulative_gates_where_it_runs_query_distance_gates(degree, monkeypatch):
    """At the Transformer++ widths and context with BF16, Mosaic runs the per-order query-distance twin
    but has no cumulative kernels: its support check is false for this recipe, which is refused on
    Mosaic. The per-order cumulative clause refuses Mosaic on its own, should that check ever accept."""
    twin = dict(attention_forget_gate_mode="query_distance", **MOSAIC_SHAPE)
    assert _config(degree, **twin).model.mosaic_decay_supported()
    _config(degree, attention_backend="mosaic", dtype="bfloat16", **twin)
    assert not _config(degree, **MOSAIC_SHAPE).model.mosaic_decay_supported()
    with pytest.raises(ValueError, match="query-distance"):
        _config(degree, attention_backend="mosaic", dtype="bfloat16", **MOSAIC_SHAPE)
    monkeypatch.setattr(ModelConfig, "mosaic_decay_supported", lambda self: True)
    with pytest.raises(ValueError, match="Per-order cumulative gates take query conditioning on the blocked"):
        _config(degree, attention_backend="mosaic", dtype="bfloat16", **MOSAIC_SHAPE)


def test_the_operation_rejects_cumulative_orders_outside_query_polynomials_and_off_blocked():
    q = jnp.zeros((1, 4, 2, 4))
    c = jnp.zeros((1, 4, 2, 5))
    sums = jnp.zeros((1, 4, 2, 5))
    with pytest.raises(ValueError, match="Per-order decay requires query-coefficient polynomial attention"):
        blocked_dot_product_attention(q, q, q, attention_normalization=False, query_log_decay=sums)
    with pytest.raises(ValueError, match="log decay"):
        blocked_dot_product_attention(q, q, q, attention_degree=4, attention_normalization=False,
                                      query_polynomial_coefficients=c, query_log_decay=sums[..., :3])
    for backend in ("mosaic", "fused"):
        with pytest.raises(ValueError, match="blocked reference backend"):
            attention_core(q, q, q, backend=backend, attention_degree=4, attention_normalization=False,
                           query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                           query_log_decay=sums)
    with pytest.raises(ValueError, match="Endpoint decay requires all three operands and query-distance mode"):
        blocked_dot_product_attention(q, q, q, attention_degree=4, attention_normalization=False,
                                      query_polynomial_coefficients=c, query_gate_input=sums, key_gate_input=sums,
                                      gate_log_rate=jnp.zeros((2, 5, 1)))


@pytest.mark.parametrize("degree", DEGREES)
def test_parameters_counts_and_starts(degree):
    """Each order's gate is a copy of the shared draw (so every other parameter, and the declared
    initialization, are the shared control's), the per-order query-distance recipe has the same
    parameters, and each extra order adds (896 + 2) * 15 * 14 = 188,580 parameters in the
    Transformer++ shape."""
    config = _config(degree)
    control = _config(degree, attention_forget_gate_per_order=False)
    distance = _config(degree, attention_forget_gate_mode="query_distance")
    assert attention_parameter_specs(config.model) == attention_parameter_specs(distance.model)
    old, new = (_flat(nnx.state(build_model(c.model, 42), nnx.Param)) for c in (control, config))
    assert old.keys() == new.keys()
    for path, value in old.items():
        if path[0].startswith("forget_gate_"):
            np.testing.assert_array_equal(new[path], np.broadcast_to(value[:, :, None, :], new[path].shape))
            assert new[path].shape[2] == degree + 1
        else:
            np.testing.assert_array_equal(new[path], value, err_msg=str(path))
    shape = dict(vocab_size=32000, max_seq_len=2048, num_layers=15, hidden_size=896, num_heads=14, num_kv_heads=14,
                 qk_head_dim=64, value_head_dim=64, intermediate_size=2304,
                 attention_coefficient_projection_initial_std=896 ** -.5)
    counts = {name: model_parameter_count(_config(degree, **shape, **changes).model) for name, changes in (
        ("per_order", {}), ("shared", {"attention_forget_gate_per_order": False}),
        ("distance", {"attention_forget_gate_mode": "query_distance"}))}
    assert counts["per_order"] - counts["shared"] == degree * 188_580 == degree * (896 + 2) * 15 * 14
    assert counts["per_order"] == counts["distance"]
    facts = acceptance.matches_softmax_backbone_initialization(config)
    for name in GATE:
        assert facts["mechanism_parameters"][name]["repeat_initial_axis"] == -2, name
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in nnx.to_flat_state(
        attention_weight_decay_mask(nnx.state(build_model(config.model, 42), nnx.Param), excluded_names=excluded))}
    assert mask[("forget_gate_kernel",)] and mask[("query_polynomial_kernel",)]
    for name in ("forget_gate_log_rate", "forget_gate_dt_bias", "query_polynomial_bias", "query_polynomial_amplitude"):
        assert not mask[(name,)], name


@pytest.mark.parametrize("degree", DEGREES)
def test_the_model_starts_as_shared_cumulative_fox(degree):
    """At initialization the per-order model is the shared cumulative FoX model: the same logits to
    rounding, the same gradient for every other parameter, and each shared gate parameter's gradient
    the sum over its order copies."""
    batch = jnp.asarray(np.random.default_rng(2).integers(0, 32, (2, 9)), jnp.int32)
    results = {}
    for name, config in (("per_order", _config(degree)),
                         ("shared", _config(degree, attention_forget_gate_per_order=False))):
        state = create_state(config, jax.random.PRNGKey(3))
        logits = np.asarray(nnx.merge(state.graphdef, state.params)(batch[:, :-1], deterministic=True))
        results[name] = logits, _flat(loop.loss_and_grads(state, batch, jax.random.PRNGKey(0))[1])
    np.testing.assert_allclose(results["per_order"][0], results["shared"][0], rtol=2e-5, atol=2e-6)
    per_order, shared = results["per_order"][1], results["shared"][1]
    assert per_order.keys() == shared.keys()
    for path, expected in shared.items():
        observed = per_order[path].sum(axis=2) if path[0] in GATE else per_order[path]
        assert np.linalg.norm(observed - expected) <= 5e-5 * np.linalg.norm(expected) + 1e-9, path


@pytest.mark.parametrize("degree", DEGREES)
def test_token_independent_gates_give_the_per_order_query_distance_model(degree):
    """With every w_alpha_r = 0 each order's log alpha is constant over tokens: the per-order cumulative
    model equals the per-order query-distance model with the same parameters, its logits and the
    gradient of every parameter but w_alpha (whose token sensitivities differ by design) agreeing."""
    batch = jnp.asarray(np.random.default_rng(4).integers(0, 32, (2, 9)), jnp.int32)
    results = {}
    for mode in ("cumulative", "query_distance"):
        state = create_state(_config(degree, attention_forget_gate_mode=mode), jax.random.PRNGKey(5))
        state.params["forget_gate_kernel"][...] = jnp.zeros_like(state.params["forget_gate_kernel"][...])
        # Different rates for every order.
        state.params["forget_gate_log_rate"][...] = state.params["forget_gate_log_rate"][...] + jnp.linspace(
            -2, 1, degree + 1)[None, None, :, None]
        logits = np.asarray(nnx.merge(state.graphdef, state.params)(batch[:, :-1], deterministic=True))
        results[mode] = logits, _flat(loop.loss_and_grads(state, batch, jax.random.PRNGKey(0))[1])
    np.testing.assert_allclose(results["cumulative"][0], results["query_distance"][0], rtol=2e-5, atol=2e-6)
    for path, expected in results["query_distance"][1].items():
        if path == ("forget_gate_kernel",):
            continue
        observed = results["cumulative"][1][path]
        assert np.linalg.norm(observed - expected) <= 5e-5 * np.linalg.norm(expected) + 1e-9, path


@pytest.mark.parametrize("degree", DEGREES)
def test_training_updates_every_order_of_every_gate_parameter(degree):
    config = _config(degree)
    initial = create_state(config, jax.random.PRNGKey(0))
    # Let the orders differ, as they do once training starts.
    rng = np.random.default_rng(6)
    for name in GATE:
        value = initial.params[name][...]
        initial.params[name][...] = value + jnp.asarray(0.1 * rng.standard_normal(value.shape), value.dtype)
    batch = jnp.arange(18).reshape(2, 9) % 32
    gradients = _flat(loop.loss_and_grads(initial, batch, jax.random.PRNGKey(0))[1])
    for name in GATE:
        gradient = gradients[(name,)]
        assert gradient.shape[:3] == (2, 2, degree + 1), name
        assert np.isfinite(gradient).all() and np.all(np.linalg.norm(gradient, axis=-1) > 0), name
        assert not np.allclose(gradient[:, :, 0], gradient[:, :, -1]), name
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    assert np.isfinite(loss)
    for name in GATE:
        moved = np.linalg.norm(np.asarray(first.params[name][...] - initial.params[name][...]), axis=-1)
        assert np.all(moved > 0), name


@pytest.mark.parametrize("degree", DEGREES)
def test_identity_and_per_order_diagnostics(degree):
    config = _config(degree)
    orders = degree + 1
    identity = attention_identity(config)
    assert identity["attention_forget_gate_per_order"] is True
    assert identity["attention_forget_gate_mode"] == "cumulative"
    assert identity["attention_formula"].startswith(
        f"sum_(i<=t) sum_(r=0..{degree}) c_r(x_t) s_ti^r prod_(j=i+1..t) alpha_jr v_i; "
        "log alpha_tr = -exp(A_log_r) softplus(x_t w_alpha_r + dt_bias_r); c_r uses")
    assert "lambda" not in identity["attention_formula"]
    assert identity["forget_gate_parameter_count"] == orders * (16 + 2) * 2 * 2
    assert identity["attention_implementation"] == "blocked_causal_strips"
    distance = attention_identity(_config(degree, attention_forget_gate_mode="query_distance"))
    assert "exp(-(t-i) lambda_tir)" in distance["attention_formula"]
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    gate = report["coefficients"]["forget_gate"]
    assert gate["finite"] and gate["axes"] == ["layer", "head", "polynomial_order"]
    assert gate["formula"].endswith("degree-r weight multiplied by prod_(j=i+1..t) alpha_jr")
    mean = np.asarray(gate["log_alpha_sample"]["mean"])
    assert mean.shape == (2, 2, orders) and np.all(np.asarray(gate["log_alpha_sample"]["max"]) <= 0)
    half_life = np.asarray(gate["mean_half_life_tokens"])
    np.testing.assert_allclose(half_life, np.log(0.5) / mean, rtol=1e-12)
    # Duplicated starts: every order has the shared gate's statistics.
    np.testing.assert_allclose(mean, np.broadcast_to(mean[..., :1], mean.shape), rtol=1e-6)
    assert np.asarray(gate["rate_exp_A_log"]).shape == (2, 2, orders)
    # The report follows the recipe's decay mode: the per-order query-distance twin's formula.
    twin = _config(degree, attention_forget_gate_mode="query_distance")
    report = attention_diagnostics(create_state(twin, jax.random.PRNGKey(0)), twin, *fixed_diagnostic_sample(twin))
    assert report["valid"], report["issues"]
    gate = report["coefficients"]["forget_gate"]
    assert gate["axes"] == ["layer", "head", "polynomial_order"]
    assert gate["formula"].endswith("degree-r weight multiplied by exp((t-i) log alpha_tir)")


def test_acceptance_selects_these_checks_on_cpu_and_gpu():
    """Per-order cumulative recipes run this module on CPU and its production-context cases on the GPU;
    the shared per-order checks stay on CPU only (their GPU cases qualify query-distance kernels).
    Per-order query-distance recipes keep their selection."""
    for degree in DEGREES:
        config = _config(degree)
        spec = acceptance.attention_spec(config, entry_module="tests")
        recipe = config_to_dict(config)
        cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
        gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
        assert acceptance.PER_ORDER_CUMULATIVE_TESTS in cpu and acceptance.PER_ORDER_CUMULATIVE_TESTS in gpu
        assert acceptance.PER_ORDER_DECAY_TESTS in cpu and acceptance.PER_ORDER_DECAY_TESTS not in gpu
        assert acceptance.BLOCKED_STRIPS_TESTS in cpu
        assert acceptance.QUERY_DISTANCE_DECAY_TESTS not in cpu + gpu
        distance = config_to_dict(_config(degree, attention_forget_gate_mode="query_distance"))
        for flag in (True, False):
            paths = acceptance.test_paths(spec, distance, resolved_backend="blocked", cpu=flag)
            assert acceptance.PER_ORDER_DECAY_TESTS in paths and acceptance.PER_ORDER_CUMULATIVE_TESTS not in paths
    gpu_cases = [name for name, test in globals().items() if name.startswith("test_")
                 and any(mark.name == "gpu" for mark in getattr(test, "pytestmark", ()))]
    assert sorted(gpu_cases) == ["test_gpu_strips_per_order_cumulative_matches_float64_at_the_production_context"]


# --- GPU qualification at the production context ------------------------------------------------------

def _production_context(degree, *, time=2048, heads=14, width=64, seed=31):
    """Observed BF16 output and dQ, dK, dV and FP32 dc and dL through the strips, the FP64 oracle of the
    same inputs, and each BF16 quantity's rounding floor. Unit-scale scores make every order
    contribute; the per-order cumulative sums are FP32, as the model's cumulative sum gives them."""
    rng = np.random.default_rng(seed + degree)
    q, k, v, cotangent = (jnp.asarray(rng.standard_normal((1, time, heads, width)), jnp.bfloat16) for _ in range(4))
    orders = degree + 1
    taylor = np.asarray([1 / math.factorial(r) for r in range(orders)])
    c = jnp.asarray(taylor * (1 + 0.3 * np.tanh(rng.standard_normal((1, time, heads, orders)))), jnp.float32)
    # Per head and order, rates from 3e-4 (half-life ~2,300 tokens) to 2 (under a token), varying by token.
    rates = np.exp(rng.uniform(math.log(3e-4), math.log(2.0), (heads, orders)))
    raw = -rates * np.exp(rng.uniform(-0.5, 0.5, (1, time, heads, orders)))
    sums = jnp.cumsum(jnp.asarray(raw, jnp.float32), axis=1)
    attend = partial(attention_core, backend="blocked", attention_degree=degree, attention_normalization=False,
                     attention_coefficient_activation="tanh", query_decay_mode="cumulative",
                     blocked_implementation="strips")

    def value_and_gradients(q, k, v, c, sums, cotangent):
        output, pullback = jax.vjp(lambda q, k, v, c, L: attend(q, k, v, query_polynomial_coefficients=c,
                                                                query_log_decay=L), q, k, v, c, sums)
        return (output, *pullback(cotangent))

    observed = jax.jit(value_and_gradients)(q, k, v, c, sums, cotangent)
    output, gradients = _numpy_vjp(*(x.astype(jnp.float32) for x in (q, k, v)), c, sums, cotangent.astype(jnp.float32))
    expected = (output, *gradients)
    floors = [_relative_l2(np.asarray(jnp.asarray(want, jnp.bfloat16).astype(jnp.float32)), want)
              for want in expected[:4]]
    return observed, expected, floors


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_strips_per_order_cumulative_matches_float64_at_the_production_context(degree):
    """BF16 Q/K/V at the production context, head count, and head width through the causal strips,
    with unit-scale scores and per-order decays whose half-lives range from under one token to the
    context: the output and dQ, dK, dV each within 1.2x its BF16 rounding floor (the host FP64 oracle
    rounded to BF16), and the FP32 coefficient and log-decay gradients within 1e-4 order by order,
    against host FP64 of the same inputs."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    observed, expected, floors = _production_context(degree)
    for name, got, want, floor in zip(("output", "dq", "dk", "dv"), observed[:4], expected[:4], floors, strict=True):
        assert got.dtype == jnp.bfloat16, name
        got = np.asarray(got.astype(jnp.float32), np.float64)
        assert np.isfinite(got).all(), name
        error = _relative_l2(got, want)
        assert error < 1.2 * floor, f"{name}: relative L2 error {error:.3e}, BF16 floor {floor:.3e}"
    for name, got, want in zip(("coefficients", "log_decay"), observed[4:], expected[4:], strict=True):
        assert got.dtype == jnp.float32 and np.isfinite(np.asarray(got)).all(), name
        for order in range(degree + 1):
            error = _relative_l2(np.asarray(got)[..., order], want[..., order])
            assert error < 1e-4, f"{name} order {order}: relative L2 error {error:.3e}"
