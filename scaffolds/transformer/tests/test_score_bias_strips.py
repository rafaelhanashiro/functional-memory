"""GPU qualification of the blocked backend's causal strips with per-degree score biases.

The causal strips (``strip_dot_product_attention``) train score-bias recipes on the blocked backend; they
are plain JAX and run on B200, B300 and H100 alike. Use ``pytest -m gpu tests/test_score_bias_strips.py``
inside a GPU allocation with ``JAX_PLATFORMS=cuda,cpu`` (acceptance's GPU stage), which gives the FP64
references the host CPU. Every case runs the production dispatch, ``attention_core`` on the blocked backend
with ``blocked_implementation="strips"`` (counted), at production precision: Q, K and V rounded to BF16
inside the program, as QK RMSNorm and RoPE hand them over; FP32 coefficients, gate inputs and shift
operands; a BF16 cotangent; degree eight, 2,048 tokens in four 512-query strips, d_K = d_V = 64, two
heads. Its output and every gradient are compared with an independent dense FP64 reference written here
(every pair, explicit (s_ti + beta_r)^r, JAX FP64 autodiff on the host) from the same rounded inputs:
the output and dQ, dK and dV against their BF16 rounding floors (the reference rounded to BF16), and
every FP32 gradient (coefficients, gate inputs and rates, shift operands) order by order, so that no
order's error hides behind the others'.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import score_bias
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.core import attention_core

pytestmark = pytest.mark.gpu
DEGREE, TIME, HEADS, WIDTH, HIDDEN = 8, 2048, 2, 64, 64
# G_r = 1 (the RoPE control's backbone), or each order's query-key endpoint gate (the gated control's).
BACKBONES = ("rope", "per_order_gate")
MODES = ("static", "query", "query_key")
SHIFT_NAMES = {"row": (score_bias.ROW_SHIFT,),
               "pair": (score_bias.QUERY_INPUT, score_bias.KEY_INPUT, score_bias.OFFSET, score_bias.AMPLITUDE)}
GATE_NAMES = ("query_gate_input", "key_gate_input", "gate_log_rate")
BF16_LEAVES = ("output", "dq", "dk", "dv")
# Bounds. BF16 leaves: relative L2 within FLOOR_FACTOR of the BF16 rounding floor, and no element further
# from the reference than MAX_FACTOR times BF16's unit roundoff (2^-8) times the largest reference
# magnitude. FP32 gradients: relative L2 below SUM_TOLERANCE in every order. The measured margins are in
# validation/README.md.
FLOOR_FACTOR = 1.2
MAX_FACTOR = 2.0
SUM_TOLERANCE = 1e-4


@pytest.fixture(autouse=True)
def _requires_a_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation (B200, B300 or H100)")


def _kind(mode):
    return "pair" if mode == "query_key" else "row"


def _leaves(backbone, kind):
    """(name, order axis, first order) of the output and of each gradient, in the operation's argument order."""
    names = [("output", None, 0), ("dq", None, 0), ("dk", None, 0), ("dv", None, 0), ("dc", -1, 0)]
    if backbone == "per_order_gate":
        names += [("dgate_query_input", -1, 0), ("dgate_key_input", -1, 0), ("dgate_log_rate", 1, 0)]
    names += ([("drow_shift", -1, 1)] if kind == "row" else
              [("dshift_query_input", -1, 1), ("dshift_key_input", -1, 1), ("doffset", -1, 1),
               ("damplitude", -1, 1)])
    return names


def _base_inputs(backbone, rng, *, integer_qk=False):
    """FP32 Q, K, V before their BF16 rounding, query coefficients, and the gate operands. Unit-scale
    scores (QK RMSNorm's scale) so that every order up to eight contributes; signed coefficients
    N(0, 1)/r!; per-order gate rates exp(A_r) softplus(u_r + w_r) from about 3e-4 to 2 (half-lives from
    under one token to the context). ``integer_qk`` draws Q and K in {-1, 0, 1}, so every score is an exact
    multiple of 1/8."""
    shape = (1, TIME, HEADS)
    if integer_qk:
        q, k = (rng.integers(-1, 2, size=(*shape, WIDTH)).astype(np.float32) for _ in range(2))
    else:
        # Not BF16-representable: the program rounds them.
        q, k = (rng.standard_normal((*shape, WIDTH)).astype(np.float32) for _ in range(2))
    v = rng.standard_normal((*shape, WIDTH)).astype(np.float32)
    orders = DEGREE + 1
    c = rng.standard_normal((*shape, orders)) / np.array([math.factorial(r) for r in range(orders)])
    gate = ()
    if backbone == "per_order_gate":
        gate = (rng.uniform(-4, -1, (*shape, orders)), .5 * rng.standard_normal((*shape, orders)),
                rng.uniform(-4, 2, (HEADS, orders, 1)))
    return [jnp.asarray(x, jnp.float32) for x in (q, k, v, c, *gate)]


def _mode_inputs(backbone, mode, *, seed):
    """Inputs with the mode's shift operands computed as the model computes them
    (``score_bias.attention_operands``) from an attention input x [1, T, D] of unit RMS and nonzero b, a,
    u, v: shifts of order one (x u and x v ~ N(0, 1))."""
    rng = np.random.default_rng(seed)
    base = _base_inputs(backbone, rng)
    x = jnp.asarray(rng.standard_normal((1, TIME, HIDDEN)), jnp.float32)
    parameters = {score_bias.OFFSET: .7 * rng.standard_normal((HEADS, DEGREE)),
                  score_bias.AMPLITUDE: 1 + .3 * rng.standard_normal((HEADS, DEGREE)),
                  score_bias.QUERY_KERNEL: rng.standard_normal((HEADS, DEGREE, HIDDEN)) / math.sqrt(HIDDEN),
                  score_bias.KEY_KERNEL: rng.standard_normal((HEADS, DEGREE, HIDDEN)) / math.sqrt(HIDDEN)}
    parameters = {name: jnp.asarray(parameters[name], jnp.float32) for name in score_bias.parameter_names(mode)}
    operands = jax.jit(partial(score_bias.attention_operands, mode))(x, parameters)
    shift = [operands[name] for name in SHIFT_NAMES[_kind(mode)]]
    cotangent = jnp.asarray(rng.standard_normal(base[2].shape), jnp.bfloat16)
    return [*base, *shift], cotangent


def _exact_zero_inputs(backbone, kind, *, seed):
    """Integer Q and K, so every score is exact, and shifts that make many shifted scores exactly zero and
    many negative. Row shifts: beta_r(t) = -s(t, i_r) for a random causal key i_r, zero on every key that
    shares i_r's score. Pair shifts: odd orders r have zero inputs (tanh(0) = 0) and offset b_r = -(r - 1)/8,
    zero where s = (r - 1)/8; even orders N(0, 1) inputs and offsets about -1."""
    rng = np.random.default_rng(seed)
    base = _base_inputs(backbone, rng, integer_qk=True)
    shape = (1, TIME, HEADS, DEGREE)
    if kind == "row":
        scores = np.einsum("bthd,bihd->bthi", np.asarray(base[0], np.float64), np.asarray(base[1], np.float64)) / 8
        chosen = rng.integers(0, np.arange(TIME)[None, :, None, None] + 1, size=shape)
        shift = [-np.take_along_axis(scores, chosen, axis=-1)]
    else:
        odd = np.arange(1, DEGREE + 1) % 2 == 1
        p, key = (rng.standard_normal(shape) * ~odd for _ in range(2))
        offset = np.where(odd, -(np.arange(1, DEGREE + 1) - 1) / 8, -1 + .5 * rng.standard_normal((HEADS, DEGREE)))
        shift = [p, key, offset, 1 + .3 * rng.standard_normal((HEADS, DEGREE))]
    cotangent = jnp.asarray(rng.standard_normal(base[2].shape), jnp.bfloat16)
    return [*base, *(jnp.asarray(x, jnp.float32) for x in shift)], cotangent


def _split(backbone, kind, operands):
    """(q, k, v, c), the gate operands, and the shift operands."""
    gates = 3 if backbone == "per_order_gate" else 0
    return operands[:4], operands[4:4 + gates], operands[4 + gates:4 + gates + len(SHIFT_NAMES[kind])]


def _strips(backbone, kind):
    """The production operation: Q, K, V rounded to BF16, then attention_core on the blocked strips."""
    def attend(*operands):
        (q, k, v, c), gate, shift = _split(backbone, kind, operands)
        keywords = dict(zip(SHIFT_NAMES[kind], shift, strict=True))
        if backbone == "per_order_gate":
            keywords.update(zip(GATE_NAMES, gate, strict=True), query_decay_mode="query_distance")
        return attention_core(*(x.astype(jnp.bfloat16) for x in (q, k, v)), backend="blocked",
                              blocked_implementation="strips", attention_degree=DEGREE,
                              attention_normalization=False, query_polynomial_coefficients=c,
                              attention_coefficient_activation="tanh", **keywords)
    return attend


def _dense(backbone, kind, q, k, v, c, *operands):
    """Independent all-pairs reference: W(t,i) = c_0 G_0 + sum_(r=1..m) c_r G_r (s_ti + beta_r(t,i))^r,
    G_r = exp(-(t - i) exp(A_r) softplus(u_r(t) + w_r(i))) or 1, every power explicit."""
    _, gate, shift = _split(backbone, kind, (q, k, v, c, *operands))
    distance = jnp.arange(TIME)[:, None] - jnp.arange(TIME)[None, :]
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    rows = lambda x: jnp.swapaxes(x, 1, 2)[..., None]  # [b, t, h] -> [b, h, t, 1]
    columns = lambda x: jnp.swapaxes(x, 1, 2)[:, :, None, :]  # [b, i, h] -> [b, h, 1, i]
    weights = 0.
    for r in range(DEGREE + 1):
        term = rows(c[..., r])
        if gate:
            query_input, key_input, log_rate = gate
            rate = jnp.exp(log_rate[:, r, 0])[None, :, None, None] * jnp.logaddexp(
                0., rows(query_input[..., r]) + columns(key_input[..., r]))
            term = term * jnp.exp(-jnp.maximum(distance, 0) * rate)
        if r:
            if kind == "row":
                beta = rows(shift[0][..., r - 1])
            else:
                p, key, offset, amplitude = shift
                beta = offset[None, :, None, None, r - 1] + amplitude[None, :, None, None, r - 1] * jnp.tanh(
                    rows(p[..., r - 1]) + columns(key[..., r - 1]))
            term = term * (scores + beta) ** r
        weights = weights + term
    return jnp.einsum("bhti,bihd->bthd", jnp.where(distance >= 0, weights, 0.), v)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _float64(backbone, kind, operands, cotangent):
    """The reference's output and VJP in FP64 on the host CPU, independent of the GPU compiler, from the
    exact inputs the strips see: Q, K, V rounded to BF16 and the FP32 operands."""
    cpu = jax.devices("cpu")[0]
    rounded = [np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)) for x in operands[:3]]
    values = [*rounded, *(np.asarray(x) for x in operands[3:])]
    with jax.enable_x64(), jax.default_matmul_precision("highest"), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(x, np.float64), cpu) for x in values]
        g = jax.device_put(np.asarray(cotangent.astype(jnp.float32), np.float64), cpu)
        output, gradients = jax.jit(partial(_vjp, partial(_dense, backbone, kind)))(exact, g)
        return [np.asarray(output), *(np.asarray(x) for x in gradients)]


def _run(backbone, kind, operands, cotangent, monkeypatch):
    """The strips' output and gradients on the GPU (the strips counted), and the FP64 reference's."""
    calls = []
    strips = reference.strip_dot_product_attention

    def counted(*args, **kwargs):
        calls.append(True)
        return strips(*args, **kwargs)

    monkeypatch.setattr(reference, "strip_dot_product_attention", counted)
    output, gradients = jax.jit(partial(_vjp, _strips(backbone, kind)))(operands, cotangent)
    assert calls == [True], "attention_core must reach the causal strips"
    assert output.dtype == jnp.bfloat16 and all(g.dtype == jnp.float32 for g in gradients)
    actual = [np.asarray(output.astype(jnp.float32)), *(np.asarray(x) for x in gradients)]
    return actual, _float64(backbone, kind, operands, cotangent)


def _relative_l2(got, want):
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _compare(backbone, kind, actual, expected, label):
    """Every leaf against its bound (each order of the order-indexed ones separately); an order whose
    reference is exactly zero must be exactly zero. Returns each leaf's measured error: the multiple of
    its BF16 floor for the BF16 leaves, the largest order's relative L2 for the FP32 gradients."""
    measured = {}
    for (name, axis, first), got, want in zip(_leaves(backbone, kind), actual, expected, strict=True):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        assert got.shape == want.shape and np.isfinite(got).all(), (label, name)
        if name in BF16_LEAVES:
            # The strips return BF16 output and dQ, dK, dV (cotangents of the BF16 rounding of Q, K, V).
            assert np.array_equal(got, np.asarray(jnp.asarray(got, jnp.bfloat16).astype(jnp.float32))), (label, name)
            rounded = np.asarray(jnp.asarray(want, jnp.bfloat16).astype(jnp.float32), np.float64)
            floor = _relative_l2(rounded, want)
            ratio = _relative_l2(got, want) / floor
            assert ratio < FLOOR_FACTOR, f"{label} {name}: {ratio:.3f} times the BF16 floor {floor:.3e}"
            worst = np.max(np.abs(got - want)) / (2 ** -8 * np.max(np.abs(want)))
            assert worst < MAX_FACTOR, f"{label} {name}: largest error {worst:.3f} BF16 roundoffs of the largest value"
            measured[name] = (ratio, worst)
            continue
        worst = 0.
        for order in range(want.shape[axis]):
            part, reference_part = np.take(got, order, axis), np.take(want, order, axis)
            if not np.any(reference_part):
                np.testing.assert_array_equal(part, 0, err_msg=f"{label} {name} order {first + order}")
                continue
            error = _relative_l2(part, reference_part)
            assert error < SUM_TOLERANCE, f"{label} {name} order {first + order}: relative L2 {error:.3e}"
            worst = max(worst, error)
        measured[name] = worst
    print(f"\n{label}: " + " ".join(f"{name}={value[0]:.3f}x/{value[1]:.2f}" if isinstance(value, tuple)
                                      else f"{name}={value:.2e}" for name, value in measured.items()))
    return measured


def _shifted_scores(kind, operands):
    """z_r = s + beta_r on every causal pair [pairs, m], in FP64 from the rounded Q and K."""
    q, k = (np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32), np.float64) for x in operands[:2])
    shift = [np.asarray(x, np.float64) for x in operands[-len(SHIFT_NAMES[kind]):]]
    scores = np.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    if kind == "row":
        beta = np.swapaxes(shift[0], 1, 2)[:, :, :, None, :]
    else:
        p, key, offset, amplitude = shift
        beta = offset[None, :, None, None] + amplitude[None, :, None, None] * np.tanh(
            np.swapaxes(p, 1, 2)[:, :, :, None] + np.swapaxes(key, 1, 2)[:, :, None])
    return (scores[..., None] + beta)[:, :, np.tril(np.ones((TIME, TIME), bool))]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backbone", BACKBONES)
def test_strips_match_float64_for_every_mode(backbone, mode, monkeypatch):
    """Each mode's shift operands as the model forms them from nonzero b, a, u, v: the output and dQ, dK,
    dV within 1.2x their BF16 rounding floors (no element off by more than 2 x 2^-8 of the largest value), and
    the FP32 coefficient, gate and shift gradients within 1e-4 relative L2 in every order."""
    operands, cotangent = _mode_inputs(backbone, mode, seed=3 + 10 * MODES.index(mode) + BACKBONES.index(backbone))
    shifted = _shifted_scores(_kind(mode), operands)
    assert (shifted < 0).mean() > .2 and (shifted > 0).mean() > .2
    actual, expected = _run(backbone, _kind(mode), operands, cotangent, monkeypatch)
    _compare(backbone, _kind(mode), actual, expected, f"{backbone} {mode}")


@pytest.mark.parametrize("kind", ("row", "pair"))
@pytest.mark.parametrize("backbone", BACKBONES)
def test_negative_and_exactly_zero_shifted_scores_match_float64(backbone, kind, monkeypatch):
    """Shifted scores exactly zero on several percent of the pairs (z^(r-1) is formed by the power itself,
    never by dividing by z) and negative on more than a third, under the same bounds."""
    operands, cotangent = _exact_zero_inputs(backbone, kind, seed=41 + 2 * BACKBONES.index(backbone)
                                             + (kind == "pair"))
    shifted = _shifted_scores(kind, operands)
    zero, negative = (shifted == 0).mean(), (shifted < 0).mean()
    assert zero > .02 and negative > .35, (zero, negative)
    actual, expected = _run(backbone, kind, operands, cotangent, monkeypatch)
    _compare(backbone, kind, actual, expected,
             f"{backbone} {kind} exact zeros ({zero:.3f} zero, {negative:.3f} negative)")
