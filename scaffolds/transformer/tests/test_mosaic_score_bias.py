"""GPU qualification of the H100 Mosaic per-degree score-bias kernels (``mosaic/score_bias.py``).

Use ``pytest -m gpu tests/test_mosaic_score_bias.py`` inside an H100 allocation. Every kernel
family (no gate or the per-order endpoint gate, times a row or a pair shift) is compared with an
independent dense FP64 reference written here (explicit (s + beta_r)^r over all pairs, JAX FP64
autodiff), order by order for the coefficient, gate, and shift gradients, with unit-scale scores,
shifts of order one, many negative and exactly zero shifted scores; against the existing query and
per-order endpoint kernels at zero shift; for bitwise causality, poisoning, and repeatability; and
for refusals. On B300 every entry point must refuse before tracing a kernel.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api
from scaffolds.transformer.attention.backends.mosaic.score_bias import (
    score_bias_per_order_endpoint_attention, score_bias_query_polynomial_attention,
)

pytestmark = pytest.mark.gpu
FAMILIES = (("none", "row"), ("none", "pair"), ("endpoint", "row"), ("endpoint", "pair"))
FAMILY_IDS = ("row", "pair", "endpoint-row", "endpoint-pair")
# Relative L2 bounds of test_per_order_decay: output, dq, dk, dv at the Mosaic BF16 bounds; every
# FP32 sum (coefficient, gate, and shift gradients) at .01, each order separately.
TOLERANCES = {"out": .006, "dq": .025, "dk": .025, "dv": .015}
SUM_TOLERANCE = .01
ZERO_SHIFT_TOLERANCE = 2e-4  # relative L2 against the existing kernels at zero shift (cf. the zero-key checks)


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _hopper():
    return jax.devices()[0].compute_capability == "9.0"


def _names(gate, shift):
    """Names of the operands after (q, k, v, c), with each one's order axis and first order."""
    names = [("u", -1, 0), ("w", -1, 0), ("log_rate", 1, 0)] if gate == "endpoint" else []
    names += [("beta", -1, 1)] if shift == "row" else [("p", -1, 1), ("qs", -1, 1), ("b", -1, 1), ("a", -1, 1)]
    return names


def _inputs(gate, shift, degree, time, *, seed=0, batch=1, heads=2, top=False, zero_shift=False, exact_zero=False,
            offset_mean=0.):
    """(q, k, v, c, gate operands..., shift operands...) and a BF16 cotangent. Unit-scale scores
    (QK RMSNorm's scale) so that every order up to the degree contributes. ``top`` keeps only the
    four highest orders' coefficients, which then dominate the output and the q, k, v gradients
    (the forward and dK/dV kernels compute them). ``exact_zero`` draws integer-valued BF16 q and k
    in {-1, 0, 1}, so every score is an exact multiple of 1/8, and sets beta_r(t) = -s(t, i_r) for a
    random causal key i_r: every key with that score has a shifted score of exactly zero, and
    about half the others are negative."""
    rng = np.random.default_rng(seed)
    orders = degree + 1
    shape = (batch, time, heads)
    if exact_zero:
        q, k = (rng.integers(-1, 2, size=(*shape, 64)).astype(np.float64) for _ in range(2))
    else:
        q, k = (rng.standard_normal((*shape, 64)) for _ in range(2))
    q, k = (jnp.asarray(x, jnp.bfloat16) for x in (q, k))
    v = jnp.asarray(rng.standard_normal((*shape, 64)), jnp.bfloat16)
    factorial = np.array([math.factorial(r) for r in range(orders)])
    c = rng.standard_normal((*shape, orders)) / factorial
    if top:
        c = c * (np.arange(orders) > degree - 4)
    operands = [c]
    if gate == "endpoint":
        operands += [rng.uniform(-4, -1, (*shape, orders)), .5 * rng.standard_normal((*shape, orders)),
                     rng.uniform(-1, 1, (heads, orders, 1))]
    if shift == "row":
        beta = rng.standard_normal((*shape, degree))
        if exact_zero:
            s = np.einsum("bthd,bihd->bhti", np.asarray(q, np.float64), np.asarray(k, np.float64)) / 8
            chosen = rng.integers(0, np.arange(time)[None, :, None, None] + 1, size=(*shape, degree))
            beta = -np.take_along_axis(s.transpose(0, 2, 1, 3), chosen, axis=-1)  # [b, t, h, r]
        operands.append(beta * (not zero_shift))
    else:
        p, qs = (rng.standard_normal((*shape, degree)) for _ in range(2))
        b, a = offset_mean + .5 * rng.standard_normal((heads, degree)), 1 + .3 * rng.standard_normal((heads, degree))
        if zero_shift:
            p, qs, b, a = 0 * p, 0 * qs, 0 * b, np.ones_like(a)
        operands += [p, qs, b, a]
    cotangent = jnp.asarray(rng.standard_normal(v.shape), jnp.bfloat16)
    return (q, k, v, *(jnp.asarray(x, jnp.float32) for x in operands)), cotangent


def _mosaic(gate, shift, precision=None):
    """The production operation over (q, k, v, c, gate operands..., shift operands...)."""
    def attend(q, k, v, c, *operands):
        gate_operands = operands[:3] if gate == "endpoint" else ()
        values = operands[len(gate_operands):]
        names = ("row_shift",) if shift == "row" else (
            "query_shift_input", "key_shift_input", "shift_offset", "shift_amplitude")
        kwargs = dict(zip(names, values), precision=precision)
        if gate == "endpoint":
            return score_bias_per_order_endpoint_attention(q, k, v, c, *gate_operands, **kwargs)
        return score_bias_query_polynomial_attention(q, k, v, c, **kwargs)
    return attend


def _dense(gate, shift, q, k, v, c, *operands):
    """Independent all-pairs reference: W = sum_r c_r G_r (s + beta_r)^r with explicit powers."""
    time = q.shape[1]
    s = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    distance = jnp.arange(time)[:, None] - jnp.arange(time)[None, :]
    rows = lambda x: jnp.swapaxes(x, 1, 2)[..., None]  # [b, t, h] -> [b, h, t, 1]
    cols = lambda x: jnp.swapaxes(x, 1, 2)[:, :, None, :]  # [b, i, h] -> [b, h, 1, i]
    if gate == "endpoint":
        u, w, log_rate, *operands = operands
    weights = 0.
    for r in range(c.shape[-1]):
        term = rows(c[..., r])
        if gate == "endpoint":
            rate = jnp.exp(log_rate[:, r, 0])[None, :, None, None] * jnp.logaddexp(0., rows(u[..., r]) + cols(w[..., r]))
            term = term * jnp.exp(-jnp.maximum(distance, 0) * rate)
        if r:
            if shift == "row":
                beta = rows(operands[0][..., r - 1])
            else:
                p, qs, b, a = operands
                head = lambda x: x[:, r - 1][None, :, None, None]
                beta = head(b) + head(a) * jnp.tanh(rows(p[..., r - 1]) + cols(qs[..., r - 1]))
            term = term * (s + beta) ** r
        weights = weights + term
    weights = jnp.where(distance >= 0, weights, 0.)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _float64(gate, shift, operands, cotangent):
    """The reference's output and VJP in FP64 on the host CPU (independent of the GPU compilers,
    whose layout normalization rejects this program in FP64), from the exact BF16/FP32 inputs."""
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(), jax.default_matmul_precision("highest"), jax.default_device(cpu):
        exact = lambda x: jax.device_put(np.asarray(x.astype(jnp.float32), np.float64), cpu)
        result = jax.jit(partial(_vjp, partial(_dense, gate, shift)))(tuple(map(exact, operands)), exact(cotangent))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _leaves(gate, shift, result):
    """[(name, array, order axis or None, first order)] of an operation's output and gradients."""
    output, grads = result
    named = [("out", output, None, 0), ("dq", grads[0], None, 0), ("dk", grads[1], None, 0), ("dv", grads[2], None, 0),
             ("dc", grads[3], -1, 0)]
    named += [(f"d{name}", grad, axis, first) for (name, axis, first), grad in zip(_names(gate, shift), grads[4:],
                                                                                  strict=True)]
    return [(name, np.asarray(x, np.float64), axis, first) for name, x, axis, first in named]


def _compare(gate, shift, actual, expected, *, label=""):
    """Relative L2 errors of every leaf (and of every order of the order-indexed ones) against
    their bounds; an order whose reference is exactly zero must be exactly zero. Returns the
    largest error of each leaf for the record."""
    measured = {}
    for (name, got, axis, first), (_, want, _, _) in zip(_leaves(gate, shift, actual), _leaves(gate, shift, expected),
                                                         strict=True):
        assert got.shape == want.shape and np.isfinite(got).all(), (label, name)
        tolerance = TOLERANCES.get(name, SUM_TOLERANCE)
        parts = ([(None, got, want)] if axis is None else
                 [(first + r, np.take(got, r, axis), np.take(want, r, axis)) for r in range(want.shape[axis])])
        worst = 0.
        for order, part, reference in parts:
            norm = np.linalg.norm(reference)
            if norm == 0:  # a zeroed order's shift and gate gradients
                np.testing.assert_array_equal(part, 0, err_msg=f"{label} {name} order {order}")
                continue
            error = np.linalg.norm(part - reference) / norm
            assert error < tolerance, (label, name, order, error)
            worst = max(worst, error)
        if axis is None:  # L2 can hide an isolated bad row: bound the worst element too
            assert np.max(np.abs(got - want)) <= 3 * tolerance * np.max(np.abs(want)), (label, name)
        measured[name] = worst
    return measured


def _refuses_here():
    """B300 has no score-bias kernel: its entry points raise before tracing one."""
    return pytest.raises(ValueError, match="H100")


CASES = (pytest.param(512, 4, False, id="T512-m4"), pytest.param(2048, 8, False, id="T2048-m8"),
         pytest.param(2048, 8, True, id="T2048-m8-top"))


@pytest.mark.parametrize("time,degree,top", CASES)
@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_output_and_every_gradient_match_float64(gate, shift, time, degree, top):
    """Shifts of order one (rows: N(0, 1); pairs: p, q ~ N(0, 1), b ~ N(0, .25), a ~ 1 + N(0, .09)).
    With ``top`` only orders m-3..m have coefficients: the zeroed orders' shift and gate gradients
    are then exactly zero, and the top orders' errors are not hidden by the others in the output
    and the q, k, v gradients."""
    operands, cotangent = _inputs(gate, shift, degree, time, seed=degree + time, top=top)
    attend = jax.jit(partial(_vjp, _mosaic(gate, shift)))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    actual = attend(operands, cotangent)
    assert actual[0].dtype == jnp.bfloat16 and all(g.dtype == x.dtype for g, x in zip(actual[1], operands))
    measured = _compare(gate, shift, actual, _float64(gate, shift, operands, cotangent),
                        label=f"{gate}-{shift} T={time} m={degree}{' top' if top else ''}")
    print(f"\n{gate}-{shift} T={time} m={degree}{' top' if top else ''}:",
          " ".join(f"{name}={error:.2e}" for name, error in measured.items()))


@pytest.mark.parametrize("precision", ("bf16x3",))
@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_fp32_precision_weights_match_float64(gate, shift, precision):
    """Weights and score gradients as three BF16 components (the kernels' other precision): the
    same bounds, degree four, 512 tokens."""
    operands, cotangent = _inputs(gate, shift, 4, 512, seed=7)
    attend = jax.jit(partial(_vjp, _mosaic(gate, shift, precision=precision)))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    measured = _compare(gate, shift, attend(operands, cotangent), _float64(gate, shift, operands, cotangent),
                        label=f"{gate}-{shift} {precision}")
    print(f"\n{gate}-{shift} {precision}:", " ".join(f"{name}={error:.2e}" for name, error in measured.items()))


@pytest.mark.parametrize("gate", ("none", "endpoint"))
def test_negative_and_exactly_zero_shifted_scores_match_float64(gate):
    """Integer-valued q, k make every score exact; beta_r(t) = -s(t, i_r) then gives exactly zero
    shifted scores on every key that shares i_r's score (z^(r-1) is formed without dividing by z),
    and about half the others negative. Degree eight, 2,048 tokens."""
    degree, time = 8, 2048
    operands, cotangent = _inputs(gate, "row", degree, time, seed=11, exact_zero=True)
    q, k, beta = (np.asarray(x, np.float64) for x in (operands[0], operands[1], operands[-1]))
    z = (np.einsum("bthd,bihd->bhti", q, k) / 8)[..., None] + beta.transpose(0, 2, 1, 3)[:, :, :, None, :]
    causal = np.tril(np.ones((time, time), bool))[None, None, :, :, None]
    pairs = causal.sum() * z.shape[0] * z.shape[1] * degree
    zero, negative = np.sum((z == 0) & causal) / pairs, np.sum((z < 0) & causal) / pairs
    assert zero > .03 and negative > .3, (zero, negative)
    attend = jax.jit(partial(_vjp, _mosaic(gate, "row")))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    measured = _compare(gate, "row", attend(operands, cotangent), _float64(gate, "row", operands, cotangent),
                        label=f"{gate}-row exact zeros")
    print(f"\n{gate}-row exact zeros ({zero:.3f} of shifted scores zero, {negative:.3f} negative):",
          " ".join(f"{name}={error:.2e}" for name, error in measured.items()))


@pytest.mark.parametrize("gate", ("none", "endpoint"))
def test_pair_shift_with_negative_offsets_matches_float64(gate):
    """Offsets near -1.5 make most pair-shifted scores negative; degree eight, 512 tokens."""
    degree, time = 8, 512
    operands, cotangent = _inputs(gate, "pair", degree, time, seed=13, offset_mean=-1.5)
    attend = jax.jit(partial(_vjp, _mosaic(gate, "pair")))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    measured = _compare(gate, "pair", attend(operands, cotangent), _float64(gate, "pair", operands, cotangent),
                        label=f"{gate}-pair negative offsets")
    print(f"\n{gate}-pair negative offsets:", " ".join(f"{name}={error:.2e}" for name, error in measured.items()))


def _difference(got, want):
    """(largest difference relative to the largest reference magnitude, relative L2 difference)."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    norm, scale = np.linalg.norm(want), np.max(np.abs(want))
    return (np.max(np.abs(got - want)) / scale if scale else float(np.any(got != want)),
            np.linalg.norm(got - want) / norm if norm else float(np.any(got != want)))


@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_zero_shift_reproduces_the_existing_kernels(gate, shift):
    """b = 0, a = 1, p = q = 0 (the runs' initialization) or a zero row shift: the ungated
    families reproduce ``api.query_polynomial_attention`` and the gated ones
    ``api.per_order_endpoint_decay_attention`` (output and every shared gradient), within
    finite-precision differences (explicit powers against Horner); degree eight, 2,048 tokens."""
    degree, time = 8, 2048
    operands, cotangent = _inputs(gate, shift, degree, time, seed=17, zero_shift=True)
    attend = jax.jit(partial(_vjp, _mosaic(gate, shift)))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    shifted = attend(operands, cotangent)
    shared = 4 + (3 if gate == "endpoint" else 0)  # q, k, v, c and the gate operands
    control = api.per_order_endpoint_decay_attention if gate == "endpoint" else api.query_polynomial_attention
    expected = jax.jit(partial(_vjp, control))(operands[:shared], cotangent)
    names = ("out", "dq", "dk", "dv", "dc", "du", "dw", "dlog_rate")
    report = []
    for name, got, want in zip(names, (shifted[0], *shifted[1][:shared]), (expected[0], *expected[1]), strict=False):
        largest, relative = _difference(got, want)
        assert np.isfinite(np.asarray(got, np.float32)).all() and relative <= ZERO_SHIFT_TOLERANCE, (name, relative)
        report.append(f"{name}: max/max|ref| {largest:.1e} L2 {relative:.1e}")
    print(f"\nzero shift {gate}-{shift} vs existing kernel:", "; ".join(report))


HEAD_OPERANDS = ("log_rate", "b", "a")  # per head: their gradients sum over every causal pair
PER_KEY_OPERANDS = ("k", "v", "w", "qs")  # per key token; the others are per query token


def _change_future(operands, gate, shift, prefix):
    """Every token input of positions >= prefix changed by finite amounts; head operands kept."""
    names = ["q", "k", "v", "c", *(name for name, _, _ in _names(gate, shift))]
    return tuple(x if name in HEAD_OPERANDS else x.at[:, prefix:].add(1. if name in ("q", "k", "v") else .5)
                 for name, x in zip(names, operands, strict=True))


@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_outputs_and_gradients_ignore_future_tokens_bitwise(gate, shift):
    """With the cotangent zero from position 300 on, changing every input of those positions leaves
    the earlier outputs and every gradient bitwise unchanged, and the later positions' gradients
    exactly zero."""
    degree, time, prefix = 4, 512, 300
    operands, cotangent = _inputs(gate, shift, degree, time, seed=19)
    attend = jax.jit(partial(_vjp, _mosaic(gate, shift)))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    restricted = cotangent.at[:, prefix:].set(0)
    output, grads = attend(operands, restricted)
    changed_output, changed_grads = attend(_change_future(operands, gate, shift, prefix), restricted)
    np.testing.assert_array_equal(np.asarray(changed_output[:, :prefix]), np.asarray(output[:, :prefix]))
    names = ["q", "k", "v", "c", *(name for name, _, _ in _names(gate, shift))]
    for name, grad, changed_grad in zip(names, grads, changed_grads, strict=True):
        np.testing.assert_array_equal(np.asarray(changed_grad), np.asarray(grad), err_msg=name)
        if name not in HEAD_OPERANDS:
            np.testing.assert_array_equal(np.asarray(grad[:, prefix:]), 0, err_msg=name)


def _poison_cases():
    """(family, operand name, position kind): query-side inputs poison their own row, key-side inputs
    every later row that attends to their key."""
    cases = [(("none", "row"), "beta", "query"), (("none", "pair"), "p", "query"), (("none", "pair"), "qs", "key"),
             (("endpoint", "row"), "beta", "query"), (("endpoint", "row"), "w", "key"),
             (("endpoint", "pair"), "p", "query"), (("endpoint", "pair"), "qs", "key")]
    return [pytest.param(family, name, kind, bad, id=f"{'-'.join(family)}-{name}-{'nan' if bad != bad else bad}")
            for family, name, kind in cases for bad in (float("nan"), float("inf"))]


@pytest.mark.parametrize("family,operand,kind,bad", _poison_cases())
def test_nonfinite_inputs_poison_only_their_causal_rows(family, operand, kind, bad):
    """A nonfinite shift (or gate) input of token 300, head 0 makes exactly the causal rows that use
    it NaN: its own row (query-side) or every row from 300 on (key-side). Every other output row,
    and, for the full cotangent, every gradient row that does not depend on it, is bitwise
    unchanged: summands of masked entries never reach a sum, in the dQ pass (key-side inputs of a
    tile's future keys) and the dK pass (query-side inputs of a transposed tile's earlier queries)."""
    gate, shift = family
    degree, time, position = 4, 512, 300
    operands, cotangent = _inputs(gate, shift, degree, time, seed=23)
    names = ["q", "k", "v", "c", *(name for name, _, _ in _names(gate, shift))]
    index = names.index(operand)
    poisoned = list(operands)
    poisoned[index] = poisoned[index].at[0, position, 0, 1].set(bad)
    attend = jax.jit(partial(_vjp, _mosaic(gate, shift)))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    clean, dirty = attend(operands, cotangent), attend(tuple(poisoned), cotangent)
    rows = np.zeros(time, bool)
    rows[position:position + 1 if kind == "query" else time] = True
    bad_output = ~np.isfinite(np.asarray(dirty[0], np.float32)).all(axis=-1)
    expected = np.zeros(bad_output.shape, bool)
    expected[0, rows, 0] = True
    np.testing.assert_array_equal(bad_output, expected)
    np.testing.assert_array_equal(np.asarray(dirty[0])[~expected], np.asarray(clean[0])[~expected])
    # Gradient rows that cannot depend on the input: per-query gradients of every other row (query
    # side) or of the earlier rows (key side); per-key gradients of the later keys (query side: a
    # query attends only to earlier keys) or of every other key (key side). Head 1 is untouched.
    positions = np.arange(time)
    for name, got, want in zip(names, dirty[1], clean[1], strict=True):
        if name in HEAD_OPERANDS:  # sums over every pair, the poisoned ones included
            continue
        if kind == "query":
            keep = positions > position if name in PER_KEY_OPERANDS else positions != position
        else:
            keep = positions != position if name in PER_KEY_OPERANDS else positions < position
        np.testing.assert_array_equal(np.asarray(got)[0, keep, 0], np.asarray(want)[0, keep, 0], err_msg=name)
        np.testing.assert_array_equal(np.asarray(got)[:, :, 1], np.asarray(want)[:, :, 1], err_msg=name)


@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_production_shape_is_bitwise_repeatable(gate, shift):
    """16 sequences of 2,048 tokens, 14 heads, d_K = 64, degree eight: twenty forward and backward
    evaluations give bitwise identical outputs and gradients (the per-order sums are read, added,
    and written back in shared memory, by every lane that holds a row)."""
    operands, cotangent = _inputs(gate, shift, 8, 2048, seed=29, batch=16, heads=14)
    attend = jax.jit(partial(_vjp, _mosaic(gate, shift)))
    if not _hopper():
        with _refuses_here():
            attend(operands, cotangent)
        return
    first = jax.tree.map(np.asarray, attend(operands, cotangent))
    assert all(np.isfinite(np.asarray(x, np.float32)).all() for x in jax.tree.leaves(first))
    for _ in range(19):
        again = jax.tree.map(np.asarray, attend(operands, cotangent))
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(again), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("change", ("both shifts", "no shift", "row dtype", "row shape", "pair offset shape",
                                    "pair input dtype", "degree", "width", "time", "precision", "gate shape"))
def test_invalid_inputs_raise(change):
    gate = "endpoint" if change == "gate shape" else "none"
    operands, _ = _inputs(gate, "row", 4, 512, seed=31, heads=1)
    q, k, v, c, *rest = operands
    beta = rest[-1]
    pair = _inputs("none", "pair", 4, 512, seed=31, heads=1)[0][4:]
    kwargs = dict(row_shift=beta)
    if change == "both shifts":
        kwargs.update(zip(("query_shift_input", "key_shift_input", "shift_offset", "shift_amplitude"), pair))
    elif change == "no shift":
        kwargs = {}
    elif change == "row dtype":
        kwargs["row_shift"] = beta.astype(jnp.bfloat16)
    elif change == "row shape":
        kwargs["row_shift"] = beta[..., :-1]
    elif change == "pair offset shape":
        kwargs = dict(zip(("query_shift_input", "key_shift_input", "shift_offset", "shift_amplitude"), pair))
        kwargs["shift_offset"] = kwargs["shift_offset"][:, :-1]
    elif change == "pair input dtype":
        kwargs = dict(zip(("query_shift_input", "key_shift_input", "shift_offset", "shift_amplitude"), pair))
        kwargs["query_shift_input"] = kwargs["query_shift_input"].astype(jnp.bfloat16)
    elif change == "degree":
        c = jnp.zeros((*c.shape[:3], api.MAX_QUERY_DEGREE + 2), jnp.float32)
        kwargs["row_shift"] = jnp.zeros((*c.shape[:3], api.MAX_QUERY_DEGREE + 1), jnp.float32)
    elif change == "width":
        q = k = jnp.zeros((*q.shape[:3], 48), jnp.bfloat16)
    elif change == "time":
        q, k, v, c, kwargs["row_shift"] = (x[:, :200] for x in (q, k, v, c, beta))
    if change == "precision":
        kwargs["precision"] = "fp8"
    with pytest.raises(ValueError):
        if gate == "endpoint":
            u, w, log_rate = rest[:3]
            score_bias_per_order_endpoint_attention(q, k, v, c, u[..., :-1], w, log_rate, **kwargs)
        else:
            score_bias_query_polynomial_attention(q, k, v, c, **kwargs)


@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_b300_refuses_before_tracing_a_kernel(gate, shift, monkeypatch):
    """On B300 (Blackwell kernels) every entry point raises before any kernel is reached: no
    fallback. Runs on H100 by presenting the Blackwell kernel set without its kernels."""
    monkeypatch.setattr(api, "_kernels", lambda: ("blackwell", None))
    operands, cotangent = _inputs(gate, shift, 8, 2048, seed=37)
    with pytest.raises(ValueError, match="H100 only"):
        jax.jit(partial(_vjp, _mosaic(gate, shift)))(operands, cotangent)
