"""B-spline score functions (attention/spline.py) against independent references.

The references never use the scaffold's basis code: a NumPy float64 Cox-de Boor recursion
(continued outside the end knots by end-piece polynomials interpolated from it), and a dense
float64 JAX attention that evaluates the spline by de Boor's algorithm on each score's span.
"""
from fractions import Fraction
from functools import partial
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer.attention import spline
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention, strip_dot_product_attention

KNOTS = (-32.0, -6.0, 0.0, 6.0, 32.0)
DEGREE = 4
QUARTIC_INIT = tuple(Fraction(*value) for value in (
    (116131, 3), (38029, 6), (-931, 6), (38, 1), (-42, 1), (2351, 6), (63935, 6), (49697, 1)))
CUBIC_INIT = tuple(Fraction(*value) for value in (
    (-14941, 3), (-2435, 3), (61, 3), (-5, 1), (137, 3), (1283, 1), (18019, 3)))
PROJECTION_STD = 0.03340765523905305  # 1/sqrt(896), the query control's draw
SPECIAL = np.asarray([-50, -32 - 1e-9, -32, -32 + 1e-9, -6 - 1e-9, -6, -6 + 1e-9, -1e-9, 0, 1e-9,
                      6 - 1e-9, 6, 6 + 1e-9, 32 - 1e-9, 32, 32 + 1e-9, 50], np.float64)


def knot_vector(knots=KNOTS, degree=DEGREE):
    return np.asarray([knots[0]] * (degree + 1) + list(knots[1:-1]) + [knots[-1]] * (degree + 1), np.float64)


def cox_de_boor(points, knots=KNOTS, degree=DEGREE):
    """[degree + 1][functions, points]: every degree's B-splines by the Cox-de Boor recursion on the
    clamped knot vector, half-open spans with the last span closed at the end knot; zero outside."""
    t = knot_vector(knots, degree)
    s = np.asarray(points, np.float64).ravel()
    table = [np.stack([((t[i] <= s) & ((s < t[i + 1]) | ((t[i + 1] == t[-1]) & (s == t[-1]))) & (t[i] < t[i + 1]))
                       .astype(np.float64) for i in range(len(t) - 1)])]
    for order in range(1, degree + 1):
        previous, current = table[-1], np.zeros((len(t) - 1 - order, s.size))
        for i in range(len(t) - 1 - order):
            if t[i + order] > t[i]:
                current[i] += (s - t[i]) / (t[i + order] - t[i]) * previous[i]
            if t[i + order + 1] > t[i + 1]:
                current[i] += (t[i + order + 1] - s) / (t[i + order + 1] - t[i + 1]) * previous[i + 1]
        table.append(current)
    return table


def end_pieces(side, knots=KNOTS, degree=DEGREE):
    """Each B_i's first (side 0) or last (side -1) piece: the polynomial interpolating Cox-de Boor at
    degree + 1 Chebyshev points of that interval (exact for a degree-p piece)."""
    left, right = (knots[0], knots[1]) if side == 0 else (knots[-2], knots[-1])
    nodes = (left + right) / 2 + (right - left) / 2 * np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    values = cox_de_boor(nodes, knots, degree)[-1]
    return [np.polynomial.Polynomial.fit(nodes, row, degree, domain=[left, right]) for row in values]


def reference_basis(points, knots=KNOTS, degree=DEGREE, derivative=False):
    """[functions, points] B_i(s) (or B_i'(s)): Cox-de Boor on [u_0, u_K], end pieces outside."""
    s = np.asarray(points, np.float64).ravel()
    table = cox_de_boor(s, knots, degree)
    if derivative:
        t, lower = knot_vector(knots, degree), table[-2]
        values = np.zeros_like(table[-1])
        for i in range(values.shape[0]):
            if t[i + degree] > t[i]:
                values[i] += degree / (t[i + degree] - t[i]) * lower[i]
            if t[i + degree + 1] > t[i + 1]:
                values[i] -= degree / (t[i + degree + 1] - t[i + 1]) * lower[i + 1]
    else:
        values = table[-1]
    for side, outside in ((0, s < knots[0]), (-1, s > knots[-1])):
        if outside.any():
            for index, piece in enumerate(end_pieces(side, knots, degree)):
                values[index, outside] = (piece.deriv() if derivative else piece)(s[outside])
    return values


def taylor(s, degree=DEGREE):
    return sum(np.asarray(s, np.float64) ** order / math.factorial(order) for order in range(degree + 1))


def module_values(points, theta, dtype=jnp.float64):
    """The scaffold's f(s) for one coefficient vector theta [n] at scores ``points``."""
    pieces = spline.to_pieces(jnp.asarray(theta, dtype), KNOTS, DEGREE, dtype)
    return spline.evaluate(jnp.asarray(points, dtype), lambda piece, order: pieces[piece, order],
                           knots=KNOTS, degree=DEGREE)


def assert_within(observed, expected, bound, label=""):
    """|observed - expected| <= bound elementwise (bound may be an array)."""
    error = np.abs(np.asarray(observed, np.float64) - np.asarray(expected, np.float64))
    excess = error - bound
    assert np.all(excess <= 0), (label, float(np.max(excess)), int(np.argmax(excess)))


def random_theta(rng, count=8):
    return np.asarray([float(value) for value in QUARTIC_INIT]) * (1 + 0.3 * rng.standard_normal(count)) \
        + 20 * rng.standard_normal(count)


# --- The basis and its Taylor start ---------------------------------------------------------------

def test_taylor_coefficients_are_the_exact_blossoms():
    """b_init derived in code from the knot vector and T_p equals the decided fractions (degree 4)
    and the cubic check values (degree 3); the coefficient count is len(knots) + degree - 1."""
    assert spline.taylor_coefficients(KNOTS, 4) == QUARTIC_INIT
    assert spline.taylor_coefficients(KNOTS, 3) == CUBIC_INIT
    assert spline.basis_count(KNOTS, 4) == 8 != 4 + 1
    assert spline.basis_count(KNOTS, 3) == 7
    assert [float(knot) for knot in spline.clamped_knot_vector(KNOTS, 4)] == list(knot_vector())
    assert spline.piece_centres(KNOTS) == (-6, 0, 0, 6)
    # The independent reference agrees: sum_i b_i B_i = T_4 inside the knots and, by its end
    # pieces, outside them.
    points = np.concatenate([np.linspace(-60, 60, 2401), SPECIAL])
    represented = np.asarray([float(value) for value in QUARTIC_INIT]) @ reference_basis(points)
    np.testing.assert_allclose(represented, taylor(points), rtol=1e-12, atol=1e-12)
    cubic = np.asarray([float(value) for value in CUBIC_INIT]) @ reference_basis(points, degree=3)
    np.testing.assert_allclose(cubic, taylor(points, 3), rtol=1e-11, atol=1e-11)


def test_reference_basis_is_a_partition_of_unity():
    points = np.linspace(-32, 32, 1001)
    basis = reference_basis(points)
    assert basis.shape == (8, points.size) and (basis >= -1e-15).all()
    np.testing.assert_allclose(basis.sum(axis=0), 1.0, rtol=0, atol=1e-13)


def test_piece_matrix_is_the_basis_on_every_piece():
    """The exact linear map's local polynomials are the Cox-de Boor B-splines on their pieces."""
    exact = spline.exact_piece_matrix(KNOTS, DEGREE)
    centres = [float(centre) for centre in spline.piece_centres(KNOTS)]
    for piece, (left, right) in enumerate(zip(KNOTS, KNOTS[1:])):
        points = np.linspace(left, right, 41)[:-1]
        local = np.asarray([[float(value) for value in power] for power in exact[piece]])  # [power, basis]
        values = sum(np.outer(local[power], (points - centres[piece]) ** power) for power in range(DEGREE + 1))
        np.testing.assert_allclose(values, reference_basis(points), rtol=0, atol=1e-13)
    assert spline.piece_matrix(KNOTS, DEGREE).dtype == np.float32 and spline.piece_matrix(KNOTS, DEGREE).shape == (4, 5, 8)
    # Exactly: at the Taylor start each piece's local polynomial is T_4 expanded at its centre c
    # (the coefficient of (s - c)^r is T_(4-r)(c) / r!), the two central pieces T_4 itself.
    for piece, centre in enumerate(spline.piece_centres(KNOTS)):
        local = [sum((exact[piece][power][index] * QUARTIC_INIT[index] for index in range(8)), Fraction(0))
                 for power in range(DEGREE + 1)]
        assert local == [sum((centre ** m / math.factorial(m) for m in range(DEGREE - power + 1)), Fraction(0))
                         / math.factorial(power) for power in range(DEGREE + 1)], piece
        if centre == 0:
            assert local == [Fraction(1, math.factorial(power)) for power in range(DEGREE + 1)], piece


@pytest.mark.parametrize("coefficients", ("taylor", "random"))
def test_values_and_gradients_match_cox_de_boor(coefficients):
    """f, df/ds and df/dtheta = B_i(s) in exact arithmetic, inside the knots, at every knot and
    beside it, and under extrapolation, for the Taylor start and for random signed coefficients."""
    rng = np.random.default_rng(7)
    theta = (np.asarray([float(value) for value in QUARTIC_INIT]) if coefficients == "taylor"
             else random_theta(rng))
    points = np.concatenate([np.linspace(-50, 50, 1001), SPECIAL])
    basis, slopes = reference_basis(points), reference_basis(points, derivative=True)
    scale = np.abs(theta) @ np.abs(basis)
    with jax.enable_x64():
        values = np.asarray(module_values(points, theta))
        slope = np.asarray(jax.vmap(jax.grad(lambda s: module_values(s, theta)))(jnp.asarray(points)))
        jacobian = np.asarray(jax.vmap(jax.grad(lambda th, s: module_values(s, th), argnums=0),
                                       in_axes=(None, 0))(jnp.asarray(theta), jnp.asarray(points)))
    assert_within(values, theta @ basis, 1e-12 * scale, "value")
    assert_within(slope, theta @ slopes, 1e-10 * (np.abs(theta) @ np.abs(slopes)) + 1e-12, "slope")
    np.testing.assert_allclose(jacobian.T, basis, rtol=1e-12, atol=1e-12)
    if coefficients == "taylor":
        np.testing.assert_allclose(values, taylor(points), rtol=1e-12)
        np.testing.assert_allclose(slope, taylor(points, 3), rtol=1e-10, atol=1e-12)


def test_float32_reproduces_t4_inside_the_knots_and_under_extrapolation():
    """The production path (FP32 b_init storage, FP32 piece map, FP32 Horner) at the Taylor start,
    relative to T_4 itself. The largest error lies near T_4's minimum 0.27 at s = -1.6, where the
    B-spline terms cancel: sum_j |b_j B_j(s)| reaches 166 T_4(s)."""
    points = np.concatenate([np.linspace(-50, 50, 20001), SPECIAL]).astype(np.float32)
    stored = np.asarray([float(value) for value in QUARTIC_INIT], np.float32)
    values = np.asarray(module_values(points, stored, jnp.float32), np.float64)
    exact = taylor(points.astype(np.float64))
    relative = np.abs(values / exact - 1)
    inside = np.abs(points) <= 32
    assert relative[inside].max() < 5e-6, relative[inside].max()
    assert relative[~inside].max() < 1e-6, relative[~inside].max()
    for point in (-50, -32, -6, 0, 6, 32, 50):
        index = np.flatnonzero(points == np.float32(point))[0]
        assert relative[index] < 2e-6, (point, relative[index])


def test_value_and_three_derivatives_are_continuous_at_interior_knots():
    """Exactly: every B_i's adjacent pieces share value and derivatives 1-3 at -6, 0, 6, and the
    fourth derivative jumps (C^3, not C^4); numerically, one-sided FP64 derivatives agree."""
    exact = spline.exact_piece_matrix(KNOTS, DEGREE)
    centres = spline.piece_centres(KNOTS)

    def derivative(piece, basis, order, at):
        x = Fraction(at) - centres[piece]
        return sum((exact[piece][power][basis] * math.perm(power, order) * x ** (power - order)
                    for power in range(order, DEGREE + 1)), Fraction(0))

    for piece, knot in enumerate(KNOTS[1:-1], start=1):
        for basis in range(8):
            for order in range(4):
                assert derivative(piece - 1, basis, order, knot) == derivative(piece, basis, order, knot)
        jumps = [derivative(piece - 1, basis, 4, knot) - derivative(piece, basis, 4, knot) for basis in range(8)]
        assert any(jump != 0 for jump in jumps)
    theta = random_theta(np.random.default_rng(3))
    with jax.enable_x64():
        function = lambda s: module_values(s, theta)
        derivatives = [function]
        for _ in range(3):
            derivatives.append(jax.grad(derivatives[-1]))
        for knot in KNOTS[1:-1]:
            for order, function in enumerate(derivatives):
                below, above = float(function(knot - 1e-7)), float(function(knot + 1e-7))
                scale = max(abs(below), abs(float(derivatives[0](knot))) * 10.0 ** -order, 1.0)
                assert abs(below - above) < 1e-5 * scale, (knot, order, below, above)


def test_end_knots_continue_their_end_pieces():
    """At +-32 the end pieces evaluate the knot itself; past them f is the same polynomial (no clipping)."""
    theta = random_theta(np.random.default_rng(11))
    points = np.asarray([-80.0, -50.0, -33.0, -32.0, -31.0, 31.0, 32.0, 33.0, 50.0, 80.0])
    with jax.enable_x64():
        values = np.asarray(module_values(points, theta))
    pieces = [end_pieces(0), end_pieces(-1)]
    first = theta @ np.stack([piece(points[:5]) for piece in pieces[0]])
    last = theta @ np.stack([piece(points[5:]) for piece in pieces[1]])
    np.testing.assert_allclose(values, np.concatenate([first, last]), rtol=1e-11)
    # B_0(-32) = 1 and B_7(32) = 1: the clamped ends interpolate their end coefficients.
    np.testing.assert_allclose(values[[3, 6]], theta[[0, 7]], rtol=1e-12)


# --- Attention against a dense float64 reference ---------------------------------------------------

def de_boor(s, theta, span, t, degree=DEGREE):
    """f on knot span ``span`` at s (its polynomial continued anywhere): de Boor's algorithm."""
    d = [theta[..., index + span - degree] for index in range(degree + 1)]
    for level in range(1, degree + 1):
        for index in range(degree, level - 1, -1):
            alpha = (s - t[index + span - degree]) / (t[index + 1 + span - level] - t[index + span - degree])
            d[index] = (1 - alpha) * d[index - 1] + alpha * d[index]
    return d[degree]


def dense_spline(s, theta, knots=KNOTS, degree=DEGREE):
    """f(s) by de Boor's algorithm on each score's span; the end spans extend past the end knots."""
    t = knot_vector(knots, degree)
    pieces = len(knots) - 1
    piece = sum((s >= knot).astype(jnp.int32) for knot in knots[1:-1])
    return sum(jnp.where(piece == index, de_boor(s, theta, degree + index, t, degree), 0.0) for index in range(pieces))


def dense_attention(q, k, v, theta):
    """sum_(i<=t) f(s_ti) v_i with theta [heads, n] (static) or [batch, time, heads, n] (query)."""
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    time = q.shape[1]
    causal = jnp.tril(jnp.ones((time, time), bool))
    coefficients = theta[None, :, None, None, :] if theta.ndim == 2 else jnp.swapaxes(theta, 1, 2)[:, :, :, None, :]
    weights = jnp.where(causal, dense_spline(jnp.where(causal, scores, 0.0), coefficients), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def attention_case(*, batch=2, time=37, heads=2, width=8, value_width=5, query=False, seed=0, wide=True):
    """Scores across every piece and past both end knots, signed coefficients, negative weights."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, time, heads, width))
    if wide:
        q *= rng.uniform(0.3, 16.0, (batch, time, heads, 1))
    k = rng.standard_normal((batch, time, heads, width))
    v = rng.standard_normal((batch, time, heads, value_width))
    base = np.stack([random_theta(rng) for _ in range(heads)])
    theta = (base[None, None] * (1 + 0.2 * np.tanh(rng.standard_normal((batch, time, heads, 8))))
             if query else base)
    return q, k, v, theta


def vjp_program(attend):
    def run(q, k, v, theta, cotangent):
        output, pullback = jax.vjp(attend, q, k, v, theta)
        return [output, *pullback(cotangent)]

    return jax.jit(run)


def spline_arguments(query):
    return dict(attention_degree=DEGREE, attention_normalization=False, spline_knots=KNOTS)


def scaffold_attend(implementation, query, **options):
    name = "query_spline_coefficients" if query else "spline_coefficients"
    function = (partial(reference._blocked_scan, compute=jnp.float64, **options) if implementation == "scan"
                else partial(strip_dot_product_attention, **options))
    return lambda q, k, v, theta: function(q, k, v, **spline_arguments(query), **{name: theta})


def relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


@pytest.mark.parametrize("time,strip,block", ((37, 8, 4), (5, 8, 4), (1, 8, 4)))
@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_attention_and_gradients_match_the_dense_reference(query, implementation, time, strip, block):
    """Output and gradients of Q, K, V and theta, exact arithmetic, ragged strips and blocks."""
    q, k, v, theta = attention_case(time=time, query=query)
    cotangent = np.random.default_rng(time).standard_normal(v.shape)
    options = {"query_block_size": block} if implementation == "scan" else {"strip_size": strip}
    with jax.enable_x64():
        arrays = [jnp.asarray(x, jnp.float64) for x in (q, k, v, theta, cotangent)]
        expected = vjp_program(dense_attention)(*arrays)
        actual = vjp_program(scaffold_attend(implementation, query, **options))(*arrays)
    for label, observed, wanted in zip(("output", "q", "k", "v", "theta"), actual, expected, strict=True):
        observed, wanted = np.asarray(observed), np.asarray(wanted)
        assert observed.dtype == np.float64 and observed.shape == wanted.shape and np.isfinite(wanted).all(), label
        np.testing.assert_allclose(observed, wanted, rtol=1e-9, atol=1e-10 * np.abs(wanted).max(), err_msg=label)
    if time == 37:
        # The case reaches every piece and both continuations, with negative weights.
        with jax.enable_x64():
            scores = np.einsum("bqhd,bkhd->bhqk", q, k)[:, :, np.tril(np.ones((time, time), bool))] / math.sqrt(8)
        assert (scores < -32).any() and (scores > 32).any() and ((scores > -6) & (scores < 0)).any()
        assert ((scores > 0) & (scores < 6)).any() and ((scores < -6) & (scores > -32)).any()


def test_negative_weights_are_signed_and_masking_is_causal():
    """Signed weights pass unclipped (no positivity), and future keys never reach earlier rows."""
    q, k, v, _ = attention_case(time=21, wide=False, seed=4)
    theta = np.tile(np.asarray([-1.0, 2.0, -3.0, 0.5, -0.25, 1.5, -2.0, 1.0]), (2, 1))
    arguments = dict(**spline_arguments(False), spline_coefficients=jnp.asarray(theta, jnp.float32))
    q, k, v = (jnp.asarray(x, jnp.float32) for x in (q, k, v))
    output, diagnostics = jax.jit(lambda q, k, v: blocked_dot_product_attention(
        q, k, v, query_block_size=8, return_diagnostics=True, **arguments))(q, k, v)
    assert int(diagnostics["negative_kernel_count"]) > 0 and float(diagnostics["kernel_min"]) < 0
    assert int(diagnostics["invalid_kernel_count"]) == 0 and np.isfinite(np.asarray(output)).all()
    with jax.enable_x64():
        expected = dense_attention(*(jnp.asarray(x, jnp.float64) for x in (q, k, v, theta)))
    assert relative_l2(output, expected) < 1e-5
    strips = jax.jit(lambda q, k, v: strip_dot_product_attention(q, k, v, strip_size=8, **arguments))(q, k, v)
    # Perturbing the future (keys and values from position 10 on) leaves rows 0-9 unchanged.
    later = jnp.arange(21)[None, :, None, None] >= 10
    k2, v2 = jnp.where(later, 5 * k, k), jnp.where(later, -3 * v, v)
    for attend in (partial(blocked_dot_product_attention, query_block_size=8),
                   partial(strip_dot_product_attention, strip_size=8)):
        attend = jax.jit(partial(attend, **arguments))
        before, after = attend(q, k, v), attend(q, k2, v2)
        np.testing.assert_array_equal(np.asarray(before)[:, :10], np.asarray(after)[:, :10])
        assert not np.allclose(np.asarray(before)[:, 10:], np.asarray(after)[:, 10:])
    assert relative_l2(strips, output) < 2e-6


def generator_case(seed=5, batch=2, time=19, heads=2, hidden=12, width=8):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, time, hidden))
    q = rng.standard_normal((batch, time, heads, width)) * rng.uniform(0.3, 4.0, (batch, time, heads, 1))
    k = rng.standard_normal((batch, time, heads, width))
    v = rng.standard_normal((batch, time, heads, 5))
    bias = np.stack([random_theta(rng) for _ in range(heads)])
    amplitude = 0.1 * np.abs(bias) * rng.standard_normal((heads, 8))
    kernel = 0.3 * rng.standard_normal((heads, hidden, 8))
    return x, q, k, v, bias, amplitude, kernel


@pytest.mark.parametrize("implementation", ("scan", "strips"))
def test_query_generator_gradients_match_the_dense_reference(implementation):
    """theta_j(x_t) = b_j + a_j tanh(w_j^T x_t) through the scaffold's FP32 generator and FP32 attention:
    output and gradients of x, Q, K, V, W, a, b against the float64 generator and dense reference."""
    from scaffolds.transformer.attention.input_dependent import project_coefficients

    x, q, k, v, bias, amplitude, kernel = generator_case()
    cotangent = np.random.default_rng(9).standard_normal(v.shape)

    def scaffold(x, q, k, v, bias, amplitude, kernel):
        theta = project_coefficients(x, kernel, bias, amplitude=amplitude, activation="tanh")
        assert theta.dtype == jnp.float32 and theta.shape == (*q.shape[:3], 8)
        options = {"query_block_size": 8} if implementation == "scan" else {"strip_size": 8}
        attend = blocked_dot_product_attention if implementation == "scan" else strip_dot_product_attention
        return attend(q, k, v, **spline_arguments(True), query_spline_coefficients=theta, **options)

    def dense(x, q, k, v, bias, amplitude, kernel):
        theta = bias + amplitude * jnp.tanh(jnp.einsum("btd,hdj->bthj", x, kernel))
        return dense_attention(q, k, v, theta)

    def run(function, arrays, g):
        output, pullback = jax.vjp(function, *arrays)
        return [output, *pullback(g)]

    arrays = [jnp.asarray(value, jnp.float32) for value in (x, q, k, v, bias, amplitude, kernel)]
    actual = jax.jit(partial(run, scaffold))(arrays, jnp.asarray(cotangent, jnp.float32))
    with jax.enable_x64():
        expected = jax.jit(partial(run, dense))([jnp.asarray(value, jnp.float64) for value in
                                                 (x, q, k, v, bias, amplitude, kernel)],
                                                jnp.asarray(cotangent, jnp.float64))
    for label, observed, wanted in zip(("output", "x", "q", "k", "v", "b", "a", "W"), actual, expected, strict=True):
        assert observed.dtype == jnp.float32 and np.isfinite(np.asarray(observed)).all(), label
        assert np.linalg.norm(np.asarray(wanted)) > 0, label
        assert relative_l2(observed, wanted) < 2e-5, (label, relative_l2(observed, wanted))


@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_float32_strips_match_the_scan(query):
    q, k, v, theta = attention_case(query=query, seed=2)
    arrays = [jnp.asarray(x, jnp.float32) for x in (q, k, v, theta)]
    cotangent = jnp.asarray(np.random.default_rng(1).standard_normal(v.shape), jnp.float32)
    name = "query_spline_coefficients" if query else "spline_coefficients"
    scan = vjp_program(lambda q, k, v, th: blocked_dot_product_attention(
        q, k, v, query_block_size=8, **spline_arguments(query), **{name: th}))(*arrays, cotangent)
    strips = vjp_program(lambda q, k, v, th: strip_dot_product_attention(
        q, k, v, strip_size=16, **spline_arguments(query), **{name: th}))(*arrays, cotangent)
    for label, observed, wanted in zip(("output", "q", "k", "v", "theta"), strips, scan, strict=True):
        assert observed.dtype == wanted.dtype == jnp.float32, label
        assert relative_l2(observed, wanted) < 2e-5, label


@pytest.mark.parametrize("case", ("score_overflow", "q_nan"))
@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_invalid_rows_follow_the_signed_polynomial_rules(query, case):
    """A nonfinite score or kernel poisons its row (head 1 at position 20) and nothing else, in the
    scan and the strips alike; finite rows agree."""
    q, k, v, theta = (np.asarray(x, np.float32) for x in attention_case(batch=1, time=37, query=query, wide=False))
    if case == "score_overflow":
        q[0, 20, 1] *= 1e12          # finite scores near 1e12: their quartic pieces overflow FP32
    else:
        q[0, 20, 1, 0] = np.nan
    name = "query_spline_coefficients" if query else "spline_coefficients"
    arguments = dict(**spline_arguments(query), **{name: jnp.asarray(theta)})
    scan = jax.jit(partial(blocked_dot_product_attention, query_block_size=8, **arguments))(q, k, v)
    strips = jax.jit(partial(strip_dot_product_attention, strip_size=8, **arguments))(q, k, v)
    expected = np.arange(37) == 20
    for output in (scan, strips):
        bad = ~np.isfinite(np.asarray(output))
        np.testing.assert_array_equal(bad[0, :, 1].any(-1), expected)
        assert not bad[:, :, 0].any()
    finite = ~expected
    np.testing.assert_allclose(np.asarray(strips)[0, finite, 1], np.asarray(scan)[0, finite, 1], rtol=2e-5, atol=1e-4)


def test_score_coverage_counts_every_measured_score_once():
    """The scan's per-head coverage over ragged blocks equals a direct count of the causal scores."""
    q, k, v, theta = attention_case(batch=2, time=37, seed=6)
    q, k, v = (jnp.asarray(x, jnp.float32) for x in (q, k, v))
    _, diagnostics = jax.jit(partial(blocked_dot_product_attention, query_block_size=8, return_diagnostics=True,
                                     **spline_arguments(False)))(q, k, v, spline_coefficients=jnp.asarray(theta, jnp.float32))
    scores = np.einsum("bqhd,bkhd->bhqk", np.asarray(q, np.float64), np.asarray(k, np.float64)) / math.sqrt(8)
    causal = np.tril(np.ones((37, 37), bool))
    # FP32 rounding of a score cannot move it across a boundary: none lies within 1e-4 of one.
    assert min(np.abs(scores[:, :, causal] - boundary).min() for boundary in (-32, -6, 0, 6, 32)) > 1e-4
    pieces = np.stack([np.sum(causal & (scores >= left) & (scores < right), axis=(0, 2, 3))
                       for left, right in zip((-np.inf, -6, 0, 6), (-6, 0, 6, np.inf))], axis=-1)
    np.testing.assert_array_equal(np.asarray(diagnostics["spline_piece_count"]), pieces)
    assert pieces.sum() == 2 * 2 * 37 * 38 // 2
    np.testing.assert_array_equal(np.asarray(diagnostics["spline_below_range_count"]),
                                  np.sum(causal & (scores < -32), axis=(0, 2, 3)))
    np.testing.assert_array_equal(np.asarray(diagnostics["spline_above_range_count"]),
                                  np.sum(causal & (scores > 32), axis=(0, 2, 3)))
    assert np.asarray(diagnostics["spline_below_range_count"]).sum() > 0
    np.testing.assert_allclose(np.asarray(diagnostics["spline_score_min"]),
                               np.min(np.where(causal, scores, np.inf), axis=(0, 2, 3)), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(diagnostics["spline_score_max"]),
                               np.max(np.where(causal, scores, -np.inf), axis=(0, 2, 3)), rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_gpu_production_precision_against_float64(query):
    """Production precision on the allocated GPU: BF16 Q/K/V at the recipe's width and context
    with FP32 coefficients near the Taylor start, queries scaled so that scores reach every piece
    and both continuations. Against the scan's arithmetic in FP64, the strips' output and Q/K/V
    gradients (training and evaluation) and the scan's output (the diagnostics' forward) stay
    within 1.25 times the BF16 rounding floor, the strips' coefficient gradients within 1e-4,
    and the scan's diagnostics count every causal score once."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    rng = np.random.default_rng(29)
    batch, time, heads, width = 1, 2048, 4, 64
    q, k, v, cotangent = (rng.standard_normal((batch, time, heads, width)) for _ in range(4))
    # Unit-variance rows give scores of standard deviation about one, as QK RMSNorm does at the
    # start; per-query scales from 0.5 to 16 (log-uniform) carry them past -32 and 32.
    q *= np.exp(rng.uniform(math.log(0.5), math.log(16.0), (batch, time, heads, 1)))
    start = np.asarray([float(value) for value in QUARTIC_INIT])
    theta = (start * (1 + 0.1 * np.tanh(rng.standard_normal((batch, time, heads, 8)))) if query
             else start * (1 + 0.1 * rng.standard_normal((heads, 8))))
    name = "query_spline_coefficients" if query else "spline_coefficients"
    q, k, v, cotangent = (jnp.asarray(x, jnp.bfloat16) for x in (q, k, v, cotangent))
    theta = jnp.asarray(theta, jnp.float32)

    def values_and_gradients(attend, q, k, v, theta, cotangent):
        output, pullback = jax.vjp(lambda q, k, v, theta: attend(q, k, v, **spline_arguments(query), **{name: theta}),
                                   q, k, v, theta)
        return [output, *pullback(cotangent.astype(output.dtype))]

    strips = jax.jit(partial(values_and_gradients, strip_dot_product_attention))(q, k, v, theta, cotangent)
    scan, diagnostics = jax.jit(lambda q, k, v, theta: blocked_dot_product_attention(
        q, k, v, return_diagnostics=True, **spline_arguments(query), **{name: theta}))(q, k, v, theta)
    with jax.enable_x64():
        exact = jax.jit(partial(values_and_gradients, partial(reference._blocked_scan, compute=jnp.float64)))(
            *(jnp.asarray(x, jnp.float64) for x in (q, k, v, theta, cotangent)))
        exact = [np.asarray(x, np.float64) for x in exact]
    floors = [relative_l2(x.astype(jnp.bfloat16), x) for x in exact[:4]]
    for label, observed, wanted, floor in zip(("output", "q", "k", "v"), strips, exact, floors):
        assert observed.dtype == jnp.bfloat16 and np.isfinite(np.asarray(observed, np.float64)).all(), label
        assert relative_l2(observed, wanted) <= 1.25 * floor, (label, relative_l2(observed, wanted), floor)
    assert strips[4].dtype == jnp.float32 and relative_l2(strips[4], exact[4]) < 1e-4, relative_l2(strips[4], exact[4])
    assert relative_l2(scan, exact[0]) <= 1.25 * floors[0], (relative_l2(scan, exact[0]), floors[0])
    pieces = np.asarray(diagnostics["spline_piece_count"])
    assert pieces.sum() == batch * heads * time * (time + 1) // 2 and (pieces > 0).all(), pieces
    assert (np.asarray(diagnostics["spline_below_range_count"]) > 0).all()
    assert (np.asarray(diagnostics["spline_above_range_count"]) > 0).all()
    assert int(diagnostics["invalid_kernel_count"]) == 0 and int(diagnostics["negative_kernel_count"]) > 0


# --- Recipes, parameters, and models ---------------------------------------------------------------

# The model sections of the spline runs' controls, the static m = 4 and query m = 4 + W decay
# Transformer++ recipes (r4339a245820e and rca97cc9db8f2), inlined so that these tests, and the
# acceptance that runs them, read no study's files. Like every historical recipe, they lack the
# basis fields.
STATIC_M4_MODEL = {
    "kind": "transformer", "vocab_size": 32000, "max_seq_len": 2048, "num_layers": 15, "hidden_size": 896,
    "num_heads": 14, "num_kv_heads": 14, "qk_head_dim": 64, "value_head_dim": 64, "intermediate_size": 2304,
    "rope_theta": 10000.0, "rms_norm_eps": 1e-06, "rms_norm_fp32": True, "dropout": 0.0,
    "tie_embeddings": False, "remat": False, "attention_backend": "mosaic", "attention_degree": 4,
    "attention_normalization": False, "attention_output_scaling": "head_rms_norm",
    "attention_mechanism": "dot_product", "gdn_conv_size": 4, "gdn_erase_mode": "tied",
    "gdn_write_activation": "sigmoid", "gdn_polynomial_degree": 1, "gla_gate_rank": 16, "gla_gate_tau": 16.0,
    "gla_chunk_size": 64, "gla_inner_chunk_size": 16, "gla_use_rope": True, "quadratic_mode": "fixed",
    "quadratic_sharing": "global", "attention_coefficients": "learned", "attention_fixed_coefficients": None,
    "attention_fixed_square_factors": None, "attention_coefficient_sharing": "per_head",
    "attention_coefficient_conditioning": "static", "attention_coefficient_activation": "identity",
    "attention_coefficient_readout": "diagonal", "attention_coefficient_readout_scale": 1.0,
    "attention_coefficient_feature_count": None, "attention_coefficient_amplitude_initial": 1.0,
    "attention_coefficient_projection_initial_std": 0.0, "attention_coefficient_tanh_bias": False,
    "positional_encoding": "rope", "path_w_rank": 32, "path_conv_size": 3,
    "attention_coefficient_amplitude_scale": "none", "attention_coefficient_scale": "factorial",
    "attention_coefficient_parameterization": "direct", "attention_coefficient_initialization": "taylor",
    "attention_denominator_degree": None, "attention_denominator_parameterization": "direct",
    "attention_denominator_constant": "fixed", "qk_norm": True, "qk_norm_type": "rms_norm",
    "qk_norm_sharing": "per_layer", "qk_norm_initial_scale": None, "qk_norm_eps": 1e-06,
    "qk_norm_head_scale": True, "initialization": "transformer_truncated_reference",
}
QUERY_M4_MODEL = {
    **STATIC_M4_MODEL, "attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh",
    "attention_forget_gate": False, "attention_coefficient_kernel_weight_decay": True,
    "attention_coefficient_amplitude_initial": 0.0, "attention_coefficient_projection_initial_std": PROJECTION_STD,
    "attention_coefficient_amplitude_scale": "factorial", "qk_norm_token_scale": "none",
}
STATIC_CONTROL, QUERY_CONTROL = "static", "query"
CONTROL_MODELS = {STATIC_CONTROL: STATIC_M4_MODEL, QUERY_CONTROL: QUERY_M4_MODEL}


def control_model(control):
    return dict(CONTROL_MODELS[control])


def spline_model(control):
    """A control's model section with the decided spline changes."""
    model = control_model(control)
    model.update(attention_backend="blocked", attention_blocked_implementation="strips",
                 attention_kernel_basis="bspline", attention_spline_knots=list(KNOTS),
                 attention_coefficient_scale="none", attention_coefficient_amplitude_scale="none")
    return model


def recipe_with(model):
    """A recipe around a model section, with the controls' batch, context, and precision."""
    return {"data": {"batch_size": 256, "sequence_length": 2049}, "model": model,
            "train": {"dtype": "bfloat16", "micro_batch_size": 16}}


def test_production_recipes_validate_and_round_trip(tmp_path):
    from scaffolds.transformer.configs import config_to_dict, load_config, save_config
    from scaffolds.transformer.configs.loader import config_dicts_equal

    for control in CONTROL_MODELS:
        path = tmp_path / f"{control}.yaml"
        path.write_text(yaml.safe_dump(recipe_with(spline_model(control)), sort_keys=False))
        config = load_config(path)
        assert config.model.attention_spline_knots == KNOTS and config.model.attention_degree == 4
        saved = tmp_path / f"{control}-saved.yaml"
        save_config(config, saved)
        assert config_dicts_equal(config_to_dict(load_config(saved)), config_to_dict(config))
        assert yaml.safe_load(saved.read_text())["model"]["attention_spline_knots"] == list(KNOTS)


@pytest.mark.parametrize("control", (STATIC_CONTROL, QUERY_CONTROL))
def test_historical_recipes_load_and_resume_unchanged(control, tmp_path):
    """A recipe and a resolved run configuration written before the basis fields existed keep the
    monomial basis and compare equal on resume."""
    from scaffolds.transformer.configs import config_to_dict, load_config
    from scaffolds.transformer.configs.loader import config_dicts_equal

    recipe = tmp_path / "config.yaml"
    recipe.write_text(yaml.safe_dump(recipe_with(control_model(control))))
    # A run's resolved configuration (config.json) as the trainer wrote it before the fields existed.
    stored = config_to_dict(load_config(recipe))
    for key in ("attention_kernel_basis", "attention_spline_knots"):
        del stored["model"][key]
    resolved = tmp_path / "config.json"
    resolved.write_text(json.dumps(stored))
    for path in (recipe, resolved):
        config = load_config(path)
        assert config.model.attention_kernel_basis == "polynomial" and config.model.attention_spline_knots is None
        assert config.model.mosaic_supported()
    assert config_dicts_equal(stored, config_to_dict(load_config(resolved)))
    assert not config_dicts_equal({**stored, "model": {**stored["model"], "attention_kernel_basis": "bspline"}},
                                  config_to_dict(load_config(resolved)))


REJECTED = {
    "normalized": {"attention_normalization": True, "attention_output_scaling": "none"},
    "mosaic": {"attention_backend": "mosaic"},
    "fused": {"attention_backend": "fused"},
    "auto": {"attention_backend": "auto"},
    "cudnn": {"attention_backend": "cudnn"},
    "path": {"positional_encoding": "path"},
    "forget_gate": {"attention_forget_gate": True},
    "coefficient_scale": {"attention_coefficient_scale": "factorial"},
    "amplitude_scale": {"attention_coefficient_amplitude_scale": "factorial"},
    "orders": {"attention_coefficient_orders": [0, 2, 4]},
    "linear_start": {"attention_coefficient_initialization": "linear"},
    "fixed": {"attention_coefficients": "fixed"},
    "global": {"attention_coefficient_sharing": "global"},
    "query_key": {"attention_coefficient_conditioning": "query_key"},
    "dense": {"attention_coefficient_readout": "dense"},
    "tanh_bias": {"attention_coefficient_tanh_bias": True},
    "sigmoid": {"attention_coefficient_activation": "sigmoid"},
    "unsorted_knots": {"attention_spline_knots": [-32, 6, 0, 32]},
    "repeated_knots": {"attention_spline_knots": [-32, -6, -6, 32]},
    "one_knot": {"attention_spline_knots": [0]},
    "no_knots": {"attention_spline_knots": None},
    "degree_zero": {"attention_degree": 0},
    "exponential": {"attention_degree": None},
    "unknown_basis": {"attention_kernel_basis": "fourier"},
}


@pytest.mark.parametrize("change", list(REJECTED))
def test_configuration_rejects_unqualified_spline_settings(change):
    from scaffolds.transformer.configs import ModelConfig

    model = {**spline_model(QUERY_CONTROL), **REJECTED[change]}
    with pytest.raises(ValueError):
        ModelConfig(**model)


def test_spline_knots_belong_to_the_bspline_basis():
    from scaffolds.transformer.configs import ModelConfig

    model = control_model(STATIC_CONTROL)
    with pytest.raises(ValueError, match="bspline kernel basis only"):
        ModelConfig(**{**model, "attention_spline_knots": list(KNOTS)})
    for control in (STATIC_CONTROL, QUERY_CONTROL):
        config = ModelConfig(**spline_model(control))
        assert not config.mosaic_supported() and not config.mosaic_decay_supported()


@pytest.mark.parametrize("setting", ("penalty", "learning_rate_scale", "release", "continuation"))
def test_run_settings_for_monomial_orders_reject_the_spline(setting):
    from scaffolds.transformer.configs import Config, ModelConfig, validate_config

    config = Config(model=ModelConfig(**spline_model(QUERY_CONTROL)))
    config.data.sequence_length = 2049
    validate_config(config)
    if setting == "penalty":
        config.optim.coefficient_penalty = 1e-3
    elif setting == "learning_rate_scale":
        config.optim.query_coefficient_learning_rate_scale = 3.0
    elif setting == "release":
        # An optimizer recipe that defines a coefficient release schedule.
        object.__setattr__(config.optim, "coefficient_release_schedule", "all")
    else:
        config.continuation.parent_checkpoint = "parent"
    with pytest.raises(ValueError, match="B-spline|bspline"):
        validate_config(config)


def tiny_model(query, *, implementation="strips", **settings):
    from scaffolds.transformer.configs import ModelConfig

    extra = (dict(attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                  attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=0.2,
                  attention_coefficient_kernel_weight_decay=True) if query else {})
    values = dict(vocab_size=64, max_seq_len=32, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2,
                  qk_head_dim=16, value_head_dim=16, intermediate_size=64, attention_backend="blocked",
                  initialization="transformer_reference", attention_blocked_implementation=implementation,
                  attention_degree=4, attention_normalization=False, attention_coefficients="learned",
                  attention_coefficient_sharing="per_head", attention_kernel_basis="bspline",
                  attention_spline_knots=KNOTS, attention_output_scaling="head_rms_norm", qk_norm=True,
                  qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True)
    values.update(extra)
    values.update(settings)
    return ModelConfig(**values)


def fp32_correctly_rounded(stored, exact):
    """|stored - exact| <= half an FP32 ulp at stored, in exact rationals."""
    spacing = Fraction(float(np.spacing(np.float32(abs(stored)))))
    return abs(Fraction(float(stored)) - exact) <= spacing / 2


@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_parameters_start_at_taylor_in_fp32_with_the_softmax_backbone(query):
    from flax import nnx

    from scaffolds.transformer.acceptance import matches_softmax_backbone_initialization
    from scaffolds.transformer.configs import Config
    from scaffolds.transformer.models import build_model, model_parameter_count

    config = tiny_model(query)
    model = build_model(config, nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    params = {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))
              if len(path) == 1 and "spline" in path[0]}
    names = {("query_spline_bias",), ("query_spline_kernel",), ("query_spline_amplitude",)} if query else {
        ("spline_coefficients",)}
    assert set(params) == names
    assert all(value.dtype == np.float32 for value in params.values())
    bias = params[("query_spline_bias",) if query else ("spline_coefficients",)]
    assert bias.shape == (2, 2, 8)
    for value in bias.reshape(-1, 8):
        assert all(fp32_correctly_rounded(stored, exact) for stored, exact in zip(value, QUARTIC_INIT))
        np.testing.assert_array_equal(value, np.asarray([float(x) for x in QUARTIC_INIT], np.float32))
    if query:
        np.testing.assert_array_equal(params[("query_spline_amplitude",)], np.zeros((2, 2, 8), np.float32))
        kernel = params[("query_spline_kernel",)]
        assert kernel.shape == (2, 2, 32, 8) and np.all(kernel != 0)
    assert sum(leaf.size for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))) == model_parameter_count(config)
    facts = matches_softmax_backbone_initialization(Config(model=config))
    assert set(facts["mechanism_parameters"]) >= {name[0] for name in names}


def test_production_query_projection_draws_n_0_1_over_896():
    """W [15, 14, 896, 8] from the mechanism stream: mean 0 and standard deviation 1/sqrt(896)."""
    from scaffolds.transformer.attention.parameters import initial_attention_parameter_values, mechanism_key

    specs = spline.parameter_specs(hidden_size=896, degree=4, knots=KNOTS, conditioning="query", amplitude_initial=0.0,
                                   projection_initial_std=PROJECTION_STD, kernel_weight_decay=True)
    values = initial_attention_parameter_values(specs, num_layers=15, num_heads=14,
                                                key=mechanism_key(jax.random.PRNGKey(42)))
    kernel = np.asarray(values["query_spline_kernel"], np.float64)
    assert values["query_spline_kernel"].dtype == jnp.float32 and kernel.shape == (15, 14, 896, 8)
    assert abs(kernel.mean()) < 4 * PROJECTION_STD / math.sqrt(kernel.size)
    assert abs(kernel.std() / PROJECTION_STD - 1) < 4 / math.sqrt(2 * kernel.size)
    # Every layer, head, and output has its own draw.
    columns = kernel.reshape(-1, 896, 8)
    assert np.abs(np.corrcoef(columns[0].T) - np.eye(8)).max() < 0.15
    np.testing.assert_array_equal(np.asarray(values["query_spline_amplitude"]), np.zeros((15, 14, 8), np.float32))
    assert {spec.name: spec.weight_decay for spec in specs} == {
        "query_spline_bias": False, "query_spline_kernel": True, "query_spline_amplitude": False}


@pytest.mark.parametrize("query,kernel_decay", ((False, False), (True, True), (True, False)))
def test_weight_decay_reaches_only_the_projection(query, kernel_decay):
    """One AdamW update with zero gradients: W shrinks by 1 - lr * decay when its decay is on;
    theta, b, and a never decay."""
    from flax import nnx

    from scaffolds.transformer.configs import Config
    from scaffolds.transformer.training.state import create_state

    model = tiny_model(query, **({"attention_coefficient_kernel_weight_decay": kernel_decay} if query else {}))
    config = Config(model=model)
    config.data.sequence_length = 33
    config.data.synthetic_vocab_size = 64
    config.optim.warmup_steps = 0
    state = create_state(config, jax.random.PRNGKey(0))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    before, after = dict(nnx.to_flat_state(state.params)), dict(nnx.to_flat_state(updated.params))
    factor = 1 - config.optim.learning_rate * config.optim.weight_decay
    names = ("query_spline_bias", "query_spline_kernel", "query_spline_amplitude") if query else ("spline_coefficients",)
    for name in names:
        old, new = np.asarray(before[(name,)][...]), np.asarray(after[(name,)][...])
        if name == "query_spline_kernel" and kernel_decay:
            np.testing.assert_allclose(new, old * factor, rtol=1e-6)
            assert not np.array_equal(new, old)
        else:
            np.testing.assert_array_equal(new, old)
    # The backbone projections keep their usual decay.
    q_proj = ("layers_0", "attention", "q_proj", "kernel")
    np.testing.assert_allclose(np.asarray(after[q_proj][...]), np.asarray(before[q_proj][...]) * factor, rtol=1e-6)


def test_initial_spline_networks_compute_the_m4_taylor_kernel():
    """At initialization the static spline, the query spline (a = 0), and the fixed Taylor m = 4
    network share every backbone tensor, and their logits agree to FP32 tolerance."""
    from flax import nnx

    from scaffolds.transformer.models import build_model

    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 64, (2, 32)), jnp.int32)
    taylor_config = tiny_model(False, attention_coefficients="fixed", attention_kernel_basis="polynomial",
                               attention_spline_knots=None)
    networks = {name: build_model(config, nnx.Rngs(params=jax.random.PRNGKey(3), dropout=jax.random.PRNGKey(4)))
                for name, config in (("taylor", taylor_config), ("static", tiny_model(False)),
                                     ("query", tiny_model(True)))}
    states = {name: dict(nnx.to_flat_state(nnx.state(model, nnx.Param))) for name, model in networks.items()}
    for path, value in states["taylor"].items():
        for name in ("static", "query"):
            np.testing.assert_array_equal(np.asarray(states[name][path][...]), np.asarray(value[...]), err_msg=str(path))
    logits = {name: np.asarray(model(tokens), np.float64) for name, model in networks.items()}
    assert relative_l2(logits["static"], logits["taylor"]) < 1e-5
    assert relative_l2(logits["query"], logits["static"]) < 1e-6


def _float64_loss_and_gradients(config, tokens):
    from flax import nnx

    from scaffolds.transformer.models import build_model

    model = build_model(config, nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = jnp.float64
    graphdef, params = nnx.split(model, nnx.Param)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)

    def loss(params):
        logits = nnx.merge(graphdef, params)(tokens[:, :-1]).astype(jnp.float64)
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(log_probabilities, tokens[:, 1:, None], axis=-1))

    return jax.jit(jax.value_and_grad(loss))(params)


@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_two_layer_model_strips_match_the_scan_in_float64(query, monkeypatch):
    """The same loss and parameter gradients with either implementation, every layer on the strips
    (counted; four 8-query strips of 32 tokens). Plain QK and no output scaling keep FP32
    normalizations out of the float64 comparison; the query generator computes in FP32 by design."""
    tokens = jnp.asarray(np.random.default_rng(5).integers(0, 64, (2, 33)), jnp.int32)
    original = reference.strip_dot_product_attention
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, strip_size=8, **kwargs)

    monkeypatch.setattr(reference, "strip_dot_product_attention", counted)
    settings = dict(qk_norm=False, qk_norm_head_scale=False, attention_output_scaling="none")
    with jax.enable_x64():
        with monkeypatch.context() as patch:
            patch.setattr(reference, "blocked_dot_product_attention",
                          partial(reference._blocked_scan, compute=jnp.float64))
            scan = _float64_loss_and_gradients(tiny_model(query, implementation="scan", **settings), tokens)
            assert calls == []
            strips = _float64_loss_and_gradients(tiny_model(query, implementation="strips", **settings), tokens)
            assert len(calls) == 2
        np.testing.assert_allclose(strips[0], scan[0], rtol=1e-11)
        names = [str(path) for path, _ in jax.tree_util.tree_flatten_with_path(scan[1])[0]]
        assert any("spline" in name for name in names)
        for name, observed, wanted in zip(names, jax.tree.leaves(strips[1]), jax.tree.leaves(scan[1]), strict=True):
            np.testing.assert_allclose(observed, wanted, rtol=1e-10, atol=1e-11 * max(np.max(np.abs(wanted)), 1.0),
                                       err_msg=name)


def test_coefficient_override_and_capture_reject_the_spline_basis():
    from flax import nnx

    from scaffolds.transformer.models import build_model

    model = build_model(tiny_model(False), nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    tokens = jnp.zeros((1, 8), jnp.int32)
    with pytest.raises(ValueError, match="B-spline"):
        model(tokens, return_coefficients=True)
    with pytest.raises(ValueError, match="B-spline"):
        model(tokens, query_coefficients=jnp.zeros((2, 1, 8, 2, 5)))


@pytest.mark.parametrize("query", (False, True), ids=("static", "query"))
def test_metadata_and_diagnostics_describe_the_spline(query):
    from scaffolds.transformer.configs import Config
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
    from scaffolds.transformer.training.state import create_state
    from scaffolds.transformer.training.tracking import attention_metrics, run_tags

    config = Config(model=tiny_model(query))
    config.data.sequence_length = 33
    config.data.synthetic_vocab_size = 64
    identity = attention_identity(config)
    kernel = "query_spline_unnormalized" if query else "learned_spline_unnormalized"
    assert identity["attention_kernel"] == kernel and identity["attention_kernel_basis"] == "bspline"
    assert identity["attention_degree"] == 4 and identity["attention_spline_basis_count"] == 8
    assert identity["attention_spline_knots"] == list(KNOTS)
    assert identity["attention_spline_knot_vector"] == list(knot_vector())
    assert identity["attention_spline_initial_coefficients"] == [str(value) for value in QUARTIC_INIT]
    assert identity["attention_implementation"] == "blocked_causal_strips" and identity["attention_signed"]
    assert identity["attention_coefficient_count"] == 2 * 2 * 8 * (32 + 2 if query else 1)
    assert "B-splines on the knots [-32, -6, 0, 6, 32]" in identity["attention_formula"]
    assert identity["attention_formula"].endswith("then per-head RMSNorm over value channels with a learned gain")
    assert kernel in run_tags(config, {})
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == ("unshifted_query_spline" if query else "unshifted_spline")
    coverage = report["spline_score_coverage"]
    assert coverage["pieces"] == ["(-inf, -6)", "[-6, 0)", "[0, 6)", "[6, inf)"]
    fractions = np.asarray(coverage["piece_fraction"])
    assert fractions.shape == (2, 2, 4)
    np.testing.assert_allclose(fractions.sum(-1), 1.0)
    assert np.asarray(coverage["below_range_fraction"]).shape == (2, 2)
    for layer in report["layers"]:
        assert sum(layer[f"spline_piece_{index}_fraction"] for index in range(4)) == pytest.approx(1.0)
        assert layer["spline_below_range_fraction"] == layer["spline_above_range_fraction"] == 0.0
    coefficients = report["coefficients"]
    assert coefficients["effective_coefficient_names"] == [f"B{index}" for index in range(8)]
    if query:
        np.testing.assert_allclose(np.asarray(coefficients["projection_bias"])[0, 0],
                                   [float(value) for value in QUARTIC_INIT], rtol=1e-7)
        assert np.all(np.asarray(coefficients["modulation_amplitude"]) == 0)
        assert np.asarray(coefficients["projection_kernel_l2"]).shape == (2, 2, 8)
        assert np.asarray(coefficients["effective_summary"]["mean"]).shape == (2, 2, 8)
    else:
        np.testing.assert_allclose(np.asarray(coefficients["effective"])[1, 1],
                                   [float(value) for value in QUARTIC_INIT], rtol=1e-7)
    metrics = attention_metrics(report)
    assert f"attention/{kernel}/layer00/spline_below_range_fraction" in metrics


def test_attention_layers_take_exactly_their_own_coefficient_arrays():
    """A spline layer refuses missing, foreign, or monomial coefficient arrays; a monomial layer
    refuses spline arrays."""
    from flax import nnx

    from scaffolds.transformer.attention.parameters import parameters_for_layer
    from scaffolds.transformer.models import build_model

    rngs = lambda: nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))
    model = build_model(tiny_model(False), rngs())
    parameters = parameters_for_layer(model, model.attention_parameter_specs, 0)
    attention, x = model.layers_0.attention, jnp.ones((1, 4, 32))
    assert attention(x, attention_parameters=parameters).shape == x.shape
    missing = {name: value for name, value in parameters.items() if name != "spline_coefficients"}
    for wrong in (missing, {**parameters, "query_spline_bias": jnp.ones((2, 8))}):
        with pytest.raises(ValueError, match="exactly its model-owned"):
            attention(x, attention_parameters=wrong)
    with pytest.raises(ValueError, match="unnormalized kernel alone"):
        attention(x, attention_parameters={**parameters, "polynomial_coefficients": jnp.ones((2, 5))})
    polynomial = build_model(tiny_model(False, attention_kernel_basis="polynomial", attention_spline_knots=None), rngs())
    own = parameters_for_layer(polynomial, polynomial.attention_parameter_specs, 0)
    assert "polynomial_coefficients" in own and not any("spline" in name for name in own)
    with pytest.raises(ValueError, match="exactly its model-owned"):
        polynomial.layers_0.attention(x, attention_parameters={**own, "spline_coefficients": jnp.ones((2, 8))})
