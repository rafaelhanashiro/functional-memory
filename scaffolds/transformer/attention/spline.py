"""Fixed B-spline score functions for unnormalized dot-product attention.

``model.attention_kernel_basis: bspline`` replaces the monomials s^r of a learned unnormalized
kernel by a fixed B-spline basis: the attention output is ``sum_(i<=t) f(s_ti) v_i`` with
``f(s) = sum_j theta_j B_j(s)``. The basis has degree p = ``attention_degree`` on the distinct
knots u_0 < ... < u_K (``attention_spline_knots``), clamped: the knot vector repeats each endpoint
p + 1 times, so there are K + p functions B_0..B_(K+p-1) (``len(knots) + degree - 1``, not
degree + 1), and f is a polynomial of degree p on each interval [u_j, u_(j+1)] with p - 1
continuous derivatives at the simple interior knots. Outside [u_0, u_K] f continues its first
or last polynomial piece; scores are never clipped. The same basis serves every layer, head,
and token; only the coefficients theta are learned, static per layer and head
(``spline_coefficients``) or generated per query token by the diagonal tanh generator of the
monomial query coefficients (``query_spline_bias``, ``query_spline_kernel``,
``query_spline_amplitude``: theta_j(x_t) = b_j + a_j tanh(w_j^T x_t)).

Evaluation never runs Cox-de Boor per score. A fixed linear map (``piece_matrix``), derived
exactly in rationals and rounded once to the compute dtype, takes theta to the local
polynomial of every piece, ``f(s) = sum_r P_jr (s - c_j)^r`` on piece j, where the centre c_j
is the point of [u_j, u_(j+1)] nearest zero (``piece_centres``; -6, 0, 0, 6 for the knots
-32, -6, 0, 6, 32). Attention scores concentrate near zero, so most evaluations use a small
local variable; at the Taylor start each piece's local polynomial is T_p expanded at c_j, the
two central pieces T_p in s itself. A score selects its piece by comparison with the
interior knots (s < u_1 first piece, ..., s >= u_(K-1) last piece; a NaN score takes the last
piece and stays NaN) and is evaluated by Horner's rule. The map runs once per head (static)
or per query token and head before any per-score work.

``taylor_coefficients`` gives the exact B-spline coefficients of the Taylor polynomial
T_p(s) = sum_(r<=p) s^r / r! by blossoming; the spline then equals T_p on the whole real
line, extrapolation included, and starts as the degree-p Taylor kernel.
``initial_coefficients(..., initialization="affine", intercept=u, slope=v)`` instead
starts at u + v*s, using that affine function's values at the Greville abscissae.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import math

import jax
import jax.numpy as jnp
import numpy as np

KERNEL_BASES = ("polynomial", "bspline")
STATIC_PARAMETERS = ("spline_coefficients",)
QUERY_PARAMETERS = ("query_spline_bias", "query_spline_kernel", "query_spline_amplitude")


def validate_knots(knots, degree) -> tuple[float, ...]:
    """The distinct knots as floats: finite, strictly increasing, at least two; degree >= 1."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("A B-spline basis requires an integer degree >= 1")
    if (not isinstance(knots, (tuple, list)) or len(knots) < 2
            or any(not isinstance(knot, (int, float)) or isinstance(knot, bool) or not math.isfinite(knot)
                   for knot in knots)):
        raise ValueError("B-spline knots must be a list of at least two finite numbers")
    knots = tuple(float(knot) for knot in knots)
    if any(right <= left for left, right in zip(knots, knots[1:])):
        raise ValueError("B-spline knots must be strictly increasing (distinct; the endpoints are clamped)")
    return knots


def _exact(knots) -> tuple[Fraction, ...]:
    # Fraction(float) is the float's exact binary value.
    return tuple(Fraction(knot) for knot in knots)


def basis_count(knots, degree: int) -> int:
    """Number of B-splines on the clamped knots: len(knots) + degree - 1."""
    return len(knots) + degree - 1


def clamped_knot_vector(knots, degree: int) -> tuple[Fraction, ...]:
    """(u_0 x (degree+1), u_1, ..., u_(K-1), u_K x (degree+1)), exact."""
    distinct = _exact(validate_knots(knots, degree))
    return (distinct[0],) * (degree + 1) + distinct[1:-1] + (distinct[-1],) * (degree + 1)


def piece_centres(knots) -> tuple[Fraction, ...]:
    """Each piece's expansion point: the point of [u_j, u_(j+1)] nearest zero."""
    distinct = _exact(knots)
    return tuple(min(max(Fraction(0), left), right) for left, right in zip(distinct, distinct[1:]))


def _add(left, right):
    size = max(len(left), len(right))
    return tuple((left[i] if i < len(left) else 0) + (right[i] if i < len(right) else 0) for i in range(size))


def _times_linear(polynomial, constant, slope):
    """polynomial(s) * (constant + slope s), ascending coefficients."""
    if not polynomial:
        return ()
    return _add(tuple(constant * value for value in polynomial), (0, *(slope * value for value in polynomial)))


@lru_cache(maxsize=None)
def _monomial_pieces(knots: tuple[float, ...], degree: int):
    """[piece][basis][power]: B_i on [u_j, u_(j+1)] in ascending powers of s, exact (Cox-de Boor)."""
    t = clamped_knot_vector(knots, degree)
    pieces = []
    for piece in range(len(knots) - 1):
        span = degree + piece  # t[span] = u_piece < t[span + 1] = u_(piece + 1)
        basis = [(Fraction(1),) if index == span else () for index in range(len(t) - 1)]
        for order in range(1, degree + 1):
            basis = [_add(_times_linear(basis[i], -t[i] / (t[i + order] - t[i]), 1 / (t[i + order] - t[i]))
                          if t[i + order] != t[i] else (),
                          _times_linear(basis[i + 1], t[i + order + 1] / (t[i + order + 1] - t[i + 1]),
                                        -1 / (t[i + order + 1] - t[i + 1]))
                          if t[i + order + 1] != t[i + 1] else ())
                     for i in range(len(t) - 1 - order)]
        pieces.append(tuple(tuple(Fraction(value) for value in function) + (Fraction(0),) * (degree + 1 - len(function))
                            for function in basis))
    return tuple(pieces)


def _shifted(polynomial, centre: Fraction):
    """Ascending coefficients of p(centre + x) in x."""
    degree = len(polynomial) - 1
    return tuple(sum((polynomial[k] * math.comb(k, r) * centre ** (k - r) for k in range(r, degree + 1)), Fraction(0))
                 for r in range(degree + 1))


@lru_cache(maxsize=None)
def exact_piece_matrix(knots: tuple[float, ...], degree: int) -> tuple:
    """M[j][r][i]: the coefficient of (s - c_j)^r in B_i on piece j, so P_jr = sum_i M[j][r][i] theta_i."""
    centres = piece_centres(knots)
    pieces = _monomial_pieces(knots, degree)
    local = [[_shifted(function, centres[piece]) for function in pieces[piece]] for piece in range(len(pieces))]
    return tuple(tuple(tuple(local[piece][index][power] for index in range(basis_count(knots, degree)))
                       for power in range(degree + 1)) for piece in range(len(pieces)))


@lru_cache(maxsize=None)
def _piece_matrix(knots: tuple[float, ...], degree: int, dtype_name: str) -> np.ndarray:
    exact = exact_piece_matrix(knots, degree)
    return np.asarray([[[float(value) for value in row] for row in power] for power in exact],
                      np.float64).astype(dtype_name)


def piece_matrix(knots, degree: int, dtype=jnp.float32) -> np.ndarray:
    """``exact_piece_matrix`` as a NumPy constant [pieces, degree + 1, basis] in ``dtype``."""
    return _piece_matrix(validate_knots(knots, degree), degree, np.dtype(dtype).name)


@lru_cache(maxsize=None)
def taylor_coefficients(knots: tuple[float, ...], degree: int) -> tuple[Fraction, ...]:
    """Exact B-spline coefficients of T_p(s) = sum_(r<=p) s^r / r! by blossoming.

    theta_i is the blossom of T_p at the knots t_(i+1)..t_(i+p): the blossom of s^r is the
    elementary symmetric polynomial e_r of those knots divided by C(p, r). The result is
    checked exactly: every piece of sum_i theta_i B_i is T_p.
    """
    knots = validate_knots(knots, degree)
    t = clamped_knot_vector(knots, degree)
    taylor = tuple(Fraction(1, math.factorial(order)) for order in range(degree + 1))
    coefficients = []
    for index in range(basis_count(knots, degree)):
        arguments = t[index + 1:index + degree + 1]
        coefficients.append(sum((taylor[order] * sum((math.prod(subset, start=Fraction(1))
                                                      for subset in combinations(arguments, order)), Fraction(0))
                                 / math.comb(degree, order) for order in range(degree + 1)), Fraction(0)))
    for piece in _monomial_pieces(knots, degree):
        represented = tuple(sum((theta * function[power] for theta, function in zip(coefficients, piece)), Fraction(0))
                            for power in range(degree + 1))
        if represented != taylor:
            raise AssertionError("Blossoming did not reproduce the Taylor polynomial")
    return tuple(coefficients)


def initial_coefficients(knots, degree: int, initialization: str = "taylor", *,
                         intercept: float | None = None, slope: float | None = None) -> tuple[Fraction, ...]:
    """Exact initial coefficients, including the polynomial extensions of the end pieces.

    An affine function is reproduced by setting theta_i = intercept + slope*g_i, where
    g_i is the mean of the degree knots t_(i+1)..t_(i+degree) (the Greville abscissa).
    The affine settings are explicit, finite numbers; they never change Taylor starts.
    """
    knots = validate_knots(knots, degree)
    if initialization == "taylor":
        if intercept is not None or slope is not None:
            raise ValueError("Spline initial intercept and slope apply only to affine initialization")
        return taylor_coefficients(knots, degree)
    if initialization != "affine":
        raise ValueError("B-spline coefficient initialization must be taylor or affine")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (intercept, slope)):
        raise ValueError("Affine spline initialization requires a finite initial intercept and slope")
    t = clamped_knot_vector(knots, degree)
    intercept, slope = Fraction(intercept), Fraction(slope)
    return tuple(intercept + slope * sum(t[index + 1:index + degree + 1], Fraction(0)) / degree
                 for index in range(basis_count(knots, degree)))


def parameter_specs(*, hidden_size: int, degree: int, knots, conditioning: str = "static",
                    amplitude_initial: float = 1.0, projection_initial_std: float = 0.0,
                    kernel_weight_decay: bool = False, initialization: str = "taylor",
                    initial_intercept: float | None = None, initial_slope: float | None = None):
    """Per layer/head FP32 arrays, starting at the selected function: static theta or query generator.

    Static: ``spline_coefficients`` [basis], the initial B-spline coefficients, without weight
    decay. Query: ``query_spline_bias`` b (those coefficients) and ``query_spline_amplitude`` a
    (amplitude_initial) [basis], without weight decay, and ``query_spline_kernel`` W
    [hidden_size, basis], N(0, projection_initial_std^2) from the mechanism stream, decayed only
    with ``kernel_weight_decay``. No factorial scaling.
    """
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    initial = tuple(float(value) for value in initial_coefficients(
        knots, degree, initialization, intercept=initial_intercept, slope=initial_slope))
    if conditioning == "static":
        return (AttentionParameterSpec(name="spline_coefficients", argument="spline_coefficients",
                                       initial=initial, sharing="per_head", weight_decay=False),)
    if conditioning != "query":
        raise ValueError("B-spline coefficients are static or query-conditioned")
    count = len(initial)
    return (
        AttentionParameterSpec(name="query_spline_bias", argument="query_spline_bias", initial=initial,
                               sharing="per_head", weight_decay=False),
        AttentionParameterSpec(name="query_spline_kernel", argument="query_spline_kernel",
                               initial=(0.0,) * (hidden_size * count), initial_shape=(hidden_size, count),
                               sharing="per_head", weight_decay=bool(kernel_weight_decay),
                               initial_std=float(projection_initial_std)),
        AttentionParameterSpec(name="query_spline_amplitude", argument="query_spline_amplitude",
                               initial=(float(amplitude_initial),) * count, sharing="per_head", weight_decay=False),
    )


def to_pieces(coefficients, knots, degree: int, dtype):
    """B-spline coefficients [..., basis] to the pieces' local coefficients [..., pieces, degree + 1].

    One fixed linear map in ``dtype`` (FP32 in every model, FP64 in exact-arithmetic tests).
    """
    matrix = piece_matrix(knots, degree, dtype)
    if coefficients.shape[-1] != matrix.shape[-1]:
        raise ValueError(f"A degree-{degree} B-spline on {len(knots)} knots has {matrix.shape[-1]} coefficients, "
                         f"got {coefficients.shape[-1]}")
    return jnp.einsum("...i,jri->...jr", coefficients.astype(dtype), jnp.asarray(matrix),
                      precision=jax.lax.Precision.HIGHEST, preferred_element_type=dtype)


def _piece_selector(scores, knots):
    """select(values) takes values[j] where a score lies in piece j (a NaN score: the last piece)."""
    below = [scores < knot for knot in knots[1:-1]]

    def select(values):
        result = values[-1]
        for piece in range(len(below) - 1, -1, -1):
            result = jnp.where(below[piece], values[piece], result)
        return result

    return select


def evaluate(scores, coefficient, *, knots, degree: int):
    """f at every score: Horner's rule in s - c_j on each score's piece j.

    ``coefficient(j, r)`` is piece j's coefficient of (s - c_j)^r, broadcastable to ``scores``.
    """
    select = _piece_selector(scores, knots)
    pieces = range(len(knots) - 1)
    local = scores - select([jnp.asarray(float(centre), scores.dtype) for centre in piece_centres(knots)])
    value = select([coefficient(piece, degree) for piece in pieces])
    for order in range(degree - 1, -1, -1):
        value = value * local + select([coefficient(piece, order) for piece in pieces])
    return value


def score_coverage(scores, measured, knots) -> dict:
    """Per-head counts of the measured scores each piece evaluates, of those outside [u_0, u_K]
    (polynomial continuation), and the per-head score extrema; [heads, pieces] and [heads]."""
    select = _piece_selector(scores, knots)
    piece = select([jnp.asarray(index, jnp.int32) for index in range(len(knots) - 1)])
    axes = (0, 2, 3)
    return {
        "spline_piece_count": jnp.stack([jnp.sum(measured & (piece == index), axis=axes, dtype=jnp.int32)
                                         for index in range(len(knots) - 1)], axis=-1),
        "spline_below_range_count": jnp.sum(measured & (scores < knots[0]), axis=axes, dtype=jnp.int32),
        "spline_above_range_count": jnp.sum(measured & (scores > knots[-1]), axis=axes, dtype=jnp.int32),
        "spline_score_min": jnp.min(jnp.where(measured, scores, jnp.inf), axis=axes),
        "spline_score_max": jnp.max(jnp.where(measured, scores, -jnp.inf), axis=axes),
    }


def coefficient_statistics(coefficients):
    """Generated query coefficients theta on the diagnostic sample, per head and basis function."""
    values = coefficients.astype(jnp.float32)
    return {
        "spline_coefficient_mean": jnp.mean(values, axis=(0, 1)),
        "spline_coefficient_std": jnp.std(values, axis=(0, 1)),
        "spline_coefficient_min": jnp.min(values, axis=(0, 1)),
        "spline_coefficient_max": jnp.max(values, axis=(0, 1)),
    }


def spline_attention(q, k, v, *, attention_degree, spline_knots, spline_coefficients=None,
                     query_spline_coefficients=None, backend="blocked", return_diagnostics=False,
                     blocked_implementation="scan"):
    """Unnormalized sum_(i<=t) f(s_ti) v_i on the blocked backend (scan or causal strips).

    ``spline_coefficients`` [basis] or [heads, basis] are static; ``query_spline_coefficients``
    [batch, time, heads, basis] are one function per query and head. No fused, Mosaic, or cuDNN
    kernel implements the basis. Diagnostics come from the scan and add the per-head score
    coverage, plus the generated coefficients' statistics.
    """
    from scaffolds.transformer.attention.backends.reference import blocked_backend_attention

    if backend != "blocked":
        raise ValueError("B-spline attention runs on the blocked backend only; no other kernel implements it")
    result = blocked_backend_attention(
        q, k, v, implementation=blocked_implementation, attention_degree=attention_degree,
        attention_normalization=False, spline_knots=spline_knots, spline_coefficients=spline_coefficients,
        query_spline_coefficients=query_spline_coefficients, return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
        return result
    output, diagnostics = result
    if query_spline_coefficients is not None:
        diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(query_spline_coefficients)))
    return output, diagnostics
