"""Normalized attention with separate learned numerator and denominator polynomials.

The weights are p_a(s_ij) / sum_(k<=i) p_b(s_ik) with a numerator
p_a(s) = sum_(r<=m) a_r s^r, learned as ``polynomial_coefficients`` (a_0..a_m), and a
denominator p_b of degree d (``attention_denominator_degree``, the numerator's degree
when unset). Its form (``attention_denominator_parameterization``) and constant
(``attention_denominator_constant``) select one learned array:

| Form, constant | Array | p_b(s) |
|---|---|---|
| direct, fixed | ``polynomial_denominator_coefficients`` b_1..b_d | 1 + sum_r b_r s^r |
| direct, learned | ``polynomial_denominator_coefficients_with_constant`` b_0..b_d | sum_r b_r s^r |
| sum_of_squares, fixed | ``polynomial_denominator_square_factors`` u_1..u_k, w_1..w_k | (1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2 |
| sum_of_squares, learned | ``polynomial_denominator_square_factors_with_constant`` q1_0..q1_k, q2_0..q2_(k-1) | q1(s)^2 + q2(s)^2 |

with d = 2k for sums of squares. A fixed constant of 1 removes the common scale of the
two polynomials; a learned one leaves it free. Both sum-of-squares forms are complete:
every nonnegative polynomial of degree 2k is |h|^2 for a complex h of degree k, rotated
so that h(0) = 1 (fixed constant) or so that its leading coefficient is real (learned).
A direct denominator is signed: a negative nonzero row sum is valid. An exactly zero or
nonfinite row sum poisons its row.

Initialization follows ``attention_coefficient_initialization``: ``taylor`` makes both
polynomials T_m and T_d (sum_r s^r / r!); ``linear`` sets a_1 = 1 and the other
numerator coefficients to 0, and the denominator to 1, so attention starts at
sum_(k<=i) s_ik v_k / (i + 1). With a fixed constant, a sum-of-squares denominator
cannot start exactly at 1 and still move: with every w_r = 0 their gradient is zero, so
its linear start sets w_1 = LINEAR_START_SQUARE_FACTOR (p_b = 1 + 1e-6 s^2). With a
learned constant it starts exactly at 1 as q1 = q2 = 1/sqrt(2), and every factor moves.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.unnormalized import initial_coefficients

DENOMINATOR_PARAMETERIZATIONS = ("direct", "sum_of_squares")
DENOMINATOR_CONSTANTS = ("fixed", "learned")
DENOMINATOR_KINDS = {
    "polynomial_denominator_coefficients": ("direct", "fixed"),
    "polynomial_denominator_coefficients_with_constant": ("direct", "learned"),
    "polynomial_denominator_square_factors": ("sum_of_squares", "fixed"),
    "polynomial_denominator_square_factors_with_constant": ("sum_of_squares", "learned"),
}
DENOMINATOR_ARGUMENTS = {kind: name for name, kind in DENOMINATOR_KINDS.items()}
LINEAR_START_SQUARE_FACTOR = 1e-3


def denominator_degree(model) -> int:
    """The denominator's degree d: ``attention_denominator_degree``, or the numerator's when unset."""
    degree = getattr(model, "attention_denominator_degree", None)
    return model.attention_degree if degree is None else degree


def denominator_parameterization(model) -> str:
    return getattr(model, "attention_denominator_parameterization", "direct")


def denominator_constant(model) -> str:
    return getattr(model, "attention_denominator_constant", "fixed")


def denominator_argument(model) -> str:
    return DENOMINATOR_ARGUMENTS[(denominator_parameterization(model), denominator_constant(model))]


def parameter_count(degree: int, form: str, constant: str) -> int:
    return degree + (constant == "learned")


def _validate_degree(degree, *, square=False) -> None:
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("A rational kernel's denominator requires an integer degree >= 1")
    if square and degree % 2:
        raise ValueError("A sum-of-squares denominator requires an even degree")


def initial_denominator(degree: int, initialization: str = "taylor") -> tuple[float, ...]:
    """Initial b_1..b_d (the constant b_0 = 1 is not a parameter): 1/r! for Taylor, zeros for linear."""
    _validate_degree(degree)
    if initialization == "linear":
        return (0.0,) * degree
    if initialization == "taylor":
        return tuple(1.0 / math.factorial(order) for order in range(1, degree + 1))
    raise ValueError(f"Unknown coefficient initialization: {initialization}")


def initial_denominator_square_factors(degree: int, initialization: str = "taylor") -> tuple[float, ...]:
    """Initial u_1..u_k, w_1..w_k (d = 2k) of a sum-of-squares denominator with constant 1.

    ``taylor`` reproduces T_d: the square factors of ``taylor_square_factors``,
    h = q1 + i q2, are rotated by the phase of h(0) so the constant of q1 is 1 and
    that of q2 is 0. ``linear`` starts at 1 + (LINEAR_START_SQUARE_FACTOR s)^2.
    """
    _validate_degree(degree, square=True)
    half = degree // 2
    if initialization == "linear":
        return (0.0,) * half + (LINEAR_START_SQUARE_FACTOR,) + (0.0,) * (half - 1)
    if initialization != "taylor":
        raise ValueError(f"Unknown coefficient initialization: {initialization}")
    from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors

    factors = np.asarray(taylor_square_factors(degree))
    h = factors[:half + 1] + 1j * np.concatenate([factors[half + 1:], [0.0]])
    h = h * np.conj(h[0]) / abs(h[0])
    if not (np.isclose(h[0].real, 1.0, atol=1e-12) and abs(h[0].imag) < 1e-12):
        raise AssertionError("The rotated factor must have constant 1")
    result = np.concatenate([h.real[1:], h.imag[1:]])
    expected = np.array([1.0 / math.factorial(order) for order in range(1, degree + 1)])
    if not np.allclose(expand_square_factors(result), expected, rtol=0, atol=1e-12):
        raise AssertionError(f"Square factors do not reproduce T_{degree}")
    return tuple(float(value) for value in result)


def initial_denominator_parameters(degree: int, form: str = "direct", constant: str = "fixed",
                                   initialization: str = "taylor") -> tuple[float, ...]:
    """Initial values of the denominator array selected by ``form`` and ``constant``."""
    if form not in DENOMINATOR_PARAMETERIZATIONS:
        raise ValueError(f"Unknown denominator parameterization: {form}")
    if constant not in DENOMINATOR_CONSTANTS:
        raise ValueError(f"Unknown denominator constant: {constant}")
    if form == "direct":
        coefficients = initial_denominator(degree, initialization)
        return coefficients if constant == "fixed" else (1.0, *coefficients)
    if constant == "fixed":
        return initial_denominator_square_factors(degree, initialization)
    _validate_degree(degree, square=True)
    half = degree // 2
    if initialization == "linear":
        # q1 = q2 = 1/sqrt(2): p_b = 1 exactly, and every factor has a nonzero gradient.
        root = 1.0 / math.sqrt(2.0)
        return (root,) + (0.0,) * half + (root,) + (0.0,) * (half - 1)
    if initialization != "taylor":
        raise ValueError(f"Unknown coefficient initialization: {initialization}")
    from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors

    return taylor_square_factors(degree)


def expand_square_factors(factors) -> np.ndarray:
    """b_1..b_d of (1 + sum u_r s^r)^2 + (sum w_r s^r)^2 for factors u, w on the last axis."""
    factors = np.asarray(factors, dtype=np.float64)
    half = factors.shape[-1] // 2
    ones = np.ones(factors.shape[:-1] + (1,))
    q1 = np.concatenate([ones, factors[..., :half]], axis=-1)
    q2 = np.concatenate([0 * ones, factors[..., half:]], axis=-1)
    out = np.zeros(factors.shape[:-1] + (2 * half + 1,))
    for i in range(half + 1):
        for j in range(half + 1):
            out[..., i + j] += q1[..., i] * q1[..., j] + q2[..., i] * q2[..., j]
    return out[..., 1:]


def parameter_specs(*, degree: int, sharing: str, initialization: str = "taylor",
                    denominator_degree: int | None = None, denominator_parameterization: str = "direct",
                    denominator_constant: str = "fixed"):
    """Numerator a_0..a_m and the denominator's parameters, neither weight-decayed."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    denominator_degree = degree if denominator_degree is None else denominator_degree
    initial = initial_denominator_parameters(denominator_degree, denominator_parameterization,
                                             denominator_constant, initialization)
    name = DENOMINATOR_ARGUMENTS[(denominator_parameterization, denominator_constant)]
    return (
        AttentionParameterSpec(
            name="polynomial_coefficients", argument="polynomial_coefficients",
            initial=initial_coefficients(degree, initialization), sharing=sharing, weight_decay=False,
        ),
        AttentionParameterSpec(name=name, argument=name, initial=initial, sharing=sharing, weight_decay=False),
    )


def select_denominator(heads: int, **denominators):
    """The one denominator array passed by name, as (array, form, constant, degree), or None.

    Arrays are [n] (shared) or [heads, n]. The degree d follows from n: n = d for a fixed
    constant, n = d + 1 for a learned one; sums of squares need even d.
    """
    unknown = set(denominators) - set(DENOMINATOR_KINDS)
    if unknown:
        raise TypeError(f"Unknown denominator arguments: {sorted(unknown)}")
    given = [(name, array) for name, array in denominators.items() if array is not None]
    if not given:
        return None
    if len(given) > 1:
        raise ValueError("Provide at most one denominator array")
    name, array = given[0]
    form, constant = DENOMINATOR_KINDS[name]
    if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[0] != heads):
        raise ValueError("Denominator parameters must have shape (n,) or (heads, n)")
    degree = array.shape[-1] - (constant == "learned")
    _validate_degree(degree, square=form == "sum_of_squares")
    return array, form, constant, degree


def denominator_factors(array: jax.Array, form: str, constant: str, *, dtype=jnp.float32) -> tuple[jax.Array, ...]:
    """Ascending coefficient arrays whose polynomial (direct) or sum of squares is p_b, in ``dtype`` (FP32 in models)."""
    array = array.astype(dtype)
    leading = (*array.shape[:-1], 1)
    ones, zeros = jnp.ones(leading, dtype), jnp.zeros(leading, dtype)
    if form == "direct":
        return (array,) if constant == "learned" else (jnp.concatenate([ones, array], axis=-1),)
    if constant == "learned":
        half = (array.shape[-1] - 1) // 2
        return array[..., :half + 1], array[..., half + 1:]
    half = array.shape[-1] // 2
    return (jnp.concatenate([ones, array[..., :half]], axis=-1),
            jnp.concatenate([zeros, array[..., half:]], axis=-1))


def expanded_denominator(array: jax.Array, form: str, constant: str) -> jax.Array:
    """Ascending coefficients c_0..c_d of p_b along the last axis, differentiable."""
    factors = denominator_factors(array, form, constant)
    if form == "direct":
        return factors[0]
    first, second = factors
    degree = 2 * (first.shape[-1] - 1)
    out = jnp.zeros((*first.shape[:-1], degree + 1), jnp.float32)
    for factor in (first, second):
        for i in range(factor.shape[-1]):
            for j in range(factor.shape[-1]):
                out = out.at[..., i + j].add(factor[..., i] * factor[..., j])
    return out


def full_denominator(denominator: jax.Array) -> jax.Array:
    """Ascending denominator coefficients 1, b_1, ..., b_d along the last axis."""
    return denominator_factors(denominator, "direct", "fixed")[0]


def square_factor_polynomials(factors: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Ascending coefficients (1, u_1, ..., u_k) and (0, w_1, ..., w_k) along the last axis."""
    return denominator_factors(factors, "sum_of_squares", "fixed")


def denominator_names(degree: int, parameterization: str = "direct", constant: str = "fixed") -> list[str]:
    if parameterization == "sum_of_squares":
        half = degree // 2
        if constant == "learned":
            return [f"q1_{order}" for order in range(half + 1)] + [f"q2_{order}" for order in range(half)]
        return [f"u{order}" for order in range(1, half + 1)] + [f"w{order}" for order in range(1, half + 1)]
    return [f"b{order}" for order in range(0 if constant == "learned" else 1, degree + 1)]
