"""Normalized attention with separate learned numerator and denominator polynomials.

The weights are p_a(s_ij) / sum_(k<=i) p_b(s_ik) with a numerator
p_a(s) = sum_(r<=m) a_r s^r, learned as ``polynomial_coefficients`` (a_0..a_m), and a
denominator of degree d (``attention_denominator_degree``, the numerator's degree when
unset) whose constant is fixed at 1, which removes the common scale of the two:

- ``direct`` (``polynomial_denominator_coefficients``, b_1..b_d):
  p_b(s) = 1 + sum_r b_r s^r. Signed: a negative nonzero row sum is valid, an
  exactly zero or nonfinite sum poisons its row.
- ``sum_of_squares`` (``polynomial_denominator_square_factors``, u_1..u_k then
  w_1..w_k, d = 2k): p_b(s) = (1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2. Every
  nonnegative polynomial of degree 2k with constant 1 has this form (write it as
  |h|^2 for a complex h of degree k and rotate h so h(0) = 1), so the
  parameterization is complete. Values are nonnegative; only a zero row sum is
  invalid.

Parameters are per layer and head, or shared. Initialization follows
``attention_coefficient_initialization``: ``taylor`` makes both polynomials T_m and
T_d (sum_r s^r / r!); ``linear`` sets a_1 = 1 and the other numerator coefficients
to 0, and the denominator to 1, so attention starts at sum_(k<=i) s_ik v_k / (i + 1).
A sum-of-squares denominator cannot start exactly at 1 and still move: with every
w_r = 0 the gradient of every w_r is zero. Its linear start therefore sets
w_1 = LINEAR_START_SQUARE_FACTOR, so p_b(s) = 1 + 1e-6 s^2 at initialization.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.unnormalized import initial_coefficients

DENOMINATOR_PARAMETERIZATIONS = ("direct", "sum_of_squares")
DENOMINATOR_ARGUMENTS = {"direct": "polynomial_denominator_coefficients",
                         "sum_of_squares": "polynomial_denominator_square_factors"}
LINEAR_START_SQUARE_FACTOR = 1e-3


def denominator_degree(model) -> int:
    """The denominator's degree d: ``attention_denominator_degree``, or the numerator's when unset."""
    degree = getattr(model, "attention_denominator_degree", None)
    return model.attention_degree if degree is None else degree


def denominator_parameterization(model) -> str:
    return getattr(model, "attention_denominator_parameterization", "direct")


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
    """Initial u_1..u_k, w_1..w_k (d = 2k) of a sum-of-squares denominator.

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
                    denominator_degree: int | None = None, denominator_parameterization: str = "direct"):
    """Numerator a_0..a_m and the denominator's d parameters, neither weight-decayed."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    denominator_degree = degree if denominator_degree is None else denominator_degree
    if denominator_parameterization == "direct":
        initial = initial_denominator(denominator_degree, initialization)
    elif denominator_parameterization == "sum_of_squares":
        initial = initial_denominator_square_factors(denominator_degree, initialization)
    else:
        raise ValueError(f"Unknown denominator parameterization: {denominator_parameterization}")
    name = DENOMINATOR_ARGUMENTS[denominator_parameterization]
    return (
        AttentionParameterSpec(
            name="polynomial_coefficients", argument="polynomial_coefficients",
            initial=initial_coefficients(degree, initialization), sharing=sharing, weight_decay=False,
        ),
        AttentionParameterSpec(name=name, argument=name, initial=initial, sharing=sharing, weight_decay=False),
    )


def validate_denominator(denominator, heads, *, square=False) -> int:
    """Check a [d] or [heads, d] denominator array and return d."""
    if denominator.ndim not in (1, 2) or (denominator.ndim == 2 and denominator.shape[0] != heads):
        raise ValueError("Denominator parameters must have shape (d,) or (heads, d)")
    degree = denominator.shape[-1]
    _validate_degree(degree, square=square)
    return degree


def full_denominator(denominator: jax.Array) -> jax.Array:
    """Ascending denominator coefficients 1, b_1, ..., b_d along the last axis."""
    denominator = denominator.astype(jnp.float32)
    return jnp.concatenate([jnp.ones((*denominator.shape[:-1], 1), jnp.float32), denominator], axis=-1)


def square_factor_polynomials(factors: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Ascending coefficients (1, u_1, ..., u_k) and (0, w_1, ..., w_k) along the last axis."""
    factors = factors.astype(jnp.float32)
    half = factors.shape[-1] // 2
    leading = (*factors.shape[:-1], 1)
    return (jnp.concatenate([jnp.ones(leading, jnp.float32), factors[..., :half]], axis=-1),
            jnp.concatenate([jnp.zeros(leading, jnp.float32), factors[..., half:]], axis=-1))


def denominator_names(degree: int, parameterization: str = "direct") -> list[str]:
    if parameterization == "sum_of_squares":
        half = degree // 2
        return [f"u{order}" for order in range(1, half + 1)] + [f"w{order}" for order in range(1, half + 1)]
    return [f"b{order}" for order in range(1, degree + 1)]
