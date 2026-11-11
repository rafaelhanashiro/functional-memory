"""Nonnegative learned polynomial kernels written as a sum of two squares.

Every polynomial of even degree m = 2k that is nonnegative on the real line is
p(s) = q1(s)^2 + q2(s)^2 with deg q1 = k and deg q2 = k - 1. Writing
p = |h|^2 for a complex degree-k polynomial h whose leading coefficient is real
makes the imaginary part of h, q2, one degree lower. The m + 1 real
coefficients of q1 and q2 are the learned parameters, one set per layer and head,
the same count as learning c_0..c_m directly; the scale is redundant under row
normalization, as it is for the direct coefficients.

The kernel is evaluated as q1(s)^2 + q2(s)^2, never through the expanded
coefficients: expanding a polynomial whose minimum is near zero cancels in
floating point and can produce small negative values, which would break the
nonnegativity the parameterization exists to guarantee. Individual kernel values
may be exactly zero (at a common real root of q1 and q2); only a row sum of zero
is invalid.

Factors are stored in ascending powers, q1 first: [q1_0, ..., q1_k, q2_0, ..., q2_{k-1}].
They are initialized so that q1^2 + q2^2 equals the Taylor polynomial
T_m(s) = sum_{r<=m} s^r / r! exactly (to rounding).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def _validate_degree(degree) -> int:
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2 or degree % 2:
        raise ValueError("A sum-of-squares polynomial kernel requires an even integer degree >= 2")
    return degree // 2


def taylor_square_factors(degree: int) -> tuple[float, ...]:
    """Factors q1, q2 (ascending powers) with q1(s)^2 + q2(s)^2 = T_m(s).

    T_m has no real roots for even m. Its roots come in conjugate pairs; h takes
    one root from each pair, choosing the upper root for the pairs ranked 1st, 3rd,
    ... and the lower root for the 2nd, 4th, ... when ordered by decreasing
    imaginary part, then q1 = Re h and q2 = Im h. For m = 2 this gives
    q1 = (1 + s) / sqrt(2), q2 = 1 / sqrt(2); for m = 4 it is the decomposition
    whose quadratic factor has all-positive coefficients. Signs are fixed so each
    factor's constant term is positive.
    """
    half = _validate_degree(degree)
    taylor = np.array([1.0 / math.factorial(order) for order in range(degree, -1, -1)])  # descending powers
    roots = np.roots(taylor)
    upper = sorted((root for root in roots if root.imag > 0), key=lambda root: -root.imag)
    if len(upper) != half:
        raise ValueError(f"T_{degree} must have {half} conjugate root pairs")
    chosen = [root if index % 2 == 0 else np.conj(root) for index, root in enumerate(upper)]
    h = math.sqrt(taylor[0]) * np.poly(chosen)  # descending powers, real leading coefficient
    q1, q2 = h.real[::-1], h.imag[::-1][:half]  # ascending; the s^k term of Im h is zero
    if not np.isclose(h.imag[0], 0.0, atol=1e-12):
        raise AssertionError("The leading coefficient of h must be real")
    q1 = q1 if q1[0] > 0 else -q1
    q2 = q2 if q2[0] > 0 else -q2
    factors = np.concatenate([q1, q2])
    if not np.allclose(effective_coefficients_numpy(factors), taylor[::-1], rtol=0, atol=1e-12):
        raise AssertionError(f"Square factors do not reproduce T_{degree}")
    return tuple(float(value) for value in factors)


def split_factors(factors, degree: int):
    """(q1, q2) along the last axis: q1 has k + 1 ascending coefficients, q2 has k."""
    half = _validate_degree(degree)
    return factors[..., :half + 1], factors[..., half + 1:]


def effective_coefficients_numpy(factors, degree: int | None = None) -> np.ndarray:
    """Coefficients c_0..c_m of q1^2 + q2^2 (ascending), for any leading batch shape."""
    factors = np.asarray(factors, dtype=np.float64)
    degree = factors.shape[-1] - 1 if degree is None else degree
    q1, q2 = split_factors(factors, degree)
    out = np.zeros(factors.shape[:-1] + (degree + 1,))
    for i in range(q1.shape[-1]):
        for j in range(q1.shape[-1]):
            out[..., i + j] += q1[..., i] * q1[..., j]
    for i in range(q2.shape[-1]):
        for j in range(q2.shape[-1]):
            out[..., i + j] += q2[..., i] * q2[..., j]
    return out


def query_square_coefficients(generated: jax.Array, degree: int) -> jax.Array:
    """Per-query c_0..c_m of q1^2 + q2^2 from generated values g = b + a tanh(W x) [..., m + 1].

    Factor k is f_k (1 + g_k) / 2 for the Taylor square factors f, so g = 1 is T_m and b and a keep
    the direct form's first-order effect on the polynomial: a step in g_k moves q_k half as far
    in relative terms, and squaring doubles it. The kernels evaluate the expanded coefficients
    in FP32, as they do every other per-query polynomial.
    """
    factors = jnp.asarray(taylor_square_factors(degree), jnp.float32) * (1.0 + generated.astype(jnp.float32)) / 2.0
    terms = [jnp.zeros(factors.shape[:-1], jnp.float32) for _ in range(degree + 1)]
    for part in split_factors(factors, degree):
        for i in range(part.shape[-1]):
            for j in range(part.shape[-1]):
                terms[i + j] = terms[i + j] + part[..., i] * part[..., j]
    return jnp.stack(terms, axis=-1)


def _horner(scores: jax.Array, coefficients: jax.Array) -> jax.Array:
    """Evaluate ascending per-head coefficients [heads, n] over [batch, heads, queries, keys] scores."""
    value = jnp.broadcast_to(coefficients[:, -1][None, :, None, None], scores.shape)
    for order in range(coefficients.shape[-1] - 2, -1, -1):
        value = value * scores + coefficients[:, order][None, :, None, None]
    return value


def square_sum_polynomial(scores: jax.Array, factors: jax.Array, degree: int, *, dtype=jnp.float32) -> jax.Array:
    """q1(s)^2 + q2(s)^2 in FP32 (``dtype``) for shared ([m+1]) or per-head ([heads, m+1]) factors."""
    factors = jnp.broadcast_to(factors, (scores.shape[1], degree + 1)).astype(dtype)
    q1, q2 = split_factors(factors, degree)
    return jnp.square(_horner(scores, q1)) + jnp.square(_horner(scores, q2))


def validate_factors(factors, degree, heads) -> None:
    _validate_degree(degree)
    if factors.shape not in ((degree + 1,), (heads, degree + 1)):
        raise ValueError("Square factors must have shape (m+1,) or (heads, m+1)")


def parameter_specs(*, degree: int, sharing: str):
    """One factor set per layer and head (or shared), initialized to reproduce T_m."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (AttentionParameterSpec(
        name="polynomial_square_factors", argument="polynomial_square_factors",
        initial=taylor_square_factors(degree), sharing=sharing, weight_decay=False,
    ),)


def factor_names(degree: int) -> list[str]:
    half = _validate_degree(degree)
    return [f"q1_{order}" for order in range(half + 1)] + [f"q2_{order}" for order in range(half)]
