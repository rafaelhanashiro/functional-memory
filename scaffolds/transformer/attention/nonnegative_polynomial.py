"""Nonnegative learned polynomial kernels written as a sum of two squares.

Every polynomial of even degree m = 2k that is nonnegative on the real line is
p(s) = q1(s)^2 + q2(s)^2 with deg q1 = k and deg q2 = k - 1. Writing
p = |h|^2 for a complex degree-k polynomial h whose leading coefficient is real
makes the imaginary part of h, q2, one degree lower. The m + 1 real
coefficients of q1 and q2 are the learned parameters, one set per layer and head,
the same count as learning c_0..c_m directly; the scale is redundant under row
normalization, as it is for the direct coefficients.

The normalized static kernel is evaluated as q1(s)^2 + q2(s)^2, never through the expanded
coefficients: expanding a polynomial whose minimum is near zero cancels in
floating point and can produce small negative values, which would break the
nonnegativity the parameterization exists to guarantee. Individual kernel values
may be exactly zero (at a common real root of q1 and q2); only a row sum of zero
is invalid.

The unnormalized static and query-conditioned paths expand their factors into FP32 coefficients
for the query-polynomial kernels. It can therefore round slightly negative near roots;
its numerical diagnostics treat it as signed. With input_power=1 the expansion is
multiplied by z through a structural leading zero, representing z h(z), z=softplus(s).

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


def static_square_factors(stored: jax.Array, degree: int, *, parameterization: str = "direct") -> jax.Array:
    """Resolve stored static factors, retaining the historical direct parameterization.

    ``relative_taylor`` stores b initialized to one and represents F_j=f_j(1+b_j)/2,
    exactly the query generator's factor map with its input-dependent branch removed.
    This preserves the query bias's optimizer scale even for tiny high-order factors.
    """
    _validate_degree(degree)
    if stored.shape[-1] != degree + 1:
        raise ValueError("Static square factors must have core_degree + 1 entries")
    if parameterization == "direct":
        return stored
    if parameterization != "relative_taylor":
        raise ValueError(f"Unknown static SoS parameterization: {parameterization}")
    return jnp.asarray(taylor_square_factors(degree), jnp.float32) * (1 + stored.astype(jnp.float32)) / 2


def static_square_coefficients(factors: jax.Array, degree: int, *, input_power: int = 0) -> jax.Array:
    """Expand actual static factors of z^input_power (q1(z)^2+q2(z)^2) in FP32.

    ``degree`` is the total degree, including the structural prefactor. Broadcasting
    the resulting per-head coefficients over query tokens sums their gradients back
    to the same static factor parameters; no token-dependent parameters are created.
    """
    if type(input_power) is not int or input_power not in (0, 1):
        raise ValueError("input_power must be the integer 0 or 1")
    core_degree = degree - input_power
    _validate_degree(core_degree)
    if factors.shape[-1] != core_degree + 1:
        raise ValueError("Static square factors must have core_degree + 1 entries")
    factors = factors.astype(jnp.float32)
    terms = [jnp.zeros(factors.shape[:-1], jnp.float32) for _ in range(core_degree + 1)]
    for part in split_factors(factors, core_degree):
        for i in range(part.shape[-1]):
            for j in range(part.shape[-1]):
                terms[i + j] = terms[i + j] + part[..., i] * part[..., j]
    if input_power:
        terms.insert(0, jnp.zeros_like(terms[0]))
    return jnp.stack(terms, axis=-1)


def unnormalized_square_attention(q, k, v, factors, *, attention_degree: int, input_power: int = 0,
                                  score_transform: str = "identity", backend: str = "blocked",
                                  return_diagnostics: bool = False, blocked_implementation: str = "scan"):
    """Static SoS on the existing unnormalized query-polynomial kernels with equal rows."""
    from scaffolds.transformer.attention.input_dependent import query_polynomial_attention

    if backend not in ("blocked", "mosaic"):
        raise ValueError("Unnormalized static SoS uses backend='blocked' or 'mosaic'")
    if score_transform not in ("identity", "softplus") or input_power and score_transform != "softplus":
        raise ValueError("Static SoS takes identity or softplus scores, with its prefactor on softplus only")
    validate_factors(factors, attention_degree - input_power, q.shape[2])
    coefficients = static_square_coefficients(factors, attention_degree, input_power=input_power)
    coefficients = jnp.broadcast_to(coefficients, (q.shape[2], attention_degree + 1))
    coefficients = jnp.broadcast_to(coefficients[None, None], (*q.shape[:3], attention_degree + 1))
    return query_polynomial_attention(q, k, v, coefficients, backend=backend,
                                      score_transform=score_transform, return_diagnostics=return_diagnostics,
                                      blocked_implementation=blocked_implementation)


def query_square_coefficients(generated: jax.Array, degree: int, *, input_power: int = 0,
                              zero_constant: bool = False, pure: bool = False) -> jax.Array:
    """Actual coefficients of z^input_power (q1(z)^2 + q2(z)^2), of TOTAL degree ``degree``.

    ``generated`` contains g_j=b_j+a_j tanh(W_j^T x), with core_degree+1 factors,
    core_degree=degree-input_power. Each factor is f_j (1+g_j)/2 for the fixed Taylor
    square factors f. Thus g=1 starts h at T_core_degree; input_power=1 starts p at
    z T_core_degree(z), not T_degree(z). The returned FP32 coefficients already include
    factorial order scaling. Padding fixes c_0 exactly to zero without a learned parameter.

    With ``zero_constant=True``, ``generated`` has core_degree-1 entries: only
    U_1..U_k and V_1..V_(k-1). Their fixed f_j retain the original Taylor factors;
    U_0,V_0 are structural zeros before expansion. Both expanded c_0 and c_1
    therefore stay exactly zero. This variant has no prefactor and does not
    initialize to T_core_degree: it removes the original factors' constants.

    With ``pure=True``, ``generated`` is each factor's fraction of its Taylor factor,
    g_j=tanh(W_j^T x+beta_j), so the factor is f_j g_j with |f_j g_j| < |f_j|. A common
    start g (tanh(beta) for W=0) starts h at g^2 T_core_degree, which per-head RMSNorm
    output scaling does not distinguish from T_core_degree.
    """
    if type(input_power) is not int or input_power not in (0, 1):
        raise ValueError("input_power must be the integer 0 or 1")
    if type(degree) is not int:
        raise ValueError("The total polynomial degree must be an integer")
    core_degree = degree - input_power
    _validate_degree(core_degree)
    if type(zero_constant) is not bool or (zero_constant and input_power):
        raise ValueError("Zero factor constants require a boolean flag and no input prefactor")
    if type(pure) is not bool or (pure and zero_constant):
        raise ValueError("Pure factor fractions require a boolean flag and every factor constant")
    if generated.shape[-1] != core_degree + 1 - 2 * int(zero_constant):
        raise ValueError("Generated square factors must match the active factor coordinates" if zero_constant else
                         "Generated square factors must have core_degree + 1 entries")
    fixed = jnp.asarray(taylor_square_factors(core_degree), jnp.float32)
    if zero_constant:
        indices = jnp.asarray([j for j in range(core_degree + 1) if j not in (0, core_degree // 2 + 1)])
        active = fixed[indices] * (1 + generated.astype(jnp.float32)) / 2
        factors = jnp.zeros((*generated.shape[:-1], core_degree + 1), jnp.float32).at[..., indices].set(active)
    elif pure:
        factors = fixed * generated.astype(jnp.float32)
    else:
        factors = fixed * (1 + generated.astype(jnp.float32)) / 2
    terms = [jnp.zeros(factors.shape[:-1], jnp.float32) for _ in range(core_degree + 1)]
    for part in split_factors(factors, core_degree):
        for i in range(part.shape[-1]):
            for j in range(part.shape[-1]):
                terms[i + j] = terms[i + j] + part[..., i] * part[..., j]
    if input_power:
        terms.insert(0, jnp.zeros_like(terms[0]))
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


def parameter_specs(*, degree: int, sharing: str, parameterization: str = "direct"):
    """One factor set per layer and head (or shared), initialized to reproduce T_m."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    initial = taylor_square_factors(degree)
    if parameterization == "relative_taylor":
        initial = (1.0,) * len(initial)
    elif parameterization != "direct":
        raise ValueError(f"Unknown static SoS parameterization: {parameterization}")
    return (AttentionParameterSpec(
        name="polynomial_square_factors", argument="polynomial_square_factors",
        initial=initial, sharing=sharing, weight_decay=False,
    ),)


def factor_names(degree: int) -> list[str]:
    half = _validate_degree(degree)
    return [f"q1_{order}" for order in range(half + 1)] + [f"q2_{order}" for order in range(half)]
