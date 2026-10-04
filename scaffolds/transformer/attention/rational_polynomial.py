"""Normalized attention with separate learned numerator and denominator polynomials.

The weights are p_a(s_ij) / sum_(j<=i) p_b(s_ij) with a numerator
p_a(s) = sum_(r<=m) a_r s^r and a denominator p_b(s) = 1 + sum_(1<=r<=m) b_r s^r.
Both are learned per layer and head (or shared): ``polynomial_coefficients`` holds
a_0..a_m and ``polynomial_denominator_coefficients`` holds b_1..b_m; the constant
b_0 = 1 is fixed, which removes the common scale of the two polynomials. Both
kernels are signed: a negative nonzero row sum of p_b is valid, an exactly zero or
nonfinite sum poisons its row, as for the shared-polynomial learned kernel.

Initialization follows ``attention_coefficient_initialization``: ``taylor`` sets
a_r = b_r = 1/r!, so the kernel starts at normalized Taylor attention; ``linear``
sets a_1 = 1 and every other coefficient to 0, so it starts at linear attention
divided by the number of visible keys, sum_(j<=i) s_ij v_j / (i + 1).
"""

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.unnormalized import initial_coefficients


def initial_denominator(degree: int, initialization: str = "taylor") -> tuple[float, ...]:
    """Initial b_1..b_m (the constant b_0 = 1 is not a parameter): 1/r! for Taylor, zeros for linear."""
    coefficients = initial_coefficients(degree, initialization)
    if initialization == "linear":
        return (0.0,) * degree
    return coefficients[1:]


def parameter_specs(*, degree: int, sharing: str, initialization: str = "taylor"):
    """Numerator a_0..a_m and denominator b_1..b_m, neither weight-decayed."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (
        AttentionParameterSpec(
            name="polynomial_coefficients", argument="polynomial_coefficients",
            initial=initial_coefficients(degree, initialization), sharing=sharing, weight_decay=False,
        ),
        AttentionParameterSpec(
            name="polynomial_denominator_coefficients", argument="polynomial_denominator_coefficients",
            initial=initial_denominator(degree, initialization), sharing=sharing, weight_decay=False,
        ),
    )


def validate_denominator(denominator, degree, heads) -> None:
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("A rational kernel requires an integer attention_degree >= 1")
    if denominator.shape not in ((degree,), (heads, degree)):
        raise ValueError("Denominator coefficients must have shape (m,) or (heads, m)")


def full_denominator(denominator: jax.Array) -> jax.Array:
    """Ascending denominator coefficients 1, b_1, ..., b_m along the last axis."""
    denominator = denominator.astype(jnp.float32)
    return jnp.concatenate([jnp.ones((*denominator.shape[:-1], 1), jnp.float32), denominator], axis=-1)


def denominator_names(degree: int) -> list[str]:
    return [f"b{order}" for order in range(1, degree + 1)]
