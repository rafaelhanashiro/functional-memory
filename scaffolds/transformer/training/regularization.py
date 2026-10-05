"""Penalty on learned polynomial coefficients, added to the training loss only.

With weight lambda = ``optim.coefficient_penalty`` the training loss is

    NLL + lambda * sum over polynomials [ a_0^2 + sum_(r>=2) r^2 a_r^2 + sum_(r>=1) r^2 b_r^2 ]

where a_r are the numerator (or the only) coefficients and b_r the direct
denominator coefficients of a rational kernel, which ``optim.coefficient_penalty_denominator:
false`` leaves out. The linear numerator coefficient a_1 is not penalized; the fixed
denominator constant has no parameter. "Polynomials" are the
model's learned coefficient sets: one per layer and head with ``per_head`` sharing,
one in total with ``global``. The validation loss, the recorded training NLL, and
the perplexities never include the penalty; its value is recorded separately.
"""

import jax
import jax.numpy as jnp

from scaffolds.transformer.training.numerics import has_learned_polynomial, has_rational_kernel


def penalty_weights(degree: int, denominator_degree: int | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Weights for a_0..a_m and for b_1..b_d (d = m unless given)."""
    denominator_degree = degree if denominator_degree is None else denominator_degree
    for value in (degree, denominator_degree):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("The coefficient penalty requires integer degrees >= 1")
    numerator = tuple(1.0 if order == 0 else 0.0 if order == 1 else float(order ** 2) for order in range(degree + 1))
    denominator = tuple(float(order ** 2) for order in range(1, denominator_degree + 1))
    return numerator, denominator


def coefficient_penalty_sum(params, model, *, include_denominator: bool = True) -> jax.Array:
    """The bracketed sum above, unweighted by lambda, over every learned polynomial."""
    from scaffolds.transformer.attention.rational_polynomial import denominator_degree, denominator_parameterization

    parameterization = getattr(model, "attention_coefficient_parameterization", "direct")
    if not has_learned_polynomial(model) or parameterization not in ("direct", "rational"):
        raise ValueError("The coefficient penalty applies to learned direct or rational polynomial coefficients")
    rational = has_rational_kernel(model)
    numerator_weights, denominator_weights = penalty_weights(
        model.attention_degree, denominator_degree(model) if rational else None)
    numerator = params["polynomial_coefficients"][...].astype(jnp.float32)
    total = jnp.sum(jnp.asarray(numerator_weights, jnp.float32) * jnp.square(numerator))
    if rational and include_denominator:
        if denominator_parameterization(model) != "direct":
            raise ValueError("The coefficient penalty is defined on direct denominator coefficients")
        denominator = params["polynomial_denominator_coefficients"][...].astype(jnp.float32)
        total = total + jnp.sum(jnp.asarray(denominator_weights, jnp.float32) * jnp.square(denominator))
    return total
