"""Penalty on learned polynomial coefficients, added to the training loss only.

With weight lambda = ``optim.coefficient_penalty`` the training loss is

    NLL + lambda * sum over polynomials [ a_0^2 + sum_(r>=2) r^2 a_r^2 + sum_(r>=1) r^2 b_r^2 ]

where a_r are the numerator (or the only) coefficients and b_r the denominator
coefficients of a rational kernel, which ``optim.coefficient_penalty_denominator:
false`` leaves out. For a sum-of-squares denominator, b_r are the coefficients of the
expanded polynomial, differentiated through its factors. The linear numerator
coefficient a_1 is not penalized; the denominator penalty requires the constant fixed
at b_0 = 1. "Polynomials" are the
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
    from scaffolds.transformer.attention.rational_polynomial import (
        denominator_argument, denominator_constant, denominator_degree, denominator_parameterization,
        expanded_denominator,
    )

    parameterization = getattr(model, "attention_coefficient_parameterization", "direct")
    if getattr(model, "attention_coefficient_scale", "none") != "none":
        raise ValueError("The coefficient penalty is defined on unscaled coefficients")
    if not has_learned_polynomial(model) or parameterization not in ("direct", "rational"):
        raise ValueError("The coefficient penalty applies to learned direct or rational polynomial coefficients")
    rational = has_rational_kernel(model)
    numerator_weights, denominator_weights = penalty_weights(
        model.attention_degree, denominator_degree(model) if rational else None)
    numerator = params["polynomial_coefficients"][...].astype(jnp.float32)
    from scaffolds.transformer.attention.parameters import coefficient_coordinates, expand_polynomial_coefficients

    numerator = expand_polynomial_coefficients(numerator, model.attention_degree,
                                               coefficient_coordinates(model))
    total = jnp.sum(jnp.asarray(numerator_weights, jnp.float32) * jnp.square(numerator))
    if rational and include_denominator:
        if denominator_constant(model) != "fixed":
            raise ValueError("The denominator penalty is defined with its constant fixed at 1")
        denominator = expanded_denominator(params[denominator_argument(model)][...],
                                           denominator_parameterization(model), "fixed")[..., 1:]
        total = total + jnp.sum(jnp.asarray(denominator_weights, jnp.float32) * jnp.square(denominator))
    return total
