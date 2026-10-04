"""Penalty on learned polynomial coefficients, added to the training loss only.

With weight lambda = ``optim.coefficient_penalty`` the training loss is

    NLL + lambda * sum over polynomials [ a_0^2 + sum_(r>=2) r^2 a_r^2 + sum_(r>=1) r^2 b_r^2 ]

where a_r are the numerator (or the only) coefficients and b_r the denominator
coefficients of a rational kernel. The linear numerator coefficient a_1 is not
penalized; the fixed denominator constant has no parameter. "Polynomials" are the
model's learned coefficient sets: one per layer and head with ``per_head`` sharing,
one in total with ``global``. The validation loss, the recorded training NLL, and
the perplexities never include the penalty; its value is recorded separately.
"""

import jax
import jax.numpy as jnp

from scaffolds.transformer.training.numerics import has_learned_polynomial, has_rational_kernel


def penalty_weights(degree: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Weights for a_0..a_m and for b_1..b_m."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("The coefficient penalty requires an integer degree >= 1")
    numerator = tuple(1.0 if order == 0 else 0.0 if order == 1 else float(order ** 2) for order in range(degree + 1))
    denominator = tuple(float(order ** 2) for order in range(1, degree + 1))
    return numerator, denominator


def coefficient_penalty_sum(params, model) -> jax.Array:
    """The bracketed sum above, unweighted by lambda, over every learned polynomial."""
    parameterization = getattr(model, "attention_coefficient_parameterization", "direct")
    if not has_learned_polynomial(model) or parameterization not in ("direct", "rational"):
        raise ValueError("The coefficient penalty applies to learned direct or rational polynomial coefficients")
    numerator_weights, denominator_weights = penalty_weights(model.attention_degree)
    numerator = params["polynomial_coefficients"][...].astype(jnp.float32)
    total = jnp.sum(jnp.asarray(numerator_weights, jnp.float32) * jnp.square(numerator))
    if has_rational_kernel(model):
        denominator = params["polynomial_denominator_coefficients"][...].astype(jnp.float32)
        total = total + jnp.sum(jnp.asarray(denominator_weights, jnp.float32) * jnp.square(denominator))
    return total
