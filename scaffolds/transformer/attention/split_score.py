"""Negative-score branches for a learned degree-eight query polynomial, a sum of squares or direct coefficients;
the positive side is the original polynomial."""
from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np


DENOMINATORS = ("positive_quartic", "sos_quartic", "signed_quartic")
BRANCHES = ("none", "exp_times_sos", "exp_times_origin", *DENOMINATORS)
SOS_FLOOR = 1e-4


def validate(model):
    branch = model.attention_negative_score_branch
    if branch not in BRANCHES:
        raise ValueError(f"Unknown negative score branch: {branch}")
    if branch == "none":
        return
    if not (model.attention_backend == "blocked" and model.attention_degree == 8
            and model.attention_coefficients == "learned" and not model.attention_normalization
            and model.attention_coefficient_conditioning == "query"
            and model.attention_coefficient_parameterization in ("sum_of_squares", "direct")
            and model.attention_coefficient_activation == "tanh"
            and model.attention_coefficient_readout == "diagonal"
            and model.attention_coefficient_input_normalization == "none"
            and model.attention_coefficient_input_source == "post_rmsnorm"
            and model.attention_kernel_basis == "polynomial" and model.positional_encoding == "rope"
            and model.attention_coefficient_input_power == 0 and not model.attention_coefficient_zero_constant
            and model.attention_score_prefactor == "none" and model.attention_score_transform == "identity"
            and model.attention_score_bias == "none" and model.attention_coefficient_gate == "none"
            and not model.attention_forget_gate and model.attention_polynomial_shift == "none"):
        raise ValueError("The split score kernels take an unnormalized raw degree-eight query polynomial (SoS or "
                         "direct) on blocked RoPE")


def initial_bias(branch):
    if branch == "sos_quartic":
        # |1 + v1*u + v2*u^2|^2 must equal (1+epsilon)T4(u) - epsilon.
        target = np.array([1., *( (1 + SOS_FLOOR) / math.factorial(r) for r in range(1, 5))])
        roots = np.roots(target[::-1])
        factors = np.polynomial.polynomial.polyfromroots(roots[roots.imag > 0])
        factors /= factors[0]
        return tuple(np.r_[factors[1:].real, factors[1:].imag])
    value = math.log(math.expm1(1.0)) if branch == "positive_quartic" else 1.0
    return (value,) * 4


def parameter_specs(config):
    from scaffolds.transformer.attention.input_dependent import parameter_specs as generator_specs

    # Four generator outputs for the nonconstant terms; no generator coordinate for D(0).
    specs = generator_specs(hidden_size=config.hidden_size, degree=3, activation="tanh",
                            amplitude_initial=config.attention_coefficient_amplitude_initial,
                            projection_initial_std=config.attention_coefficient_projection_initial_std,
                            kernel_weight_decay=config.attention_coefficient_kernel_weight_decay)
    return tuple(replace(spec, name=spec.name.replace("query_polynomial", "negative_quartic"),
                         argument=spec.argument.replace("query_polynomial", "negative_quartic"),
                         initial=initial_bias(config.attention_negative_score_branch)
                         if spec.name == "query_polynomial_bias" else spec.initial) for spec in specs)


def denominator_parameters(raw, branch):
    if branch == "sos_quartic":
        return raw
    values = jax.nn.softplus(raw) if branch == "positive_quartic" else raw
    return values / jnp.asarray([1, 2, 6, 24], values.dtype)


def weights(scores, positive_coefficients, negative_parameters, branch):
    """Rows broadcast over keys; inactive operands see zero, and the score-zero slope uses the positive side."""
    negative = scores < 0
    u = jnp.where(negative, -scores, 0.)
    x = scores if branch == "exp_times_sos" else jnp.where(negative, 0., scores)
    positive = positive_coefficients[-1]
    for coefficient in reversed(positive_coefficients[:-1]):
        positive = positive * x + coefficient
    if branch in ("exp_times_origin", "exp_times_sos"):
        return positive * jnp.exp(-u), None
    a, b, c, d = negative_parameters
    if branch == "sos_quartic":
        real = 1. + u * (a + u * b)
        imag = u * (c + u * d)
        denominator = (SOS_FLOOR + real * real + imag * imag) / (1. + SOS_FLOOR)
    else:
        denominator = 1. + u * (a + u * (b + u * (c + u * d)))
    return jnp.where(negative, positive_coefficients[0] / denominator, positive), denominator
