"""Row-normalized causal attention with a learned polynomial kernel: p(QK^T / sqrt(d_k)) V / rowsum.

The coefficients c_0..c_m are the same model-owned ``polynomial_coefficients`` array
as the unnormalized mechanism's, shared ([m+1]) or per head ([heads, m+1]), and the
kernel may be signed. Weights divide by the signed row sum: negative nonzero sums
are valid, an exactly zero or nonfinite sum poisons its row, as for the unrestricted
quadratic and odd Taylor kernels. Fused CUDA execution evaluates causal tiles and
returns gradients of Q, K, V, and every learned coefficient.

``polynomial_square_factors`` instead parameterizes an even-degree kernel as
q1(s)^2 + q2(s)^2 (see ``nonnegative_polynomial``): the kernel is nonnegative by
construction, and only a zero row sum is invalid.

``polynomial_denominator_coefficients`` (b_1..b_m, see ``rational_polynomial``) adds a
separate denominator 1 + sum_r b_r s^r: the weights are p_a(s) / sum_j p_b(s_j) with
the numerator's coefficients a_0..a_m in ``polynomial_coefficients``. Both polynomials
are signed; only an exactly zero or nonfinite denominator sum is invalid.
"""

import jax.numpy as jnp


def normalized_learned_polynomial_attention(q, k, v, *, attention_degree, polynomial_coefficients=None,
                                            polynomial_square_factors=None,
                                            polynomial_denominator_coefficients=None, backend="blocked",
                                            return_diagnostics=False):
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
    from scaffolds.transformer.attention.nonnegative_polynomial import validate_factors
    from scaffolds.transformer.attention.rational_polynomial import validate_denominator
    from scaffolds.transformer.attention.unnormalized import _validate_coefficients

    if (polynomial_coefficients is None) == (polynomial_square_factors is None):
        raise ValueError("Provide exactly one of polynomial_coefficients and polynomial_square_factors")
    heads = q.shape[2]
    if polynomial_coefficients is not None:
        _validate_coefficients(polynomial_coefficients, attention_degree, heads)
        learned = {"polynomial_coefficients": polynomial_coefficients}
        parameters, parameterization = polynomial_coefficients, "direct"
        if polynomial_denominator_coefficients is not None:
            validate_denominator(polynomial_denominator_coefficients, attention_degree, heads)
            learned["polynomial_denominator_coefficients"] = polynomial_denominator_coefficients
            parameterization = "rational"
    else:
        if polynomial_denominator_coefficients is not None:
            raise ValueError("A rational kernel's numerator is its direct polynomial_coefficients, not square factors")
        validate_factors(polynomial_square_factors, attention_degree, heads)
        learned = {"polynomial_square_factors": polynomial_square_factors}
        parameters, parameterization = polynomial_square_factors, "sum_of_squares"
    if backend == "blocked":
        return blocked_dot_product_attention(
            q, k, v, attention_degree=attention_degree, attention_normalization=True,
            return_diagnostics=return_diagnostics, **learned,
        )
    if backend != "fused":
        raise ValueError("Normalized learned polynomial attention requires backend='blocked' or 'fused'")
    from scaffolds.transformer.attention.backends.fused_normalized_polynomial import (
        fused_normalized_polynomial_attention,
    )

    # Broadcasting a shared array sums its per-head gradients in the VJP.
    parameters = jnp.broadcast_to(parameters, (heads, attention_degree + 1)).astype(jnp.float32)
    if parameterization == "rational":
        denominator = jnp.broadcast_to(polynomial_denominator_coefficients, (heads, attention_degree))
        parameters = jnp.concatenate([parameters, denominator.astype(jnp.float32)], axis=-1)
    output = fused_normalized_polynomial_attention(q, k, v, parameters, parameterization=parameterization)
    if not return_diagnostics:
        return output
    _, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=attention_degree, attention_normalization=True,
        return_diagnostics=True, **learned,
    )
    return output, diagnostics
