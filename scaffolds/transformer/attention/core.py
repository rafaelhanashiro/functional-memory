"""Complete attention operations on post-RoPE Q/K/V arrays.

Mechanisms consume [batch, time, heads, channels] arrays and return an array
with V's shape. They own score/kernel mathematics and backend dispatch; the
Transformer owns projections and model parameters. Dispatch uses static Python
configuration and disappears when JAX traces the training step.
"""

import jax


def attention_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    backend: str = "blocked",
    attention_degree: int | None = None,
    attention_normalization: bool = True,
    quadratic_mode: str = "fixed",
    coefficients: jax.Array | None = None,
    polynomial_coefficients: jax.Array | None = None,
    polynomial_square_factors: jax.Array | None = None,
    query_polynomial_coefficients: jax.Array | None = None,
    key_polynomial_factors: jax.Array | None = None,
    attention_coefficient_activation: str = "identity",
    return_diagnostics: bool = False,
    dropout=None,
    deterministic: bool = True,
    rngs=None,
    weights_dtype=None,
    score_kernel=None,
    **denominators,
):
    """Apply the mechanism selected by the existing flat model configuration.

    ``denominators`` holds at most one rational-kernel denominator array, named as in
    ``rational_polynomial.DENOMINATOR_KINDS``.
    """
    from scaffolds.transformer.attention.rational_polynomial import DENOMINATOR_KINDS

    unknown = set(denominators) - set(DENOMINATOR_KINDS)
    if unknown:
        raise TypeError(f"Unknown attention arguments: {sorted(unknown)}")
    denominators = {name: array for name, array in denominators.items() if array is not None}
    if not isinstance(attention_normalization, bool):
        raise ValueError("attention_normalization must be a boolean")
    if score_kernel is not None:
        from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

        if (not callable(score_kernel) or attention_degree is not None or quadratic_mode != "fixed"
                or coefficients is not None or polynomial_coefficients is not None or polynomial_square_factors is not None
                or query_polynomial_coefficients is not None or key_polynomial_factors is not None or denominators):
            raise ValueError("A callable scalar score kernel cannot be combined with polynomial options")
        if backend == "triton_scalar":
            from scaffolds.transformer.attention.backends.triton_scalar import basis_kernel, scalar_attention
            output = scalar_attention(q, k, v, kernel=basis_kernel(score_kernel, implementation="auto"),
                                      attention_normalization=attention_normalization)
            if not return_diagnostics:
                return output
            _, diagnostics = blocked_dot_product_attention(
                q, k, v, score_kernel=score_kernel, attention_normalization=attention_normalization,
                return_diagnostics=True)
            return output, diagnostics
        if backend != "blocked":
            raise ValueError("A custom scalar score kernel requires blocked or triton_scalar execution")
        return blocked_dot_product_attention(
            q, k, v, score_kernel=score_kernel, attention_normalization=attention_normalization,
            attention_degree=attention_degree, quadratic_mode=quadratic_mode, coefficients=coefficients,
            polynomial_coefficients=polynomial_coefficients, polynomial_square_factors=polynomial_square_factors,
            query_polynomial_coefficients=query_polynomial_coefficients, key_polynomial_factors=key_polynomial_factors,
            return_diagnostics=return_diagnostics, **denominators,
        )
    if key_polynomial_factors is not None and query_polynomial_coefficients is None:
        raise ValueError("Key factors require query coefficients")
    if query_polynomial_coefficients is not None:
        from scaffolds.transformer.attention.input_dependent import (
            query_key_polynomial_attention, query_linear_attention, query_polynomial_attention,
        )

        if (attention_normalization or attention_degree is None or attention_degree < 1 or quadratic_mode != "fixed"
                or coefficients is not None or polynomial_coefficients is not None
                or polynomial_square_factors is not None or denominators):
            raise ValueError("Query coefficients define an unnormalized polynomial kernel alone")
        if query_polynomial_coefficients.shape[-1] != attention_degree + 1:
            raise ValueError("Query coefficient count must match attention degree")
        if key_polynomial_factors is not None:
            if attention_coefficient_activation != "tanh":
                raise ValueError("Key factors require tanh query coefficients")
            return query_key_polynomial_attention(q, k, v, query_polynomial_coefficients, key_polynomial_factors,
                                                  backend=backend, return_diagnostics=return_diagnostics)
        if attention_coefficient_activation == "identity":
            if attention_degree != 1:
                raise ValueError("Identity query coefficients require degree-one attention")
            operation = query_linear_attention
        elif attention_coefficient_activation == "tanh":
            operation = query_polynomial_attention
        else:
            raise ValueError("Unknown attention coefficient activation")
        return operation(q, k, v, query_polynomial_coefficients, backend=backend,
                         return_diagnostics=return_diagnostics)
    if not attention_normalization:
        from scaffolds.transformer.attention.unnormalized import unnormalized_attention

        if quadratic_mode != "fixed" or coefficients is not None:
            raise ValueError("Unnormalized attention requires the exponential or a fixed Taylor kernel")
        if polynomial_square_factors is not None:
            raise ValueError("A sum-of-squares kernel is implemented for row-normalized attention only")
        if denominators:
            raise ValueError("A rational kernel's denominator is its row sum; it requires normalized attention")
        return unnormalized_attention(
            q, k, v, attention_degree=attention_degree, backend=backend,
            return_diagnostics=return_diagnostics, polynomial_coefficients=polynomial_coefficients,
        )
    if polynomial_coefficients is not None or polynomial_square_factors is not None or denominators:
        from scaffolds.transformer.attention.learned_polynomial import normalized_learned_polynomial_attention

        if quadratic_mode != "fixed" or coefficients is not None:
            raise ValueError("Learned polynomial coefficients exclude the learned quadratic modes")
        return normalized_learned_polynomial_attention(
            q, k, v, attention_degree=attention_degree, polynomial_coefficients=polynomial_coefficients,
            polynomial_square_factors=polynomial_square_factors,
            backend=backend, return_diagnostics=return_diagnostics, **denominators,
        )
    if quadratic_mode != "fixed":
        from scaffolds.transformer.attention.learned_quadratic import learned_quadratic_attention

        if attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        return learned_quadratic_attention(
            q, k, v, coefficients=coefficients, quadratic_mode=quadratic_mode,
            backend=backend, return_diagnostics=return_diagnostics,
        )
    if coefficients is not None:
        raise ValueError("Fixed attention does not accept learned coefficients")
    if attention_degree is not None:
        from scaffolds.transformer.attention.taylor import taylor_attention

        return taylor_attention(
            q, k, v, attention_degree=attention_degree, backend=backend,
            return_diagnostics=return_diagnostics,
        )
    from scaffolds.transformer.attention.softmax import softmax_attention

    return softmax_attention(
        q, k, v, backend=backend, return_diagnostics=return_diagnostics,
        dropout=dropout, deterministic=deterministic, rngs=rngs,
        weights_dtype=weights_dtype,
    )
