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
    query_log_decay: jax.Array | None = None,
    query_decay_mode: str = "cumulative",
    query_gate_input: jax.Array | None = None,
    key_gate_input: jax.Array | None = None,
    gate_log_rate: jax.Array | None = None,
    spline_knots: tuple[float, ...] | None = None,
    spline_coefficients: jax.Array | None = None,
    query_spline_coefficients: jax.Array | None = None,
    score_bias_row_shift: jax.Array | None = None,
    score_bias_query_input: jax.Array | None = None,
    score_bias_key_input: jax.Array | None = None,
    score_bias_offset: jax.Array | None = None,
    score_bias_amplitude: jax.Array | None = None,
    attention_coefficient_activation: str = "identity",
    return_diagnostics: bool = False,
    dropout=None,
    deterministic: bool = True,
    rngs=None,
    weights_dtype=None,
    blocked_implementation: str = "scan",
    score_transform: str = "identity",
    **denominators,
):
    """Apply the mechanism selected by the existing flat model configuration.

    ``denominators`` holds at most one rational-kernel denominator array, named as in
    ``rational_polynomial.DENOMINATOR_KINDS``. ``blocked_implementation`` selects the blocked
    backend's execution ("scan" or "strips"; see ``backends.reference``); other backends ignore it.
    ``spline_knots`` with static ``spline_coefficients`` or ``query_spline_coefficients`` selects
    the B-spline score function (``spline``), which defines the kernel alone. The ``score_bias_*``
    operands (attention/score_bias.py: a row shift, or the query-key pair inputs with their offset
    and amplitude) shift each order r >= 1 of tanh query coefficients; every other kernel rejects them.
    ``score_transform`` (attention/score_transform.py) evaluates tanh query coefficients' polynomial at
    z = phi(s), without a forget gate, key factors or score biases, on the blocked backend; every other
    mechanism and backend rejects a transform other than "identity".
    """
    from scaffolds.transformer.attention.backends.reference import BLOCKED_IMPLEMENTATIONS
    from scaffolds.transformer.attention.rational_polynomial import DENOMINATOR_KINDS
    from scaffolds.transformer.attention.score_transform import validate as validate_transform

    unknown = set(denominators) - set(DENOMINATOR_KINDS)
    if unknown:
        raise TypeError(f"Unknown attention arguments: {sorted(unknown)}")
    if blocked_implementation not in BLOCKED_IMPLEMENTATIONS:
        raise ValueError(f"Unknown blocked attention implementation: {blocked_implementation}")
    score_bias = {name: value for name, value in (
        ("score_bias_row_shift", score_bias_row_shift), ("score_bias_query_input", score_bias_query_input),
        ("score_bias_key_input", score_bias_key_input), ("score_bias_offset", score_bias_offset),
        ("score_bias_amplitude", score_bias_amplitude)) if value is not None}
    if score_bias and (query_polynomial_coefficients is None or key_polynomial_factors is not None
                       or attention_coefficient_activation != "tanh"):
        # No other kernel may ignore them: it would silently train the unshifted polynomial.
        raise ValueError("Per-degree score biases apply only to tanh query-coefficient polynomial attention")
    # The historical calls are unchanged without a transform.
    transform = {} if validate_transform(score_transform) == "identity" else {"score_transform": score_transform}
    if transform and (query_polynomial_coefficients is None or key_polynomial_factors is not None
                      or attention_coefficient_activation != "tanh" or score_bias or query_log_decay is not None
                      or any(value is not None for value in (query_gate_input, key_gate_input, gate_log_rate))):
        # No other kernel may ignore it: it would silently train the untransformed polynomial.
        raise ValueError("A score transform applies only to tanh query-coefficient polynomial attention without a "
                         "forget gate, key factors, or score biases")
    # The historical scan keeps every mechanism's established call; strips add the selection.
    strips = {} if blocked_implementation == "scan" else {"blocked_implementation": blocked_implementation}
    denominators = {name: array for name, array in denominators.items() if array is not None}
    if not isinstance(attention_normalization, bool):
        raise ValueError("attention_normalization must be a boolean")
    if spline_knots is not None or spline_coefficients is not None or query_spline_coefficients is not None:
        from scaffolds.transformer.attention.spline import spline_attention

        if (attention_normalization or quadratic_mode != "fixed" or coefficients is not None
                or polynomial_coefficients is not None or polynomial_square_factors is not None
                or query_polynomial_coefficients is not None or key_polynomial_factors is not None
                or query_log_decay is not None or query_decay_mode != "cumulative"
                or any(value is not None for value in (query_gate_input, key_gate_input, gate_log_rate))
                or denominators):
            raise ValueError("B-spline coefficients define an unnormalized kernel alone, without decay")
        return spline_attention(q, k, v, attention_degree=attention_degree, spline_knots=spline_knots,
                                spline_coefficients=spline_coefficients,
                                query_spline_coefficients=query_spline_coefficients, backend=backend,
                                return_diagnostics=return_diagnostics, blocked_implementation=blocked_implementation)
    if key_polynomial_factors is not None and query_polynomial_coefficients is None:
        raise ValueError("Key factors require query coefficients")
    if query_decay_mode not in ("cumulative", "query_distance"):
        raise ValueError(f"Unknown query decay mode: {query_decay_mode}")
    endpoint = any(value is not None for value in (query_gate_input, key_gate_input, gate_log_rate))
    if endpoint and (any(value is None for value in (query_gate_input, key_gate_input, gate_log_rate))
                     or query_log_decay is not None or query_decay_mode != "query_distance"):
        raise ValueError("Endpoint decay requires all three operands and query-distance mode without log decay")
    if query_decay_mode != "cumulative" and query_log_decay is None and not endpoint:
        raise ValueError("A query-distance decay mode requires query_log_decay")
    if (query_log_decay is not None or endpoint) and query_polynomial_coefficients is None:
        # Softmax and the unnormalized exponential add the decay to every score: L_t - L_i (the
        # Forgetting Transformer), or (t - i) log alpha_t in the query-distance mode, or with endpoint
        # operands (t - i) log alpha_ti, log alpha_ti = -exp(gate_log_rate) softplus(u_t + v_i).
        if (attention_degree is not None or quadratic_mode != "fixed" or coefficients is not None
                or polynomial_coefficients is not None or polynomial_square_factors is not None or denominators
                or key_polynomial_factors is not None):
            raise ValueError("The forget gate requires query coefficients, softmax, or the unnormalized exponential")
        if backend != "blocked":
            raise ValueError("Softmax and exponential attention with a forget gate run on the blocked backend")
        from scaffolds.transformer.attention.backends.reference import blocked_backend_attention

        decay = (dict(query_gate_input=query_gate_input, key_gate_input=key_gate_input, gate_log_rate=gate_log_rate)
                 if endpoint else dict(query_log_decay=query_log_decay))
        return blocked_backend_attention(q, k, v, implementation=blocked_implementation,
                                         attention_normalization=attention_normalization,
                                         query_decay_mode=query_decay_mode, return_diagnostics=return_diagnostics,
                                         **decay)
    if (query_log_decay is not None or endpoint) and key_polynomial_factors is not None:
        raise ValueError("The forget gate requires query coefficients without key factors")
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
                                                  backend=backend, return_diagnostics=return_diagnostics, **strips)
        if attention_coefficient_activation == "identity":
            if attention_degree != 1:
                raise ValueError("Identity query coefficients require degree-one attention")
            if query_log_decay is not None or endpoint:
                raise ValueError("The forget gate requires nonlinear query coefficients")
            return query_linear_attention(q, k, v, query_polynomial_coefficients, backend=backend,
                                          return_diagnostics=return_diagnostics, **strips)
        if attention_coefficient_activation not in ("tanh", "sigmoid", "pure_tanh", "affine_sigmoid"):
            raise ValueError("Unknown attention coefficient activation")
        return query_polynomial_attention(q, k, v, query_polynomial_coefficients, backend=backend,
                                          return_diagnostics=return_diagnostics, log_decay=query_log_decay,
                                          decay_mode=query_decay_mode, query_gate_input=query_gate_input,
                                          key_gate_input=key_gate_input, gate_log_rate=gate_log_rate,
                                          **score_bias, **strips, **transform)
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
            return_diagnostics=return_diagnostics, polynomial_coefficients=polynomial_coefficients, **strips,
        )
    if polynomial_coefficients is not None or polynomial_square_factors is not None or denominators:
        from scaffolds.transformer.attention.learned_polynomial import normalized_learned_polynomial_attention

        if quadratic_mode != "fixed" or coefficients is not None:
            raise ValueError("Learned polynomial coefficients exclude the learned quadratic modes")
        return normalized_learned_polynomial_attention(
            q, k, v, attention_degree=attention_degree, polynomial_coefficients=polynomial_coefficients,
            polynomial_square_factors=polynomial_square_factors,
            backend=backend, return_diagnostics=return_diagnostics, **strips, **denominators,
        )
    if quadratic_mode != "fixed":
        from scaffolds.transformer.attention.learned_quadratic import learned_quadratic_attention

        if attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        return learned_quadratic_attention(
            q, k, v, coefficients=coefficients, quadratic_mode=quadratic_mode,
            backend=backend, return_diagnostics=return_diagnostics, **strips,
        )
    if coefficients is not None:
        raise ValueError("Fixed attention does not accept learned coefficients")
    if attention_degree is not None:
        from scaffolds.transformer.attention.taylor import taylor_attention

        return taylor_attention(
            q, k, v, attention_degree=attention_degree, backend=backend,
            return_diagnostics=return_diagnostics, **strips,
        )
    from scaffolds.transformer.attention.softmax import softmax_attention

    return softmax_attention(
        q, k, v, backend=backend, return_diagnostics=return_diagnostics,
        dropout=dropout, deterministic=deterministic, rngs=rngs,
        weights_dtype=weights_dtype, **strips,
    )
