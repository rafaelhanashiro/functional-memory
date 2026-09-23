"""Unnormalized causal k(QK^T / sqrt(d_k)) V: exp or its Taylor truncation."""


def unnormalized_attention(q, k, v, *, attention_degree=None, backend="blocked", return_diagnostics=False):
    """``attention_degree=None`` selects unshifted exp; an integer m selects p_m."""
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if backend == "blocked":
        return blocked_dot_product_attention(
            q, k, v, attention_degree=attention_degree, attention_normalization=False,
            return_diagnostics=return_diagnostics,
        )
    if backend != "fused":
        raise ValueError("Unnormalized attention requires backend='blocked' or 'fused'")
    from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

    output = fused_unnormalized_attention(q, k, v, attention_degree=attention_degree)
    if not return_diagnostics:
        return output
    _, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=attention_degree, attention_normalization=False,
        return_diagnostics=True,
    )
    return output, diagnostics
