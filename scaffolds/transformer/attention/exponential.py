"""Unshifted causal exp(QK^T / sqrt(d_k)) V without row normalization."""


def exponential_attention(q, k, v, *, backend="blocked", return_diagnostics=False):
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if backend == "blocked":
        return blocked_dot_product_attention(
            q, k, v, attention_normalization=False, return_diagnostics=return_diagnostics,
        )
    if backend != "fused":
        raise ValueError("Unnormalized attention requires backend='blocked' or 'fused'")
    from scaffolds.transformer.attention.backends.fused_exponential import fused_unnormalized_attention

    output = fused_unnormalized_attention(q, k, v)
    if not return_diagnostics:
        return output
    _, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_normalization=False, return_diagnostics=True,
    )
    return output, diagnostics
