"""Row-normalized causal attention with a learned polynomial kernel: p(QK^T / sqrt(d_k)) V / rowsum.

The coefficients c_0..c_m are the same model-owned ``polynomial_coefficients`` array
as the unnormalized mechanism's, shared ([m+1]) or per head ([heads, m+1]), and the
kernel may be signed. Weights divide by the signed row sum: negative nonzero sums
are valid, an exactly zero or nonfinite sum poisons its row, as for the unrestricted
quadratic and odd Taylor kernels. Only the blocked reference evaluates this
mechanism; there is no fused kernel.
"""


def normalized_learned_polynomial_attention(q, k, v, *, attention_degree, polynomial_coefficients,
                                            backend="blocked", return_diagnostics=False):
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
    from scaffolds.transformer.attention.unnormalized import _validate_coefficients

    _validate_coefficients(polynomial_coefficients, attention_degree, q.shape[2])
    if backend != "blocked":
        raise ValueError("Normalized learned polynomial coefficients have no fused kernel; use backend='blocked'")
    return blocked_dot_product_attention(
        q, k, v, attention_degree=attention_degree, attention_normalization=True,
        polynomial_coefficients=polynomial_coefficients, return_diagnostics=return_diagnostics,
    )
