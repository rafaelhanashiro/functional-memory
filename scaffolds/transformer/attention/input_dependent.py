"""Unnormalized degree-one attention with query-dependent affine coefficients.

The incoming attention activation (after the block's pre-attention RMSNorm)
produces c0,c1 per query and head. The score operands, including QK normalization
and RoPE, retain the ordinary dot-product path. Coefficient projection and basis
combination use FP32; the final attention output has V's dtype.
"""

import jax
import jax.numpy as jnp


def parameter_specs(*, hidden_size: int):
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (
        AttentionParameterSpec(
            name="query_polynomial_bias", argument="query_polynomial_bias",
            initial=(1.0, 1.0), sharing="per_head", weight_decay=False,
        ),
        AttentionParameterSpec(
            name="query_polynomial_kernel", argument="query_polynomial_kernel",
            initial=(0.0,) * (hidden_size * 2), initial_shape=(hidden_size, 2),
            sharing="per_head", weight_decay=False,
        ),
    )


def project_coefficients(x, kernel, bias):
    """FP32 affine projection [B,T,D] x [H,D,2] + [H,2] -> [B,T,H,2]."""
    if (x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1:] != (x.shape[-1], 2)
            or bias.shape != (kernel.shape[0], 2)):
        raise ValueError("Query coefficient projection requires x [B,T,D], kernel [H,D,2], bias [H,2]")
    return jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    ) + bias.astype(jnp.float32)


def coefficient_statistics(coefficients):
    """Actual coefficients on the diagnostic sample, per head and order."""
    values = coefficients.astype(jnp.float32)
    return {
        "query_coefficient_mean": jnp.mean(values, axis=(0, 1)),
        "query_coefficient_std": jnp.std(values, axis=(0, 1)),
        "query_coefficient_min": jnp.min(values, axis=(0, 1)),
        "query_coefficient_max": jnp.max(values, axis=(0, 1)),
    }


def query_linear_attention(q, k, v, coefficients, *, backend="blocked", return_diagnostics=False):
    """sum_(i<=t) (c0_t + c1_t * <q_t,k_i>/sqrt(d_K)) v_i.

    The fused path reuses the linear learned-polynomial kernel and adds the
    constant basis via a FP32 prefix sum. Its forward tiles also check the actual
    c0+c1*s kernel, so nonfinite weights cannot disappear behind zero values.
    Both bases must be representable in FP32 before their combination; extreme
    coefficients/cancellation can therefore fail earlier than direct evaluation.
    Its VJP propagates through both bases,
    both coefficients, and the coefficient projection. The blocked path directly
    evaluates the query-dependent polynomial and supplies kernel diagnostics.
    """
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if coefficients.shape != (*q.shape[:3], 2):
        raise ValueError("Query coefficients must have shape [batch,time,heads,2]")
    if backend == "blocked":
        result = blocked_dot_product_attention(
            q, k, v, attention_degree=1, attention_normalization=False,
            query_polynomial_coefficients=coefficients, return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            output, diagnostics = result
        else:
            return result
    elif backend == "fused":
        from scaffolds.transformer.attention.backends.fused_learned_polynomial import (
            fused_learned_polynomial_attention,
        )

        linear = fused_learned_polynomial_attention(
            q, k, v, jnp.broadcast_to(jnp.asarray([0.0, 1.0], jnp.float32), (q.shape[2], 2)),
            keep_output_float32=True, query_kernel_guard=coefficients,
        )
        prefix = jnp.cumsum(v.astype(jnp.float32), axis=1)
        c = coefficients.astype(jnp.float32)
        output = (c[..., 0, None] * prefix + c[..., 1, None] * linear).astype(v.dtype)
        output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        if not return_diagnostics:
            return output
        _, diagnostics = blocked_dot_product_attention(
            q, k, v, attention_degree=1, attention_normalization=False,
            query_polynomial_coefficients=coefficients, return_diagnostics=True,
        )
    else:
        raise ValueError("Query-conditioned attention requires a blocked or fused backend")
    diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
    return output, diagnostics
