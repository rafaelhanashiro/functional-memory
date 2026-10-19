"""Unnormalized polynomials with query-dependent coefficients.

The incoming attention activation (after the block's pre-attention RMSNorm)
produces coefficients per query and head. The score operands, including QK normalization
and RoPE, retain the ordinary dot-product path. Coefficient projection and basis
combination use FP32; the final attention output has V's dtype.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp


def parameter_specs(*, hidden_size: int, degree: int = 1, activation: str = "identity",
                    amplitude_initial: float = 1.0, projection_initial_std: float = 0.0,
                    tanh_bias: bool = False, conditioning: str = "query"):
    """b starts at 1/r!; W at zero or N(0, projection_initial_std^2); tanh's a at amplitude_initial
    and its optional inner bias beta at zero. "query_key" adds the key factor's W_k (same draw
    rule, independent draws) and a_k (same start as a)."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    count = degree + 1
    specs = (
        AttentionParameterSpec(
            name="query_polynomial_bias", argument="query_polynomial_bias",
            initial=tuple(1.0 / math.factorial(r) for r in range(count)), sharing="per_head", weight_decay=False,
        ),
        AttentionParameterSpec(
            name="query_polynomial_kernel", argument="query_polynomial_kernel",
            initial=(0.0,) * (hidden_size * count), initial_shape=(hidden_size, count),
            sharing="per_head", weight_decay=False, initial_std=float(projection_initial_std),
        ),
    )
    if activation == "tanh":
        specs += (AttentionParameterSpec(
            name="query_polynomial_amplitude", argument="query_polynomial_amplitude",
            initial=(float(amplitude_initial),) * count, sharing="per_head", weight_decay=False,
        ),)
        if tanh_bias:
            specs += (AttentionParameterSpec(
                name="query_polynomial_tanh_bias", argument="query_polynomial_tanh_bias",
                initial=(0.0,) * count, sharing="per_head", weight_decay=False,
            ),)
    elif activation != "identity":
        raise ValueError(f"Unknown coefficient activation: {activation}")
    elif tanh_bias:
        raise ValueError("A bias inside tanh requires tanh coefficients")
    if conditioning == "query_key":
        if activation != "tanh":
            raise ValueError("Query-key coefficients require tanh coefficients")
        specs += (
            AttentionParameterSpec(
                name="key_polynomial_kernel", argument="key_polynomial_kernel",
                initial=(0.0,) * (hidden_size * count), initial_shape=(hidden_size, count),
                sharing="per_head", weight_decay=False, initial_std=float(projection_initial_std),
            ),
            AttentionParameterSpec(
                name="key_polynomial_amplitude", argument="key_polynomial_amplitude",
                initial=(float(amplitude_initial),) * count, sharing="per_head", weight_decay=False,
            ),
        )
    elif conditioning != "query":
        raise ValueError(f"Unknown input-dependent coefficient conditioning: {conditioning}")
    return specs


def project_coefficients(x, kernel, bias, *, amplitude=None, activation="identity", tanh_bias=None):
    """FP32 c=b+xW or c=b+a*tanh(xW+beta); the inner bias beta is optional and otherwise absent."""
    if (x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1] != x.shape[-1]
            or bias.shape != (kernel.shape[0], kernel.shape[-1])):
        raise ValueError("Query coefficient projection requires x [B,T,D], kernel [H,D,R], bias [H,R]")
    projected = jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    if activation == "tanh":
        if amplitude is None or amplitude.shape != bias.shape:
            raise ValueError("Tanh coefficients require a model-owned amplitude with the bias shape")
        if tanh_bias is not None:
            if tanh_bias.shape != bias.shape:
                raise ValueError("The bias inside tanh must have the outer bias shape [H,R]")
            projected = projected + tanh_bias.astype(jnp.float32)
        projected = amplitude.astype(jnp.float32) * jnp.tanh(projected)
    elif activation != "identity" or amplitude is not None or tanh_bias is not None:
        raise ValueError("Identity coefficients cannot receive an amplitude or inner bias; unknown activation otherwise")
    return projected + bias.astype(jnp.float32)


def project_key_factors(x, kernel, amplitude):
    """FP32 per-key factors f = 1 + a*tanh(xW) with shape [B,T,H,R]; exactly one while a = 0."""
    if (x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1] != x.shape[-1]
            or amplitude.shape != (kernel.shape[0], kernel.shape[-1])):
        raise ValueError("Key factor projection requires x [B,T,D], kernel [H,D,R], amplitude [H,R]")
    projected = jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    return 1.0 + amplitude.astype(jnp.float32) * jnp.tanh(projected)


def query_key_polynomial_attention(q, k, v, coefficients, key_factors, *, backend="blocked",
                                   return_diagnostics=False):
    """sum_(i<=t) sum_r c_r(x_t) f_r(x_i) s_ti^r v_i: per-query coefficients times per-key factors.

    The blocked reference evaluates the pairwise coefficient directly. The fused and Mosaic
    query-coefficient kernels run once per order r with the key factor folded into the
    values, v_i^(r) = f_r(x_i) v_i, and the coefficient vector c_r(x_t) e_r; the m+1 partial
    outputs are summed in FP32, so a nonfinite row in any pass poisons that row.
    """
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3] or coefficients.shape[-1] < 2:
        raise ValueError("Query coefficients require [batch,time,heads,degree+1], degree >= 1")
    if key_factors.shape != coefficients.shape:
        raise ValueError("Key factors must match the query coefficient shape [batch,time,heads,degree+1]")
    degree = coefficients.shape[-1] - 1
    reference = partial(blocked_dot_product_attention, q, k, v, attention_degree=degree,
                        attention_normalization=False, query_polynomial_coefficients=coefficients,
                        key_polynomial_factors=key_factors)
    if backend == "blocked":
        result = reference(return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return result
        output, diagnostics = result
    elif backend in ("fused", "mosaic"):
        if backend == "fused":
            from scaffolds.transformer.attention.backends.fused_query_polynomial import (
                fused_query_polynomial_attention as attend,
            )
        else:
            from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention as attend

        coefficients32, factors32 = coefficients.astype(jnp.float32), key_factors.astype(jnp.float32)
        total = None
        for order in range(degree + 1):
            selected = jnp.zeros_like(coefficients32).at[..., order].set(coefficients32[..., order])
            values = (v.astype(jnp.float32) * factors32[..., order, None]).astype(v.dtype)
            partial_output = attend(q, k, values, selected).astype(jnp.float32)
            total = partial_output if total is None else total + partial_output
        output = total.astype(v.dtype)
        output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        if not return_diagnostics:
            return output
        _, diagnostics = reference(return_diagnostics=True)
    else:
        raise ValueError("Query-key polynomial attention requires a blocked, fused or mosaic backend")
    diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
    diagnostics.update(jax.lax.stop_gradient(key_factor_statistics(key_factors)))
    return output, diagnostics


def key_factor_statistics(key_factors):
    """Actual per-key factors on the diagnostic sample, per head and order."""
    values = key_factors.astype(jnp.float32)
    return {
        "key_factor_mean": jnp.mean(values, axis=(0, 1)),
        "key_factor_std": jnp.std(values, axis=(0, 1)),
        "key_factor_min": jnp.min(values, axis=(0, 1)),
        "key_factor_max": jnp.max(values, axis=(0, 1)),
    }


def query_polynomial_attention(q, k, v, coefficients, *, backend="blocked", return_diagnostics=False):
    """Apply a per-query polynomial directly in one fused attention operation."""
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3]
            or coefficients.shape[-1] < 2):
        raise ValueError("Query coefficients require [batch,time,heads,degree+1], degree >= 1")
    degree = coefficients.shape[-1] - 1
    if backend == "blocked":
        result = blocked_dot_product_attention(
            q, k, v, attention_degree=degree, attention_normalization=False,
            query_polynomial_coefficients=coefficients, return_diagnostics=return_diagnostics,
        )
        if not return_diagnostics:
            return result
        output, diagnostics = result
    elif backend in ("fused", "mosaic"):
        if backend == "fused":
            from scaffolds.transformer.attention.backends.fused_query_polynomial import (
                fused_query_polynomial_attention as attend,
            )
        else:
            from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention as attend

        output = attend(q, k, v, coefficients)
        if not return_diagnostics:
            return output
        _, diagnostics = blocked_dot_product_attention(
            q, k, v, attention_degree=degree, attention_normalization=False,
            query_polynomial_coefficients=coefficients, return_diagnostics=True,
        )
    else:
        raise ValueError("Query-conditioned polynomial attention requires a blocked, fused or mosaic backend")
    diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
    return output, diagnostics


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
