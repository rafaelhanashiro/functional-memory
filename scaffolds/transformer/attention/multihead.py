"""Shared Q/K/V projections, positional encoding, and output projection."""

from typing import Literal
import math

import jax
import jax.numpy as jnp
from flax import nnx

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.output_scaling import OUTPUT_SCALINGS, scale_attention_output
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.attention.rational_polynomial import DENOMINATOR_KINDS
from scaffolds.transformer.attention.taylor import _validate_degree
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.training import telemetry

AttentionBackend = Literal["blocked", "fused", "manual", "auto", "xla", "cudnn"]


def attention_from_config(config, *, rngs: nnx.Rngs):
    """Construct the configured module without exposing mechanism fields to blocks."""
    if config.attention_mechanism == "gla":
        from scaffolds.transformer.attention.gla import GatedLinearAttention

        return GatedLinearAttention(
            config.hidden_size, config.num_heads,
            config.num_heads * config.qk_head_dim, config.num_heads * config.value_head_dim,
            gate_rank=config.gla_gate_rank, gate_tau=config.gla_gate_tau,
            chunk_size=config.gla_chunk_size, inner_chunk_size=config.gla_inner_chunk_size,
            rope_theta=config.rope_theta, rms_norm_eps=config.rms_norm_eps,
            dropout=config.dropout, use_rope=config.gla_use_rope,
            backend=config.attention_backend, rngs=rngs,
        )
    if config.attention_mechanism == "gated_deltanet":
        from scaffolds.transformer.attention.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet(
            config.hidden_size, config.num_heads, config.qk_head_dim, config.value_head_dim,
            conv_size=config.gdn_conv_size, norm_eps=config.rms_norm_eps,
            backend=config.attention_backend, erase_mode=config.gdn_erase_mode,
            write_activation=config.gdn_write_activation,
            polynomial_degree=config.gdn_polynomial_degree, rngs=rngs,
        )
    from scaffolds.transformer.attention.basis import projection_from_config

    score_kernel = None if config.attention_basis is None else projection_from_config(config.attention_basis)
    return MultiHeadAttention(
        config.hidden_size, config.num_heads, config.num_kv_heads,
        config.qk_head_dim, config.value_head_dim, rope_theta=config.rope_theta,
        dropout=config.dropout, backend=config.attention_backend,
        attention_degree=config.attention_degree, quadratic_mode=config.quadratic_mode,
        attention_normalization=config.attention_normalization,
        attention_fixed_coefficients=config.attention_fixed_coefficients,
        attention_coefficient_conditioning=config.attention_coefficient_conditioning,
        attention_coefficient_activation=config.attention_coefficient_activation,
        attention_fixed_square_factors=config.attention_fixed_square_factors,
        qk_norm=config.qk_norm, qk_norm_eps=config.qk_norm_eps,
        attention_output_scaling=config.attention_output_scaling, rms_norm_eps=config.rms_norm_eps,
        score_kernel=score_kernel,
        rngs=rngs,
    )


class MultiHeadAttention(nnx.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        qk_head_dim: int, value_head_dim: int, rope_theta: float = 10000.0,
        dropout: float = 0.0, backend: AttentionBackend = "blocked",
        attention_degree: int | None = None, quadratic_mode: str = "fixed",
        attention_normalization: bool = True, qk_norm: bool = False,
        qk_norm_eps: float = 1e-6, *, attention_fixed_coefficients: tuple[float, ...] | None = None,
        attention_fixed_square_factors: tuple[float, ...] | None = None,
        attention_output_scaling: str = "none", rms_norm_eps: float = 1e-6,
        attention_coefficient_conditioning: str = "static",
        attention_coefficient_activation: str = "identity",
        score_kernel=None,
        rngs: nnx.Rngs,
    ):
        _validate_degree(attention_degree)
        if score_kernel is not None and (
                not callable(score_kernel) or backend != "blocked" or attention_degree is not None
                or quadratic_mode != "fixed" or attention_fixed_coefficients is not None
                or attention_fixed_square_factors is not None or attention_coefficient_conditioning != "static"):
            raise ValueError("A scalar score kernel requires blocked attention without polynomial options")
        self.score_kernel = score_kernel
        if not isinstance(attention_normalization, bool):
            raise ValueError("attention_normalization must be a boolean")
        if not isinstance(qk_norm, bool):
            raise ValueError("qk_norm must be a boolean")
        if qk_norm and attention_degree == 0:
            raise ValueError("QK-norm requires a kernel that reads scores; degree 0 ignores Q/K")
        if not attention_normalization and (quadratic_mode != "fixed" or backend not in ("blocked", "fused", "mosaic")):
            raise ValueError("Unnormalized attention requires a fixed exp/Taylor kernel and a blocked, fused or mosaic backend")
        if attention_output_scaling not in OUTPUT_SCALINGS:
            raise ValueError(f"Unknown attention output scaling: {attention_output_scaling}")
        if attention_output_scaling == "inverse_position" and attention_normalization:
            raise ValueError("Inverse-position output scaling applies only to unnormalized attention")
        if quadratic_mode not in ("fixed", "positive", "unrestricted"):
            raise ValueError(f"Unknown quadratic mode: {quadratic_mode}")
        if quadratic_mode != "fixed" and attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        if attention_fixed_coefficients is not None:
            if quadratic_mode != "fixed" or attention_degree is None or attention_degree < 1:
                raise ValueError("Fixed polynomial coefficients require degree >= 1 without a quadratic mode")
            if (not isinstance(attention_fixed_coefficients, (tuple, list))
                    or len(attention_fixed_coefficients) != attention_degree + 1):
                raise ValueError("Fixed polynomial coefficients must contain attention_degree + 1 coefficients")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in attention_fixed_coefficients):
                raise ValueError("Fixed polynomial coefficients must be finite numeric values")
            attention_fixed_coefficients = tuple(float(value) for value in attention_fixed_coefficients)
        if attention_fixed_square_factors is not None:
            if attention_fixed_coefficients is not None:
                raise ValueError("Specify either fixed polynomial coefficients or fixed square factors")
            if (quadratic_mode != "fixed" or not attention_normalization or attention_degree is None
                    or attention_degree < 2 or attention_degree % 2):
                raise ValueError("Fixed square factors require normalized, even-degree attention without a quadratic mode")
            if (not isinstance(attention_fixed_square_factors, (tuple, list))
                    or len(attention_fixed_square_factors) != attention_degree + 1):
                raise ValueError("Fixed square factors must contain attention_degree + 1 factors")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in attention_fixed_square_factors):
                raise ValueError("Fixed square factors must be finite numeric values")
            attention_fixed_square_factors = tuple(float(value) for value in attention_fixed_square_factors)
        if min(hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if num_heads != num_kv_heads:
            raise ValueError("This experiment requires full MHA (num_heads == num_kv_heads)")
        if qk_head_dim % 2:
            raise ValueError("qk_head_dim must be even for RoPE")
        if backend not in ("blocked", "fused", "mosaic", "manual", "auto", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {backend}")
        if backend not in ("blocked", "fused", "mosaic") and attention_degree is not None:
            raise ValueError("Taylor attention requires backend='blocked', 'fused' or 'mosaic'")
        if backend != "manual" and dropout != 0.0:
            raise ValueError("Blocked/fused attention requires dropout=0")
        if attention_coefficient_conditioning not in ("static", "query", "query_key"):
            raise ValueError("Unknown attention coefficient conditioning")
        if attention_coefficient_activation not in ("identity", "tanh"):
            raise ValueError("Unknown attention coefficient activation")
        if attention_coefficient_activation != "identity" and attention_coefficient_conditioning == "static":
            raise ValueError("Tanh coefficients require query conditioning")
        if attention_coefficient_conditioning == "query_key" and attention_coefficient_activation != "tanh":
            raise ValueError("Query-key coefficients require the tanh activation")
        if attention_coefficient_conditioning in ("query", "query_key") and (
                attention_normalization or attention_degree is None or attention_degree < 1 or quadratic_mode != "fixed"
                or attention_fixed_coefficients is not None or attention_fixed_square_factors is not None):
            raise ValueError("Query-conditioned coefficients require unnormalized polynomial attention")
        if (attention_coefficient_conditioning == "query" and attention_coefficient_activation == "identity"
                and attention_degree != 1):
            raise ValueError("Identity query coefficients require degree-one attention")
        self.attention_coefficient_conditioning = attention_coefficient_conditioning
        self.attention_coefficient_activation = attention_coefficient_activation
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.rope_theta = rope_theta
        self.backend = backend
        self.attention_degree = attention_degree
        # Python floats are static module data, never optimizer parameters.
        self.attention_fixed_coefficients = attention_fixed_coefficients
        self.attention_fixed_square_factors = attention_fixed_square_factors
        self.attention_normalization = attention_normalization
        self.quadratic_mode = quadratic_mode
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.attention_output_scaling = attention_output_scaling
        self.rms_norm_eps = rms_norm_eps
        self.dtype = jnp.float32
        # Preserve the reference constructor order/default initialization exactly.
        self.q_proj = nnx.Linear(hidden_size, num_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, num_kv_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, num_kv_heads * value_head_dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout)
        self.o_proj = nnx.Linear(num_heads * value_head_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None, return_diagnostics: bool = False,
        coefficients: jax.Array | None = None,
        attention_parameters: dict[str, jax.Array] | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        batch, seq_len, _ = x.shape
        parameters = {} if attention_parameters is None else dict(attention_parameters)
        log_scale = parameters.pop("qk_norm_log_scale", None)
        if self.qk_norm != (log_scale is not None):
            raise ValueError("QK-norm requires exactly its model-owned qk_norm_log_scale parameter")
        output_norm_scale = parameters.pop("output_norm_scale", None)
        if (self.attention_output_scaling == "head_rms_norm") != (output_norm_scale is not None):
            raise ValueError("Head RMSNorm output scaling requires exactly its model-owned output_norm_scale parameter")
        query_kernel = parameters.pop("query_polynomial_kernel", None)
        query_bias = parameters.pop("query_polynomial_bias", None)
        query_amplitude = parameters.pop("query_polynomial_amplitude", None)
        query_tanh_bias = parameters.pop("query_polynomial_tanh_bias", None)
        key_kernel = parameters.pop("key_polynomial_kernel", None)
        key_amplitude = parameters.pop("key_polynomial_amplitude", None)
        if self.attention_coefficient_conditioning in ("query", "query_key"):
            from scaffolds.transformer.attention.input_dependent import project_coefficients, project_key_factors

            if query_kernel is None or query_bias is None:
                raise ValueError("Query-conditioned attention requires its model-owned kernel and bias")
            parameters["query_polynomial_coefficients"] = project_coefficients(
                x, query_kernel, query_bias, amplitude=query_amplitude,
                activation=self.attention_coefficient_activation, tanh_bias=query_tanh_bias)
            if self.attention_coefficient_conditioning == "query_key":
                if key_kernel is None or key_amplitude is None:
                    raise ValueError("Query-key attention requires its model-owned key kernel and amplitude")
                parameters["key_polynomial_factors"] = project_key_factors(x, key_kernel, key_amplitude)
            elif key_kernel is not None or key_amplitude is not None:
                raise ValueError("Query-conditioned attention cannot receive key factor parameters")
        elif any(value is not None for value in (query_kernel, query_bias, query_amplitude, query_tanh_bias,
                                                 key_kernel, key_amplitude)):
            raise ValueError("Static coefficients cannot receive query coefficient parameters")
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.backend != "manual":
            q, k, v = (operand.astype(self.dtype) for operand in (q, k, v))
        q = q.reshape(batch, seq_len, self.num_heads, self.qk_head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.qk_head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.value_head_dim)
        if self.qk_norm:
            # Cosine scores: the mechanisms' q'.k'/sqrt(d_K) becomes exp(gamma) cos(q, k).
            q, k = cosine_score_operands(q, k, log_scale, eps=self.qk_norm_eps)
        positions = jnp.arange(seq_len)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        if self.backend != "manual":
            q, k = q.astype(self.dtype), k.astype(self.dtype)
        if coefficients is not None:
            if "coefficients" in parameters:
                raise ValueError("Coefficients were supplied twice")
            parameters = {**parameters, "coefficients": coefficients}
        if self.attention_fixed_coefficients is not None:
            if any(name in parameters for name in ("coefficients", "polynomial_coefficients", "polynomial_square_factors",
                                                *DENOMINATOR_KINDS)):
                raise ValueError("Fixed polynomial coefficients cannot be combined with supplied attention coefficients")
            parameters["polynomial_coefficients"] = jnp.asarray(self.attention_fixed_coefficients, dtype=jnp.float32)
        if self.attention_fixed_square_factors is not None:
            if any(name in parameters for name in ("coefficients", "polynomial_coefficients", "polynomial_square_factors",
                                                *DENOMINATOR_KINDS)):
                raise ValueError("Fixed square factors cannot be combined with supplied attention coefficients")
            parameters["polynomial_square_factors"] = jnp.asarray(self.attention_fixed_square_factors, dtype=jnp.float32)
        result = attention_core(
            q, k, v, backend=self.backend, attention_degree=self.attention_degree,
            attention_normalization=self.attention_normalization,
            attention_coefficient_activation=self.attention_coefficient_activation,
            quadratic_mode=self.quadratic_mode, **parameters,
            return_diagnostics=return_diagnostics, dropout=self.dropout,
            deterministic=deterministic, rngs=rngs, weights_dtype=x.dtype,
            score_kernel=self.score_kernel,
        )
        y, diagnostics = result if return_diagnostics else (result, None)
        telemetry.record_attention_output(
            y, head_rms_norm_eps=self.rms_norm_eps if self.attention_output_scaling == "head_rms_norm" else None)
        y = scale_attention_output(y, self.attention_output_scaling,
                                   output_norm_scale=output_norm_scale, eps=self.rms_norm_eps)
        output = self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
        return (output, diagnostics) if return_diagnostics else output
