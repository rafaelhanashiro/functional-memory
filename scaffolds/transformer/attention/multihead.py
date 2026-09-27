"""Shared Q/K/V projections, positional encoding, and output projection."""

from typing import Literal
import math

import jax
import jax.numpy as jnp
from flax import nnx

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.attention.taylor import _validate_degree
from scaffolds.transformer.attention.positional import apply_rope

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
    return MultiHeadAttention(
        config.hidden_size, config.num_heads, config.num_kv_heads,
        config.qk_head_dim, config.value_head_dim, rope_theta=config.rope_theta,
        dropout=config.dropout, backend=config.attention_backend,
        attention_degree=config.attention_degree, quadratic_mode=config.quadratic_mode,
        attention_normalization=config.attention_normalization,
        attention_fixed_coefficients=config.attention_fixed_coefficients,
        qk_norm=config.qk_norm, qk_norm_eps=config.qk_norm_eps,
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
        rngs: nnx.Rngs,
    ):
        _validate_degree(attention_degree)
        if not isinstance(attention_normalization, bool):
            raise ValueError("attention_normalization must be a boolean")
        if not isinstance(qk_norm, bool):
            raise ValueError("qk_norm must be a boolean")
        if qk_norm and attention_degree == 0:
            raise ValueError("QK-norm requires a kernel that reads scores; degree 0 ignores Q/K")
        if not attention_normalization and (quadratic_mode != "fixed" or backend not in ("blocked", "fused")):
            raise ValueError("Unnormalized attention requires a fixed exp/Taylor kernel and a blocked/fused backend")
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
        if min(hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if num_heads != num_kv_heads:
            raise ValueError("This experiment requires full MHA (num_heads == num_kv_heads)")
        if qk_head_dim % 2:
            raise ValueError("qk_head_dim must be even for RoPE")
        if backend not in ("blocked", "fused", "manual", "auto", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {backend}")
        if backend not in ("blocked", "fused") and attention_degree is not None:
            raise ValueError("Taylor attention requires backend='blocked' or 'fused'")
        if backend != "manual" and dropout != 0.0:
            raise ValueError("Blocked/fused attention requires dropout=0")
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
        self.attention_normalization = attention_normalization
        self.quadratic_mode = quadratic_mode
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
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
            if any(name in parameters for name in ("coefficients", "polynomial_coefficients", "polynomial_square_factors")):
                raise ValueError("Fixed polynomial coefficients cannot be combined with supplied attention coefficients")
            parameters["polynomial_coefficients"] = jnp.asarray(self.attention_fixed_coefficients, dtype=jnp.float32)
        result = attention_core(
            q, k, v, backend=self.backend, attention_degree=self.attention_degree,
            attention_normalization=self.attention_normalization,
            quadratic_mode=self.quadratic_mode, **parameters,
            return_diagnostics=return_diagnostics, dropout=self.dropout,
            deterministic=deterministic, rngs=rngs, weights_dtype=x.dtype,
        )
        y, diagnostics = result if return_diagnostics else (result, None)
        output = self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
        return (output, diagnostics) if return_diagnostics else output
