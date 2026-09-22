"""Shared Q/K/V projections, positional encoding, and output projection."""

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from experiments.shared_transformer.attention.core import attention_core
from experiments.shared_transformer.attention.taylor import _validate_degree
from experiments.shared_transformer.attention.positional import apply_rope

AttentionBackend = Literal["blocked", "fused", "manual", "auto", "xla", "cudnn"]


def attention_from_config(config, *, rngs: nnx.Rngs):
    """Construct the configured module without exposing mechanism fields to blocks."""
    return MultiHeadAttention(
        config.hidden_size, config.num_heads, config.num_kv_heads,
        config.qk_head_dim, config.value_head_dim, rope_theta=config.rope_theta,
        dropout=config.dropout, backend=config.attention_backend,
        attention_degree=config.attention_degree, quadratic_mode=config.quadratic_mode,
        rngs=rngs,
    )


class MultiHeadAttention(nnx.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        qk_head_dim: int, value_head_dim: int, rope_theta: float = 10000.0,
        dropout: float = 0.0, backend: AttentionBackend = "blocked",
        attention_degree: int | None = None, quadratic_mode: str = "fixed", *, rngs: nnx.Rngs,
    ):
        _validate_degree(attention_degree)
        if quadratic_mode not in ("fixed", "positive", "unrestricted"):
            raise ValueError(f"Unknown quadratic mode: {quadratic_mode}")
        if quadratic_mode != "fixed" and attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
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
        self.quadratic_mode = quadratic_mode
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
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.backend != "manual":
            q, k, v = (operand.astype(self.dtype) for operand in (q, k, v))
        q = q.reshape(batch, seq_len, self.num_heads, self.qk_head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.qk_head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.value_head_dim)
        positions = jnp.arange(seq_len)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        if self.backend != "manual":
            q, k = q.astype(self.dtype), k.astype(self.dtype)
        parameters = {} if attention_parameters is None else attention_parameters
        if coefficients is not None:
            if "coefficients" in parameters:
                raise ValueError("Coefficients were supplied twice")
            parameters = {**parameters, "coefficients": coefficients}
        result = attention_core(
            q, k, v, backend=self.backend, attention_degree=self.attention_degree,
            quadratic_mode=self.quadratic_mode, **parameters,
            return_diagnostics=return_diagnostics, dropout=self.dropout,
            deterministic=deterministic, rngs=rngs, weights_dtype=x.dtype,
        )
        y, diagnostics = result if return_diagnostics else (result, None)
        output = self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
        return (output, diagnostics) if return_diagnostics else output
