"""Full MHA with independent query/key and value head widths."""

import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from experiments.key_dimension.models.components.positional import apply_rope


AttentionBackend = Literal["manual", "auto", "xla", "cudnn"]


def padded_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    backend: AttentionBackend = "xla",
) -> jax.Array:
    """Use public SDPA without changing the mathematical Q/K or value width.

    Inputs have shape [batch, time, heads, channels] and already contain RoPE.
    Zero padding makes the channel dimensions equal, as JAX's public API
    requires. The explicit scale always uses the *true* query/key dimension.
    """
    qk_dim, value_dim = q.shape[-1], v.shape[-1]
    if k.shape[-1] != qk_dim:
        raise ValueError("Query and key channel dimensions must match")
    if backend not in ("auto", "xla", "cudnn"):
        raise ValueError(f"Unknown fused attention backend: {backend}")
    # cuDNN also requires channel alignment; production d_V=64 already meets it.
    width = ((max(qk_dim, value_dim) + 7) // 8) * 8

    def pad(array):
        return jnp.pad(array, ((0, 0), (0, 0), (0, 0), (0, width - array.shape[-1])))

    y = jax.nn.dot_product_attention(
        pad(q), pad(k), pad(v),
        scale=1.0 / math.sqrt(qk_dim),
        is_causal=True,
        implementation=None if backend == "auto" else backend,
    )
    return y[..., :value_dim]


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        value_head_dim: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        backend: AttentionBackend = "manual",
        *,
        rngs: nnx.Rngs,
    ):
        if min(hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if num_heads != num_kv_heads:
            raise ValueError("This experiment requires full MHA (num_heads == num_kv_heads)")
        if qk_head_dim % 2:
            raise ValueError("qk_head_dim must be even for RoPE")
        if backend not in ("manual", "auto", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {backend}")
        if backend != "manual" and dropout != 0.0:
            raise ValueError("Fused attention currently requires dropout=0")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.rope_theta = rope_theta
        self.backend = backend
        # The training dtype walker configures BF16 compute without casting params.
        self.dtype = jnp.float32
        # Preserve OFM's constructor order and NNX default initializers exactly.
        self.q_proj = nnx.Linear(hidden_size, num_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, num_kv_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, num_kv_heads * value_head_dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout)
        self.o_proj = nnx.Linear(num_heads * value_head_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        batch, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.backend != "manual":
            q, k, v = (operand.astype(self.dtype) for operand in (q, k, v))
        q = q.reshape(batch, seq_len, self.num_heads, self.qk_head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.qk_head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.value_head_dim)

        # RoPE's frequency grid depends on the true dimension, before padding.
        positions = jnp.arange(seq_len)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        if self.backend != "manual":
            q, k = q.astype(self.dtype), k.astype(self.dtype)
            y = padded_dot_product_attention(q, k, v, backend=self.backend)
        else:
            scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.qk_head_dim)
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
            weights = nnx.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
            weights = self.dropout(weights, deterministic=deterministic, rngs=rngs)
            y = jnp.einsum("bhts,bshd->bthd", weights, v)
        return self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
