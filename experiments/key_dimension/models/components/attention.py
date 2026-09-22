"""Shared attention with the historical direct-constructor defaults."""

from experiments.shared_transformer.models.components import attention as _shared
from experiments.shared_transformer.models.components.attention import (
    AttentionBackend,
    QUERY_BLOCK_SIZE,
    blocked_dot_product_attention,
    padded_dot_product_attention,
)


class MultiHeadAttention(_shared.MultiHeadAttention):
    def __init__(
        self, hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim,
        rope_theta=10000.0, dropout=0.0, backend="manual", *, rngs, **kwargs,
    ):
        super().__init__(
            hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim,
            rope_theta=rope_theta, dropout=dropout, backend=backend, rngs=rngs, **kwargs,
        )


def __getattr__(name):
    return getattr(_shared, name)


__all__ = list(_shared.__all__)
