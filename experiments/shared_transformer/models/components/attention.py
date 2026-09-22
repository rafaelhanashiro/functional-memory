"""Compatibility imports for the shared attention package."""

from experiments.shared_transformer.attention.backends.reference import (
    QUERY_BLOCK_SIZE, blocked_dot_product_attention,
)
from experiments.shared_transformer.attention.learned_quadratic import _quadratic_polynomial
from experiments.shared_transformer.attention.multihead import AttentionBackend, MultiHeadAttention
from experiments.shared_transformer.attention.softmax import padded_dot_product_attention
from experiments.shared_transformer.attention.taylor import _taylor_polynomial, _validate_degree

__all__ = [
    "AttentionBackend", "MultiHeadAttention", "QUERY_BLOCK_SIZE",
    "blocked_dot_product_attention", "padded_dot_product_attention",
]
