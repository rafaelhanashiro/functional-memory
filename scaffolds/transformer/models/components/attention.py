"""Compatibility imports for the shared attention package."""

from scaffolds.transformer.attention.backends.reference import (
    QUERY_BLOCK_SIZE, blocked_dot_product_attention,
)
from scaffolds.transformer.attention.learned_quadratic import _quadratic_polynomial
from scaffolds.transformer.attention.multihead import AttentionBackend, MultiHeadAttention
from scaffolds.transformer.attention.softmax import padded_dot_product_attention
from scaffolds.transformer.attention.taylor import _taylor_polynomial, _validate_degree

__all__ = [
    "AttentionBackend", "MultiHeadAttention", "QUERY_BLOCK_SIZE",
    "blocked_dot_product_attention", "padded_dot_product_attention",
]
