"""Exact softmax attention with reference, SDPA, and legacy dropout paths."""

import math

import jax
import jax.numpy as jnp
from flax import nnx


def padded_dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, backend: str = "xla",
) -> jax.Array:
    """Original public SDPA path retained solely for baseline parity checks."""
    qk_dim, value_dim = q.shape[-1], v.shape[-1]
    if k.shape[-1] != qk_dim:
        raise ValueError("Query and key channel dimensions must match")
    if backend not in ("auto", "xla", "cudnn"):
        raise ValueError(f"Unknown fused attention backend: {backend}")
    width = ((max(qk_dim, value_dim) + 7) // 8) * 8

    def pad(array):
        return jnp.pad(array, ((0, 0), (0, 0), (0, 0), (0, width - array.shape[-1])))

    y = jax.nn.dot_product_attention(
        pad(q), pad(k), pad(v), scale=1.0 / math.sqrt(qk_dim),
        is_causal=True, implementation=None if backend == "auto" else backend,
    )
    return y[..., :value_dim]


def softmax_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, backend: str = "blocked",
    return_diagnostics: bool = False, dropout=None, deterministic: bool = True,
    rngs=None, weights_dtype=None, blocked_implementation: str = "scan",
):
    """Exact softmax with legacy dropout arithmetic and optional diagnostics.

    ``blocked_implementation`` ("scan" or "strips") selects the blocked backend's execution.
    """
    from scaffolds.transformer.attention.backends.reference import (
        blocked_backend_attention, blocked_dot_product_attention,
    )

    if backend == "blocked":
        return blocked_backend_attention(q, k, v, implementation=blocked_implementation,
                                         return_diagnostics=return_diagnostics)
    if backend == "manual":
        seq_len = q.shape[1]
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
        weights = nnx.softmax(scores.astype(jnp.float32), axis=-1).astype(
            v.dtype if weights_dtype is None else weights_dtype
        )
        if dropout is not None:
            weights = dropout(weights, deterministic=deterministic, rngs=rngs)
        output = jnp.einsum("bhts,bshd->bthd", weights, v)
    else:
        output = padded_dot_product_attention(q, k, v, backend="cudnn" if backend == "fused" else backend)
    if not return_diagnostics:
        return output
    # All mechanisms can report diagnostics to the common trainer without
    # replacing the output of the selected implementation.
    _, diagnostics = blocked_dot_product_attention(q, k, v, return_diagnostics=True)
    return output, diagnostics
