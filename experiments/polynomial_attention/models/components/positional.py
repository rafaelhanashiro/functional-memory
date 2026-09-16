import jax
import jax.numpy as jnp


def rope_frequencies(head_dim: int, positions: jax.Array, theta: float = 10000.0) -> tuple[jax.Array, jax.Array]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    dim = jnp.arange(0, head_dim // 2, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (2.0 * dim / head_dim))
    freqs = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x: jax.Array, positions: jax.Array, theta: float = 10000.0) -> jax.Array:
    """Apply split-half RoPE to `[batch, time, heads, head_dim]` tensors."""

    head_dim = x.shape[-1]
    cos, sin = rope_frequencies(head_dim, positions, theta)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
