import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.scale = nnx.Param(jnp.ones((hidden_size,)), weight_decay=False)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * self.scale
