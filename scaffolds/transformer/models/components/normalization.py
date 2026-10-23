import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, fp32: bool = False):
        self.scale = nnx.Param(jnp.ones((hidden_size,)), weight_decay=False)
        self.eps = eps
        self.fp32 = fp32

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.fp32:
            x = x.astype(jnp.float32)
        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * self.scale
