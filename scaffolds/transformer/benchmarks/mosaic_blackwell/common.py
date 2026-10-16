"""Shared inputs, FP64 reference, and timing for the Mosaic Blackwell prototypes."""
import math, sys, time, traceback
import numpy as np, jax, jax.numpy as jnp
print(jax.devices()[0].device_kind, flush=True)
def inputs(B, T, H, D, seed=0):
    ks = jax.random.split(jax.random.key(seed), 3)
    q, k, v = (jax.random.normal(kk, (B, T, H, D), jnp.float32) for kk in ks)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * 8 * math.sqrt(D)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    return tuple(x.astype(jnp.bfloat16) for x in (q, k, v))
def reference(q, k, v, degree):
    q, k, v = (np.asarray(x.astype(jnp.float32), np.float64) for x in (q, k, v))
    T, D = q.shape[1], q.shape[-1]
    s = np.einsum("bthd,bshd->bhts", q, k) / math.sqrt(D)
    p = np.zeros_like(s); term = np.ones_like(s)
    for r in range(degree + 1):
        p += term; term = term * s / (r + 1)
    p = np.where(np.tril(np.ones((T, T), bool)), p, 0.0)
    return np.einsum("bhts,bshd->bthd", p, v) / p.sum(-1).transpose(0, 2, 1)[..., None], p.sum(-1).transpose(0, 2, 1)
def bench(f, *a, n=10):
    jax.block_until_ready(f(*a)); t = time.perf_counter()
    for _ in range(n): o = f(*a)
    jax.block_until_ready(o); return (time.perf_counter() - t) / n * 1e3
