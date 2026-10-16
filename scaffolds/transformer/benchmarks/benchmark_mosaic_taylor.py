"""Forward and forward+backward time of normalized Taylor attention, Mosaic GPU against
Pallas-Triton, at the production per-GPU shape (16 x 2048 x 16 heads, value width 64).

    python -m scaffolds.transformer.benchmarks.benchmark_mosaic_taylor   # in an H100 or B300 allocation
"""
import math
import time

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.mosaic.taylor import taylor_attention
from scaffolds.transformer.attention.backends.fused_taylor import fused_polynomial_attention
print(jax.devices()[0].device_kind, flush=True)
def bench(f, *a, n=10):
    jax.block_until_ready(f(*a)); t = time.perf_counter()
    for _ in range(n): o = f(*a)
    jax.block_until_ready(o); return (time.perf_counter() - t) / n * 1e3
def inputs(width):
    ks = jax.random.split(jax.random.key(0), 4)
    q, k = (jax.random.normal(kk, (16, 2048, 16, width), jnp.float32) for kk in ks[:2])
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * math.sqrt(width) * math.sqrt(width)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v, do = (jax.random.normal(kk, (16, 2048, 16, 64), jnp.bfloat16) for kk in ks[2:])
    return q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v, do
q, k, v, do = inputs(64)  # warm the GPU so the first measured cell is not slowed
for fn in (fused_polynomial_attention, taylor_attention):
    bench(jax.jit(lambda q, k, v: fn(q, k, v, attention_degree=2)), q, k, v)
for width in (8, 32, 64, 128):
    q, k, v, do = inputs(width)
    for degree in (1, 2, 4, 16):
        row = f"d_K={width:<3} m={degree:<2}"
        for name, fn in (("Triton", fused_polynomial_attention), ("Mosaic", taylor_attention)):
            f = lambda q, k, v: fn(q, k, v, attention_degree=degree)
            g = jax.jit(jax.grad(lambda q, k, v: jnp.sum(f(q, k, v).astype(jnp.float32) * do), argnums=(0, 1, 2)))
            try:
                row += f" | {name} fwd {bench(jax.jit(f), q, k, v):6.3f} fwd+bwd {bench(g, q, k, v):6.3f}"
            except Exception as e:
                row += f" | {name} {type(e).__name__}: {str(e)[:80]}"
        print(row, flush=True)
