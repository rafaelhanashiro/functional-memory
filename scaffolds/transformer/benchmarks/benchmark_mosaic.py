"""Forward and forward+backward time of the Mosaic GPU attention kernels at both weight
precisions against their Pallas-Triton counterparts and cuDNN softmax, at the production
per-GPU shape (16 x 2048 x 16 heads, value width 64): normalized Taylor attention over Q/K
widths and degrees, then query-conditioned polynomial attention at width 64 (gradients
include the coefficients). Times are milliseconds.

    python -m scaffolds.transformer.benchmarks.benchmark_mosaic   # in an H100 or B300 allocation
"""
from functools import partial
import math
import time

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention, taylor_attention
from scaffolds.transformer.attention.backends.fused_query_polynomial import fused_query_polynomial_attention
from scaffolds.transformer.attention.backends.fused_taylor import fused_polynomial_attention
from scaffolds.transformer.attention.backends.mosaic.common import BF16, BF16X3
from scaffolds.transformer.attention.softmax import padded_dot_product_attention
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
def softmax(q, k, v):  # the production softmax path
    return padded_dot_product_attention(q, k, v, backend="cudnn")


for width in (8, 16, 32, 64, 128):
    q, k, v, do = inputs(width)
    g = jax.jit(jax.grad(lambda q, k, v: jnp.sum(softmax(q, k, v).astype(jnp.float32) * do), argnums=(0, 1, 2)))
    print(f"d_K={width:<3} cuDNN softmax | fwd {bench(jax.jit(softmax), q, k, v):6.3f} fwd+bwd {bench(g, q, k, v):6.3f}",
          flush=True)
    for degree in (1, 2, 4, 16):
        row = f"d_K={width:<3} m={degree:<2}"
        for name, fn in (("Triton", fused_polynomial_attention),
                         ("Mosaic bf16x3", partial(taylor_attention, precision=BF16X3)),
                         ("Mosaic bf16", partial(taylor_attention, precision=BF16))):
            f = lambda q, k, v: fn(q, k, v, attention_degree=degree)
            g = jax.jit(jax.grad(lambda q, k, v: jnp.sum(f(q, k, v).astype(jnp.float32) * do), argnums=(0, 1, 2)))
            try:
                row += f" | {name} fwd {bench(jax.jit(f), q, k, v):6.3f} fwd+bwd {bench(g, q, k, v):6.3f}"
            except Exception as e:
                row += f" | {name} {type(e).__name__}: {str(e)[:80]}"
        print(row, flush=True)
q, k, v, do = inputs(64)
for degree in (1, 2, 3, 4, 8):
    c = jnp.tile(jnp.array([1 / math.factorial(r) for r in range(degree + 1)], jnp.float32), (16, 2048, 16, 1))
    row = f"query m={degree:<2}"
    for name, fn in (("Triton", fused_query_polynomial_attention),
                     ("Mosaic bf16x3", partial(query_polynomial_attention, precision=BF16X3)),
                     ("Mosaic bf16", partial(query_polynomial_attention, precision=BF16))):
        g = jax.jit(jax.grad(lambda q, k, v, c: jnp.sum(fn(q, k, v, c).astype(jnp.float32) * do), argnums=(0, 1, 2, 3)))
        row += f" | {name} fwd {bench(jax.jit(fn), q, k, v, c):6.3f} fwd+bwd {bench(g, q, k, v, c):6.3f}"
    print(row, flush=True)

from scaffolds.transformer.attention.unnormalized import unnormalized_attention

for degree in (3, 8):  # static per-head learned coefficients (the query kernel with equal coefficients)
    c = jnp.tile(jnp.array([1 / math.factorial(r) for r in range(degree + 1)], jnp.float32), (16, 1))
    row = f"static m={degree:<2}"
    for name, backend in (("Triton", "fused"), ("Mosaic bf16", "mosaic")):
        fn = partial(unnormalized_attention, attention_degree=degree, backend=backend)
        f = jax.jit(lambda q, k, v, c: fn(q, k, v, polynomial_coefficients=c))
        g = jax.jit(jax.grad(lambda q, k, v, c: jnp.sum(fn(q, k, v, polynomial_coefficients=c).astype(jnp.float32) * do),
                             argnums=(0, 1, 2, 3)))
        row += f" | {name} fwd {bench(f, q, k, v, c):6.3f} fwd+bwd {bench(g, q, k, v, c):6.3f}"
    print(row, flush=True)
