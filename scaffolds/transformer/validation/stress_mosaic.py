"""Call one Mosaic kernel repeatedly at the production per-GPU shape (16 x 2048 x 16 heads,
d_K = 64, sharp scores). Progress is printed every EVERY calls, so a hang shows up as a
stalled counter, and each printed result must equal the first call's bitwise (a race that
overwrites a buffer early would change it). Kernels: forward, dq, and dkv (Hopper) or dk and
dv (Blackwell).

    STRESS_DEGREE=m python stress_mosaic.py FAMILY KERNEL PRECISION CALLS [EVERY]  # m: 3 (query), 2 (Taylor)
"""
import math, os, sys, time
import jax, jax.numpy as jnp, numpy as np
from scaffolds.transformer.attention.backends.mosaic.api import _kernels, _scale
from scaffolds.transformer.attention.backends.mosaic.common import TAYLOR, QUERY

family, kernel, precision, calls = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
_, kernels = _kernels()
keys = jax.random.split(jax.random.key(0), 5)
shape = (16, 2048, 16, 64)
q, k = (jax.random.normal(key, shape, jnp.float32) for key in keys[:2])
q = (q / jnp.linalg.norm(q, axis=-1, keepdims=True) * 4 * math.sqrt(64)).astype(jnp.bfloat16)
k = (k / jnp.linalg.norm(k, axis=-1, keepdims=True)).astype(jnp.bfloat16)
v, do = (jax.random.normal(key, shape, jnp.bfloat16) for key in keys[2:4])
degree = int(os.environ.get("STRESS_DEGREE", 3 if family == QUERY else 2))
if family == QUERY:
    taylor = [1 / math.factorial(r) for r in range(degree + 1)]
    c = jnp.tile(jnp.array(taylor, jnp.float32), (16, 16, 2048, 1)).transpose(3, 0, 1, 2)
    fwd_vec = bwd_vec = c
else:
    fwd_vec = None
    bwd_vec = jnp.stack([jax.random.normal(keys[4], (16, 16, 2048), jnp.float32), jnp.full((16, 16, 2048), 1e-3, jnp.float32)])
common = dict(family=family, degree=degree, scale=_scale(q), precision=precision)
if kernel == "forward":
    run = jax.jit(lambda: kernels.forward(q, k, v, fwd_vec, **common))
elif kernel == "dq":
    run = jax.jit(lambda: kernels.dq(q, k, v, do, bwd_vec, **common))
elif kernel == "dkv":
    run = jax.jit(lambda: kernels.dkv(q, k, v, do, bwd_vec, **common))
else:  # Blackwell splits dK and dV
    run = jax.jit(lambda: kernels.dkv(q, k, v, do, bwd_vec, which=kernel, **common))
print("compiling", flush=True)
reference = jax.tree.map(np.asarray, run())
print("first call done", flush=True)
start = time.time()
for i in range(1, calls + 1):
    out = run()
    if i % EVERY == 0:
        same = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(jax.tree.leaves(jax.tree.map(np.asarray, out)), jax.tree.leaves(reference)))
        print(f"{family} {kernel} {precision} {i} calls {time.time() - start:.0f}s identical={same}", flush=True)
        if not same:
            sys.exit(2)
jax.block_until_ready(out)
print(f"{family} {kernel} {precision} done {calls} calls", flush=True)
