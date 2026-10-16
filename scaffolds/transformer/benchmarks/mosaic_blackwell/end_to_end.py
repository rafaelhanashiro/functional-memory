"""End-to-end: Mosaic Blackwell normalized Taylor attention (fwd + dQ + dV + dK) vs Triton and cuDNN."""
import sys
from functools import partial
import numpy as np, jax, jax.numpy as jnp

from scaffolds.transformer.benchmarks.mosaic_blackwell import taylor_forward as fwd, taylor_dq as bdq, taylor_dkv as bkv
from scaffolds.transformer.benchmarks.mosaic_blackwell.common import inputs, bench
from scaffolds.transformer.attention.backends import fused_taylor as ft

@partial(jax.custom_vjp, nondiff_argnums=(3,))
def mosaic_attention(q, k, v, degree):
    return fwd.forward(q, k, v, degree=degree, stages=3)[0].astype(v.dtype)
def _f(q, k, v, degree):
    out, norm = fwd.forward(q, k, v, degree=degree, stages=3)
    return out.astype(v.dtype), (q, k, v, out, norm)
def _b(degree, res, do):
    q, k, v, out, norm = res
    delta = jnp.sum(do.astype(jnp.float32) * out, axis=-1).transpose(0, 2, 1)
    invz = (1.0 / norm).transpose(0, 2, 1)
    return (bdq.dq(q, k, v, do, delta, invz, degree=degree),
            bkv.dkv(q, k, v, do, delta, invz, which="dk", degree=degree),
            bkv.dkv(q, k, v, do, delta, invz, which="dv", degree=degree))
mosaic_attention.defvjp(_f, _b)

Q, K, V = inputs(16, 2048, 16, 64)
DO = jax.random.normal(jax.random.key(9), V.shape, jnp.bfloat16)
for degree in (2, 4):
    fns = {"Mosaic": lambda q, k, v: mosaic_attention(q, k, v, degree),
           "Triton": lambda q, k, v: ft.fused_polynomial_attention(q, k, v, attention_degree=degree)}
    grads = {}
    for name, fn in fns.items():
        g = jax.jit(jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v).astype(jnp.float32) * DO), argnums=(0, 1, 2)))
        grads[name] = g(Q, K, V)
        print(f"m={degree} {name}: fwd {bench(jax.jit(fn), Q, K, V):.3f} ms, fwd+bwd {bench(g, Q, K, V):.3f} ms", flush=True)
    diffs = [float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))) / jnp.max(jnp.abs(b.astype(jnp.float32))))
             for a, b in zip(grads["Mosaic"], grads["Triton"])]
    print(f"m={degree} Mosaic vs Triton gradient max rel diff (dQ, dK, dV): {[f'{d:.1e}' for d in diffs]}", flush=True)
cud = lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
gc = jax.jit(jax.grad(lambda q, k, v: jnp.sum(cud(q, k, v).astype(jnp.float32) * DO), argnums=(0, 1, 2)))
print(f"cuDNN softmax: fwd {bench(jax.jit(cud), Q, K, V):.3f} ms, fwd+bwd {bench(gc, Q, K, V):.3f} ms", flush=True)
