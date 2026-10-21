"""Isolate small-shape CUDA failures in polynomial GDN, one case per process.

Run ``--case matmul|forward|gradient --warps 2|4`` in an allocated GPU process.
Every stage synchronizes before reporting success, so a later compilation cannot
hide an earlier asynchronous launch failure. This is a diagnostic, not a timing
or numerical-qualification benchmark.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np


def stage(name, function):
    print(json.dumps({"stage": name, "status": "starting"}), flush=True)
    started = time.perf_counter()
    try:
        result = jax.block_until_ready(function())
        arrays = [np.asarray(value) for value in jax.tree.leaves(result)
                  if hasattr(value, "shape") and hasattr(value, "dtype")]
        summary = [{"shape": list(value.shape), "dtype": str(value.dtype),
                    "finite": bool(np.isfinite(value).all())} for value in arrays]
        print(json.dumps({"stage": name, "status": "complete",
                          "seconds": time.perf_counter() - started, "arrays": summary}), flush=True)
        return result
    except Exception as error:
        print(json.dumps({"stage": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"}), flush=True)
        raise


def matmul_case():
    rng = np.random.default_rng(71)
    matrix, keys, cotangent = stage("matmul_inputs", lambda: tuple(jnp.asarray(value, jnp.float32) for value in (
        rng.normal(size=(1, 1, 1, 64, 64)),
        rng.normal(size=(1, 1, 1, 64, 8)),
        rng.normal(size=(1, 1, 1, 64, 8)),
    )))

    def transformed(matrix, keys):
        with jax.default_matmul_precision("highest"):
            return jnp.swapaxes(matrix, -1, -2) @ keys

    stage("matmul_forward", lambda: transformed(matrix, keys))
    _, pullback = stage("matmul_vjp_setup", lambda: jax.vjp(transformed, matrix, keys))
    stage("matmul_vjp", lambda: pullback(cotangent))


def polynomial_operands():
    """Reproduce the first failing qualification fixture: seed 71, FP32 K=V=8."""
    rng = np.random.default_rng(71)
    shape = (1, 64, 1, 8)
    q, k = (rng.normal(size=shape) for _ in range(2))
    q, k = (value / np.linalg.norm(value, axis=-1, keepdims=True) for value in (q, k))
    v = rng.normal(size=shape)
    g = -rng.uniform(.002, .4, size=shape[:3])
    beta = rng.uniform(-.8, .8, size=shape[:3] + (3,))
    erase = rng.uniform(.02, .98, size=shape[:3])
    arguments = tuple(jnp.asarray(value, jnp.float32) for value in (q, k, v, g, beta, erase))
    return arguments, jnp.asarray(rng.normal(size=shape), jnp.float32)


def forward_case(warps):
    from scaffolds.transformer.attention.backends import polynomial_gdn as backend

    (q, k, v, g, beta, erase), _ = stage("forward_inputs", polynomial_operands)
    inverse = stage("inverse_forward", lambda: backend._inverse_forward(k, erase))

    def prepare():
        with jax.default_matmul_precision("highest"):
            return backend._token_view(
                jnp.swapaxes(backend._chunk_view(inverse), -1, -2) @ backend._chunk_view(k))

    transformed = stage("transformed_keys", prepare)
    prefix = stage("decay_cumsum", lambda: jnp.cumsum(g, axis=1))
    stage("polynomial_forward", lambda: backend._forward(
        q, k, v, prefix, beta, erase, inverse, transformed, 8 ** -.5,
        precision="tf32x3", num_warps=warps))


def gradient_case(warps):
    from scaffolds.transformer.attention.backends.polynomial_gdn import polynomial_gated_delta_rule

    arguments, cotangent = stage("gradient_inputs", polynomial_operands)
    function = partial(polynomial_gated_delta_rule, scale=8 ** -.5,
                       precision="tf32x3", num_warps=warps)
    stage("public_forward", lambda: function(*arguments))
    stage("all_six_gradients", lambda: jax.grad(
        lambda *values: jnp.sum(function(*values) * cotangent), argnums=tuple(range(6)))(*arguments))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("matmul", "forward", "gradient"), required=True)
    parser.add_argument("--warps", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    print(json.dumps({"case": args.case, "warps": args.warps,
                      "devices": [str(device) for device in jax.local_devices()]}), flush=True)
    if args.case == "matmul":
        matmul_case()
    elif args.case == "forward":
        forward_case(args.warps)
    else:
        gradient_case(args.warps)
    print(json.dumps({"case": args.case, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
