"""CPU oracle and cancellation diagnostics for repeated-key GDN qualification.

Run from the repository root with its locked transformer environment:
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 scaffolds/transformer/.venv/bin/python \
    scaffolds/transformer/validation/gdn_precision_cpu.py

No GPU implementation is simulated here. The scan checks the analytic oracle;
the exact chunk decomposition isolates its separately stored key-gradient terms.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

NAMES = ("dq", "dk", "dv", "dg", "dbeta", "derase")
SCALE = 64 ** -0.5


def operands(dtype):
    """Identical RNG consumption and casts to seed-44 GPU qualification case."""
    rng = np.random.default_rng(44)
    shape = (1, 128, 2, 64)
    rng.standard_normal(shape)
    rng.standard_normal(shape)
    value = rng.standard_normal(shape)
    rng.uniform(0, np.log(16), (1, 1, 2))
    rng.standard_normal((1, 128, 2))
    beta = 1 / (1 + np.exp(-rng.standard_normal((1, 128, 2))))
    cotangent = rng.standard_normal(shape)
    key = np.zeros(shape)
    key[..., 0] = 1
    cast = lambda x: jnp.asarray(x, dtype)
    return (cast(key), cast(key), cast(value), jnp.zeros((1, 128, 2), jnp.float32),
            cast(beta), jnp.ones((1, 128, 2), dtype)), cast(cotangent)


def oracle(args, cotangent):
    q, k, v, _, beta, _ = (np.asarray(x, np.float64) for x in args)
    cot = np.asarray(cotangent, np.float64)
    association = beta[..., None] * v
    previous = np.concatenate((np.zeros_like(association[:, :1]), association[:, :-1]), axis=1)
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dq[..., 0] = SCALE * np.sum(cot * association, axis=-1)
    dk[..., 0] = SCALE * np.sum(cot * (association - 2 * previous), axis=-1)
    return (SCALE * association, (dq, dk, SCALE * beta[..., None] * cot,
            np.zeros_like(beta), SCALE * np.sum(cot * v, axis=-1),
            -SCALE * np.sum(cot * previous, axis=-1)))


def scan(q, k, v, g, beta, erase):
    dtype = q.dtype
    g = g.astype(dtype)
    def step(state, xs):
        qi, ki, vi, gi, bi, ei = xs
        state = state * jnp.exp(gi)[..., None, None]
        pred = jnp.einsum("bhk,bhkv->bhv", ki, state)
        update = bi[..., None] * vi - ei[..., None] * pred
        state = state + jnp.einsum("bhk,bhv->bhkv", ki, update)
        return state, jnp.einsum("bhk,bhkv->bhv", qi * SCALE, state)
    initial = jnp.zeros((q.shape[0], q.shape[2], q.shape[3], v.shape[-1]), dtype)
    _, out = jax.lax.scan(step, initial, tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta, erase)))
    return jnp.moveaxis(out, 0, 1)


def metric(actual, expected):
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    error = np.linalg.norm(actual - expected)
    norm = np.linalg.norm(expected)
    return {"l2_error": float(error), "reference_l2": float(norm),
            "relative_l2": float(error / norm) if norm else None,
            "absolute_max": float(np.max(np.abs(actual - expected)))}


def chunked_split_keys(q, read_key, wy_key, v, beta, erase):
    """Exact zero-decay two-chunk algebra, with separate appearances of k.

    The read/state-update uses read_key. The WY inverse and W use wy_key.
    Their gradients sum to the total key gradient when both aliases equal k.
    """
    q, read_key, wy_key, v = (jnp.transpose(x[0], (1, 0, 2)) for x in (q, read_key, wy_key, v))
    beta, erase = (jnp.transpose(x[0], (1, 0)) for x in (beta, erase))
    state = jnp.zeros((q.shape[0], q.shape[-1], v.shape[-1]), q.dtype)
    outputs = []
    for start in (0, 64):
        sl = slice(start, start + 64)
        qi, kr, kw, vi, bi, ei = (x[:, sl] for x in (q, read_key, wy_key, v, beta, erase))
        system = jnp.eye(64, dtype=q.dtype) + jnp.tril(ei[..., None] * (kw @ jnp.swapaxes(kw, -1, -2)), -1)
        inverse = jnp.linalg.inv(system)
        w = inverse @ (ei[..., None] * kw)
        u = inverse @ (bi[..., None] * vi)
        vnew = u - w @ state
        out = qi @ state + jnp.tril(qi @ jnp.swapaxes(kr, -1, -2)) @ vnew
        outputs.append(out * SCALE)
        state = state + jnp.swapaxes(kr, -1, -2) @ vnew
    return jnp.transpose(jnp.concatenate(outputs, axis=1), (1, 0, 2))[None]


def main():
    report = {"case": "seed44_repeated_e1_keys_T128_H2_K64_V64_g0_erase1",
              "jax_version": jax.__version__, "backend": jax.default_backend(),
              "scan": {}, "chunk_gradient_cancellation": {},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dependency_lock_sha256": hashlib.sha256((Path(__file__).parent.parent / "uv.lock").read_bytes()).hexdigest()}
    with jax.default_matmul_precision("highest"):
        for input_name, input_dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
            args, cot = operands(input_dtype)
            expected_out, expected_grads = oracle(args, cot)
            for compute_name, compute_dtype in (("float64", jnp.float64), ("float32", jnp.float32),
                                               ("bfloat16", jnp.bfloat16)):
                compute_args = tuple(x.astype(compute_dtype) for x in args)
                compute_cot = cot.astype(compute_dtype)
                loss = lambda *xs: jnp.sum(scan(*xs).astype(compute_dtype) * compute_cot)
                out = jax.jit(scan)(*compute_args)
                grads = jax.jit(jax.grad(loss, argnums=tuple(range(6))))(*compute_args)
                report["scan"][f"input_{input_name}_compute_{compute_name}"] = {
                    "output": metric(out, expected_out),
                    **{name: metric(actual, expected) for name, actual, expected in zip(NAMES, grads, expected_grads)},
                }
            args64 = tuple(x.astype(jnp.float64) for x in args)
            q, k, v, _, beta, erase = args64
            cot64 = cot.astype(jnp.float64)
            loss = lambda read_key, wy_key: jnp.sum(chunked_split_keys(q, read_key, wy_key, v, beta, erase) * cot64)
            parts = jax.jit(jax.grad(loss, argnums=(0, 1)))(k, k)
            expected = expected_grads[1]
            direct, wy = (np.asarray(x, np.float64) for x in parts)
            entry = {
                "exact_parts_sum": metric(direct + wy, expected),
                "direct_part_l2": float(np.linalg.norm(direct)),
                "wy_part_l2": float(np.linalg.norm(wy)),
                "sum_of_part_norms_over_result_norm": float((np.linalg.norm(direct) + np.linalg.norm(wy)) / np.linalg.norm(expected)),
            }
            for name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
                rounded = [np.asarray(jnp.asarray(x, dtype), np.float64) for x in parts]
                entry[f"round_each_part_to_{name}_then_sum_float64"] = metric(sum(rounded), expected)
                entry[f"sum_exact_parts_then_round_to_{name}"] = metric(jnp.asarray(direct + wy, dtype), expected)
            report["chunk_gradient_cancellation"][input_name] = entry
    output = Path(__file__).parent / "runs/gdn-independent-precision-cpu/diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
