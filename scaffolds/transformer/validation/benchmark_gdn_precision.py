"""Measure repaired tied/independent GDN kernels on one allocated GPU.

The default large case is the attention shape per GPU for the canonical recipe
on two devices: B=32, T=2048, H=16, K=V=64. Features, gates and attention outputs
use BF16; log decay uses FP32, and gradients match their inputs' dtypes.
Compilation, warmup, input creation and finite checks
are excluded from synchronized timings. Compiler memory estimates cover this
attention call, not a complete model or a measured GPU allocator peak.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
    fla_gated_delta_rule, fla_independent_gated_delta_rule,
)
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, time_program, write_report


def operands(batch: int, length: int, heads: int, width: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (batch, length, heads, width)

    def normal():
        return rng.standard_normal(shape, dtype=np.float32)

    q, k = normal(), normal()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = normal()
    gate_shape = shape[:-1]
    rate = np.exp(rng.uniform(0, np.log(16), (1, 1, heads)).astype(np.float32))
    g = -rate * np.logaddexp(0, rng.standard_normal(gate_shape, dtype=np.float32) * .5 - 4)
    beta, erase = (1 / (1 + np.exp(-rng.standard_normal(gate_shape, dtype=np.float32)))
                   for _ in range(2))
    cast = lambda value: jnp.asarray(value, jnp.bfloat16)
    return (cast(q), cast(k), cast(v), jnp.asarray(g, jnp.float32), cast(beta), cast(erase)), cast(normal())


def finite_summary(result) -> list[dict]:
    """Only small on-device reductions are copied to the host, outside timing."""
    summary = []
    for value in jax.tree.leaves(result):
        value32 = value.astype(jnp.float32)
        finite, maximum = jax.device_get((jnp.all(jnp.isfinite(value32)), jnp.max(jnp.abs(value32))))
        summary.append({"shape": list(value.shape), "dtype": str(value.dtype),
                        "finite": bool(finite), "max_abs": float(maximum) if finite else None})
    if not all(item["finite"] for item in summary):
        raise FloatingPointError(f"Nonfinite benchmark output or gradient: {summary}")
    return summary


def benchmark(args):
    devices = jax.local_devices(backend="gpu")
    if len(devices) != 1:
        raise ValueError(f"Kernel benchmark requires exactly one visible GPU; found {len(devices)}")
    report = {
        "format": "gdn-precision-kernel-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(devices[0]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(Path("scaffolds/transformer/uv.lock").read_bytes()).hexdigest(),
        "launch_source_commit": os.environ.get("GDN_BENCHMARK_SOURCE"),
        "configuration": {
            "batches": args.batches, "length": args.length, "heads": args.heads, "width": args.width,
            "dtype": "bfloat16", "log_decay_dtype": "float32", "seed": args.seed,
            "modes": args.modes, "warmup": args.warmup, "iterations": args.iterations,
            "timing": "compiled call plus host dispatch; synchronize all outputs after every call",
            "backward_objective": "sum(float32(output) * float32(cotangent)); all input gradients",
            "memory": "compiled.memory_analysis estimates; not measured allocator peaks or full-model memory",
        },
        "runtime_environment": {name: os.environ.get(name) for name in (
            "XLA_FLAGS", "TRITON_F32_DEFAULT", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "TRANSFORMER_HOST_STAGING",
        )},
        "measurements": [],
    }
    write_report(args.output, report)
    try:
        for batch in args.batches:
            all_inputs, cotangent = operands(batch, args.length, args.heads, args.width, args.seed)
            jax.block_until_ready((all_inputs, cotangent))
            scale = args.width ** -.5
            for mode in args.modes:
                implementation = fla_gated_delta_rule if mode == "tied" else fla_independent_gated_delta_rule
                inputs = all_inputs[:5] if mode == "tied" else all_inputs

                def forward(*values):
                    return implementation(*values, scale)

                def objective(*values):
                    output = forward(*values[:-1])
                    loss = jnp.sum(output.astype(jnp.float32) * values[-1].astype(jnp.float32))
                    return loss, output

                backward = jax.value_and_grad(objective, argnums=tuple(range(len(inputs))), has_aux=True)
                for operation, function, arguments in (
                    ("forward", forward, inputs),
                    ("forward_and_backward", backward, (*inputs, cotangent)),
                ):
                    print(f"START batch={batch} mode={mode} operation={operation}", flush=True)
                    timing, result = time_program(function, arguments, warmup=args.warmup,
                                                  iterations=args.iterations)
                    summary = finite_summary(result)
                    report["measurements"].append({
                        "batch": batch, "mode": mode, "operation": operation,
                        **timing, "outputs": summary,
                    })
                    write_report(args.output, report)
                    print(f"DONE batch={batch} mode={mode} operation={operation} "
                          f"median_ms={timing['median_ms']:.3f} p90_ms={timing['p90_ms']:.3f}", flush=True)
                    del result
            del all_inputs, inputs, cotangent, arguments
        report["status"] = "complete"
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(args.output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", nargs="+", choices=("tied", "independent"), default=["tied", "independent"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()
    if min(*args.batches, args.length, args.heads, args.width, args.warmup, args.iterations) < 1:
        parser.error("Shapes and timing counts must be positive")
    if args.length % 64:
        parser.error("Sequence length must be a multiple of the GDN chunk size (64)")
    benchmark(args)


if __name__ == "__main__":
    main()
