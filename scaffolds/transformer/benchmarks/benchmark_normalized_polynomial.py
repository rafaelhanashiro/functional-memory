"""Compare fused and blocked normalized learned polynomials on one allocated GPU.

The default is the production per-device attention shape, with Taylor-initialized
parameters and QK-normalized scores at scale sqrt(d_K). Timings include coefficient
or square-factor gradients but exclude input generation, compilation, diagnostics,
and transfers. This measures attention, not a complete Transformer update.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.learned_polynomial import normalized_learned_polynomial_attention
from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.benchmarks.benchmark_attention import (
    error_norms, provenance, time_program, write_report,
)


def _forward(q, k, v, parameters, *, degree, parameterization, backend):
    argument = "polynomial_coefficients" if parameterization == "direct" else "polynomial_square_factors"
    return normalized_learned_polynomial_attention(
        q, k, v, attention_degree=degree, backend=backend, **{argument: parameters})


def _value_and_vjp(forward, q, k, v, parameters, cotangent):
    output, pullback = jax.vjp(forward, q, k, v, parameters)
    return output, pullback(cotangent)


def benchmark(args):
    device = jax.local_devices(backend="gpu")[0]
    shape = (args.batch_size, args.sequence_length, args.num_heads, args.head_dim)
    report = {
        "format": "normalized-learned-polynomial-kernel-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(device), "configuration": vars(args) | {"output": str(args.output)},
        "shape": list(shape), "results": [],
        "timing": "synchronized compiled calls; compilation and diagnostics excluded",
        "scope": "attention only, including Q/K/V and every coefficient or square-factor gradient",
        "input_distribution": "independent Gaussian Q/K/V, QK-norm with scale sqrt(d_K); Taylor parameters",
        "memory": "compiler buffer estimates; not measured allocator peaks",
    }
    root = Path(__file__).resolve().parents[3]
    report["provenance"]["source_sha256"][str(Path(__file__).relative_to(root))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    write_report(args.output, report)
    try:
        rng = np.random.default_rng(args.seed)
        dtype = getattr(jnp, args.dtype)
        q, k, v, cotangent = [jnp.asarray(rng.standard_normal(shape), dtype) for _ in range(4)]
        q, k = cosine_score_operands(q, k, jnp.array([math.log(math.sqrt(args.head_dim))]), eps=1e-6)
        jax.block_until_ready((q, k, v, cotangent))
        for parameterization in args.parameterizations:
            for degree in args.degrees:
                if parameterization == "sum_of_squares" and degree % 2:
                    continue
                initial = ([1 / math.factorial(r) for r in range(degree + 1)]
                           if parameterization == "direct" else taylor_square_factors(degree))
                parameters = jnp.broadcast_to(jnp.asarray(initial, jnp.float32), (args.num_heads, degree + 1))
                entry = {"degree": degree, "parameterization": parameterization, "implementations": {}}
                report["results"].append(entry)
                results = {}
                for backend in args.implementations:
                    forward = partial(_forward, degree=degree, parameterization=parameterization, backend=backend)
                    operands = (q, k, v, parameters)
                    timings, outputs = {}, {}
                    for operation, function, arguments in (
                        ("forward", forward, operands),
                        ("forward_backward", partial(_value_and_vjp, forward), (*operands, cotangent)),
                    ):
                        report["active_stage"] = dict(degree=degree, parameterization=parameterization,
                                                      backend=backend, operation=operation)
                        write_report(args.output, report)
                        timings[operation], outputs[operation] = time_program(
                            function, arguments, warmup=args.warmup, iterations=args.iterations)
                        print(f"{parameterization} m={degree} {backend} {operation}: "
                              f"{timings[operation]['median_ms']:.3f} ms", flush=True)
                    entry["implementations"][backend] = timings
                    results[backend] = outputs
                    write_report(args.output, report)
                if {"blocked", "fused"}.issubset(results):
                    actual, reference = results["fused"]["forward_backward"], results["blocked"]["forward_backward"]
                    entry["parity_against_blocked"] = {
                        "output": error_norms(actual[0], reference[0]),
                        **{f"gradient_{name}": error_norms(a, b)
                           for name, a, b in zip(("q", "k", "v", "parameters"), actual[1], reference[1], strict=True)},
                    }
                    entry["speedup_blocked_over_fused"] = {
                        op: entry["implementations"]["blocked"][op]["median_ms"] /
                        entry["implementations"]["fused"][op]["median_ms"]
                        for op in ("forward", "forward_backward")}
                    if not all(x["all_finite"] for x in entry["parity_against_blocked"].values()):
                        raise FloatingPointError(f"Nonfinite benchmark result: {parameterization}, degree {degree}")
                write_report(args.output, report)
                del results
                jax.clear_caches()
        report["status"] = "complete"
        report.pop("active_stage", None)
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(args.output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--parameterizations", choices=("direct", "sum_of_squares"), nargs="+",
                        default=["direct", "sum_of_squares"])
    parser.add_argument("--implementations", choices=("blocked", "fused"), nargs="+", default=["blocked", "fused"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.degrees + [args.batch_size, args.sequence_length, args.num_heads, args.head_dim, args.warmup]) < 1:
        parser.error("Degrees, dimensions, and warmup must be positive")
    if args.iterations < 10:
        parser.error("At least ten timed calls are required")
    benchmark(args)


if __name__ == "__main__":
    main()
