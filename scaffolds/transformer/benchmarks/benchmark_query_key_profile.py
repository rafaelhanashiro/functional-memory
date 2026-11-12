"""Matched resident-array attention diagnostics for the 200M / 4B setup.

Compare cuDNN softmax with the production query-key kernel and an isolated
32x32 backward-tile override. Forward geometry and arithmetic stay unchanged.
Optional traces capture one warmed forward/backward call per configuration.
This measures attention, not coefficient projections or full training updates.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import math
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
from scaffolds.transformer.attention.softmax import padded_dot_product_attention
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, time_program, write_report
from scaffolds.transformer.benchmarks import tune_query_key as tuning


def softmax(q, k, v):
    return padded_dot_product_attention(q, k, v, backend="cudnn")


def query_only(q, k, v, coefficients):
    from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
    return query_polynomial_attention(q, k, v, coefficients)


def backward_program(forward, operand_count):
    def objective(*arguments):
        output = forward(*arguments[:operand_count])
        value = jnp.sum(output.astype(jnp.float32) * arguments[-1].astype(jnp.float32))
        # Keep the actual forward output observable through the custom VJP.
        return value, output
    return jax.value_and_grad(objective, argnums=tuple(range(operand_count)), has_aux=True)


def profile(program, arguments, destination, label):
    compiled = jax.jit(program).lower(*arguments).compile()
    jax.block_until_ready(compiled(*arguments))
    destination.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(destination), create_perfetto_trace=True):
        with jax.profiler.TraceAnnotation("attention_forward_backward", configuration=label):
            jax.block_until_ready(compiled(*arguments))
    return {"directory": str(destination.resolve()), "calls": 1,
            "annotation": "attention_forward_backward", "timing_excluded": True}


def measure(forward, operands, cotangent, entry, report, args):
    backward = backward_program(forward, len(operands))
    outputs = {}
    for name, program, arguments in (
        ("forward", forward, operands),
        ("forward_backward", backward, (*operands, cotangent)),
    ):
        report["active_stage"] = {"configuration": entry["id"], "operation": name}
        write_report(args.output, report)
        print(f"configuration={entry['id']} operation={name}: compiling and timing", flush=True)
        entry[name], outputs[name] = time_program(
            program, arguments, warmup=args.warmup, iterations=args.iterations)
        print(f"configuration={entry['id']} operation={name} "
              f"median_ms={entry[name]['median_ms']:.6f}", flush=True)
        write_report(args.output, report)
    entry["all_finite"] = all(bool(jax.device_get(jnp.all(jnp.isfinite(array))))
                              for array in jax.tree.leaves(outputs))
    if args.profile_dir is not None:
        report["active_stage"]["operation"] = "profile"
        write_report(args.output, report)
        try:
            entry["profile"] = profile(backward, (*operands, cotangent),
                                       args.profile_dir / entry["id"], entry["id"])
        except Exception as error:
            # Timings and parity remain useful if profiler support is unavailable.
            entry["profile"] = {"error": f"{type(error).__name__}: {error}"}
    return outputs


def benchmark(args):
    devices = jax.local_devices(backend="gpu")
    if args.device_index >= len(devices):
        raise ValueError(f"Requested GPU {args.device_index}; only {len(devices)} visible")
    device = devices[args.device_index]
    report = {
        "format": "query-key-matched-attention-profile", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(device),
        "configuration": {
            **{key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()},
            "shape": [args.batch_size, args.sequence_length, args.num_heads, args.head_dim],
            "dtype": "bfloat16", "scope": "attention only; excludes projections and training runtime",
            "timing": "resident compiled calls, synchronized; compilation, parity, profiles excluded",
            "inputs": "tune_query_key.inputs: full-factorial varying query coefficients and key factors",
            "score_formula": "score_scale*cosine(q,k), normalized before BF16 conversion",
            "precision": "unchanged production FP32 polynomial and BF16x3 mixed products",
            "objective": "dynamic cotangent; forward output and every operand gradient returned",
            "parity": "32x32 backward versus immutable production snapshots; all five gradients, per degree/head",
            "limits": "synthetic inputs; no trained-checkpoint parity or register/spill counters",
        },
        "results": [],
    }
    root = Path(__file__).resolve().parents[3]
    for source in (Path(__file__), Path(tuning.__file__)):
        report["provenance"]["source_sha256"][str(source.resolve().relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for package in ("jax-cuda13-plugin", "nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cudnn-cu13"):
        try:
            report["provenance"]["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    started = time.monotonic()
    write_report(args.output, report)
    try:
        with jax.default_device(device):
            first_arrays = tuning.inputs(args, args.degrees[0])
            entry = {"id": "softmax_cudnn", "status": "running", "gradients": ["q", "k", "v"]}
            report["results"].append(entry)
            outputs = measure(softmax, first_arrays[:3], first_arrays[-1], entry, report, args)
            entry["status"] = "accepted" if entry["all_finite"] else "numerical_failure"
            del outputs, first_arrays
            for degree in args.degrees:
                arrays = tuning.inputs(args, degree)
                reference = None
                for variant in ("production", "backward32"):
                    tuning.set_candidate(None)
                    if variant == "backward32":
                        kernel._backward_tile_sizes = lambda dq, dv, **kwargs: (32, 32)
                    entry = {
                        "id": f"query_key_m{degree}_{variant}", "degree": degree, "status": "running",
                        "gradients": list(tuning.INPUT_NAMES),
                        "forward_tiles": list(kernel._tile_sizes(args.head_dim, args.head_dim)),
                        "backward_tiles": list(kernel._backward_tile_sizes(
                            args.head_dim, args.head_dim, degree=degree, time=args.sequence_length,
                            dtype=jnp.bfloat16, interpret=False)),
                        "num_warps": kernel.NUM_WARPS, "num_stages": kernel.NUM_STAGES,
                    }
                    report["results"].append(entry)
                    outputs = measure(tuning.forward, arrays[:5], arrays[-1], entry, report, args)
                    if variant == "production":
                        reference = tuning.host_snapshot(outputs)
                    entry["parity"], accepted = tuning.compare(outputs, reference)
                    entry["status"] = "accepted" if accepted else "numerical_failure"
                    del outputs
                    write_report(args.output, report)
                if args.include_query_mosaic:
                    tuning.set_candidate(None)
                    entry = {"id": f"query_m{degree}_mosaic", "degree": degree, "status": "running",
                             "gradients": ["q", "k", "v", "coefficients"],
                             "comparison": "different mechanism; no parity comparison to query-key"}
                    report["results"].append(entry)
                    outputs = measure(query_only, arrays[:4], arrays[-1], entry, report, args)
                    entry["status"] = "accepted" if entry["all_finite"] else "numerical_failure"
                    del outputs
                del arrays, reference
            report["status"] = ("complete" if all(e["status"] == "accepted" for e in report["results"])
                                else "numerical_failure")
            report.pop("active_stage", None)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        tuning.set_candidate(None)
        report["elapsed_seconds"] = time.monotonic() - started
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--include-query-mosaic", action="store_true")
    parser.add_argument("--degrees", nargs="+", type=int, default=[8, 12])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--score-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.num_heads, args.head_dim) <= 0 or args.head_dim > 64:
        parser.error("Require positive dimensions and head-dim <= 64")
    if args.device_index < 0 or args.warmup < 3 or args.iterations < 10:
        parser.error("Require device-index >= 0, warmup >= 3, iterations >= 10")
    if not math.isfinite(args.score_scale) or args.score_scale <= 0:
        parser.error("Require a positive finite score-scale")
    if len(set(args.degrees)) != len(args.degrees) or any(d not in (8, 12) for d in args.degrees):
        parser.error("Require distinct degrees from 8 and 12")
    result = benchmark(args)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
