"""Tune fused Taylor-attention tiles on one allocated GPU (never a login node).

  python -u -m scaffolds.transformer.benchmarks.tune_attention \
    --degrees 2 16 --output /path/on/nfs/tile-tuning.json

The fixed 64x64, four-warp, two-stage reference is always measured first for
each degree. Reference outputs and Q/K/V gradients are copied to independent
host arrays before compilation caches are cleared. Candidate tiles change only
launch geometry. Numerical failures and compilation/resource errors are retained
in the report and excluded from performance rankings. Timings exclude compilation,
input generation, host snapshots, and numerical comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import gc
import hashlib
import math
from pathlib import Path
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.benchmarks.benchmark_attention import error_norms, provenance, time_program, write_report
from scaffolds.transformer.attention.backends import fused_taylor as kernel


BASELINE = (64, 64, 4, 2)
DEFAULT_TILES = ((32, 64), (64, 32), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128))
GLOBALS = ("BLOCK_Q", "BLOCK_K", "NUM_WARPS", "NUM_STAGES")
# Same BF16 relative-L2 and relative-max limits as test_fused_attention.py.
TOLERANCES = {"forward": 0.006, "gradient_q": 0.025, "gradient_k": 0.025, "gradient_v": 0.015}


def tile_argument(value: str) -> tuple[int, int]:
    try:
        q, k = (int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a tile such as 64x128") from error
    if any(dimension < 16 or dimension > 256 or dimension & (dimension - 1) for dimension in (q, k)):
        raise argparse.ArgumentTypeError("Tile dimensions must be powers of two from 16 to 256")
    return q, k


def candidate_config(candidate: tuple[int, int, int, int]) -> dict:
    q, k, warps, stages = candidate
    return {"id": f"q{q}_k{k}_w{warps}_s{stages}", "block_q": q, "block_k": k,
            "num_warps": warps, "num_stages": stages}


def set_candidate(candidate: tuple[int, int, int, int]) -> None:
    # Globals are captured while tracing custom_vjp/Pallas. Clearing every JAX
    # compilation/tracing cache is essential before reusing input signatures.
    jax.clear_caches()
    gc.collect()
    for name, value in zip(GLOBALS, candidate, strict=True):
        setattr(kernel, name, value)


def host_snapshot(value):
    def copy(array):
        snapshot = np.array(jax.device_get(array), copy=True)
        snapshot.setflags(write=False)
        return snapshot
    return jax.tree.map(copy, value)


def measure_candidate(candidate, degree, arrays, entry, args, report) -> dict:
    set_candidate(candidate)
    q, k, v, cotangent = arrays

    def forward(q, k, v):
        return kernel.fused_polynomial_attention(q, k, v, attention_degree=degree)

    def objective(q, k, v, cotangent):
        return jnp.sum(forward(q, k, v).astype(jnp.float32) * cotangent.astype(jnp.float32))

    programs = (
        ("forward", forward, (q, k, v)),
        ("forward_backward", jax.value_and_grad(objective, argnums=(0, 1, 2)), arrays),
    )
    outputs = {}
    for operation, program, arguments in programs:
        report["active_stage"] = {"degree": degree, "candidate": entry["config"]["id"], "operation": operation}
        write_report(args.output, report)
        print(f"degree={degree} candidate={entry['config']['id']} operation={operation}: compiling and timing", flush=True)
        timing, output = time_program(program, arguments, warmup=args.warmup, iterations=args.iterations)
        entry[operation] = timing
        outputs[operation] = output
        print(f"degree={degree} candidate={entry['config']['id']} operation={operation} median_ms={timing['median_ms']:.6f}", flush=True)
        write_report(args.output, report)
    return outputs


def compare_outputs(outputs: dict, reference: dict) -> tuple[dict, bool]:
    actual_value, actual_grads = outputs["forward_backward"]
    reference_value, reference_grads = reference["forward_backward"]
    pairs = [("forward", outputs["forward"], reference["forward"]), ("objective", actual_value, reference_value)]
    pairs += [(f"gradient_{name}", actual, expected)
              for name, actual, expected in zip(("q", "k", "v"), actual_grads, reference_grads, strict=True)]
    parity = {}
    for name, actual, expected in pairs:
        # Move the host baseline only for the untimed comparison. It remains
        # host-owned and independent of candidate results and cleared caches.
        expected_device = jax.device_put(expected)
        norms = error_norms(actual, expected_device)
        tolerance = TOLERANCES.get(name)
        accepted = norms["all_finite"]
        if tolerance is not None:
            relative = norms["relative_l2_error"]
            accepted = accepted and relative is not None and relative <= tolerance
            accepted = accepted and norms["max_abs_error"] is not None and norms["reference_max_abs"] is not None
            if accepted:
                accepted = norms["max_abs_error"] <= 3 * tolerance * norms["reference_max_abs"]
            # Gate each batch/head independently: a small bad head must not
            # disappear in an aggregate norm over the production tensor.
            actual32, expected32 = actual.astype(jnp.float32), expected_device.astype(jnp.float32)
            difference = actual32 - expected32
            axes = (1, 3)
            squared_error = jnp.sum(jnp.square(difference), axis=axes)
            squared_reference = jnp.sum(jnp.square(expected32), axis=axes)
            relative_per_head = jnp.sqrt(squared_error / jnp.where(squared_reference > 0, squared_reference, 1.0))
            head_max_error = jnp.max(jnp.abs(difference), axis=axes)
            head_max_reference = jnp.max(jnp.abs(expected32), axis=axes)
            head_passed = (squared_error <= tolerance ** 2 * squared_reference) & (head_max_error <= 3 * tolerance * head_max_reference)
            head_stats = jax.device_get({"all_passed": jnp.all(head_passed),
                                         "max_relative_l2": jnp.max(relative_per_head),
                                         "nonzero_error_with_zero_reference": jnp.any((squared_reference == 0) & (squared_error != 0))})
            head_relative = float(head_stats["max_relative_l2"])
            norms["per_batch_head"] = {
                "all_passed": bool(head_stats["all_passed"]),
                "max_relative_l2_error": head_relative if math.isfinite(head_relative) else None,
                "nonzero_error_with_zero_reference": bool(head_stats["nonzero_error_with_zero_reference"]),
            }
            accepted = accepted and bool(head_stats["all_passed"])
        norms["relative_l2_tolerance"] = tolerance
        norms["relative_max_tolerance"] = 3 * tolerance if tolerance is not None else None
        norms["accepted"] = bool(accepted)
        parity[name] = norms
    return parity, all(norms["accepted"] for norms in parity.values())


def update_best(degree_entry: dict) -> None:
    accepted = [entry for entry in degree_entry["candidates"] if entry["status"] == "accepted"]
    degree_entry["best"] = {
        operation: {
            "candidate": best["config"], "median_ms": best[operation]["median_ms"],
            "speedup_over_baseline": degree_entry["baseline"][operation]["median_ms"] / best[operation]["median_ms"],
        }
        for operation in ("forward", "forward_backward")
        if accepted
        for best in [min(accepted, key=lambda entry: entry[operation]["median_ms"])]
    }


def benchmark(args) -> dict:
    devices = jax.local_devices(backend="gpu")
    if args.device_index >= len(devices):
        raise ValueError(f"Requested GPU {args.device_index}; found {len(devices)} GPUs")
    device = devices[args.device_index]
    shape = (args.batch_size, args.sequence_length, args.num_heads, args.head_dim)
    requested = [(q, k, warps, stages) for q, k in args.tiles for warps in args.warps for stages in args.stages]
    candidates = [BASELINE] + [candidate for candidate in requested if candidate != BASELINE]
    report = {
        "format": "polynomial-attention-tile-tuning", "schema_version": 1, "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(), "provenance": provenance(device),
        "configuration": {
            "shape": list(shape), "layout": "batch,time,heads,channels", "dtype": "bfloat16",
            "degrees": args.degrees, "seed": args.seed, "qk_scale": args.qk_scale,
            "warmup": args.warmup, "iterations": args.iterations,
            "baseline": candidate_config(BASELINE),
            "requested_candidates": [candidate_config(candidate) for candidate in requested],
            "evaluated_candidates": [candidate_config(candidate) for candidate in candidates],
            "baseline_policy": "always included; measured once per degree before other candidates",
            "reference_storage": "independent read-only host arrays; never cached device outputs",
            "cache_policy": "jax.clear_caches before every candidate, then set BLOCK_Q/BLOCK_K/NUM_WARPS/NUM_STAGES",
            "input_distribution": "fixed independent normal(0,1); Q/K scaled before BF16 conversion",
            "objective": "sum(float32(output) * float32(cotangent)); Q/K/V gradients",
            "parity": {"relative_l2_tolerances": TOLERANCES, "relative_max_multiplier": 3,
                       "scope": "global and every batch/head; exact zero references require exact zero errors",
                       "objective": "finite-only gate; cancellation-sensitive scalar error recorded",
                       "limitation": "tile-invariance check against the same semantic kernel, not an independent correctness proof"},
            "timing": "synchronized compiled calls; compilation, reference transfers, and parity checks excluded",
            "memory": "compiler estimates from time_program; not measured allocator peak",
        },
        "results": [],
    }
    path = Path(__file__).resolve()
    report["provenance"]["source_sha256"][str(path.relative_to(path.parents[2]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    original = tuple(getattr(kernel, name) for name in GLOBALS)
    write_report(args.output, report)
    try:
        with jax.default_device(device):
            keys = jax.random.split(jax.random.PRNGKey(args.seed), 4)
            arrays = tuple((jax.random.normal(key, shape, dtype=jnp.float32) *
                            (args.qk_scale if index < 2 else 1.0)).astype(jnp.bfloat16)
                           for index, key in enumerate(keys))
            jax.block_until_ready(arrays)
            for degree in args.degrees:
                baseline = {"config": candidate_config(BASELINE), "status": "running", "is_baseline": True}
                degree_entry = {"degree": degree, "baseline": baseline, "candidates": [baseline]}
                report["results"].append(degree_entry)
                write_report(args.output, report)
                baseline_outputs = measure_candidate(BASELINE, degree, arrays, baseline, args, report)
                reference = host_snapshot(baseline_outputs)
                report["active_stage"]["operation"] = "parity"
                baseline["parity"], accepted = compare_outputs(baseline_outputs, reference)
                del baseline_outputs
                if not accepted:
                    baseline["status"] = "numerical_failure"
                    raise FloatingPointError(f"Nonfinite baseline output/gradient at degree {degree}")
                baseline["status"] = "accepted"
                update_best(degree_entry)
                write_report(args.output, report)
                for candidate in candidates[1:]:
                    entry = {"config": candidate_config(candidate), "status": "running", "is_baseline": False}
                    degree_entry["candidates"].append(entry)
                    outputs = None
                    write_report(args.output, report)
                    try:
                        outputs = measure_candidate(candidate, degree, arrays, entry, args, report)
                        report["active_stage"]["operation"] = "parity"
                        entry["parity"], accepted = compare_outputs(outputs, reference)
                        entry["status"] = "accepted" if accepted else "numerical_failure"
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        resource = any(token in message.lower() for token in (
                            "outofresources", "out of resource", "resource_exhausted", "resource exhausted",
                            "shared memory size", "too many resources", "exceeds maximum",
                        ))
                        entry["status"] = "unsupported_resources" if resource else "error"
                        entry["error"] = message
                        entry["traceback"] = traceback.format_exc()
                        entry["failed_stage"] = dict(report.get("active_stage", {}))
                        print(f"degree={degree} candidate={entry['config']['id']} {entry['status']}: {message}", flush=True)
                    finally:
                        del outputs
                    update_best(degree_entry)
                    print(f"degree={degree} candidate={entry['config']['id']} status={entry['status']} best={degree_entry['best']}", flush=True)
                    write_report(args.output, report)
                del reference
        report["status"] = "complete"
        report["candidate_status_counts"] = dict(Counter(entry["status"] for degree in report["results"] for entry in degree["candidates"]))
        report.pop("active_stage", None)
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        write_report(args.output, report)
        raise
    finally:
        set_candidate(original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 16])
    parser.add_argument("--tiles", type=tile_argument, nargs="+", default=list(DEFAULT_TILES))
    parser.add_argument("--warps", type=int, choices=(4, 8), nargs="+", default=[4, 8])
    parser.add_argument("--stages", type=int, choices=(1, 2, 3, 4), nargs="+", default=[2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--qk-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.batch_size, args.sequence_length, args.num_heads, args.head_dim)) or args.head_dim > 64:
        parser.error("Shape dimensions must be positive; head-dim <=64 is required so global tiles apply")
    if any(degree < 0 or degree % 2 for degree in args.degrees):
        parser.error("Degrees must be nonnegative even integers")
    if args.device_index < 0 or args.warmup < 3 or args.iterations < 10:
        parser.error("Require device-index >=0, at least 3 warmups, and at least 10 timed iterations")
    if not math.isfinite(args.qk_scale) or args.qk_scale <= 0:
        parser.error("qk-scale must be positive and finite")
    for values in (args.degrees, args.tiles, args.warps, args.stages):
        if len(values) != len(set(values)):
            parser.error("Degrees, tiles, warps, and stages must not contain duplicates")
    benchmark(args)


if __name__ == "__main__":
    main()
