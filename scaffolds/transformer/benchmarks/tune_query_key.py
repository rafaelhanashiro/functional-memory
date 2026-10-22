"""Tune the existing one-pass query-key polynomial kernel on one allocated GPU.

  python -m scaffolds.transformer.benchmarks.tune_query_key \
    --output scaffolds/transformer/validation/runs/query-key-tuning/tuning.json

Only tile geometry, warps, and stages change. The production FP32 polynomial,
BF16x3 mixed products, and custom VJP remain unchanged. All five input gradients
are checked against independent host snapshots of the production geometry.
This is a tile-invariance benchmark, not an independent correctness proof or a
complete Transformer benchmark. Timings exclude compilation and comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import gc
import hashlib
import math
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, time_program, write_report


BASELINE = (64, 64, 4, 2)
DEFAULT_CANDIDATES = (
    BASELINE, (32, 64, 4, 2), (64, 32, 4, 2), (32, 32, 4, 2),
    (128, 32, 4, 2), (64, 64, 8, 2), (64, 128, 4, 2),
)
INPUT_NAMES = ("q", "k", "v", "coefficients", "factors")
TOLERANCES = {"forward": 0.006, "backward_forward": 0.006,
              "gradient_q": 0.025, "gradient_k": 0.025, "gradient_v": 0.015,
              "gradient_coefficients": 0.025, "gradient_factors": 0.025}
ORIGINAL_CONFIG = (kernel._tile_sizes, getattr(kernel, "_backward_tile_sizes", None),
                   kernel.NUM_WARPS, kernel.NUM_STAGES)


def candidate_argument(value):
    try:
        candidate = tuple(int(part) for part in value.lower().split("x"))
        q, k, warps, stages = candidate
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected QxKxWARPSxSTAGES, e.g. 32x64x4x2") from error
    if any(size < 16 or size > 256 or size & (size - 1) for size in (q, k)):
        raise argparse.ArgumentTypeError("Tile sizes must be powers of two between 16 and 256")
    if warps not in (4, 8) or stages not in (1, 2, 3, 4):
        raise argparse.ArgumentTypeError("Require 4 or 8 warps, and 1 through 4 stages")
    return candidate


def candidate_config(candidate):
    if candidate is None:
        return {"id": "production", "uses_kernel_defaults": True}
    q, k, warps, stages = candidate
    return {"id": f"q{q}_k{k}_w{warps}_s{stages}", "block_q": q, "block_k": k,
            "num_warps": warps, "num_stages": stages}


def set_candidate(candidate):
    # Globals are captured during tracing, including through custom_vjp.
    jax.clear_caches()
    gc.collect()
    if candidate is None:
        kernel._tile_sizes, backward, kernel.NUM_WARPS, kernel.NUM_STAGES = ORIGINAL_CONFIG
        if backward is not None:
            kernel._backward_tile_sizes = backward
        return
    q, k, warps, stages = candidate
    kernel._tile_sizes = lambda dq, dv: (q, k)
    if ORIGINAL_CONFIG[1] is not None:
        kernel._backward_tile_sizes = lambda dq, dv, **kwargs: (q, k)
    kernel.NUM_WARPS, kernel.NUM_STAGES = warps, stages


def host_snapshot(value):
    def copy(array):
        result = np.array(jax.device_get(array), copy=True)
        result.setflags(write=False)
        return result
    return jax.tree.map(copy, value)


def inputs(args, degree):
    """Synthetic nonconstant full-factorial coefficients and relative key factors."""
    shape = (args.batch_size, args.sequence_length, args.num_heads, args.head_dim)
    coeff_shape = (*shape[:3], degree + 1)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 8)
    q, k, v, cotangent = [jax.random.normal(key, shape, jnp.float32) for key in keys[:4]]
    # QK-normalized score = 8*cosine for width64. Normalize before casting,
    # matching the finite-precision operands presented to the attention kernel.
    norm = lambda x: x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True))
    operand_scale = math.sqrt(args.score_scale * math.sqrt(args.head_dim))
    q, k = norm(q) * operand_scale, norm(k) * operand_scale
    factorials = jnp.asarray([math.factorial(r) for r in range(degree + 1)], jnp.float32)
    beta = 1.0 + 0.2 * jnp.tanh(jax.random.normal(keys[4], (args.num_heads, degree + 1)))
    alpha = 0.4 * jnp.tanh(jax.random.normal(keys[5], (args.num_heads, degree + 1)))
    coefficients = (beta + alpha * jnp.tanh(jax.random.normal(keys[6], coeff_shape))) / factorials
    factors = 1.0 + 0.3 * jnp.tanh(jax.random.normal(keys[7], coeff_shape))
    return jax.block_until_ready((q.astype(jnp.bfloat16), k.astype(jnp.bfloat16),
                                 v.astype(jnp.bfloat16), coefficients, factors,
                                 cotangent.astype(jnp.bfloat16)))


def forward(q, k, v, coefficients, factors):
    return kernel.fused_query_key_polynomial_attention(q, k, v, coefficients, factors)


def objective(q, k, v, coefficients, factors, cotangent):
    output = forward(q, k, v, coefficients, factors)
    value = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
    # Returning the output prevents forward elimination in the custom VJP.
    return value, output


def measure(candidate, arrays, entry, report, args):
    set_candidate(candidate)
    programs = (("forward", forward, arrays[:5]),
                ("forward_backward", jax.value_and_grad(objective, argnums=(0, 1, 2, 3, 4),
                                                        has_aux=True), arrays))
    results = {}
    for name, program, arguments in programs:
        report["active_stage"]["operation"] = name
        write_report(args.output, report)
        print(f"degree={report['active_stage']['degree']} candidate={entry['config']['id']} "
              f"operation={name}: compiling and timing", flush=True)
        timing, result = time_program(program, arguments, warmup=args.warmup, iterations=args.iterations)
        entry[name] = timing
        results[name] = result
        print(f"degree={report['active_stage']['degree']} candidate={entry['config']['id']} "
              f"operation={name} median_ms={timing['median_ms']:.6f}", flush=True)
        write_report(args.output, report)
    return results


@jax.jit
def comparison_arrays(actual, expected):
    """Scaled reductions avoid overflowing the squared norm of high-order gradients."""
    actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(expected)), jnp.finfo(jnp.float32).tiny)
    delta, ref = (actual - expected) / scale, expected / scale

    def reductions(axes):
        error_sq = jnp.sum(jnp.square(delta), axis=axes)
        reference_sq = jnp.sum(jnp.square(ref), axis=axes)
        max_error = jnp.max(jnp.abs(delta), axis=axes)
        max_ref = jnp.max(jnp.abs(ref), axis=axes)
        return error_sq, reference_sq, max_error, max_ref

    return {"all_finite": jnp.all(jnp.isfinite(actual)) & jnp.all(jnp.isfinite(expected)),
            "scale": scale,
            "global": reductions(None), "batch_head": reductions((1, 3)),
            "degree": reductions((0, 1, 2)), "head_degree": reductions((0, 1))}


def reduction_summary(values, tolerance):
    error_sq, reference_sq, max_error, max_reference = [np.asarray(v, np.float64) for v in values]
    finite = (np.isfinite(error_sq) & np.isfinite(reference_sq)
              & np.isfinite(max_error) & np.isfinite(max_reference))
    relative = np.sqrt(np.divide(error_sq, reference_sq, out=np.zeros_like(error_sq), where=reference_sq > 0))
    zero_mismatch = (reference_sq == 0) & (error_sq != 0)
    passed = finite & ~zero_mismatch & (error_sq <= tolerance ** 2 * reference_sq)
    passed &= max_error <= 3 * tolerance * max_reference
    maximum = float(np.max(relative))
    return {"accepted": bool(np.all(passed)), "max_relative_l2_error": maximum if math.isfinite(maximum) else None,
            "nonzero_error_with_zero_reference": bool(np.any(zero_mismatch)),
            "failed_count": int(np.sum(~passed)), "count": int(passed.size)}


def compare_array(actual, expected, tolerance, by_degree):
    values = jax.device_get(comparison_arrays(actual, jax.device_put(expected)))
    result = {"all_finite": bool(values["all_finite"]), "relative_l2_tolerance": tolerance,
              "relative_max_tolerance": 3 * tolerance,
              "global": reduction_summary(values["global"], tolerance),
              "per_batch_head": reduction_summary(values["batch_head"], tolerance)}
    if by_degree:
        result["per_head_degree"] = reduction_summary(values["head_degree"], tolerance)
        result["per_degree"] = [dict(degree=degree, **reduction_summary(tuple(v[degree] for v in values["degree"]), tolerance))
                                for degree in range(actual.shape[-1])]
    result["accepted"] = result["all_finite"] and result["global"]["accepted"] and result["per_batch_head"]["accepted"]
    if by_degree:
        result["accepted"] &= result["per_head_degree"]["accepted"] and all(v["accepted"] for v in result["per_degree"])
    return result


def compare(outputs, reference):
    (value, actual_output), gradients = outputs["forward_backward"]
    (expected_value, expected_output), expected_gradients = reference["forward_backward"]
    pairs = [("forward", outputs["forward"], reference["forward"]),
             ("backward_forward", actual_output, expected_output)]
    pairs += [(f"gradient_{name}", actual, expected)
              for name, actual, expected in zip(INPUT_NAMES, gradients, expected_gradients, strict=True)]
    parity = {name: compare_array(actual, expected, TOLERANCES[name],
                                  name in ("gradient_coefficients", "gradient_factors"))
              for name, actual, expected in pairs}
    value, expected_value = float(jax.device_get(value)), float(expected_value)
    parity["objective"] = {"accepted": math.isfinite(value) and math.isfinite(expected_value),
                           "actual": value if math.isfinite(value) else None,
                           "reference": expected_value if math.isfinite(expected_value) else None,
                           "absolute_error": abs(value - expected_value) if math.isfinite(value - expected_value) else None,
                           "policy": "finite only; scalar objective can cancel"}
    return parity, all(item["accepted"] for item in parity.values())


def update_best(entry):
    accepted = [candidate for candidate in entry["candidates"] if candidate["status"] == "accepted"]
    entry["best"] = {}
    for operation in ("forward", "forward_backward"):
        if accepted:
            best = min(accepted, key=lambda c: c[operation]["median_ms"])
            entry["best"][operation] = {"candidate": best["config"], "median_ms": best[operation]["median_ms"],
                                        "speedup_over_baseline": entry["baseline"][operation]["median_ms"] / best[operation]["median_ms"]}


def benchmark(args):
    devices = jax.local_devices(backend="gpu")
    if args.device_index >= len(devices):
        raise ValueError(f"Requested GPU {args.device_index}; only {len(devices)} visible")
    device = devices[args.device_index]
    requested = [BASELINE] + [c for c in args.candidates if c != BASELINE]
    if args.include_production:
        requested.append(None)
    report = {"format": "query-key-polynomial-tile-tuning", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "provenance": provenance(device),
              "configuration": {"shape": [args.batch_size, args.sequence_length, args.num_heads, args.head_dim],
                                "dtype": "bfloat16", "degrees": args.degrees, "seed": args.seed,
                                "score_scale": args.score_scale, "warmup": args.warmup, "iterations": args.iterations,
                                "candidates": [candidate_config(c) for c in requested],
                                "coefficient_formula": "(beta + alpha*tanh(z))/r!: beta=1+.2*tanh(N), alpha=.4*tanh(N), z=N",
                                "key_factor_formula": "1+.3*tanh(N), independently varying by batch/token/head/degree",
                                "score_formula": "score_scale*cosine(q,k), normalized before BF16 conversion",
                                "objective": "sum(float32(output)*float32(dynamic_cotangent)); output returned as aux; gradients all five operands",
                                "precision": "unchanged production FP32 polynomial and BF16x3 mixed products",
                                "parity": "global and every batch/head; coefficient/factor gradients also each degree and head/degree; limits relative L2 and 3x relative max",
                                "reference": "production 64x64/4warp/2stage, independent immutable host copies; not an independent correctness oracle",
                                "timing": "synchronized resident compiled calls; no compilation, copies, or comparisons",
                                "max_minutes": args.max_minutes, "budget_policy": "checked between candidates; an ongoing compilation is not interrupted"},
              "results": []}
    source = Path(__file__).resolve()
    report["provenance"]["source_sha256"][str(source.relative_to(source.parents[3]))] = hashlib.sha256(source.read_bytes()).hexdigest()
    started = time.monotonic()
    write_report(args.output, report)
    budget_exhausted = False
    try:
        with jax.default_device(device):
            for degree in args.degrees:
                arrays = inputs(args, degree)
                entry = {"degree": degree, "candidates": []}
                report["results"].append(entry)
                reference = None
                for candidate in requested:
                    if time.monotonic() - started >= args.max_minutes * 60:
                        budget_exhausted = True
                        break
                    candidate_entry = {"config": candidate_config(candidate), "status": "running",
                                       "is_baseline": candidate == BASELINE}
                    entry["candidates"].append(candidate_entry)
                    if candidate == BASELINE:
                        entry["baseline"] = candidate_entry
                    report["active_stage"] = {"degree": degree, "candidate": candidate_entry["config"]["id"]}
                    outputs = None
                    try:
                        outputs = measure(candidate, arrays, candidate_entry, report, args)
                        report["active_stage"]["operation"] = "parity"
                        if candidate == BASELINE:
                            reference = host_snapshot(outputs)
                        candidate_entry["parity"], accepted = compare(outputs, reference)
                        candidate_entry["status"] = "accepted" if accepted else "numerical_failure"
                        if candidate == BASELINE and not accepted:
                            raise FloatingPointError("The production-geometry baseline is nonfinite")
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        resource = any(s in message.lower() for s in ("outofresources", "out of resource", "resource_exhausted",
                                                                     "resource exhausted", "shared memory", "too many resources"))
                        candidate_entry.update(status="unsupported_resources" if resource else "error", error=message,
                                               traceback=traceback.format_exc(), failed_stage=dict(report["active_stage"]))
                        print(f"degree={degree} candidate={candidate_entry['config']['id']} failed: {message}", flush=True)
                        if candidate == BASELINE:
                            raise
                    finally:
                        del outputs
                    update_best(entry)
                    report["elapsed_seconds"] = time.monotonic() - started
                    write_report(args.output, report)
                    print(f"degree={degree} candidate={candidate_entry['config']['id']} status={candidate_entry['status']} "
                          f"best={entry['best']}", flush=True)
                del arrays, reference
                if budget_exhausted:
                    break
        report["status"] = "budget_exhausted" if budget_exhausted else "complete"
        report["candidate_status_counts"] = dict(Counter(c["status"] for e in report["results"] for c in e["candidates"]))
        report.pop("active_stage", None)
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(args.output, report)
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_report(args.output, report)
        raise
    finally:
        set_candidate(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--candidates", type=candidate_argument, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--include-production", action="store_true",
                        help="Also benchmark the current production defaults, without geometry overrides")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--score-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=18.0)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.num_heads, args.head_dim) <= 0 or args.head_dim > 64:
        parser.error("Positive shapes and head-dim <=64 required")
    if any(not 1 <= degree <= 16 for degree in args.degrees):
        parser.error("Degrees must be between 1 and 16")
    if args.device_index < 0 or args.warmup < 3 or args.iterations < 10:
        parser.error("Require nonnegative device index, >=3 warmups, >=10 timed iterations")
    if not all(math.isfinite(v) and v > 0 for v in (args.score_scale, args.max_minutes)):
        parser.error("Score scale and time budget must be finite and positive")
    if len(set(args.degrees)) != len(args.degrees) or len(set(args.candidates)) != len(args.candidates):
        parser.error("Degrees and candidates must not contain duplicates")
    benchmark(args)


if __name__ == "__main__":
    main()
