"""Diagnose full-batch versus batch-split GDN arithmetic without throughput timing.

Compare the production BF16 fixture and the same quantized values promoted to
FP32. The FP32 case removes final BF16 output/gradient rounding; it does not
change the fixture values or rescale the explicit output cotangent.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends import polynomial_gdn
from scaffolds.transformer.benchmarks import benchmark_poly_gdn_tuning as benchmark
from scaffolds.transformer.benchmarks import upstream_inputs
from scaffolds.transformer.benchmarks.benchmark_training_throughput import _execution_provenance


@jax.jit
def distribution_metrics(actual, expected):
    """Count differing elements and expose their locations along the batch axis."""
    a, e = actual.astype(jnp.float32), expected.astype(jnp.float32)
    difference = a - e
    mismatch = a != e
    axes = tuple(range(1, a.ndim))
    scale = jnp.maximum(jnp.max(jnp.abs(e)), jnp.finfo(jnp.float32).tiny)
    expected_rms = jnp.sqrt(jnp.mean(jnp.square(e / scale), axis=axes))
    difference_rms = jnp.sqrt(jnp.mean(jnp.square(difference / scale), axis=axes))
    result = {
        "elements": jnp.asarray(a.size),
        "mismatch_count": jnp.count_nonzero(mismatch),
        "mismatch_count_by_batch": jnp.count_nonzero(mismatch, axis=axes),
        "relative_rms_by_batch": difference_rms / jnp.maximum(expected_rms, jnp.finfo(jnp.float32).tiny),
        "max_abs_by_batch": jnp.max(jnp.abs(difference), axis=axes),
        "sign_change_count": jnp.count_nonzero(mismatch & (jnp.signbit(a) != jnp.signbit(e))),
    }
    if actual.dtype == jnp.bfloat16 and expected.dtype == jnp.bfloat16:
        def ordered(value):
            bits = jax.lax.bitcast_convert_type(value, jnp.uint16).astype(jnp.int32)
            magnitude = bits & 0x7fff
            return jnp.where((bits & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + magnitude)
        distance = jnp.abs(ordered(actual) - ordered(expected))
        multiple = distance > 1
        result.update(
            max_bf16_ulp=jnp.max(distance),
            one_ulp_count=jnp.count_nonzero(distance == 1),
            multiple_ulp_count=jnp.count_nonzero(multiple),
            multiple_ulp_reference_abs_max=jnp.max(jnp.where(multiple, jnp.abs(e), 0)),
            multiple_ulp_max_abs_difference=jnp.max(jnp.where(multiple, jnp.abs(difference), 0)),
        )
    return result


def diagnostic_case(runner, name, operands, cotangent):
    args = runner.args
    case = {"status": "running", "input_dtypes": [str(value.dtype) for value in operands],
            "cotangent_dtype": str(cotangent.dtype), "compilations": {}, "chunk_sizes": [],
            "chunk_call_seconds": []}
    runner.report["cases"][name] = case
    runner.active(case=name, phase="preparing")
    forward = partial(polynomial_gdn.polynomial_gated_delta_rule, scale=args.key_dim ** -.5,
                      precision="tf32x3", num_warps=2, min_ctas_per_sm=0)
    function = partial(benchmark.value_and_vjp, forward)
    executable, full = runner.compile("full_forward_backward", function, (*operands, cotangent), case["compilations"])
    del executable
    executables, host_parts = {}, []
    try:
        for start in range(0, args.batch, args.reference_batch_size):
            stop = min(start + args.reference_batch_size, args.batch)
            size = stop - start
            arguments = tuple(value[start:stop] for value in (*operands, cotangent))
            runner.active(effective_batch=size, phase="split_reference")
            operation = f"split_forward_backward_b{size}"
            if size not in executables:
                executable, result = runner.compile(operation, function, arguments, case["compilations"])
                executables[size] = executable
                seconds = case["compilations"][operation]["first_call_seconds"]
            else:
                started = time.perf_counter()
                result = jax.block_until_ready(executables[size](*arguments))
                seconds = time.perf_counter() - started
            host_parts.append(jax.device_get(result))
            case["chunk_sizes"].append(size)
            case["chunk_call_seconds"].append(seconds)
            del result, arguments
            runner.save()
        split = jax.tree.map(lambda *parts: jnp.asarray(np.concatenate(parts, axis=0)), *host_parts)
        host_parts.clear()
        runner.active(phase="comparing")
        case["strict_accuracy"] = benchmark.compare_result(full, split, backward=True, rms_limit=1e-5, max_limit=1e-5)
        case["distribution"] = {}
        for channel, actual, expected in zip(("output", *benchmark.INPUT_NAMES),
                                            (full[0], *full[1]), (split[0], *split[1]), strict=True):
            raw = jax.device_get(distribution_metrics(actual, expected))
            case["distribution"][channel] = {key: np.asarray(value).tolist() for key, value in raw.items()}
            runner.save()
        case["status"] = "complete"
        print(json.dumps({"case": name, "strict_passed": case["strict_accuracy"]["passed"],
                          "all_finite": case["strict_accuracy"]["all_finite"]}), flush=True)
        del full, split
        return case
    except Exception as error:
        case.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        executables.clear()
        host_parts.clear()
        runner.context.pop("effective_batch", None)
        runner.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--reference-batch-size", type=int, default=16)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--experiment", default="attention_scaling")
    args = parser.parse_args()
    if min(args.batch, args.reference_batch_size, args.length, args.heads, args.key_dim, args.value_dim) < 1:
        parser.error("Dimensions must be positive")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("Run with one allocated GPU")
    provenance = _execution_provenance(devices[0], args.experiment)
    provenance["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"format": "polynomial-gdn-batch-split-diagnostic", "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "configuration": vars(args) | {"output": str(args.output)},
              "provenance": provenance, "cases": {},
              "operator": {"degree": 3, "precision": "tf32x3", "num_warps": 2, "min_ctas_per_sm": 0},
              "scope": "Numerical diagnostic only; compilation and first-call durations are not throughput measurements."}
    runner = benchmark.Runner(args, report)
    runner.save()
    started = time.perf_counter()
    try:
        host = upstream_inputs.kernel_inputs(args.batch, args.length, args.heads, args.key_dim,
                                             args.seed, value_width=args.value_dim)
        coefficients = upstream_inputs.polynomial_coefficients(args.batch, args.length, args.heads, 3, args.seed)
        report["input_sha256"] = {name: benchmark.array_digest(value) for name, value in host.items()}
        report["polynomial_coefficients_sha256"] = benchmark.array_digest(coefficients)
        operands = tuple(jnp.asarray(coefficients if name == "beta" else host[name],
                                    jnp.float32 if name == "g" else jnp.bfloat16) for name in benchmark.INPUT_NAMES)
        cotangent = jnp.asarray(host["do"], jnp.bfloat16)
        del host, coefficients
        bf16 = diagnostic_case(runner, "production_bf16", operands, cotangent)
        promoted = tuple(value.astype(jnp.float32) for value in operands)
        promoted_cotangent = cotangent.astype(jnp.float32)
        report["promotion_preserves_values_exactly"] = all(bool(value) for value in jax.device_get([
            jnp.all(original.astype(jnp.float32) == value)
            for original, value in zip((*operands, cotangent), (*promoted, promoted_cotangent), strict=True)]))
        del operands, cotangent
        fp32 = diagnostic_case(runner, "same_quantized_values_fp32", promoted, promoted_cotangent)
        report["fp32_strict_passed"] = fp32["strict_accuracy"]["passed"]
        report["all_finite"] = bf16["strict_accuracy"]["all_finite"] and fp32["strict_accuracy"]["all_finite"]
        if not report["promotion_preserves_values_exactly"] or not report["all_finite"] or not report["fp32_strict_passed"]:
            raise FloatingPointError("Promoted FP32 split reference or finite/value-preservation check failed")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(finished_at_utc=datetime.now(UTC).isoformat(), wall_seconds=time.perf_counter() - started)
        report.pop("active_stage", None)
        runner.save()


if __name__ == "__main__":
    main()
