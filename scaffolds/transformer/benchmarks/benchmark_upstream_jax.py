"""Compare our attention kernels with stock JAX/cuDNN causal softmax on one GPU.

This times attention only. Inputs, normalization, gate/coefficient generation,
compilation, transfers, and finite checks are outside the measured calls. Forward
plus backward returns the output and gradients of every floating-point operand,
using an explicit output cotangent rather than timing a scalar loss reduction.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
import hashlib
import math
from pathlib import Path
import re
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.benchmarks.benchmark_attention import compiled_memory, write_report
from scaffolds.transformer.benchmarks.benchmark_training_throughput import _execution_provenance
from scaffolds.transformer.benchmarks import upstream_inputs


def _value_and_vjp(forward, *arguments):
    operands, cotangent = arguments[:-1], arguments[-1]
    output, pullback = jax.vjp(forward, *operands)
    return output, pullback(cotangent)


def _summary(samples_ms, tokens):
    values = np.asarray(samples_ms, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Timing samples must be finite and positive")
    return {
        "timed_calls": len(values), "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)), "p10_ms": float(np.quantile(values, .1)),
        "p90_ms": float(np.quantile(values, .9)), "min_ms": float(values.min()),
        "max_ms": float(values.max()), "tokens_per_second": float(tokens * 1000 / values.mean()),
    }


def _dispatch(compiled, *, cudnn, backward):
    """Retain call targets, not the potentially large generated HLO/assembly."""
    hlo = compiled.as_text()
    cudnn_targets = sorted(set(re.findall(r"__cudnn\$[A-Za-z0-9_]+", hlo)))
    if cudnn and not any("fmha" in name and "Backward" not in name for name in cudnn_targets):
        raise RuntimeError("Compiled softmax did not contain a cuDNN attention forward call")
    if cudnn and backward and not any("fmha" in name and "Backward" in name for name in cudnn_targets):
        raise RuntimeError("Compiled softmax VJP did not contain a cuDNN attention backward call")
    return {
        "source": "compiled.as_text", "cudnn_targets": cudnn_targets,
        "custom_call_targets": sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo))),
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
    }


def _check_result(result, operands, *, backward):
    if backward:
        output, gradients = result
        if len(gradients) != len(operands):
            raise RuntimeError("Benchmark did not return every input gradient")
        if any(gradient.shape != operand.shape for gradient, operand in zip(gradients, operands, strict=True)):
            raise RuntimeError("Gradient shape differs from its input")
    else:
        output, gradients = result, ()
    arrays = (output, *gradients)
    finite = [bool(value) for value in jax.device_get([jnp.all(jnp.isfinite(x)) for x in arrays])]
    return {
        "all_finite": all(finite), "finite_by_output_and_gradient": finite,
        "gradient_count": len(gradients),
        "output_and_gradient_shapes": [list(x.shape) for x in arrays],
        "output_and_gradient_dtypes": [str(x.dtype) for x in arrays],
    }


def benchmark(args):
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("This benchmark requires exactly one GPU")
    source = _execution_provenance(devices[0], args.experiment)
    root = Path(__file__).resolve().parents[3]
    for path in (Path(__file__), Path(upstream_inputs.__file__)):
        source["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    source["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shape = (args.batch, args.length, args.heads, args.width)
    report = {
        "format": "upstream-jax-attention-kernel-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": source, "configuration": vars(args) | {"output": str(args.output)},
        "shape": list(shape), "layout": "batch,time,heads,channels", "results": {},
        "timing": "Compiled call plus host dispatch, synchronized after every call; each program compiles once.",
        "scope": "Attention forward or forward plus all input gradients; no scalar loss reduction.",
        "exclusions": ["input generation and host/device transfer", "Q/K normalization and score scaling",
                       "gate and coefficient generation, including sigmoid/tanh", "compilation and warmup",
                       "finite checks", "projections, convolution, output normalization, and other model layers"],
        "comparison": "Different attention mathematics; throughput comparison, not numerical equivalence.",
        "memory": "Compiler buffer estimates and allocator statistics, not isolated per-kernel measured peaks.",
    }
    write_report(args.output, report)
    try:
        host = upstream_inputs.kernel_inputs(*shape, seed=args.seed)
        report["input_sha256"] = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in host.items()}
        operands = {name: jnp.asarray(value, jnp.float32 if name == "g" else jnp.bfloat16)
                    for name, value in host.items()}
        scaled_q = jnp.asarray(host["q"] * args.width, jnp.bfloat16)
        signed_beta = jnp.asarray(2 * host["beta"] - 1, jnp.bfloat16)
        del host
        coefficients = jnp.broadcast_to(
            jnp.asarray([1., 1., .5, 1 / 6], jnp.float32), (*shape[:-1], 4))
        q, k, v, g, beta, erase, cotangent = (operands[name] for name in ("q", "k", "v", "g", "beta", "erase", "do"))
        jax.block_until_ready((operands, scaled_q, signed_beta, coefficients))
        report["input_dtypes"] = {name: str(value.dtype) for name, value in operands.items()}
        report["input_distribution"] = {
            "common": "upstream_inputs.kernel_inputs; unit Q/K, Gaussian V/cotangent, sigmoid beta/erase, negative log decay",
            "gdn": "unit Q/K; scale=1/sqrt(width); signed write is (2*FP32 beta-1) cast to BF16",
            "softmax_and_mosaic": "Q multiplied by width in FP32 before BF16 cast; scale=1/sqrt(width), giving order-one scores",
            "mosaic_coefficients": [1., 1., .5, 1 / 6],
            "mosaic_coefficients_description": "Per-query FP32 Taylor coefficients, supplied as differentiable inputs; tanh excluded",
        }
        from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
            fla_gated_delta_rule, fla_independent_gated_delta_rule,
        )
        from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention

        modes = (
            ("gdn_tied_sigmoid_jax_bridge", partial(fla_gated_delta_rule, scale=1 / math.sqrt(args.width)),
             (q, k, v, g, beta), ("q", "k", "v", "g", "beta")),
            ("gdn_independent_tanh_jax_bridge", partial(fla_independent_gated_delta_rule, scale=1 / math.sqrt(args.width)),
             (q, k, v, g, signed_beta, erase), ("q", "k", "v", "g", "beta", "erase")),
            ("query_polynomial_cubic_mosaic", query_polynomial_attention,
             (scaled_q, k, v, coefficients), ("q", "k", "v", "coefficients")),
            ("softmax_cudnn", partial(jax.nn.dot_product_attention, is_causal=True, implementation="cudnn"),
             (scaled_q, k, v), ("q", "k", "v")),
        )
        for name, forward, inputs, input_names in modes:
            entry = {"differentiated_inputs": list(input_names), "gradient_count": len(inputs), "operations": {},
                     "precision": ("BF16 public operands/output; GDN bridge promotes Q/K/V/gates/intermediates to FP32 and uses TF32x3"
                                   if name.startswith("gdn_") else
                                   "BF16 Q/K/V/output; FP32 polynomial coefficients, scores and accumulators with three-part BF16 products"
                                   if name.endswith("mosaic") else "Stock JAX/cuDNN BF16 causal flash attention")}
            report["results"][name] = entry
            for operation, function, arguments in (
                ("forward", forward, inputs),
                ("forward_backward", partial(_value_and_vjp, forward), (*inputs, cotangent)),
            ):
                backward = operation == "forward_backward"
                report["active_stage"] = {"implementation": name, "operation": operation, "stage": "compiling"}
                write_report(args.output, report)
                started = time.perf_counter()
                compiled = jax.jit(function).lower(*arguments).compile()
                timing = {"compilation_seconds": time.perf_counter() - started,
                          "compiled_memory": compiled_memory(compiled),
                          "dispatch": _dispatch(compiled, cudnn=name == "softmax_cudnn", backward=backward),
                          "rounds": []}
                entry["operations"][operation] = timing
                for round_index in range(args.rounds):
                    report["active_stage"] = {"implementation": name, "operation": operation,
                                              "round": round_index + 1, "stage": "timing"}
                    write_report(args.output, report)
                    for _ in range(args.warmup):
                        jax.block_until_ready(compiled(*arguments))
                    samples = []
                    for _ in range(args.iterations):
                        started = time.perf_counter()
                        result = compiled(*arguments)
                        jax.block_until_ready(result)
                        samples.append((time.perf_counter() - started) * 1000)
                    checks = _check_result(result, inputs, backward=backward)
                    round_result = {"round": round_index + 1, "warmup_calls": args.warmup,
                                    "samples_ms": samples, "summary": _summary(samples, args.batch * args.length),
                                    "checks": checks}
                    timing["rounds"].append(round_result)
                    write_report(args.output, report)
                    if not checks["all_finite"]:
                        raise FloatingPointError(f"Nonfinite {name} {operation} result")
                    print(f"{name} {operation} round={round_index + 1}: "
                          f"{round_result['summary']['median_ms']:.3f} ms", flush=True)
                    del result
                timing["summary"] = _summary(
                    [sample for measured_round in timing["rounds"] for sample in measured_round["samples_ms"]],
                    args.batch * args.length)
                del compiled
                write_report(args.output, report)
        report.update(status="complete", gpu_memory_stats=devices[0].memory_stats())
        report.pop("active_stage", None)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--experiment", default="attention_scaling")
    args = parser.parse_args()
    if min(args.batch, args.length, args.heads, args.width, args.warmup, args.iterations, args.rounds) < 1:
        parser.error("Dimensions and timing counts must be positive")
    if args.width != 64:
        parser.error("This matched Q/K/V-width comparison requires width 64 for Mosaic")
    benchmark(args)


if __name__ == "__main__":
    main()
