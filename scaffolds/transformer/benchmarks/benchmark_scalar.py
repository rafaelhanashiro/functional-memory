"""Same-GPU scalar attention comparison at training shapes, including all VJPs.

Each invocation owns one function/shape/mode. Matched implementations use the
same resident operands and cotangent. Warmup/compilation and profiles are excluded
from synchronized timing samples. Failed cases remain visible in the JSON.
"""
import argparse
from functools import partial
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.basis import ExpProjection, sine, projection_from_config
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.backends.triton_scalar import (
    Tuning, basis_kernel, exponential_kernel, polynomial_kernel, scalar_attention, uniform_rbf_kernel,
)
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.benchmarks.benchmark_attention import write_report
from scaffolds.transformer.benchmarks.benchmark_training_throughput import _execution_provenance, digest
from scaffolds.transformer.configs import load_config


RUNS = {3: "r10b2cb22687c", 5: "r1b486c356e1a", 9: "r27f0d0f59eb1", 20: "r873f76c0a542"}


def case(name):
    recipe = None
    if name.startswith("rbf"):
        path = Path("experiments/attention_scaling/runs") / RUNS[int(name[3:])] / "config.yaml"
        cfg = load_config(path)
        projection = projection_from_config(cfg.model.attention_basis)
        recipe = {"path": str(path), "sha256": digest(path)}
        return basis_kernel(projection), dict(score_kernel=projection), recipe
    if name == "sine":
        projection = ExpProjection((sine(1.),), (1.,), 0., 1., 16, 32, 1e-12, {})
        return basis_kernel(projection), dict(score_kernel=projection), recipe
    if name == "exp":
        return exponential_kernel(), dict(attention_degree=None), recipe
    degree = int(name[4:])
    return polynomial_kernel(tuple(1/math.factorial(i) for i in range(degree+1))), dict(attention_degree=degree), recipe


def operations(name, mode, precise=False, tune=False):
    kernel, reference_args, recipe = case(name)
    normalized = mode == "sum"
    def finish(raw, gain):
        return scale_attention_output(raw, "head_rms_norm", output_norm_scale=gain) if mode == "rms" else raw
    result = {"blocked": lambda q, k, v, g: finish(blocked_dot_product_attention(
        q, k, v, attention_normalization=normalized, **reference_args), g)}
    result["triton_separate"] = lambda q, k, v, g: finish(scalar_attention(
        q, k, v, kernel=kernel, attention_normalization=normalized), g)
    if mode == "rms":
        result["triton_fused"] = lambda q, k, v, g: scalar_attention(
            q, k, v, kernel=kernel, output_norm_scale=g, streaming=False)
        result["triton_streamed"] = lambda q, k, v, g: scalar_attention(
            q, k, v, kernel=kernel, output_norm_scale=g, streaming=True)
    if precise:
        result["triton_precise"] = lambda q, k, v, g: scalar_attention(
            q, k, v, kernel=kernel, attention_normalization=normalized,
            output_norm_scale=g if mode == "rms" else None, precision="tf32x3")
    if name.startswith("rbf"):
        recurrence = uniform_rbf_kernel(reference_args["score_kernel"])
        result["triton_recurrence"] = lambda q, k, v, g: scalar_attention(
            q, k, v, kernel=recurrence, attention_normalization=normalized,
            output_norm_scale=g if mode == "rms" else None, streaming=False)
        result["triton_recurrence_streamed"] = lambda q, k, v, g: scalar_attention(
            q, k, v, kernel=recurrence, attention_normalization=normalized,
            output_norm_scale=g if mode == "rms" else None, streaming=True)
    if name.startswith("poly") or name == "exp":
        degree = reference_args["attention_degree"]
        if normalized:
            if degree is None:
                from scaffolds.transformer.attention.softmax import padded_dot_product_attention
                result["cudnn"] = lambda q, k, v, g: padded_dot_product_attention(q, k, v, backend="cudnn")
            else:
                from scaffolds.transformer.attention.backends.fused_taylor import fused_polynomial_attention
                from scaffolds.transformer.attention.backends.mosaic.api import taylor_attention
                result["pallas"] = lambda q, k, v, g: fused_polynomial_attention(q, k, v, attention_degree=degree)
                result["mosaic"] = lambda q, k, v, g: taylor_attention(q, k, v, attention_degree=degree)
        else:
            from scaffolds.transformer.attention.unnormalized import unnormalized_attention
            result["pallas"] = lambda q, k, v, g: finish(unnormalized_attention(
                q, k, v, attention_degree=degree, backend="fused"), g)
            if degree is not None:
                result["mosaic"] = lambda q, k, v, g: finish(unnormalized_attention(
                    q, k, v, attention_degree=degree, backend="mosaic"), g)
    if tune:
        for bm, bn, warps, stages in ((64, 32, 4, 3), (64, 64, 4, 3), (128, 32, 4, 3),
                                    (128, 64, 8, 3), (128, 64, 4, 2), (64, 64, 8, 3)):
            tuning = Tuning(forward_q=bm, forward_k=bn, forward_warps=warps, stages=stages)
            def candidate(q, k, v, g, tuning=tuning):
                return scalar_attention(q, k, v, kernel=kernel, tuning=tuning,
                    attention_normalization=normalized, output_norm_scale=g if mode == "rms" else None)
            result[f"triton_tile_{bm}_{bn}_w{warps}_s{stages}"] = candidate
    return result, recipe


def measurements(fn, inputs, count):
    for _ in range(3):
        jax.block_until_ready(fn(*inputs))
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        output = jax.block_until_ready(fn(*inputs))
        samples.append((time.perf_counter() - start)*1000)
    return {"samples_ms": samples, "median_ms": float(np.median(samples)),
            "p10_ms": float(np.quantile(samples,.1)), "p90_ms": float(np.quantile(samples,.9))}


def benchmark(args):
    device, = jax.local_devices()
    report = {"format": "scalar-attention-comparison", "version": 1, "status": "running",
              "provenance": _execution_provenance(device, "attention_scaling"),
              "function": args.kernel, "mode": args.mode,
              "shape": [args.batch, args.length, 16, 64], "dtype": "bfloat16", "seed": 1729,
              "warmup": 3, "iterations": args.iterations, "cases": [],
              "scope": "Resident synthetic attention operands; forward and forward plus Q/K/V/gain backward. Compilation and profiling excluded."}
    implementations, report["recipe"] = operations(args.kernel, args.mode, args.precise, args.tune)
    shape = tuple(report["shape"])
    rng = np.random.default_rng(report["seed"])
    q, k = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
    q = q / np.sqrt(np.sum(q*q, -1, keepdims=True)+1e-6)*64
    k = k / np.sqrt(np.sum(k*k, -1, keepdims=True)+1e-6)
    v, go = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
    inputs = (*[jnp.asarray(x, jnp.bfloat16) for x in (q, k, v)], jnp.ones((16, 64), jnp.float32))
    go = jnp.asarray(go, jnp.bfloat16)
    del q, k, v
    reference = None
    write_report(args.output, report)
    for label, operation in implementations.items():
        entry = {"implementation": label, "status": "running"}
        report["cases"].append(entry)
        write_report(args.output, report)
        try:
            forward = jax.jit(operation)
            def joint(*values):
                output, vjp = jax.vjp(operation, *values)
                return (output, *vjp(go))
            backward = jax.jit(joint)
            start = time.perf_counter()
            actual = jax.block_until_ready(backward(*inputs))
            entry["compile_and_first_seconds"] = time.perf_counter() - start
            if label == "blocked":
                reference = actual
            if reference is None:
                raise RuntimeError("Blocked reference failed; comparison is unavailable")
            errors = []
            for a, b in zip(actual, reference, strict=True):
                if not bool(jnp.all(jnp.isfinite(a))):
                    raise FloatingPointError("Nonfinite output or gradient")
                errors.append(float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32)) /
                                    jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)), 1e-12)))
            entry["relative_l2_errors_vs_blocked"] = dict(zip(("output", "dq", "dk", "dv", "dgain"), errors))
            # Also independently qualified on small inputs by test_triton_scalar.
            if max(errors) > .05:
                raise AssertionError(f"Production-shape error exceeds .05: {errors}")
            entry["forward"] = measurements(forward, inputs, args.iterations)
            entry["forward_backward"] = measurements(backward, inputs, args.iterations)
            entry["status"] = "complete"
            print(args.kernel, args.mode, args.batch, label,
                  entry["forward"]["median_ms"], entry["forward_backward"]["median_ms"],
                  "max_error", max(errors), flush=True)
            if args.profile and label in ("blocked", "triton_fused", "triton_streamed", "triton_recurrence", "mosaic"):
                trace = args.profile / label
                with jax.profiler.trace(str(trace), create_perfetto_trace=True):
                    for step in range(3):
                        with jax.profiler.StepTraceAnnotation("attention", step_num=step):
                            jax.block_until_ready(backward(*inputs))
                entry["profile"] = str(trace)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(label, entry["error"], flush=True)
        finally:
            write_report(args.output, report)
    report["status"] = "complete" if all(c["status"] == "complete" for c in report["cases"]) else "completed_with_failures"
    report["gpu_memory_stats"] = device.memory_stats()
    write_report(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, choices=("rbf3", "rbf5", "rbf9", "rbf20", "sine", "poly3", "poly8", "exp"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--mode", choices=("none", "sum", "rms"), default="rms")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--precise", action="store_true")
    parser.add_argument("--tune", action="store_true")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
