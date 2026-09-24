"""Benchmark blocked and fused Taylor attention on one allocated GPU.

Run as a module from the repository root, for example:
  python -m scaffolds.transformer.benchmarks.benchmark_attention \
    --output /path/on/nfs/attention-benchmark.json

The default shape is the local microbatch on each GPU in the four-H100 sweep.
Compilation, input generation, and parity reductions are outside timed samples.
This measures attention alone, not a Transformer update or end-to-end training.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.models.components.attention import blocked_dot_product_attention


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(device) -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    sources = [Path(__file__).resolve(), root / "scaffolds/transformer/uv.lock"]
    sources += sorted((root / "scaffolds/transformer/attention").rglob("*.py"))
    versions = {}
    for name in ("jax", "jaxlib", "flax", "numpy", "jax-cuda12-plugin", "nvidia-cudnn-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "source_commit": commit,
        "source_dirty": dirty,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources if path.is_file()},
        "package_versions": versions,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": {"id": device.id, "platform": device.platform, "kind": device.device_kind,
                   "process_index": device.process_index, "description": str(device)},
        "visible_devices": [{"id": item.id, "kind": item.device_kind} for item in jax.local_devices()],
        "environment": {name: os.environ.get(name) for name in (
            "SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
        )},
    }


def compiled_memory(compiled) -> dict:
    """Report compiler buffer estimates; these are not measured allocator peaks."""
    try:
        stats = compiled.memory_analysis()
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if stats is None:
        return {"available": False}
    fields = (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
        "alias_size_in_bytes", "generated_code_size_in_bytes",
        "host_argument_size_in_bytes", "host_output_size_in_bytes",
        "host_temp_size_in_bytes", "host_alias_size_in_bytes",
    )
    return {
        "available": True,
        "source": "compiled.memory_analysis",
        "is_measured_allocator_peak": False,
        **{name: int(getattr(stats, name)) for name in fields if getattr(stats, name, None) is not None},
    }


def time_program(function, arguments: tuple, *, warmup: int, iterations: int) -> tuple[dict, object]:
    compile_start = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    compilation_seconds = time.perf_counter() - compile_start
    memory = compiled_memory(compiled)
    for _ in range(warmup):
        jax.block_until_ready(compiled(*arguments))
    samples_ms = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples_ms.append((time.perf_counter() - start) * 1000)
    return {
        "compilation_seconds": compilation_seconds,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "median_ms": float(np.median(samples_ms)),
        "p10_ms": float(np.quantile(samples_ms, 0.1)),
        "p90_ms": float(np.quantile(samples_ms, 0.9)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "compiled_memory": memory,
    }, result


def error_norms(actual, reference) -> dict:
    """Reduce on device, copying only comparison scalars back to the host."""
    actual, reference = actual.astype(jnp.float32), reference.astype(jnp.float32)
    difference = actual - reference
    measured = jax.device_get({
        "all_finite": jnp.all(jnp.isfinite(actual)) & jnp.all(jnp.isfinite(reference)),
        "l2_error": jnp.sqrt(jnp.sum(jnp.square(difference))),
        "reference_l2": jnp.sqrt(jnp.sum(jnp.square(reference))),
        "max_abs_error": jnp.max(jnp.abs(difference)),
        "reference_max_abs": jnp.max(jnp.abs(reference)),
    })
    result = {key: (bool(value) if key == "all_finite" else float(value)) for key, value in measured.items()}
    # A zero reference gradient is expected at degree zero. Do not invent a
    # relative error when the reference norm is zero and the difference is not.
    result["relative_l2_error"] = (
        result["l2_error"] / result["reference_l2"] if result["reference_l2"] else
        (0.0 if result["l2_error"] == 0 else None)
    )
    return {key: (value if not isinstance(value, float) or math.isfinite(value) else None)
            for key, value in result.items()}


def attention_function(name: str, degree: int):
    if name == "blocked":
        implementation = blocked_dot_product_attention
    else:
        # Import lazily so the blocked baseline can run while a fused kernel
        # is being developed, and --help does not need a completed kernel.
        from scaffolds.transformer.models.components.fused_attention import fused_polynomial_attention
        implementation = fused_polynomial_attention

    def forward(q, k, v):
        return implementation(q, k, v, attention_degree=degree)

    return forward


def benchmark(args) -> dict:
    devices = jax.local_devices(backend="gpu")
    if args.device_index >= len(devices):
        raise ValueError(f"Requested local GPU {args.device_index}; only {len(devices)} are visible")
    device = devices[args.device_index]
    shape = (args.batch_size, args.sequence_length, args.num_heads, args.head_dim)
    report = {
        "format": "polynomial-attention-kernel-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(device),
        "configuration": {
            "shape": list(shape), "layout": "batch,time,heads,channels", "dtype": args.dtype,
            "seed": args.seed, "qk_scale": args.qk_scale, "degrees": args.degrees,
            "implementations": args.implementations, "warmup": args.warmup, "iterations": args.iterations,
            "input_distribution": "independent normal(0,1); Q/K scaled before dtype conversion",
            "backward_objective": "sum(float32(attention(Q,K,V)) * float32(cotangent)); gradients w.r.t. Q,K,V",
            "timing": "compiled call plus host dispatch, synchronized after every call; compilation excluded",
            "memory": "compiler buffer estimates only; no measured allocator peak",
        },
        "results": [],
    }
    write_report(args.output, report)
    try:
        with jax.default_device(device):
            dtype = getattr(jnp, args.dtype)
            keys = jax.random.split(jax.random.PRNGKey(args.seed), 4)
            arrays = tuple((jax.random.normal(key, shape, dtype=jnp.float32) *
                            (args.qk_scale if index < 2 else 1.0)).astype(dtype)
                           for index, key in enumerate(keys))
            q, k, v, cotangent = jax.block_until_ready(arrays)
            for degree in args.degrees:
                entry = {"degree": degree, "implementations": {}}
                report["results"].append(entry)
                outputs = {}
                for name in args.implementations:
                    forward = attention_function(name, degree)

                    def objective(q, k, v, cotangent):
                        return jnp.sum(forward(q, k, v).astype(jnp.float32) * cotangent.astype(jnp.float32))

                    programs = (
                        ("forward", forward, (q, k, v)),
                        ("forward_backward", jax.value_and_grad(objective, argnums=(0, 1, 2)), (q, k, v, cotangent)),
                    )
                    entry["implementations"][name] = {}
                    outputs[name] = {}
                    for operation, program, arguments in programs:
                        report["active_stage"] = {"degree": degree, "implementation": name, "operation": operation}
                        write_report(args.output, report)
                        timing, result = time_program(program, arguments, warmup=args.warmup, iterations=args.iterations)
                        entry["implementations"][name][operation] = timing
                        outputs[name][operation] = result
                        print(f"degree={degree} implementation={name} operation={operation} "
                              f"median_ms={timing['median_ms']:.4f} p90_ms={timing['p90_ms']:.4f}", flush=True)
                        write_report(args.output, report)
                if "blocked" in outputs and "fused" in outputs:
                    actual, reference = outputs["fused"], outputs["blocked"]
                    actual_value, actual_grads = actual["forward_backward"]
                    reference_value, reference_grads = reference["forward_backward"]
                    entry["parity_against_blocked"] = {
                        "forward": error_norms(actual["forward"], reference["forward"]),
                        "objective": error_norms(actual_value, reference_value),
                        **{f"gradient_{name}": error_norms(a, b)
                           for name, a, b in zip(("q", "k", "v"), actual_grads, reference_grads, strict=True)},
                    }
                    entry["speedup_blocked_over_fused"] = {
                        operation: entry["implementations"]["blocked"][operation]["median_ms"] /
                        entry["implementations"]["fused"][operation]["median_ms"]
                        for operation in ("forward", "forward_backward")
                    }
                    if not all(value["all_finite"] for value in entry["parity_against_blocked"].values()):
                        raise FloatingPointError(f"Nonfinite output or Q/K/V gradient at degree {degree}")
                    print(f"degree={degree} parity={json.dumps(entry['parity_against_blocked'], sort_keys=True)}", flush=True)
                write_report(args.output, report)
                del outputs
        report["status"] = "complete"
        report.pop("active_stage", None)
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(args.output, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--implementations", choices=("blocked", "fused"), nargs="+", default=["blocked", "fused"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--qk-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.batch_size, args.sequence_length, args.num_heads, args.head_dim)):
        parser.error("All shape dimensions must be positive")
    if any(degree < 0 or degree % 2 for degree in args.degrees):
        parser.error("Degrees must be nonnegative even integers")
    if args.device_index < 0 or args.warmup < 1 or args.iterations < 10:
        parser.error("Require device-index >= 0, warmup >= 1, and at least 10 timed iterations")
    if not math.isfinite(args.qk_scale) or args.qk_scale <= 0:
        parser.error("qk-scale must be positive and finite")
    if len(set(args.degrees)) != len(args.degrees) or len(set(args.implementations)) != len(args.implementations):
        parser.error("Degrees and implementations must not contain duplicates")
    benchmark(args)


if __name__ == "__main__":
    main()
