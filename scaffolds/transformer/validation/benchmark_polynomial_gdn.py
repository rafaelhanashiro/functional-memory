"""Measure exact quadratic GDN kernel stages and warp choices at training shape.

Runs one independent attention layer on one GPU. It uses the production local
microbatch and sequence/head widths, including the complete value-and-gradient
operation. Warp choices change launch geometry only. PTXAS reports register and
spill usage for each compiled local polynomial kernel when available.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends import polynomial_gdn as backend
from scaffolds.transformer.attention.backends import polynomial_gdn_kernels as kernels
from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--check-kernels", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("This benchmark requires an allocated GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "shape": {"batch": args.batch, "length": args.length, "heads": args.heads, "key_width": args.width,
                  "value_width": args.width},
        "dtype": "bfloat16",
        "variants": {}, "compiled_resources": [],
    }
    context = {"variant": "preparation", "stage": "inverse"}

    def save():
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # The JAX bridge compiles directly to PTX, so ordinary Triton cache files do
    # not contain these kernels. Inspect a second PTXAS compilation without
    # changing the executable used by the benchmark.
    original_compile = bridge.compile_ttir_to_ptx_inplace
    def inspect_compile(*compile_args, **compile_kwargs):
        result = original_compile(*compile_args, **compile_kwargs)
        if result.name.startswith(("forward", "backward_queries", "backward_keys")):
            resource = {**context, "kernel": result.name, "shared_bytes": result.shared_mem_bytes}
            ptxas = os.environ.get("TRITON_PTXAS_PATH")
            if ptxas:
                with tempfile.TemporaryDirectory(prefix="polynomial-gdn-ptx-") as temp:
                    ptx_path = Path(temp) / "kernel.ptx"
                    ptx_path.write_text(result.binary)
                    target = re.search(r"\.target\s+(sm_\w+)", result.binary)
                    command = [ptxas, "-v", str(ptx_path), "-o", str(Path(temp) / "kernel.cubin")]
                    if target:
                        command.extend(["--gpu-name", target.group(1)])
                    compiled = subprocess.run(command, capture_output=True, text=True)
                    resource["ptxas_returncode"] = compiled.returncode
                    resource["ptxas_report"] = compiled.stderr.strip()
                    for field, expression in (
                        ("registers", r"Used (\d+) registers"),
                        ("spill_store_bytes", r"(\d+) bytes spill stores"),
                        ("spill_load_bytes", r"(\d+) bytes spill loads"),
                        ("stack_frame_bytes", r"(\d+) bytes stack frame"),
                    ):
                        found = re.search(expression, compiled.stderr)
                        if found:
                            resource[field] = int(found.group(1))
            report["compiled_resources"].append(resource)
            save()
        return result
    bridge.compile_ttir_to_ptx_inplace = inspect_compile

    shape = (args.batch, args.length, args.heads, args.width)
    rng = np.random.default_rng(127)
    def random_array(shape, scale=1.):
        return rng.standard_normal(shape, dtype=np.float32) * scale
    q_np, k_np = random_array(shape), random_array(shape)
    q_np /= np.linalg.norm(q_np, axis=-1, keepdims=True)
    k_np /= np.linalg.norm(k_np, axis=-1, keepdims=True)
    q, k, v, dout = (jnp.asarray(a, jnp.bfloat16) for a in (q_np, k_np, random_array(shape), random_array(shape)))
    gate_shape = shape[:3]
    g = jnp.asarray(-rng.uniform(.001, .1, gate_shape), jnp.float32)
    beta = jnp.asarray(random_array((*gate_shape, 3), .2), jnp.float32)
    erase = jnp.asarray(rng.uniform(.1, .9, gate_shape), jnp.float32)
    k32 = k.astype(jnp.float32)
    prefix = jnp.cumsum(g, axis=1)
    inverse = jax.jit(backend._inverse_forward)(k32, erase)
    jax.block_until_ready(inverse)
    with jax.default_matmul_precision("highest"):
        transformed = backend._token_view(jnp.swapaxes(backend._chunk_view(inverse), -1, -2) @ backend._chunk_view(k32))
    chunks = args.length // backend.CHUNK_SIZE
    history_shape = jax.ShapeDtypeStruct(
        (args.batch, args.heads, chunks * (chunks + 1) // 2, backend.CHUNK_SIZE, args.width), jnp.float32)
    scalar_shape = jax.ShapeDtypeStruct((), jnp.float32)
    scale = args.width ** -.5
    raw_inputs = (q, k32, v, prefix, beta, erase, inverse, transformed)
    full_inputs = (q, k, v, g, beta, erase)
    baseline = None

    def measure(label, function, values):
        context["stage"] = label
        started = time.perf_counter()
        executable = jax.jit(function).lower(*values).compile()
        compile_seconds = time.perf_counter() - started
        for _ in range(2):
            result = executable(*values)
            jax.block_until_ready(result)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = executable(*values)
            jax.block_until_ready(result)
            samples.append(time.perf_counter() - started)
        timing = {"seconds": samples, "median_seconds": statistics.median(samples),
                  "minimum_seconds": min(samples), "compile_seconds": compile_seconds}
        try:
            timing["memory_analysis"] = dataclasses.asdict(executable.memory_analysis())
        except TypeError:
            timing["memory_analysis"] = str(executable.memory_analysis())
        print(json.dumps({"variant": context["variant"], "stage": label, **timing}), flush=True)
        return result, timing

    original_call = backend._polynomial_call
    per_stage = {}
    variants = [(f"warps_{warps}", {name: warps for name in ("forward", "backward_queries", "backward_keys")})
                for warps in args.warps]
    for variant_index in range(len(variants) + 1):
        if variant_index == len(variants):
            if len(args.warps) < 2:
                break
            # Forward and history share a function, so choose their combined
            # timing; backward stages are independent launch choices.
            best_forward = min(args.warps, key=lambda w: per_stage[w]["forward"] + per_stage[w]["history"])
            choices = {"forward": best_forward,
                       "backward_queries": min(args.warps, key=lambda w: per_stage[w]["backward_queries"]),
                       "backward_keys": min(args.warps, key=lambda w: per_stage[w]["backward_keys"])}
            variant = "mixed_best"
        else:
            variant, choices = variants[variant_index]
        context["variant"] = variant
        entry = {"num_warps": choices, "timings": {}}
        report["variants"][variant] = entry

        def configured_call(kernel, *operands, out, q, v, name, **extra):
            return bridge.triton_call(
                *operands, kernel=kernel, out_shape=out,
                grid=(q.shape[1] // backend.CHUNK_SIZE, q.shape[0] * q.shape[2]),
                name=f"polynomial_gdn_benchmark::{variant}::{name}",
                default_dot_input_precision="tf32x3", num_warps=choices[kernel.__name__],
                num_stages=1, **backend._metadata(q, v), **extra)
        backend._polynomial_call = configured_call

        def history_call(*a):
            return configured_call(kernels.forward, *a, float(scale),
                                   out=[scalar_shape, history_shape], q=q, v=v, name="recompute_queries",
                                   SAVE_HISTORY=True, COMPUTE_OUTPUT=False)[1]
        _, entry["timings"]["forward"] = measure("forward", lambda *a: backend._forward(*a, scale), raw_inputs)
        history, entry["timings"]["history"] = measure("history", history_call, raw_inputs)
        query_inputs = (k32, v, prefix, beta, erase, inverse, transformed, dout, history)
        def query_call(*a):
            return configured_call(
                kernels.backward_queries, *a, float(scale),
                out=[jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct(prefix.shape, jnp.float32), history_shape],
                q=q, v=v, name="backward_queries")
        query_results, entry["timings"]["backward_queries"] = measure("backward_queries", query_call, query_inputs)
        adjoint_history = query_results[-1]
        key_inputs = (*query_inputs, adjoint_history)
        def key_call(*a):
            return configured_call(
                kernels.backward_keys, *a, float(scale),
                out=[jax.ShapeDtypeStruct(k.shape, jnp.float32), jax.ShapeDtypeStruct(v.shape, v.dtype),
                     jax.ShapeDtypeStruct(beta.shape, beta.dtype), jax.ShapeDtypeStruct(erase.shape, erase.dtype),
                     jax.ShapeDtypeStruct(inverse.shape, jnp.float32), jax.ShapeDtypeStruct(transformed.shape, jnp.float32),
                     jax.ShapeDtypeStruct(prefix.shape, jnp.float32)],
                q=q, v=v, name="backward_keys", GROUP=0, BQ=64, MODE=0)
        _, entry["timings"]["backward_keys"] = measure("backward_keys", key_call, key_inputs)
        def key_values_call(*a):
            return configured_call(
                kernels.backward_keys, *a, float(scale),
                out=[scalar_shape, jax.ShapeDtypeStruct(v.shape, v.dtype),
                     jax.ShapeDtypeStruct(beta.shape, beta.dtype), scalar_shape, scalar_shape, scalar_shape,
                     jax.ShapeDtypeStruct(prefix.shape, jnp.float32)],
                q=q, v=v, name="backward_key_values", GROUP=1, BQ=64, MODE=0)
        def key_geometry_call(*a):
            return configured_call(
                kernels.backward_keys, *a, float(scale),
                out=[jax.ShapeDtypeStruct(k.shape, jnp.float32), scalar_shape, scalar_shape,
                     jax.ShapeDtypeStruct(erase.shape, erase.dtype), jax.ShapeDtypeStruct(inverse.shape, jnp.float32),
                     jax.ShapeDtypeStruct(transformed.shape, jnp.float32), scalar_shape],
                q=q, v=v, name="backward_key_geometry", GROUP=2, BQ=64, MODE=0)
        _, entry["timings"]["backward_key_values"] = measure("backward_key_values", key_values_call, key_inputs)
        _, entry["timings"]["backward_key_geometry"] = measure("backward_key_geometry", key_geometry_call, key_inputs)
        for mode, label in ((1, "diagonal"), (2, "cross")):
            def geometry_part(*a):
                return configured_call(
                    kernels.backward_keys, *a, float(scale),
                    out=[jax.ShapeDtypeStruct(k.shape, jnp.float32), scalar_shape, scalar_shape,
                         jax.ShapeDtypeStruct(erase.shape, erase.dtype),
                         jax.ShapeDtypeStruct(inverse.shape, jnp.float32) if mode == 1 else scalar_shape,
                         jax.ShapeDtypeStruct(transformed.shape, jnp.float32) if mode == 2 else scalar_shape,
                         scalar_shape], q=q, v=v, name=f"backward_key_geometry_{label}", GROUP=2, BQ=64, MODE=mode)
            _, entry["timings"][f"backward_key_geometry_{label}"] = measure(
                f"backward_key_geometry_{label}", geometry_part, key_inputs)
        _, entry["timings"]["inverse_backward"] = measure(
            "inverse_backward", lambda k,e,a,da: backend._inverse_bwd((k,e,a),da),
            (k32, erase, inverse, jnp.ones_like(inverse)))
        del history, adjoint_history, query_inputs, query_results, key_inputs

        def objective(*a):
            return jnp.sum(backend.polynomial_gated_delta_rule(*a, scale).astype(jnp.float32) * dout.astype(jnp.float32))
        full_result, entry["timings"]["value_and_grad"] = measure(
            "value_and_grad", jax.value_and_grad(objective, argnums=tuple(range(6))), full_inputs)
        host_result = jax.tree.map(lambda x: np.asarray(x, np.float32), full_result)
        if baseline is None:
            baseline = host_result
        else:
            errors = []
            for actual, expected in zip(jax.tree.leaves(host_result), jax.tree.leaves(baseline), strict=True):
                numerator = np.linalg.norm((actual - expected).ravel().astype(np.float64))
                denominator = np.linalg.norm(expected.ravel().astype(np.float64))
                relative = float(numerator / max(denominator, 1e-30))
                errors.append(relative)
                if relative > 3e-3:
                    raise AssertionError(f"Launch geometry changed a gradient: {variant}, relative_error={relative}")
            entry["relative_difference_from_baseline"] = errors
        del full_result, host_result
        if variant_index < len(variants):
            per_stage[args.warps[variant_index]] = {
                name: timing["median_seconds"] for name, timing in entry["timings"].items()}
        save()
    backend._polynomial_call = original_call
    if args.check_kernels:
        xml_path = args.output.with_name(args.output.stem + "-gpu-tests.xml")
        started = time.perf_counter()
        checked = subprocess.run([
            sys.executable, "-m", "pytest", "-q", "-c", "scaffolds/transformer/pyproject.toml",
            "-m", "gpu", "scaffolds/transformer/tests/test_polynomial_gdn_kernel.py",
            f"--junitxml={xml_path}",
        ])
        report["kernel_tests"] = {"returncode": checked.returncode, "seconds": time.perf_counter() - started,
                                  "junit_xml": str(xml_path)}
        if checked.returncode:
            report["status"] = "failed_kernel_checks"
            save()
            raise SystemExit(checked.returncode)
    report["status"] = "passed"
    save()
    print(f"benchmark_result={args.output}", flush=True)


if __name__ == "__main__":
    main()
