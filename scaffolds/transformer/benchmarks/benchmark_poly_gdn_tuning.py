"""Profile opt-in cubic GDN precision and launch choices on one GPU.

Each timed call uses precompiled code and resident inputs. Forward/backward
returns the output and all six input gradients with an explicit cotangent.
Native FLA uses the same shared input fixture in benchmark_upstream_torch, but
is a linear, tied-gate reference workload, not the same mathematical operator.
Stage timings are isolated calls with resident intermediates; their sum is not
an end-to-end timing. Accuracy is checked against tf32x3 at --reference-warps
(default two, the qualified Blackwell setting) and --reference-min-ctas
(default zero, no occupancy hint) on the exact same cast operands.
Independent small FP64 tests remain separate qualification.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.benchmarks.benchmark_attention import compiled_memory, write_report
from scaffolds.transformer.benchmarks.benchmark_training_throughput import _execution_provenance
from scaffolds.transformer.benchmarks import upstream_inputs


INPUT_NAMES = ("q", "k", "v", "g", "beta", "erase")
PRECISIONS = ("tf32x3", "bf16_read", "bf16_values")


def array_digest(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def json_safe(value):
    """Keep failed numerical checks in valid JSON instead of losing the report."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Timings must be finite and positive")
    return {"samples": len(values), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)), "minimum_seconds": float(values.min()),
            "p10_seconds": float(np.quantile(values, .1)), "p90_seconds": float(np.quantile(values, .9))}


def value_and_vjp(forward, *arguments):
    output, pullback = jax.vjp(forward, *arguments[:-1])
    return output, pullback(arguments[-1])


@jax.jit
def error_metrics(actual, expected):
    """Scalar-only transfer; FP32 reductions with a scale that avoids squaring overflow."""
    actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
    difference = actual - expected
    reference_max = jnp.max(jnp.abs(expected))
    scale = jnp.maximum(reference_max, jnp.finfo(jnp.float32).tiny)
    delta_rms = jnp.sqrt(jnp.mean(jnp.square(difference / scale)))
    reference_rms = jnp.sqrt(jnp.mean(jnp.square(expected / scale)))
    max_abs = jnp.max(jnp.abs(difference))
    relative_rms = delta_rms / jnp.maximum(reference_rms, jnp.finfo(jnp.float32).tiny)
    relative_max = max_abs / scale
    return {"finite": jnp.all(jnp.isfinite(actual)) & jnp.all(jnp.isfinite(expected)),
            "relative_rms": relative_rms, "max_abs": max_abs, "relative_max": relative_max,
            "reference_rms": reference_rms * scale, "reference_max": reference_max}


def compare_result(actual, expected, *, backward, rms_limit, max_limit):
    names = ("output", *INPUT_NAMES) if backward else ("output",)
    actual_values = (actual[0], *actual[1]) if backward else (actual,)
    expected_values = (expected[0], *expected[1]) if backward else (expected,)
    channels = {}
    for name, value, reference in zip(names, actual_values, expected_values, strict=True):
        if value.shape != reference.shape:
            raise ValueError(f"{name} result shape {value.shape} differs from reference {reference.shape}")
        raw = jax.device_get(error_metrics(value, reference))
        result = {key: bool(item) if key == "finite" else float(item) for key, item in raw.items()}
        result.update(shape=list(value.shape), dtype=str(value.dtype))
        result["passed"] = result["finite"] and result["relative_rms"] <= rms_limit and result["relative_max"] <= max_limit
        channels[name] = result
    return {"passed": all(item["passed"] for item in channels.values()),
            "all_finite": all(item["finite"] for item in channels.values()),
            "limits": {"relative_rms": rms_limit, "relative_max": max_limit}, "channels": channels}


def finite_result(result):
    return all(bool(value) for value in jax.device_get(
        [jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(result)]))


def runtime_ptxas_flags():
    """Match pinned JAX's literal-space split and optional device-debug flag."""
    flags = [flag for flag in os.environ.get("JAX_TRITON_PTXAS_EXTRA_FLAGS", "").split(" ") if flag]
    if (os.environ.get("JAX_TRITON_PTXAS_DEVICE_DEBUG") in ("1", "true")
            and "--device-debug" not in flags and "-g" not in flags):
        flags.append("--device-debug")
    return flags


@contextmanager
def record_resources(report, context, save, enabled):
    """Inspect the actual PTX passed to JAX, including bridge cache hits.

    A separate PTXAS invocation extracts register/spill information without
    modifying the timed executable or any installed package. Its time is part
    of reported compilation, never measured operator time.
    """
    if not enabled:
        yield
        return
    from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge
    original = bridge.triton_kernel_call_lib.TritonKernel
    seen = set()

    def inspect_kernel(*args, **kwargs):
        kernel = original(*args, **kwargs)
        ptx_indices = [index for index, value in enumerate(args)
                       if isinstance(value, str) and ".version " in value]
        if not ptx_indices:
            return kernel
        index = ptx_indices[0]
        ptx = args[index]
        digest = hashlib.sha256(ptx.encode()).hexdigest()
        identity = (context.get("shape"), context.get("variant"), context.get("operation"), digest)
        if identity in seen:
            return kernel
        seen.add(identity)
        extra_flags = os.environ.get("JAX_TRITON_PTXAS_EXTRA_FLAGS", "")
        resource = {**context, "kernel": args[0], "num_warps": args[1],
                    "shared_bytes": args[index - 1], "ptx_sha256": digest,
                    "min_ctas_directive": re.findall(r"\.minnctapersm\s+(\d+)", ptx),
                    "runtime_ptxas_extra_flags": extra_flags,
                    "runtime_ptxas_device_debug": os.environ.get("JAX_TRITON_PTXAS_DEVICE_DEBUG")}
        assembler = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH") or os.environ.get("TRITON_PTXAS_PATH")
        if assembler:
            try:
                with tempfile.TemporaryDirectory(prefix="poly-gdn-resource-") as directory:
                    source = Path(directory) / "kernel.ptx"
                    source.write_text(ptx)
                    target = re.search(r"\.target\s+(sm_\w+)", ptx)
                    command = [assembler, "-v", str(source), "-o", str(Path(directory) / "kernel.cubin")]
                    if target:
                        command += ["--gpu-name", target.group(1)]
                    command += runtime_ptxas_flags()
                    resource["inspection_ptxas_command"] = command
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    resource.update(ptxas_returncode=result.returncode, ptxas_report=result.stderr.strip())
                    for field, expression in (("registers", r"Used (\d+) registers"),
                                              ("spill_store_bytes", r"(\d+) bytes spill stores"),
                                              ("spill_load_bytes", r"(\d+) bytes spill loads"),
                                              ("stack_frame_bytes", r"(\d+) bytes stack frame")):
                        found = re.search(expression, result.stderr)
                        resource[field] = int(found.group(1)) if found else None
            except (OSError, subprocess.SubprocessError) as error:
                resource["resource_inspection_error"] = f"{type(error).__name__}: {error}"
        else:
            resource["resource_inspection_error"] = "No locked PTXAS path provided"
        report["compiled_resources"].append(resource)
        save()
        return kernel

    bridge.triton_kernel_call_lib.TritonKernel = inspect_kernel
    try:
        yield
    finally:
        bridge.triton_kernel_call_lib.TritonKernel = original


class Runner:
    def __init__(self, args, report):
        self.args, self.report = args, report
        self.context = {}

    def save(self):
        write_report(self.args.output, json_safe(self.report))

    def active(self, **fields):
        self.context.update(fields)
        self.report["active_stage"] = self.context.copy()
        self.save()

    def compile(self, name, function, arguments, destination):
        self.active(operation=name, phase="compiling")
        started = time.perf_counter()
        timing = {"status": "compiling", "rounds": []}
        destination[name] = timing
        self.save()
        try:
            executable = jax.jit(function).lower(*arguments).compile()
        finally:
            timing["compilation_seconds"] = time.perf_counter() - started
            self.save()
        hlo = executable.as_text()
        timing.update(status="compiled", compiled_memory=compiled_memory(executable),
                      hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                      custom_call_targets=sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo))))
        self.active(phase="checking_first_call")
        started = time.perf_counter()
        result = jax.block_until_ready(executable(*arguments))
        timing["first_call_seconds"] = time.perf_counter() - started
        timing["first_call_finite"] = finite_result(result)
        self.save()
        if not timing["first_call_finite"]:
            raise FloatingPointError(f"Nonfinite first call: {name}")
        return executable, result

    def measure(self, name, executable, arguments, timing, round_index):
        self.active(operation=name, round=round_index + 1, phase="warmup")
        warmup = []
        for _ in range(self.args.warmup):
            started = time.perf_counter()
            result = jax.block_until_ready(executable(*arguments))
            warmup.append(time.perf_counter() - started)
        self.active(phase="timing")
        samples = []
        for _ in range(self.args.iterations):
            started = time.perf_counter()
            result = jax.block_until_ready(executable(*arguments))
            samples.append(time.perf_counter() - started)
        # Checks and JSON I/O are outside every measured region.
        finite = finite_result(result)
        trial = {"round": round_index + 1, "warmup_seconds": warmup,
                 "samples_seconds": samples, "summary": summarize(samples), "all_finite": finite,
                 "finite_check_scope": "All returned arrays of the last measured call; identical resident inputs every call."}
        timing["rounds"].append(trial)
        timing["summary"] = summarize([sample for trial in timing["rounds"] for sample in trial["samples_seconds"]])
        timing["status"] = "complete" if len(timing["rounds"]) == self.args.rounds else "running"
        self.save()
        print(json.dumps({**self.context, "mean_ms": trial["summary"]["mean_seconds"] * 1000,
                          "all_finite": finite}), flush=True)
        if not finite:
            raise FloatingPointError(f"Nonfinite measured result: {name}")
        return result


def stage_profile(runner, entry, operands, cotangent, precision, warps, min_ctas, scale):
    """Use exactly the production stage calls, including separated geometry passes."""
    from scaffolds.transformer.attention.backends import polynomial_gdn as backend
    from scaffolds.transformer.attention.backends import polynomial_gdn_kernels as kernels
    q, k, v, g, beta, erase = operands
    k32 = k.astype(jnp.float32)
    # The public backend promotes the model's BF16 gates before these stages.
    beta, erase = beta.astype(jnp.float32), erase.astype(jnp.float32)
    stages = entry.setdefault("stages", {})
    options = dict(precision=precision, num_warps=warps, min_ctas_per_sm=min_ctas)

    def measure(name, function, arguments):
        executable, result = runner.compile(name, function, arguments, stages)
        for round_index in range(runner.args.rounds):
            result = runner.measure(name, executable, arguments, stages[name], round_index)
        return result

    inverse = measure("inverse_forward", backend._inverse_forward, (k32, erase))

    def preparation(k, g, inverse):
        with jax.default_matmul_precision("highest"):
            transformed = backend._token_view(jnp.swapaxes(backend._chunk_view(inverse), -1, -2)
                                               @ backend._chunk_view(k))
        return jnp.cumsum(g, axis=1), transformed

    prefix, transformed = measure("prefix_and_transformed_keys", preparation, (k32, g, inverse))
    chunks = q.shape[1] // backend.CHUNK_SIZE
    history_shape = jax.ShapeDtypeStruct(
        (q.shape[0], q.shape[2], chunks * (chunks + 1) // 2, backend.CHUNK_SIZE, q.shape[-1]), jnp.float32)
    shape = lambda x, dtype=None: jax.ShapeDtypeStruct(x.shape, x.dtype if dtype is None else dtype)
    dummy = jax.ShapeDtypeStruct((), jnp.float32)
    raw = (q, k32, v, prefix, beta, erase, inverse, transformed)
    measure("attention_forward", lambda *values: backend._forward(*values, scale, **options), raw)

    def call(kernel, values, out, name, **extra):
        return backend._polynomial_call(kernel, *values, float(scale), out=out, q=q, v=v, name=name,
                                        NBETA=beta.shape[-1], **options, **extra)

    # Match the production backward dependency: the history-only kernel receives dO.
    history_inputs = (q, k32, cotangent, prefix, beta, erase, inverse, transformed)
    history = measure("history", lambda *values: call(
        kernels.forward, values, [dummy, history_shape], "recompute_queries",
        SAVE_HISTORY=True, COMPUTE_OUTPUT=False)[1], history_inputs)
    query_inputs = (k32, v, prefix, beta, erase, inverse, transformed, cotangent, history)
    query_result = measure("backward_queries", lambda *values: call(
        kernels.backward_queries, values, [shape(q), shape(prefix), history_shape], "backward_queries"), query_inputs)
    key_inputs = (*query_inputs, query_result[-1])
    def values_backward(*values):
        result = call(kernels.backward_keys, values,
            [dummy, shape(v), shape(beta), dummy, dummy, dummy, shape(prefix)],
            "backward_key_values", GROUP=1, BQ=64, MODE=0)
        return result[1], result[2], result[6]

    def geometry_backward(*values, diagonal):
        result = call(kernels.backward_keys, values,
            [shape(k32), dummy, dummy, shape(erase), shape(inverse) if diagonal else dummy,
             dummy if diagonal else shape(transformed), dummy],
            "backward_key_geometry_diagonal" if diagonal else "backward_key_geometry_cross",
            GROUP=2, BQ=64, MODE=1 if diagonal else 2)
        return result[0], result[3], result[4] if diagonal else result[5]

    measure("backward_values", values_backward, key_inputs)
    diagonal = measure("backward_diagonal", partial(geometry_backward, diagonal=True), key_inputs)
    cross = measure("backward_cross", partial(geometry_backward, diagonal=False), key_inputs)
    # In production R=A^T K adds a local inverse cotangent before inverse backward.
    def transformed_backward(k, a, dr):
        with jax.default_matmul_precision("highest"):
            keys, matrix, gradient = map(backend._chunk_view, (k, a, dr))
            return (backend._token_view(matrix @ gradient),
                    backend._token_view(keys @ jnp.swapaxes(gradient, -1, -2)))
    _, inverse_from_transformed = measure("transformed_keys_backward", transformed_backward, (k32, inverse, cross[2]))
    measure("inverse_backward", lambda k, e, a, da: backend._inverse_bwd((k, e, a), da),
            (k32, erase, inverse, diagonal[2] + inverse_from_transformed))
    entry["stage_scope"] = ("Isolated precompiled stages with resident intermediates; sum excludes scheduling/fusion effects "
                            "and the final gradient additions/reverse cumulative sum.")


def _batch_split_accuracy(runner, check, forward, operands, cotangent, reference, *,
                          prefix, rms_limit, max_limit):
    """Compare one dtype's full call with unchanged slices, releasing chunk histories."""
    size = runner.args.reference_batch_size
    check.update(requested_batch_size=size, chunk_sizes=[], chunk_call_seconds=[])
    check.setdefault("compilations", {})
    executables, host_parts = {}, []
    function = partial(value_and_vjp, forward)
    try:
        for start in range(0, operands[0].shape[0], size):
            stop = min(start + size, operands[0].shape[0])
            actual_size = stop - start
            runner.active(variant="reference", effective_batch=actual_size)
            arguments = tuple(value[start:stop] for value in (*operands, cotangent))
            name = f"{prefix}_b{actual_size}"
            if actual_size not in executables:
                executable, result = runner.compile(name, function, arguments, check["compilations"])
                executables[actual_size] = executable
                seconds = check["compilations"][name]["first_call_seconds"]
            else:
                runner.active(operation=name, phase="checking_split_reference")
                started = time.perf_counter()
                result = jax.block_until_ready(executables[actual_size](*arguments))
                seconds = time.perf_counter() - started
            host_parts.append(jax.device_get(result))
            check["chunk_sizes"].append(actual_size)
            check["chunk_call_seconds"].append(seconds)
            del result, arguments
            runner.save()
        # Transfer/checks are outside every throughput interval. No batch mean
        # or gradient rescaling is applied to the explicit-cotangent VJPs.
        expected = jax.tree.map(lambda *parts: jnp.asarray(np.concatenate(parts, axis=0)), *host_parts)
        check["accuracy"] = compare_result(reference, expected, backward=True,
                                            rms_limit=rms_limit, max_limit=max_limit)
        del expected
        runner.save()
        if not check["accuracy"]["passed"]:
            raise FloatingPointError(f"Full-batch reference differs from the concatenated batch-split reference ({prefix})")
    finally:
        executables.clear()
        host_parts.clear()
        runner.context.pop("effective_batch", None)


def check_batch_split_reference(runner, entry, forward, operands, cotangent, reference):
    """Check global batch offsets in production dtype and strictly in FP32.

    The FP32 check promotes the already-quantized production inputs and the
    explicit cotangent without changing their values. Both checks cover the
    output and all six VJPs and must pass before any variant is timed.
    """
    if not runner.args.reference_batch_size:
        return
    check = {"status": "running", "passed": False,
             "policy": "production_dtype_and_promoted_fp32",
             "scope": "Concatenate exact-reference output and all six VJPs from unchanged batch slices; no gradient rescaling."}
    entry["batch_split_reference"] = check
    promoted = None
    try:
        _batch_split_accuracy(runner, check, forward, operands, cotangent, reference,
            prefix="batch_split_reference", rms_limit=runner.args.exact_relative_rms,
            max_limit=runner.args.exact_relative_max)
        promoted = {"status": "running", "compilations": {},
                    "scope": "Same already-quantized production operands and cotangent promoted to FP32; unchanged reference backend settings."}
        check["promoted_fp32"] = promoted
        source_arguments = (*operands, cotangent)
        arguments = tuple(value.astype(jnp.float32) for value in source_arguments)
        # Compare in the promoted dtype, rather than casting back and hiding a
        # changed FP32 value that happens to round to the original BF16 value.
        preserved = jax.device_get([jnp.all(original == value) for original, value in
                                    zip(source_arguments, arguments, strict=True)])
        promoted["promotion_channels"] = {
            name: {"source_dtype": str(original.dtype), "target_dtype": str(value.dtype),
                   "values_preserved_exactly": bool(same)}
            for name, original, value, same in zip((*INPUT_NAMES, "cotangent"),
                source_arguments, arguments, preserved, strict=True)}
        promoted["promotion_preserves_values_exactly"] = all(bool(same) for same in preserved)
        runner.save()
        if not promoted["promotion_preserves_values_exactly"]:
            raise FloatingPointError("FP32 promotion changed production input or cotangent values")
        runner.active(variant="reference", effective_batch=operands[0].shape[0])
        executable, promoted_reference = runner.compile("batch_split_promoted_fp32_full",
            partial(value_and_vjp, forward), arguments, promoted["compilations"])
        del executable
        _batch_split_accuracy(runner, promoted, forward, arguments[:-1], arguments[-1], promoted_reference,
            prefix="batch_split_promoted_fp32", rms_limit=1e-5, max_limit=1e-5)
        del promoted_reference, arguments
        promoted["status"] = "complete"
        check.update(status="complete", passed=True)
    except Exception as error:
        check.update(status="failed", error=f"{type(error).__name__}: {error}")
        if promoted is not None:
            promoted.update(status="failed", error=check["error"])
        raise
    finally:
        runner.context.pop("effective_batch", None)
        runner.save()


def benchmark(args):
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("Run with exactly one allocated GPU")
    from scaffolds.transformer.attention.backends import polynomial_gdn as backend
    root = Path(__file__).resolve().parents[3]
    source = _execution_provenance(devices[0], args.experiment)
    for path in (Path(__file__), Path(upstream_inputs.__file__)):
        source["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    source["runtime_environment"].update({name: os.environ.get(name) for name in (
        "POLY_GDN_VALUE_PRECISION", "POLY_GDN_NUM_WARPS", "POLY_GDN_MIN_CTAS_PER_SM",
        "JAX_TRITON_PTXAS_EXTRA_FLAGS", "JAX_TRITON_PTXAS_DEVICE_DEBUG")})
    source["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"format": "polynomial-gdn-kernel-tuning", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "provenance": source,
              "configuration": vars(args) | {"output": str(args.output)}, "shapes": {}, "compiled_resources": [],
              "comparison": "Cubic polynomial GDN and native FLA linear tied GDN are different mathematics.",
              "timing": "Resident compiled call plus host dispatch; synchronize every call; compile/transfer/checks excluded.",
              "accuracy_reference": {"precision": "tf32x3", "num_warps": args.reference_warps,
                                     "min_ctas_per_sm": args.reference_min_ctas, "same_cast_inputs": True},
              "accuracy_scope": "Production-size parity to exact backend; independent FP64 oracle tests required separately.",
              "memory": "Compiler estimates and device allocator statistics, not measured isolated kernel peaks."}
    runner = Runner(args, report)
    runner.save()
    started = time.perf_counter()
    try:
        with record_resources(report, runner.context, runner.save, args.resources):
            for batch in args.batches:
                for width in args.key_dims:
                    name = f"b{batch}_t{args.length}_h{args.heads}_k{width}_v{args.value_dim}"
                    shape_entry = {"status": "running", "shape": {"batch": batch, "length": args.length,
                        "heads": args.heads, "key_dim": width, "value_dim": args.value_dim, "degree": args.degree},
                        "variants": {}}
                    report["shapes"][name] = shape_entry
                    runner.active(shape=name, variant="reference", operation="fixture", phase="preparing")
                    try:
                        benchmark_shape(runner, shape_entry, batch, width, backend)
                        states = [variant["status"] for variant in shape_entry["variants"].values()]
                        shape_entry["status"] = "complete" if all(state == "complete" for state in states) else "completed_with_failures"
                    except Exception as error:
                        shape_entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                        print(json.dumps({"shape": name, "error": shape_entry["error"]}), flush=True)
                    finally:
                        shape_entry["gpu_memory_stats"] = devices[0].memory_stats()
                        runner.save()
                        jax.clear_caches()
        report["status"] = "complete" if all(entry["status"] == "complete" for entry in report["shapes"].values()) else "completed_with_failures"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(finished_at_utc=datetime.now(UTC).isoformat(), total_wall_seconds=time.perf_counter() - started)
        report.pop("active_stage", None)
        runner.save()
    return report


def benchmark_shape(runner, entry, batch, width, backend):
    args = runner.args
    host = upstream_inputs.kernel_inputs(batch, args.length, args.heads, width, args.seed, value_width=args.value_dim)
    coefficients = upstream_inputs.polynomial_coefficients(batch, args.length, args.heads, args.degree, args.seed)
    entry["input_sha256"] = {name: array_digest(value) for name, value in host.items()}
    entry["polynomial_coefficients_sha256"] = array_digest(coefficients)
    entry["input_distribution"] = "Shared unit Q/K, Gaussian V/dO, negative log decay, sigmoid erase; independent tanh(.2*N(0,1)) coefficients."
    # Match production: projected coefficients and erasure are cast to the
    # model's BF16 compute dtype; only log decay remains FP32 at the interface.
    operands = tuple(jnp.asarray(coefficients if name == "beta" else host[name],
                                jnp.float32 if name == "g" else jnp.bfloat16) for name in INPUT_NAMES)
    cotangent = jnp.asarray(host["do"], jnp.bfloat16)
    entry["input_dtypes"] = {name: str(value.dtype) for name, value in zip(INPUT_NAMES, operands, strict=True)}
    entry["cotangent_dtype"] = str(cotangent.dtype)
    del host, coefficients
    jax.block_until_ready((operands, cotangent))
    scale = width ** -.5
    entry["scale"] = scale
    forward = partial(backend.polynomial_gated_delta_rule, scale=scale, precision="tf32x3",
                      num_warps=args.reference_warps, min_ctas_per_sm=args.reference_min_ctas)
    reference_operations = entry.setdefault("reference", {})
    executable, reference = runner.compile("forward_backward", partial(value_and_vjp, forward),
                                            (*operands, cotangent), reference_operations)
    del executable
    check_batch_split_reference(runner, entry, forward, operands, cotangent, reference)
    compiled = {}
    for precision, warps, min_ctas in product(args.precisions, args.warps, args.min_ctas):
        variant = f"{precision}_w{warps}_cta{min_ctas}"
        runner.active(variant=variant)
        case = {"status": "compiling", "precision": precision, "num_warps": warps,
                "min_ctas_per_sm": min_ctas, "operations": {}}
        entry["variants"][variant] = case
        functions = {}
        try:
            forward = partial(backend.polynomial_gated_delta_rule, scale=scale, precision=precision,
                              num_warps=warps, min_ctas_per_sm=min_ctas)
            for operation, function, values in (
                ("forward", forward, operands),
                ("forward_backward", partial(value_and_vjp, forward), (*operands, cotangent))):
                executable, actual = runner.compile(operation, function, values, case["operations"])
                exact = precision == "tf32x3"
                checks = compare_result(actual, reference if operation == "forward_backward" else reference[0],
                    backward=operation == "forward_backward",
                    rms_limit=args.exact_relative_rms if exact else args.max_relative_rms,
                    max_limit=args.exact_relative_max if exact else args.max_relative_max)
                case["operations"][operation]["accuracy"] = checks
                functions[operation] = (executable, values)
                del actual
            case["accuracy_passed"] = all(item["accuracy"]["passed"] for item in case["operations"].values())
            case["status"] = "ready"
            compiled[variant] = functions
        except Exception as error:
            case.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(json.dumps({"variant": variant, "error": case["error"]}), flush=True)
        runner.save()
    for round_index in range(args.rounds):
        variants = list(compiled) if round_index % 2 == 0 else list(reversed(compiled))
        for variant in variants:
            case = entry["variants"][variant]
            if case["status"] == "failed":
                continue
            runner.active(variant=variant)
            try:
                for operation, (executable, values) in compiled[variant].items():
                    runner.measure(operation, executable, values, case["operations"][operation], round_index)
                case["status"] = ("complete" if case["accuracy_passed"] else "rejected_accuracy") if round_index == args.rounds - 1 else "running"
            except Exception as error:
                case.update(status="failed", error=f"{type(error).__name__}: {error}")
            runner.save()
    # Release full-call executables/results before allocating isolated history workspaces.
    compiled.clear()
    del reference
    if args.profile_stages:
        for variant, case in entry["variants"].items():
            if case["status"] == "failed" or args.profile_variants and variant not in args.profile_variants:
                continue
            runner.active(variant=variant)
            try:
                stage_profile(runner, case, operands, cotangent, case["precision"], case["num_warps"],
                              case["min_ctas_per_sm"], scale)
                case["stage_status"] = "complete"
            except Exception as error:
                case.update(stage_status="failed", stage_error=f"{type(error).__name__}: {error}")
                if case["status"] == "complete":
                    case["status"] = "failed_stage_profile"
            runner.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[16])
    parser.add_argument("--key-dims", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--degree", type=int, choices=(2, 3), default=3)
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    parser.add_argument("--warps", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2],
                        help="Sweep launch choices; unsupported hardware settings are rejected before launch")
    parser.add_argument("--reference-warps", type=int, choices=(1, 2, 4, 8), default=2,
                        help="Exact tf32x3 parity reference; two warps is the qualified Blackwell setting")
    parser.add_argument("--min-ctas", type=int, nargs="+", choices=(0, 1, 2), default=[0],
                        help="Minimum CTAs per SM occupancy hints; zero disables the hint")
    parser.add_argument("--reference-min-ctas", type=int, choices=(0, 1, 2), default=0,
                        help="Exact parity reference occupancy hint; zero disables the hint")
    parser.add_argument("--reference-batch-size", type=int, default=0,
                        help="Check production-dtype and same-quantized FP32 batch-split parity before timing; zero disables")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--profile-variants", nargs="+", help="Optional precision_wN_ctaN names to profile")
    parser.add_argument("--resources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-relative-rms", type=float, default=.01)
    parser.add_argument("--max-relative-max", type=float, default=.05)
    parser.add_argument("--exact-relative-rms", type=float, default=.003)
    parser.add_argument("--exact-relative-max", type=float, default=.01)
    parser.add_argument("--experiment", default="attention_scaling")
    args = parser.parse_args()
    # Confirmation may name the same baseline and selected setting. Preserve
    # requested order while compiling and recording every unique case once.
    for name in ("batches", "key_dims", "precisions", "warps", "min_ctas"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    if min(*args.batches, *args.key_dims, args.value_dim, args.length, args.heads,
           args.warmup, args.iterations, args.rounds) < 1:
        parser.error("Dimensions and timing counts must be positive")
    if args.reference_batch_size < 0:
        parser.error("Reference batch size must be zero or positive")
    if args.length % 64 or max(*args.key_dims, args.value_dim) > 64:
        parser.error("The backend requires T divisible by64 and K,V<=64")
    if min(args.max_relative_rms, args.max_relative_max, args.exact_relative_rms, args.exact_relative_max) <= 0:
        parser.error("Accuracy thresholds must be positive")
    result = benchmark(args)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
