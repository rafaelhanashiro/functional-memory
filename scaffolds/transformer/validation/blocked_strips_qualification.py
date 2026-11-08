"""GPU qualification of the blocked backend's causal strips against its scan.

Stages (run from the repository root with PYTHONPATH=.):
  kernel  BF16 inputs at [2, 2048, 4, 64]: relative L2 error of the scan's and the strips'
          output and gradients against the scan's own arithmetic in FP64, for the forget-gate
          mechanisms that train on the blocked backend, a static polynomial, and the quartic
          B-spline score function (static and query coefficients near its Taylor start, queries
          scaled so that scores reach every piece and both continuations).
  model   One forward+backward micro-step of a recipe's full model (training.loop.loss_and_grads)
          per implementation, each in a fresh process: compile time, steady-state s/update,
          allocator peak, and the loss and gradient norm the two implementations reach.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SHAPE = tuple(int(x) for x in os.environ.get("BLOCKED_STRIPS_SHAPE", "2,2048,4,64").split(","))
CASES = ("fox_softmax", "fox_exp", "fox_query_m4", "static_m4", "static_spline_m4", "query_spline_m4")
SPLINE_KNOTS = (-32.0, -6.0, 0.0, 6.0, 32.0)


def _inputs(case: str, seed: int = 0):
    """Operands in FP32 (rounded to BF16 where the model feeds BF16) and the keyword arguments."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    batch, length, heads, width = SHAPE
    q, k, v, g = (rng.standard_normal(SHAPE).astype(np.float32) for _ in range(4))
    if case.endswith("_spline_m4"):
        # Per-query scales from 0.5 to 16 (log-uniform): scores reach every piece and pass +-32.
        q = q * np.exp(rng.uniform(math.log(0.5), math.log(16.0), (*SHAPE[:3], 1))).astype(np.float32)
    # Unit-RMS rows give score standard deviation about one, as QK RMSNorm does at initialization.
    q, k, v, g = (jnp.asarray(x, jnp.bfloat16) for x in (q, k, v, g))
    log_alpha = np.log(1.0 / (1.0 + np.exp(-rng.normal(3.0, 1.0, (batch, length, heads)))))
    cumulative = jnp.asarray(np.cumsum(log_alpha, axis=1), jnp.float32)
    arrays = {}
    if case == "fox_softmax":
        arguments = {"attention_normalization": True}
        arrays["query_log_decay"] = cumulative
    elif case == "fox_exp":
        arguments = {"attention_normalization": False}
        arrays["query_log_decay"] = cumulative
    elif case == "fox_query_m4":
        arguments = {"attention_degree": 4, "attention_normalization": False}
        b = np.array([1, 1, 1 / 2, 1 / 6, 1 / 24], np.float32)
        a = 0.3 * b * np.tanh(rng.standard_normal((batch, length, heads, 5))).astype(np.float32)
        arrays["query_polynomial_coefficients"] = jnp.asarray(b + a, jnp.float32)
        arrays["query_log_decay"] = cumulative
    elif case == "static_m4":
        arguments = {"attention_degree": 4, "attention_normalization": False}
        coefficients = np.array([1, 1, 1 / 2, 1 / 6, 1 / 24], np.float32)
        arrays["polynomial_coefficients"] = jnp.asarray(
            coefficients * (1 + 0.1 * rng.standard_normal((heads, 5))), jnp.float32)
    elif case in ("static_spline_m4", "query_spline_m4"):
        from scaffolds.transformer.attention.spline import taylor_coefficients

        arguments = {"attention_degree": 4, "attention_normalization": False, "spline_knots": SPLINE_KNOTS}
        start = np.asarray([float(value) for value in taylor_coefficients(SPLINE_KNOTS, 4)])
        if case == "static_spline_m4":
            arrays["spline_coefficients"] = jnp.asarray(start * (1 + 0.1 * rng.standard_normal((heads, 8))),
                                                        jnp.float32)
        else:
            arrays["query_spline_coefficients"] = jnp.asarray(
                start * (1 + 0.1 * np.tanh(rng.standard_normal((batch, length, heads, 8)))), jnp.float32)
    else:
        raise ValueError(case)
    return (q, k, v, g), arrays, arguments


def _relative_l2(actual, reference) -> float:
    actual = np.asarray(actual, np.float64)
    reference = np.asarray(reference, np.float64)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def kernel_stage(output: Path) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from scaffolds.transformer.attention.backends import reference as ref

    report = {"stage": "kernel", "shape": SHAPE, "device": str(jax.devices()[0].device_kind), "cases": {}}
    for case in CASES:
        (q, k, v, g), arrays, arguments = _inputs(case)
        names = sorted(arrays)

        def vjp(function, q, k, v, g, *values):
            def attend(q, k, v, *values):
                return function(q, k, v, **dict(zip(names, values)), **arguments)

            out, pullback = jax.vjp(attend, q, k, v, *values)
            return (out, *pullback(g.astype(out.dtype)))

        def upcast(x):
            return x.astype(jnp.float64)

        exact = jax.jit(lambda *xs: vjp(lambda q, k, v, **a: ref._blocked_scan(q, k, v, compute=jnp.float64, **a),
                                        *map(upcast, xs)))
        scan = jax.jit(lambda *xs: vjp(ref.blocked_dot_product_attention, *xs))
        strips = jax.jit(lambda *xs: vjp(ref.strip_dot_product_attention, *xs))
        operands = (q, k, v, g, *(arrays[name] for name in names))
        reference_values = [np.asarray(x) for x in jax.device_get(exact(*operands))]
        labels = ["output", "dq", "dk", "dv", *(f"d_{name}" for name in names)]
        floor = {label: _relative_l2(np.asarray(jnp.asarray(value, jnp.bfloat16).astype(jnp.float32)), value)
                 for label, value in zip(labels[:4], reference_values[:4])}
        entry = {"floor": floor}
        for implementation, program in (("scan", scan), ("strips", strips)):
            started = time.perf_counter()
            values = jax.device_get(program(*operands))
            entry[implementation] = {
                "relative_l2": {label: _relative_l2(value, expected)
                                for label, value, expected in zip(labels, values, reference_values)},
                "finite": bool(all(np.all(np.isfinite(np.asarray(x, np.float64))) for x in values)),
                "first_call_seconds": time.perf_counter() - started,
            }
        report["cases"][case] = entry
        summary = {impl: {label: round(entry[impl]["relative_l2"][label] / floor[label], 3)
                          for label in ("output", "dq", "dk", "dv")} for impl in ("scan", "strips")}
        print(f"kernel {case} error/floor {json.dumps(summary)}", flush=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def model_case(recipe: Path, implementation: str, micro_batch: int, output: Path) -> dict:
    import jax
    import jax.numpy as jnp
    import optax

    from scaffolds.transformer.validation.blocked_attention_benchmark import (
        load_model_config, measure, micro_step_program,
    )

    config = load_model_config(recipe, micro_batch=micro_batch, backend="blocked")
    config.model.attention_blocked_implementation = implementation
    state, program, _ = micro_step_program(config, variant=None)
    batch = jnp.asarray(np.random.default_rng(config.train.seed).integers(
        0, config.model.vocab_size, (micro_batch, config.data.sequence_length), dtype=np.int32))
    rng = jax.random.PRNGKey(config.train.seed)
    timing, (nll, grads), _ = measure(program, (state.params, batch, rng), warmup=2, iterations=8)
    nll, grad_norm = (float(x) for x in jax.device_get((nll, optax.tree.norm(grads))))
    stats = jax.local_devices()[0].memory_stats() or {}
    result = {"recipe": str(recipe), "implementation": implementation, "micro_batch": micro_batch,
              "timing": timing, "nll": nll, "grad_norm": grad_norm,
              "seconds_per_update": timing["median_ms"] * config.data.batch_size / micro_batch / 1e3,
              "peak_bytes_in_use": stats.get("peak_bytes_in_use"), "device": str(jax.devices()[0].device_kind)}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def model_stage(recipes: list[str], micro_batch: int, output: Path) -> dict:
    report = {"stage": "model", "micro_batch": micro_batch, "cases": []}
    for recipe in recipes:
        for implementation in ("scan", "strips"):
            path = output.parent / f".case-{Path(recipe).parent.name}-{implementation}-{os.getpid()}.json"
            command = [sys.executable, "-m", "scaffolds.transformer.validation.blocked_strips_qualification",
                       "model-case", "--recipe", recipe, "--implementation", implementation,
                       "--micro-batch", str(micro_batch), "--output", str(path)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if path.exists():
                case = json.loads(path.read_text())
                path.unlink()
            else:
                case = {"recipe": recipe, "implementation": implementation, "status": "crashed",
                        "returncode": completed.returncode, "stderr_tail": completed.stderr[-3000:]}
            report["cases"].append(case)
            keys = ("seconds_per_update", "nll", "grad_norm", "peak_bytes_in_use")
            summary = {key: case.get(key) for key in keys}
            if "timing" in case:
                summary["compile_seconds"] = case["timing"].get("compile_seconds")
            print(f"model {Path(recipe).parent.name} {implementation} {json.dumps(summary)}", flush=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    kernel = sub.add_parser("kernel")
    kernel.add_argument("--output", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--recipes", nargs="+", required=True)
    model.add_argument("--micro-batch", type=int, default=16)
    model.add_argument("--output", type=Path, required=True)
    case = sub.add_parser("model-case")
    case.add_argument("--recipe", type=Path, required=True)
    case.add_argument("--implementation", choices=("scan", "strips"), required=True)
    case.add_argument("--micro-batch", type=int, required=True)
    case.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "kernel":
        kernel_stage(args.output)
    elif args.stage == "model":
        model_stage(args.recipes, args.micro_batch, args.output)
    else:
        model_case(args.recipe, args.implementation, args.micro_batch, args.output)


if __name__ == "__main__":
    main()
