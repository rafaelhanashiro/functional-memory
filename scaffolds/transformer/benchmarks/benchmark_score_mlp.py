"""Per-layer score-MLP attention time at the Transformer++ attention shape, fused and blocked.

jit forward + VJP (the output and every operand gradient) of one layer and micro-batch:
micro-batch 16, 14 heads, 2,048 tokens, 64-channel BF16 Q/K/V, FP32 units around the recipe
constants (the GPU tests' production inputs), both forms at J = 4 and 8, and plain
unnormalized exp, on the fused and blocked backends. Attention time per update is 15 layers
times 16 micro-batches, without remat recomputation or the rest of the model. Unless
``--no-sweep``, it also times the fused kernels under other tiles, warps, and stages (module
globals patched at trace time), forward alone and forward + VJP. Compilation and input
generation are outside the timed samples. Run it on one GPU through
``validation/qualify_score_mlp_fused.sbatch`` (step ``benchmark``), or directly:

    python -m scaffolds.transformer.benchmarks.benchmark_score_mlp \
      --output scaffolds/transformer/validation/runs/score-mlp/fused-throughput.json
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import os
from pathlib import Path
import shlex
import socket
import sys

import jax

from scaffolds.transformer.attention import score_mlp
from scaffolds.transformer.attention.backends import fused_score_mlp as fused
from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, time_program, write_report
from scaffolds.transformer.tests.test_score_mlp_fused import _attend, _production_inputs, _vjp

LAYERS, MICRO_BATCHES = 15, 16
SHAPE = {"batch": 16, "time": 2048, "heads": 14, "qk_head_dim": 64, "value_head_dim": 64}
WIDTHS = (4, 8)
# (BLOCK_Q, BLOCK_K, NUM_WARPS, NUM_STAGES); production first.
PRODUCTION = (fused.BLOCK_Q, fused.BLOCK_K, fused.NUM_WARPS, fused.NUM_STAGES)
CANDIDATES = (PRODUCTION, (32, 32, 8, 2), (32, 32, 4, 1), (32, 32, 4, 3), (64, 32, 4, 2), (32, 64, 4, 2),
              (64, 64, 4, 2))
SWEEP_CASES = (("exp_residual", 8), ("kernel", 8), ("exp_residual", 4), ("kernel", 4))


def _inputs(form, width):
    return _production_inputs(form, width, batch=SHAPE["batch"], heads=SHAPE["heads"],
                              channels=SHAPE["qk_head_dim"])


def _forward_and_vjp(attend, cotangent):
    return lambda *x: _vjp(attend, x, cotangent)


def _timed(function, arguments, args):
    timing, _ = time_program(function, arguments, warmup=args.warmup, iterations=args.iterations)
    return {key: round(timing[key], 3) for key in ("median_ms", "p10_ms", "p90_ms", "compilation_seconds")}


def _set_candidate(candidate):
    # The kernel reads these globals while tracing, including through its custom VJP.
    jax.clear_caches()
    gc.collect()
    fused.BLOCK_Q, fused.BLOCK_K, fused.NUM_WARPS, fused.NUM_STAGES = candidate


def throughput(args) -> dict:
    timings = {}
    for form in score_mlp.FORMS:
        for width in WIDTHS:
            operands, cotangent = _inputs(form, width)
            for backend in ("fused", "blocked"):
                name = f"{backend}_{form}-{width}"
                timings[name] = _timed(_forward_and_vjp(_attend(form, backend), cotangent), operands, args)
                print(name, timings[name], flush=True)
    (q, k, v, *_), cotangent = _inputs("exp_residual", 4)
    for name, attend in (("fused_exp", fused_unnormalized_attention),
                         ("blocked_exp", lambda *x: blocked_dot_product_attention(*x, attention_normalization=False))):
        timings[name] = _timed(_forward_and_vjp(attend, cotangent), (q, k, v), args)
        print(name, timings[name], flush=True)
    return timings


def tile_sweep(args) -> dict:
    """Forward alone and forward + VJP of the fused kernels per setting; the backward is their difference."""
    sweep = {}
    try:
        for form, width in SWEEP_CASES:
            operands, cotangent = _inputs(form, width)
            for candidate in CANDIDATES:
                _set_candidate(candidate)
                key = "{}-{} tiles={}x{} warps={} stages={}".format(form, width, *candidate)
                try:
                    forward = _timed(_attend(form), operands, args)
                    both = _timed(_forward_and_vjp(_attend(form), cotangent), operands, args)
                    sweep[key] = {"forward_ms": forward["median_ms"], "forward_and_vjp_ms": both["median_ms"],
                                  "backward_ms": round(both["median_ms"] - forward["median_ms"], 3)}
                except Exception as error:  # noqa: BLE001 - a setting that fails to compile or fit is a result
                    sweep[key] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
                print(key, sweep[key], flush=True)
    finally:
        _set_candidate(PRODUCTION)
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--no-sweep", action="store_true", help="skip the tile, warp, and stage sweep")
    args = parser.parse_args()
    device = jax.devices()[0]
    source = provenance(device)
    report = {
        "producer": "scaffolds/transformer/benchmarks/benchmark_score_mlp.py",
        "command": shlex.join(["python", "-m", "scaffolds.transformer.benchmarks.benchmark_score_mlp", *sys.argv[1:]]),
        "created_utc": datetime.now(UTC).isoformat(),
        "gpu_type": device.device_kind, "node": socket.gethostname().split(".")[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "source_commit": source["source_commit"],
        "provenance": source, "shape": SHAPE,
        "kernel_settings": dict(zip(("block_q", "block_k", "num_warps", "num_stages"), PRODUCTION)),
        "measure": (f"jit forward + VJP of one layer and micro-batch, median of {args.iterations} calls after "
                    f"{args.warmup} warm calls, milliseconds"),
        "note": f"{LAYERS} layers x {MICRO_BATCHES} micro-batches per update; excludes remat recomputation "
                "and the rest of the model",
    }
    timings = throughput(args)
    report.update(milliseconds={name: timing["median_ms"] for name, timing in timings.items()},
                  attention_seconds_per_update={name: round(timing["median_ms"] * LAYERS * MICRO_BATCHES / 1e3, 2)
                                                for name, timing in timings.items()},
                  timings=timings)
    write_report(args.output, report)
    if not args.no_sweep:
        report["tile_sweep"] = {
            "measure": "fused kernels with BLOCK_Q, BLOCK_K, NUM_WARPS, NUM_STAGES patched at trace time; medians, ms",
            "milliseconds": tile_sweep(args)}
        write_report(args.output, report)


if __name__ == "__main__":
    main()
