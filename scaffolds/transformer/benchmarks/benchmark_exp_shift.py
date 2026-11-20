"""Per-layer time of the fused unnormalized exponential, unshifted and shifted by its row maximum.

jit forward + VJP (output and Q/K/V gradients; the shifted kernel's shift held constant) and forward alone, of one
layer and micro-batch at the reference shape's attention: micro-batch 16, 14 heads, 2,048 tokens, BF16 Q/K at each
key width and 64-channel values. Attention time per update is 15 layers times 16 micro-batches, without the rest of
the model. Compilation and input generation are outside the timed samples. Run it on one GPU through
``validation/qualify_exp_shift.sbatch``, or directly:

    python -m scaffolds.transformer.benchmarks.benchmark_exp_shift --output <report.json>
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, time_program, write_report

LAYERS, MICRO_BATCHES = 15, 16
BATCH, TIME, HEADS, VALUE_WIDTH = 16, 2048, 14, 64
WIDTHS = (4, 8, 16, 64)


def _inputs(width):
    rng = np.random.default_rng(width)
    q, k = (jnp.asarray(0.7 * rng.standard_normal((BATCH, TIME, HEADS, width)), jnp.bfloat16) for _ in range(2))
    v, cotangent = (jnp.asarray(rng.standard_normal((BATCH, TIME, HEADS, VALUE_WIDTH)), jnp.bfloat16)
                    for _ in range(2))
    return q, k, v, cotangent


def _forward_and_vjp(shifted, q, k, v, cotangent):
    if shifted:
        (output, shift), pullback = jax.vjp(partial(fused_unnormalized_attention, exp_shift="row_max"), q, k, v)
        return pullback((cotangent, jnp.zeros_like(shift)))
    output, pullback = jax.vjp(fused_unnormalized_attention, q, k, v)
    return pullback(cotangent)


def _forward(shifted, q, k, v):
    return fused_unnormalized_attention(q, k, v, exp_shift="row_max" if shifted else "none")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    device = jax.devices()[0]
    cases = []
    for width in WIDTHS:
        q, k, v, cotangent = _inputs(width)
        for shifted in (False, True):
            case = {"qk_head_dim": width, "exp_shift": "row_max" if shifted else "none"}
            for name, function, arguments in (("forward_and_vjp", partial(_forward_and_vjp, shifted), (q, k, v, cotangent)),
                                               ("forward", partial(_forward, shifted), (q, k, v))):
                timing, _ = time_program(function, arguments, warmup=args.warmup, iterations=args.iterations)
                case[name] = {key: round(timing[key], 3)
                              for key in ("median_ms", "p10_ms", "p90_ms", "compilation_seconds")}
            case["attention_seconds_per_update"] = round(
                case["forward_and_vjp"]["median_ms"] * LAYERS * MICRO_BATCHES / 1000, 3)
            print(case, flush=True)
            cases.append(case)
        del q, k, v, cotangent
    report = {
        "format": "fused-exponential-shift-throughput", "version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "shape": {"batch": BATCH, "time": TIME, "heads": HEADS, "value_head_dim": VALUE_WIDTH, "dtype": "bfloat16",
                  "layers": LAYERS, "micro_batches_per_update": MICRO_BATCHES},
        "cases": cases, "provenance": provenance(device),
    }
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
