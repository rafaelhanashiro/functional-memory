"""Per-layer PaTH attention time: FLA's fused kernels against the JAX implementation.

Forward and forward + backward (gradients of every input, coefficients included) at the
production per-GPU shape (32 x 2048 tokens, 16 heads, width 64, BF16 q/k/v, FP32 w, beta,
and coefficients), for softmax and the tanh query polynomial at m = 2 and 4. RoPE-free
references at the same shape: cuDNN softmax and the Mosaic query-polynomial kernel.
Compilation, autotuning, and input generation are outside the timed samples. It then
measures both implementations' errors against FP32 HIGHEST (m = 1-4 and softmax), repeats
the fused forward and backward on fixed inputs to report which results repeat bitwise,
and, given ``--upstream``, compares the bridge with FLA's own PyTorch kernels.

    python -m scaffolds.transformer.benchmarks.benchmark_path --output report.json  # one GPU
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from scaffolds.transformer.benchmarks.benchmark_attention import provenance, time_program, write_report


def operands(batch, length, heads, width, degree, seed=0):
    keys = jax.random.split(jax.random.key(seed), 7)
    q, k = (jax.random.normal(key, (batch, length, heads, width), jnp.float32) for key in keys[:2])
    # Cosine-normalized scores, as with QK normalization: |s| <= 2.
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * math.sqrt(2 * math.sqrt(width))
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True) * math.sqrt(2 * math.sqrt(width))
    v, do = (jax.random.normal(key, (batch, length, heads, width), jnp.bfloat16) for key in keys[2:4])
    w = jax.random.normal(keys[4], (batch, length, heads, width), jnp.float32)
    w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)
    beta = 2 * jax.nn.sigmoid(jax.random.normal(keys[5], (batch, length, heads), jnp.float32))
    taylor = jnp.asarray([1 / math.factorial(r) for r in range(max(degree, 1) + 1)], jnp.float32)
    c = taylor + 0.1 * jax.random.normal(keys[6], (batch, length, heads, taylor.size), jnp.float32)
    return dict(q=q.astype(jnp.bfloat16), k=k.astype(jnp.bfloat16), v=v, w=w, beta=beta, c=c, do=do)


def cases(degree):
    from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention
    from scaffolds.transformer.attention.path import path_attention

    kernel = "softmax" if degree == 0 else "query_polynomial"
    names = ("q", "k", "v", "w", "beta") + (("c",) if degree else ())

    def path(implementation):
        def attend(q, k, v, w, beta, c=None):
            return implementation(q, k, v, w, beta, kernel=kernel, coefficients=c)
        return attend

    result = {"path_fused": (path(fused_path_attention), names),
              "path_jax": (path(path_attention), names)}
    if degree == 0:
        from scaffolds.transformer.attention.softmax import padded_dot_product_attention

        result["rope_free_cudnn_softmax"] = (partial(padded_dot_product_attention, backend="cudnn"), ("q", "k", "v"))
    else:
        from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention

        result["rope_free_mosaic_query_polynomial"] = (query_polynomial_attention, ("q", "k", "v", "c"))
    return result


def benchmark(args) -> dict:
    device = jax.devices()[0]
    report = {"created_utc": datetime.now(UTC).isoformat(), "provenance": provenance(device),
              "shape": {"batch": args.batch, "length": args.length, "heads": args.heads, "width": args.width},
              "timing": {"warmup": args.warmup, "iterations": args.iterations}, "results": []}
    for degree in args.degrees:
        data = operands(args.batch, args.length, args.heads, args.width, degree)
        for name, (function, inputs) in cases(degree).items():
            arguments = tuple(data[key] for key in inputs)
            cotangent = data["do"]

            def forward_backward(*xs, function=function):
                output, pull = jax.vjp(function, *xs)
                return output, pull(cotangent.astype(output.dtype))

            row = {"implementation": name, "kernel": "softmax" if degree == 0 else "query_polynomial",
                   "degree": degree or None}
            for mode, program in (("forward", function), ("forward_backward", forward_backward)):
                timing, _ = time_program(program, arguments, warmup=args.warmup, iterations=args.iterations)
                row[mode] = {key: timing[key] for key in ("median_ms", "p10_ms", "p90_ms", "compilation_seconds")}
            print(f"m={degree} {name:36s} fwd {row['forward']['median_ms']:8.3f} ms  "
                  f"fwd+bwd {row['forward_backward']['median_ms']:8.3f} ms", flush=True)
            report["results"].append(row)
            write_report(args.output, report)
    return report


def accuracy(degrees, *, batch=2, length=2048, heads=4, width=64) -> dict:
    """Relative L2 error of the fused and JAX implementations (BF16 q/k/v) against FP32 HIGHEST."""
    import numpy as np

    from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention
    from scaffolds.transformer.attention.path import path_attention

    report = {"shape": {"batch": batch, "length": length, "heads": heads, "width": width}, "results": {}}
    for degree in sorted(set(degrees) | {1, 3}):
        data = operands(batch, length, heads, width, degree, seed=1)
        data.update(q=(data["q"].astype(jnp.float32) * 2).astype(jnp.bfloat16),  # QK-norm's initial scores, 8 cos(q, k)
                    k=(data["k"].astype(jnp.float32) * 2).astype(jnp.bfloat16))
        names = ("q", "k", "v", "w", "beta") + (("c",) if degree else ())
        kernel = "softmax" if degree == 0 else "query_polynomial"

        def program(implementation, cast=None, precision=None):
            def run(*xs):
                xs = xs if cast is None else tuple(x.astype(cast) for x in xs)
                def attend(*x):
                    return implementation(*x[:5], kernel=kernel, coefficients=x[5] if degree else None,
                                          **({"precision": precision} if precision else {}))
                output, pull = jax.vjp(attend, *xs)
                return output, *pull(data["do"].astype(output.dtype))
            return jax.jit(run)

        arguments = tuple(data[n] for n in names)
        with jax.default_matmul_precision("highest"):
            reference = program(path_attention, jnp.float32, "highest")(*arguments)
        rows = {}
        for name, values in (("fused", program(fused_path_attention)(*arguments)),
                             ("jax", program(path_attention)(*arguments))):
            rows[name] = {label: float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64))
                                       / np.linalg.norm(np.asarray(b, np.float64)))
                          for label, a, b in zip(("output", *(f"d{n}" for n in names)), values, reference)}
        report["results"]["softmax" if degree == 0 else f"m{degree}"] = rows
        print(f"accuracy m={degree}: {rows}", flush=True)
    return report


def repeatability(degrees, repeats: int, shape: dict) -> dict:
    """Run-to-run spread of the fused output and gradients on fixed inputs at the benchmark shape."""
    import numpy as np

    from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention

    report = {}
    for degree in degrees:
        data = operands(shape["batch"], shape["length"], shape["heads"], shape["width"], degree)
        names = ("q", "k", "v", "w", "beta") + (("c",) if degree else ())
        kernel = "softmax" if degree == 0 else "query_polynomial"

        def run(*xs):
            output, pull = jax.vjp(lambda *x: fused_path_attention(*x[:5], kernel=kernel, coefficients=x[5] if degree
                                                                   else None), *xs)
            return output, *pull(data["do"])

        program = jax.jit(run)
        first = [np.asarray(x, np.float32) for x in program(*(data[n] for n in names))]
        rows = {name: {"bitwise_repeats": 0, "max_relative_l2": 0.0} for name in ("output", *(f"d{n}" for n in names))}
        for _ in range(repeats):
            again = [np.asarray(x, np.float32) for x in program(*(data[n] for n in names))]
            for (name, row), a, b in zip(rows.items(), again, first):
                row["bitwise_repeats"] += int(np.array_equal(a, b))
                row["max_relative_l2"] = max(row["max_relative_l2"], float(np.linalg.norm(a - b) / np.linalg.norm(b)))
        report["softmax" if degree == 0 else f"m{degree}"] = {"repeats": repeats, "results": rows}
        print(f"repeatability m={degree}: {rows}", flush=True)
    return report


def upstream_parity(path: Path) -> dict:
    """Compare the bridge with FLA's PyTorch ``parallel_path_attn`` on the same inputs."""
    import numpy as np

    from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention

    data = np.load(path)
    bf16 = lambda name: jnp.asarray(data[name], jnp.bfloat16)
    arguments = (bf16("q"), bf16("k"), bf16("v"), jnp.asarray(data["w"]), jnp.asarray(data["beta"]))

    def forward_backward(*xs):
        output, pull = jax.vjp(partial(fused_path_attention, kernel="softmax"), *xs)
        return output, *pull(bf16("do"))

    ours = jax.jit(forward_backward)(*arguments)
    rows = {}
    for name, value in zip(("output", "dq", "dk", "dv", "dw", "dbeta"), ours):
        mine, theirs = np.asarray(value, np.float32), data[f"fla_{name}"]
        rows[name] = {"bitwise_equal": bool(np.array_equal(mine, theirs)),
                      "relative_l2": float(np.linalg.norm(mine - theirs) / np.linalg.norm(theirs)),
                      "max_abs": float(np.max(np.abs(mine - theirs))), "fla_max_abs": float(np.max(np.abs(theirs)))}
        print(f"bridge vs FLA {name}: {rows[name]}", flush=True)
    return {"shape": list(data["q"].shape), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--degrees", type=int, nargs="+", default=[0, 2, 4], help="0 is softmax")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--upstream", type=Path, help="parity file from benchmark_upstream_path.py")
    parser.add_argument("--repeats", type=int, default=10, help="repeatability runs (0 skips)")
    args = parser.parse_args()
    report = benchmark(args)
    report["accuracy"] = accuracy(args.degrees)
    write_report(args.output, report)
    if args.repeats:
        report["repeatability"] = repeatability(args.degrees, args.repeats, report["shape"])
        write_report(args.output, report)
    if args.upstream is not None:
        report["upstream_parity"] = upstream_parity(args.upstream)
        write_report(args.output, report)


if __name__ == "__main__":
    main()
