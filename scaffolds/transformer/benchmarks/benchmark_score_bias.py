"""Throughput and register use of the H100 Mosaic per-degree score-bias kernels.

``kernels`` times each Mosaic kernel (forward, dQ, dK/dV) of the four score-bias families (no gate
or the per-order endpoint gate, times a row or a pair shift) and of their controls (the query
polynomial and the per-order endpoint gate without shifts) at the production per-GPU shape: 16
sequences of 2,048 tokens, 14 heads, Q/K and value widths 64, degree 8 (``--degree``).
``attention`` times the attention operation alone through the public entry points (warmed forward,
and forward plus the VJP of every operand). ``registers`` compiles every kernel once with ptxas's
verbose report (``MOSAIC_GPU_DUMP_PTXAS``) and records its registers and spill bytes; run it in its
own process, since a kernel compiled earlier in the process is not compiled (or reported) again.

    python -m scaffolds.transformer.benchmarks.benchmark_score_bias kernels --output k.json
    python -m scaffolds.transformer.benchmarks.benchmark_score_bias attention --output a.json
    python -m scaffolds.transformer.benchmarks.benchmark_score_bias registers --output r.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

SHAPE = {"batch": 16, "time": 2048, "heads": 14, "width": 64, "value_width": 64}
FAMILIES = (("none", "row"), ("none", "pair"), ("endpoint", "row"), ("endpoint", "pair"))


def _timed(function, *args, repeats):
    import jax

    jax.block_until_ready(function(*args))  # compile and warm
    jax.block_until_ready(function(*args))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append((time.perf_counter() - started) * 1e3)
    return samples


def _operands(degree, seed=0):
    """Production-shape inputs: unit-scale scores, Taylor-scaled coefficients, per-order endpoint
    gate inputs as in benchmark_decay, row shifts and pair inputs of order one."""
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    batch, time_, heads, width = (SHAPE[name] for name in ("batch", "time", "heads", "width"))
    orders = degree + 1
    tokens = (batch, time_, heads)
    q, k = (jnp.asarray(rng.standard_normal((*tokens, width)), jnp.bfloat16) for _ in range(2))
    v, do = (jnp.asarray(rng.standard_normal((*tokens, 64)), jnp.bfloat16) for _ in range(2))
    taylor = np.array([1 / math.factorial(r) for r in range(orders)])
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    return {
        "q": q, "k": k, "v": v, "do": do,
        "c": f32(taylor * (1 + .5 * np.tanh(rng.standard_normal((*tokens, orders))))),
        "gate": (f32(rng.uniform(-5, -1, (*tokens, orders))), f32(.5 * rng.standard_normal((*tokens, orders))),
                 f32(rng.uniform(-1, 1, (heads, orders, 1)))),
        "row": {"row_shift": f32(.5 * rng.standard_normal((*tokens, degree)))},
        "pair": {"query_shift_input": f32(rng.standard_normal((*tokens, degree))),
                 "key_shift_input": f32(rng.standard_normal((*tokens, degree))),
                 "shift_offset": f32(.5 * rng.standard_normal((heads, degree))),
                 "shift_amplitude": f32(1 + .3 * rng.standard_normal((heads, degree)))},
    }


def _operations(degree):
    """{name: (function, operands)} of the public entry points: four families and two controls."""
    from scaffolds.transformer.attention.backends.mosaic import api, score_bias

    x = _operands(degree)
    base = (x["q"], x["k"], x["v"], x["c"])
    cases = {"query": (api.query_polynomial_attention, base),
             "per_order_endpoint_decay": (api.per_order_endpoint_decay_attention, (*base, *x["gate"]))}
    for gate, shift in FAMILIES:
        names = tuple(x[shift])
        if gate == "none":
            function = lambda q, k, v, c, *s, names=names: score_bias.score_bias_query_polynomial_attention(
                q, k, v, c, **dict(zip(names, s)))
            operands = (*base, *x[shift].values())
        else:
            function = lambda q, k, v, c, u, w, a, *s, names=names: (
                score_bias.score_bias_per_order_endpoint_attention(q, k, v, c, u, w, a, **dict(zip(names, s))))
            operands = (*base, *x["gate"], *x[shift].values())
        cases[score_bias.score_bias_family(gate, shift)] = (function, operands)
    return cases, x["do"]


def _kernel_calls(degree):
    """{family: {pass: call}} of each Mosaic H100 kernel with packed operands."""
    import jax
    from scaffolds.transformer.attention.backends.mosaic import api, common, hopper, score_bias

    if api._kernels()[0] != "hopper":
        raise SystemExit("These kernels run on H100 only")
    x = _operands(degree)
    q, k, v, do, c = (x[name] for name in ("q", "k", "v", "do", "c"))
    log_rate = x["gate"][2][..., 0]
    rate = -jax.numpy.exp(log_rate)
    packed = {common.QUERY: (api._coefficient_vectors(c), None, None),
              common.PER_ORDER_ENDPOINT_DECAY: api._per_order_endpoint_operands(c, *x["gate"][:2], rate)}
    for gate, shift in FAMILIES:
        family = score_bias.score_bias_family(gate, shift)
        operands = {"row": {"row_shift": x["row"]["row_shift"]},
                    "pair": dict(zip(("shift_query", "shift_key", "shift_offset", "shift_amplitude"),
                                     x["pair"].values()))}[shift]
        if gate == "endpoint":
            operands |= {"gate_query": x["gate"][0], "gate_key": x["gate"][1], "gate_rate": rate}
        packed[family] = score_bias._pack(family, c, operands)
    calls = {}
    for family, (vectors, key_vectors, heads) in packed.items():
        options = dict(family=family, degree=degree, scale=1 / math.sqrt(q.shape[-1]), precision="bf16")
        calls[family] = {
            "forward": lambda o=options, a=(vectors, key_vectors, heads): hopper.forward(q, k, v, *a, **o),
            "dq": lambda o=options, a=(vectors, key_vectors, heads): hopper.dq(q, k, v, do, *a, **o),
            "dkv": lambda o=options, a=(vectors, key_vectors, heads): hopper.dkv(q, k, v, do, *a, **o)}
    return calls


def kernels(degree, repeats):
    import numpy as np

    results = {}
    for family, passes in _kernel_calls(degree).items():
        results[family] = {name: _timed(call, repeats=repeats) for name, call in passes.items()}
        print(family, {name: round(float(np.median(x)), 3) for name, x in results[family].items()}, flush=True)
    return {"kernels_ms": results}


def attention(degree, repeats):
    import jax
    import numpy as np

    cases, do = _operations(degree)
    results = {}
    for name, (function, operands) in cases.items():
        def backward(*x, function=function):
            return jax.vjp(function, *x)[1](do)

        results[name] = {"forward_ms": _timed(jax.jit(function), *operands, repeats=repeats),
                         "forward_backward_ms": _timed(jax.jit(backward), *operands, repeats=repeats)}
        print(name, {key: round(float(np.median(value)), 3) for key, value in results[name].items()}, flush=True)
    return {"cases": results}


def registers(degree, dump=None):
    """ptxas's registers and spill bytes of every kernel, from a dump directory of this process
    (``dump``, which keeps each kernel's PTX and ptxas report, else a temporary directory)."""
    import jax

    dump = Path(tempfile.mkdtemp(prefix="score-bias-ptxas-") if dump is None else dump)
    dump.mkdir(parents=True, exist_ok=True)
    os.environ["MOSAIC_GPU_DUMP_PTXAS"] = "1"
    os.environ["MOSAIC_GPU_DUMP_TO"] = str(dump)
    pattern = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
                         r"|Used (\d+) registers")
    results = {}
    for family, passes in _kernel_calls(degree).items():
        results[family] = {}
        for name, call in passes.items():
            before = set(dump.glob("*.ptxas"))
            jax.block_until_ready(call())
            reports = sorted(set(dump.glob("*.ptxas")) - before)
            if len(reports) != 1:
                raise RuntimeError(f"{family} {name}: expected one new ptxas report, found {len(reports)}")
            stats = {"report": reports[0].name}
            for match in pattern.finditer(reports[0].read_text()):
                if match.group(4):
                    stats["registers"] = int(match.group(4))
                else:
                    stats.update(stack_bytes=int(match.group(1)), spill_store_bytes=int(match.group(2)),
                                 spill_load_bytes=int(match.group(3)))
            results[family][name] = stats
        print(family, results[family], flush=True)
    return {"ptxas": results, "note": "registers is the launch allocation (384 threads); compute warpgroups "
                                      "raise theirs to 232 with setmaxnreg"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("kernels", "attention", "registers"))
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dump", type=Path, help="registers: keep each kernel's PTX and ptxas report here")
    args = parser.parse_args()
    import jax

    device = jax.local_devices()[0]
    report = {"format": "score-bias-attention-benchmark", "mode": args.mode, "device": device.device_kind,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
              "jax": jax.__version__, "shape": {**SHAPE, "degree": args.degree}, "precision": "bf16",
              "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "source_dirty": bool(subprocess.run(["git", "status", "--porcelain", "scaffolds"], capture_output=True,
                                                  text=True).stdout.strip())}
    if args.mode == "kernels":
        report.update(kernels(args.degree, args.repeats))
    elif args.mode == "attention":
        report.update(attention(args.degree, args.repeats))
    else:
        report.update(registers(args.degree, args.dump))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
