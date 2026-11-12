"""Throughput and register use of the Mosaic transformed query-polynomial kernels.

``kernels`` times each Mosaic kernel of the query polynomial (the control) and of its softplus and abs transformed
families (``common.QUERY_SOFTPLUS``, ``common.QUERY_ABS``) on the GPU it runs on, H100 (forward, dQ, dK/dV) or
B300 (forward, dQ, dK, dV with the stages the entry points use), at the production per-GPU shape of the
reference shape: 16 sequences of 2,048 tokens, 14 heads, Q/K and value widths 64, degree 8 (``--degree``), BF16
weight products. ``attention`` times the public entry points (warmed forward, and forward plus the VJP of every
operand). ``registers`` compiles every kernel once with ptxas's verbose report and records its registers, spill
bytes and PTX ISA version; run it in its own process (a kernel compiled earlier is not compiled again). Run under
the launcher's runtime, which pins ``CUDA_HOME`` to the locked toolkit; reports record the toolkit
(``benchmark_score_bias``).

    python -m scaffolds.transformer.benchmarks.benchmark_score_transform kernels --output k.json
    python -m scaffolds.transformer.benchmarks.benchmark_score_transform attention --output a.json
    python -m scaffolds.transformer.benchmarks.benchmark_score_transform registers --output r.json
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

from scaffolds.transformer.benchmarks.benchmark_score_bias import SHAPE, _ptx_isa, _timed, _toolkit

TRANSFORMS = ("identity", "softplus", "abs")


def _operands(degree, seed=0):
    """Production-shape inputs: unit-scale scores and Taylor-scaled coefficients, as benchmark_score_bias."""
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    tokens = tuple(SHAPE[name] for name in ("batch", "time", "heads"))
    q, k = (jnp.asarray(rng.standard_normal((*tokens, SHAPE["width"])), jnp.bfloat16) for _ in range(2))
    v, do = (jnp.asarray(rng.standard_normal((*tokens, SHAPE["value_width"])), jnp.bfloat16) for _ in range(2))
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
    c = jnp.asarray(taylor * (1 + .5 * np.tanh(rng.standard_normal((*tokens, degree + 1)))), jnp.float32)
    return q, k, v, do, c


def _family(transform):
    from scaffolds.transformer.attention.backends.mosaic import api, common

    return common.QUERY if transform == "identity" else api.TRANSFORM_FAMILY[transform]


def _kernel_calls(degree):
    """{transform: {pass: call}} of each Mosaic kernel of the GPU, with the stages the entry points use."""
    from scaffolds.transformer.attention.backends.mosaic import api

    arch, kernels = api._kernels()
    q, k, v, do, c = _operands(degree)
    vectors = api._coefficient_vectors(c)
    calls = {}
    for transform in TRANSFORMS:
        options = dict(family=_family(transform), degree=degree, scale=1 / math.sqrt(q.shape[-1]), precision="bf16")
        if arch == "hopper":
            calls[transform] = {"forward": lambda o=options: kernels.forward(q, k, v, vectors, **o),
                                "dq": lambda o=options: kernels.dq(q, k, v, do, vectors, **o),
                                "dkv": lambda o=options: kernels.dkv(q, k, v, do, vectors, **o)}
        else:
            stages = 2 if q.shape[-1] > 64 else 3  # as api._backward
            calls[transform] = {
                "forward": lambda o=options: kernels.forward(q, k, v, vectors, **o),
                "dq": lambda o=options: kernels.dq(q, k, v, do, vectors, **o, stages=stages),
                "dk": lambda o=options: kernels.dkv(q, k, v, do, vectors, which="dk", **o, stages=stages),
                "dv": lambda o=options: kernels.dkv(q, k, v, do, vectors, which="dv", **o)}
    return arch, calls


def kernels(degree, repeats):
    import numpy as np

    arch, calls = _kernel_calls(degree)
    results = {}
    for transform, passes in calls.items():
        results[transform] = {name: _timed(call, repeats=repeats) for name, call in passes.items()}
        print(transform, {name: round(float(np.median(x)), 3) for name, x in results[transform].items()}, flush=True)
    return {"architecture": arch, "kernels_ms": results}


def attention(degree, repeats):
    import jax
    import numpy as np
    from scaffolds.transformer.attention.backends.mosaic import api

    q, k, v, do, c = _operands(degree)
    results = {}
    for transform in TRANSFORMS:
        function = (api.query_polynomial_attention if transform == "identity" else
                    lambda q, k, v, c, t=transform: api.transformed_query_polynomial_attention(q, k, v, c, transform=t))

        def backward(*x, function=function):
            return jax.vjp(function, *x)[1](do)

        results[transform] = {"forward_ms": _timed(jax.jit(function), q, k, v, c, repeats=repeats),
                              "forward_backward_ms": _timed(jax.jit(backward), q, k, v, c, repeats=repeats)}
        print(transform, {key: round(float(np.median(value)), 3) for key, value in results[transform].items()},
              flush=True)
    return {"cases": results}


def registers(degree, dump=None):
    """ptxas's registers and spill bytes of every kernel, from a dump directory of this process (``dump``, which
    keeps each kernel's PTX and ptxas report, else a temporary directory)."""
    import jax

    dump = Path(tempfile.mkdtemp(prefix="score-transform-ptxas-") if dump is None else dump)
    dump.mkdir(parents=True, exist_ok=True)
    os.environ["MOSAIC_GPU_DUMP_PTXAS"] = "1"
    os.environ["MOSAIC_GPU_DUMP_TO"] = str(dump)
    pattern = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
                         r"|Used (\d+) registers")
    arch, calls = _kernel_calls(degree)
    results = {}
    for transform, passes in calls.items():
        results[transform] = {}
        for name, call in passes.items():
            before = set(dump.glob("*.ptxas"))
            jax.block_until_ready(call())
            reports = sorted(set(dump.glob("*.ptxas")) - before)
            if len(reports) != 1:
                raise RuntimeError(f"{transform} {name}: expected one new ptxas report, found {len(reports)}")
            stats = {"report": reports[0].name, "ptx_isa": _ptx_isa(reports[0])}
            for match in pattern.finditer(reports[0].read_text()):
                if match.group(4):
                    stats["registers"] = int(match.group(4))
                else:
                    stats.update(stack_bytes=int(match.group(1)), spill_store_bytes=int(match.group(2)),
                                 spill_load_bytes=int(match.group(3)))
            results[transform][name] = stats
        print(transform, results[transform], flush=True)
    return {"architecture": arch, "ptxas": results}


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
    report = {"format": "score-transform-attention-benchmark", "mode": args.mode, "device": device.device_kind,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
              "jax": jax.__version__, "toolkit": _toolkit(), "shape": {**SHAPE, "degree": args.degree},
              "precision": "bf16",
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
