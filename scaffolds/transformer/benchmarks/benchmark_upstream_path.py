"""FLA's own ``parallel_path_attn`` in CUDA PyTorch (the upstream runtime), on one GPU.

Times forward and forward + backward (gradients of q, k, v, w, beta) at the production
per-GPU shape with BF16 q/k/v and FP32 w, beta, as ``benchmark_path`` times the JAX bridge,
and writes FLA's output and gradients on fixed inputs for ``benchmark_path --upstream``.
Runs in ``benchmarks/upstream_runtime`` (CUDA 13: B300 on Kennel); no scaffold code is used.

    python scaffolds/transformer/benchmarks/benchmark_upstream_path.py --output timing.json --parity parity.npz
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def operands(batch, length, heads, width, seed):
    rng = np.random.default_rng(seed)
    q, k = (rng.standard_normal((batch, length, heads, width)) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) * math.sqrt(2 * math.sqrt(width)) for x in (q, k))
    v, do = (rng.standard_normal((batch, length, heads, width)) for _ in range(2))
    w = rng.standard_normal((batch, length, heads, width))
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    beta = 2 / (1 + np.exp(-rng.standard_normal((batch, length, heads))))
    return dict(q=q.astype(np.float32), k=k.astype(np.float32), v=v.astype(np.float32), w=w.astype(np.float32),
                beta=beta.astype(np.float32), do=do.astype(np.float32))


def tensors(torch, data):
    bf16 = {"q", "k", "v", "do"}
    return {name: torch.tensor(value, device="cuda").to(torch.bfloat16 if name in bf16 else torch.float32)
            for name, value in data.items()}


def run(torch, attention, x, backward):
    inputs = [x[name].detach().requires_grad_(backward) for name in ("q", "k", "v", "w", "beta")]
    output, _ = attention(*inputs, scale=x["q"].shape[-1] ** -0.5)
    if not backward:
        return output, ()
    gradients = torch.autograd.grad(output, inputs, x["do"])
    return output, gradients


def timed(torch, function, *, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e3)
    return {"median_ms": float(np.median(samples)), "p10_ms": float(np.quantile(samples, .1)),
            "p90_ms": float(np.quantile(samples, .9))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    import torch
    from fla.ops.path_attn.parallel import parallel_path_attn

    shape = dict(batch=32, length=2048, heads=16, width=64)
    x = tensors(torch, operands(**shape, seed=0))
    report = {"created_utc": datetime.now(UTC).isoformat(), "shape": shape,
              "device": torch.cuda.get_device_name(0), "node": os.environ.get("SLURMD_NODENAME"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "packages": {name: metadata.version(name) for name in ("torch", "triton", "flash-linear-attention")},
              "upstream": json.loads(metadata.distribution("flash-linear-attention").read_text("direct_url.json") or "{}"),
              "timing": {"warmup": args.warmup, "iterations": args.iterations}}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        report[mode] = timed(torch, lambda: run(torch, parallel_path_attn, x, backward),
                             warmup=args.warmup, iterations=args.iterations)
        print(f"FLA parallel_path_attn {mode}: {report[mode]['median_ms']:.3f} ms", flush=True)
    parity = operands(batch=2, length=2048, heads=4, width=64, seed=1)
    output, gradients = run(torch, parallel_path_attn, tensors(torch, parity), True)
    results = {f"fla_{name}": value.float().cpu().numpy() for name, value in
               zip(("output", "dq", "dk", "dv", "dw", "dbeta"), (output, *gradients))}
    args.parity.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.parity, **parity, **results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
