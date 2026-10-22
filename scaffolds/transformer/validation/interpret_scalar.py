"""Run scalar kernel algebra through Triton's CPU interpreter and FP64 oracle.

This complements GPU qualification; it does not exercise GPU code generation,
tensor-core rounding, the JAX custom call, or the device memory model.
Run with TRITON_INTERPRET=1 JAX_PLATFORMS=cpu before importing Triton.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from scaffolds.transformer.attention.backends import triton_scalar_kernels as kernels
from scaffolds.transformer.tests.test_triton_scalar import kernel_and_oracle, oracle


def check(name, mode, streaming):
    spec, evaluate = kernel_and_oracle(name)
    rng = np.random.default_rng(31)
    shape = (1, 137, 2, 32)
    q, k, v, go = [torch.from_numpy(rng.normal(size=shape).astype(np.float32)) for _ in range(4)]
    gain = torch.from_numpy(rng.uniform(.6, 1.4, size=shape[-2:]).astype(np.float32))
    out, n = torch.empty_like(v), torch.empty(shape)
    stats = torch.empty((*shape[:3], 2))
    adj, delta = torch.empty(shape), torch.empty(shape[:3])
    dg = torch.empty((1, 2, 2, 32))
    dq, dk, dv = (torch.empty_like(v) for _ in range(3))
    common = dict(T=137, H=2, DQ=32, DV=32, SCALE=32**-.5, MODE=mode,
                  FN=spec.function, PARAMS=spec.parameters, EXP=spec.exponential, PRECISE=False)
    kernels.forward[(2, 2)](q, k, v, gain, out, n, stats, **common,
                           EPS=1e-6, STREAM=streaming, BM=128, BN=64)
    kernels.preprocess[(2, 2)](n, stats, gain, go, adj, delta, dg, T=137, H=2, DV=32, MODE=mode, BM=128)
    kernels.backward_q[(2, 2)](q, k, v, adj, delta, stats, dq, **common, BM=128, BN=32)
    kernels.backward_kv[(2, 2)](q, k, v, adj, delta, stats, dk, dv, **common, BM=32, BN=128)
    actual = (out, dq, dk, dv, dg.sum((0, 2)) if mode == 2 else torch.zeros_like(gain))
    expected = oracle(*(x.float().numpy() for x in (q, k, v, gain, go)), evaluate, {0: "none", 1: "sum", 2: "rms"}[mode], 1e-6)
    errors = {}
    for label, a, b in zip(("output", "dq", "dk", "dv", "dgain"), actual, expected, strict=True):
        a = np.asarray(a.float(), np.float64)
        error = np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)
        if not np.isfinite(a).all() or error >= 1e-3:
            raise AssertionError((name, mode, streaming, label, error))
        errors[label] = float(error)
    return {"dtype": "float32", "function": name, "mode": mode, "streaming": streaming, "relative_l2_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("Set TRITON_INTERPRET=1 before starting this process")
    torch.set_num_threads(2)
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "scope": __doc__, "status": "running", "cases": []}
    for name in ("rbf3", "rbf5", "rbf9", "rbf20", "sine", "poly", "exp",
                 "rbf3_recurrence", "rbf5_recurrence", "rbf9_recurrence", "rbf20_recurrence"):
        for mode in (0, 1, 2):
            if name == "sine" and mode == 1:
                continue
            for streaming in (False, True):
                result = check(name, mode, streaming)
                report["cases"].append(result)
                print(name, mode, streaming, max(result["relative_l2_errors"].values()), flush=True)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
