"""Check that the GDN bridge on ``triton_port`` launches exactly what the previous bridge did.

Loads ``attention/backends/fla_gated_delta_rule.py`` from a reference commit beside the
current module and, for the tied and independent-gate operations (forward and backward,
BF16 and FP32), compares every Triton custom call in their lowered programs: the compiled
kernels and autotune candidates, grids, parameters, output shapes, zeroed and aliased
buffers. Then it runs both on the same inputs and compares outputs and gradients
bitwise. Run on one GPU:

    python -m scaffolds.transformer.validation.compare_gdn_port --commit 7fb44dc --output report.json
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends import fla_gated_delta_rule as current

ROOT = Path(__file__).resolve().parents[3]
MODULE = "scaffolds/transformer/attention/backends/fla_gated_delta_rule.py"


def previous_module(commit: str):
    source = subprocess.check_output(["git", "show", f"{commit}:{MODULE}"], cwd=ROOT, text=True)
    spec = importlib.util.spec_from_loader("previous_fla_gated_delta_rule", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, f"{commit}:{MODULE}", "exec"), module.__dict__)
    sys.modules[spec.name] = module
    return module


def launches(program, *arguments):
    """Every Triton custom call in the lowered program, without SSA names or source locations.

    Each call's backend configuration is jaxlib's serialized kernel call: the compiled
    kernel (or every autotune candidate), grid, parameters, zeroed and aliased buffers.
    """
    module = jax.jit(program).lower(*arguments).compiler_ir("stablehlo")
    text = module.operation.get_asm(enable_debug_info=False)
    return [line.split("{", 1)[1].strip() for line in text.splitlines() if "custom_call @triton_kernel_call" in line]


def _operands(dtype, *, batch=2, length=512, heads=4, key_dim=64, value_dim=64, seed=0):
    rng = np.random.default_rng(seed)
    q, k = (rng.standard_normal((batch, length, heads, key_dim)) for _ in range(2))
    q, k = (z / np.linalg.norm(z, axis=-1, keepdims=True) for z in (q, k))
    v = rng.standard_normal((batch, length, heads, value_dim))
    g = -np.log1p(np.exp(rng.standard_normal((batch, length, heads)) * 0.5 - 4))
    beta, erase = (1 / (1 + np.exp(-rng.standard_normal((batch, length, heads)))) for _ in range(2))
    cotangent = rng.standard_normal((batch, length, heads, value_dim))
    cast = lambda z, d=dtype: jnp.asarray(z, d)
    return (cast(q), cast(k), cast(v), cast(g, jnp.float32), cast(beta), cast(erase)), cast(cotangent)


def programs(module, independent, cotangent, scale):
    operation = module.fla_independent_gated_delta_rule if independent else module.fla_gated_delta_rule

    def run(*arguments):
        output, pull = jax.vjp(lambda *x: operation(*x, scale), *arguments)
        return output, pull(cotangent)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = previous_module(args.commit)
    report = {"created_utc": datetime.now(UTC).isoformat(), "reference_commit": args.commit,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "device_kind": jax.devices()[0].device_kind, "cases": []}
    passed = True
    for dtype in (jnp.bfloat16, jnp.float32):
        for independent in (False, True):
            operands, cotangent = _operands(dtype)
            operands = operands if independent else operands[:5]
            scale = operands[0].shape[-1] ** -0.5
            new, old = (programs(module, independent, cotangent, scale) for module in (current, previous))
            new_launches, old_launches = launches(new, *operands), launches(old, *operands)
            same_launches = new_launches == old_launches
            new_values, old_values = (jax.tree.leaves(jax.jit(program)(*operands)) for program in (new, old))
            bitwise = [bool(np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32), equal_nan=True))
                       for a, b in zip(new_values, old_values, strict=True)]
            case = {"dtype": jnp.dtype(dtype).name, "independent": independent, "launches": len(new_launches),
                    "identical_launches": same_launches, "bitwise_equal_outputs_and_gradients": bitwise}
            if not same_launches:
                case["differing_launches"] = sum(a != b for a, b in zip(new_launches, old_launches))
                case["previous_launches"] = len(old_launches)
            passed &= same_launches and all(bitwise)
            report["cases"].append(case)
            print(json.dumps(case, default=str), flush=True)
    report["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
