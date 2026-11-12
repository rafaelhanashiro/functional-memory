"""Export the complete scalar VJP to CUDA MLIR on a CPU host, without execution."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
from jax import export

from scaffolds.launch.frozen import frozen_hash, changes
from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge
from scaffolds.transformer.attention.backends.triton_scalar import _attention, Tuning
from scaffolds.transformer.tests.test_triton_scalar import kernel_and_oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", action="store_true", help="Compile BF16 training shapes and RMS variants")
    args = parser.parse_args()
    # Cross-compilation specifies the target explicitly; no GPU is selected.
    bridge._get_cuda_compute_capability = lambda device: 90
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if changes(root, source, "attention_scaling", "transformer"):
        raise ValueError("CUDA export requires committed, unchanged scaffold source")
    report = {"source_commit": source,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_execution_sha256": frozen_hash(root, source, "attention_scaling", "transformer"),
              "status": "running", "target": "sm90", "scope": __doc__, "cases": []}
    shape = (16, 2048, 16, 64) if args.training else (1, 137, 2, 32)
    dtype_choices = (jnp.bfloat16,) if args.training else (jnp.bfloat16, jnp.float32)
    names = ("rbf3", "rbf5", "rbf9", "rbf20", "sine", "poly3", "poly8", "exp") if args.training else (
        "rbf3", "rbf5", "rbf9", "rbf20", "sine", "poly", "exp")
    names += ("rbf3_recurrence", "rbf5_recurrence", "rbf9_recurrence", "rbf20_recurrence")
    variants = ((0, False, False), (1, True, False), (2, False, False), (2, True, False), (2, True, True)) if args.training else (
        (0, False, False), (1, True, False), (2, True, False))
    report["shape"] = shape
    report['tuning'] = asdict(Tuning())
    block_q = Tuning().forward_q
    scratch_size = 384 * ((shape[1] + block_q - 1) // block_q) * shape[0] * shape[2]
    for dtype in dtype_choices:
        for name in names:
            if args.training:
                from scaffolds.transformer.benchmarks.benchmark_scalar import case
                kernel, reference, _ = case(name.removesuffix("_recurrence"))
                if name.endswith("_recurrence"):
                    from scaffolds.transformer.attention.backends.triton_scalar import uniform_rbf_kernel
                    kernel = uniform_rbf_kernel(reference["score_kernel"])
            else:
                kernel, _ = kernel_and_oracle(name)
            for mode, streaming, precise in variants:
                if name == "sine" and mode == 1:
                    continue
                def operation(q, k, v, gain):
                    return _attention(q, k, v, gain, kernel, mode, streaming, precise, 1e-6, Tuning())
                def joint(q, k, v, gain, go):
                    y, vjp = jax.vjp(operation, q, k, v, gain)
                    return (y, *vjp(go))
                aval = jax.ShapeDtypeStruct(shape, dtype)
                gain = jax.ShapeDtypeStruct(shape[-2:], jnp.float32)
                exported = export.export(jax.jit(joint), platforms=("cuda",),
                    disabled_checks=(export.DisabledSafetyCheck.custom_call("triton_kernel_call"),))(
                        aval, aval, aval, gain, aval)
                mlir = exported.mlir_module()
                assert f"tensor<{scratch_size}xui8>" in mlir
                assert len(exported.out_avals) == 5  # Scratch must remain private.
                report["cases"].append({"dtype": str(jnp.dtype(dtype)), "function": name, "mode": mode,
                    "streaming": streaming, "precise": precise,
                    "mlir_sha256": hashlib.sha256(mlir.encode()).hexdigest()})
                print(dtype, name, mode, streaming, precise, "compiled", flush=True)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
