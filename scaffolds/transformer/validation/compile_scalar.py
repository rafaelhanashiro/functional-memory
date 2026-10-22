"""Export the complete scalar VJP to CUDA MLIR on a CPU host, without execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
from jax import export

from scaffolds.launch.frozen import frozen_hash
from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge
from scaffolds.transformer.attention.backends.triton_scalar import _attention, Tuning
from scaffolds.transformer.tests.test_triton_scalar import kernel_and_oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Cross-compilation specifies the target explicitly; no GPU is selected.
    bridge._get_cuda_compute_capability = lambda device: 90
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {"source_commit": source,
              "frozen_execution_sha256": frozen_hash(root, source, "attention_scaling", "transformer"),
              "status": "running", "target": "sm90", "scope": __doc__, "cases": []}
    for dtype in (jnp.bfloat16, jnp.float32):
        for name in ("rbf3", "rbf5", "rbf9", "rbf20", "sine", "poly", "exp"):
            kernel, _ = kernel_and_oracle(name)
            for mode in (0, 1, 2):
                if name == "sine" and mode == 1:
                    continue
                def operation(q, k, v, gain):
                    return _attention(q, k, v, gain, kernel, mode, mode != 0, False, 1e-6, Tuning())
                def joint(q, k, v, gain, go):
                    y, vjp = jax.vjp(operation, q, k, v, gain)
                    return (y, *vjp(go))
                aval = jax.ShapeDtypeStruct((1, 137, 2, 32), dtype)
                gain = jax.ShapeDtypeStruct((2, 32), jnp.float32)
                exported = export.export(jax.jit(joint), platforms=("cuda",),
                    disabled_checks=(export.DisabledSafetyCheck.custom_call("triton_kernel_call"),))(
                        aval, aval, aval, gain, aval)
                mlir = exported.mlir_module()
                assert "tensor<1536xui8>" in mlir  # Four CTAs, three descriptors each.
                assert len(exported.out_avals) == 5  # Scratch must remain private.
                report["cases"].append({"dtype": str(jnp.dtype(dtype)), "function": name, "mode": mode,
                    "mlir_sha256": hashlib.sha256(mlir.encode()).hexdigest()})
                print(dtype, name, mode, "compiled", flush=True)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
