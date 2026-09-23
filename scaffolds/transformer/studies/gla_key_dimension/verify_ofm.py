"""Verify extraction parity against the recorded, read-only OFM Git revision.

Run on CPU with the shared scaffold environment and ``--ofm-root PATH``.
This source audit is optional; training has no dependency on an OFM checkout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.attention.gla import GatedLinearAttention


def verify(ofm_root: Path) -> dict:
    source_manifest = Path(__file__).resolve().parents[2] / "attention/backends/gla_sources.json"
    provenance = json.loads(source_manifest.read_text())["ofm"]
    commit = provenance["commit"]

    def source(path):
        raw = subprocess.check_output(["git", "-C", str(ofm_root), "show", f"{commit}:{path}"])
        recorded_hash = provenance["files"].get(path)
        if recorded_hash is not None and hashlib.sha256(raw).hexdigest() != recorded_hash:
            raise ValueError(f"Recorded OFM source hash mismatch: {path}")
        return raw.decode()

    tree = ast.parse(source("models/architectures/gla.py"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name in ("HeadRMSNorm", "GatedLinearAttention")]
    if len(tree.body) != 2:
        raise ValueError("Expected exactly the two recorded OFM attention classes")
    upstream = types.ModuleType("_ofm_gla_parity_reference")
    sys.modules[upstream.__name__] = upstream
    upstream.__dict__.update(jax=jax, jnp=jnp, nnx=nnx)
    exec(compile(source("models/components/positional.py"), "ofm/positional.py", "exec"), upstream.__dict__)
    exec(compile(tree, "ofm/gla.py", "exec"), upstream.__dict__)
    cases = []
    for dtype in (jnp.float32, jnp.bfloat16):
        for rope in (False, True):
            settings = dict(hidden_size=16, num_heads=2, key_dim=8, value_dim=12,
                            gate_rank=4, chunk_size=4, use_rope=rope)
            original = upstream.GatedLinearAttention(**settings, rngs=nnx.Rngs(3))
            extracted = GatedLinearAttention(**settings, rngs=nnx.Rngs(3))
            for model in (original, extracted):
                for _, module in nnx.iter_graph(model):
                    if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
                        module.dtype = dtype
            original_graph, original_state = nnx.split(original, nnx.Param)
            extracted_graph, extracted_state = nnx.split(extracted, nnx.Param)
            for left, right in zip(jax.tree.leaves(original_state), jax.tree.leaves(extracted_state), strict=True):
                np.testing.assert_array_equal(left, right)
            inputs = jax.random.normal(jax.random.key(9), (1, 9, 16)).astype(dtype)
            np.testing.assert_array_equal(original(inputs), extracted(inputs))

            def gradients(graph, state):
                return jax.grad(lambda params: jnp.square(
                    nnx.merge(graph, params)(inputs).astype(jnp.float32)
                ).mean())(state)

            original_grads = gradients(original_graph, original_state)
            extracted_grads = gradients(extracted_graph, extracted_state)
            for left, right in zip(jax.tree.leaves(original_grads), jax.tree.leaves(extracted_grads), strict=True):
                np.testing.assert_array_equal(left, right)
            cases.append({"dtype": dtype.__name__, "rope": rope,
                          "parameters_exact": True, "outputs_exact": True, "gradients_exact": True})
    return {
        "ofm_commit": commit, "source_hash_verified": True,
        "source_files": {name: digest for name, digest in provenance["files"].items()
                         if name in ("models/architectures/gla.py", "models/components/positional.py")},
        "jax_version": jax.__version__, "flax_version": version("flax"),
        "platform": jax.default_backend(),
        "command": [sys.executable, "-m", "scaffolds.transformer.studies.gla_key_dimension.verify_ofm",
                    "--ofm-root", str(ofm_root)],
        "environment": {"JAX_PLATFORMS": "cpu"},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofm-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.ofm_root), indent=2))


if __name__ == "__main__":
    main()
