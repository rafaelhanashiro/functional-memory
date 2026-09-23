"""Verify initialization against the independent in-repository OFM port."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import gc
import hashlib
import json
import os
import importlib
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


REFERENCE_COMMIT = "c2b0d6e5514995cbe7b5e564f902201cfde1ba4e"
OFM_SOURCE_COMMIT = "32ea743b7b3aed1d631116dfaa3c4b85101baf32"
ROOT = Path(__file__).resolve().parents[4]


def _parameter_fingerprint(build_model, model_config, seed: int) -> dict:
    """Build and fingerprint one model, releasing all its arrays on return."""
    import jax
    import numpy as np
    from flax import nnx

    rng = jax.random.PRNGKey(seed)
    model = build_model(
        model_config,
        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)),
    )
    parameters = nnx.state(model, nnx.Param)
    digest = hashlib.sha256()
    parameter_count = 0
    tensor_count = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]:
        array = np.asarray(leaf)
        if array.dtype != np.float32:
            raise ValueError(f"Initial parameter {path} must be FP32; got {array.dtype}")
        digest.update(str(path).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
        parameter_count += array.size
        tensor_count += 1
    return {
        "parameter_count": int(parameter_count),
        "parameter_tensor_count": tensor_count,
        "parameter_sha256": digest.hexdigest(),
    }


def verify_initialization(config_path: str | Path) -> dict:
    import jax

    from scaffolds.transformer.configs.loader import load_config
    from scaffolds.transformer.models import build_model

    if jax.default_backend() != "cpu":
        raise RuntimeError("Initialization verification must run on the CPU")
    from scaffolds.transformer.models import model_parameter_count

    config = load_config(config_path)
    expected_parameter_count = model_parameter_count(config.model)
    # Export immutable independent source, not a current compatibility adapter.
    with tempfile.TemporaryDirectory(prefix="ofm-initialization-reference-") as directory:
        archive = subprocess.check_output(["git", "archive", REFERENCE_COMMIT, "experiments/key_dimension"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(directory, filter="data")
        sys.path.insert(0, directory)
        try:
            reference_configs = importlib.import_module("experiments.key_dimension.configs.base")
            reference_models = importlib.import_module("experiments.key_dimension.models")
        finally:
            sys.path.remove(directory)
    ReferenceModelConfig = reference_configs.ModelConfig
    REFERENCE_OFM_SOURCE_COMMIT = reference_configs.OFM_SOURCE_COMMIT
    build_reference = reference_models.build_model
    reference_values = {
        field.name: getattr(config.model, field.name)
        for field in fields(ReferenceModelConfig)
        if field.name != "attention_backend"
    }
    reference_config = ReferenceModelConfig(
        **reference_values, attention_backend="cudnn",
    )
    # Initialization consumes no attention operations; CPU blocked selection is
    # sufficient even if the production recipe requests automatic GPU dispatch.
    from dataclasses import replace
    actual = _parameter_fingerprint(build_model, replace(config.model, attention_backend="blocked"), config.train.seed)
    # Do not retain two 1 GB parameter trees (or their initializer executables)
    # at once. _parameter_fingerprint returns only scalars and a digest.
    gc.collect()
    jax.clear_caches()
    reference = _parameter_fingerprint(
        build_reference, reference_config, config.train.seed,
    )
    gc.collect()
    jax.clear_caches()

    for name, fingerprint in (("experiment", actual), ("reference", reference)):
        if fingerprint["parameter_count"] != expected_parameter_count:
            raise AssertionError(
                f"{name} parameter count {fingerprint['parameter_count']} "
                f"does not match the recipe count {expected_parameter_count}"
            )
    if actual != reference:
        raise AssertionError(
            f"Initial parameters differ from the frozen OFM port: "
            f"experiment={actual!r}, reference={reference!r}"
        )
    return {
        "status": "passed",
        "backend": "cpu",
        "seed": config.train.seed,
        "parameter_count": actual["parameter_count"],
        "parameter_sha256": actual["parameter_sha256"],
        "experiment": actual,
        "reference": reference,
        "experiment_model_config": asdict(config.model),
        "reference_model_config": asdict(reference_config),
        "reference_module": "experiments.key_dimension.models",
        "reference_source_commit": REFERENCE_COMMIT,
        "reference_ofm_source_commit": REFERENCE_OFM_SOURCE_COMMIT,
        "canonical_ofm_run_source_commit": OFM_SOURCE_COMMIT,
        "hash_format": (
            "SHA-256 of each JAX flattened parameter path, shape, and C-order "
            "FP32 bytes, in tree order"
        ),
        "scaffold_genealogy": (
            "The key_dimension scaffold ports OFM's Transformer; "
            "polynomial_attention preserves its parameter constructors, and "
            "unnormalized_attention derives from that scaffold. This check "
            "establishes exact initial "
            "parameter identity with the independent in-repository "
            "key_dimension port exported from its frozen Git commit, without "
            "importing the adjacent OFM repository."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Set this before importing JAX or either model family; GPU allocations may
    # invoke this worker as a subprocess while retaining JAX_PLATFORMS=cuda.
    os.environ["JAX_PLATFORMS"] = "cpu"
    result = verify_initialization(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
