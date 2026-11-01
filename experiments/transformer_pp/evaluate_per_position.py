"""Per-position validation NLL of retained checkpoints, at their training length and beyond it.

For each model and position t, L_t is the mean over sequences of -log p(x_t | x_<t). The
training-length pass scores the model's own recorded validation set and must reproduce its
recorded validation NLL. The evaluation-length pass scores longer packed rows cut from the
same token stream, with the model's length limit raised; no parameter depends on it.
Nothing is trained or updated.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from scaffolds.records import capture_artifacts, update_record, validate_record
from scaffolds.transformer.configs import Config
from scaffolds.transformer.configs.base import validate_config
from scaffolds.transformer.configs.loader import _update_dataclass
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent

ROOT = Path(__file__).resolve().parents[2]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(spec: dict) -> None:
    root = Path(spec["checkpoint_path"])
    for item in spec["checkpoint_files"]:
        path = root / item["path"]
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint changed: {path}")


def load_model_config(spec: dict, *, max_seq_len: int | None = None) -> Config:
    """The model's frozen recipe, optionally with a raised length limit.

    ``ignored_config_keys`` maps ``section.key`` to its recorded value for settings this
    scaffold does not define; each must hold exactly that value and must not affect the
    forward pass (for example another branch's weight-decay switch at its default).
    """
    raw = yaml.safe_load(Path(spec["config_path"]).read_text())
    config = Config()
    for dotted, expected in (spec.get("ignored_config_keys") or {}).items():
        section, key = dotted.split(".", 1)
        if key in {field.name for field in fields(type(getattr(config, section)))}:
            raise ValueError(f"{dotted} is defined by this scaffold and cannot be ignored")
        if raw.get(section, {}).get(key, "<absent>") != expected:
            raise ValueError(f"{dotted} is {raw.get(section, {}).get(key, '<absent>')!r}, expected {expected!r}")
        raw[section].pop(key)
    if raw["model"].get("attention_mechanism") == "dot_product":
        if "gdn_polynomial_degree" not in {field.name for field in fields(type(config.model))}:
            raw["model"].pop("gdn_polynomial_degree", None)
    _update_dataclass(config, raw, context=spec["config_path"])
    if max_seq_len is not None:
        config.model.max_seq_len = max_seq_len
        config.data.sequence_length = max_seq_len + 1
    validate_config(config)
    return config


def per_position_nll(graph, params, manifest, *, sequence_length: int, sequences: int, microbatch: int):
    """Mean next-token NLL at every position over the first ``sequences`` packed rows."""
    if sequences % microbatch:
        raise ValueError("sequences must be a multiple of the microbatch")

    @jax.jit
    def token_nll(state, tokens):
        logits = nnx.merge(graph, state)(tokens[:, :-1], deterministic=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:])

    batches = PackedBatchIterator(manifest, batch_size=microbatch, sequence_length=sequence_length)
    total = np.zeros(sequence_length - 1, dtype=np.float64)
    for _ in range(sequences // microbatch):
        values = np.asarray(token_nll(params, jnp.asarray(next(batches))), dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite per-token loss")
        total += values.sum(axis=0)
    return total / sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise FileExistsError("Evaluation already completed")
    evaluation = yaml.safe_load((run / "config.yaml").read_text())["evaluation"]
    validate_record(run, checkpoint_hashes=False)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use one GPU")
    runtime = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.uname().nodename, "gpu_type": jax.devices()[0].device_kind, "gpu_count": 1,
        "python_executable": sys.executable, "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "versions": {name: version(name) for name in ("jax", "flax", "optax", "orbax-checkpoint")},
        "environment": {key: os.environ.get(key) for key in ("CUDA_HOME", "XLA_FLAGS", "TRANSFORMER_CUDA_ENVIRONMENT")},
        "optimizer_updates": 0,
    }
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [runtime]})
    lengths = evaluation["lengths"]
    for spec in lengths.values():
        if sha256(spec["manifest"]) != spec["manifest_sha256"]:
            raise ValueError(f"Packed-data manifest changed: {spec['manifest']}")
    outcomes = []
    for model in evaluation["models"]:
        if sha256(model["config_path"]) != model["config_sha256"]:
            raise ValueError(f"Recipe changed: {model['config_path']}")
        print(f"Verifying checkpoint of {model['role']} ({model['run_id']})", flush=True)
        verify_checkpoint(model)
        for name, spec in lengths.items():
            limit = spec["sequence_length"] - 1
            config = load_model_config(model, max_seq_len=None if name == "training" else limit)
            if config.data.sequence_length != spec["sequence_length"]:
                raise ValueError(f"{model['role']} trains at {config.data.sequence_length}, not {spec['sequence_length']}")
            graph, params = restore_parent(config, Path(model["checkpoint_path"]))
            started = time.perf_counter()
            curve = per_position_nll(graph, params, spec["manifest"], sequence_length=spec["sequence_length"],
                                     sequences=spec["sequences"], microbatch=spec["microbatch_size"])
            nll = float(curve.mean())
            row = {"role": model["role"], "source_run_id": model["run_id"], "checkpoint_step": model["step"],
                   "length": name, "sequence_length": spec["sequence_length"], "sequences": spec["sequences"],
                   "prediction_tokens": spec["sequences"] * limit, "attention_backend": config.model.attention_backend,
                   "nll": nll, "perplexity": math.exp(nll), "seconds": round(time.perf_counter() - started, 1)}
            if name == "training":
                error = abs(nll - model["recorded_validation_nll"])
                row["recorded_validation_nll"] = model["recorded_validation_nll"]
                row["replay_absolute_error"] = error
                if error > evaluation["replay_nll_atol"]:
                    raise ValueError(f"{model['role']} replay differs: {nll} versus {model['recorded_validation_nll']}")
            else:
                trained = model["training_sequence_length"] - 1
                row["nll_within_training_length"] = float(curve[:trained].mean())
                row["nll_beyond_training_length"] = float(curve[trained:].mean()) if curve.size > trained else None
            print(json.dumps(row), flush=True)
            outcomes.append({key: value for key, value in row.items()})
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({**row, "per_position_nll": curve.tolist()}, sort_keys=True, allow_nan=False) + "\n")
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_updates": 0, "measurements": outcomes,
               "interpretation": evaluation["interpretation"]}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "evaluation_outcome": outcome})
    print("Per-position evaluation complete", flush=True)


if __name__ == "__main__":
    main()
