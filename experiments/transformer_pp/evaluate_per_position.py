"""Per-position validation NLL of retained checkpoints, at their training length and beyond it.

For each model and position t, L_t is the mean over sequences of -log p(x_t | x_<t). The
training-length pass scores the model's own recorded validation set and must reproduce its
recorded validation NLL. The evaluation-length pass scores longer packed rows cut from the
same token stream, with the model's length limit raised; no parameter depends on it.
Nothing is trained or updated.
"""
from __future__ import annotations

import argparse
from copy import copy
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

from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
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


def extend_forward_length(graph, params, *, max_seq_len: int):
    """Raise only a restored dot-product model's inference length guard.

    Source recipes and constructors retain full training validation. Mosaic's training
    limit includes backward shared memory, which is irrelevant to these forward-only
    evaluations; its forward kernels instead check hardware and tiled input shapes.
    RoPE positions are computed from the actual input length, so no parameter or cached
    position table changes. This helper must never be used to enable longer training.
    """
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("Evaluation length must be a positive integer")
    model = nnx.merge(graph, params)
    config = model.config
    if config.attention_mechanism != "dot_product" or config.positional_encoding not in ("rope", "none"):
        raise ValueError("Length extrapolation supports dot-product attention with RoPE or no positional encoding")
    if max_seq_len < config.max_seq_len:
        raise ValueError("Evaluation length cannot shorten the restored model's length guard")
    model.config = copy(config)
    model.config.max_seq_len = max_seq_len
    return nnx.split(model, nnx.Param)


def per_position_nll(graph, params, manifest, *, sequence_length: int, sequences: int, microbatch: int,
                     qualification: dict | None = None, details: dict | None = None):
    """Mean next-token NLL, with an optional same-input causal-prefix qualification."""
    if sequences % microbatch:
        raise ValueError("sequences must be a multiple of the microbatch")
    if qualification is not None:
        prefix = qualification["prefix_tokens"]
        if isinstance(prefix, bool) or not isinstance(prefix, int) or not 0 < prefix < sequence_length - 1:
            raise ValueError("Qualification prefix must be positive and shorter than the evaluated context")
        for key in ("mean_nll_atol", "max_token_nll_atol"):
            if not math.isfinite(qualification[key]) or qualification[key] < 0:
                raise ValueError(f"Qualification {key} must be finite and nonnegative")
    details = {} if details is None else details

    @jax.jit
    def token_nll(state, tokens):
        logits = nnx.merge(graph, state)(tokens[:, :-1], deterministic=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:])

    batches = PackedBatchIterator(manifest, batch_size=microbatch, sequence_length=sequence_length)
    total = np.zeros(sequence_length - 1, dtype=np.float64)
    batch_count = sequences // microbatch
    steady_seconds = 0.0
    for index in range(batch_count):
        tokens = jnp.asarray(next(batches))
        started = time.perf_counter()
        values = np.asarray(token_nll(params, tokens), dtype=np.float64)
        elapsed = time.perf_counter() - started
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite per-token loss in batch {index} at context {sequence_length - 1}")
        if index == 0:
            details["first_batch_seconds_including_compile"] = round(elapsed, 3)
            if qualification is not None:
                started = time.perf_counter()
                truncated = np.asarray(token_nll(params, tokens[:, :prefix + 1]), dtype=np.float64)
                difference = values[:, :prefix] - truncated
                result = {"prefix_tokens": prefix, "sequences": microbatch,
                          "mean_nll_absolute_error": float(abs(difference.mean())),
                          "mean_absolute_token_nll_error": float(np.abs(difference).mean()),
                          "max_token_nll_absolute_error": float(np.abs(difference).max()),
                          "seconds_including_compile": round(time.perf_counter() - started, 3),
                          "mean_nll_atol": qualification["mean_nll_atol"],
                          "max_token_nll_atol": qualification["max_token_nll_atol"]}
                result["passed"] = bool(np.isfinite(difference).all()
                                         and result["mean_nll_absolute_error"] <= qualification["mean_nll_atol"]
                                         and result["max_token_nll_absolute_error"] <= qualification["max_token_nll_atol"])
                details["causal_prefix_qualification"] = result
                print(json.dumps({"causal_prefix_qualification": result}), flush=True)
                if not result["passed"]:
                    raise ValueError(f"Causal-prefix qualification failed: {result}")
        else:
            steady_seconds += elapsed
        total += values.sum(axis=0)
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == batch_count:
            print(json.dumps({"evaluation_batches_done": index + 1, "evaluation_batches_total": batch_count,
                              "context_length": sequence_length - 1, "last_batch_seconds": round(elapsed, 3)}), flush=True)
    details["subsequent_batches_seconds"] = round(steady_seconds, 3)
    details["subsequent_batch_count"] = max(0, batch_count - 1)
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
    record = load_run(run)
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    lengths = evaluation["lengths"]
    for spec in lengths.values():
        if sha256(spec["manifest"]) != spec["manifest_sha256"]:
            raise ValueError(f"Packed-data manifest changed: {spec['manifest']}")
    outcomes = []
    completed = {}
    metrics_path = run / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["role"], row["length"])
            if key in completed:
                raise ValueError(f"Duplicate persisted evaluation: {key}")
            completed[key] = row
    expected = {(model["role"], name) for model in evaluation["models"] for name in lengths}
    if completed.keys() - expected:
        raise ValueError("Persisted measurements include an undeclared model/length")
    for model in evaluation["models"]:
        if sha256(model["config_path"]) != model["config_sha256"]:
            raise ValueError(f"Recipe changed: {model['config_path']}")
        print(f"Verifying checkpoint of {model['role']} ({model['run_id']})", flush=True)
        verify_checkpoint(model)
        for name, spec in lengths.items():
            limit = spec["sequence_length"] - 1
            previous = completed.get((model["role"], name))
            if previous is not None:
                if (previous["source_run_id"] != model["run_id"] or previous["checkpoint_step"] != model["step"]
                        or previous["sequence_length"] != spec["sequence_length"]
                        or previous["sequences"] != spec["sequences"]
                        or len(previous["per_position_nll"]) != limit
                        or not np.isfinite(previous["per_position_nll"]).all()):
                    raise ValueError(f"Persisted measurement does not match frozen recipe: {model['role']}/{name}")
                outcomes.append({key: value for key, value in previous.items() if key != "per_position_nll"})
                print(f"Reusing completed evaluation of {model['role']} at {limit} tokens", flush=True)
                continue
            config = load_model_config(model)
            if name == "training" and config.data.sequence_length != spec["sequence_length"]:
                raise ValueError(f"{model['role']} trains at {config.data.sequence_length}, not {spec['sequence_length']}")
            backend = resolve_attention_backend(config.model, dtype=config.train.dtype, sequence_length=limit)
            if backend != config.model.attention_backend:
                raise ValueError("Evaluation must retain the source checkpoint's explicit backend")
            graph, params = restore_parent(config, Path(model["checkpoint_path"]))
            if name != "training":
                graph, params = extend_forward_length(graph, params, max_seq_len=limit)
            started = time.perf_counter()
            details = {}
            curve = per_position_nll(graph, params, spec["manifest"], sequence_length=spec["sequence_length"],
                                     sequences=spec["sequences"], microbatch=spec["microbatch_size"], details=details,
                                     qualification=None if name == "training" else evaluation.get("qualification"))
            nll = float(curve.mean())
            row = {"role": model["role"], "source_run_id": model["run_id"], "checkpoint_step": model["step"],
                   "length": name, "sequence_length": spec["sequence_length"], "sequences": spec["sequences"],
                   "prediction_tokens": spec["sequences"] * limit, "attention_backend": config.model.attention_backend,
                   "nll": nll, "perplexity": math.exp(nll), "seconds": round(time.perf_counter() - started, 1), **details}
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
                stream.flush()
                os.fsync(stream.fileno())
            del graph, params
            jax.clear_caches()
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_updates": 0, "measurements": outcomes,
               "interpretation": evaluation["interpretation"]}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "evaluation_outcome": outcome})
    print("Per-position evaluation complete", flush=True)


if __name__ == "__main__":
    main()
