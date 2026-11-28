"""Measure the effect of resetting a trained static polynomial to Taylor on its frozen backbone."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.transformer_pp.evaluate_calibration_mean import sequence_nll
from experiments.transformer_pp.evaluate_per_position import load_model_config, sha256, verify_checkpoint
from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import capture_artifacts, load_run, regenerate_index, update_record
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def reset_to_taylor(params, model):
    """Set the single learned factorial-coordinate tensor to one and reuse every other leaf."""
    if (model.attention_coefficients != "learned" or model.attention_coefficient_scale != "factorial"
            or model.attention_coefficient_conditioning != "static" or model.attention_coefficient_orders is not None
            or model.attention_coefficient_parameterization != "direct" or model.attention_coefficient_sharing != "per_head"):
        raise ValueError("This reset requires a full per-head static polynomial in direct factorial coordinates")
    expected = (model.num_layers, model.num_heads, model.attention_degree + 1)
    leaves = []
    reset_count = 0
    for path, value in nnx.to_flat_state(params):
        if path == ("polynomial_coefficients",):
            if value.shape != expected:
                raise ValueError(f"Expected a coefficient tensor of shape {expected}, got {value.shape}")
            value = value.replace(jnp.ones_like(value[...]))
            reset_count += 1
        leaves.append((path, value))
    if reset_count != 1:
        raise ValueError(f"Expected one static coefficient tensor, found {reset_count}")
    return nnx.from_flat_state(leaves)


def score_parent(role, spec, recipe):
    parent_dir = STUDY / "runs" / spec["parent_run_id"]
    checkpoint = load_run(parent_dir)["selected_checkpoint"]
    config_file = parent_dir / "config.yaml"
    if (sha256(config_file) != spec["parent_config_sha256"]
            or (checkpoint["sha256"], checkpoint["step"], checkpoint["validation_nll"])
            != (spec["checkpoint_sha256"], spec["checkpoint_step"], spec["validation_nll"])):
        raise ValueError(f"{role} checkpoint or recipe does not match its pinned selection")
    config = load_model_config({"config_path": str(config_file)})
    n, micro = recipe["evaluation_sequences"], recipe["microbatch"]
    if (sha256(config.data.eval_manifest) != recipe["eval_manifest_sha256"]
            or config.data.sequence_length != recipe["context_length"] + 1
            or n != config.data.batch_size * config.train.eval_batches or recipe["start_sequence"] != 0
            or micro < 1 or n % micro or recipe["replay_nll_atol"] <= 0):
        raise ValueError("Evaluation must use complete microbatches on the original full validation prefix")
    backend = resolve_attention_backend(config.model, dtype=config.train.dtype)
    if backend != "mosaic":
        raise ValueError(f"{role} resolved backend {backend} differs from its training backend")
    if role == "taylor" and (config.model.attention_coefficients != "fixed"
                            or config.model.attention_fixed_coefficients is not None
                            or config.model.attention_coefficient_initialization != "taylor"):
        raise ValueError("The control must be a fixed Taylor checkpoint")
    print(f"Verifying and restoring {role} from {checkpoint['path']}", flush=True)
    verify_checkpoint({"checkpoint_path": checkpoint["path"], "checkpoint_files": checkpoint["files"]})
    graph, params = restore_parent(config, Path(checkpoint["path"]))
    conditions = {role + "_native": params}
    if role == "static":
        conditions["static_reset"] = reset_to_taylor(params, config.model)

    @jax.jit
    def forward(state, rows):
        logits = nnx.merge(graph, state)(rows[:, :-1], deterministic=True)
        return sequence_nll(logits, rows)

    batches = PackedBatchIterator(config.data.eval_manifest, batch_size=micro, sequence_length=config.data.sequence_length,
                                  start_sequence=recipe["start_sequence"])
    losses = {name: np.empty(n, dtype=np.float64) for name in conditions}
    for first in range(0, n, micro):
        rows = jnp.asarray(next(batches))
        for name, state in conditions.items():
            values = np.asarray(forward(state, rows), dtype=np.float64)
            if values.shape != (micro,) or not np.isfinite(values).all():
                raise FloatingPointError(f"{name} produced invalid losses in packed rows {first}:{first+micro}")
            losses[name][first:first+micro] = values
        processed = first + micro
        if first == 0 or processed % (32 * micro) == 0 or processed == n:
            print(f"{role}: scored {processed}/{n} sequences", flush=True)
    metrics = {}
    for name, per_sequence in losses.items():
        nll = float(per_sequence.mean())
        metrics[name] = {"condition": name, "parent_run_id": spec["parent_run_id"], "nll": nll,
                         "perplexity": math.exp(nll), "sequence_nll": per_sequence.tolist()}
    native = metrics[role + "_native"]
    error = abs(native["nll"] - spec["validation_nll"])
    native.update(recorded_validation_nll=spec["validation_nll"], replay_absolute_nll_error=error)
    if error > recipe["replay_nll_atol"]:
        raise ValueError(f"{role} native NLL {native['nll']} differs from recorded {spec['validation_nll']} by {error}")
    print(f"{role} native validation replay |delta NLL| = {error:.7g}", flush=True)
    return metrics


def paired_comparison(left, right):
    difference = np.asarray(left["sequence_nll"]) - right["sequence_nll"]
    return {"left": left["condition"], "right": right["condition"], "delta_nll": float(difference.mean()),
            "paired_sequence_se": float(difference.std(ddof=1) / np.sqrt(difference.size)),
            "delta_perplexity": left["perplexity"] - right["perplexity"]}


def run_evaluation(run):
    record = load_run(run)
    if record["status"] == "complete":
        print("Evaluation already complete", flush=True)
        return
    config_file = run / "config.yaml"
    if sha256(config_file) != record["config"]["sha256"]:
        raise ValueError("Evaluation recipe has changed")
    recipe = yaml.safe_load(config_file.read_text())["evaluation"]
    if set(recipe["models"]) != {"static", "taylor"}:
        raise ValueError("Expected one static parent and one fixed-Taylor parent")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("The parents' Mosaic kernels require one GPU")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if modified := changes(ROOT, commit, STUDY.name, "transformer"):
        raise ValueError(f"Evaluation source differs from commit {commit}: {modified}")
    runtime = {"source_commit": commit, "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
               "started_utc": datetime.now(UTC).isoformat(), "partition": os.getenv("SLURM_JOB_PARTITION"),
               "slurm_job_id": os.getenv("SLURM_JOB_ID"), "node": os.uname().nodename, "gpu_count": 1,
               "gpu_type": devices[0].device_kind, "resolved_attention_backend": "mosaic", "optimizer_updates": 0}
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    metrics = {}
    for role in ("static", "taylor"):
        metrics.update(score_parent(role, recipe["models"][role], recipe))

    completion = {
        "status": "complete", "evaluation_sequences": recipe["evaluation_sequences"],
        "evaluation_tokens": recipe["evaluation_sequences"] * recipe["context_length"], "optimizer_updates": 0,
        "comparisons": {name: paired_comparison(metrics["static_reset"], metrics[reference]) for name, reference in
                        (("reset_minus_static_native", "static_native"), ("reset_minus_taylor_native", "taylor_native"))},
    }
    metadata = {**runtime, "start_sequence": recipe["start_sequence"], "context_length": recipe["context_length"],
                "evaluation_sequences": recipe["evaluation_sequences"], "microbatch": recipe["microbatch"],
                "eval_manifest_sha256": recipe["eval_manifest_sha256"], "parent_checkpoints": recipe["models"],
                "sequence_order": "ascending packed validation row", "replay_nll_atol": recipe["replay_nll_atol"]}
    for filename, contents in (
        ("metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in metrics.values())),
        ("metadata.json", json.dumps(metadata, indent=2, allow_nan=False) + "\n"),
        ("completion.json", json.dumps(completion, indent=2, allow_nan=False) + "\n"),
    ):
        temporary = run / (filename + ".tmp")
        temporary.write_text(contents)
        temporary.replace(run / filename)
    update_record(run, {**capture_artifacts(run), "status": "complete"})
    regenerate_index(STUDY)
    print(json.dumps(completion["comparisons"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    run = parser.parse_args().run.resolve()
    with (run / "config.yaml").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_evaluation(run)


if __name__ == "__main__":
    main()
