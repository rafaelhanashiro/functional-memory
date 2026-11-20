"""Evaluate a frozen checkpoint with every query polynomial fixed to its head's calibration mean."""
from __future__ import annotations

import argparse
import fcntl
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from experiments.transformer_pp.evaluate_per_position import load_model_config, sha256, verify_checkpoint
from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import capture_artifacts, load_run, regenerate_index, update_record
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def sequence_nll(logits, rows):
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), rows[:, 1:])
    return token_loss.mean(axis=1)


def evaluate(graph, params, calibration, recipients, *, microbatch, sequence_seed, model_dtype="float32"):
    """Average expanded coefficients over calibration tokens, then score the held-out rows."""
    if calibration.ndim != 2 or recipients.ndim != 2 or calibration.shape[1] != recipients.shape[1]:
        raise ValueError("Calibration and evaluation require packed rows of equal length")
    if (microbatch < 1 or min(len(recipients), len(calibration)) < microbatch or calibration.shape[1] < 2
            or len(recipients) % microbatch or len(calibration) % microbatch or len(recipients) % 2):
        raise ValueError("Use complete microbatches and an even number of held-out sequences")

    score_domain = {"coefficient_score_transform": nnx.merge(graph, params).config.attention_score_transform}

    @jax.jit
    def calibration_mean(state, rows):
        _, coeff = nnx.merge(graph, state)(rows[:, :-1], return_coefficients=True, **score_domain)
        return coeff.mean(axis=(1, 2))  # [layer, head, monomial order]

    @jax.jit
    def native(state, rows):
        logits, coeff = nnx.merge(graph, state)(rows[:, :-1], return_coefficients=True, **score_domain)
        return sequence_nll(logits, rows), coeff

    @jax.jit
    def qualify_replay(state, rows):
        model = nnx.merge(graph, state)
        inputs = rows[:, :-1]
        ordinary = model(inputs)
        captured, coefficients = model(inputs, return_coefficients=True, **score_domain)
        replayed = model(inputs, query_coefficients=coefficients, **score_domain)
        return tuple(sequence_nll(logits, rows) for logits in (ordinary, captured, replayed))

    @jax.jit
    def intervene(state, rows, coefficients):
        logits = nnx.merge(graph, state)(rows[:, :-1], query_coefficients=coefficients, **score_domain)
        return sequence_nll(logits, rows)

    means = np.mean([
        np.asarray(calibration_mean(params, calibration[i:i + microbatch]))
        for i in range(0, len(calibration), microbatch)
    ], axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(means).all():
        raise FloatingPointError("Calibration produced nonfinite coefficients")
    means = jnp.asarray(means)[:, None, None, :, :]
    order = np.random.default_rng(sequence_seed).permutation(len(recipients))
    native_nll, mean_nll, identity_nll = np.empty((3, len(order)), dtype=np.float64)
    # Separately compiled BF16 GPU forwards have the same 0.002-nat tolerance as
    # the coefficient-fit diagnostic. Joint replay is checked independently.
    external_atol = 0.002 if model_dtype == "bfloat16" and jax.default_backend() == "gpu" else 2e-4
    for start in range(0, len(order), microbatch):
        batch = slice(start, start + microbatch)
        rows = jnp.asarray(recipients[order[batch]])
        native_nll[batch], coefficients = native(params, rows)
        if start == 0:
            ordinary, captured, replayed = map(np.asarray, qualify_replay(params, rows))
            np.testing.assert_allclose(captured, ordinary, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(replayed, ordinary, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(native_nll[batch], ordinary, atol=external_atol, rtol=0)
            capture_error = float(np.max(np.abs(ordinary - native_nll[batch])))
            joint_error = float(np.max(np.abs(replayed - ordinary)))
        mean_nll[batch] = intervene(params, rows, jnp.broadcast_to(means, coefficients.shape))
        identity_nll[batch] = intervene(params, rows, coefficients)
        np.testing.assert_allclose(identity_nll[batch], native_nll[batch], atol=external_atol, rtol=0,
                                   err_msg="Identity replay changed held-out NLL")
    if not all(np.isfinite(values).all() for values in (mean_nll, native_nll, identity_nll)):
        raise FloatingPointError("An evaluation produced a nonfinite loss")

    # Report the historical uncertainty estimator: one paired difference averaged
    # over each consecutive block of two sequences, in the frozen recipe's order.
    def paired_difference(reference):
        blocks = (mean_nll - reference).reshape(-1, 2).mean(axis=1)
        error = float(blocks.std(ddof=1) / np.sqrt(len(blocks))) if len(blocks) > 1 else None
        return float(blocks.mean()), error

    delta, se = paired_difference(native_nll)
    replay_delta, replay_se = paired_difference(identity_nll)
    return order, [
        {"condition": "native", "layer_scope": "all_layers", "nll": float(native_nll.mean()),
         "sequence_nll": native_nll.tolist(), "capture_max_abs_sequence_nll_error": capture_error,
         "joint_replay_max_abs_sequence_nll_error": joint_error, "external_replay_atol": external_atol,
         "identity_sequence_nll": identity_nll.tolist(),
         "identity_max_abs_sequence_nll_error": float(np.max(np.abs(identity_nll - native_nll)))},
        {"condition": "calibration_mean", "layer_scope": "all_layers", "nll": float(mean_nll.mean()),
         "sequence_nll": mean_nll.tolist(), "delta_nll": delta, "delta_nll_vs_identity_replay": replay_delta,
         "paired_se": se, "paired_se_vs_identity_replay": replay_se},
    ]


def run_evaluation(run):
    record = load_run(run)
    if record["status"] == "complete":
        raise ValueError(f"{run.name} already completed")
    config_file = run / "config.yaml"
    if sha256(config_file) != record["config"]["sha256"]:
        raise ValueError("Diagnostic recipe changed")
    recipe = yaml.safe_load(config_file.read_text())["evaluation"]
    if recipe.get("layer_scope") != "all_layers":
        raise ValueError("This diagnostic applies the calibration means to all layers together")
    parent_dir = STUDY / "runs" / recipe["parent_run_id"]
    parent = load_run(parent_dir)
    selected = parent["selected_checkpoint"]
    parent_file = parent_dir / "config.yaml"
    if (sha256(parent_file) != recipe["parent_config_sha256"]
            or selected["sha256"] != recipe["checkpoint_sha256"] or selected["step"] != recipe["checkpoint_step"]):
        raise ValueError("Parent recipe or checkpoint selection changed")
    config = load_model_config({"config_path": str(parent_file)})
    if sha256(config.data.eval_manifest) != recipe["eval_manifest_sha256"]:
        raise ValueError("Validation manifest changed")
    verify_checkpoint({"checkpoint_path": selected["path"], "checkpoint_files": selected["files"]})
    batch = recipe["microbatch"]
    calibration_count = recipe["calibration_sequences"]
    count = recipe["evaluation_sequences"]
    iterator = PackedBatchIterator(config.data.eval_manifest, batch_size=calibration_count + count,
                                   sequence_length=config.data.sequence_length)
    rows = next(iterator)
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("The recorded parent checkpoints require one GPU for Mosaic evaluation")
    backend = resolve_attention_backend(config.model, dtype=config.train.dtype)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if modified := changes(ROOT, commit, STUDY.name, "transformer"):
        raise ValueError(f"Evaluation source differs from its commit: {modified}")
    runtime = {"started_utc": datetime.now(UTC).isoformat(), "source_commit": commit,
               "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
               "slurm_job_id": os.getenv("SLURM_JOB_ID"), "partition": os.getenv("SLURM_JOB_PARTITION"),
               "node": os.uname().nodename, "gpu_type": devices[0].device_kind, "gpu_count": 1,
               "resolved_attention_backend": backend, "optimizer_updates": 0}
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    graph, params = restore_parent(config, Path(selected["path"]))
    order, metrics = evaluate(graph, params, rows[:calibration_count], rows[calibration_count:],
                              microbatch=batch, sequence_seed=recipe["pairing_seed"], model_dtype=config.train.dtype)
    recipients = order + calibration_count
    metadata = {"calibration_row_range": [0, calibration_count], "recipient_rows": recipients.tolist(),
                "context_length": rows.shape[1] - 1}
    for name, content in (
        ("metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in metrics)),
        ("metadata.json", json.dumps(metadata, indent=2) + "\n"),
        ("completion.json", json.dumps({"status": "complete", "layers": config.model.num_layers,
                                       "evaluation_sequences": count, "native_nll": metrics[0]["nll"]}) + "\n"),
    ):
        temporary = run / (name + ".tmp")
        temporary.write_text(content)
        temporary.replace(run / name)
    update_record(run, {**capture_artifacts(run), "status": "complete"})
    regenerate_index(STUDY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    # Record writers lock the directory. A separate file lock excludes duplicate evaluators.
    descriptor = os.open(run / "config.yaml", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_evaluation(run)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
