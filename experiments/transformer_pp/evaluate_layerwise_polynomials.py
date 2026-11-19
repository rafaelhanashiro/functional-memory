"""Measure held-out NLL with native, exchanged, or mean polynomials at one layer at a time."""
from __future__ import annotations

import argparse
import fcntl
from datetime import UTC, datetime
from functools import partial
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


def evaluate(graph, params, calibration, recipients, *, microbatch, pairing_seed):
    """Rows contain next-token targets. Returned indices are relative to the recipient split."""
    if microbatch < 2 or microbatch % 2:
        raise ValueError("Use an even microbatch for reciprocal donor pairs")
    if calibration.ndim != 2 or recipients.ndim != 2 or calibration.shape[1] != recipients.shape[1]:
        raise ValueError("Calibration and evaluation require packed rows of equal length")
    if min(len(recipients), len(calibration)) < microbatch or recipients.shape[1] < 2 or any(
            size % microbatch for size in (len(calibration), len(recipients))):
        raise ValueError("Use nonempty complete microbatches with next-token targets")

    @jax.jit
    def calibration_mean(state, rows):
        _, coeff = nnx.merge(graph, state)(rows[:, :-1], deterministic=True, return_coefficients=True)
        return coeff.mean(axis=(1, 2))  # [layer, head, monomial degree]

    @jax.jit
    def native(state, rows):
        logits, coeff = nnx.merge(graph, state)(rows[:, :-1], deterministic=True, return_coefficients=True)
        return sequence_nll(logits, rows), coeff

    @jax.jit
    def ordinary_nll(state, rows):
        return sequence_nll(nnx.merge(graph, state)(rows[:, :-1], deterministic=True), rows)

    @partial(jax.jit, static_argnames=("layer",))
    def intervene(state, rows, coefficients, layer):
        logits = nnx.merge(graph, state)(rows[:, :-1], deterministic=True, query_coefficients=coefficients,
                                         coefficient_override_layer=layer)
        return sequence_nll(logits, rows)

    means = np.mean([
        np.asarray(calibration_mean(params, calibration[i:i + microbatch]))
        for i in range(0, len(calibration), microbatch)
    ], axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(means).all():
        raise FloatingPointError("Calibration produced nonfinite coefficients")
    means = jnp.asarray(means)
    order = np.random.default_rng(pairing_seed).permutation(len(recipients))
    partner = np.arange(microbatch) ^ 1
    native_nll = np.empty(len(order), dtype=np.float64)
    conditions = ("swapped", "calibration_mean")
    losses = {name: np.empty((len(means), len(order)), dtype=np.float64) for name in conditions}
    replay_error = np.zeros(len(means))
    for start in range(0, len(order), microbatch):
        batch = slice(start, start + microbatch)
        rows = jnp.asarray(recipients[order[batch]])
        native_nll[batch], coeff = native(params, rows)
        if start == 0:
            ordinary = np.asarray(ordinary_nll(params, rows))
            capture_error = float(np.max(np.abs(ordinary - native_nll[batch])))
            np.testing.assert_allclose(native_nll[batch], ordinary, atol=2e-4, rtol=0,
                                       err_msg="Coefficient capture changed native NLL")
        for layer in range(len(means)):
            own = coeff[layer]
            swapped = own[partner]
            mean = jnp.broadcast_to(means[layer], own.shape)
            losses["swapped"][layer, batch] = intervene(params, rows, swapped, layer)
            losses["calibration_mean"][layer, batch] = intervene(params, rows, mean, layer)
            if start == 0:
                replay = np.asarray(intervene(params, rows, own, layer))
                replay_error[layer] = np.max(np.abs(replay - native_nll[batch]))
                np.testing.assert_allclose(replay, native_nll[batch], atol=2e-4, rtol=0,
                                           err_msg=f"Native coefficient replay failed at layer {layer}")
        if start == 0 or (start + microbatch) % 32 == 0:
            print(f"Scored {start + microbatch}/{len(order)} held-out sequences", flush=True)
    if not np.isfinite(native_nll).all() or any(not np.isfinite(x).all() for x in losses.values()):
        raise FloatingPointError("An evaluation produced a nonfinite loss")

    records = [{"condition": "native", "nll": float(native_nll.mean()), "sequence_nll": native_nll.tolist(),
                "capture_max_abs_sequence_nll_error": capture_error,
                "identity_max_abs_sequence_nll_error_by_layer": replay_error.tolist()}]
    for layer in range(len(means)):
        for condition in conditions:
            values = losses[condition][layer]
            delta_per_pair = (values - native_nll).reshape(-1, 2).mean(axis=1)
            records.append({"layer_index": layer, "condition": condition, "nll": float(values.mean()),
                            "delta_nll": float(delta_per_pair.mean()),
                            "paired_se": (float(delta_per_pair.std(ddof=1) / np.sqrt(len(delta_per_pair)))
                                          if len(delta_per_pair) > 1 else None),
                            "sequence_nll": values.tolist()})
    return order, records


def run_evaluation(run):
    record = load_run(run)
    if record["status"] == "complete":
        raise ValueError(f"{run.name} already completed")
    config_file = run / "config.yaml"
    if sha256(config_file) != record["config"]["sha256"]:
        raise ValueError("Diagnostic recipe changed")
    recipe = yaml.safe_load(config_file.read_text())["evaluation"]
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
                              microbatch=batch, pairing_seed=recipe["pairing_seed"])
    recipients = order + calibration_count
    metadata = {"calibration_row_range": [0, calibration_count], "recipient_rows": recipients.tolist(),
                "donor_rows": recipients[np.arange(count) ^ 1].tolist(), "context_length": rows.shape[1] - 1}
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
