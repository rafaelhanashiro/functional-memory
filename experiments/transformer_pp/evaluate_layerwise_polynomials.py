"""Measure held-out NLL with native, exchanged, or mean polynomials at one or all layers."""
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


def evaluate(graph, params, calibration, recipients, *, microbatch, pairing_seed, model_dtype="float32",
             layer_scope="one_at_a_time"):
    """Rows contain next-token targets. Returned indices are relative to the recipient split."""
    if microbatch < 2 or microbatch % 2:
        raise ValueError("Use an even microbatch for reciprocal donor pairs")
    if layer_scope not in ("one_at_a_time", "all_layers"):
        raise ValueError(f"Unknown layer scope: {layer_scope}")
    if calibration.ndim != 2 or recipients.ndim != 2 or calibration.shape[1] != recipients.shape[1]:
        raise ValueError("Calibration and evaluation require packed rows of equal length")
    if min(len(recipients), len(calibration)) < microbatch or recipients.shape[1] < 2 or any(
            size % microbatch for size in (len(calibration), len(recipients))):
        raise ValueError("Use nonempty complete microbatches with next-token targets")

    coefficient_domain = {"coefficient_score_transform": nnx.merge(graph, params).config.attention_score_transform}

    @jax.jit
    def calibration_mean(state, rows):
        _, coeff = nnx.merge(graph, state)(rows[:, :-1], deterministic=True, return_coefficients=True, **coefficient_domain)
        return coeff.mean(axis=(1, 2))  # [layer, head, monomial degree]

    @jax.jit
    def native(state, rows):
        logits, coeff = nnx.merge(graph, state)(rows[:, :-1], deterministic=True, return_coefficients=True, **coefficient_domain)
        return sequence_nll(logits, rows), coeff

    @jax.jit
    def qualify_replay(state, rows):
        model = nnx.merge(graph, state)
        inputs = rows[:, :-1]
        ordinary = model(inputs, deterministic=True)
        captured, coefficients = model(inputs, deterministic=True, return_coefficients=True, **coefficient_domain)
        replayed = model(inputs, deterministic=True, query_coefficients=coefficients, **coefficient_domain)
        return tuple(sequence_nll(logits, rows) for logits in (ordinary, captured, replayed))

    @partial(jax.jit, static_argnames=("layer",))
    def intervene(state, rows, coefficients, layer):
        logits = nnx.merge(graph, state)(rows[:, :-1], deterministic=True, query_coefficients=coefficients,
                                         coefficient_override_layer=layer, **coefficient_domain)
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
    # None passes the full [layer, batch, token, head, degree] table to the model.
    targets = (None,) if layer_scope == "all_layers" else range(len(means))
    loss_shape = (len(targets), len(order))
    losses = {name: np.empty(loss_shape, dtype=np.float64) for name in conditions}
    identity_nll = np.empty(loss_shape, dtype=np.float64)
    # The coefficient-fit diagnostic qualifies separately compiled BF16 GPU forwards at
    # 0.002 nats per sequence. Same-compilation identity still gets the strict check.
    external_atol = 0.002 if model_dtype == "bfloat16" and jax.default_backend() == "gpu" else 2e-4
    for start in range(0, len(order), microbatch):
        batch = slice(start, start + microbatch)
        rows = jnp.asarray(recipients[order[batch]])
        native_nll[batch], coeff = native(params, rows)
        if start == 0:
            ordinary, captured, replayed = map(np.asarray, qualify_replay(params, rows))
            np.testing.assert_allclose(captured, ordinary, atol=1e-5, rtol=1e-5,
                                       err_msg="Joint coefficient capture changed native NLL")
            np.testing.assert_allclose(replayed, ordinary, atol=1e-5, rtol=1e-5,
                                       err_msg="Joint coefficient replay changed native NLL")
            joint_error = float(np.max(np.abs(replayed - ordinary)))
            capture_error = float(np.max(np.abs(ordinary - native_nll[batch])))
            np.testing.assert_allclose(native_nll[batch], ordinary, atol=external_atol, rtol=0,
                                       err_msg="Coefficient capture changed native NLL")
        for index, layer in enumerate(targets):
            if layer is None:
                own = coeff
                swapped = jnp.take(own, partner, axis=1)
                mean = jnp.broadcast_to(means[:, None, None, :, :], own.shape)
            else:
                own = coeff[layer]
                swapped = own[partner]
                mean = jnp.broadcast_to(means[layer], own.shape)
            losses["swapped"][index, batch] = intervene(params, rows, swapped, layer)
            losses["calibration_mean"][index, batch] = intervene(params, rows, mean, layer)
            replay = np.asarray(intervene(params, rows, own, layer))
            identity_nll[index, batch] = replay
            target = "All layers" if layer is None else f"Layer {layer}"
            np.testing.assert_allclose(replay, native_nll[batch], atol=external_atol, rtol=0,
                                       err_msg=f"{target}: external replay failed for recipient rows {order[batch]}")
            if start == 0:
                error = np.max(np.abs(replay - native_nll[batch]))
                print(f"{target}: identity replay max |ΔNLL| {error:.6g}", flush=True)
        if start == 0 or (start + microbatch) % 32 == 0:
            print(f"Scored {start + microbatch}/{len(order)} held-out sequences", flush=True)
    if not np.isfinite(native_nll).all() or any(not np.isfinite(x).all() for x in losses.values()):
        raise FloatingPointError("An evaluation produced a nonfinite loss")

    records = [{"condition": "native", "nll": float(native_nll.mean()), "sequence_nll": native_nll.tolist(),
                "capture_max_abs_sequence_nll_error": capture_error,
                "joint_replay_max_abs_sequence_nll_error": joint_error,
                "external_replay_atol": external_atol}]
    errors = np.abs(identity_nll - native_nll).max(axis=1)
    if layer_scope == "all_layers":
        records[0].update(layer_scope=layer_scope, identity_sequence_nll=identity_nll[0].tolist(),
                          identity_max_abs_sequence_nll_error=float(errors[0]))
    else:
        records[0].update(identity_sequence_nll_by_layer=identity_nll.tolist(),
                          identity_max_abs_sequence_nll_error_by_layer=errors.tolist())
    for index, layer in enumerate(targets):
        for condition in conditions:
            values = losses[condition][index]
            delta_per_pair = (values - native_nll).reshape(-1, 2).mean(axis=1)
            delta_vs_replay = (values - identity_nll[index]).reshape(-1, 2).mean(axis=1)
            scope = {"layer_scope": layer_scope} if layer is None else {"layer_index": layer}
            records.append({**scope, "condition": condition, "nll": float(values.mean()),
                            "delta_nll": float(delta_per_pair.mean()),
                            "delta_nll_vs_identity_replay": float(delta_vs_replay.mean()),
                            "paired_se_vs_identity_replay": (float(delta_vs_replay.std(ddof=1) / np.sqrt(len(delta_vs_replay)))
                                                             if len(delta_vs_replay) > 1 else None),
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
                              microbatch=batch, pairing_seed=recipe["pairing_seed"], model_dtype=config.train.dtype,
                              layer_scope=recipe.get("layer_scope", "one_at_a_time"))
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
