"""Capture one layer's queries, keys and coefficients from coefficient-oracle fits.

Each fit's selected coefficient table replays on its frozen parent over the first sequences
of the fitting batch, and the per-sequence NLL must match the fit's own recorded replay. The
layer's queries and keys on the first sequence, as the attention core receives them, are then
kept with the coefficients each of its query positions used. Parameters and checkpoints are
read-only; no optimizer exists.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorstore as ts
import yaml

from experiments.transformer_pp.evaluate_per_position import sha256, verify_checkpoint
from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import capture_artifacts, load_run, regenerate_index, update_record
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import parent_config, restore_parent

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent


def write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def coefficient_table(checkpoint: Path, step: int) -> np.ndarray:
    """The fit's first coefficient chunk, in effective monomial coefficients c_r."""
    identity = json.loads((checkpoint / "coefficient_identity.json").read_text())
    if identity["coordinates"] != "effective_monomial_c" or identity["step"] != step:
        raise ValueError(f"Unexpected coefficient checkpoint {checkpoint}: {identity}")
    store = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "path": "coefficients.0",
                                           "base": {"driver": "file", "path": f"{checkpoint / 'best'}/"}}}
    return np.asarray(ts.open(store, open=True, read=True).result().read().result())


def capture(graph, params, tokens, table, *, layer, layers):
    """Per-sequence NLL under the coefficient table, and the layer's first-sequence queries and keys."""
    native = multihead.attention_core

    def forward(params, tokens, table):
        captured, calls = [], [0]

        def core(q, k, v, **kwargs):
            if calls[0] == layer:
                captured.extend([q[0].astype(jnp.float32), k[0].astype(jnp.float32)])
            calls[0] += 1
            return native(q, k, v, **kwargs)

        with patch.object(multihead, "attention_core", core):
            logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True,
                                              query_coefficients=jnp.swapaxes(table, 0, 1))
        if calls[0] != layers:
            raise ValueError(f"Expected {layers} attention calls, traced {calls[0]}")
        nll = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)
        return nll, *captured

    return jax.device_get(jax.jit(forward)(params, jnp.asarray(tokens), jnp.asarray(table)))


def run_diagnostic(run: Path) -> None:
    record = load_run(run)
    if record["status"] == "complete":
        raise ValueError(f"{run.name} already completed")
    config_file = run / "config.yaml"
    if sha256(config_file) != record["config"]["sha256"]:
        raise ValueError("Diagnostic recipe changed")
    recipe = yaml.safe_load(config_file.read_text())["diagnostic"]
    data = recipe["data"]
    if sha256(data["manifest"]) != data["manifest_sha256"]:
        raise ValueError("Validation manifest changed")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("The fits' Mosaic forward needs exactly one GPU")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if modified := changes(ROOT, commit, STUDY.name, "transformer"):
        raise ValueError(f"Diagnostic source differs from its commit: {modified}")
    runtime = {"started_utc": datetime.now(UTC).isoformat(), "source_commit": commit,
               "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
               "slurm_job_id": os.getenv("SLURM_JOB_ID"), "partition": os.getenv("SLURM_JOB_PARTITION"),
               "node": os.uname().nodename, "gpu_type": devices[0].device_kind, "gpu_count": 1,
               "resolved_attention_backends": {}, "optimizer_updates": 0}
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    layer, sequences, tolerance = recipe["layer"], recipe["replay_sequences"], recipe["replay_nll_tolerance"]
    rows = next(iter(PackedBatchIterator(data["manifest"], batch_size=data["batch_size"],
                                         sequence_length=data["sequence_length"])))
    if hashlib.sha256(rows.astype(np.int32).tobytes()).hexdigest() != data["batch_sha256"]:
        raise ValueError("The fitting batch differs from the pinned batch")
    operands, metrics = {}, []
    for fit in recipe["fits"]:
        role, fit_dir = fit["role"], STUDY / "runs" / fit["run_id"]
        fit_record = load_run(fit_dir)
        selected, source = fit_record["selected_checkpoint"], fit_record["source_checkpoint"]
        parent_file = STUDY / "runs" / fit_record["parent_run_id"] / "config.yaml"
        pins = {"config": (sha256(fit_dir / "config.yaml"), fit["config_sha256"]),
                "recorded replay": (sha256(fit_dir / "batch_functional_probe.json"), fit["replay_sha256"]),
                "checkpoint": (selected["sha256"], fit["checkpoint_sha256"]),
                "parent": (fit_record["parent_run_id"], fit["parent_run_id"]),
                "parent config": (sha256(parent_file), fit["parent_config_sha256"]),
                "parent checkpoint": (source["sha256"], fit["parent_checkpoint_sha256"])}
        if changed := [name for name, (actual, pinned) in pins.items() if actual != pinned]:
            raise ValueError(f"The {role} fit changed: {changed}")
        for checkpoint in (selected, source):
            verify_checkpoint({"checkpoint_path": checkpoint["path"], "checkpoint_files": checkpoint["files"]})
        config = parent_config(parent_file)
        if (Path(config.data.eval_manifest).resolve() != Path(data["manifest"]).resolve()
                or config.data.sequence_length != data["sequence_length"]):
            raise ValueError(f"The {role} fit's parent reads other data")
        table = coefficient_table(Path(selected["path"]), fit["checkpoint_step"])
        positions = data["sequence_length"] - 1
        if table.ndim == 3:  # One static [layer, head, order] table for every sequence and position.
            table = np.broadcast_to(table[None, :, None], (sequences, table.shape[0], positions, *table.shape[1:]))
        table = table[:sequences]
        if table.shape[:3] != (sequences, config.model.num_layers, positions):
            raise ValueError(f"Unexpected {role} coefficient table shape {table.shape}")
        backend = resolve_attention_backend(config.model, dtype=config.train.dtype)
        runtime["resolved_attention_backends"][role] = backend
        print(f"Replaying the {role} fit {fit['run_id']} on parent {fit['parent_run_id']}", flush=True)
        graph, params = restore_parent(config, Path(source["path"]))
        nll, q, k = capture(graph, params, rows[:sequences], table, layer=layer, layers=config.model.num_layers)
        probe = json.loads((fit_dir / "batch_functional_probe.json").read_text())
        recorded = np.asarray(probe["measurements"]["fitted_raw"]["per_sequence_nll"][:sequences])
        error = float(np.max(np.abs(nll - recorded)))
        if not error <= tolerance:
            raise ValueError(f"The {role} replay differs from its recorded NLL by {error}")
        operands |= {f"{role}_q": q, f"{role}_k": k, f"{role}_coefficients": np.asarray(table[0, layer])}
        metrics.append({"role": role, "fit_run_id": fit["run_id"], "checkpoint_step": fit["checkpoint_step"],
                        "parent_run_id": fit["parent_run_id"], "attention_backend": backend,
                        "layer": layer, "indexing": "zero_based", "captured_sequence": 0,
                        "replay_nll": nll.tolist(), "recorded_nll": recorded.tolist(),
                        "replay_max_abs_nll_error": error, "query_shape": list(q.shape)})
        print(json.dumps(metrics[-1]), flush=True)
        del graph, params
        gc.collect()
        jax.clear_caches()
    temporary = run / "operands.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **operands)
    temporary.replace(run / "operands.npz")
    write(run / "metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in metrics))
    write(run / "completion.json", json.dumps({
        "status": "complete", "fits": [row["role"] for row in metrics],
        "max_replay_nll_error": max(row["replay_max_abs_nll_error"] for row in metrics)}, indent=2) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete"})
    regenerate_index(STUDY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    # Record writers lock the directory. A separate file lock excludes duplicate captures.
    descriptor = os.open(run / "config.yaml", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_diagnostic(run)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
