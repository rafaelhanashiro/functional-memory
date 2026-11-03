"""Attention weights of one query over every key position, per layer, averaged over heads.

For each model, one eager forward of one packed row through the model's own backend captures
what every attention call receives (post-RoPE Q/K and each query's effective monomial
coefficients). For the chosen query position t, each key's weight is w = sum_r c_r(t) s^r with
s = q.k/sqrt(d_K); its share of the query's total |w| is averaged over heads and stored per
layer and key position. Nothing is trained or updated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.transformer_pp.diagnose_attention_scores import AttentionRecorder
from experiments.transformer_pp.evaluate_per_position import (
    extend_forward_length, load_model_config, sha256, verify_checkpoint,
)
from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent

ROOT = Path(__file__).resolve().parents[5]


def query_shares(q, k, coefficients, *, position: int) -> jax.Array:
    """|w| shares [heads, position + 1] of one query over its causal keys."""
    query = q[:, position].astype(jnp.float32)  # [H, D]
    keys = k[:, :position + 1].astype(jnp.float32)  # [H, T, D]
    scores = jnp.einsum("hd,hkd->hk", query, keys) / math.sqrt(q.shape[-1])
    c = coefficients[:, position].astype(jnp.float32)[:, None, :]  # [H, 1, m+1]
    weights = c[..., -1]
    for order in range(c.shape[-1] - 2, -1, -1):
        weights = weights * scores + c[..., order]
    magnitude = jnp.abs(weights)
    return magnitude / jnp.sum(magnitude, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise FileExistsError("Position weights already completed")
    spec = yaml.safe_load((run / "config.yaml").read_text())["position_weights"]
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
        "optimizer_updates": 0,
    }
    record = load_run(run)
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    if sha256(spec["manifest"]) != spec["manifest_sha256"]:
        raise ValueError(f"Packed-data manifest changed: {spec['manifest']}")
    batches = PackedBatchIterator(spec["manifest"], batch_size=1, sequence_length=spec["sequence_length"])
    for _ in range(spec["row"]):
        next(batches)
    tokens = jnp.asarray(next(batches))
    eos = json.loads(Path(spec["manifest"]).read_text())["metadata"]["eos_token_id"]
    eos_positions = np.flatnonzero(np.asarray(tokens[0, :-1]) == eos).tolist()
    position, limit = spec["query_position"], spec["sequence_length"] - 1
    rows = []
    for model_spec in spec["models"]:
        if sha256(model_spec["config_path"]) != model_spec["config_sha256"]:
            raise ValueError(f"Recipe changed: {model_spec['config_path']}")
        verify_checkpoint(model_spec)
        config = load_model_config(model_spec)
        graph, params = restore_parent(config, Path(model_spec["checkpoint_path"]))
        if limit > config.model.max_seq_len:
            graph, params = extend_forward_length(graph, params, max_seq_len=limit)
        model = nnx.merge(graph, params)
        with AttentionRecorder() as recorder:
            model(tokens[:, :-1], deterministic=True)
        for layer, (q, k, arguments) in enumerate(recorder.calls):
            coefficients = arguments.get("query_polynomial_coefficients")
            if coefficients is None or arguments.get("attention_normalization"):
                raise ValueError("Expected unnormalized query-coefficient attention")
            shares = np.asarray(query_shares(jnp.moveaxis(q[0], 1, 0), jnp.moveaxis(k[0], 1, 0),
                                             jnp.moveaxis(coefficients[0], 1, 0), position=position), np.float64)
            rows.append({"model": model_spec["name"], "model_run_id": model_spec["run_id"], "layer": layer,
                         "row": spec["row"], "query_position": position, "eos_positions": eos_positions,
                         "head_mean_share": np.round(shares.mean(axis=0), 9).tolist(),
                         "share_beyond_2048_by_head": shares[:, :position + 1 - 2048].sum(axis=1).round(6).tolist()})
        del model, graph, params, recorder
        jax.clear_caches()
    with (run / "metrics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0,
               "interpretation": spec["interpretation"]}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "position_weights_outcome": outcome})
    print("Position weights complete", flush=True)


if __name__ == "__main__":
    main()
