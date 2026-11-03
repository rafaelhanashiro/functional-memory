"""How query polynomial attention spreads its weight over key distance, within and past training.

For each case (a checkpoint, a row length, and query-position windows), single-row eager
forwards through the model's own backend capture what every attention call receives: post-RoPE
Q/K and each query's effective monomial coefficients. For every query in a window, each key's
weight is w = sum_r c_r(t) s^r with s = q.k/sqrt(d_K); its share of the query's total |w| is
accumulated in distance bins, beside the share of keys each bin holds (the uniform baseline)
and the query's effective number of keys exp(entropy of the shares). Results are averaged over
queries, heads and rows, per layer. Nothing is trained or updated.
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

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.transformer_pp.diagnose_attention_scores import AttentionRecorder
from experiments.transformer_pp.evaluate_per_position import (
    extend_forward_length, load_model_config, sha256, verify_checkpoint,
)
from flax import nnx
from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent

ROOT = Path(__file__).resolve().parents[5]


def window_profile(q, k, coefficients, *, start: int, end: int, edges: tuple[int, ...]):
    """Per-head mean |w| share and key share per distance bin, and mean n_eff / n, for queries [start, end)."""
    queries = q[:, start:end].astype(jnp.float32)  # [H, Q, D]
    keys = k[:, :end].astype(jnp.float32)  # [H, T, D]
    scores = jnp.einsum("hqd,hkd->hqk", queries, keys) / math.sqrt(q.shape[-1])
    c = coefficients[:, start:end].astype(jnp.float32)[:, :, None, :]  # [H, Q, 1, m+1]
    weights = c[..., -1]
    for order in range(c.shape[-1] - 2, -1, -1):
        weights = weights * scores + c[..., order]
    distance = jnp.arange(start, end)[:, None] - jnp.arange(end)[None, :]
    causal = distance >= 0
    magnitude = jnp.where(causal[None], jnp.abs(weights), 0.0)
    share = magnitude / jnp.sum(magnitude, axis=-1, keepdims=True)
    keys_per_query = jnp.sum(causal, axis=-1).astype(jnp.float32)
    bins, uniform = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        member = causal & (distance >= low) & (distance < high)
        bins.append(jnp.mean(jnp.sum(jnp.where(member[None], share, 0.0), axis=-1), axis=-1))
        uniform.append(jnp.mean(jnp.sum(member, axis=-1) / keys_per_query))
    entropy = -jnp.sum(jnp.where(share > 0, share * jnp.log(share), 0.0), axis=-1)
    effective = jnp.mean(jnp.exp(entropy) / keys_per_query[None], axis=-1)
    return jnp.stack(bins, axis=-1), jnp.stack(uniform), effective


def profile_case(case: dict, spec: dict, edges: tuple[int, ...]) -> list[dict]:
    limit = case["sequence_length"] - 1
    config = load_model_config(spec)
    graph, params = restore_parent(config, Path(spec["checkpoint_path"]))
    if limit > config.model.max_seq_len:
        graph, params = extend_forward_length(graph, params, max_seq_len=limit)
    model = nnx.merge(graph, params)
    batches = PackedBatchIterator(case["manifest"], batch_size=1, sequence_length=case["sequence_length"])
    sums = {}
    for _ in range(case["rows"]):
        tokens = jnp.asarray(next(batches))
        with AttentionRecorder() as recorder:
            model(tokens[:, :-1], deterministic=True)
        for layer, (q, k, arguments) in enumerate(recorder.calls):
            coefficients = arguments.get("query_polynomial_coefficients")
            if coefficients is None or arguments.get("attention_normalization"):
                raise ValueError("The profile expects unnormalized query-coefficient attention")
            head_q, head_k = jnp.moveaxis(q[0], 1, 0), jnp.moveaxis(k[0], 1, 0)
            head_c = jnp.moveaxis(coefficients[0], 1, 0)
            for start, end in case["windows"]:
                shares, uniform, effective = (np.asarray(value, np.float64) for value in
                                              window_profile(head_q, head_k, head_c, start=start, end=end, edges=edges))
                key = (layer, start, end)
                previous = sums.get(key)
                sums[key] = (shares, uniform, effective) if previous is None else tuple(
                    a + b for a, b in zip(previous, (shares, uniform, effective)))
        del recorder
    rows = []
    for (layer, start, end), (shares, uniform, effective) in sorted(sums.items()):
        rows.append({"case": case["name"], "model_run_id": spec["run_id"], "sequence_length": case["sequence_length"],
                     "rows": case["rows"], "layer": layer, "query_start": start, "query_end": end,
                     "bin_edges": list(edges), "weight_share_by_head": (shares / case["rows"]).tolist(),
                     "key_share": (uniform / case["rows"]).tolist(),
                     "effective_key_fraction_by_head": (effective / case["rows"]).tolist()})
    del model, graph, params
    jax.clear_caches()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise FileExistsError("Profile already completed")
    profile = yaml.safe_load((run / "config.yaml").read_text())["profile"]
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
    models = {spec["name"]: spec for spec in profile["models"]}
    for spec in models.values():
        if sha256(spec["config_path"]) != spec["config_sha256"]:
            raise ValueError(f"Recipe changed: {spec['config_path']}")
        verify_checkpoint(spec)
    edges = tuple(profile["bin_edges"])
    metrics_path = run / "metrics.jsonl"
    done = set()
    if metrics_path.exists():
        done = {json.loads(line)["case"] for line in metrics_path.read_text().splitlines()}
    for case in profile["cases"]:
        if case["name"] in done:
            continue
        if sha256(case["manifest"]) != case["manifest_sha256"]:
            raise ValueError(f"Packed-data manifest changed: {case['manifest']}")
        print(f"Profiling {case['name']}", flush=True)
        rows = profile_case(case, models[case["model"]], edges)
        with metrics_path.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        summary = {(row["query_start"], row["query_end"]): [] for row in rows}
        for row in rows:
            summary[(row["query_start"], row["query_end"])].append(np.mean(row["weight_share_by_head"], axis=0))
        for window, values in summary.items():
            print(json.dumps({"case": case["name"], "window": window,
                              "mean_weight_share": np.round(np.mean(values, axis=0), 4).tolist()}), flush=True)
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0,
               "cases": [case["name"] for case in profile["cases"]], "interpretation": profile["interpretation"]}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "profile_outcome": outcome})
    print("Weight profile complete", flush=True)


if __name__ == "__main__":
    main()
