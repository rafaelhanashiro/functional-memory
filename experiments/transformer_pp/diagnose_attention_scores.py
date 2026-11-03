"""Attention scores and weights of retained checkpoints by key distance, past the training length.

For every model, layer and head this records the scores s_ti = <q_t, k_i> / sqrt(d_K) that the
attention kernel receives (after QK normalization, head scaling and RoPE), binned by the
distance t - i, and the share of each late query's kernel weight that every distance bin
receives. Shares use the weight magnitude |w(s)| normalized over the query's keys, which is
what the per-head output RMSNorm leaves of an unnormalized kernel. Each model runs its own
backend; scores and weights are recomputed from the arrays its kernel receives, captured
from the attention call. Nothing is trained or updated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
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

from experiments.transformer_pp.evaluate_per_position import load_model_config, sha256, verify_checkpoint
from scaffolds.records import capture_artifacts, update_record, validate_record
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.taylor import _taylor_polynomial
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent

ROOT = Path(__file__).resolve().parents[2]
UNSUPPORTED = ("coefficients", "polynomial_coefficients", "polynomial_square_factors",
               "key_polynomial_factors", "query_log_decay")


def kernel_kind(arguments: dict) -> str:
    """Name the weight function an attention call applies to its scores."""
    if any(arguments.get(name) is not None for name in UNSUPPORTED) or arguments.get("quadratic_mode", "fixed") != "fixed":
        raise ValueError("This diagnostic covers exp, softmax, fixed Taylor and query-conditioned kernels")
    if arguments.get("query_polynomial_coefficients") is not None:
        return "query"
    if arguments.get("attention_degree") is None:
        return "softmax" if arguments.get("attention_normalization", True) else "exp"
    return "taylor"


@partial(jax.jit, static_argnames=("kind", "degree", "bin_width", "bins", "training_length"))
def head_statistics(q, k, coefficients, *, kind, degree, bin_width, bins, training_length):
    """Distance-binned score sums and late-query weight shares for one head.

    ``q`` and ``k`` are [time, width]; ``coefficients`` is [time, degree + 1] for query kernels.
    Returns per-bin count, score sum, score square sum, minimum and maximum; the per-bin sum
    over late queries (t >= training_length) of their normalized weight shares; and the sum
    over late queries of the share on keys at distances >= training_length.
    """
    time, width = q.shape
    scores = (q.astype(jnp.float32) @ k.astype(jnp.float32).T) / math.sqrt(width)
    distance = jnp.arange(time)[:, None] - jnp.arange(time)[None, :]
    causal = distance >= 0
    index = jnp.where(causal, jnp.minimum(distance // bin_width, bins - 1), bins).reshape(-1)

    def binned(values, reduce=jax.ops.segment_sum):
        return reduce(values.reshape(-1), index, num_segments=bins + 1)[:bins]

    safe = jnp.where(causal, scores, 0.0)
    if kind in ("exp", "softmax"):
        weight = jnp.exp(safe - jnp.max(jnp.where(causal, scores, -jnp.inf), axis=1, keepdims=True))
    elif kind == "taylor":
        weight = _taylor_polynomial(safe, degree)
    else:
        weight = jnp.broadcast_to(coefficients[:, degree, None], safe.shape)
        for order in range(degree - 1, -1, -1):
            weight = weight * safe + coefficients[:, order, None]
    magnitude = jnp.where(causal, jnp.abs(weight), 0.0)
    share = magnitude / jnp.sum(magnitude, axis=1, keepdims=True)
    late = jnp.arange(time)[:, None] >= training_length
    return (binned(causal.astype(jnp.float32)), binned(safe), binned(safe * safe),
            binned(jnp.where(causal, scores, jnp.inf), jax.ops.segment_min),
            binned(jnp.where(causal, scores, -jnp.inf), jax.ops.segment_max),
            binned(jnp.where(late & causal, share, 0.0)),
            jnp.sum(jnp.where(late & (distance >= training_length), share, 0.0)))


class AttentionRecorder:
    """Capture the arrays every attention call receives during an eager forward pass."""

    def __init__(self):
        self.calls, self._original = [], None

    def __enter__(self):
        self._original = multihead.attention_core

        def record(q, k, v, **arguments):
            self.calls.append((q, k, arguments))
            return self._original(q, k, v, **arguments)

        multihead.attention_core = record
        return self

    def __exit__(self, *_):
        multihead.attention_core = self._original


def diagnose(graph, params, batches, *, sequences, bin_width, bins, training_length):
    """Accumulate per-layer statistics over ``sequences`` single-sequence forward passes."""
    model, layers = nnx.merge(graph, params), None
    for _ in range(sequences):
        tokens = jnp.asarray(next(batches))
        with AttentionRecorder() as recorder:
            model(tokens[:, :-1], deterministic=True)
        if layers is None:
            layers = [None] * len(recorder.calls)
        for layer, (q, k, arguments) in enumerate(recorder.calls):
            kind, degree = kernel_kind(arguments), arguments.get("attention_degree") or 0
            batch, time, heads, _ = q.shape
            if batch != 1:
                raise ValueError("Use one sequence per forward pass")
            coefficients = arguments.get("query_polynomial_coefficients")
            per_head = []
            for head in range(heads):
                values = head_statistics(
                    q[0, :, head], k[0, :, head],
                    coefficients[0, :, head].astype(jnp.float32) if coefficients is not None else jnp.zeros((time, 1)),
                    kind=kind, degree=degree, bin_width=bin_width, bins=bins, training_length=training_length)
                per_head.append([np.asarray(value, np.float64) for value in values])
            stacked = [np.stack([values[index] for values in per_head]) for index in range(7)]
            if layers[layer] is None:
                layers[layer] = {"kind": kind, "degree": degree, "time": time, "heads": heads, "values": stacked}
            else:
                previous = layers[layer]["values"]
                layers[layer]["values"] = [previous[0] + stacked[0], previous[1] + stacked[1], previous[2] + stacked[2],
                                           np.minimum(previous[3], stacked[3]), np.maximum(previous[4], stacked[4]),
                                           previous[5] + stacked[5], previous[6] + stacked[6]]
    return layers


def summarize(layer: dict, *, sequences: int, bin_width: int, training_length: int) -> dict:
    """Pool heads per bin and give per-head scalars for seen (< training length) and unseen distances."""
    count, total, square, minimum, maximum, late_share, far = layer["values"]
    late_rows = sequences * (layer["time"] - training_length)
    seen = np.arange(count.shape[1]) * bin_width < training_length

    def moments(selection):
        n = count[:, selection].sum(axis=1)
        mean = total[:, selection].sum(axis=1) / n
        return mean, np.sqrt(np.maximum(square[:, selection].sum(axis=1) / n - mean ** 2, 0.0))

    seen_mean, seen_std = moments(seen)
    unseen_mean, unseen_std = moments(~seen)
    return {
        "kind": layer["kind"], "degree": layer["degree"], "heads": layer["heads"],
        "score_count": count.sum(axis=0).tolist(), "score_sum": total.sum(axis=0).tolist(),
        "score_sum_squares": square.sum(axis=0).tolist(),
        "score_min": minimum.min(axis=0).tolist(), "score_max": maximum.max(axis=0).tolist(),
        "late_query_weight_share": (late_share.mean(axis=0) / late_rows).tolist(),
        "head_far_weight_share": (far / late_rows).tolist(),
        "head_score_mean_seen": seen_mean.tolist(), "head_score_std_seen": seen_std.tolist(),
        "head_score_mean_unseen": unseen_mean.tolist(), "head_score_std_unseen": unseen_std.tolist(),
        "head_max_abs_score_seen": np.maximum(np.abs(minimum[:, seen]), np.abs(maximum[:, seen])).max(axis=1).tolist(),
        "head_max_abs_score_unseen": np.maximum(np.abs(minimum[:, ~seen]), np.abs(maximum[:, ~seen])).max(axis=1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise FileExistsError("Diagnostic already completed")
    diagnostic = yaml.safe_load((run / "config.yaml").read_text())["diagnostic"]
    validate_record(run, checkpoint_hashes=False)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use one GPU")
    runtime = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.uname().nodename, "gpu_type": jax.devices()[0].device_kind, "gpu_count": 1,
        "python_executable": sys.executable, "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "versions": {name: version(name) for name in ("jax", "flax", "orbax-checkpoint")},
        "optimizer_updates": 0,
    }
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [runtime]})
    data = diagnostic["data"]
    if sha256(data["manifest"]) != data["manifest_sha256"]:
        raise ValueError(f"Packed-data manifest changed: {data['manifest']}")
    settings = {key: diagnostic[key] for key in ("bin_width", "bins", "training_length")}
    for model in diagnostic["models"]:
        if sha256(model["config_path"]) != model["config_sha256"]:
            raise ValueError(f"Recipe changed: {model['config_path']}")
        verify_checkpoint(model)
        config = load_model_config(model, max_seq_len=data["sequence_length"] - 1)
        graph, params = restore_parent(config, Path(model["checkpoint_path"]))
        batches = PackedBatchIterator(data["manifest"], batch_size=1, sequence_length=data["sequence_length"])
        print(f"Diagnosing {model['role']} ({model['run_id']}, backend {config.model.attention_backend})", flush=True)
        layers = diagnose(graph, params, batches, sequences=data["sequences"], **settings)
        for index, layer in enumerate(layers):
            if layer["kind"] != model["kernel"]:
                raise ValueError(f"{model['role']} layer {index} applies {layer['kind']}, not {model['kernel']}")
            row = {"role": model["role"], "source_run_id": model["run_id"], "checkpoint_step": model["step"],
                   "layer": index, "sequences": data["sequences"], **settings,
                   **summarize(layer, sequences=data["sequences"], bin_width=settings["bin_width"],
                               training_length=settings["training_length"])}
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        far = np.mean([np.mean(json.loads(line)["head_far_weight_share"]) for line in open(run / "metrics.jsonl")
                       if json.loads(line)["role"] == model["role"]])
        print(json.dumps({"role": model["role"], "mean_far_weight_share": float(far)}), flush=True)
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_updates": 0, "interpretation": diagnostic["interpretation"]}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "diagnostic_outcome": outcome})
    print("Attention-score diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
