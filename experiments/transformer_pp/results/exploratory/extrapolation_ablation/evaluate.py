"""Locate why query polynomial attention breaks past its 2,048-token training length.

Two inference-time edits of the validation-best query + W decay (m = 4) checkpoint, scored
per position at 16,384 tokens on the rows its recipe selects (the original 16k evaluation's
held-out rows, or long documents the model never trained on):

- ``window``: each query attends only to its most recent 2,048 keys (distance t - i < 2,048),
  so no key lies beyond the training distance and no query sees more keys than in training.
- ``clamp``: each query attends to every key, but a key farther than 2,047 tokens back gets
  the RoPE phase of distance 2,047, so every relative phase was seen in training while the
  number of keys keeps growing.

A ``full`` pass with the same implementation and the original causal mask and RoPE is the
control: at 2,048 tokens it must replay the recorded validation NLL, and at 16,384 tokens its
first 2,048 positions must match the reference Mosaic evaluation of the same rows. At 2,048 tokens ``window``
and ``clamp`` are mathematically identical to ``full``; ``clamp`` is replayed there too, since
its implementation differs. Nothing is trained or updated; only the attention call is replaced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.transformer_pp.evaluate_per_position import (
    extend_forward_length, load_model_config, manifest_metadata, per_position_nll, require_unseen, sha256,
    verify_checkpoint,
)
from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.training.coefficient_fit import restore_parent

ROOT = Path(__file__).resolve().parents[5]
MODES = ("full", "window", "clamp")


def polynomial_attention(q, k, v, coefficients, *, mode: str, window: int, rope_theta: float,
                         block: int = 256) -> jax.Array:
    """Unnormalized query-coefficient polynomial attention with an ablated mask or RoPE.

    ``q, k, v`` are [batch, time, heads, width]; ``coefficients`` [batch, time, heads, m+1]
    are each query's effective monomial coefficients. ``full`` and ``window`` receive post-RoPE
    Q/K; ``clamp`` receives them before RoPE and applies it here: true positions for keys
    within ``window - 1`` tokens, and the fixed relative phase of distance ``window - 1`` for
    farther keys (queries rotated to position ``window - 1``, keys left at position 0).
    Scores, weights and the value sum are FP32, as in the blocked reference; the result has
    V's dtype.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown attention ablation: {mode}")
    batch, length, heads, width = q.shape
    block = math.gcd(length, block)
    scale = 1.0 / math.sqrt(width)
    if mode == "clamp":
        positions = jnp.arange(length)
        near_q = apply_rope(q, positions, rope_theta).astype(q.dtype)
        near_k = apply_rope(k, positions, rope_theta).astype(k.dtype)
        far_q = apply_rope(q, jnp.full((length,), window - 1), rope_theta).astype(q.dtype)
        far_k = k
    else:
        near_q, near_k = q, k
    values = v.astype(jnp.float32)
    coefficients = coefficients.astype(jnp.float32)
    keys = jnp.arange(length)

    def weights(queries, all_keys, query_coefficients):
        scores = jnp.einsum("bqhd,bkhd->bhqk", queries, all_keys, preferred_element_type=jnp.float32) * scale
        c = jnp.transpose(query_coefficients, (0, 2, 1, 3))[..., None, :]  # [B, H, Q, 1, m+1]
        result = c[..., -1]
        for order in range(c.shape[-1] - 2, -1, -1):
            result = result * scores + c[..., order]
        return result

    def one_block(start):
        query_block = lax.dynamic_slice_in_dim(near_q, start, block, axis=1)
        coefficient_block = lax.dynamic_slice_in_dim(coefficients, start, block, axis=1)
        distance = (start + jnp.arange(block))[:, None] - keys[None, :]
        causal = distance >= 0
        near = causal & (distance < window) if mode != "full" else causal
        w = jnp.where(near, weights(query_block, near_k, coefficient_block), 0.0)
        if mode == "clamp":
            far_block = lax.dynamic_slice_in_dim(far_q, start, block, axis=1)
            w = w + jnp.where(causal & (distance >= window), weights(far_block, far_k, coefficient_block), 0.0)
        return jnp.einsum("bhqk,bkhd->bqhd", w, values)

    out = lax.map(one_block, jnp.arange(0, length, block))
    return jnp.moveaxis(out, 0, 1).reshape(batch, length, heads, -1).astype(v.dtype)


@contextmanager
def ablated_attention(mode: str, *, window: int, rope_theta: float, block: int):
    """Replace the attention module's kernel call (and, for ``clamp``, its RoPE) for one pass."""
    def core(q, k, v, *, query_polynomial_coefficients=None, attention_normalization=None,
             return_diagnostics=False, **_):
        if query_polynomial_coefficients is None or attention_normalization or return_diagnostics:
            raise ValueError("The ablation expects unnormalized query-coefficient attention without diagnostics")
        return polynomial_attention(q, k, v, query_polynomial_coefficients, mode=mode, window=window,
                                    rope_theta=rope_theta, block=block)

    original_core, original_rope = multihead.attention_core, multihead.apply_rope
    multihead.attention_core = core
    if mode == "clamp":
        multihead.apply_rope = lambda x, positions, theta: x
    try:
        yield
    finally:
        multihead.attention_core, multihead.apply_rope = original_core, original_rope


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
        "optimizer_updates": 0,
    }
    record = load_run(run)
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    lengths, model = evaluation["lengths"], evaluation["model"]
    for spec in lengths.values():
        if sha256(spec["manifest"]) != spec["manifest_sha256"]:
            raise ValueError(f"Packed-data manifest changed: {spec['manifest']}")
    if sha256(model["config_path"]) != model["config_sha256"]:
        raise ValueError(f"Recipe changed: {model['config_path']}")
    reference = evaluation["reference"]
    if sha256(reference["metrics_path"]) != reference["metrics_sha256"]:
        raise ValueError(f"Reference measurements changed: {reference['metrics_path']}")
    reference_curve = next(np.asarray(row["per_position_nll"], dtype=np.float64)
                           for row in map(json.loads, Path(reference["metrics_path"]).read_text().splitlines())
                           if row["role"] == reference["role"] and row["length"] == "evaluation")
    print(f"Verifying checkpoint of {model['run_id']}", flush=True)
    verify_checkpoint(model)
    metrics_path = run / "metrics.jsonl"
    completed = {}
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            completed[(row["mode"], row["length"])] = row
    window, trained = evaluation["window"], model["training_sequence_length"] - 1
    outcomes = []
    for item in evaluation["passes"]:
        mode, name = item["mode"], item["length"]
        spec = lengths[name]
        limit = spec["sequence_length"] - 1
        if (mode, name) in completed:
            outcomes.append({key: value for key, value in completed[(mode, name)].items() if key != "per_position_nll"})
            print(f"Reusing completed {mode} pass at {limit} tokens", flush=True)
            continue
        config = load_model_config(model)
        start = spec.get("start_sequence", 0)
        details = {"start_sequence": start} if start else {}
        if "long_documents_from" in manifest_metadata(spec["manifest"]):
            details.update(require_unseen(config, spec["manifest"], start))
        graph, params = restore_parent(config, Path(model["checkpoint_path"]))
        if name != "training":
            graph, params = extend_forward_length(graph, params, max_seq_len=limit)
        started = time.perf_counter()
        with ablated_attention(mode, window=window, rope_theta=config.model.rope_theta, block=evaluation["block_size"]):
            curve = per_position_nll(graph, params, spec["manifest"], sequence_length=spec["sequence_length"],
                                     sequences=spec["sequences"], microbatch=spec["microbatch_size"], details=details,
                                     start_sequence=start)
        nll = float(curve.mean())
        row = {"mode": mode, "length": name, "source_run_id": model["run_id"], "checkpoint_step": model["step"],
               "window": window, "sequence_length": spec["sequence_length"], "sequences": spec["sequences"],
               "prediction_tokens": spec["sequences"] * limit, "nll": nll, "perplexity": math.exp(nll),
               "seconds": round(time.perf_counter() - started, 1), **details}
        if name == "training":
            error = abs(nll - model["recorded_validation_nll"])
            row.update(recorded_validation_nll=model["recorded_validation_nll"], replay_absolute_error=error)
            if error > evaluation["replay_nll_atol"]:
                raise ValueError(f"{mode} replay differs: {nll} versus {model['recorded_validation_nll']}")
        else:
            row["nll_within_training_length"] = float(curve[:trained].mean())
            row["nll_beyond_training_length"] = float(curve[trained:].mean())
            if mode == "full":
                prefix_error = abs(float(curve[:trained].mean()) - float(reference_curve[:trained].mean()))
                row["reference_prefix_mean_nll_absolute_error"] = prefix_error
                row["reference_mean_nll_absolute_error"] = abs(nll - float(reference_curve.mean()))
                if prefix_error > evaluation["reference_prefix_nll_atol"]:
                    raise ValueError(f"Full control differs from the original evaluation by {prefix_error} nats")
        print(json.dumps(row), flush=True)
        outcomes.append(dict(row))
        with metrics_path.open("a") as stream:
            stream.write(json.dumps({**row, "per_position_nll": curve.tolist()}, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        del graph, params
        jax.clear_caches()
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_updates": 0, "measurements": outcomes, "interpretation": evaluation["interpretation"]}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "evaluation_outcome": outcome})
    print("Extrapolation ablation complete", flush=True)


if __name__ == "__main__":
    main()
