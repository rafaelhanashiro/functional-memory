"""Forward-only score calibration for fixed-knot polynomial splines.

Every snapshot sees the same declared training rows and sampling indices. Query rows
are sampled uniformly without replacement; all their causal keys contribute to
histograms, moments and tail-weight shares. An independent uniform sample of causal
pairs supplies score quantiles. Quantiles of per-query extrema use the sampled query
rows. Scores are recomputed in FP32/HIGHEST from the operands the original attention
backend receives; they are diagnostic arithmetic, not a bitwise kernel trace.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
import yaml

from experiments.transformer_pp.diagnose_attention_scores import AttentionRecorder
from experiments.transformer_pp.evaluate_per_position import load_model_config, sha256, verify_checkpoint
from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.data.packed import PackedBatchIterator, load_packed_manifest
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.coefficient_fit import restore_parent

ROOT = Path(__file__).resolve().parents[2]
SAMPLING_LAW = {
    "query_rows": "uniform without replacement within each sequence; all causal keys retained",
    "pair_samples": "independent uniform causal-pair sample with replacement within each sequence",
    "snapshots": "identical data rows and sampling indices for every model and state",
    "weight_share": "mean across sampled queries of sum_outside(abs(p(score)))/sum_causal(abs(p(score))); zero-weight rows contribute zero and are counted separately",
}


def sample_indices(length: int, queries: int, pairs: int, seed: int, sequence_index: int):
    if not 0 < queries <= length or pairs <= 0:
        raise ValueError("Sampling requires 0 < queries <= context length and positive pair count")
    rng = np.random.default_rng(np.random.SeedSequence([seed, sequence_index]))
    rows = np.sort(rng.choice(length, size=queries, replace=False)).astype(np.int32)
    flat = rng.integers(length * (length + 1) // 2, size=pairs, dtype=np.int64)
    pair_q = np.floor((np.sqrt(8.0 * flat + 1.0) - 1.0) / 2.0).astype(np.int64)
    pair_k = flat - pair_q * (pair_q + 1) // 2
    return rows, pair_q.astype(np.int32), pair_k.astype(np.int32)


def _head_statistics(q, k, coefficients, rows, pair_q, pair_k, radii, edges):
    scores = jnp.matmul(q[rows].astype(jnp.float32), k.astype(jnp.float32).T,
                        precision=jax.lax.Precision.HIGHEST) / math.sqrt(q.shape[-1])
    causal = jnp.arange(k.shape[0])[None, :] <= rows[:, None]
    polynomial_scores = jnp.where(causal, scores, 0.0)
    selected = coefficients[rows].astype(jnp.float32)
    weight = jnp.broadcast_to(selected[:, -1, None], scores.shape)
    for order in range(selected.shape[-1] - 2, -1, -1):
        weight = weight * polynomial_scores + selected[:, order, None]
    magnitude = jnp.where(causal, jnp.abs(weight), 0.0)
    denominator = magnitude.sum(axis=1)
    outside = causal[None] & (jnp.abs(scores)[None] > radii[:, None, None])
    tail_share = jnp.sum(jnp.where(outside, magnitude[None], 0.0), axis=2) / jnp.where(denominator > 0, denominator, 1.0)[None]
    safe = jnp.where(causal, scores, 0.0)
    samples = jnp.sum(q[pair_q].astype(jnp.float32) * k[pair_k].astype(jnp.float32), axis=-1)
    samples = samples / math.sqrt(q.shape[-1])
    minimum = jnp.min(jnp.where(causal, scores, jnp.inf), axis=1)
    maximum = jnp.max(jnp.where(causal, scores, -jnp.inf), axis=1)
    absmax = jnp.maximum(jnp.abs(minimum), jnp.abs(maximum))
    return {
        "score_count": causal.sum(), "score_sum": safe.sum(), "score_square_sum": (safe * safe).sum(),
        "zero_weight_query_count": jnp.sum(denominator == 0),
        "histogram_counts": jnp.histogram(scores, bins=edges, weights=causal.astype(jnp.int32))[0],
        "histogram_underflow": jnp.sum(causal & (scores < edges[0])),
        "histogram_overflow": jnp.sum(causal & (scores > edges[-1])),
        "pair_outside_count": jnp.sum(outside, axis=(1, 2)),
        "weight_share_sum": jnp.sum(tail_share, axis=1),
        "query_peak_outside_count": jnp.sum(absmax[None] > radii[:, None], axis=1),
        "pair_samples": samples, "query_min": minimum, "query_max": maximum, "query_absmax": absmax,
        "finite": (jnp.all(jnp.isfinite(scores)) & jnp.all(jnp.isfinite(samples))
                   & jnp.all(jnp.isfinite(weight)) & jnp.all(jnp.isfinite(denominator))
                   & jnp.all(jnp.isfinite(tail_share))),
    }


@jax.jit
def layer_statistics(q, k, coefficients, rows, pair_q, pair_k, radii, edges):
    """One layer, q/k [T,H,D], effective coefficients [T,H,R]; outputs have leading H."""
    return jax.vmap(_head_statistics, in_axes=(1, 1, 1, None, None, None, None, None))(
        q, k, coefficients, rows, pair_q, pair_k, radii, edges)


def effective_coefficients(q, arguments, conditioning):
    """Read the already-scaled coefficients captured at the actual attention call."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Score calibration uses one sequence per forward pass")
    if arguments.get("attention_normalization", True):
        raise ValueError("Expected unnormalized polynomial attention")
    if any(arguments.get(name) is not None for name in (
            "query_log_decay", "query_degree_log_decay", "query_gate_input", "key_gate_input",
            "key_polynomial_factors", "polynomial_square_factors", "coefficients")):
        raise ValueError("Calibration controls must be plain direct static/query polynomials without FoX")
    degree = arguments.get("attention_degree")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("Expected a positive polynomial degree")
    expected = (*q.shape[:3], degree + 1)
    if conditioning == "query":
        values = arguments.get("query_polynomial_coefficients")
        if values is None or values.shape != expected or arguments.get("polynomial_coefficients") is not None:
            raise ValueError("Missing or malformed query-dependent coefficients")
    elif conditioning == "static":
        values = arguments.get("polynomial_coefficients")
        if values is None or arguments.get("query_polynomial_coefficients") is not None:
            raise ValueError("Missing static learned coefficients")
        if values.shape not in ((degree + 1,), (q.shape[2], degree + 1)):
            raise ValueError(f"Malformed static coefficients: {values.shape}")
        values = jnp.broadcast_to(values, expected)
    else:
        raise ValueError(f"Unsupported conditioning: {conditioning}")
    return values[0].astype(jnp.float32)


def _initial_parameters(config):
    rng = jax.random.PRNGKey(config.train.seed)
    model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    return nnx.split(model, nnx.Param)


def _check_source(config):
    model = config.model
    if (model.attention_mechanism != "dot_product" or model.attention_normalization
            or model.attention_coefficients != "learned"
            or model.attention_coefficient_parameterization != "direct"
            or model.attention_coefficient_conditioning not in ("static", "query")
            or model.attention_forget_gate or model.positional_encoding not in ("rope", "none")
            or model.remat):
        raise ValueError("Expected an eager, ungated, unnormalized learned static/query polynomial control")
    if not model.qk_norm or model.qk_norm_type != "rms_norm" or not model.qk_norm_head_scale:
        raise ValueError("Controls must use QK RMSNorm with learned per-head score scaling")
    if model.attention_output_scaling != "head_rms_norm":
        raise ValueError("Controls must use attention-output head RMSNorm")


def _accumulate(accumulator, values):
    if not np.all(values["finite"]):
        raise ValueError("Nonfinite score/weight statistic")
    vectors = {"pair_samples", "query_min", "query_max", "query_absmax"}
    for key, value in values.items():
        if key == "finite":
            continue
        value = np.asarray(value)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Nonfinite diagnostic statistic: {key}")
        if key in vectors:
            accumulator.setdefault(key, []).append(value)
        else:
            cast = value.astype(np.int64 if np.issubdtype(value.dtype, np.integer) else np.float64)
            accumulator[key] = accumulator.get(key, 0) + cast


def summarize(accumulator, *, quantiles, radii):
    """Yield one head summary; quantiles are exact for the declared retained samples."""
    samples = {key: np.concatenate(accumulator[key], axis=1)
               for key in ("pair_samples", "query_min", "query_max", "query_absmax")}
    for head in range(samples["pair_samples"].shape[0]):
        count = int(accumulator["score_count"][head])
        query_count = samples["query_min"].shape[1]
        mean = float(accumulator["score_sum"][head] / count)
        variance = float(accumulator["score_square_sum"][head] / count - mean * mean)
        def quantile(values):
            return {str(p): float(x) for p, x in zip(quantiles, np.quantile(values, quantiles))}
        row = {
            "head": head, "score_count": count, "score_mean": mean,
            "score_std": math.sqrt(max(0.0, variance)),
            "score_min": float(samples["query_min"][head].min()),
            "score_max": float(samples["query_max"][head].max()),
            "sampled_pair_count": samples["pair_samples"].shape[1], "sampled_query_count": query_count,
            "zero_weight_query_count": int(accumulator["zero_weight_query_count"][head]),
            "score_quantiles": quantile(samples["pair_samples"][head]),
            "absolute_score_quantiles": quantile(np.abs(samples["pair_samples"][head])),
            **{key + "_quantiles": quantile(samples[key][head])
               for key in ("query_min", "query_max", "query_absmax")},
            "histogram_counts": accumulator["histogram_counts"][head].astype(np.int64).tolist(),
            "histogram_underflow": int(accumulator["histogram_underflow"][head]),
            "histogram_overflow": int(accumulator["histogram_overflow"][head]),
            "radius_statistics": [{
                "radius": float(radius),
                "causal_pair_fraction_outside": float(accumulator["pair_outside_count"][head, index] / count),
                "mean_query_absolute_weight_share_outside": float(accumulator["weight_share_sum"][head, index] / query_count),
                "query_peak_fraction_outside": float(accumulator["query_peak_outside_count"][head, index] / query_count),
            } for index, radius in enumerate(radii)],
        }
        if sum(row["histogram_counts"]) + row["histogram_underflow"] + row["histogram_overflow"] != count:
            raise ValueError("Histogram does not account for every sampled-row causal score")
        yield row


def _self_check():
    """Independent NumPy dense reference on a small fixture, run on the allocated device."""
    rng = np.random.default_rng(17)
    q = rng.normal(size=(9, 2, 4)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    coefficients = rng.normal(size=(9, 2, 3)).astype(np.float32)
    rows, pq, pk = sample_indices(9, 5, 31, 42, 0)
    radii, edges = np.array([0.25, 1.0, 3.0], np.float32), np.array([-8, -1, 0, 1, 8], np.float32)
    actual = jax.device_get(layer_statistics(q, k, coefficients, rows, pq, pk, radii, edges))
    errors = []
    for head in range(2):
        dense = q[:, head].astype(np.float64) @ k[:, head].astype(np.float64).T / 2
        valid = [dense[t, :t + 1] for t in rows]
        weights = [np.abs(sum(coefficients[t, head, r] * dense[t, :t + 1] ** r for r in range(3))) for t in rows]
        values = np.concatenate(valid)
        expected = {
            "score_count": len(values), "score_sum": values.sum(), "score_square_sum": (values * values).sum(),
            "zero_weight_query_count": 0,
            "histogram_counts": np.histogram(values, edges)[0],
            "histogram_underflow": (values < edges[0]).sum(), "histogram_overflow": (values > edges[-1]).sum(),
            "pair_outside_count": [(np.abs(values) > radius).sum() for radius in radii],
            "weight_share_sum": [sum(w[np.abs(s) > radius].sum() / w.sum() for w, s in zip(weights, valid)) for radius in radii],
            "query_peak_outside_count": [sum(np.max(np.abs(s)) > radius for s in valid) for radius in radii],
            "pair_samples": dense[pq, pk], "query_min": [s.min() for s in valid],
            "query_max": [s.max() for s in valid], "query_absmax": [np.max(np.abs(s)) for s in valid],
            "finite": True,
        }
        for name, want in expected.items():
            got = actual[name][head]
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5, err_msg=name)
            errors.append(float(np.max(np.abs(np.asarray(got, float) - np.asarray(want, float)))))
    return {"status": "passed", "method": "allocated-device statistics against independent NumPy FP64 dense reference",
            "max_absolute_error": max(errors), "rtol": 2e-5, "atol": 2e-5}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise FileExistsError("Calibration is already complete")
    diagnostic = yaml.safe_load((run / "config.yaml").read_text())["diagnostic"]
    validate_record(run, checkpoint_hashes=False)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Calibration requires one GPU")
    data, sampling = diagnostic["data"], diagnostic["sampling"]
    radii = np.asarray(diagnostic["radii"], np.float32)
    edges = np.asarray(diagnostic["histogram_edges"], np.float32)
    quantiles = diagnostic["quantiles"]
    if (radii.ndim != 1 or len(radii) == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0)
            or np.any(np.diff(radii) <= 0) or edges.ndim != 1 or len(edges) < 2
            or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0)
            or not quantiles or any(not 0 <= p <= 1 for p in quantiles)):
        raise ValueError("Invalid radii, histogram edges or quantiles")
    if data["sequences"] <= 0 or data.get("sequence_stride", 1) <= 0:
        raise ValueError("Positive sequence count and stride are required")
    runtime = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"), "node": os.uname().nodename,
        "gpu_type": jax.devices()[0].device_kind, "gpu_count": 1, "optimizer_updates": 0,
        "python_executable": sys.executable, "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "versions": {name: version(name) for name in ("jax", "flax", "orbax-checkpoint")},
        "environment": {key: os.environ.get(key) for key in ("CUDA_HOME", "XLA_FLAGS", "TRANSFORMER_CUDA_ENVIRONMENT")},
    }
    record = load_run(run)
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*record.get("attempts", []), runtime]})
    acceptance = _self_check()
    update_record(run, {"acceptance": acceptance})
    print(json.dumps({"statistics_acceptance": acceptance}), flush=True)
    if sha256(data["manifest"]) != data["manifest_sha256"]:
        raise ValueError("Training manifest has changed")
    print(json.dumps({"phase": "verify_training_data", "manifest": data["manifest"]}), flush=True)
    validated_manifest = load_packed_manifest(data["manifest"])
    print(json.dumps({"phase": "training_data_verified"}), flush=True)
    manifest = validated_manifest[1]
    sequence_rows = [data.get("start_sequence", 0) + i * data.get("sequence_stride", 1) for i in range(data["sequences"])]
    if min(sequence_rows) < 0 or max(sequence_rows) >= manifest["num_sequences"]:
        raise ValueError("Calibration row lies outside the training corpus")
    snapshot_summaries = []
    temporary_metrics = run / ".metrics.jsonl.tmp"
    # A retry restarts this read-only diagnostic; only a complete set is published.
    with temporary_metrics.open("w") as stream:
        for spec in diagnostic["models"]:
            if sha256(spec["config_path"]) != spec["config_sha256"]:
                raise ValueError(f"Source recipe changed: {spec['config_path']}")
            print(json.dumps({"phase": "verify_checkpoint", "role": spec["role"],
                              "source_run_id": spec["run_id"]}), flush=True)
            verify_checkpoint(spec)
            config = load_model_config(spec)
            _check_source(config)
            if (Path(config.data.train_manifest).resolve() != Path(data["manifest"]).resolve()
                    or config.data.sequence_length != data["sequence_length"]
                    or config.model.max_seq_len != data["sequence_length"] - 1):
                raise ValueError("Calibration must use the control's training corpus and context length")
            for state in spec.get("states", ["initialization", "checkpoint"]):
                if state not in ("initialization", "checkpoint"):
                    raise ValueError(f"Unknown snapshot state: {state}")
                started = time.monotonic()
                graph, params = (_initial_parameters(config) if state == "initialization"
                                 else restore_parent(config, Path(spec["checkpoint_path"])))
                model = nnx.merge(graph, params)
                accumulators = [dict() for _ in range(config.model.num_layers)]
                print(json.dumps({"role": spec["role"], "state": state, "phase": "forward",
                                  "backend": model.config.attention_backend}), flush=True)
                for sequence_index, sequence_row in enumerate(sequence_rows):
                    batch = next(PackedBatchIterator(data["manifest"], batch_size=1,
                        sequence_length=data["sequence_length"], start_sequence=sequence_row,
                        validated_manifest=validated_manifest))
                    indices = sample_indices(data["sequence_length"] - 1, sampling["queries_per_sequence"],
                                             sampling["pairs_per_sequence"], sampling["seed"], sequence_index)
                    with AttentionRecorder() as recorder:
                        logits = model(jnp.asarray(batch[:, :-1]), deterministic=True)
                    if len(recorder.calls) != config.model.num_layers:
                        raise ValueError("Did not capture exactly one attention call per layer")
                    if not bool(jax.device_get(jnp.all(jnp.isfinite(logits)))):
                        raise ValueError("Source model produced nonfinite logits")
                    for layer, (q, k, arguments) in enumerate(recorder.calls):
                        if q.shape != k.shape or q.shape[2] != config.model.num_heads:
                            raise ValueError("Expected matched query/key heads for these controls")
                        coefficients = effective_coefficients(q, arguments, config.model.attention_coefficient_conditioning)
                        values = jax.device_get(layer_statistics(q[0], k[0], coefficients, *indices, radii, edges))
                        _accumulate(accumulators[layer], values)
                    print(json.dumps({"role": spec["role"], "state": state, "sequences_done": sequence_index + 1,
                                      "sequences_total": len(sequence_rows), "training_row": sequence_row,
                                      "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
                    del logits, recorder
                for layer, accumulator in enumerate(accumulators):
                    for row in summarize(accumulator, quantiles=quantiles, radii=radii):
                        row.update({"role": spec["role"], "state": state, "source_run_id": spec["run_id"],
                                    "checkpoint_step": 0 if state == "initialization" else spec["step"],
                                    "initialization_kind": "reconstructed_from_source_recipe" if state == "initialization" else None,
                                    "layer": layer, "sequences": len(sequence_rows),
                                    "degree": config.model.attention_degree})
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                snapshot_summaries.append({"role": spec["role"], "state": state, "source_run_id": spec["run_id"],
                                           "initialization_kind": "reconstructed_from_source_recipe" if state == "initialization" else None,
                                           "elapsed_seconds": round(time.monotonic() - started, 3),
                                           "layers": config.model.num_layers, "heads": config.model.num_heads,
                                           "backend": model.config.attention_backend})
                del model, graph, params, accumulators
                gc.collect()
    os.replace(temporary_metrics, run / "metrics.jsonl")
    metadata = {"status": "complete", "runtime": runtime, "statistics_acceptance": acceptance,
                "sampling_law": SAMPLING_LAW, "training_sequence_rows": sequence_rows,
                "histogram_edges": edges.tolist(), "snapshots": snapshot_summaries}
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_updates": 0, "snapshots": snapshot_summaries, "interpretation": diagnostic["interpretation"]}
    _write_json(run / "metadata.json", metadata)
    _write_json(run / "completion.json", outcome)
    update_record(run, {**capture_artifacts(run), "status": "complete", "diagnostic_outcome": outcome})
    print("Spline score calibration complete", flush=True)


if __name__ == "__main__":
    main()
