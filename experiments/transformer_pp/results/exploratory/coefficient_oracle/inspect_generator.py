"""Measure fitted query coefficients and ablate their dependence, without training.

Run on one GPU; compact measurements are saved beside the selected run's metrics.
The mean-coefficient ablation is not a refitted static optimum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml

from scaffolds.records import validate_record
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import (
    parent_config, restore_parent, split_generator_parameters,
)


ROOT = Path(__file__).resolve().parents[5]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(values):
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def parameter_summary(generator):
    values = {str(path[-1]): np.asarray(value[...], dtype=np.float64)
              for path, value in nnx.to_flat_state(generator)}
    b = values["query_polynomial_bias"]
    a = values["query_polynomial_amplitude"]
    return {
        "bias_rms_deviation_from_one": rms(b - 1),
        "bias_mean_absolute_deviation_from_one": float(np.mean(np.abs(b - 1))),
        "amplitude_rms": rms(a),
        "amplitude_mean_absolute": float(np.mean(np.abs(a))),
        "projection_rms": rms(values["query_polynomial_kernel"]),
    }


def coefficient_summary(coefficients, factorials):
    # Taylor has value one in every coordinate after multiplying by r!.
    relative = np.asarray(coefficients, dtype=np.float64) * factorials
    mean = relative.mean(axis=2, keepdims=True)
    deviations = relative - 1
    variations = relative - mean
    unit = relative / np.maximum(np.linalg.norm(relative, axis=-1, keepdims=True), 1e-30)
    taylor_unit = np.ones(relative.shape[-1]) / math.sqrt(relative.shape[-1])
    cosine = np.sum(unit * taylor_unit, axis=-1)
    return {
        "coordinates": "r! times effective monomial coefficients; Taylor is all ones",
        "rms_deviation_from_taylor": rms(deviations),
        "mean_absolute_deviation_from_taylor": float(np.mean(np.abs(deviations))),
        "rms_query_variation_about_layer_head_order_mean": rms(variations),
        "query_variation_fraction_of_squared_taylor_deviation": float(
            np.mean(variations ** 2) / np.mean(deviations ** 2)),
        "per_order_rms_deviation_from_taylor": [rms(deviations[..., r])
                                                 for r in range(relative.shape[-1])],
        "per_order_rms_query_variation": [rms(variations[..., r])
                                           for r in range(relative.shape[-1])],
        "negative_coefficient_fraction": float(np.mean(relative < 0)),
        "cosine_with_taylor_direction_quantiles": np.quantile(cosine, [0, .1, .5, .9, 1]).tolist(),
        "rms_unit_direction_query_variation": rms(unit - unit.mean(axis=2, keepdims=True)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "generator_dependence.json"
    if output.exists():
        raise FileExistsError(output)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use one GPU")
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle = recipe["oracle"]
    record = json.loads((run / "record.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    if oracle["parameterization"] != "query_coefficient_generator" or oracle["sequence_count"] != 1:
        raise ValueError("Requires a single-sequence native generator fit")
    validate_record(run, checkpoint_hashes=False)
    parent_recipe = Path(oracle["parent_config"])
    if sha256(parent_recipe) != oracle["parent_config_sha256"]:
        raise ValueError("Parent recipe checksum changed")
    validate_record(parent_recipe.parent, checkpoint_hashes=False)
    parent_record = json.loads((parent_recipe.parent / "record.json").read_text())
    if parent_record["selected_checkpoint"]["sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint identity changed")
    checkpoint = Path(record["selected_checkpoint"]["path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    identity = json.loads((checkpoint / "coefficient_identity.json").read_text())
    if (identity["coordinates"] != "stored_generator_parameters"
            or identity["step"] != completion["best_step"]):
        raise ValueError("Unexpected selected generator checkpoint")
    config = parent_config(parent_recipe)
    if sha256(config.data.eval_manifest) != oracle["eval_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    batch = next(iter(PackedBatchIterator(config.data.eval_manifest, batch_size=1,
        sequence_length=config.data.sequence_length, start_sequence=oracle["data_start"])))
    batch_hash = hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest()
    if batch_hash != oracle["batch_sha256"]:
        raise ValueError("Fitting sequence changed")
    print("Restoring frozen backbone and fitted generator", flush=True)
    graph, source = restore_parent(config, Path(oracle["parent_checkpoint"]))
    original_generator, frozen = split_generator_parameters(source)
    with ocp.StandardCheckpointer() as checkpointer:
        fitted = checkpointer.restore(checkpoint / "best", target={
            "generator_parameters": [original_generator],
        })["generator_parameters"][0]
    tokens = jax.device_put(batch)

    @jax.jit
    def capture(generator, backbone, inputs):
        logits, coefficients = nnx.merge(graph, backbone, generator)(
            inputs[:, :-1], deterministic=True, return_coefficients=True)
        nll = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), inputs[:, 1:]).mean()
        return nll, coefficients

    @jax.jit
    def evaluate_override(generator, backbone, inputs, coefficients):
        logits = nnx.merge(graph, backbone, generator)(inputs[:, :-1], deterministic=True,
                                                      query_coefficients=coefficients)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), inputs[:, 1:]).mean()

    print("Capturing native coefficients before and after fitting", flush=True)
    before_nll, before = jax.device_get(capture(original_generator, frozen, tokens))
    after_nll, after = jax.device_get(capture(fitted, frozen, tokens))
    tolerance = .002 if config.train.dtype == "bfloat16" else 1e-5
    if abs(float(before_nll) - completion["initial_nll"]) > tolerance:
        raise ValueError("Initial forward differs from fitting record")
    if abs(float(after_nll) - completion["best_fit_nll"]) > tolerance:
        raise ValueError("Fitted forward differs from selected checkpoint loss")
    factorials = np.asarray([math.factorial(r) for r in range(config.model.attention_degree + 1)])
    mean = np.broadcast_to(after.mean(axis=2, keepdims=True), after.shape).copy()
    taylor = np.broadcast_to(1. / factorials, after.shape).astype(np.float32).copy()
    measurements = {}
    for label, coefficients in (("fitted_coefficients_replayed", after),
                                ("query_mean_coefficients", mean),
                                ("fixed_taylor_coefficients", taylor)):
        value = float(evaluate_override(fitted, frozen, tokens, jax.device_put(coefficients)))
        measurements[label] = {"nll": value, "perplexity": math.exp(value)}
        print(label + ": " + json.dumps(measurements[label]), flush=True)
    if abs(measurements["fitted_coefficients_replayed"]["nll"] - float(after_nll)) > tolerance:
        raise ValueError("Captured coefficient replay differs from native forward")
    if not all(np.isfinite(value).all() for value in (before, after)):
        raise ValueError("Nonfinite coefficients")
    output.write_text(json.dumps({
        "schema_version": 1, "run_id": run.name, "parent_run_id": oracle["parent_run_id"],
        "selected_checkpoint_step": identity["step"],
        "checkpoint_sha256": record["selected_checkpoint"]["sha256"],
        "parent_checkpoint_sha256": oracle["parent_checkpoint_sha256"],
        "batch_sha256": batch_hash, "sequence_count": 1, "prediction_positions": batch.shape[1] - 1,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": jax.devices()[0].device_kind, "cuda_environment": os.environ.get("TRANSFORMER_CUDA_ENVIRONMENT"),
        "coefficient_shape": list(after.shape), "nll_replay_atol": tolerance,
        "native_forward": {"before_nll": float(before_nll), "after_nll": float(after_nll)},
        "parameter_summary": {"before": parameter_summary(original_generator), "after": parameter_summary(fitted)},
        "coefficient_summary": {"before": coefficient_summary(before, factorials),
                                "after": coefficient_summary(after, factorials)},
        "ablations": measurements,
        "ablation_definition": "Replace all layers' effective coefficients by their fitted-sequence query mean per layer/head/order; or by 1/r!. Recompute activations, with every parameter fixed. All ablations use the same compiled override graph.",
        "limits": "One fitted sequence only; averaging ablation is not a refitted static optimum or a test of generalization. Coefficient distances depend on coordinates; head RMSNorm makes positive common coefficient scales mostly redundant.",
    }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
