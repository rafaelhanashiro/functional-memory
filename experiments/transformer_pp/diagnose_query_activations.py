"""Forward-only probe of one head's tanh coefficient inputs on a pinned held-out row.

The native model forward is unchanged. Temporary Python wrappers return the exact
pre-attention normalization inputs/outputs and coefficient-generator operands as JIT
auxiliaries. Parent parameters and checkpoint files are read-only; no optimizer exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.transformer_pp.evaluate_per_position import load_model_config, sha256, verify_checkpoint
from scaffolds.records import capture_artifacts, update_record, validate_record
from scaffolds.transformer.attention import input_dependent
from scaffolds.transformer.models.components.normalization import RMSNorm
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import restore_parent
from scaffolds.transformer.training.diagnostics import fixed_diagnostic_sample
from scaffolds.transformer.training.loop import cross_entropy_loss

ROOT = Path(__file__).resolve().parents[2]
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
# Declared before inspecting replay results. Historical moments used the blocked
# diagnostic backend; this probe preserves the parent recipe's native backend.
REFERENCE_MOMENT_TOLERANCE = 0.02
MANUAL_COEFFICIENT_TOLERANCE = 5e-6


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise FloatingPointError("Probe summaries require nonempty finite arrays")
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))}}


def capture_forward(graphdef, *, layer, head, layers):
    """Return scalar NLL and captures; no traced arrays escape the returned pytree."""
    original_projection, original_norm = input_dependent.project_coefficients, RMSNorm.__call__

    def forward(params, batch):
        model = nnx.merge(graphdef, params)
        selected_norm = getattr(model, model.layer_names[layer]).attention_norm
        captured, calls, norm_calls = {}, [0], [0]

        def normalized(module, value):
            output = original_norm(module, value)
            if module is selected_norm:
                norm_calls[0] += 1
                captured["x_before"] = value[0]
                captured["x_after"] = output[0]
                captured["gain"] = module.scale[...]
            return output

        def projected(x, kernel, bias, **kwargs):
            index = calls[0]
            calls[0] += 1
            coefficients = original_projection(x, kernel, bias, **kwargs)
            if index == layer:
                if (kwargs.get("activation") != "tanh" or kwargs.get("mixing") is not None
                        or kwargs.get("tanh_bias") is not None or kwargs.get("orders") is not None
                        or kwargs.get("gate_kernel") is not None):
                    raise ValueError("Probe requires ordinary diagonal tanh query coefficients")
                # Match the native full-head FP32 projection, then select the head.
                z = jnp.einsum("btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
                               precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
                features = jax.lax.optimization_barrier(jnp.tanh(z))
                captured.update(x_projection=x[0], kernel=kernel[head],
                                amplitude=kwargs["amplitude"][head], bias_effective=bias[head],
                                z=z[0, :, head], tanh=features[0, :, head],
                                coefficients=coefficients[0, :, head])
            return coefficients

        with patch.object(RMSNorm, "__call__", normalized), \
                patch.object(input_dependent, "project_coefficients", projected):
            logits = model(batch[:, :-1], deterministic=True)
        if calls[0] != layers or norm_calls[0] != 1 or "kernel" not in captured:
            raise ValueError(f"Wrong capture counts: projection={calls[0]}, norm={norm_calls[0]}")
        captured["bias_raw"] = model.query_polynomial_bias[...][layer, head]
        return cross_entropy_loss(logits, batch[:, 1:]), captured

    return jax.jit(forward)


def reference_report(spec, *, layer, head, arrays, factorials, parameterization="direct"):
    """Audit checkpoint parameters strictly and historical sample moments descriptively."""
    path = Path(spec["reference_metrics_path"])
    if sha256(path) != spec["reference_metrics_sha256"]:
        raise ValueError(f"Reference measurements changed: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in records if row.get("step") == spec["step"]
            and "coefficients" in row.get("attention_diagnostics", {})]
    if len(rows) != 1:
        raise ValueError(f"Expected one coefficient reference at step {spec['step']}, found {len(rows)}")
    diagnostics = rows[0]["attention_diagnostics"]
    sample = diagnostics["sample"]
    if sample["record_sha256"] != spec["reference_sample_record_sha256"]:
        raise ValueError("Reference coefficient moments use a different sample")
    coefficients = diagnostics["coefficients"]
    parameter_errors = {}
    for field, actual in (("modulation_amplitude", arrays["amplitude"]),
                          ("projection_bias", arrays["bias_raw"]),
                          ("projection_kernel_l2", np.linalg.norm(arrays["kernel"].astype(np.float64), axis=0))):
        expected = np.asarray(coefficients[field], np.float64)[layer, head]
        error = float(np.max(np.abs(actual - expected)))
        parameter_errors[field] = error
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Restored {field} does not match checkpoint-step measurements: {error}")
    if parameterization == "sum_of_squares":
        return {"checkpoint_parameters_match": True, "parameter_max_abs_errors": parameter_errors,
                "sample_sha256": sample["record_sha256"], "reference_checkpoint_step": spec["step"],
                "all_moments_within_tolerance": None,
                "moment_comparison_note": "Not compared: historical moments are expanded polynomial coefficients; the probe captures the pre-expansion factor generators g = b + a*tanh(z)."}
    raw = arrays["coefficients"].astype(np.float64) * factorials
    scale = np.maximum(np.abs(arrays["bias_raw"]) + np.abs(arrays["amplitude"]), 1.0)
    moments = {}
    for name, reduce in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        observed = reduce(raw, axis=0)
        expected = np.asarray(coefficients["effective_summary"][name], np.float64)[layer, head] * factorials
        difference = observed - expected
        normalized_error = np.abs(difference) / scale
        moments[name] = {"observed_factorial_rescaled": observed.tolist(),
                         "reference_factorial_rescaled": expected.tolist(),
                         "difference_factorial_rescaled": difference.tolist(),
                         "absolute_error_over_generator_scale": normalized_error.tolist(),
                         "within_tolerance": bool(np.all(normalized_error <= REFERENCE_MOMENT_TOLERANCE))}
    return {"checkpoint_parameters_match": True, "parameter_max_abs_errors": parameter_errors,
            "sample_sha256": sample["record_sha256"], "reference_checkpoint_step": spec["step"],
            "moment_tolerance": REFERENCE_MOMENT_TOLERANCE,
            "moment_error_scale": "max(abs(raw_b_r) + abs(raw_a_r), 1), after multiplying coefficients by r!",
            "moment_comparison_is_descriptive": True,
            "moment_comparison_note": "Historical coefficient moments use the blocked diagnostic forward; this probe uses the native recipe backend.",
            "all_moments_within_tolerance": all(row["within_tolerance"] for row in moments.values()),
            "moments": moments}


def summarize_capture(arrays, *, epsilon, parameterization="direct"):
    x = arrays["x_after"].astype(np.float64)
    before = arrays["x_before"].astype(np.float64)
    gain = arrays["gain"].astype(np.float64)
    kernel = arrays["kernel"].astype(np.float64)
    z, features = arrays["z"].astype(np.float64), arrays["tanh"].astype(np.float64)
    coefficients = arrays["coefficients"].astype(np.float64)
    degree = kernel.shape[-1] - 1
    factorials = np.asarray([math.factorial(r) for r in range(degree + 1)], np.float64)
    if parameterization == "sum_of_squares":
        factorials = np.ones(degree + 1, np.float64)
    elif parameterization != "direct":
        raise ValueError(f"Unsupported coefficient parameterization: {parameterization}")
    if not np.array_equal(arrays["x_after"], arrays["x_projection"]):
        raise ValueError("Captured RMSNorm output differs from the actual coefficient input")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("A captured activation or parameter is nonfinite")
    reconstruction = (arrays["bias_raw"].astype(np.float64)
                      + arrays["amplitude"].astype(np.float64) * np.tanh(z)) / factorials
    scale = np.maximum(np.abs(arrays["bias_raw"]) + np.abs(arrays["amplitude"]), 1.0)
    coefficient_errors = np.max(np.abs(coefficients - reconstruction), axis=0) * factorials / scale
    if np.any(coefficient_errors > MANUAL_COEFFICIENT_TOLERANCE):
        raise ValueError(f"Manual coefficient reconstruction differs from native: {coefficient_errors}")
    # FP64 reconstructions audit the captured FP32 arithmetic; they do not alter it.
    z_reference = x @ kernel
    normalization_reference = before / np.sqrt(np.mean(before * before, axis=-1, keepdims=True) + epsilon) * gain
    norm = np.linalg.norm(x, axis=-1)
    before_norm = np.linalg.norm(before, axis=-1)
    w_norm = np.linalg.norm(kernel, axis=0)
    mean_x = x.mean(axis=0)
    mean_projection = mean_x @ kernel
    centered = (x - mean_x) @ kernel
    denominators = norm[:, None] * w_norm[None, :]
    alignment = np.divide(z, denominators, out=np.zeros_like(z), where=denominators > 0)
    mean_x_norm = np.linalg.norm(mean_x)
    rows = []
    for r in range(degree + 1):
        denominator = mean_x_norm * w_norm[r]
        rows.append({"degree": r, "w_norm": float(w_norm[r]),
                     "gain_weighted_w_norm": float(np.linalg.norm(gain * kernel[:, r])),
                     "raw_a": float(arrays["amplitude"][r]), "raw_b": float(arrays["bias_raw"][r]),
                     "z": summary(z[:, r]), "tanh": summary(features[:, r]),
                     "tanh_derivative": summary(1.0 - features[:, r] ** 2),
                     "alignment_cos": summary(alignment[:, r]),
                     "coefficient": summary(coefficients[:, r]),
                     "saturation_abs_z_gt2": float(np.mean(np.abs(z[:, r]) > 2.0)),
                     "saturation_abs_z_gt3": float(np.mean(np.abs(z[:, r]) > 3.0)),
                     "saturation_abs_tanh_ge099": float(np.mean(np.abs(features[:, r]) >= .99)),
                     "saturation_tanh_ge099": float(np.mean(features[:, r] >= .99)),
                     "saturation_tanh_le_neg099": float(np.mean(features[:, r] <= -.99)),
                     "mean_activation_projection": float(mean_projection[r]),
                     "centered_activation_projection": summary(centered[:, r]),
                     "mean_activation_alignment_cos": float(mean_projection[r] / denominator) if denominator else None,
                     "mean_projection_squared_share": float(mean_projection[r] ** 2 / np.mean(z[:, r] ** 2))
                     if np.any(z[:, r]) else None})
    return {"dimension": x.shape[-1], "prediction_tokens": x.shape[0], "degree": degree,
            "x_norm_before": summary(before_norm), "x_norm_after": summary(norm),
            "x_rms_before": summary(before_norm / math.sqrt(x.shape[-1])),
            "x_rms_after": summary(norm / math.sqrt(x.shape[-1])), "gain": summary(gain),
            "gain_rms": float(np.sqrt(np.mean(gain * gain))),
            "mean_activation_norm": float(mean_x_norm),
            "centered_activation_norm": summary(np.linalg.norm(x - mean_x, axis=-1)),
            "degrees": rows,
            "native_capture_validation": {
                "rmsnorm_output_equals_generator_input": True,
                "manual_coefficient_max_normalized_error": float(coefficient_errors.max()),
                "manual_coefficient_normalized_errors": coefficient_errors.tolist(),
                "manual_coefficient_tolerance": MANUAL_COEFFICIENT_TOLERANCE,
                "projection_fp64_reconstruction_max_abs_error": float(np.max(np.abs(z - z_reference))),
                "rmsnorm_fp64_reconstruction_max_abs_error": float(np.max(np.abs(x - normalization_reference))),
                "mean_centered_decomposition_max_abs_error": float(np.max(np.abs(z - mean_projection - centered)))}}


def require_supported(config, *, layer, head):
    model = config.model
    if not (0 <= layer < model.num_layers and 0 <= head < model.num_heads):
        raise ValueError("Selected layer/head is out of range")
    parameterization = model.attention_coefficient_parameterization
    if parameterization not in {"direct", "sum_of_squares"}:
        raise ValueError(f"Unsupported parameterization: {parameterization}")
    scale = "none" if parameterization == "sum_of_squares" else "factorial"
    transform = model.attention_score_transform
    input_power = model.attention_coefficient_input_power
    if (transform, input_power) not in {("identity", 0), ("softplus", 1)}:
        raise ValueError("Probe supports raw scores or softplus scores with a prefactor")
    if input_power and parameterization != "sum_of_squares":
        raise ValueError("The prefactor probe requires a sum-of-squares core")
    required = {"attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh",
                "attention_coefficient_scale": scale,
                "attention_coefficient_amplitude_scale": scale, "attention_coefficient_gate": "none",
                "attention_coefficient_readout": "diagonal", "attention_coefficient_input_normalization": "none",
                "attention_score_bias": "none", "remat": False, "rms_norm_fp32": True,
                "attention_backend": "mosaic"}
    for field, expected in required.items():
        if getattr(model, field) != expected:
            raise ValueError(f"Unsupported probe recipe: {field}={getattr(model, field)!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    for name in ("completion.json", "metrics.jsonl", "activations.npz"):
        if (run / name).exists():
            raise FileExistsError(f"Refusing to overwrite probe output {run / name}")
    diagnostic = yaml.safe_load((run / "config.yaml").read_text())["diagnostic"]
    validate_record(run, checkpoint_hashes=False)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use exactly one GPU")
    runtime = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "started_utc": utc(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "partition": os.environ.get("SLURM_JOB_PARTITION"), "node": os.uname().nodename,
               "gpu_type": jax.devices()[0].device_kind, "gpu_count": 1,
               "python_executable": sys.executable, "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
               "versions": {name: version(name) for name in ("jax", "flax", "orbax-checkpoint")},
               "environment": {name: os.environ.get(name) for name in
                               ("TRANSFORMER_CUDA_ENVIRONMENT", "CUDA_HOME", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
               "resolved_attention_backends": {}, "optimizer_updates": 0}
    attempts = json.loads((run / "record.json").read_text()).get("attempts", [])
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [*attempts, runtime]})
    write_json(run / "metadata.json", {"status": "running", "runtime": runtime})
    data = diagnostic["data"]
    if sha256(data["manifest"]) != data["manifest_sha256"]:
        raise ValueError("Pinned validation manifest changed")
    if data.get("start_sequence", 0) != 0 or data.get("sequences", 1) != 1:
        raise ValueError("This probe replays exactly the first held-out sequence")
    layer, head = diagnostic["layer"], diagnostic["head"]
    archive, outcomes, shared_sample_hash = {}, [], None
    for spec in diagnostic["models"]:
        print(f"Restoring {spec['role']} ({spec['run_id']}, checkpoint {spec['step']})", flush=True)
        for key in ("config", "reference_metrics", "data_identity"):
            if key + "_path" in spec and sha256(spec[key + "_path"]) != spec[key + "_sha256"]:
                raise ValueError(f"Pinned source {key} changed")
        verify_checkpoint(spec)
        config = load_model_config(spec)
        require_supported(config, layer=layer, head=head)
        runtime["resolved_attention_backends"][spec["role"]] = config.model.attention_backend
        if (Path(config.data.eval_manifest).resolve() != Path(data["manifest"]).resolve()
                or config.data.sequence_length != data["sequence_length"]):
            raise ValueError("Diagnostic data does not match the parent validation recipe")
        sample, sample_provenance = fixed_diagnostic_sample(config)
        sample_hash = hashlib.sha256(sample.astype("<i4").tobytes()).hexdigest()
        if sample_hash != spec["reference_sample_record_sha256"] or sample_hash != data["record_sha256"]:
            raise ValueError("Held-out sequence differs from the pinned diagnostic record")
        if shared_sample_hash is not None and sample_hash != shared_sample_hash:
            raise ValueError("Parents received different held-out sequences")
        shared_sample_hash = sample_hash
        graph, params = restore_parent(config, Path(spec["checkpoint_path"]))
        before_hash = state_sha256(params)
        forward = capture_forward(graph, layer=layer, head=head, layers=config.model.num_layers)
        loss, captured = forward(params, jnp.asarray(sample))
        loss, captured = jax.device_get((loss, captured))
        if not math.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite native sequence loss")
        after_hash = state_sha256(params)
        if after_hash != before_hash:
            raise ValueError("Forward probe changed parent parameters")
        arrays = {key: np.asarray(value) for key, value in captured.items()}
        parameterization = config.model.attention_coefficient_parameterization
        input_power = config.model.attention_coefficient_input_power
        core_degree = config.model.attention_degree - input_power
        results = summarize_capture(arrays, epsilon=config.model.rms_norm_eps, parameterization=parameterization)
        factorials = np.asarray([math.factorial(r) for r in range(core_degree + 1)], np.float64)
        reference = reference_report(spec, layer=layer, head=head, arrays=arrays, factorials=factorials,
                                     parameterization=parameterization)
        if parameterization == "sum_of_squares":
            half = core_degree // 2
            labels = [f"U{r}" for r in range(half + 1)] + [f"V{r}" for r in range(half)]
        else:
            labels = [f"r{r}" for r in range(core_degree + 1)]
        for entry, label in zip(results["degrees"], labels, strict=True):
            entry["coordinate_label"] = label
        row = {"format": "query_activation_probe", "version": 1, "role": spec["role"],
               "coefficient_parameterization": parameterization,
               "score_transform": config.model.attention_score_transform,
               "input_power": input_power, "total_degree": config.model.attention_degree,
               "coordinate_definition": "SoS factor generator" if parameterization == "sum_of_squares" else "polynomial order",
               "captured_coefficient_definition": "g_j = b_j + a_j*tanh(z_j), before SoS expansion" if parameterization == "sum_of_squares" else "c_r = (b_r + a_r*tanh(z_r))/r!",
               "source_run_id": spec["run_id"], "checkpoint_step": spec["step"],
               "parent_source_commit": spec["source_commit"], "checkpoint_sha256": spec.get("checkpoint_sha256"),
               "layer": layer, "head": head, "indexing": "zero_based", "sequences": 1,
               "attention_backend": config.model.attention_backend,
               "coefficient_projection_dtype": "float32", "coefficient_projection_precision": "HIGHEST",
               "sample": sample_provenance,
               "single_sequence_nll": float(loss), "single_sequence_perplexity": math.exp(float(loss)),
               "parent_parameter_sha256_before": before_hash, "parent_parameter_sha256_after": after_hash,
               "parent_parameters_unchanged": True, "optimizer_updates": 0,
               "x_definition": "exact pre-attention RMSNorm output including learned channel gains, shared across coefficient heads",
               "reference_validation": reference, **results}
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        archive.update({spec["role"] + "__" + key: value for key, value in arrays.items()})
        outcomes.append({"role": spec["role"], "parent_parameters_unchanged": True,
                         "checkpoint_parameters_match": reference["checkpoint_parameters_match"],
                         "all_reference_moments_within_tolerance": reference["all_moments_within_tolerance"]})
        print(json.dumps({"role": spec["role"], "x_norm_after_mean": results["x_norm_after"]["mean"],
                          "coefficient_capture_max_error": results["native_capture_validation"]["manual_coefficient_max_normalized_error"],
                          "reference_moments_within_tolerance": reference["all_moments_within_tolerance"]}), flush=True)
        del forward, params, graph, captured, arrays
        gc.collect()
        jax.clear_caches()
    archive["sample_tokens"] = sample
    np.savez_compressed(run / "activations.npz", **archive)
    outcome = {"status": "complete", "finished_utc": utc(), "optimizer_updates": 0,
               "sample_record_sha256": shared_sample_hash, "models": outcomes,
               "interpretation": diagnostic.get("interpretation", "One held-out sequence at one selected head; not a whole-model saturation estimate.")}
    write_json(run / "completion.json", outcome)
    write_json(run / "metadata.json", {"status": "complete", "runtime": runtime, "sample": sample_provenance})
    update_record(run, {**capture_artifacts(run), "status": "complete", "runtime": runtime,
                        "diagnostic_outcome": outcome})
    print("Query activation probe complete", flush=True)


if __name__ == "__main__":
    main()
