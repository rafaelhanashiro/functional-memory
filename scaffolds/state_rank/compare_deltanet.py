"""Matched pretrained/random Gated DeltaNet state spectra; no weights are saved."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from scaffolds.state_rank import compare_random as common
from scaffolds.state_rank import random_models, spectrum

from scaffolds.state_rank.artifacts import CONFIG_DIRECTORY, retained_path, resolved_config_path

DEFAULT_CONFIG = CONFIG_DIRECTORY / "random_vs_pretrained_deltanet.json"
METRICS = ("participation_ratio", "threshold_rank", "energy_rank90", "energy_rank99", "log_trace")
EXPERIMENT = "state_rank/random_vs_pretrained_deltanet"
INITIALIZATION_DETAILS = {
    "implementation": "pinned FLA GatedDeltaNetModel reference constructor/post_init",
    "linear_convolution_embedding": "Normal(0, config.initializer_range); biases zero",
    "normalization_weights": "ones",
    "A_log": "log(Uniform(0, 16)) independently per layer/head",
    "dt_bias": "inverse_softplus(log-uniform dt in [0.001, 0.1], clamped at 0.0001)",
    "residual_rescaling": "none (reference default prenorm_residual_strategy=None)",
    "parameter_generation": "seeded CPU FP32, full independent model for each seed",
}
# State evolution is FP32; tolerances accommodate possible kernel/hardware drift.
# FP64 SVD should agree much more closely, but cannot undo FP32 state differences.
# Discrete ranks may move by ONE only when an actual boundary crossing is
# confined to the explicit narrow band; all non-boundary changes fail.
BASELINE_TOLERANCES = {
    "normalized_spectrum_max_absolute_error": 2e-5,
    "participation_ratio_max_relative_error": 5e-4,
    "log_trace_max_absolute_error": 5e-4,
    "threshold_relative_singular_boundary_atol": 2e-6,
    "energy_cumulative_boundary_atol": 5e-5,
    "maximum_rank_difference": 1,
}


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported comparison configuration schema")
    if (config.get("model_dtype") != "float32" or config.get("analysis_dtype") != "float64"
            or config.get("attention_implementation") != "fused_recurrent" or config.get("allow_tf32") is not False):
        raise ValueError("Gated DeltaNet requires FP32 fused_recurrent inference, FP64 analysis, and TF32 off")
    for name in ("layers", "seeds"):
        values = config[name]
        if (not values or any(type(v) is not int or v < 0 for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{name} must contain distinct nonnegative integers")
    if config["layers"] != sorted(config["layers"]):
        raise ValueError("Comparison layers must be sorted")
    if config["baseline_layer"] not in config["layers"]:
        raise ValueError("The retained baseline layer must be measured")
    if type(config["tokens"]) is not int or config["tokens"] < 2:
        raise ValueError("tokens must be an integer >= 2")
    if type(config["input_count"]) is not int or config["input_count"] < 1:
        raise ValueError("input_count must be a positive integer")
    if type(config.get("svd_workers", 1)) is not int or config.get("svd_workers", 1) < 1:
        raise ValueError("svd_workers must be a positive integer")
    prefixes = config["spectrum_prefixes"]
    if (not prefixes or prefixes != sorted(set(prefixes))
            or any(type(t) is not int or not 1 <= t <= config["tokens"] for t in prefixes)):
        raise ValueError("Spectrum prefixes must be sorted distinct in-range integers")


def prepare_manifest(config_path, project_root, output=None):
    """Copy exact retained inputs and freeze source/configuration/data identity."""
    project_root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    baseline = retained_path(project_root, config["baseline_directory"])
    old_metadata = json.loads((baseline / "metadata.json").read_text())
    required = {"model": config["model_id"], "model_revision": config["model_revision"],
                "layer": config["baseline_layer"], "tokens": config["tokens"],
                "model_dtype": "float32", "analysis_dtype": "float64",
                "attention_implementation": "fused_recurrent"}
    for key, value in required.items():
        if old_metadata.get(key) != value:
            raise ValueError(f"Retained baseline {key} differs from this comparison")
    if old_metadata.get("threshold_rank_singular_tolerance") != spectrum.RANK_TOL:
        raise ValueError("Retained baseline singular-value rank threshold differs")
    output = Path(output).resolve() if output else project_root / config["output_directory"]
    provenance = {
        "config": config, "source_commit": common.source_commit(),
        "dependency_lock_sha256": common.sha256_file(project_root / "uv.lock"),
        "inputs_sha256": common.sha256_file(baseline / "inputs.json"),
        "baseline_metadata_sha256": common.sha256_file(baseline / "metadata.json"),
        "baseline_spectra_sha256": common.sha256_file(baseline / "spectra.npz"),
    }
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["identity"] != identity:
            raise ValueError("Existing comparison has a different source/configuration/input identity")
        return path
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs.json").write_bytes((baseline / "inputs.json").read_bytes())
    common.write_json(resolved_config_path(output, writing=True), config)
    common.write_json(path, {
        "schema_version": 1, "experiment": EXPERIMENT, "identity": identity, **provenance,
        "project_root": str(project_root), "execution_root": str(Path(__file__).resolve().parents[2]),
        "output_directory": str(output), "baseline_directory": str(baseline),
        "created_utc": common.utc_now(), "conditions": common.condition_specs(config), "jobs": {},
        "initialization": "Pinned FLA/Hugging Face reference initialization; not the original training initialization",
        "state_definition": "native recurrent state before output norm/gate/projection; convolution buffers excluded from spectra",
        "checkpoint_retention": "none", "baseline_tolerances": BASELINE_TOLERANCES,
        "initialization_details": INITIALIZATION_DETAILS,
    })
    return path


def validate_manifest(manifest):
    validate_config(manifest["config"])
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Manifest is not a Gated DeltaNet comparison")
    if manifest.get("baseline_tolerances") != BASELINE_TOLERANCES:
        raise ValueError("Baseline comparison tolerances differ from the frozen implementation")
    common.validate_manifest(manifest)


def validate_loading_info(loading, num_layers):
    """Allow only the historical unused D vectors and AutoModel's omitted readout."""
    allowed = {"lm_head.weight"} | {f"layers.{i}.attn.D" for i in range(num_layers)}
    unexpected = set(loading["unexpected_keys"]) - allowed
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"] or unexpected:
        raise RuntimeError(f"Pretrained Gated DeltaNet failed exact loading: {loading}")
    return sorted(loading["unexpected_keys"])


def _relative_singular_values(eigenvalues):
    return np.sqrt(np.divide(eigenvalues, eigenvalues[..., :1], out=np.zeros_like(eigenvalues),
                             where=eigenvalues[..., :1] > 0))


def compare_baseline(result, normalized_eigenvalues, reference, index):
    """Audit every prefix/head against retained layer 12, including rank boundaries."""
    old_spectrum = np.asarray(reference["normalized_eigenvalues"][index])
    current = np.asarray(normalized_eigenvalues)
    if current.shape != old_spectrum.shape:
        raise ValueError("Baseline spectrum shape differs from the measured state spectra")
    if not np.isfinite(current).all() or np.any(current < 0):
        raise ValueError("Measured normalized spectra must be finite and nonnegative")
    current_log = np.asarray(result["log_trace"])
    old_log = np.asarray(reference["log_trace"][index])
    both_zero = np.isneginf(current_log) & np.isneginf(old_log)
    log_difference = np.zeros_like(current_log, dtype=float)
    np.subtract(current_log, old_log, out=log_difference, where=~both_zero)
    log_error = float(np.max(np.abs(log_difference)))
    errors = {
        "normalized_spectrum_max_absolute_error": float(np.max(np.abs(current - old_spectrum))),
        "participation_ratio_max_relative_error": float(np.max(
            np.abs(result["participation_ratio"] - reference["participation_ratio"][index])
            / np.maximum(reference["participation_ratio"][index], 1))),
        "log_trace_max_absolute_error": log_error,
    }
    current_sigma, old_sigma = _relative_singular_values(current), _relative_singular_values(old_spectrum)
    current_cdf, old_cdf = current.cumsum(axis=-1), old_spectrum.cumsum(axis=-1)
    boundary_failure = False
    for metric, target in (("threshold_rank", spectrum.RANK_TOL), ("energy_rank90", .9), ("energy_rank99", .99)):
        actual, old = np.asarray(result[metric]), np.asarray(reference[metric][index])
        if actual.shape != current.shape[:-1] or old.shape != actual.shape:
            raise ValueError(f"Baseline {metric} shape differs")
        changed = actual != old
        difference = np.abs(actual - old)
        if metric == "threshold_rank":
            crossed = (current_sigma > target) != (old_sigma > target)
            near = ((np.abs(current_sigma - target) <= BASELINE_TOLERANCES["threshold_relative_singular_boundary_atol"])
                    & (np.abs(old_sigma - target) <= BASELINE_TOLERANCES["threshold_relative_singular_boundary_atol"]))
            allowed = np.any(crossed, axis=-1) & np.all(~crossed | near, axis=-1)
        else:
            boundary = np.clip(np.minimum(actual, old).astype(int) - 1, 0, current.shape[-1] - 1)[..., None]
            current_boundary = np.take_along_axis(current_cdf, boundary, axis=-1)[..., 0]
            old_boundary = np.take_along_axis(old_cdf, boundary, axis=-1)[..., 0]
            allowed = ((current_boundary >= target) != (old_boundary >= target))
            allowed &= np.abs(current_boundary - target) <= BASELINE_TOLERANCES["energy_cumulative_boundary_atol"]
            allowed &= np.abs(old_boundary - target) <= BASELINE_TOLERANCES["energy_cumulative_boundary_atol"]
        allowed &= difference <= BASELINE_TOLERANCES["maximum_rank_difference"]
        failures = changed & ~allowed
        errors[f"{metric}_mismatches"] = int(changed.sum())
        errors[f"{metric}_boundary_tolerated_mismatches"] = int((changed & allowed).sum())
        errors[f"{metric}_nonboundary_mismatches"] = int(failures.sum())
        errors[f"{metric}_maximum_difference"] = float(difference.max())
        boundary_failure |= bool(failures.any())
    errors["passed"] = bool(not boundary_failure and all(
        errors[key] <= BASELINE_TOLERANCES[key] for key in (
            "normalized_spectrum_max_absolute_error", "participation_ratio_max_relative_error",
            "log_trace_max_absolute_error")))
    return errors


def _measure_input(*args, **kwargs):
    from scaffolds.state_rank.deltanet_states import compute_gdn_spectra_layers
    return compute_gdn_spectra_layers(*args, **kwargs)


def analyze_model(model, inputs, layers, spectrum_prefixes, *, baseline=None, baseline_layer=12, svd_workers=1):
    """Combine per-input all-layer measurements while retaining only selected spectra."""
    all_inputs, checks, per_input_validation = [], [], []
    coordinates = None
    for index, record in enumerate(inputs):
        print(f"input={index + 1}/{len(inputs)} extracting_all_layers tokens={len(record['token_ids'])}", flush=True)
        arrays, validation = _measure_input(
            model, torch.tensor([record["token_ids"]]), layers, spectrum_prefixes,
            full_spectrum_layers=(baseline_layer,) if baseline is not None else (), svd_workers=svd_workers,
        )
        full_spectra = validation.pop("full_spectra", {})
        if baseline is not None:
            if baseline_layer not in full_spectra:
                raise RuntimeError("Measurement helper did not retain the baseline layer's full spectra")
            layer_index = list(arrays["layers"]).index(baseline_layer)
            result = {name: arrays[name][layer_index] for name in METRICS}
            check = compare_baseline(result, full_spectra[baseline_layer], baseline, index)
            checks.append({"input_index": index, **check})
            if not check["passed"]:
                raise ValueError(f"Pretrained baseline reproduction failed: {check}")
        del full_spectra
        current_coordinates = {key: arrays[key] for key in ("layers", "head_indices", "prefixes", "spectrum_prefixes")}
        if coordinates is None:
            coordinates = current_coordinates
        elif any(not np.array_equal(value, coordinates[key]) for key, value in current_coordinates.items()):
            raise ValueError("State measurement axes changed between inputs")
        all_inputs.append({key: value for key, value in arrays.items() if key not in current_coordinates})
        per_input_validation.append({"input_index": index, **validation})
        for layer_index, layer in enumerate(arrays["layers"]):
            print(f"input={index + 1} layer={layer} final_pr={arrays['participation_ratio'][layer_index,-1].mean():.6f} "
                  f"final_rank={arrays['threshold_rank'][layer_index,-1].mean():.3f}", flush=True)
    if not all_inputs:
        raise ValueError("At least one saved input is required")
    combined = {key: np.stack([entry[key] for entry in all_inputs], axis=1) for key in all_inputs[0]}
    combined.update(coordinates)
    combined["token_ids"] = np.asarray([record["token_ids"] for record in inputs], dtype=np.int64)
    return combined, {
        "baseline_checks": checks, "baseline_tolerances": BASELINE_TOLERANCES,
        "maximum_prefix_relative_error": max(v["maximum_prefix_relative_error"] for v in per_input_validation),
        "input_checks": per_input_validation,
    }


def run_condition(manifest_path, condition_name, *, require_cuda=False):
    manifest = json.loads(Path(manifest_path).read_text())
    validate_manifest(manifest)
    condition = next((c for c in manifest["conditions"] if c["name"] == condition_name), None)
    if condition is None:
        raise ValueError(f"Unknown condition: {condition_name}")
    output = Path(manifest["output_directory"]) / condition_name
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["identity"] != manifest["identity"]:
            raise ValueError("Condition belongs to a different comparison")
        if previous["status"] == "complete":
            if common.sha256_file(output / "metrics.npz") != previous["metrics_sha256"]:
                raise ValueError("Completed metrics were changed")
            print(f"already_complete={condition_name}", flush=True)
            return previous
    # Native fused recurrence is CUDA-only, including ordinary local invocation.
    if not torch.cuda.is_available():
        raise RuntimeError("Native Gated DeltaNet comparison requires a visible CUDA GPU")
    import fla.models.gated_deltanet  # register the pinned FLA architecture
    from transformers import AutoConfig, AutoModel

    device = torch.device("cuda")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = manifest["config"]
    output.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    metadata = {
        "schema_version": 1, "identity": manifest["identity"], **condition,
        "condition": condition_name, "status": "running", "started_utc": common.utc_now(),
        "source_commit": manifest["source_commit"], "model": config["model_id"],
        "model_revision": config["model_revision"], "model_dtype": "float32", "analysis_dtype": "float64",
        "attention_implementation": "fused_recurrent", "allow_tf32": False,
        "initialization": manifest["initialization"] if condition["weights"] == "random" else "pinned pretrained checkpoint",
        "layers": config["layers"], "head_type": "recurrent", "state_layout": "value,key",
        "threshold_rank_singular_tolerance": spectrum.RANK_TOL,
        "participation_ratio_convention": "(sum(sigma^2))^2 / sum(sigma^4)",
        "baseline_tolerances": BASELINE_TOLERANCES, "svd_workers": config.get("svd_workers", 1),
        "slurm_job_id": job_id, "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "hostname": socket.gethostname(), "device": str(device), "gpu": torch.cuda.get_device_name(),
        "inputs_sha256": manifest["inputs_sha256"], "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "fla_revision": spectrum.FLA_REVISION,
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "numpy", "matplotlib", "flash-linear-attention", "triton")},
        "log_path": str(Path(manifest["project_root"]) / "results/logs" / EXPERIMENT / condition_name / f"slurm-{job_id}.out") if job_id else None,
        "checkpoint_retention": "none",
        "initialization_details": INITIALIZATION_DETAILS if condition["weights"] == "random" else None,
    }
    common.write_json(metadata_path, metadata)
    started = time.monotonic()
    try:
        model_config = AutoConfig.from_pretrained(config["model_id"], revision=config["model_revision"])
        if model_config.model_type != "gated_deltanet":
            raise ValueError("Pinned model must be a Gated DeltaNet")
        model_config.attn_mode = "fused_recurrent"
        if any(layer >= model_config.num_hidden_layers for layer in config["layers"]):
            raise ValueError("Requested layer exceeds pinned model depth")
        if config["tokens"] > model_config.max_position_embeddings:
            raise ValueError("Requested tokens exceed the pinned model context")
        inputs = random_models.load_saved_inputs(
            Path(manifest["output_directory"]) / "inputs.json", config["tokens"], model_config.vocab_size,
            model_config.bos_token_id, expected_count=config["input_count"],
        )
        baseline_path = Path(manifest["baseline_directory"]) / "spectra.npz"
        with np.load(baseline_path, allow_pickle=False) as old:
            if not np.array_equal(old["token_ids"], np.asarray([item["token_ids"] for item in inputs])):
                raise ValueError("Saved inputs differ from retained baseline token IDs")
            if not np.array_equal(old["head_indices"], np.arange(model_config.num_v_heads or model_config.num_heads)):
                raise ValueError("Retained baseline does not cover all recurrent heads")
        if condition["weights"] == "random":
            model = random_models.build_random_gated_deltanet(model_config, condition["seed"])
            metadata["unused_checkpoint_keys"] = []
        else:
            model, loading = AutoModel.from_pretrained(
                config["model_id"], revision=config["model_revision"], config=model_config,
                dtype=torch.float32, use_safetensors=True, output_loading_info=True,
            )
            metadata["unused_checkpoint_keys"] = validate_loading_info(loading, model_config.num_hidden_layers)
            model.eval()
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("All model parameters must be FP32")
        metadata["parameter_sha256"] = random_models.parameter_fingerprint(model)
        metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        metadata["model_config"] = model.config.to_dict()
        common.write_json(metadata_path, metadata)
        model.to(device).eval()
        if condition["weights"] == "pretrained":
            with np.load(baseline_path, allow_pickle=False) as baseline:
                arrays, validation = analyze_model(
                    model, inputs, config["layers"], config["spectrum_prefixes"], baseline=baseline,
                    baseline_layer=config["baseline_layer"], svd_workers=config.get("svd_workers", 1),
                )
        else:
            arrays, validation = analyze_model(
                model, inputs, config["layers"], config["spectrum_prefixes"], svd_workers=config.get("svd_workers", 1),
            )
        temporary = output / "metrics.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / "metrics.npz")
        metadata.update(status="complete", completed_utc=common.utc_now(), elapsed_seconds=time.monotonic() - started,
                        metrics_sha256=common.sha256_file(output / "metrics.npz"), validation=validation,
                        metrics_shape=list(arrays["participation_ratio"].shape),
                        spectrum_shape=list(arrays["normalized_eigenvalues"].shape))
        common.write_json(metadata_path, metadata)
        print(f"completed={condition_name} elapsed_seconds={metadata['elapsed_seconds']:.2f}", flush=True)
        return metadata
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", updated_utc=common.utc_now())
        common.write_json(metadata_path, metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--condition")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(prepare_manifest(args.config, args.project_root, args.output))
    elif args.manifest and args.condition:
        run_condition(args.manifest, args.condition, require_cuda=args.require_cuda)
    else:
        parser.error("Use --prepare or provide --manifest and --condition")


if __name__ == "__main__":
    main()
