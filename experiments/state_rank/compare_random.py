"""Matched random/pretrained Llama state-rank conditions; no model weights are saved."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from experiments.state_rank import spectrum

import numpy as np
import torch

from experiments.state_rank.random_models import build_random_llama, load_saved_inputs, parameter_fingerprint

DEFAULT_CONFIG = Path(__file__).parent / "configs/random_vs_pretrained.json"
METRICS = ("participation_ratio", "threshold_rank", "energy_rank90", "energy_rank99",
           "log_trace", "key_norms", "value_norms")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True,
    ).strip()


def condition_specs(config):
    return [{"name": "pretrained", "weights": "pretrained", "seed": None}] + [
        {"name": f"random-seed{seed}", "weights": "random", "seed": seed}
        for seed in config["seeds"]
    ]


def prepare_manifest(config_path, project_root, output=None):
    """Resolve inputs and provenance before scheduling; refuse incompatible reuse."""
    project_root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported comparison configuration schema")
    if (config["model_dtype"] != "float32" or config["analysis_dtype"] != "float64"
            or config["attention_implementation"] != "eager" or config["allow_tf32"]):
        raise ValueError("The matched comparison requires FP32 eager inference, FP64 analysis, and TF32 off")
    for name in ("layers", "seeds"):
        values = config[name]
        if (not values or any(type(v) is not int or v < 0 for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{name} must contain distinct nonnegative integers")
    if config["layers"] != sorted(config["layers"]):
        raise ValueError("Comparison layers must be sorted for depth plots")
    if config["baseline_layer"] not in config["layers"]:
        raise ValueError("The retained baseline layer must be measured")
    prefixes = config["spectrum_prefixes"]
    if not prefixes or prefixes != sorted(set(prefixes)) or any(type(t) is not int or not 1 <= t <= config["tokens"] for t in prefixes):
        raise ValueError("Spectrum prefixes must be sorted distinct in-range integers")
    baseline = project_root / config["baseline_directory"]
    old_metadata = json.loads((baseline / "metadata.json").read_text())
    if (old_metadata["model"] != config["model_id"]
            or old_metadata["model_revision"] != config["model_revision"]
            or old_metadata["layer"] != config["baseline_layer"]
            or old_metadata["tokens"] != config["tokens"]):
        raise ValueError("Pinned model, layer, or tokens differ from the retained baseline")
    output = Path(output).resolve() if output else project_root / config["output_directory"]
    provenance = {
        "config": config,
        "source_commit": source_commit(),
        "dependency_lock_sha256": sha256_file(project_root / "uv.lock"),
        "inputs_sha256": sha256_file(baseline / "inputs.json"),
        "baseline_metadata_sha256": sha256_file(baseline / "metadata.json"),
        "baseline_spectra_sha256": sha256_file(baseline / "spectra.npz"),
    }
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["identity"] != identity:
            raise ValueError("Existing comparison has a different source/configuration/input identity")
        return path
    output.mkdir(parents=True, exist_ok=True)
    # Copy the exact saved inputs once; no tokenization or dataset access occurs.
    (output / "inputs.json").write_bytes((baseline / "inputs.json").read_bytes())
    write_json(output / "config.json", config)
    write_json(path, {
        "schema_version": 1, "experiment": "state_rank/random_vs_pretrained",
        "identity": identity, **provenance,
        "project_root": str(project_root), "execution_root": str(Path(__file__).resolve().parents[2]),
        "output_directory": str(output), "baseline_directory": str(baseline),
        "created_utc": utc_now(), "conditions": condition_specs(config), "jobs": {},
        "initialization": "Hugging Face reference initialization; not Meta's original random weights",
        "state_definition": "unnormalized softmax numerator; post-RoPE keys; before output processing",
        "checkpoint_retention": "none",
    })
    return path


def validate_manifest(manifest):
    protected = {key: manifest[key] for key in (
        "config", "source_commit", "dependency_lock_sha256", "inputs_sha256",
        "baseline_metadata_sha256", "baseline_spectra_sha256",
    )}
    identity = hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest()
    if manifest["identity"] != identity or manifest["conditions"] != condition_specs(manifest["config"]):
        raise ValueError("Comparison manifest identity or condition list was changed")
    if manifest["source_commit"] != source_commit():
        raise ValueError("Execution source commit differs from the frozen manifest")
    project_root = Path(manifest["project_root"])
    baseline = Path(manifest["baseline_directory"])
    for path, field in (
        (project_root / "uv.lock", "dependency_lock_sha256"),
        (Path(manifest["output_directory"]) / "inputs.json", "inputs_sha256"),
        (baseline / "metadata.json", "baseline_metadata_sha256"),
        (baseline / "spectra.npz", "baseline_spectra_sha256"),
    ):
        if sha256_file(path) != manifest[field]:
            raise ValueError(f"Comparison provenance changed: {path}")


def compare_baseline(keys, values, result, reference, index):
    """Fail the first GPU condition if the historical layer-8 result drifts."""
    errors = {}
    for name, current in (("keys", keys), ("values", values)):
        old = reference[name][index]
        errors[f"{name}_relative_error"] = float(np.linalg.norm(current - old) / max(np.linalg.norm(old), 1e-300))
    old_spectrum = reference["normalized_eigenvalues"][index]
    errors["normalized_spectrum_max_absolute_error"] = float(np.max(np.abs(result["normalized_eigenvalues"] - old_spectrum)))
    old_pr = reference["participation_ratio"][index]
    errors["participation_ratio_max_relative_error"] = float(np.max(np.abs(result["participation_ratio"] - old_pr) / np.maximum(old_pr, 1)))
    for name in ("threshold_rank", "energy_rank90", "energy_rank99"):
        errors[f"{name}_mismatches"] = int(np.count_nonzero(result[name] != reference[name][index]))
    errors["passed"] = (
        max(errors["keys_relative_error"], errors["values_relative_error"]) <= spectrum.CACHE_RTOL
        and errors["normalized_spectrum_max_absolute_error"] <= 2e-5
        and errors["participation_ratio_max_relative_error"] <= 5e-4
        and all(errors[f"{name}_mismatches"] == 0 for name in ("threshold_rank", "energy_rank90", "energy_rank99"))
    )
    return errors


def analyze_model(model, inputs, layers, spectrum_prefixes, *, baseline=None, baseline_layer=8):
    """Keep scalar curves and selected spectra; discard raw K/V after each input."""
    metrics_by_input = []
    baseline_checks = []
    maximum_direct_error = 0.0
    selected = np.asarray(spectrum_prefixes) - 1
    for index, record in enumerate(inputs):
        print(f"input={index + 1}/{len(inputs)} extracting_all_layers tokens={len(record['token_ids'])}", flush=True)
        caches = spectrum.extract_kv_layers(model, torch.tensor([record["token_ids"]]), layers)
        per_layer = []
        for layer in layers:
            keys, values = caches.pop(layer)
            result = spectrum.compute_llama_spectra(
                keys, values, check_prefixes=spectrum_prefixes, scale=model.layers[layer].self_attn.scaling,
            )
            result["value_norms"] = np.linalg.norm(values, axis=-1)
            maximum_direct_error = max(maximum_direct_error, float(np.nanmax(result["direct_check_relative_error"])))
            if baseline is not None and layer == baseline_layer:
                check = compare_baseline(keys, values, result, baseline, index)
                baseline_checks.append({"input_index": index, **check})
                if not check["passed"]:
                    raise ValueError(f"Pretrained baseline reproduction failed: {check}")
            per_layer.append({name: result[name] for name in METRICS} | {
                "normalized_eigenvalues": result["normalized_eigenvalues"][selected],
            })
            print(f"input={index + 1} layer={layer} final_pr={result['participation_ratio'][-1].mean():.6f} "
                  f"final_rank={result['threshold_rank'][-1].mean():.3f}", flush=True)
        metrics_by_input.append({name: np.stack([r[name] for r in per_layer]) for name in per_layer[0]})
    arrays = {name: np.stack([r[name] for r in metrics_by_input], axis=1) for name in metrics_by_input[0]}
    arrays.update(
        layers=np.asarray(layers), head_indices=np.arange(model.config.num_key_value_heads),
        prefixes=np.arange(1, len(inputs[0]["token_ids"]) + 1), spectrum_prefixes=np.asarray(spectrum_prefixes),
        token_ids=np.asarray([item["token_ids"] for item in inputs], dtype=np.int64),
    )
    return arrays, {"maximum_direct_relative_error": maximum_direct_error, "baseline_checks": baseline_checks}


def run_condition(manifest_path, condition_name, *, require_cuda=False):
    from transformers import AutoConfig, AutoModel
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
            if sha256_file(output / "metrics.npz") != previous["metrics_sha256"]:
                raise ValueError("Completed metrics were changed")
            print(f"already_complete={condition_name}", flush=True)
            return previous
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Slurm comparison requires a visible CUDA GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = manifest["config"]
    output.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    metadata = {
        "schema_version": 1, "identity": manifest["identity"], **condition,
        "condition": condition_name, "status": "running", "started_utc": utc_now(),
        "source_commit": manifest["source_commit"], "model": config["model_id"],
        "model_revision": config["model_revision"], "model_dtype": "float32", "analysis_dtype": "float64",
        "attention_implementation": "eager", "allow_tf32": False, "initialization": manifest["initialization"] if condition["weights"] == "random" else "pinned pretrained checkpoint",
        "layers": config["layers"], "threshold_rank_singular_tolerance": spectrum.RANK_TOL,
        "participation_ratio_convention": "(sum(sigma^2))^2 / sum(sigma^4)",
        "slurm_job_id": job_id, "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "hostname": socket.gethostname(), "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "inputs_sha256": manifest["inputs_sha256"], "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "matplotlib")},
        "log_path": str(Path(manifest["project_root"]) / "results/logs/state_rank/random_vs_pretrained" / condition_name / f"slurm-{job_id}.out") if job_id else None,
    }
    write_json(metadata_path, metadata)
    started = time.monotonic()
    try:
        model_config = AutoConfig.from_pretrained(config["model_id"], revision=config["model_revision"])
        if any(layer >= model_config.num_hidden_layers for layer in config["layers"]):
            raise ValueError("Requested layer exceeds pinned model depth")
        inputs = load_saved_inputs(
            Path(manifest["output_directory"]) / "inputs.json", config["tokens"], model_config.vocab_size,
            model_config.bos_token_id, expected_count=config["input_count"],
        )
        baseline_path = Path(manifest["baseline_directory"]) / "spectra.npz"
        with np.load(baseline_path) as old:
            if not np.array_equal(old["token_ids"], np.asarray([item["token_ids"] for item in inputs])):
                raise ValueError("Saved inputs differ from retained baseline token IDs")
        if condition["weights"] == "random":
            model = build_random_llama(model_config, condition["seed"])
        else:
            model, loading = AutoModel.from_pretrained(
                config["model_id"], revision=config["model_revision"], config=model_config,
                dtype=torch.float32, attn_implementation="eager", use_safetensors=True, output_loading_info=True,
            )
            if (loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]
                    or set(loading["unexpected_keys"]) - {"lm_head.weight"}):
                raise RuntimeError(f"Pretrained model failed exact loading: {loading}")
            model.eval()
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("All model parameters must be FP32")
        metadata["parameter_sha256"] = parameter_fingerprint(model)
        metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        metadata["model_config"] = model.config.to_dict()
        write_json(metadata_path, metadata)
        model.to(device).eval()
        if condition["weights"] == "pretrained":
            with np.load(baseline_path) as baseline:
                arrays, validation = analyze_model(model, inputs, config["layers"], config["spectrum_prefixes"],
                                                   baseline=baseline, baseline_layer=config["baseline_layer"])
        else:
            arrays, validation = analyze_model(model, inputs, config["layers"], config["spectrum_prefixes"])
        temporary = output / "metrics.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / "metrics.npz")
        metadata.update(status="complete", completed_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                        metrics_sha256=sha256_file(output / "metrics.npz"), validation=validation,
                        metrics_shape=list(arrays["participation_ratio"].shape),
                        spectrum_shape=list(arrays["normalized_eigenvalues"].shape))
        write_json(metadata_path, metadata)
        print(f"completed={condition_name} elapsed_seconds={metadata['elapsed_seconds']:.2f}", flush=True)
        return metadata
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", updated_utc=utc_now())
        write_json(metadata_path, metadata)
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
