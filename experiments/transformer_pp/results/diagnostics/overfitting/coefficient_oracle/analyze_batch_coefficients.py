"""Inspect fitted batch coefficient geometry on CPU, without loading the backbone.

Read evenly spaced queries from every sequence's retained free-coefficient table.
Measurements stay beside the run; the sequence split tests transfer of a PCA basis,
not language-model generalization or the ability to predict coefficients from inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

# NumPy and TensorStore are the only array runtimes used; no accelerator is opened.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(min(4, max(1, int(os.environ.get(_name, "4")))))

import numpy as np
import tensorstore as ts
import yaml

from scaffolds.records import capture_artifacts, load_run, update_record, validate_record


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
OUTPUT = "batch_coefficient_geometry.json"
QUANTILES = [0.0, 0.1, 0.5, 0.9, 1.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return parent
    raise ValueError(f"Run is outside a repository: {path}")


def checked_checkpoint(run: Path, record: dict, recipe: dict) -> tuple[Path, dict]:
    # A post-training hook can run while training.log is still open. Validate the
    # outcome and identity, then check scientific inputs and checkpoint metadata;
    # do not compare the live log with an earlier artifact hash.
    validate_record(run, verify_hashes=False, checkpoint_hashes=False)
    if sha256(run / "config.yaml") != record["config"]["sha256"]:
        raise ValueError("The frozen recipe checksum changed")
    completion = json.loads((run / "completion.json").read_text())
    if completion.get("status") != "complete":
        raise ValueError("Completion does not describe a completed fit")
    selected = record.get("selected_checkpoint")
    if not selected or selected.get("kind") != "directory":
        raise ValueError("A retained coefficient checkpoint is required")
    checkpoint = Path(selected["path"])
    if not checkpoint.is_absolute():
        checkpoint = repository(run) / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    actual = {p.relative_to(checkpoint).as_posix(): p.stat().st_size
              for p in checkpoint.rglob("*") if p.is_file()}
    expected = {item["path"]: item["size_bytes"] for item in selected["files"]}
    if actual != expected:
        raise ValueError("Retained checkpoint file names or sizes differ from its record")
    identity_path = checkpoint / "coefficient_identity.json"
    identity = json.loads(identity_path.read_text())
    identity_entry = next(item for item in selected["files"]
                          if item["path"] == "coefficient_identity.json")
    if sha256(identity_path) != identity_entry["sha256"]:
        raise ValueError("Coefficient checkpoint identity checksum changed")
    if (identity.get("coordinates") != "effective_monomial_c"
            or identity.get("layout") != "microbatches of [example,layer,query,head,order]"
            or identity.get("step") != completion["best_step"]
            or identity.get("parent_run_id") != recipe["oracle"]["parent_run_id"]):
        raise ValueError("The retained checkpoint is not the selected independent batch fit")
    return checkpoint, completion


def sample_coefficients(checkpoint: Path, recipe: dict, positions: np.ndarray) -> np.ndarray:
    """Read one saved microbatch at a time, retaining only sampled queries."""
    directory = checkpoint / "best"
    metadata = json.loads((directory / "_METADATA").read_text())
    if not metadata.get("use_ocdbt") or metadata.get("use_zarr3"):
        raise ValueError("Expected the scaffold's OCDBT/Zarr-v2 coefficient checkpoint")
    arrays = []
    for entry in metadata["tree_metadata"].values():
        keys = entry["key_metadata"]
        if len(keys) != 2 or keys[0]["key"] != "coefficients":
            raise ValueError("Checkpoint contains an unexpected coefficient tree")
        arrays.append((int(keys[1]["key"]), entry["value_metadata"]["write_shape"]))
    arrays.sort()
    if [index for index, _ in arrays] != list(range(len(arrays))):
        raise ValueError("Saved coefficient microbatches are not contiguous")
    model, oracle = recipe["model"], recipe["oracle"]
    tail = [model["num_layers"], recipe["data"]["sequence_length"] - 1,
            model["num_heads"], model["attention_degree"] + 1]
    if any(shape[1:] != tail or shape[0] <= 0 for _, shape in arrays):
        raise ValueError("Coefficient shape disagrees with the frozen model recipe")
    count = int(oracle["sequence_count"])
    if sum(shape[0] for _, shape in arrays) != count:
        raise ValueError("Checkpoint does not contain every fitting sequence")
    sampled = np.empty((count, tail[0], len(positions), tail[2], tail[3]), np.float64)
    offset = 0
    for index, shape in arrays:
        array = ts.open({"driver": "zarr", "kvstore": {
            "driver": "ocdbt", "base": {"driver": "file", "path": str(directory) + "/"},
            "path": f"coefficients.{index}",
        }}, open=True, read=True).result()
        if list(array.shape) != shape:
            raise ValueError("Array storage shape differs from checkpoint metadata")
        sampled[offset:offset + shape[0]] = array[:, :, positions, :, :].read().result()
        offset += shape[0]
    if not np.isfinite(sampled).all():
        raise ValueError("Fitted coefficients are not finite")
    return sampled


def pca(values: np.ndarray) -> dict:
    mean = values.mean(axis=0)
    centered = values - mean
    eigenvalues, basis = np.linalg.eigh(centered.T @ centered / len(values))
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    basis = basis[:, ::-1]
    # Fix arbitrary eigenvector signs for stable serialized basis vectors.
    pivots = np.argmax(np.abs(basis), axis=0)
    basis *= np.where(basis[pivots, np.arange(basis.shape[1])] < 0, -1, 1)
    total = float(eigenvalues.sum())
    cumulative = np.cumsum(eigenvalues) / total if total > 1e-24 else np.ones(len(mean))
    return {"mean": mean, "basis": basis, "eigenvalues": eigenvalues,
            "total_variance": total, "cumulative": cumulative,
            "components_90": int(np.searchsorted(cumulative, .9) + 1) if total > 1e-24 else 0,
            "components_95": int(np.searchsorted(cumulative, .95) + 1) if total > 1e-24 else 0}


def analyze(sampled: np.ndarray, train_sequences: int) -> dict:
    """Return coefficient-space diagnostics; no model forward is evaluated."""
    negative = np.mean(sampled < 0, axis=(0, 1, 2, 3)).tolist()
    sampled *= np.asarray([math.factorial(r) for r in range(sampled.shape[-1])])
    norms = np.linalg.norm(sampled, axis=-1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("A sampled coefficient vector has undefined direction")
    sampled /= norms
    per_head = []
    for layer in range(sampled.shape[1]):
        for head in range(sampled.shape[3]):
            values = sampled[:, layer, :, head, :]
            all_fit = pca(values.reshape(-1, values.shape[-1]))
            train = pca(values[:train_sequences].reshape(-1, values.shape[-1]))
            test = values[train_sequences:].reshape(-1, values.shape[-1]) - train["mean"]
            test_energy = np.mean(np.square(test @ train["basis"]), axis=0)
            baseline_error = float(np.mean(np.sum(np.square(test), axis=-1)))
            transferred = (np.cumsum(test_energy) / baseline_error if baseline_error > 1e-24
                           else np.ones(values.shape[-1]))
            total = all_fit["total_variance"]
            def explained_by_mean(axis):
                numerator = np.mean(np.sum(np.square(values.mean(axis=axis) - all_fit["mean"]), axis=-1))
                return float(numerator / total) if total > 1e-24 else 0.0
            per_head.append({
                "layer": layer + 1, "head": head + 1,
                "mean_unit_coefficients": all_fit["mean"].tolist(),
                "total_direction_variance": total,
                "pca_eigenvalues": all_fit["eigenvalues"].tolist(),
                "pca_cumulative_variance_fraction": all_fit["cumulative"].tolist(),
                "components_90_percent": all_fit["components_90"],
                "components_95_percent": all_fit["components_95"],
                "train_mean_unit_coefficients": train["mean"].tolist(),
                "train_pca_components": train["basis"].T.tolist(),
                "train_pca_cumulative_variance_fraction": train["cumulative"].tolist(),
                "test_cumulative_error_reduction_fraction": transferred.tolist(),
                "variance_fraction_explained_by_sequence_mean": explained_by_mean(1),
                "variance_fraction_explained_by_query_position_mean": explained_by_mean(0),
            })
    aggregate_keys = (
        "pca_cumulative_variance_fraction", "components_90_percent", "components_95_percent",
        "test_cumulative_error_reduction_fraction", "variance_fraction_explained_by_sequence_mean",
        "variance_fraction_explained_by_query_position_mean",
    )
    return {"per_head": per_head, "summary": {
        "head_count": len(per_head), "quantile_probabilities": QUANTILES,
        "negative_coefficient_fraction_per_order": negative,
        **{key + "_quantiles": np.quantile([item[key] for item in per_head], QUANTILES, axis=0).tolist()
           for key in aggregate_keys},
    }}


def atomic_write(path: Path, data: dict, overwrite: bool) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # An atomic link refuses a destination created by another analysis process.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--train-sequences", type=int, help="default: first three quarters of the sequences")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--if-applicable", action="store_true", help="skip unrelated/incomplete fits and matching existing output")
    parser.add_argument("--dry-run", action="store_true", help="compute and print a summary without writing files or records")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    record = load_run(run)
    oracle = recipe.get("oracle", {})
    if (record.get("status") != "complete"
            or oracle.get("parameterization") != "free_per_example_query_coefficients"
            or int(oracle.get("sequence_count", 0)) <= 1):
        if args.if_applicable:
            print("Batch coefficient geometry: not applicable; skipped.")
            return
        raise ValueError("Requires a complete independent free-coefficient batch fit")
    checkpoint, completion = checked_checkpoint(run, record, recipe)
    count = int(oracle["sequence_count"])
    train_sequences = args.train_sequences if args.train_sequences is not None else 3 * count // 4
    length = int(recipe["data"]["sequence_length"]) - 1
    if not 0 < args.queries <= length or not 0 < train_sequences < count:
        raise ValueError("Invalid query sample count or sequence split")
    positions = ((np.arange(args.queries) + .5) * length / args.queries).astype(int)
    settings = {
        "coordinates": "r! times effective monomial coefficients, divided by their positive L2 norm; sign preserved",
        "centering": "separately within each layer/head; arithmetic mean over sampled queries and sequences",
        "sequence_count": count, "query_positions_zero_based": positions.tolist(),
        "train_sequence_indices": list(range(train_sequences)),
        "test_sequence_indices": list(range(train_sequences, count)),
        "quantile_probabilities": QUANTILES,
        "test_error_reference": "squared coefficient-direction error about the training-sequence mean",
    }
    identity = {"config_sha256": record["config"]["sha256"],
                "checkpoint_sha256": record["selected_checkpoint"]["sha256"],
                "selected_checkpoint_step": completion["best_step"],
                "analysis_script_sha256": sha256(Path(__file__)), "settings": settings}
    output = run / OUTPUT
    if output.exists() and not args.overwrite:
        existing = json.loads(output.read_text())
        if args.if_applicable and all(existing.get(key) == value for key, value in identity.items()):
            print(f"Batch coefficient geometry: matching output exists at {output}; skipped.")
            return
        raise FileExistsError(f"Existing geometry differs or overwrite was not requested: {output}")
    measurements = analyze(sample_coefficients(checkpoint, recipe, positions), train_sequences)
    result = {
        "schema_version": 1, "run_id": run.name, "parent_run_id": oracle["parent_run_id"], **identity,
        "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "checkpoint_identity_sha256": sha256(checkpoint / "coefficient_identity.json"),
        "checkpoint_verification": "recorded file names and sizes; coefficient_identity.json SHA256",
        "completion_sha256": sha256(run / "completion.json"),
        "fitting_batch_sha256": json.loads((run / "data_identity.json").read_text())["batch_sha256"],
        "coefficient_array_shape": [count, recipe["model"]["num_layers"], length,
                                    recipe["model"]["num_heads"], recipe["model"]["attention_degree"] + 1],
        "limits": [
            "Subsampled query positions; every fitting sequence is represented.",
            "PCA is in normalized factorial coefficient coordinates, without observed attention scores.",
            "Coefficient reconstruction error does not establish preserved attention outputs or fitting perplexity.",
            "The sequence split tests a basis on other independently fitted coefficient tables, not unseen language-model inputs.",
            "This does not test whether a causal generator can predict the fitted coefficients.",
        ], **measurements,
    }
    if not args.dry_run:
        atomic_write(output, result, args.overwrite)
        update_record(run, capture_artifacts(run))
    summary = measurements["summary"]
    print(json.dumps({
        "run_id": run.name, "dry_run": args.dry_run,
        "median_pca_cumulative_variance_fraction": summary["pca_cumulative_variance_fraction_quantiles"][2],
        "median_test_cumulative_error_reduction_fraction": summary["test_cumulative_error_reduction_fraction_quantiles"][2],
        "median_components_90_percent": summary["components_90_percent_quantiles"][2],
        "median_components_95_percent": summary["components_95_percent_quantiles"][2],
        "output": None if args.dry_run else str(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
