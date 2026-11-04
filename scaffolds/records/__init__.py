"""Run identity, provenance, and derived indexes shared by every scaffold.

``sweep.yaml`` holds study metadata and an ordered ``run_ids`` list. A run owns
its immutable ``config.yaml`` and mutable authoritative ``record.json``. Metrics
are raw trainer output; ``runs.csv`` is always regenerated from those inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import uuid

import yaml


RUN_ID = re.compile(r"r[0-9a-f]{12}\Z")
INDEX_COLUMNS = (
    "run_id", "label", "status", "home_cluster", "model_id", "condition",
    "attention_mechanism", "attention_family", "attention_kernel", "qk_head_dim",
    "gdn_erase_mode", "gdn_write_activation",
    "attention_degree", "attention_normalization", "attention_output_scaling", "qk_norm",
    "qk_norm_type", "qk_norm_sharing", "qk_norm_head_scale", "qk_norm_token_scale", "initialization",
    "coefficient_mode", "coefficient_conditioning", "coefficient_activation",
    "coefficient_readout", "coefficient_feature_count", "coefficient_readout_scale", "coefficient_kernel_weight_decay", "attention_forget_gate", "attention_forget_gate_mode",
    "attention_forget_gate_form", "score_mlp_width", "score_mlp_form", "score_mlp_kernel_weight_decay",
    "coefficient_parameterization", "coefficient_initialization", "coefficient_sharing", "coefficient_orders",
    "denominator_degree", "denominator_parameterization", "denominator_constant", "coefficient_penalty",
    "coefficient_penalty_denominator", "query_coefficient_learning_rate_scale",
    "quadratic_mode", "weight_decay_policy", "seed", "training_steps", "prediction_tokens", "context_length",
    "source_commit",
    "terminal_step", "terminal_validation_nll", "terminal_validation_perplexity",
    "terminal_fit_nll", "terminal_fit_perplexity",
    "feature_selection", "active_features", "terminal_validation_relative_output_rms", "best_validation_relative_output_rms",
    "config", "record", "metrics",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _study_path(path: str | Path) -> Path:
    path = Path(path)
    return path.parent if path.name == "sweep.yaml" else path


def _run_path(path: str | Path) -> Path:
    path = Path(path)
    return path.parent if path.name == "record.json" else path


def _repository(path: Path) -> Path | None:
    # A checkout has .git/HEAD; a worktree has a .git file. A bare .git directory
    # elsewhere (e.g. a stray /tmp/.git on a shared node) is not a checkout.
    for parent in (path.resolve(), *path.resolve().parents):
        marker = parent / ".git"
        if marker.is_file() or (marker / "HEAD").is_file():
            return parent
    return None


def _stored_path(path: Path) -> str:
    root = _repository(path)
    return str(path.resolve().relative_to(root)) if root else str(path.resolve())


def _artifact_path(run: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = _repository(run)
    if root and path.parts and path.parts[0] in {"experiments", "scaffolds", "results"}:
        return root / path
    return run / path


@contextmanager
def _locked(directory: Path):
    """Serialize record edits without leaving a lock file in a study."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _new_file_mode() -> int:
    """The mode an ordinary new file gets under this process's umask."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Umask:"):
                return 0o666 & ~int(line.split()[1], 8)
    except OSError:
        pass
    return 0o644


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        # mkstemp creates the file private (0600); give it ordinary new-file permissions so
        # rewritten records stay readable wherever the umask shares the tree.
        os.fchmod(descriptor, _new_file_mode())
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) if path.suffix == ".yaml" else json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_run(path: str | Path) -> dict[str, Any]:
    """Read one run's authoritative record without changing it."""
    run = _run_path(path)
    record = _read_mapping(run / "record.json")
    if record.get("run_id") != run.name:
        raise ValueError(f"Run identity does not match its directory: {run}")
    return record


def load_study(path: str | Path) -> dict[str, Any]:
    """Combine study metadata and run records in memory for analysis callers.

    Original study fields, including ``format``, are preserved. ``runs`` is a
    list of authoritative per-run records, never another persisted manifest.
    """
    study = _study_path(path)
    common = _read_mapping(study / "sweep.yaml")
    run_ids = common.get("run_ids", [])
    if not isinstance(run_ids, list) or any(not isinstance(x, str) or not RUN_ID.fullmatch(x) for x in run_ids):
        raise ValueError(f"Invalid run_ids in {study / 'sweep.yaml'}")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"Duplicate run IDs in {study / 'sweep.yaml'}")
    if "runs" in common:
        raise ValueError("sweep.yaml must reference run_ids instead of duplicating run records")
    return {**common, "runs": [load_run(study / "runs" / run_id) for run_id in run_ids]}


def run_directory(study: str | Path, label_or_id: str) -> Path:
    """Resolve a permanent ID, or a unique descriptive label, within a study."""
    study = _study_path(study)
    manifest = load_study(study)
    matches = [r for r in manifest["runs"] if r["run_id"] == label_or_id]
    if not matches:
        matches = [r for r in manifest["runs"] if r.get("label", r.get("name")) == label_or_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one run for {label_or_id!r}, found {len(matches)} in {study}")
    return study / "runs" / matches[0]["run_id"]


def create_run(study: str | Path, config: Mapping[str, Any], label: str | None = None) -> Path:
    """Allocate a fresh execution, save its complete recipe, and register its ID.

    Callers supply a fully resolved scientific recipe. No model defaults are
    inferred here. Transformer output paths are filled when a ``train`` mapping
    is present; all other configuration values remain as supplied.
    """
    study = _study_path(study)
    study.mkdir(parents=True, exist_ok=True)
    (study / "runs").mkdir(exist_ok=True)
    with _locked(study):
        sweep_path = study / "sweep.yaml"
        common = _read_mapping(sweep_path) if sweep_path.exists() else {
            "schema_version": 1, "experiment": study.name, "run_ids": [],
        }
        registered = set(common.get("run_ids", []))
        for _ in range(100):
            run_id = "r" + uuid.uuid4().hex[:12]
            if run_id in registered:
                continue
            run = study / "runs" / run_id
            try:
                run.mkdir()
            except FileExistsError:
                continue
            break
        else:
            raise RuntimeError("Could not allocate a unique run ID after 100 attempts")
        recipe = deepcopy(dict(config))
        if isinstance(recipe.get("train"), dict):
            recipe["train"]["workdir"] = _stored_path(run)
            recipe["train"]["checkpoint_dir"] = _stored_path(run / "checkpoints")
        elif "output_directory" in recipe:
            recipe["output_directory"] = _stored_path(run)
        _atomic_write(run / "config.yaml", yaml.safe_dump(recipe, sort_keys=False))
        record = {
            "record_schema_version": 1, "run_id": run_id, "label": label or run_id,
            "name": label or run_id, "status": "planned",
            "config": {"path": _stored_path(run / "config.yaml"), "sha256": _sha256(run / "config.yaml")},
        }
        _atomic_write(run / "record.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
        common.setdefault("run_ids", []).append(run_id)
        _atomic_write(sweep_path, yaml.safe_dump(common, sort_keys=False))
    regenerate_index(study)
    return run


def _merge(destination: dict[str, Any], changes: Mapping[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
            _merge(destination[key], value)
        else:
            destination[key] = deepcopy(value)


def update_record(path: str | Path, changes: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically merge provenance updates; lists are replaced, never appended.

    Identity and the frozen recipe descriptor cannot be changed. Standalone
    trainer smoke runs may create a record outside a registered study. Index
    regeneration is explicit so a trainer does not rescan all study metrics.
    """
    run = _run_path(path)
    run.mkdir(parents=True, exist_ok=True)
    with _locked(run):
        record = load_run(run) if (run / "record.json").exists() else {
            "record_schema_version": 1, "run_id": run.name, "label": run.name, "name": run.name,
        }
        if "config" not in record and (run / "config.yaml").exists():
            record["config"] = {"path": _stored_path(run / "config.yaml"), "sha256": _sha256(run / "config.yaml")}
        for key in ("run_id", "label", "name", "config"):
            if key in changes and key in record and changes[key] != record[key]:
                raise ValueError(f"Cannot change a run's immutable {key}")
        _merge(record, changes)
        _atomic_write(run / "record.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _tree_descriptor(directory: Path) -> dict[str, Any]:
    files = []
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        checksum = _sha256(path)
        files.append({"path": relative, "sha256": checksum, "size_bytes": path.stat().st_size})
        digest.update(relative.encode("utf-8") + b"\0" + checksum.encode("ascii") + b"\n")
    return {
        "kind": "directory", "path": _stored_path(directory), "sha256": digest.hexdigest(),
        "hash_format": "sha256(relative-path NUL file-sha256 NEWLINE), sorted by relative path",
        "files": files, "file_count": len(files), "size_bytes": sum(item["size_bytes"] for item in files),
    }


def capture_artifacts(
    path: str | Path, checkpoint_directory: str | Path | None = None, include_log: bool = False,
) -> dict[str, Any]:
    """Hash present raw outputs and return updates for ``update_record``.

    Call after output files are closed, and after final checkpoint pruning.
    ``checkpoint_directory`` names the selected step's directory, rather than
    its checkpoint-manager root. The caller chooses the checkpoint; this helper
    never substitutes the latest checkpoint for the validation best. Call with
    ``include_log=True`` after training exits and the launcher has flushed its
    final output. ``record.json`` is excluded to avoid a checksum cycle.
    """
    run = _run_path(path)
    artifacts = []
    updates: dict[str, Any] = {}
    candidates = sorted({*run.glob("*.json"), *run.glob("*.jsonl"), *run.glob("*.npz")})
    for artifact in candidates:
        if artifact.name == "record.json" or not artifact.is_file():
            continue
        descriptor = {"path": _stored_path(artifact), "sha256": _sha256(artifact), "size_bytes": artifact.stat().st_size}
        artifacts.append(descriptor)
        if artifact.name in {"metrics.jsonl", "metadata.json", "completion.json", "best_validation.json", "retention.json", "data_identity.json", "wandb_run.json"}:
            updates[artifact.stem] = descriptor
    if checkpoint_directory is not None:
        checkpoint = Path(checkpoint_directory)
        if not checkpoint.is_absolute() and not checkpoint.is_dir():
            checkpoint = _artifact_path(run, str(checkpoint))
        if not checkpoint.is_dir():
            raise ValueError(f"Selected checkpoint is unavailable: {checkpoint}")
        updates["selected_checkpoint"] = _tree_descriptor(checkpoint)
    if include_log and (run / "training.log").is_file():
        log = run / "training.log"
        updates["log"] = {"path": _stored_path(log), "sha256": _sha256(log), "size_bytes": log.stat().st_size}
        artifacts.append(updates["log"])
    updates["artifacts"] = artifacts
    return updates


def _tree_sizes(directory: Path) -> dict[str, int]:
    """Relative path and size of every file in a tree, read from metadata alone."""
    return {path.relative_to(directory).as_posix(): path.stat().st_size
            for path in directory.rglob("*") if path.is_file()}


FAILURE_STATUSES = frozenset({"numerical_failure", "acceptance_numerical_failure"})


def _check_outcome(run: Path, record: Mapping[str, Any]) -> None:
    """The record's status is the run's single outcome; its outcome files must agree with it."""
    status = record.get("status")
    failures = sorted(path.name for path in run.glob("numerical_failure_step_*.json"))
    if (run / "completion.json").is_file() and status != "complete":
        raise ValueError(f"{record['run_id']} has a completion record but status {status!r}")
    if failures and status not in FAILURE_STATUSES:
        raise ValueError(f"{record['run_id']} has status {status!r} but numerical failure records {failures}")
    if status == "numerical_failure" and len(failures) != 1:
        raise ValueError(f"{record['run_id']} is a numerical failure with {len(failures)} failure records")


def validate_record(path: str | Path, verify_hashes: bool = True, checkpoint_hashes: bool = True) -> dict[str, Any]:
    """Validate identity and recipe integrity, plus hashes of available artifacts.

    Checkpoints and logs can live on another cluster. Missing references are
    returned as ``unavailable``; a missing canonical recipe is always an error.
    Re-hashing a checkpoint reads all of it. With ``checkpoint_hashes=False`` a
    checkpoint tree is instead compared with its recorded file names and sizes,
    and is reported as ``size_checked`` rather than ``verified``.
    """
    run = _run_path(path)
    record = load_run(run)
    if not RUN_ID.fullmatch(record["run_id"]):
        raise ValueError(f"Invalid canonical run ID: {record['run_id']}")
    if not isinstance(record.get("label"), str) or not record["label"]:
        raise ValueError("A run requires a nonempty label")
    recipe = record.get("config")
    if not isinstance(recipe, dict) or not isinstance(recipe.get("path"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(recipe.get("sha256", ""))):
        raise ValueError("A run requires a config path and SHA256")
    if _artifact_path(run, recipe["path"]).resolve() != (run / "config.yaml").resolve():
        raise ValueError("The canonical recipe must be the run's config.yaml")
    if not (run / "config.yaml").is_file():
        raise ValueError(f"Missing canonical recipe in {run}")
    _check_outcome(run, record)
    result: dict[str, Any] = {"run_id": record["run_id"], "verified": [], "size_checked": [], "unavailable": []}
    if not verify_hashes:
        return result

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                artifact = _artifact_path(run, value["path"])
                if value.get("kind") == "directory":
                    if artifact.is_dir() and checkpoint_hashes:
                        actual = _tree_descriptor(artifact)
                        if any(actual[key] != value.get(key) for key in ("sha256", "files", "file_count", "size_bytes")):
                            raise ValueError(f"Checkpoint tree checksum mismatch: {artifact}")
                        result["verified"].append(value["path"])
                    elif artifact.is_dir():
                        recorded = {item["path"]: item["size_bytes"] for item in value.get("files", [])}
                        if _tree_sizes(artifact) != recorded:
                            raise ValueError(f"Checkpoint tree file names or sizes differ from the record: {artifact}")
                        result["size_checked"].append(value["path"])
                    else:
                        result["unavailable"].append(value["path"])
                    return
                elif artifact.is_file():
                    if _sha256(artifact) != value["sha256"]:
                        raise ValueError(f"SHA256 mismatch: {artifact}")
                    result["verified"].append(value["path"])
                elif not artifact.exists():
                    result["unavailable"].append(value["path"])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(record)
    for key in ("verified", "size_checked", "unavailable"):
        result[key] = sorted(set(result[key]))
    return result


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _attention_family(model: Mapping[str, Any], record: Mapping[str, Any], kernel: Any) -> Any:
    """Name the scientific family without identifying custom coefficients by label."""
    explicit = _first(record.get("polynomial_family"), record.get("attention_family"))
    if explicit is not None:
        return explicit
    mechanism = model.get("attention_mechanism")
    if mechanism is not None and mechanism != "dot_product":
        return mechanism
    families = {
        "gla": "gla", "softmax": "softmax", "exp_unnormalized": "exponential",
        "taylor": "taylor", "taylor_unnormalized": "taylor",
        "learned_quadratic": "learned_polynomial", "learned_polynomial": "learned_polynomial",
        "learned_polynomial_unnormalized": "learned_polynomial",
        "learned_polynomial_sum_of_squares": "learned_polynomial",
        "learned_polynomial_rational": "learned_polynomial",
        "fixed_polynomial": "polynomial", "fixed_polynomial_unnormalized": "polynomial",
        "fixed_polynomial_sum_of_squares": "polynomial",
        "score_mlp_unnormalized": "score_mlp", "exp_score_mlp_residual_unnormalized": "score_mlp",
    }
    if kernel in families:
        return families[kernel]
    # A score MLP keeps fixed coefficients and no degree, which would otherwise read as exp.
    if model.get("attention_score_mlp_width") is not None:
        return "score_mlp"
    if model.get("attention_fixed_coefficients") is not None or model.get("attention_fixed_square_factors") is not None:
        return "polynomial"
    if model.get("attention_coefficients") == "learned" or model.get("quadratic_mode") in {"positive", "unrestricted"}:
        return "learned_polynomial"
    if model.get("attention_coefficients") == "fixed" and "attention_degree" in model:
        if model["attention_degree"] is not None:
            return "taylor"
        if "attention_normalization" in model:
            return "softmax" if model["attention_normalization"] else "exponential"
    return None


def _index_row(study: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    run = study / "runs" / record["run_id"]
    config = _read_mapping(run / "config.yaml")
    model = config.get("model", {})
    model_id = model if isinstance(model, str) else model.get("model_id") if isinstance(model, dict) else None
    model = model if isinstance(model, dict) else {}
    train = config.get("train", {})
    train = train if isinstance(train, dict) else {}
    optim = config.get("optim", {})
    optim = optim if isinstance(optim, dict) else {}
    data = config.get("data", {})
    data = data if isinstance(data, dict) else {}
    condition = config.get("condition", {})
    condition_name = condition.get("name") if isinstance(condition, dict) else condition
    condition_seed = condition.get("seed") if isinstance(condition, dict) else None
    metadata = _read_mapping(run / "metadata.json") if (run / "metadata.json").exists() else {}
    completion = _read_mapping(run / "completion.json") if (run / "completion.json").exists() else {}
    terminal = record.get("terminal") or {}
    validations = []
    metrics_path = run / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                metric = json.loads(line)
                if "validation_nll" in metric:
                    validations.append(metric)
    last_validation = max(validations, key=lambda x: x.get("step", -1), default={})
    sources = (terminal, completion, metadata, last_validation)
    get = lambda key: _first(*(source.get(key) for source in sources))
    mode = _first(model.get("attention_coefficients"), record.get("coefficient_mode"))
    if model.get("quadratic_mode") in {"positive", "unrestricted"}:
        mode = "learned"
    elif mode is None and model.get("quadratic_mode") == "fixed":
        mode = "fixed"
    kernel = _first(record.get("attention_kernel"), metadata.get("attention_kernel"), model.get("attention_kernel"))
    sharing = _first(record.get("coefficient_sharing"), metadata.get("attention_coefficient_sharing"), model.get("attention_coefficient_sharing"), model.get("quadratic_sharing"))
    budget = (optim.get("total_steps"), data.get("batch_size"), data.get("sequence_length"))
    prediction_tokens = budget[0] * budget[1] * (budget[2] - 1) if all(type(value) is int for value in budget) else None
    if config.get("experiment_kind") == "teacher_activation_coefficient_distillation":
        # Updates fit coefficient targets on repeated teacher activations; they
        # are not additional language-model pretraining tokens.
        prediction_tokens = None
    fit = config.get("fit", {}) if config.get("experiment_kind") == "frozen_pretrained_head_distillation" else {}
    oracle = config.get("oracle", {})
    shared_batch_coefficients = (isinstance(oracle, dict)
                                 and oracle.get("parameterization") == "shared_batch_query_coefficients")
    static_coefficients = isinstance(oracle, dict) and oracle.get("parameterization") == "static_coefficients"
    free_coefficients = (isinstance(oracle, dict)
                         and oracle.get("parameterization") == "free_per_example_query_coefficients")
    coefficient_table = shared_batch_coefficients or static_coefficients or free_coefficients
    if free_coefficients:
        mode = "learned"
        kernel = "learned_polynomial_unnormalized"
        sharing = "per_layer_per_head_per_sequence_per_position"

    def relative_output_rms(rows):
        if not isinstance(rows, list) or not rows:
            return None
        values = [row.get("output_relative_rms") for row in rows]
        if any(value is None for value in values) or any(row.get("invalid_fraction", 0) for row in rows):
            return None
        return (sum(value * value for value in values) / len(values)) ** 0.5

    return {
        "run_id": record["run_id"], "label": record.get("label", record.get("name")),
        "status": _first(record.get("status"), completion.get("status"), metadata.get("status")),
        "home_cluster": record.get("home_cluster"),
        "model_id": _first(model_id, record.get("model_id"), metadata.get("model_id")),
        "condition": _first(condition_name, record.get("condition"), metadata.get("condition")),
        "attention_mechanism": _first(model.get("attention_mechanism"), metadata.get("attention_mechanism")),
        "attention_family": "learned_polynomial" if free_coefficients else _attention_family(model, record, kernel),
        "attention_kernel": kernel,
        "gdn_erase_mode": _first(model.get("gdn_erase_mode"), get("gdn_erase_mode"))
            if model.get("attention_mechanism") == "gated_deltanet" else None,
        "gdn_write_activation": _first(model.get("gdn_write_activation"), get("gdn_linear_write_activation"))
            if model.get("attention_mechanism") == "gated_deltanet" else None,
        "qk_head_dim": _first(model.get("qk_head_dim"), record.get("qk_head_dim"), metadata.get("qk_head_dim")),
        "attention_degree": _first(model.get("attention_degree"), record.get("attention_degree"), metadata.get("attention_degree")),
        "attention_normalization": _first(model.get("attention_normalization"), record.get("attention_normalization"), metadata.get("attention_normalization")),
        "attention_output_scaling": model.get("attention_output_scaling"),
        "qk_norm": _first(model.get("qk_norm"), record.get("qk_norm"), metadata.get("qk_norm")),
        "qk_norm_type": model.get("qk_norm_type"),
        "qk_norm_sharing": model.get("qk_norm_sharing"),
        "qk_norm_head_scale": model.get("qk_norm_head_scale"),
        "qk_norm_token_scale": model.get("qk_norm_token_scale"),
        "initialization": model.get("initialization"),
        "coefficient_mode": mode,
        "coefficient_conditioning": "sequence_position" if free_coefficients else "none" if static_coefficients else "position" if shared_batch_coefficients else model.get("attention_coefficient_conditioning"),
        "coefficient_activation": "identity" if coefficient_table else model.get("attention_coefficient_activation"),
        "coefficient_readout": None if coefficient_table else model.get("attention_coefficient_readout"),
        "coefficient_feature_count": None if coefficient_table else model.get("attention_coefficient_feature_count"),
        "coefficient_readout_scale": None if coefficient_table else model.get("attention_coefficient_readout_scale"),
        "coefficient_kernel_weight_decay": None if coefficient_table else model.get("attention_coefficient_kernel_weight_decay"),
        "attention_forget_gate": model.get("attention_forget_gate"),
        "attention_forget_gate_mode": model.get("attention_forget_gate_mode", "cumulative")
            if model.get("attention_forget_gate") else None,
        "attention_forget_gate_form": model.get("attention_forget_gate_form", "input")
            if model.get("attention_forget_gate") else None,
        "score_mlp_width": model.get("attention_score_mlp_width"),
        "score_mlp_form": model.get("attention_score_mlp_form", "kernel")
            if model.get("attention_score_mlp_width") is not None else None,
        "score_mlp_kernel_weight_decay": model.get("attention_score_mlp_kernel_weight_decay", False)
            if model.get("attention_score_mlp_width") is not None else None,
        "coefficient_parameterization": oracle["parameterization"] if coefficient_table else _first(model.get("attention_coefficient_parameterization"), record.get("attention_coefficient_parameterization")),
        "coefficient_initialization": "captured_parent_coefficients" if free_coefficients else "parent_checkpoint" if static_coefficients and oracle.get("static_initialization") == "checkpoint" else "mean_parent_token_coefficients" if static_coefficients else "mean_parent_batch_coefficients" if shared_batch_coefficients else model.get("attention_coefficient_initialization"),
        "coefficient_sharing": sharing,
        "coefficient_orders": json.dumps(model["attention_coefficient_orders"], separators=(",", ":"))
            if not coefficient_table and model.get("attention_coefficient_orders") is not None else None,
        "denominator_degree": model.get("attention_denominator_degree"),
        "denominator_parameterization": model.get("attention_denominator_parameterization"),
        "denominator_constant": model.get("attention_denominator_constant"),
        "coefficient_penalty": optim.get("coefficient_penalty"),
        "coefficient_penalty_denominator": optim.get("coefficient_penalty_denominator"),
        "query_coefficient_learning_rate_scale": optim.get("query_coefficient_learning_rate_scale"),
        "quadratic_mode": model.get("quadratic_mode"),
        "weight_decay_policy": optim.get("weight_decay_policy"),
        "seed": _first(train.get("seed"), config.get("seed"), condition_seed, record.get("seed"), metadata.get("seed")),
        "training_steps": _first(optim.get("total_steps"), fit.get("steps")), "prediction_tokens": prediction_tokens,
        "context_length": model.get("max_seq_len") if optim.get("total_steps") is not None else None,
        "source_commit": _first(record.get("source_commit"), metadata.get("source_commit"), train.get("source_commit")),
        "terminal_step": _first(terminal.get("step"), record.get("training_updates"), get("step")),
        "terminal_validation_nll": _first(record.get("terminal_validation_nll"), get("validation_nll")),
        "terminal_validation_perplexity": _first(record.get("terminal_validation_perplexity"), get("validation_perplexity")),
        "terminal_fit_nll": _first(record.get("terminal_fit_nll"), get("fit_nll")),
        "terminal_fit_perplexity": _first(record.get("terminal_fit_perplexity"), get("fit_perplexity")),
        "feature_selection": fit.get("variant"), "active_features": fit.get("budget"),
        "terminal_validation_relative_output_rms": relative_output_rms(completion.get("terminal_validation")),
        "best_validation_relative_output_rms": relative_output_rms(completion.get("best_validation")),
        "config": f"runs/{record['run_id']}/config.yaml",
        "record": f"runs/{record['run_id']}/record.json",
        "metrics": f"runs/{record['run_id']}/metrics.jsonl" if metrics_path.exists() else "",
    }


# Record subtrees that describe past executions rather than where retained files
# live now: submission commands, Slurm attempts, resolved runtime configurations,
# and the acceptance directories whose tests a later attempt reused.
EXECUTION_HISTORY = ("submission", "analysis_submissions", "attempts", "execution", "acceptance.summary")


def rename_study_paths(record: Mapping[str, Any], old: str, new: str, roots: tuple[str, ...] = ()) -> tuple[dict[str, Any], int]:
    """Point a record's retained-file locations at a renamed study folder.

    Repository-relative paths under ``experiments/<old>/``, absolute paths under
    ``<root>/experiments/<old>/`` for each of ``roots``, and ``parent_study``/``study``
    names equal to ``old`` move to ``new``. Locations under other roots (another
    cluster's artifact root, a worktree) and the subtrees in ``EXECUTION_HISTORY``
    stay as they are. Returns the rewritten record and the number of changed strings.
    """
    prefixes = [(f"experiments/{old}/", f"experiments/{new}/")]
    prefixes += [(f"{root.rstrip('/')}/experiments/{old}/", f"{root.rstrip('/')}/experiments/{new}/") for root in roots]
    count = 0

    def visit(value: Any, path: list[str]) -> Any:
        nonlocal count
        dotted = ".".join(path)
        if any(dotted == key or dotted.startswith(key + ".") for key in EXECUTION_HISTORY):
            return value
        if isinstance(value, dict):
            return {key: visit(item, path + [key]) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, path + ["[]"]) for item in value]
        if isinstance(value, str):
            if path and path[-1] in ("parent_study", "study") and value == old:
                count += 1
                return new
            for before, after in prefixes:
                if value.startswith(before):
                    count += 1
                    return after + value[len(before):]
                if value == before[:-1]:
                    count += 1
                    return after[:-1]
        return value

    return visit(deepcopy(dict(record)), []), count


def rename_study(study: str | Path, old: str, new: str, artifact_roots: tuple[str, ...] = ()) -> list[str]:
    """Rewrite records, selections, and the membership file after ``git mv experiments/<old> experiments/<new>``.

    ``study`` is the renamed folder. The checkout root always counts as a root whose
    ``experiments/<old>/`` tree moved; ``artifact_roots`` adds others (for example a
    cluster's artifact root, after moving that tree too). Frozen recipes are inputs
    and keep their paths; execution history keeps its paths; the record says where
    each retained file lives now. Returns one line per changed file.
    """
    study = _study_path(study).resolve()
    root = _repository(study)
    if root is None:
        raise ValueError(f"Not inside a Git checkout: {study}")
    roots = (str(root), *artifact_roots)
    changes: list[str] = []
    for run in sorted(path for path in (study / "runs").iterdir() if path.is_dir()):
        record_path = run / "record.json"
        if not record_path.is_file():
            continue
        original = record_path.read_text()
        record = json.loads(original)
        renamed, count = rename_study_paths(record, old, new, roots)
        if count:
            if json.dumps(record, indent=2, sort_keys=True) + "\n" != original:
                raise ValueError(f"{record_path} is not in the records' canonical JSON style; rewrite it by hand")
            _atomic_write(record_path, json.dumps(renamed, indent=2, sort_keys=True) + "\n")
            changes.append(f"{run.name}: {count} locations")
    study_pattern = re.compile(rf"^(\s*-?\s*study: ){re.escape(old)}$", re.MULTILINE)
    for directory in (root / "results", study / "results"):
        if not directory.is_dir():
            continue
        for selection in sorted(directory.rglob("*.yaml")):
            if "temp" in selection.relative_to(root).parts:
                continue
            text = selection.read_text()
            updated = study_pattern.sub(rf"\g<1>{new}", text.replace(f"experiments/{old}/", f"experiments/{new}/"))
            if updated != text:
                _atomic_write(selection, updated)
                changes.append(str(selection.relative_to(root)))
    sweep = study / "sweep.yaml"
    if sweep.is_file():
        text = sweep.read_text()
        updated = re.sub(rf"^experiment: {re.escape(old)}$", f"experiment: {new}", text, flags=re.MULTILINE)
        if updated != text:
            _atomic_write(sweep, updated)
            changes.append(str(sweep.relative_to(root)))
    return changes


def regenerate_index(study: str | Path) -> Path:
    """Write a byte-stable CSV from registered configs, records, and raw metrics.

    Missing legacy settings stay empty. No current model defaults are silently
    applied to historical runs.
    """
    study = _study_path(study)
    with _locked(study):
        manifest = load_study(study)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=INDEX_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in sorted(manifest["runs"], key=lambda item: item["run_id"]):
            writer.writerow(_index_row(study, record))
        _atomic_write(study / "runs.csv", output.getvalue())
    return study / "runs.csv"


__all__ = ["capture_artifacts", "create_run", "load_run", "load_study", "regenerate_index", "run_directory", "update_record", "validate_record"]
