"""Collect one checksummed, comparable terminal result per declared heatmap cell."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


_COMPLETE = {"complete", "completed"}
_NUMERICAL_FAILURE = {"numerical-failure", "numerical_failure"}
# The scaffold's current defaults. Runs trained under anything else are shown as
# legacy cells until their experiment is rerun.
CANONICAL_WEIGHT_DECAY_POLICY = "exclude_embeddings_norms_biases_v1"
_REFERENCE_KERNELS = {"softmax": "softmax", "softmax_no_qk_norm": "softmax",
                      "gla": "gla", "unnormalized_exp": "exp_unnormalized"}


def source_artifact_set_sha256(rows: list[dict]) -> str:
    """Pin all selected run files, including files older manifests do not hash."""
    entries = sorted((artifact["path"], artifact["sha256"])
                     for row in rows for artifact in row.get("artifacts", {}).values())
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _match(value: object, expected: object, description: str) -> None:
    if value != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {value!r}")


def _optional_match(record: dict, key: str, expected: object, description: str) -> None:
    if key in record:
        _match(record[key], expected, f"{description} {key}")


def _artifact(repo_root: Path, experiment: str, run_name: str, name: str, run: dict) -> dict:
    relative = f"experiments/{experiment}/runs/{run_name}/{name}"
    declared = run.get(name.removesuffix(".jsonl").removesuffix(".json"))
    if name == "completion.json" and isinstance(run.get("completion_artifact"), dict):
        declared = run["completion_artifact"]
    if name.startswith("numerical_failure_step_"):
        declared = run.get("failure")
    expected_sha = None
    if isinstance(declared, dict):
        _match(declared.get("path"), relative, f"Artifact path for {run_name}")
        expected_sha = declared.get("sha256")
    elif isinstance(declared, str):
        _match(declared, relative, f"Artifact path for {run_name}")
    run_files = run.get("run_files", {})
    if name in run_files:
        if expected_sha is not None:
            _match(run_files[name], expected_sha, f"Manifest hashes for {relative}")
        expected_sha = run_files[name]
    path = repo_root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing canonical run artifact: {relative}")
    actual_sha = _digest(path)
    if expected_sha is not None:
        _match(actual_sha, expected_sha, f"Artifact SHA256 for {relative}")
    return {"path": relative, "sha256": actual_sha}


def _expected_degree(panel: str, column: str) -> int | None:
    match = re.fullmatch(r"m(\d+)", column)
    if match:
        return int(match.group(1))
    if panel == "normalized_learned":
        return 2
    return None


def _identity(record: dict, *, source: str, width: int, degree: int | None, label: str) -> None:
    _match(record.get("source_commit"), source, f"{label} source commit")
    _match(record.get("qk_head_dim"), width, f"{label} d_K")
    _optional_match(record, "attention_degree", degree, label)


def _mechanism(cell: dict, run: dict, metadata: dict, label: str) -> None:
    panel, column = cell["panel"], cell["column"]
    if panel == "normalized_learned":
        mode, separator, sharing = column.partition("_")
        if not separator or mode not in {"positive", "unrestricted"} or sharing not in {"global", "per_head"}:
            raise ValueError(f"Unknown learned quadratic column: {column}")
        for record in (run, metadata):
            _match(record.get("quadratic_mode"), mode, f"{label} quadratic mode")
            _match(record.get("quadratic_sharing"), sharing, f"{label} quadratic sharing")
    expected_kernel = {
        "unnormalized_fixed": "taylor_unnormalized",
        "unnormalized_learned": "learned_polynomial_unnormalized",
    }.get(panel)
    if panel == "references":
        expected_kernel = _REFERENCE_KERNELS.get(column)
    if expected_kernel == "softmax":
        # Width-sweep softmax metadata predates the kernel field; its degree identity is checked above.
        _optional_match(metadata, "attention_kernel", expected_kernel, f"{label} kernel")
    elif expected_kernel is not None:
        _match(metadata.get("attention_kernel"), expected_kernel, f"{label} kernel")
    if panel == "unnormalized_learned":
        _match(metadata.get("attention_coefficients"), "learned", f"{label} coefficients")


def _defaults_status(repo_root: Path, experiment: str, name: str, column: dict, label: str) -> dict:
    """How a run relates to the current defaults: its resolved decay policy and QK-norm.

    The trainer's `config.json` is the resolved configuration; runs recorded before it
    existed trained with decay on all parameters and without QK-norm. A column may
    declare `qk_norm: false` (a plain-score reference) or `qk_norm: null` (a mechanism
    QK-norm does not apply to); otherwise QK-norm is expected.
    """
    config_path = repo_root / f"experiments/{experiment}/runs/{name}/config.json"
    if config_path.exists():
        config = _json(config_path)
        policy = config.get("optim", {}).get("weight_decay_policy")
        qk_norm = bool(config.get("model", {}).get("qk_norm", False))
    else:
        policy, qk_norm = None, False
    expects = column.get("qk_norm", True)
    reasons = []
    if policy != CANONICAL_WEIGHT_DECAY_POLICY:
        reasons.append("weight decay on all parameters")
    if expects is True and not qk_norm:
        reasons.append("no QK-norm")
    if expects is False and qk_norm:
        raise ValueError(f"{label} uses QK-norm but its column is the plain-score reference")
    return {"weight_decay_policy": policy, "qk_norm": qk_norm, "legacy": bool(reasons), "legacy_reasons": reasons}


def _data_identity(metadata: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        _match(metadata.get("data_identity", {}).get(key), value,
               f"{label} data identity {key}")


def _terminal(record: dict, step: int, tokens: int, label: str) -> tuple[float, float]:
    _match(record.get("step", record.get("final_step")), step, f"{label} step")
    _match(record.get("training_tokens"), tokens, f"{label} training tokens")
    nll = record.get("validation_nll")
    ppl = record.get("validation_perplexity")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in (nll, ppl)):
        raise ValueError(f"Missing finite terminal validation in {label}")
    return float(nll), float(ppl)


def _manifest_terminal(run: dict, step: int, tokens: int, nll: float, ppl: float) -> None:
    name = run["name"]
    if isinstance(run.get("terminal"), dict):
        _match(_terminal(run["terminal"], step, tokens, f"{name} manifest terminal"),
               (nll, ppl), f"{name} terminal values")
    else:
        _match(run.get("terminal_validation_nll"), nll, f"{name} manifest NLL")
        _match(run.get("terminal_validation_perplexity"), ppl, f"{name} manifest PPL")
        _match(run.get("training_updates"), step, f"{name} training updates")
        _match(run.get("prediction_tokens"), tokens, f"{name} prediction tokens")


def _collect_referenced(repo_root: Path, cell: dict, run: dict, step: int, tokens: int,
                        seed: int, data_identity: dict, column: dict | None = None) -> dict:
    experiment, name = cell["experiment"], cell["run"]
    width, degree = cell["d_k"], _expected_degree(cell["panel"], cell["column"])
    label = f"{experiment}/{name}"
    canonical_dir = f"experiments/{experiment}/runs/{name}"
    _match(run.get("name"), name, f"{label} run name")
    _optional_match(run, "run_record", canonical_dir, label)
    _optional_match(run, "workdir", canonical_dir, label)
    _match(run.get("seed"), seed, f"{label} seed")
    if "qk_head_dim" in run:
        _match(run["qk_head_dim"], width, f"{label} manifest d_K")
    if "attention_degree" in run:
        _match(run["attention_degree"], degree, f"{label} manifest degree")
    source = run.get("source_commit")
    if not isinstance(source, str) or not source:
        raise ValueError(f"Missing source commit for {label}")

    raw_status = run.get("status")
    has_terminal = isinstance(run.get("terminal"), dict) or "terminal_validation_perplexity" in run
    has_failure = isinstance(run.get("failure"), dict)
    if raw_status in _NUMERICAL_FAILURE or has_failure:
        status = "numerical_failure"
        if raw_status not in _NUMERICAL_FAILURE or not has_failure:
            raise ValueError(f"Contradictory numerical failure record: {label}")
    elif raw_status in _COMPLETE or (raw_status is None and has_terminal):
        status = "complete"
        if not has_terminal:
            raise ValueError(f"Completed run lacks manifest terminal validation: {label}")
    else:
        status = "incomplete"
        if has_terminal or raw_status in _COMPLETE:
            raise ValueError(f"Contradictory incomplete record: {label}")

    result = {"status": status, "terminal_ppl": None, "terminal_nll": None,
              "failure_step": None, "experiment": experiment, "run": name,
              "source_commit": source, "artifacts": {},
              **_defaults_status(repo_root, experiment, name, column or {}, label)}
    if status == "incomplete":
        metadata_path = repo_root / f"experiments/{experiment}/runs/{name}/metadata.json"
        if metadata_path.exists():
            result["artifacts"]["metadata"] = _artifact(repo_root, experiment, name, "metadata.json", run)
            metadata = _json(metadata_path)
            _match(metadata.get("seed"), seed, f"{label} metadata seed")
            _identity(metadata, source=source, width=width, degree=degree, label=label)
            _data_identity(metadata, data_identity, label)
            _mechanism(cell, run, metadata, label)
        return result

    result["artifacts"]["metadata"] = _artifact(repo_root, experiment, name, "metadata.json", run)
    metadata = _json(repo_root / result["artifacts"]["metadata"]["path"])
    _match(metadata.get("seed"), seed, f"{label} metadata seed")
    _identity(metadata, source=source, width=width, degree=degree, label=f"{label} metadata")
    _data_identity(metadata, data_identity, label)
    _mechanism(cell, run, metadata, label)
    _optional_match(metadata, "target_steps", step, f"{label} metadata")
    _optional_match(metadata, "target_tokens", tokens, f"{label} metadata")

    if status == "numerical_failure":
        _match(metadata.get("status"), "failed", f"{label} failure metadata status")
        failure_ref = run["failure"]
        failure_name = Path(failure_ref["path"]).name
        if not re.fullmatch(r"numerical_failure_step_\d+\.json", failure_name):
            raise ValueError(f"Unexpected numerical failure file: {label}")
        result["artifacts"]["failure"] = _artifact(repo_root, experiment, name, failure_name, run)
        failure = _json(repo_root / result["artifacts"]["failure"]["path"])
        failed_step = failure.get("step")
        if not isinstance(failed_step, int) or failed_step < 1 or failed_step >= step:
            raise ValueError(f"Invalid numerical failure step for {label}: {failed_step}")
        _match(failure_ref.get("step"), failed_step, f"{label} manifest failure step")
        _match(metadata.get("step"), failed_step, f"{label} metadata failure step")
        _optional_match(failure, "attention_degree", degree, f"{label} failure")
        if not failure.get("error") or not metadata.get("error"):
            raise ValueError(f"Missing numerical failure diagnosis for {label}")
        if (repo_root / f"experiments/{experiment}/runs/{name}/completion.json").exists():
            raise ValueError(f"Numerical failure has a completion artifact: {label}")
        result["failure_step"] = failed_step
        return result

    _match(metadata.get("status"), "complete", f"{label} metadata status")
    _match(metadata.get("step"), step, f"{label} metadata step")
    _match(metadata.get("training_tokens"), tokens, f"{label} metadata tokens")
    result["artifacts"]["metrics"] = _artifact(repo_root, experiment, name, "metrics.jsonl", run)
    result["artifacts"]["completion"] = _artifact(repo_root, experiment, name, "completion.json", run)
    metrics_path = repo_root / result["artifacts"]["metrics"]["path"]
    validations = []
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "validation":
            _identity(record, source=source, width=width, degree=degree,
                      label=f"{label} validation line {line_number}")
            validations.append(record)
    if not validations:
        raise ValueError(f"No validation records in {label}")
    last = validations[-1]
    terminal_nll, terminal_ppl = _terminal(last, step, tokens, f"{label} last validation")
    completion = _json(repo_root / result["artifacts"]["completion"]["path"])
    _identity(completion, source=source, width=width, degree=degree, label=f"{label} completion")
    _match(_terminal(completion, step, tokens, f"{label} completion"),
           (terminal_nll, terminal_ppl), f"{label} completion validation")
    _manifest_terminal(run, step, tokens, terminal_nll, terminal_ppl)
    result["terminal_nll"] = terminal_nll
    result["terminal_ppl"] = terminal_ppl
    return result


def collect_rows(repo_root: Path, config: dict) -> list[dict]:
    """Return the complete declared grid, with source runs checked against pinned manifests."""
    repo_root = Path(repo_root)
    widths = config["d_k"]
    if not widths or len(widths) != len(set(widths)) or any(type(width) is not int or width <= 0 for width in widths):
        raise ValueError("d_k must contain unique positive integers")
    step, tokens, seed = config["terminal_step"], config["training_tokens"], config["seed"]
    if type(step) is not int or step <= 0 or type(tokens) is not int or tokens <= 0:
        raise ValueError("Terminal budget must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    data_identity = config.get("data_identity", {})
    if not isinstance(data_identity, dict):
        raise ValueError("Data identity must be an object")
    expected = []
    panel_ids = set()
    for panel in config["panels"]:
        panel_id = panel["id"]
        if panel_id in panel_ids:
            raise ValueError(f"Duplicate panel: {panel_id}")
        panel_ids.add(panel_id)
        column_ids = set()
        for column in panel["columns"]:
            column_id = column["id"]
            if column_id in column_ids:
                raise ValueError(f"Duplicate column: {panel_id}/{column_id}")
            column_ids.add(column_id)
            expected.extend((panel_id, column_id, width) for width in widths)
    if not expected:
        raise ValueError("Empty heatmap grid")
    expected_set = set(expected)

    manifests = {}
    for experiment, reference in config["source_manifests"].items():
        path = f"experiments/{experiment}/provenance/manifest.json"
        _match(reference.get("path"), path, f"Source manifest path for {experiment}")
        actual = _digest(repo_root / path)
        _match(actual, reference.get("sha256"), f"Source manifest SHA256 for {experiment}")
        manifest = _json(repo_root / path)
        _match(manifest.get("experiment"), experiment, f"Source manifest identity for {experiment}")
        indexed = {}
        for run in manifest.get("runs", []):
            name = run["name"]
            if name in indexed:
                raise ValueError(f"Duplicate source run: {experiment}/{name}")
            indexed[name] = run
        manifests[experiment] = indexed

    columns = {(panel["id"], column["id"]): column for panel in config["panels"] for column in panel["columns"]}
    references = {}
    used_runs = set()
    for cell in config["cells"]:
        coordinate = (cell["panel"], cell["column"], cell["d_k"])
        if coordinate not in expected_set or coordinate in references:
            raise ValueError(f"Duplicate or out-of-grid cell reference: {coordinate}")
        experiment, name = cell["experiment"], cell["run"]
        if experiment not in manifests or name not in manifests[experiment]:
            raise ValueError(f"Unknown source run: {experiment}/{name}")
        if (experiment, name) in used_runs:
            raise ValueError(f"Source run is referenced twice: {experiment}/{name}")
        used_runs.add((experiment, name))
        references[coordinate] = (cell, manifests[experiment][name])

    rows = []
    for panel, column, width in expected:
        coordinate = (panel, column, width)
        row = {"panel": panel, "column": column, "d_k": width,
               "status": "not_run", "terminal_ppl": None, "terminal_nll": None,
               "failure_step": None, "experiment": None, "run": None,
               "source_commit": None, "artifacts": {}, "weight_decay_policy": None,
               "qk_norm": None, "legacy": False, "legacy_reasons": []}
        if coordinate in references:
            cell, run = references[coordinate]
            row.update(_collect_referenced(repo_root, cell, run, step, tokens, seed, data_identity,
                                           columns[(panel, column)]))
        rows.append(row)
    if expected_sha := config.get("source_artifact_set_sha256"):
        _match(source_artifact_set_sha256(rows), expected_sha, "Source artifact set SHA256")
    return rows
