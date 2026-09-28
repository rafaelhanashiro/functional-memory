"""Collect one checksummed, comparable terminal result per declared heatmap cell."""

from __future__ import annotations

import hashlib
import json
import math
import re
import yaml
from scaffolds.records import load_run, load_study
from scaffolds.records.selections import verify_pinned_runs
from pathlib import Path


# The scaffold's current defaults. Runs trained under anything else are shown as
# legacy cells until their experiment is rerun.
CANONICAL_WEIGHT_DECAY_POLICY = "exclude_embeddings_norms_biases_v1"
_REFERENCE_KERNELS = {"softmax": "softmax", "softmax_no_qk_norm": "softmax",
                      "gla": "gla", "unnormalized_exp": "exp_unnormalized"}


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


def _artifact(repo_root: Path, experiment: str, run_name: str, name: str) -> dict:
    relative = f"experiments/{experiment}/runs/{run_name}/{name}"
    path = repo_root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing canonical run artifact: {relative}")
    return {"path": relative, "sha256": _digest(path)}


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


def _mechanism(cell: dict, model: dict, metadata: dict, label: str) -> None:
    panel, column = cell["panel"], cell["column"]
    if panel == "normalized_learned":
        mode, separator, sharing = column.partition("_")
        if not separator or mode not in {"positive", "unrestricted"} or sharing not in {"global", "per_head"}:
            raise ValueError(f"Unknown learned quadratic column: {column}")
        for record in (model, metadata):
            _match(record.get("quadratic_mode"), mode, f"{label} quadratic mode")
            _match(record.get("quadratic_sharing"), sharing, f"{label} quadratic sharing")
    if panel == "normalized_nonnegative":
        # The positive quadratic 0.5 (1 + (a s + b)^2) or the sum of squares q1^2 + q2^2, per head.
        kernel = metadata.get("attention_kernel")
        if kernel == "learned_quadratic":
            for record in (model, metadata):
                _match(record.get("quadratic_mode"), "positive", f"{label} quadratic mode")
                _match(record.get("quadratic_sharing"), "per_head", f"{label} quadratic sharing")
        else:
            _match(kernel, "learned_polynomial_sum_of_squares", f"{label} kernel")
            _match(metadata.get("attention_coefficient_sharing"), "per_head", f"{label} coefficient sharing")
        return
    expected_kernel = {
        "unnormalized_fixed": "taylor_unnormalized",
        "unnormalized_learned": "learned_polynomial_unnormalized",
        "normalized_learned_taylor": "learned_polynomial",
        "unnormalized_fixed_hermite": "fixed_polynomial_unnormalized",
    }.get(panel)
    # Taylor kernels use the implicit 1/r! coefficients; Hermite recipes state theirs, as
    # expanded coefficients or, for normalized degree 2, as fixed square factors.
    stated = model.get("attention_fixed_coefficients") is not None or model.get("attention_fixed_square_factors") is not None
    if panel == "unnormalized_fixed" and stated:
        raise ValueError(f"{label} states fixed coefficients; a Taylor cell uses 1/r!")
    if panel in ("normalized_fixed_hermite", "unnormalized_fixed_hermite") and not stated:
        raise ValueError(f"{label} states no fixed Hermite coefficients")
    if panel == "normalized_fixed_hermite":
        _match(metadata.get("attention_kernel") in ("fixed_polynomial", "fixed_polynomial_sum_of_squares"), True,
               f"{label} fixed normalized kernel")
    if panel == "references":
        expected_kernel = _REFERENCE_KERNELS.get(column)
    if expected_kernel == "softmax":
        # Width-sweep softmax metadata predates the kernel field; its degree identity is checked above.
        _optional_match(metadata, "attention_kernel", expected_kernel, f"{label} kernel")
    elif expected_kernel is not None:
        _match(metadata.get("attention_kernel"), expected_kernel, f"{label} kernel")
    if panel in ("unnormalized_learned", "normalized_learned_taylor"):
        _match(metadata.get("attention_coefficients"), "learned", f"{label} coefficients")


def _defaults_status(config: dict, column: dict, label: str) -> dict:
    """How a run relates to the current defaults: its resolved decay policy and QK-norm.

    The run's `config.yaml` states both; a recipe without a policy decays all
    parameters, and one without `qk_norm` uses plain scores. A column may
    declare `qk_norm: false` (a plain-score reference) or `qk_norm: null` (a mechanism
    QK-norm does not apply to); otherwise QK-norm is expected.
    """
    policy = config.get("optim", {}).get("weight_decay_policy")
    qk_norm = bool(config.get("model", {}).get("qk_norm", False))
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


def _collect_referenced(repo_root: Path, cell: dict, run: dict, step: int, tokens: int,
                        seed: int, data_identity: dict, column: dict | None = None) -> dict:
    experiment, name = cell["experiment"], cell["run"]
    width, degree = cell["d_k"], _expected_degree(cell["panel"], cell["column"])
    label = f"{experiment}/{name}"
    directory = repo_root / f"experiments/{experiment}/runs/{name}"
    _match(run.get("run_id"), name, f"{label} run ID")
    _match(run.get("seed"), seed, f"{label} seed")
    source = run.get("source_commit")
    if not isinstance(source, str) or not source:
        raise ValueError(f"Missing source commit for {label}")
    config = yaml.safe_load((directory / "config.yaml").read_text())
    model = config.get("model", {})
    _match(model.get("qk_head_dim"), width, f"{label} recipe d_K")
    _optional_match(model, "attention_degree", degree, f"{label} recipe")
    failures = sorted(directory.glob("numerical_failure_step_*.json"))
    if failures:
        status = "numerical_failure"
        if (directory / "completion.json").exists():
            raise ValueError(f"Numerical failure has a completion artifact: {label}")
    elif (directory / "completion.json").exists():
        status = "complete"
    else:
        status = "incomplete"

    result = {"status": status, "terminal_ppl": None, "terminal_nll": None,
              "failure_step": None, "experiment": experiment, "run": name,
              "source_commit": source, "artifacts": {},
              **_defaults_status(config, column or {}, label)}
    if status == "incomplete":
        metadata_path = repo_root / f"experiments/{experiment}/runs/{name}/metadata.json"
        if metadata_path.exists():
            result["artifacts"]["metadata"] = _artifact(repo_root, experiment, name, "metadata.json")
            metadata = _json(metadata_path)
            _match(metadata.get("seed"), seed, f"{label} metadata seed")
            _identity(metadata, source=source, width=width, degree=degree, label=label)
            _data_identity(metadata, data_identity, label)
            _mechanism(cell, model, metadata, label)
        return result

    result["artifacts"]["metadata"] = _artifact(repo_root, experiment, name, "metadata.json")
    metadata = _json(repo_root / result["artifacts"]["metadata"]["path"])
    _match(metadata.get("seed"), seed, f"{label} metadata seed")
    _identity(metadata, source=source, width=width, degree=degree, label=f"{label} metadata")
    _data_identity(metadata, data_identity, label)
    _mechanism(cell, model, metadata, label)
    _optional_match(metadata, "target_steps", step, f"{label} metadata")
    _optional_match(metadata, "target_tokens", tokens, f"{label} metadata")

    if status == "numerical_failure":
        _match(metadata.get("status"), "failed", f"{label} failure metadata status")
        result["artifacts"]["failure"] = _artifact(repo_root, experiment, name, failures[-1].name)
        failure = _json(repo_root / result["artifacts"]["failure"]["path"])
        failed_step = failure.get("step")
        if not isinstance(failed_step, int) or failed_step < 1 or failed_step >= step:
            raise ValueError(f"Invalid numerical failure step for {label}: {failed_step}")
        _match(metadata.get("step"), failed_step, f"{label} metadata failure step")
        _optional_match(failure, "attention_degree", degree, f"{label} failure")
        if not failure.get("error") or not metadata.get("error"):
            raise ValueError(f"Missing numerical failure diagnosis for {label}")
        result["failure_step"] = failed_step
        return result

    _match(metadata.get("status"), "complete", f"{label} metadata status")
    _match(metadata.get("step"), step, f"{label} metadata step")
    _match(metadata.get("training_tokens"), tokens, f"{label} metadata tokens")
    result["artifacts"]["metrics"] = _artifact(repo_root, experiment, name, "metrics.jsonl")
    result["artifacts"]["completion"] = _artifact(repo_root, experiment, name, "completion.json")
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
    result["terminal_nll"] = terminal_nll
    result["terminal_ppl"] = terminal_ppl
    return result


def collect_rows(repo_root: Path, config: dict) -> list[dict]:
    """Return the complete declared grid, with each cell's run checked against its pinned inputs."""
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

    cell_runs = {(cell["experiment"], cell["run"]) for cell in config["cells"]}
    pins = config.get("runs", [])
    if {(entry["study"], entry["run_id"]) for entry in pins} != cell_runs or len(pins) != len(cell_runs):
        raise ValueError("Pinned runs must be exactly the cells' runs; run `python -m scaffolds.records refresh`")
    verify_pinned_runs(repo_root, pins)

    for experiment in sorted({experiment for experiment, _ in cell_runs}):
        manifest = load_study(repo_root / "experiments" / experiment)
        _match(manifest.get("experiment"), experiment, f"Source study identity for {experiment}")

    columns = {(panel["id"], column["id"]): column for panel in config["panels"] for column in panel["columns"]}
    references = {}
    used_runs = set()
    for cell in config["cells"]:
        coordinate = (cell["panel"], cell["column"], cell["d_k"])
        if coordinate not in expected_set or coordinate in references:
            raise ValueError(f"Duplicate or out-of-grid cell reference: {coordinate}")
        experiment, name = cell["experiment"], cell["run"]
        # The pinned run, not the study's current run list: a rerun cohort replaces a
        # figure's run only once its own run at that coordinate completes.
        directory = repo_root / "experiments" / experiment / "runs" / name
        if not (directory / "record.json").is_file():
            raise ValueError(f"Unknown source run: {experiment}/{name}")
        if (experiment, name) in used_runs:
            raise ValueError(f"Source run is referenced twice: {experiment}/{name}")
        used_runs.add((experiment, name))
        references[coordinate] = (cell, load_run(directory))

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
    return rows
