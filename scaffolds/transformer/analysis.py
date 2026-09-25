"""Shared analysis helpers: committed metrics, retained artifacts, and figures.

Experiment adapters build their summaries and figures from `runs/` with these
helpers; every derived output must regenerate byte-identically.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


RETAINED_FORMAT = "functional-memory-retained-experiment"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def kernel_label(degree: int | None) -> str:
    return "Exact softmax" if degree is None else f"m={degree}"


def committed_records(workdir: Path, *, metrics_path: Path | None = None) -> list[dict]:
    metrics = metrics_path or workdir / "metrics.jsonl"
    if not metrics.exists():
        return []
    by_step = {}
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "validation":
            continue
        step = record["step"]
        if record.get("checkpoint_step") != step:
            raise ValueError(f"Validation event lacks a matching checkpoint: {metrics}:{step}")
        if step in by_step and by_step[step] != record:
            raise ValueError(f"Conflicting committed validation events: {metrics}:{step}")
        if not math.isfinite(record["validation_nll"]) or not math.isfinite(record["validation_perplexity"]):
            raise ValueError(f"Non-finite committed validation metrics: {metrics}:{step}")
        by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def summarize_run(
    run: dict, source_commit: str, *, metrics_path: Path | None = None,
    completion_path: Path | None = None,
) -> tuple[dict, list[dict]]:
    workdir = Path(run["workdir"])
    records = committed_records(workdir, metrics_path=metrics_path)
    for record in records:
        if (record.get("source_commit") != source_commit or record.get("qk_head_dim") != run["qk_head_dim"]
                or "attention_degree" not in record or record["attention_degree"] != run["attention_degree"]):
            raise ValueError(f"Metric identity does not match sweep: {workdir}")
        if record.get("parameter_count") != run["parameter_count"]:
            raise ValueError(f"Metric parameter count does not match sweep: {workdir}")
    metadata_path = workdir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    status = metadata.get("status", "incomplete" if records or (workdir / "config.json").exists() else "pending")
    if status == "complete":
        status = "incomplete"  # Completion is accepted only with its validated terminal record below.
    row = {
        "run": run["name"],
        "kernel": kernel_label(run["attention_degree"]),
        "attention_degree": run["attention_degree"],
        "qk_head_dim": run["qk_head_dim"],
        "value_head_dim": run["value_head_dim"],
        "parameter_count": run["parameter_count"],
        "source_commit": source_commit,
        "status": status,
        "error": metadata.get("error") or "",
        "training_tokens": metadata.get("training_tokens", ""),
        "terminal_nll": "",
        "terminal_ppl": "",
        "best_step": "",
        "best_ppl": "",
        "latest_tokens_per_second": "",
    }
    if records:
        best = min(records, key=lambda record: record["validation_nll"])
        row.update(best_step=best["step"], best_ppl=best["validation_perplexity"],
                   latest_tokens_per_second=records[-1].get("tokens_per_second", ""))
    completion_path = completion_path or workdir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        expected_step = run["config"]["optim"]["total_steps"]
        final = next((record for record in records if record["step"] == expected_step), None)
        if (completion.get("source_commit") != source_commit or "attention_degree" not in completion
                or completion["attention_degree"] != run["attention_degree"]
                or completion.get("parameter_count") != run["parameter_count"]):
            raise ValueError(f"Completion identity does not match sweep: {completion_path}")
        if completion.get("final_step") != expected_step or completion.get("training_tokens") != run["training_tokens"]:
            raise ValueError(f"Completion does not match the fixed training budget: {completion_path}")
        if final is None or final["training_tokens"] != run["training_tokens"]:
            raise ValueError(f"Completion lacks its committed terminal validation: {completion_path}")
        row.update(status="complete", error="", training_tokens=final["training_tokens"],
                   terminal_nll=final["validation_nll"], terminal_ppl=final["validation_perplexity"])
    return row, records


def retained_root(manifest: dict, manifest_path: Path | None) -> Path:
    """Repository root of a record at `experiments/<experiment>/provenance/manifest.json`."""
    if manifest_path is None:
        return REPOSITORY_ROOT
    for directory in manifest_path.resolve().parents:
        if directory.name == manifest["experiment"] and directory.parent.name == "experiments":
            return directory.parent.parent
    raise ValueError("A record's manifest must live under experiments/<experiment>/")


def default_manifest_path(record: Path) -> Path:
    return record / "provenance/manifest.json"


def update_results_readme(path: Path, lines: list[str]) -> None:
    """Replace only the generated block, preserving authored methods and usage."""
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    title = lines[0] if lines and lines[0].startswith("# ") else "# Results"
    body = "\n".join(lines[1:] if lines and lines[0].startswith("# ") else lines).strip()
    block = f"{begin}\n{body}\n{end}"
    text = path.read_text() if path.exists() else f"{title}\n"
    if begin in text or end in text:
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
            raise ValueError(f"Malformed generated results markers: {path}")
        start, stop = text.index(begin), text.index(end) + len(end)
        text = text[:start] + block + text[stop:]
    else:
        text += ("\n" if text.endswith("\n") else "\n\n") + block + "\n"
    path.write_text(text)


def artifact_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Retained artifact path must be repository-relative: {relative_path}")
    return root / path


def checked_artifact(root: Path, artifact: dict) -> Path:
    path = artifact_path(root, artifact["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError(f"Retained artifact checksum does not match: {path}")
    return path


def summarize_retained_run(run: dict, root: Path) -> tuple[dict, list[dict]]:
    import yaml

    config_path = checked_artifact(root, run["resolved_config"])
    metrics_path = checked_artifact(root, run["metrics"])
    config = yaml.safe_load(config_path.read_text())
    if (config["train"]["seed"] != run["seed"]
            or config["train"]["source_commit"] != run["source_commit"]
            or any(config["model"][field] != run[field]
                   for field in ("attention_degree", "qk_head_dim", "value_head_dim"))):
        raise ValueError(f"Resolved configuration identity does not match provenance: {config_path}")
    terminal = run["terminal"]
    expected_tokens = (config["optim"]["total_steps"] * config["data"]["batch_size"]
                       * (config["data"]["sequence_length"] - 1))
    if terminal["step"] != config["optim"]["total_steps"] or terminal["training_tokens"] != expected_tokens:
        raise ValueError(f"Retained terminal budget does not match configuration: {config_path}")
    # The retained artifact paths, not the recipe's workdir, locate the metrics;
    # regenerating figures needs neither datasets nor checkpoints.
    adapted = dict(run, config=config, workdir=str(metrics_path.parent), training_tokens=expected_tokens)
    row, records = summarize_run(
        adapted, run["source_commit"], metrics_path=metrics_path,
        completion_path=artifact_path(root, run["completion"]),
    )
    final = next((record for record in records if record["step"] == terminal["step"]), None)
    if row["status"] != "complete" or final is None or any(final[field] != terminal[field] for field in terminal):
        raise ValueError(f"Retained terminal metrics do not match committed validation: {metrics_path}")
    selected = run["selected_checkpoint"]
    artifact_path(root, selected["path"])
    best = next((record for record in records if record["step"] == selected["step"]), None)
    if (best is None or best["validation_nll"] != min(record["validation_nll"] for record in records)
            or any(best[field] != selected[field] for field in ("validation_nll", "validation_perplexity"))):
        raise ValueError(f"Retained checkpoint is not validation-best: {metrics_path}")
    row.update(best_step=selected["step"], best_ppl=selected["validation_perplexity"])
    return row, records


def retained_plot_manifest(manifest: dict, manifest_path: Path | None = None) -> dict:
    """Adapt a concluded attention study, including numerical failures, for plots."""
    import yaml

    root = retained_root(manifest, manifest_path)
    adapted = dict(manifest, project_root=str(root))
    for group in ("controls", "runs"):
        adapted[group] = []
        for run in manifest[group]:
            if run.get("source_provenance"):
                checked_artifact(root, run["source_provenance"])
            config = yaml.safe_load(checked_artifact(root, run["resolved_config"]).read_text())
            identity = ("attention_degree", "qk_head_dim", "value_head_dim")
            if (config["train"]["source_commit"] != run["source_commit"]
                    or config["train"]["seed"] != run["seed"]
                    or any(config["model"][key] != run[key] for key in identity)):
                raise ValueError(f"Retained configuration identity mismatch: {run['name']}")
            metadata_path = checked_artifact(root, run["metadata"])
            workdir = metadata_path.parent
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("source_commit") != run["source_commit"] or metadata.get("parameter_count") != run["parameter_count"]:
                raise ValueError(f"Retained metadata identity mismatch: {run['name']}")
            if run.get("metrics"):
                if checked_artifact(root, run["metrics"]) != workdir / "metrics.jsonl":
                    raise ValueError("Retained metrics must be in their run directory")
            elif (workdir / "metrics.jsonl").exists():
                raise ValueError(f"Unrecorded retained validation metrics: {run['name']}")
            if run.get("completion_artifact"):
                checked_artifact(root, run["completion_artifact"])
            if run["status"] == "complete":
                summarize_retained_run(run, root)
            else:
                if run.get("terminal") is not None or run.get("completion") is not None:
                    raise ValueError(f"Failed run cannot have a terminal endpoint: {run['name']}")
                failure = json.loads(checked_artifact(root, run["failure"]).read_text())
                if (metadata.get("status") != "failed" or metadata.get("step") != run["failure"]["step"]
                        or failure["step"] != run["failure"]["step"]
                        or failure.get("attention_degree") != run["attention_degree"]):
                    raise ValueError(f"Retained numerical failure mismatch: {run['name']}")
                if run.get("launcher_failure"):
                    checked_artifact(root, run["launcher_failure"])
                selected = run.get("selected_checkpoint")
                records = committed_records(workdir)
                if selected:
                    best = min(records, key=lambda record: (record["validation_nll"], -record["step"]))
                    if any(best[key] != selected[key] for key in ("step", "validation_nll", "validation_perplexity")):
                        raise ValueError(f"Retained failed-run checkpoint is not validation-best: {run['name']}")
                    artifact_path(root, selected["path"])
            adapted[group].append(dict(run, config=config, workdir=str(workdir), _retained_checked=True,
                training_tokens=config["optim"]["total_steps"] * config["data"]["batch_size"] * (config["data"]["sequence_length"] - 1)))
    return adapted


def save_figure(figure, stem: Path, *, dpi: int = 180, **options) -> list[Path]:
    """Write PNG and PDF; PDFs omit their creation timestamp so they regenerate byte-identically."""
    written = []
    for suffix in ("png", "pdf"):
        path = Path(f"{stem}.{suffix}")
        figure.savefig(path, dpi=dpi, metadata={"CreationDate": None} if suffix == "pdf" else None, **options)
        written.append(path)
    return written


GPU_TYPES = ("B300", "H200", "H100", "A100", "A6000")


def terminal_row(run_directory: Path) -> dict:
    """Terminal and validation-best metrics of one completed run, from its run files."""
    run_directory = Path(run_directory)
    completion = json.loads((run_directory / "completion.json").read_text())
    best = json.loads((run_directory / "best_validation.json").read_text())
    metadata = json.loads((run_directory / "metadata.json").read_text())
    kinds = {device["kind"] for device in metadata["execution_runtime"]["devices"]}
    gpu_type = next((name for name in GPU_TYPES if all(name in kind for kind in kinds)), "/".join(sorted(kinds)))
    return {
        "parameters": completion["parameter_count"], "updates": completion["final_step"],
        "prediction_tokens": completion["training_tokens"],
        "terminal_validation_nll": completion["validation_nll"],
        "terminal_validation_perplexity": completion["validation_perplexity"],
        "best_step": best["step"], "best_validation_nll": best["validation_nll"],
        "best_validation_perplexity": best["validation_perplexity"],
        "source_commit": completion["source_commit"], "slurm_job_id": completion["slurm_job_id"],
        "gpu_count": completion["data_parallel_devices"], "gpu_type": gpu_type,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import csv

    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])


def regeneration_differences(record: Path, output: Path) -> list[str]:
    """Files regenerated into `output` that differ from, or are missing in, the committed record.

    Adapters regenerate into a scratch directory (with a copy of the README when the
    generator updates its results block) and assert that this list is empty.
    """
    record, output = Path(record), Path(output)
    produced = sorted(path.relative_to(output) for path in output.rglob("*") if path.is_file())
    if not produced:
        return ["<nothing regenerated>"]
    return [str(relative) for relative in produced
            if not (record / relative).is_file() or (record / relative).read_bytes() != (output / relative).read_bytes()]
