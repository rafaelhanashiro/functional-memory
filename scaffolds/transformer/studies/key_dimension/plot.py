"""Summarize committed validation events and plot completed sweep endpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scaffolds.transformer.studies.polynomial_attention.plot import (
    RETAINED_FORMAT, artifact_path, checked_artifact, check_dependency_lock,
    default_manifest_path as retained_manifest_path, retained_root, update_results_readme,
)


def committed_records(workdir: Path) -> list[dict]:
    metrics = workdir / "metrics.jsonl"
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


def summarize_run(run: dict, source_commit: str) -> tuple[dict, list[dict]]:
    workdir = Path(run["workdir"])
    records = committed_records(workdir)
    for record in records:
        if record.get("source_commit") != source_commit or record.get("qk_head_dim") != run["qk_head_dim"]:
            raise ValueError(f"Metric identity does not match sweep: {workdir}")
        if record.get("parameter_count") != run["parameter_count"]:
            raise ValueError(f"Metric parameter count does not match sweep: {workdir}")
    row = {
        "run": run["name"],
        "qk_head_dim": run["qk_head_dim"],
        "value_head_dim": run["value_head_dim"],
        "parameter_count": run["parameter_count"],
        "status": "incomplete" if records or (workdir / "config.json").exists() else "pending",
        "training_tokens": "",
        "terminal_nll": "",
        "terminal_ppl": "",
        "best_step": "",
        "best_ppl": "",
    }
    if records:
        best = min(records, key=lambda record: record["validation_nll"])
        row.update(best_step=best["step"], best_ppl=best["validation_perplexity"])
    completion_path = workdir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        expected_step = run["config"]["optim"]["total_steps"]
        final = next((record for record in records if record["step"] == expected_step), None)
        if completion.get("final_step") != expected_step or completion.get("training_tokens") != run["training_tokens"]:
            raise ValueError(f"Completion does not match the fixed training budget: {completion_path}")
        if final is None or final["training_tokens"] != run["training_tokens"]:
            raise ValueError(f"Completion lacks its committed terminal validation: {completion_path}")
        row.update(status="complete", training_tokens=final["training_tokens"], terminal_nll=final["validation_nll"], terminal_ppl=final["validation_perplexity"])
    return row, records


def comparison_recipe(run: dict) -> dict:
    """Scientific settings shared by runs that vary only query/key width."""
    config = run["config"]
    return {
        "training_tokens": run["training_tokens"],
        "value_head_dim": run["value_head_dim"],
        "model": {key: value for key, value in config.get("model", {}).items() if key != "qk_head_dim"},
        "data": {key: value for key, value in config.get("data", {}).items() if key not in {"train_manifest", "eval_manifest"}},
        "optim": config["optim"],
        "train": {key: config.get("train", {}).get(key) for key in ("dtype", "seed", "micro_batch_size", "eval_batches")},
    }


def aggregation_sources(manifest: dict) -> list[tuple[dict, dict | None]]:
    """Read pinned source manifests without rewriting their contents or runs."""
    sources = [(manifest, None)]
    names, widths = set(), set()
    for source, _ in sources:
        for run in source["runs"]:
            if run["name"] in names or run["qk_head_dim"] in widths:
                raise ValueError(f"Duplicate run name or query/key dimension across manifests: {run['name']}")
            names.add(run["name"])
            widths.add(run["qk_head_dim"])
        for reference in source.get("reference_manifests", []):
            path = Path(reference["path"])
            if not path.is_absolute():
                raise ValueError(f"Reference manifest path must be absolute: {path}")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != reference["sha256"]:
                raise ValueError(f"Reference manifest SHA-256 mismatch: {path}")
            sources.append((json.loads(content), reference))
    if not names:
        raise ValueError("No runs to aggregate")
    if len(sources) > 1:
        recipe = next(comparison_recipe(run) for source, _ in sources for run in source["runs"])
        data_identity = {
            key: value["sha256"] for key, value in manifest.get("data_identity", {}).items()
        }
        for source, _ in sources:
            if {key: value["sha256"] for key, value in source.get("data_identity", {}).items()} != data_identity:
                raise ValueError("Referenced sweep data identity does not match current sweep")
            if any(comparison_recipe(run) != recipe for run in source["runs"]):
                raise ValueError("Referenced sweep recipe or training budget does not match current sweep")
    return sources


def default_manifest_path(results_dir: Path = Path("results/key_dimension")) -> Path:
    path = retained_manifest_path(results_dir)
    if path.exists() and path.name == "provenance.json":
        provenance = json.loads(path.read_text())
        if provenance.get("plot_manifest"):
            return results_dir / provenance["plot_manifest"]
    return path


def retained_summaries(manifest: dict, root: Path) -> list[tuple[dict, list[dict]]]:
    import yaml

    check_dependency_lock(manifest, root)
    summaries = []
    for run in sorted(manifest["runs"], key=lambda run: run["qk_head_dim"]):
        config_path = checked_artifact(root, run["resolved_config"])
        metrics = checked_artifact(root, run["metrics"])
        config = yaml.safe_load(config_path.read_text())
        if (config["train"]["seed"] != run["seed"]
                or config["train"]["source_commit"] != run["source_commit"]
                or any(config["model"][key] != run[key] for key in ("qk_head_dim", "value_head_dim"))):
            raise ValueError(f"Resolved configuration identity does not match provenance: {config_path}")
        terminal = run["terminal"]
        expected_tokens = config["optim"]["total_steps"] * config["data"]["batch_size"] * (config["data"]["sequence_length"] - 1)
        if terminal["step"] != config["optim"]["total_steps"] or terminal["training_tokens"] != expected_tokens:
            raise ValueError("Retained terminal budget does not match configuration")
        completion = artifact_path(root, run["completion"])
        if completion != metrics.parent / "completion.json" or metrics.name != "metrics.jsonl":
            raise ValueError("Retained metrics and completion must use canonical run filenames")
        adapted = dict(run, config=config, workdir=str(metrics.parent), training_tokens=expected_tokens)
        row, records = summarize_run(adapted, run["source_commit"])
        final = next((record for record in records if record["step"] == terminal["step"]), None)
        if row["status"] != "complete" or final is None or any(final[key] != value for key, value in terminal.items()):
            raise ValueError("Retained terminal metrics do not match committed validation")
        selected = run["selected_checkpoint"]
        best = next((record for record in records if record["step"] == selected["step"]), None)
        if (best is None or best["validation_nll"] != min(record["validation_nll"] for record in records)
                or any(best[key] != selected[key] for key in ("validation_nll", "validation_perplexity"))):
            raise ValueError("Retained checkpoint is not validation-best")
        row.update(best_step=selected["step"], best_ppl=selected["validation_perplexity"])
        summaries.append((row, records))
    return summaries


def aggregate(manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None) -> list[dict]:
    retained = manifest.get("format") == RETAINED_FORMAT
    if retained:
        root = retained_root(manifest, manifest_path)
        summaries = retained_summaries(manifest, root)
        sources = []
        output_dir = output_dir or root / "results" / manifest["experiment"]
    else:
        sources = aggregation_sources(manifest)
        runs = sorted(
            ((run, source["source_commit"]) for source, _ in sources for run in source["runs"]),
            key=lambda item: item[0]["qk_head_dim"],
        )
        summaries = [summarize_run(run, source_commit) for run, source_commit in runs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or Path(manifest["project_root"]) / "results/key_dimension"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    rows = [row for row, _ in summaries]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6, 4))
    complete = [row for row in rows if row["status"] == "complete"]
    if complete:
        axis.plot([row["qk_head_dim"] for row in complete], [row["terminal_ppl"] for row in complete], "o-")
    else:
        axis.text(0.5, 0.5, "No completed runs", transform=axis.transAxes, ha="center")
    axis.set_xscale("log", base=2)
    ticks = [row["qk_head_dim"] for row in rows]
    axis.set_xticks(ticks, [str(value) for value in ticks])
    axis.set_xlabel("Query/key dimension per head")
    axis.set_ylabel("Terminal validation perplexity")
    axis.set_title("200M-reference Transformer · fixed 10B-token budget")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(figures_dir / f"ppl_vs_dk.{extension}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for row, records in summaries:
        if records:
            axis.plot([record["training_tokens"] / 1e9 for record in records], [record["validation_perplexity"] for record in records], label=f"dK={row['qk_head_dim']}", linestyle="-" if row["status"] == "complete" else "--")
    axis.set_xlabel("Prediction tokens (billions)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Fixed held-out subset; dashed curves are incomplete runs")
    axis.grid(alpha=0.2)
    if any(records for _, records in summaries):
        axis.legend()
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(figures_dir / f"learningcurves.{extension}", dpi=180)
    plt.close(figure)
    lines = ["# Key-dimension sweep", ""]
    retained = retained or (output_dir / "provenance.json").exists()
    if retained:
        lines.extend([
            f"[Retention provenance]({'provenance/manifest.json' if manifest.get('format') == RETAINED_FORMAT else 'provenance.json'}) is the authority for resolved configuration and retained checkpoint locations. "
            "Historical source manifests below are archived launch records.", "",
        ])
    if not sources:
        lines.extend(["Source commits and seeds are recorded per run in provenance.", ""])
    elif len(sources) == 1:
        lines.extend([f"Source: `{manifest['source_commit']}`. Identity: `{manifest['identity']}`.", ""])
    else:
        lines.extend(["Archived launch manifests:" if retained else "Source manifests:", ""])
        for source, reference in sources:
            label = "Current sweep" if reference is None else f"Reference `{reference['path']}` (SHA-256 `{reference['sha256']}`)"
            lines.append(f"- {label}: source `{source['source_commit']}`; identity `{source['identity']}`.")
        lines.append("")
    lines.extend([
        f"{len(complete)}/{len(rows)} runs complete. Endpoints use committed terminal validation at the common token budget; pending and incomplete runs are excluded.", "",
        "| dK | Parameters | Status | Terminal PPL |", "|---:|---:|---|---:|",
    ])
    lines.extend(f"| {row['qk_head_dim']} | {row['parameter_count']:,} | {row['status']} | {row['terminal_ppl']:.4f} |" if row["status"] == "complete" else f"| {row['qk_head_dim']} | {row['parameter_count']:,} | {row['status']} | — |" for row in rows)
    lines.extend(["", "[Machine-readable results](summary.csv) · [Terminal PPL](figures/ppl_vs_dk.pdf) · [Learning curves](figures/learningcurves.pdf)", ""])
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Sweep manifest; defaults to the retained plot manifest, or results/key_dimension/sweep.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest is not None else default_manifest_path()
    aggregate(json.loads(manifest_path.read_text()), args.output_dir, manifest_path=manifest_path)


if __name__ == "__main__":
    main()
