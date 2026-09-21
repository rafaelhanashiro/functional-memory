"""Summarize committed validation events and plot Taylor-degree endpoints."""
from __future__ import annotations

import argparse
import csv
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
    """Locate a portable results/<experiment>/ manifest without its old worktree."""
    if manifest_path is None:
        return REPOSITORY_ROOT
    result_dir = manifest_path.resolve().parent
    if result_dir.name != manifest["experiment"] or result_dir.parent.name != "results":
        raise ValueError("Retained manifest must live under results/<experiment>/")
    return result_dir.parent.parent


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
    # The original config's workdir is historical. The retained artifact paths
    # are authoritative, and regenerating figures needs neither datasets nor checkpoints.
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


def aggregate(
    manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None,
) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    retained = manifest.get("format") == RETAINED_FORMAT
    if retained:
        root = retained_root(manifest, manifest_path)
        lock_hash = manifest.get("dependency_lock_sha256")
        if lock_hash:
            checked_artifact(root, {"path": f"experiments/{manifest['experiment']}/uv.lock", "sha256": lock_hash})
        summaries = [summarize_retained_run(run, root) for run in manifest["runs"]]
        output_dir = output_dir or root / "results" / manifest["experiment"]
    else:
        summaries = [
            summarize_run(run, run["config"].get("train", {}).get("source_commit", manifest["source_commit"]))
            for run in manifest["runs"]
        ]
        output_dir = output_dir or Path(manifest["project_root"]) / "results/polynomial_attention"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row, _ in summaries]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6, 4))
    complete = [row for row in rows if row["status"] == "complete"]
    polynomial = sorted((row for row in complete if row["attention_degree"] is not None), key=lambda row: row["attention_degree"])
    softmax = next((row for row in complete if row["attention_degree"] is None), None)
    if polynomial:
        axis.plot([row["attention_degree"] for row in polynomial], [row["terminal_ppl"] for row in polynomial], "o-", label="Taylor attention")
    if softmax is not None:
        axis.axhline(softmax["terminal_ppl"], color="black", linestyle="--", label="Exact softmax")
    if complete:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No completed runs", transform=axis.transAxes, ha="center")
    ticks = [row["attention_degree"] for row in rows if row["attention_degree"] is not None]
    axis.set_xticks(ticks, [str(value) for value in ticks])
    axis.set_xlabel("Taylor degree m (terms 0 through m)")
    axis.set_ylabel("Terminal validation perplexity")
    axis.set_title("200M Transformer · fixed 10B-token budget")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"ppl_vs_degree.{extension}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for row, records in summaries:
        if records:
            axis.plot([record["training_tokens"] / 1e9 for record in records],
                      [record["validation_perplexity"] for record in records], label=row["kernel"],
                      linestyle="-" if row["status"] == "complete" else "--")
    axis.set_xlabel("Prediction tokens (billions)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Fixed held-out subset; dashed curves have not completed")
    axis.grid(alpha=0.2)
    if any(records for _, records in summaries):
        axis.legend()
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"learningcurves.{extension}", dpi=180)
    plt.close(figure)
    provenance_description = (
        "[Provenance](provenance.json) is the retained record of per-run source commits, seeds, "
        "resolved configurations, data identity, and selected checkpoints."
        if retained else f"Source: `{manifest['source_commit']}`. Identity: `{manifest['identity']}`."
    )
    lines = [
        "# Taylor attention sweep", "",
        provenance_description, "",
        f"{len(complete)}/{len(rows)} runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.", "",
        "The degree-zero control is causal prefix averaging; its Q/K projections receive no attention-derived learning signal.", "",
        "| Kernel | Parameters | Status | Terminal PPL |", "|---|---:|---|---:|",
    ]
    lines.extend(f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | {row['terminal_ppl']:.4f} |" if row["status"] == "complete"
                 else f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | — |" for row in rows)
    if retained:
        lines += [
            "", "Only the validation-best checkpoint is retained for each run under "
            "`experiments/polynomial_attention/checkpoints/<run>/` (on shared storage, ignored by Git). "
            "Terminal metrics above remain the scientific endpoint even when the best checkpoint is earlier.",
            "", "| Run | Seed | Source commit | Best step | Best PPL |",
            "|---|---:|---|---:|---:|",
            *[f"| {run['name']} | {run['seed']} | `{run['source_commit']}` | "
              f"{row['best_step']} | {row['best_ppl']:.4f} |" for run, row in zip(manifest["runs"], rows)],
            "", "Resolved configurations and reproduction instructions are in the "
            "[experiment scaffold](../../experiments/polynomial_attention/README.md). "
            "[Performance measurements](performance/README.md) document the execution backends and hardware.",
        ]
    earlier = [row for row in rows if row["source_commit"] != manifest.get("source_commit")]
    if earlier and not retained:
        lines += ["", "Retained results from earlier source revisions:", "", *[
            f"- {row['kernel']}: `{row['source_commit']}`." for row in earlier
        ]]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures:", "", *[f"- {row['kernel']}: {row['error']}" for row in failures]]
    lines.extend(["", "[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)", ""])
    (output_dir / "README.md").write_text("\n".join(lines))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    result_dir = REPOSITORY_ROOT / "results/polynomial_attention"
    default_manifest = result_dir / "provenance.json"
    if not default_manifest.exists():
        default_manifest = result_dir / "sweep.json"
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
