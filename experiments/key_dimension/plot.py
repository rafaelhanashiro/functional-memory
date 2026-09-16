"""Summarize committed validation events and plot completed sweep endpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


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


def aggregate(manifest: dict, output_dir: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or Path(manifest["project_root"]) / "results/key_dimension"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_run(run, manifest["source_commit"]) for run in manifest["runs"]]
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
        figure.savefig(output_dir / f"ppl_vs_dk.{extension}", dpi=180)
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
        figure.savefig(output_dir / f"learningcurves.{extension}", dpi=180)
    plt.close(figure)
    lines = [
        "# Key-dimension sweep", "",
        f"Source: `{manifest['source_commit']}`. Identity: `{manifest['identity']}`.", "",
        f"{len(complete)}/{len(rows)} runs complete. Endpoints use committed terminal validation at the common token budget; pending and incomplete runs are excluded.", "",
        "| dK | Parameters | Status | Terminal PPL |", "|---:|---:|---|---:|",
    ]
    lines.extend(f"| {row['qk_head_dim']} | {row['parameter_count']:,} | {row['status']} | {row['terminal_ppl']:.4f} |" if row["status"] == "complete" else f"| {row['qk_head_dim']} | {row['parameter_count']:,} | {row['status']} | — |" for row in rows)
    lines.extend(["", "[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_dk.pdf) · [Learning curves](learningcurves.pdf)", ""])
    (output_dir / "README.md").write_text("\n".join(lines))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/key_dimension/sweep.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
