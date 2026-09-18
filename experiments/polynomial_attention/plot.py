"""Summarize committed validation events and plot Taylor-degree endpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def kernel_label(degree: int | None) -> str:
    return "Exact softmax" if degree is None else f"m={degree}"


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
    completion_path = workdir / "completion.json"
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


def aggregate(manifest: dict, output_dir: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or Path(manifest["project_root"]) / "results/polynomial_attention"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_run(run, manifest["source_commit"]) for run in manifest["runs"]]
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
    lines = [
        "# Taylor attention sweep", "",
        f"Source: `{manifest['source_commit']}`. Identity: `{manifest['identity']}`.", "",
        f"{len(complete)}/{len(rows)} runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.", "",
        "The degree-zero control is causal prefix averaging; its Q/K projections receive no attention-derived learning signal.", "",
        "| Kernel | Parameters | Status | Terminal PPL |", "|---|---:|---|---:|",
    ]
    lines.extend(f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | {row['terminal_ppl']:.4f} |" if row["status"] == "complete"
                 else f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | — |" for row in rows)
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures:", "", *[f"- {row['kernel']}: {row['error']}" for row in failures]]
    lines.extend(["", "[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)", ""])
    (output_dir / "README.md").write_text("\n".join(lines))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/polynomial_attention/sweep.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
