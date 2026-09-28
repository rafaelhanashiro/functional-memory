"""Fixed odd Taylor comparison with stability and throughput trajectories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from functools import partial

from scaffolds.transformer.analysis import (
    committed_records, selected_best, selected_run_directory, selected_views, summarize_run as base_summary,
    update_results_readme,
)

locate_run = partial(selected_run_directory, Path(__file__).resolve().parent)

EXPERIMENT = "odd_taylor_attention"


def label(run: dict) -> str:
    return "Exact softmax" if run["attention_degree"] is None else f"Fixed Taylor m={run['attention_degree']}"

def summarize_run(run: dict) -> tuple[dict, list[dict]]:
    row, records = base_summary(run, run["source_commit"])
    row.update(kernel=label(run), quadratic_mode=run.get("quadratic_mode", "fixed"),
               quadratic_sharing=run.get("quadratic_sharing", "none"),
               coefficient_count=run.get("coefficient_count", 0),
               negative_weight_fraction="", min_abs_normalizer="", cancellation_ratio_max="",
               score_min="", score_max="")
    best = selected_best(run, records)
    if best is not None:
        row.update(best_step=best["step"], best_ppl=best["validation_perplexity"])
    if not run.get("control"):
        completion_path = Path(run["workdir"]) / "completion.json"
        identities = records + ([json.loads(completion_path.read_text())] if completion_path.exists() else [])
        for record in identities:
            if any(record.get(key) != run[key] for key in ("attention_degree",)):
                raise ValueError(f"Metric identity does not match the odd_taylor_attention runs: {run['name']}")
    row["failure_step"] = run["failure"]["step"] if run["failure"] is not None and row["status"] != "complete" else ""
    if records:
        layers = records[-1].get("attention_diagnostics", {}).get("layers", [])
        for key, reduction in (("negative_weight_fraction", max), ("min_abs_normalizer", min),
                               ("cancellation_ratio_max", max), ("score_min", min), ("score_max", max)):
            values = [layer[key] for layer in layers if key in layer]
            if values:
                row[key] = reduction(values)
    if run["status"] == "numerical-failure":
        row.update(status="numerical-failure", error=run["error"])
    return row, records


def aggregate(manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir
    figures_dir.mkdir(exist_ok=True)
    runs = [*manifest["controls"], *manifest["runs"]]
    summaries = [summarize_run(run) for run in runs]
    rows = [row for row, _ in summaries]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def save(figure, name):
        figure.tight_layout()
        for extension in ("pdf", "png"):
            # PDFs omit their creation timestamp so regenerated figures are byte-identical.
            figure.savefig(figures_dir / f"{name}.{extension}", dpi=180,
                           metadata={"CreationDate": None} if extension == "pdf" else None)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, ylabel in zip(axes, ("terminal_ppl", "terminal_nll"), ("Terminal perplexity", "Terminal NLL")):
        indices = [index for index, row in enumerate(rows) if row["status"] == "complete"]
        axis.bar(indices, [rows[index][metric] for index in indices], color=[f"C{index}" for index in indices])
        axis.set_xticks(range(len(rows)), [row["kernel"] for row in rows], rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.set_title("Completed runs only · seed 0")
    save(figure, "terminal_metrics")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, (row, records) in enumerate(summaries):
        if not records:
            continue
        for axis, metric, ylabel in zip(axes, ("validation_perplexity", "validation_nll"), ("Validation perplexity", "Validation NLL")):
            axis.plot([record["training_tokens"] / 1e9 for record in records], [record[metric] for record in records],
                      label=row["kernel"], color=f"C{index}", linestyle="-" if row["status"] == "complete" else "--")
            axis.set_xlabel("Prediction tokens (billions)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
    if any(records for _, records in summaries):
        axes[0].legend(fontsize=8)
    save(figure, "learning_curves")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    diagnostics = (("negative_weight_fraction", max, "Max negative-weight fraction"),
                   ("min_abs_normalizer", min, "Min absolute denominator"),
                   ("cancellation_ratio_max", max, "Max cancellation ratio"))
    for index, (row, records) in enumerate(summaries):
        for axis, (key, reduction, ylabel) in zip(list(axes.flat)[:3], diagnostics):
            events = [(record, [layer[key] for layer in record.get("attention_diagnostics", {}).get("layers", []) if key in layer]) for record in records]
            events = [(record, values) for record, values in events if values]
            if events:
                axis.plot([record["training_tokens"] / 1e9 for record, _ in events],
                          [reduction(values) for _, values in events], label=row["kernel"], color=f"C{index}")
            axis.set_ylabel(ylabel)
        score_events = [(record, record.get("attention_diagnostics", {}).get("layers", [])) for record in records]
        score_events = [(record, layers) for record, layers in score_events if layers]
        if score_events:
            x = [record["training_tokens"] / 1e9 for record, _ in score_events]
            axes[1, 1].plot(x, [max(layer["score_max"] for layer in layers) for _, layers in score_events], label=row["kernel"], color=f"C{index}")
            axes[1, 1].plot(x, [min(layer["score_min"] for layer in layers) for _, layers in score_events], color=f"C{index}", linestyle=":")
    axes[1, 1].set_ylabel("Score extrema (dotted = minimum)")
    for axis in axes.flat:
        axis.set_xlabel("Prediction tokens (billions)")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    save(figure, "stability")

    figure, axis = plt.subplots(figsize=(8, 4))
    for index, (row, records) in enumerate(summaries):
        events = [record for record in records if record.get("tokens_per_second") is not None]
        if events:
            axis.plot([record["training_tokens"] / 1e9 for record in events], [record["tokens_per_second"] for record in events], label=row["kernel"], color=f"C{index}")
    axis.set_xlabel("Prediction tokens (billions)")
    axis.set_ylabel("Training prediction tokens / second")
    axis.grid(alpha=0.2)
    if axis.lines:
        axis.legend(fontsize=8)
    save(figure, "throughput")

    lines = ["# Odd Taylor attention", "", f"Source: `{manifest['source_commit']}`. Frozen execution set: `{manifest['identity']}`.", "",
             "Fixed degree-one and degree-three Taylor models use the dK × Taylor-degree record's 200M-parameter, 10B-token, seed-0 recipe at d_K = 64 under the scaffold's current defaults (QK-norm and selective weight decay). Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.", "",
             "| Kernel | Parameters | Status | Terminal NLL | Terminal PPL | Failure step |", "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        metrics = f"{row['terminal_nll']:.6f} | {row['terminal_ppl']:.4f}" if row["status"] == "complete" else "— | —"
        lines.append(f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | {metrics} | {row['failure_step']} |")
    lines += ["", "Fixed coefficients: p₁(s)=1+s; p₃(s)=1+s+s²/2+s³/6. Scores are QK-norm cosine scores s = exp(γ)·cos(q, k), with γ learned per layer and head from log 8, so |s| ≤ exp(γ); weights use the signed row sum. Negative weights and negative nonzero denominators are allowed; zero/nonfinite denominators and nonfinite computations fail without epsilon, clipping, or absolute-value normalization.", "",
              "Each run follows the full 19,074-update schedule from initialization under the shared launcher, which checkpoints every 500 updates and requeues after preemption. One seed per variant does not estimate seed variation.", "",
              "The stability plots use the fixed first held-out diagnostic sequence. Both odd-degree runs failed before their first validation at update 1,000, so only the controls have validation and diagnostic curves.", "",
              "Both odd-degree runs failed on four B300s on akita, after earlier attempts were preempted during acceptance. Control throughput reflects its recorded allocations and is not a matched hardware benchmark. Hardware histories are in the run records.", "",
              "The controls are retained in the [dK × Taylor-degree record](../key_dimension_taylor/README.md); their source commits and artifact checksums are in their run records:", ""]
    lines += [f"- {label(run)}: `{run['source_commit']}`; {run.get('control_note', 'control')}." for run in manifest["controls"]]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures:", "", *[f"- {row['kernel']}: {row['error']}" for row in failures]]
    lines += ["", "[Metrics](summary.csv) · [Terminal metrics](terminal_metrics.pdf) · [Learning curves](learning_curves.pdf) · [Stability](stability.pdf) · [Throughput](throughput.pdf)", ""]
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/odd_taylor_attention/sweep.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(selected_views(Path(__file__).resolve().parent, args.manifest), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
