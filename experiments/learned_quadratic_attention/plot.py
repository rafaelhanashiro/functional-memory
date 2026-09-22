"""Six-label comparison with coefficient, stability, and throughput trajectories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.polynomial_attention.plot import summarize_run as historical_summary
from experiments.learned_quadratic_attention.sweep import EXPERIMENT, validate_control


def label(run: dict) -> str:
    if run.get("retained_completed"):
        return "Fixed Taylor quadratic" if run["attention_degree"] == 2 else "Exact softmax"
    return f"{run['quadratic_mode'].capitalize()} · {'global' if run['quadratic_sharing'] == 'global' else 'per head/layer'}"


def summarize_run(run: dict, source_commit: str) -> tuple[dict, list[dict]]:
    if run.get("retained_completed"):
        validate_control(run)
    row, records = historical_summary(run, source_commit)
    row.update(kernel=label(run), quadratic_mode=run.get("quadratic_mode", "fixed"),
               quadratic_sharing=run.get("quadratic_sharing", "none"),
               coefficient_count=run.get("coefficient_count", 0),
               negative_weight_fraction="", min_abs_normalizer="", cancellation_ratio_max="",
               score_min="", score_max="")
    if not run.get("retained_completed"):
        if records:
            # Match the learned training and retention policy on exact ties.
            best = min(records, key=lambda record: (record["validation_nll"], -record["step"]))
            row.update(best_step=best["step"], best_ppl=best["validation_perplexity"])
        completion_path = Path(run["workdir"]) / "completion.json"
        identities = records + ([json.loads(completion_path.read_text())] if completion_path.exists() else [])
        for record in identities:
            if any(record.get(key) != run[key] for key in ("quadratic_mode", "quadratic_sharing")):
                raise ValueError(f"Metric family/sharing does not match learned sweep: {run['name']}")
        failure_path = Path(run["workdir"]) / "preflight_failure.json"
        if not records and failure_path.exists():
            failure = json.loads(failure_path.read_text())
            row.update(status="preflight-ineligible", error=failure["error"])
    if records:
        layers = records[-1].get("attention_diagnostics", {}).get("layers", [])
        for key, reduction in (("negative_weight_fraction", max), ("min_abs_normalizer", min),
                               ("cancellation_ratio_max", max), ("score_min", min), ("score_max", max)):
            values = [layer[key] for layer in layers if key in layer]
            if values:
                row[key] = reduction(values)
    return row, records


def aggregate(manifest: dict, output_dir: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = output_dir or Path(manifest["project_root"]) / "results" / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [*manifest["controls"], *manifest["runs"]]
    summaries = [summarize_run(run, run["config"]["train"]["source_commit"]) for run in runs]
    rows = [row for row, _ in summaries]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def save(figure, name):
        figure.tight_layout()
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"{name}.{extension}", dpi=180)
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

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    coefficients = []
    for axis, run, (_, records) in zip(axes.flat, manifest["runs"], summaries[len(manifest["controls"]):], strict=True):
        events = [record for record in records if "coefficients" in record.get("attention_diagnostics", {})]
        if events:
            x = [record["training_tokens"] / 1e9 for record in events]
            raw = np.asarray([record["attention_diagnostics"]["coefficients"]["raw"] for record in events]).reshape(len(events), -1, 2)
            names = events[-1]["attention_diagnostics"]["coefficients"]["raw_parameter_names"]
            for index, name in enumerate(names):
                axis.plot(x, raw[:, :, index].mean(axis=1), label=f"{name} mean")
                axis.fill_between(x, raw[:, :, index].min(axis=1), raw[:, :, index].max(axis=1), alpha=0.15)
            axis.legend()
            coefficients.extend({"run": run["name"], "step": record["step"], "training_tokens": record["training_tokens"],
                                 **record["attention_diagnostics"]["coefficients"]} for record in events)
        axis.set_title(label(run))
        axis.set_xlabel("Prediction tokens (billions)")
        axis.set_ylabel("Raw coefficient; band = head/layer range")
        axis.grid(alpha=0.2)
    save(figure, "coefficient_trajectories")
    (output_dir / "coefficients.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in coefficients))

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

    lines = ["# Learned quadratic attention", "", f"Source: `{manifest['source_commit']}`. Sweep identity: `{manifest['identity']}`.", "",
             "Four learned variants match the historical 200M-parameter, 10B-token seed-0 recipe. Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.", "",
             "| Kernel | Parameters | Status | Terminal NLL | Terminal PPL |", "|---|---:|---|---:|---:|"]
    for row in rows:
        metrics = f"{row['terminal_nll']:.6f} | {row['terminal_ppl']:.4f}" if row["status"] == "complete" else "— | —"
        lines.append(f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | {metrics} |")
    lines += ["", "Positive: p(s)=0.5[1+(as+b)²], initialized at a=b=1. Unrestricted: p(s)=1+bs+cs², initialized at b=1,c=0.5. Coefficients are FP32 without weight decay; global sharing adds 2 parameters and per-head/per-layer sharing adds 320.", "",
              "Coefficient magnitudes must be interpreted together with score scales: Q/K rescaling can absorb some coefficient movement. The constant scale of a normalized polynomial is unidentifiable. Positive and unrestricted families use different optimization coordinates, so equal learning rates do not isolate positivity alone. One seed per variant does not estimate seed variation.", "",
              "The stability plots use the fixed first held-out diagnostic sequence. Negative-weight fractions are computed after signed normalization; minimum absolute denominators and cancellation ratios expose near-zero signed sums.", "",
              "Learned runs use four H100s on dogo. Historical control throughput reflects each control's recorded allocations, including any hardware migration; it is not a matched hardware benchmark. Hardware histories are preserved in the control provenance.", "",
              "Historical controls retain original source revisions and artifact SHA256 checksums in `sweep.json`:", ""]
    lines += [f"- {label(run)}: `{run['config']['train']['source_commit']}`; original sweep `{run['original_sweep_identity']}`." for run in manifest["controls"]]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures:", "", *[f"- {row['kernel']}: {row['error']}" for row in failures]]
    lines += ["", "[Metrics](summary.csv) · [Terminal metrics](terminal_metrics.pdf) · [Learning curves](learning_curves.pdf) · [Coefficients](coefficient_trajectories.pdf) · [Stability](stability.pdf) · [Throughput](throughput.pdf)", ""]
    (output_dir / "README.md").write_text("\n".join(lines))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/learned_quadratic_attention/sweep.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
