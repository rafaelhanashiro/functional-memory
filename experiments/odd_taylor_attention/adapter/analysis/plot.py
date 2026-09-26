"""Fixed odd Taylor comparison with stability and throughput trajectories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scaffolds.transformer.analysis import (
    RETAINED_FORMAT, artifact_path, checked_artifact, committed_records, default_manifest_path,
    retained_plot_manifest, retained_root, summarize_retained_run, summarize_run as base_summary,
    update_results_readme,
)

EXPERIMENT = "odd_taylor_attention"


def label(run: dict) -> str:
    return "Exact softmax" if run["attention_degree"] is None else f"Fixed Taylor m={run['attention_degree']}"


def summarize_run(run: dict, source_commit: str) -> tuple[dict, list[dict]]:
    if run.get("retained_completed") and not run.get("_retained_checked"):
        raise ValueError(f"Control {run['name']} must be verified by retained_plot_manifest")
    row, records = base_summary(run, source_commit)
    row.update(kernel=label(run), quadratic_mode=run.get("quadratic_mode", "fixed"),
               quadratic_sharing=run.get("quadratic_sharing", "none"),
               coefficient_count=run.get("coefficient_count", 0),
               negative_weight_fraction="", min_abs_normalizer="", cancellation_ratio_max="",
               score_min="", score_max="")
    if not run.get("retained_completed"):
        if records:
            # Match odd-degree training and retention on exact ties.
            best = min(records, key=lambda record: (record["validation_nll"], -record["step"]))
            row.update(best_step=best["step"], best_ppl=best["validation_perplexity"])
        completion_path = Path(run["workdir"]) / "completion.json"
        identities = records + ([json.loads(completion_path.read_text())] if completion_path.exists() else [])
        for record in identities:
            if any(record.get(key) != run[key] for key in ("attention_degree",)):
                raise ValueError(f"Metric degree does not match odd Taylor sweep: {run['name']}")
        failure_path = Path(run["workdir"]) / "preflight_failure.json"
        if not records and failure_path.exists():
            failure = json.loads(failure_path.read_text())
            row.update(status="preflight-ineligible", error=failure["error"])
    failure = Path(run["workdir"]) / "production_failure.json"
    if failure.exists():
        record = json.loads(failure.read_text())
        row.update(status="numerical-failure" if record.get("numerical_failures") else "failed",
                   error=record["error"])
    row["failure_step"] = ""
    failures = [json.loads(path.read_text())
                for path in Path(run["workdir"]).glob("numerical_failure_step_*.json")]
    preflight_failure = Path(run["workdir"]) / "preflight_failure.json"
    if preflight_failure.exists():
        failures.extend(detail["record"] for detail in
                        json.loads(preflight_failure.read_text()).get("numerical_failures", []))
    if failures and row.get("status") != "complete":
        row["failure_step"] = max((failure.get("step", 0) for failure in failures), default="")
    if records:
        layers = records[-1].get("attention_diagnostics", {}).get("layers", [])
        for key, reduction in (("negative_weight_fraction", max), ("min_abs_normalizer", min),
                               ("cancellation_ratio_max", max), ("score_min", min), ("score_max", max)):
            values = [layer[key] for layer in layers if key in layer]
            if values:
                row[key] = reduction(values)
    if run.get("_retained_checked") and run["status"] == "numerical-failure":
        row.update(status="numerical-failure", error=run["failure"].get("error", row["error"]))
    return row, records


def aggregate(manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if manifest.get("format") == RETAINED_FORMAT:
        manifest = retained_plot_manifest(manifest, manifest_path)
    output_dir = output_dir or Path(manifest["project_root"]) / "experiments" / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
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

    lines = ["# Odd Taylor attention", "", f"Source: `{manifest['source_commit']}`. Sweep identity: `{manifest['identity']}`.", "",
             "Fixed degree-one and degree-three Taylor models match the degree sweep's 200M-parameter, 10B-token seed-0 recipe. Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.", "",
             "| Kernel | Parameters | Status | Terminal NLL | Terminal PPL | Failure step |", "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        metrics = f"{row['terminal_nll']:.6f} | {row['terminal_ppl']:.4f}" if row["status"] == "complete" else "— | —"
        lines.append(f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | {metrics} | {row['failure_step']} |")
    lines += ["", "Fixed coefficients: p₁(s)=1+s; p₃(s)=1+s+s²/2+s³/6. Scores are unshifted after RoPE and weights use the signed row sum. Negative weights and negative nonzero denominators are allowed; zero/nonfinite denominators and nonfinite computations fail without epsilon, clipping, or absolute-value normalization.", "",
              "Each model uses the full 19,074-update learning-rate schedule throughout a 5,000-update stability stage and subsequent continuation. Continuation requires finite committed validation and exact checkpoint, next-batch, and RNG restoration. One seed per variant does not estimate seed variation.", "",
              "The stability plots use the fixed first held-out diagnostic sequence. Negative-weight fractions are computed after signed normalization; minimum absolute denominators and cancellation ratios expose near-zero signed sums.", "",
              "Odd-degree runs use four H100s on dogo. Control throughput reflects its recorded allocations, including hardware migration, and is not a matched hardware benchmark. Hardware histories are preserved in the control provenance.", "",
              "The fixed controls are degree-sweep runs retained in the [dK × Taylor-degree record](../key_dimension_taylor/README.md); their source commits and artifact checksums are in the provenance manifest:", ""]
    lines += [f"- {label(run)}: `{run['config']['train']['source_commit']}`; "
              + (f"degree-sweep identity `{run['original_sweep_identity']}`." if run.get("original_sweep_identity")
                 else f"{run.get('control_note', 'retained control')}.") for run in manifest["controls"]]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures:", "", *[f"- {row['kernel']}: {row['error']}" for row in failures]]
    lines += ["", "[Metrics](summary.csv) · [Terminal metrics](figures/terminal_metrics.pdf) · [Learning curves](figures/learning_curves.pdf) · [Stability](figures/stability.pdf) · [Throughput](figures/throughput.pdf)", ""]
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest_path(Path("experiments/odd_taylor_attention")))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
