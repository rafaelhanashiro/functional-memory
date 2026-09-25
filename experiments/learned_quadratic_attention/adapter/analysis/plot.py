"""Six-label comparison with coefficient, stability, and throughput trajectories."""
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

EXPERIMENT = "learned_quadratic_attention"


def label(run: dict) -> str:
    if run.get("retained_completed"):
        return "Fixed Taylor quadratic" if run["attention_degree"] == 2 else "Exact softmax"
    return f"{run['quadratic_mode'].capitalize()} · {'global' if run['quadratic_sharing'] == 'global' else 'per head/layer'}"


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
    if run.get("_retained_checked") and run["status"] == "numerical-failure":
        row.update(status="numerical-failure", error=run["failure"].get("error", row["error"]))
    return row, records


def aggregate(manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
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
        axis.set_title(label(run))
        axis.set_xlabel("Prediction tokens (billions)")
        axis.set_ylabel("Raw coefficient; band = head/layer range")
        axis.grid(alpha=0.2)
    save(figure, "coefficient_trajectories")

    # Final per-layer/head coefficients of completed per-head positive runs, from
    # their terminal metrics records. Both parameters are initialized at 1.
    from matplotlib.colors import TwoSlopeNorm
    per_head = [(run, records) for run, (row, records) in zip(manifest["runs"], summaries[len(manifest["controls"]):])
                if run["config"]["model"].get("quadratic_sharing") == "per_head" and run["config"]["model"].get("quadratic_mode") == "positive"
                and row["status"] == "complete" and records and "coefficients" in records[-1].get("attention_diagnostics", {})]
    if per_head:
        figure, axes = plt.subplots(len(per_head), 2, figsize=(13, 4.8 * len(per_head)), squeeze=False)
        for (run, records), line in zip(per_head, axes):
            raw = np.asarray(records[-1]["attention_diagnostics"]["coefficients"]["raw"])
            layers, heads = raw.shape[:2]
            panels = (("a: score multiplier (initialized at 1)", raw[..., 0], plt.colormaps["Blues"], None),
                      ("b: shape parameter (initialized at 1)", raw[..., 1], plt.colormaps["RdBu_r"],
                       TwoSlopeNorm(1.0, min(float(raw[..., 1].min()), 0.999), max(float(raw[..., 1].max()), 1.001))))
            for axis, (title, values, cmap, norm) in zip(line, panels):
                image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
                axis.set_xticks(range(heads), [str(head + 1) for head in range(heads)])
                axis.set_yticks(range(layers), [str(layer + 1) for layer in range(layers)])
                axis.set_xlabel("Head")
                axis.set_ylabel("Layer")
                axis.set_title(f"{label(run)} · {title}")
                figure.colorbar(image, ax=axis, shrink=0.85)
        figure.suptitle(f"Final learned coefficients at {records[-1]['training_tokens'] / 1e9:.0f}B tokens, seed 0")
        save(figure, "per_head_coefficients")

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
             "Four learned variants match the degree sweep's 200M-parameter, 10B-token seed-0 recipe. Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.", "",
             "| Kernel | Parameters | Status | Terminal NLL | Terminal PPL |", "|---|---:|---|---:|---:|"]
    for row in rows:
        metrics = f"{row['terminal_nll']:.6f} | {row['terminal_ppl']:.4f}" if row["status"] == "complete" else "— | —"
        lines.append(f"| {row['kernel']} | {row['parameter_count']:,} | {row['status']} | {metrics} |")
    lines += ["", "Positive: p(s)=0.5[1+(as+b)²], initialized at a=b=1. Unrestricted: p(s)=1+bs+cs², initialized at b=1,c=0.5. Coefficients are FP32 without weight decay; global sharing adds 2 parameters and per-head/per-layer sharing adds 320.", "",
              "Coefficient magnitudes must be interpreted together with score scales: Q/K rescaling can absorb some coefficient movement. The constant scale of a normalized polynomial is unidentifiable. Positive and unrestricted families use different optimization coordinates, so equal learning rates do not isolate positivity alone. One seed per variant does not estimate seed variation.", "",
              "The stability plots use the fixed first held-out diagnostic sequence. Negative-weight fractions are computed after signed normalization; minimum absolute denominators and cancellation ratios expose near-zero signed sums.", "",
              "Learned runs use four H100s on dogo. Control throughput reflects each control's recorded allocations, including any hardware migration; it is not a matched hardware benchmark. Hardware histories are preserved in the control provenance.", "",
              "The fixed controls are degree-sweep runs retained in the [dK × Taylor-degree record](../key_dimension_taylor/README.md); their source commits and artifact checksums are in the provenance manifest:", ""]
    lines += [f"- {label(run)}: `{run['config']['train']['source_commit']}`; degree-sweep identity `{run['original_sweep_identity']}`." for run in manifest["controls"]]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures:", "", *[f"- {row['kernel']}: {row['error']}" for row in failures]]
    lines += ["", "[Metrics](summary.csv) · [Terminal metrics](figures/terminal_metrics.pdf) · [Learning curves](figures/learning_curves.pdf) · [Coefficients](figures/coefficient_trajectories.pdf) · [Stability](figures/stability.pdf) · [Throughput](figures/throughput.pdf)", ""]
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest_path(Path("experiments/learned_quadratic_attention")))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
