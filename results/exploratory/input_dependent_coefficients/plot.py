"""Regenerate the explicitly selected input-dependent coefficient comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
FIELDS = (
    "run_id", "role", "degree", "coefficient_function", "status", "terminal_step",
    "training_tokens", "terminal_validation_nll", "terminal_validation_perplexity",
    "best_validation_step", "best_validation_perplexity", "terminal_train_nll_sample_mean",
    "control_run_id", "perplexity_percent_above_control", "valid_diagnostics",
    "diagnostic_count", "mean_coefficient_std", "saturated_mean_branch_fraction",
    "mean_tanh_derivative_from_moments", "metrics_sha256", "completion_sha256",
)


def load_runs():
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    rows, curves = [], {}
    for entry in selection["runs"]:
        path = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((path / "config.yaml").read_text())
        model = config["model"]
        if model["initialization"] != "transformer_reference":
            raise ValueError(f"Non-reference initialization: {entry['run_id']}")
        if not model["qk_norm"] or model["attention_normalization"]:
            raise ValueError(f"Unexpected QK or attention normalization: {entry['run_id']}")
        if model["attention_output_scaling"] != "head_rms_norm" or config["optim"]["warmup_steps"] != 400:
            raise ValueError(f"Unexpected output normalization or warmup: {entry['run_id']}")
        metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        validation = [row for row in metrics if row.get("event") == "validation"]
        curves[entry["run_id"]] = validation
        completion = json.loads((path / "completion.json").read_text())
        best = json.loads((path / "best_validation.json").read_text())
        if completion["final_step"] != config["optim"]["total_steps"]:
            raise ValueError(f"Selected run is incomplete: {entry['run_id']}")
        if list(path.glob("numerical_failure_step_*.json")):
            raise ValueError(f"Selected run has a numerical failure: {entry['run_id']}")
        function = (model.get("attention_coefficient_activation", "identity")
                    if model.get("attention_coefficient_conditioning", "static") == "query" else "static")
        function = "affine" if function == "identity" else function
        diagnostics = [row["attention_diagnostics"] for row in validation]
        row = dict(
            run_id=entry["run_id"], role=entry["role"], degree=model["attention_degree"],
            coefficient_function=function, status="complete", terminal_step=completion["final_step"],
            training_tokens=completion["training_tokens"],
            terminal_validation_nll=completion["validation_nll"],
            terminal_validation_perplexity=completion["validation_perplexity"],
            best_validation_step=best["step"], best_validation_perplexity=best["validation_perplexity"],
            terminal_train_nll_sample_mean=validation[-1]["train_nll_sample_mean"],
            control_run_id=entry.get("control", ""),
            valid_diagnostics=sum(bool(diag["valid"]) for diag in diagnostics),
            diagnostic_count=len(diagnostics),
            metrics_sha256=entry["sources"]["metrics.jsonl"]["sha256"],
            completion_sha256=entry["sources"]["completion.json"]["sha256"],
        )
        coefficients = diagnostics[-1]["coefficients"]
        if function != "static":
            summary = coefficients["effective_summary"]
            row["mean_coefficient_std"] = float(np.mean(summary["std"]))
        if function == "tanh":
            amplitude = np.asarray(coefficients["modulation_amplitude"])
            bias = np.asarray(coefficients["projection_bias"])
            mean = (np.asarray(summary["mean"]) - bias) / amplitude
            std = np.asarray(summary["std"]) / np.abs(amplitude)
            # Each branch is one layer/head/order; these moments cover one pinned sequence.
            row["saturated_mean_branch_fraction"] = float(np.mean(np.abs(mean) > .99))
            row["mean_tanh_derivative_from_moments"] = float(np.mean(np.clip(1 - mean**2 - std**2, 0, 1)))
        rows.append(row)
    by_id = {row["run_id"]: row for row in rows}
    for row in rows:
        if row["control_run_id"]:
            control = by_id[row["control_run_id"]]
            if control["degree"] != row["degree"] or control["coefficient_function"] != "static":
                raise ValueError(f"Control is not degree matched: {row['run_id']}")
            row["perplexity_percent_above_control"] = 100 * (
                row["terminal_validation_perplexity"] / control["terminal_validation_perplexity"] - 1
            )
    return rows, curves


def render(rows, curves):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.hashsalt": "input-dependent-coefficients"})
    styles = {"static": ("#276FBF", "-", "Static learned"),
              "tanh": ("#D65F00", "-", "Input-dependent tanh"),
              "affine": ("#707070", ":", "Input-dependent affine")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for degree, ax in enumerate(axes, 1):
        for function in ("static", "tanh", "affine"):
            for row in rows:
                if row["degree"] != degree or row["coefficient_function"] != function:
                    continue
                points = curves[row["run_id"]]
                color, linestyle, label = styles[function]
                ax.plot([p["training_tokens"] / 1e9 for p in points],
                        [p["validation_nll"] for p in points], color=color, linestyle=linestyle,
                        linewidth=1.8, label=label)
                ax.plot(points[-1]["training_tokens"] / 1e9, points[-1]["validation_nll"],
                        "o", color=color, markersize=3)
        matched = {row["coefficient_function"]: row for row in rows if row["degree"] == degree}
        tanh, static = matched["tanh"], matched["static"]
        ax.set_title(f"Degree {degree}", fontweight="bold")
        ax.text(.98, .96, f"Final PPL: {static['terminal_validation_perplexity']:.2f} static\n"
                f"{tanh['terminal_validation_perplexity']:.2f} tanh (+{tanh['perplexity_percent_above_control']:.2f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_xlabel("Training tokens (billions)")
        ax.set_xlim(0, 10.25)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Validation NLL (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("Input-dependent coefficients: matched degree, warmup 400, output RMSNorm", fontsize=12)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(DIRECTORY / "learning_curves.png", dpi=180, metadata={"Software": "Matplotlib"})
    fig.savefig(DIRECTORY / "learning_curves.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                            "Creator": "Matplotlib"})
    plt.close(fig)


def update_table(rows):
    path = DIRECTORY / "README.md"
    text = path.read_text()
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    prefix, rest = text.split(begin)
    _, suffix = rest.split(end)
    lines = ["| Degree | Coefficients | Terminal validation NLL | Terminal PPL | Best validation PPL | Best step |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r: (r["degree"], {"static": 0, "affine": 1, "tanh": 2}[r["coefficient_function"]])):
        lines.append(f"| {row['degree']} | {row['coefficient_function']} | {row['terminal_validation_nll']:.6f} | "
                     f"{row['terminal_validation_perplexity']:.6f} | {row['best_validation_perplexity']:.6f} | "
                     f"{row['best_validation_step']:,} |")
    path.write_text(prefix + begin + "\n\n" + "\n".join(lines) + "\n\n" + end + suffix)


def main():
    rows, curves = load_runs()
    with (DIRECTORY / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, curves)
    update_table(rows)


if __name__ == "__main__":
    main()
