"""Regenerate the explicitly selected input-dependent coefficient comparison.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.input_dependent_coefficients.plot
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
FIELDS = (
    "run_id", "role", "degree", "coefficient_function", "start", "attention_backend", "variant", "status",
    "terminal_step", "training_tokens", "terminal_validation_nll", "terminal_validation_perplexity",
    "best_validation_step", "best_validation_perplexity", "terminal_train_nll_sample_mean",
    "control_run_id", "perplexity_percent_above_control", "valid_diagnostics",
    "diagnostic_count", "mean_coefficient_std", "median_token_variation", "median_abs_amplitude",
    "saturated_mean_branch_fraction", "mean_tanh_derivative_from_moments", "metrics_sha256", "completion_sha256",
)
# Variant key -> (table name, color, line style, legend label); the order is the table and legend order.
VARIANTS = {
    "static": ("static", "#276FBF", "-", "Static learned"),
    "affine": ("affine", "#707070", ":", "Input-dependent affine"),
    "tanh_original": ("tanh, original start, fused kernel", "#D65F00", "-", "Tanh, original start, fused kernel"),
    "tanh_original_blocked": ("tanh, original start, blocked reference", "#D65F00", "--",
                              "Tanh, original start, blocked reference"),
    "tanh_zero_amplitude": ("tanh, zero-amplitude start, blocked reference", "#1B9E77", "-",
                            "Tanh, zero-amplitude start, blocked reference"),
    "tanh_zero_amplitude_wide": ("tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel", "#1B9E77", (0, (5, 2.5)),
                                 "Tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel"),
    "tanh_zero_amplitude_inner_bias": ("tanh, zero-amplitude start, bias inside tanh, Mosaic kernel", "#1B9E77", ":",
                                       "Tanh, zero-amplitude start, bias inside tanh, Mosaic kernel"),
    "tanh_zero_amplitude_saturating_3": ("tanh, zero-amplitude start, W ~ N(0, 9/d), Mosaic kernel", "#80CDC1", "-",
                                         "Tanh, zero-amplitude start, W ~ N(0, 9/d), Mosaic kernel"),
    "tanh_zero_amplitude_saturating_10": ("tanh, zero-amplitude start, W ~ N(0, 100/d), Mosaic kernel", "#80CDC1", "--",
                                          "Tanh, zero-amplitude start, W ~ N(0, 100/d), Mosaic kernel"),
    "tanh_zero_amplitude_order_scaled": ("tanh, zero-amplitude start, W ~ N(0, 1/d), amplitude a_r/r!, Mosaic kernel",
                                         "#E7298A", "-",
                                         "Tanh, zero-amplitude start, W ~ N(0, 1/d), amplitude a_r/r!, Mosaic kernel"),
    "tanh_query_key": ("tanh query-key, zero-amplitude start, W ~ N(0, 1/d)", "#7B3294", "-",
                       "Tanh query-key, zero-amplitude start, W ~ N(0, 1/d)"),
}


def variant_of(model) -> tuple[str, str, str]:
    """(coefficient function, start, variant key) from a run's model configuration."""
    function = (model.get("attention_coefficient_activation", "identity")
                if model.get("attention_coefficient_conditioning", "static") in ("query", "query_key") else "static")
    function = "affine" if function == "identity" else function
    if model.get("attention_coefficient_conditioning") == "query_key":
        return "tanh", "zero_amplitude", "tanh_query_key"
    if function != "tanh":
        return function, "", function
    start = "zero_amplitude" if model.get("attention_coefficient_amplitude_initial", 1.0) == 0 else "original"
    blocked = model["attention_backend"] == "blocked"
    if start == "original":
        return function, start, "tanh_original_blocked" if blocked else "tanh_original"
    if model.get("attention_coefficient_tanh_bias", False):
        return function, start, "tanh_zero_amplitude_inner_bias"
    if model.get("attention_coefficient_amplitude_scale", "none") == "factorial":
        return function, start, "tanh_zero_amplitude_order_scaled"
    spread = model.get("attention_coefficient_projection_initial_std", 0.0) * math.sqrt(model["hidden_size"])
    if spread > 5:
        return function, start, "tanh_zero_amplitude_saturating_10"
    if spread > 2:
        return function, start, "tanh_zero_amplitude_saturating_3"
    if spread > 0.5:
        return function, start, "tanh_zero_amplitude_wide"
    if not blocked:
        raise ValueError("The small-draw zero-amplitude runs in this selection use the blocked reference")
    return function, start, "tanh_zero_amplitude"


def load_runs(selection: dict | None = None):
    selection = selection or yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
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
        if json.loads((path / "record.json").read_text())["status"] != "complete":
            raise ValueError(f"Selected run is not recorded as complete: {entry['run_id']}")
        function, start, variant = variant_of(model)
        diagnostics = [row["attention_diagnostics"] for row in validation]
        row = dict(
            run_id=entry["run_id"], role=entry["role"], degree=model["attention_degree"],
            coefficient_function=function, start=start, attention_backend=model["attention_backend"],
            variant=variant, status="complete", terminal_step=completion["final_step"],
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
            # Per layer/head/order, the coefficient's standard deviation across the pinned
            # sequence's tokens relative to its mean; the median over all branches.
            row["median_token_variation"] = float(np.median(
                np.asarray(summary["std"]) / np.abs(np.asarray(summary["mean"]))))
        if function == "tanh":
            # The amplitude that multiplies tanh: a_r, or a_r/r! with order scaling.
            amplitude = np.asarray(coefficients.get("effective_modulation_amplitude",
                                                    coefficients["modulation_amplitude"]))
            bias = np.asarray(coefficients["projection_bias"])
            row["median_abs_amplitude"] = float(np.median(np.abs(amplitude)))
            mean = (np.asarray(summary["mean"]) - bias) / amplitude
            std = np.asarray(summary["std"]) / np.abs(amplitude)
            # Each branch is one layer/head/order; these moments cover one pinned sequence and
            # are not informative where the amplitude is near zero.
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


def render(rows, curves, output=DIRECTORY):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.hashsalt": "input-dependent-coefficients"}):
        _render(rows, curves, output)


def _render(rows, curves, output):
    degrees = sorted({row["degree"] for row in rows})
    columns = 3 if len(degrees) > 4 else 2
    fig, axes = plt.subplots(2, columns, figsize=(5 * columns, 7.8))
    axes = axes.ravel()
    for ax in axes[len(degrees):]:
        ax.axis("off")
    for degree, ax in zip(degrees, axes):
        matched = {}
        for variant, (_, color, linestyle, label) in VARIANTS.items():
            for row in rows:
                if row["degree"] != degree or row["variant"] != variant:
                    continue
                matched[variant] = row
                points = curves[row["run_id"]]
                ax.plot([p["training_tokens"] / 1e9 for p in points],
                        [p["validation_nll"] for p in points], color=color, linestyle=linestyle,
                        linewidth=1.8, label=label)
                ax.plot(points[-1]["training_tokens"] / 1e9, points[-1]["validation_nll"],
                        "o", color=color, markersize=3)
        ax.set_title(f"Degree {degree}", fontweight="bold")
        final = [f"Final PPL: {matched['static']['terminal_validation_perplexity']:.2f} static"]
        zero = "tanh_zero_amplitude" if "tanh_zero_amplitude" in matched else "tanh_zero_amplitude_wide"
        for variant, name in (("tanh_original", "original start"), (zero, "zero-amplitude start"),
                              ("tanh_zero_amplitude_order_scaled", "a_r/r!"), ("tanh_query_key", "query-key")):
            if variant in matched:
                row = matched[variant]
                final.append(f"{row['terminal_validation_perplexity']:.2f} {name} "
                             f"({row['perplexity_percent_above_control']:+.2f}%)")
        ax.text(.98, .96, "\n".join(final), transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_xlabel("Training tokens (billions)")
        ax.set_xlim(0, 10.25)
        ax.grid(alpha=.18)
    for ax in axes[::columns]:
        ax.set_ylabel("Validation NLL (lower is better)")
    seen = {}
    for ax in axes[:len(degrees)]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    order = [VARIANTS[key][3] for key in VARIANTS if VARIANTS[key][3] in seen]
    handles, labels = [seen[label] for label in order], order
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("Input-dependent coefficients: matched degree, warmup 400, output RMSNorm", fontsize=12)
    fig.tight_layout(rect=(0, .16, 1, .94))
    fig.savefig(output / "learning_curves.png", dpi=180, metadata={"Software": "Matplotlib"})
    fig.savefig(output / "learning_curves.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                            "Creator": "Matplotlib"})
    plt.close(fig)


def update_table(rows, output=DIRECTORY):
    path = output / "README.md"
    text = path.read_text()
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    prefix, rest = text.split(begin)
    _, suffix = rest.split(end)
    lines = ["| Degree | Variant | Terminal validation NLL | Terminal PPL | vs static | Best validation PPL | Best step |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    order = list(VARIANTS)
    for row in sorted(rows, key=lambda r: (r["degree"], order.index(r["variant"]))):
        relative = ("" if row["variant"] == "static" else f"{row['perplexity_percent_above_control']:+.2f}%")
        lines.append(f"| {row['degree']} | {VARIANTS[row['variant']][0]} | {row['terminal_validation_nll']:.6f} | "
                     f"{row['terminal_validation_perplexity']:.6f} | {relative} | {row['best_validation_perplexity']:.6f} | "
                     f"{row['best_validation_step']:,} |")
    path.write_text(prefix + begin + "\n\n" + "\n".join(lines) + "\n\n" + end + suffix)


def main(output=DIRECTORY):
    rows, curves = load_runs()
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, curves, output)
    update_table(rows, output)


if __name__ == "__main__":
    main()
