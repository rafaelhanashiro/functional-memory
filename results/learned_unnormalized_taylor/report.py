"""Regenerate this record's summary, figures, and README results block from runs/.

Controls are the committed fixed-coefficient runs of `unnormalized_taylor_attention`
(read as data). Coefficient statistics come from the per-layer, per-head arrays each
validation records; kernel shapes are compared scale-free, as p(s) / p(0) = p(s) / c_0.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from functools import partial
from scaffolds.transformer.analysis import selected_views, selected_run_directory

from scaffolds.transformer.analysis import verify_result_selection

import numpy as np

from scaffolds.transformer.analysis import save_figure, terminal_row, update_results_readme, write_csv

locate_run = partial(selected_run_directory, Path(__file__).resolve().parent)

OUTPUT = Path(__file__).resolve().parent
RECORD = OUTPUT.parents[1] / "experiments" / "attention_scaling"
CONTROLS = RECORD
DEGREES = (1, 2, 3, 4)
LEARNED, FIXED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
FIELDS = ["run", "attention_degree", "parameters", "updates", "prediction_tokens", "terminal_validation_nll",
          "terminal_validation_perplexity", "best_step", "best_validation_perplexity",
          "fixed_control_validation_perplexity", "perplexity_change_vs_fixed", "median_constant", "median_linear",
          "median_quadratic", "median_cubic", "median_quartic", "median_kernel_minimum_over_constant",
          "heads_with_negative_kernel", "median_layer_negative_kernel_fraction", "maximum_absolute_score",
          "fixed_control_maximum_absolute_score", "source_commit", "slurm_job_id", "hardware"]


def run_name(degree: int) -> str:
    return f"200m-m{degree:03d}-seed0"


def validations(run_directory: Path) -> list[dict]:
    rows = [json.loads(line) for line in (run_directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    by_step = {row["step"]: row for row in rows if row.get("event") == "validation"}
    return [by_step[step] for step in sorted(by_step)]


def coefficients(record: dict) -> np.ndarray:
    """Learned coefficients of one validation as [heads over all layers, m + 1]."""
    report = record["attention_diagnostics"]["coefficients"]
    values = np.asarray(report["effective"], dtype=np.float64)
    return values.reshape(-1, values.shape[-1])


def score_range(record: dict) -> tuple[float, float]:
    layers = record["attention_diagnostics"]["layers"]
    return min(layer["score_min"] for layer in layers), max(layer["score_max"] for layer in layers)


def kernel_minimum_over_constant(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Each head's minimum of p(s) / c_0 over the scores observed on the diagnostic sequence."""
    grid = np.linspace(low, high, 20001)
    kernels = sum(values[:, [order]] * grid[None, :] ** order for order in range(values.shape[1]))
    return kernels.min(axis=1) / values[:, 0]


def negative_kernel_fraction(record: dict) -> float:
    fractions = []
    for layer in record["attention_diagnostics"]["layers"]:
        if "negative_kernel_fraction" in layer:
            fractions.append(layer["negative_kernel_fraction"])
        elif layer["kernel_min"] > 0:
            fractions.append(0.0)  # Positive kernels (fixed even Taylor) record no signed-kernel counts.
        else:
            raise ValueError("A signed kernel lacks its negative-value fraction")
    return float(np.median(fractions))


def rows(record: Path = RECORD, controls: Path = CONTROLS) -> list[dict]:
    manifest = selected_views(Path(__file__).resolve().parent, record)
    nodes = {run["name"]: run["attempts"][-1]["node"] for run in manifest["runs"]}
    result = []
    for degree in DEGREES:
        run_directory, control_directory = locate_run(record, run_name(degree)), locate_run(controls, "fixed-" + run_name(degree))
        row = terminal_row(run_directory)
        control = terminal_row(control_directory)
        final, control_final = validations(run_directory)[-1], validations(control_directory)[-1]
        values = coefficients(final)
        minimum = kernel_minimum_over_constant(values, *score_range(final))
        medians = np.median(values, axis=0)
        names = ("constant", "linear", "quadratic", "cubic", "quartic")
        row.update(
            run=run_name(degree), attention_degree=degree,
            fixed_control_validation_perplexity=control["terminal_validation_perplexity"],
            perplexity_change_vs_fixed=row["terminal_validation_perplexity"] - control["terminal_validation_perplexity"],
            **{f"median_{name}": float(medians[order]) if order <= degree else "" for order, name in enumerate(names)},
            # Odd polynomials are unbounded below; only even degrees have a kernel floor.
            median_kernel_minimum_over_constant=float(np.median(minimum)) if degree % 2 == 0 else "",
            heads_with_negative_kernel=int(np.sum(minimum < 0)),
            median_layer_negative_kernel_fraction=negative_kernel_fraction(final),
            maximum_absolute_score=max(abs(value) for value in score_range(final)),
            fixed_control_maximum_absolute_score=max(abs(value) for value in score_range(control_final)),
            hardware=f"{row['gpu_count']}x{row['gpu_type']} {nodes[run_name(degree)]}",
        )
        result.append(row)
    return result


def _plain_log(axis) -> None:
    """Log axis with plain-number ticks (20, 50, 100) rather than powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def figures(record: Path = RECORD, controls: Path = CONTROLS, output: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "0", "path.simplify": False})
    directory = (output or OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    series = {degree: (validations(locate_run(record, run_name(degree))),
                       validations(locate_run(controls, "fixed-" + run_name(degree)))) for degree in DEGREES}
    line = dict(linewidth=2, solid_capstyle="round", solid_joinstyle="round")

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for axis, degree in zip(axes, DEGREES):
        for rows_, color, label in ((series[degree][1], FIXED, "Fixed Taylor coefficients"),
                                    (series[degree][0], LEARNED, "Learned per-head coefficients")):
            steps = [row["step"] / 1000 for row in rows_]
            axis.plot(steps, [row["validation_perplexity"] for row in rows_], color=color, label=label, **line)
            axis.plot(steps[-1:], [rows_[-1]["validation_perplexity"]], "o", color=color, markersize=5,
                      markeredgecolor="white", markeredgewidth=1.5)
        _plain_log(axis)
        axis.set_title(f"m = {degree}", color=INK, fontsize=10)
        axis.set_xlabel("Update (thousands)", color=INK, fontsize=9)
        _style(axis)
    axes[0].set_ylabel("Held-out perplexity (log scale)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    figure.tight_layout()
    save_figure(figure, directory / "learning_curves")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
    for column, degree in enumerate(DEGREES):
        for rows_, color, label in ((series[degree][1], FIXED, "Fixed Taylor coefficients"),
                                    (series[degree][0], LEARNED, "Learned per-head coefficients")):
            steps = [row["step"] / 1000 for row in rows_]
            axes[0, column].plot(steps, [negative_kernel_fraction(row) for row in rows_], color=color, label=label, **line)
            axes[1, column].plot(steps, [max(abs(value) for value in score_range(row)) for row in rows_],
                                 color=color, label=label, **line)
        axes[0, column].set_title(f"m = {degree}", color=INK, fontsize=10)
        axes[0, column].set_ylim(-0.02, 0.75)
        _plain_log(axes[1, column])
        axes[1, column].set_xlabel("Update (thousands)", color=INK, fontsize=9)
        for axis in axes[:, column]:
            _style(axis)
    axes[0, 0].set_ylabel("Negative kernel values\n(median layer fraction)", color=INK, fontsize=9)
    axes[1, 0].set_ylabel("Largest |score| (log scale)", color=INK, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    save_figure(figure, directory / "kernel_usage")
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    grid = np.linspace(-6, 6, 601)
    for axis, degree in zip(axes, DEGREES):
        values = coefficients(series[degree][0][-1])
        kernels = sum(values[:, [order]] * grid[None, :] ** order for order in range(degree + 1)) / values[:, [0]]
        for index, kernel in enumerate(kernels):
            axis.plot(grid, kernel, color=LEARNED, linewidth=0.6, alpha=0.3,
                      label=f"Learned heads ({len(kernels)})" if index == 0 else None)
        taylor = sum(grid ** order / math.factorial(order) for order in range(degree + 1))
        axis.plot(grid, taylor, color=FIXED, label="Taylor, 1/r!", **line)
        axis.axhline(0, color=MUTED, linewidth=0.8)
        axis.set_ylim(-2, 8)
        axis.set_title(f"m = {degree}", color=INK, fontsize=10)
        axis.set_xlabel("Score s", color=INK, fontsize=9)
        _style(axis)
    axes[0].set_ylabel("Kernel p(s) / p(0)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    save_figure(figure, directory / "learned_kernels")
    plt.close(figure)


def results_lines(table: list[dict]) -> list[str]:
    lines = [
        "Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0, against the "
        "fixed-coefficient unnormalized Taylor control of the same degree. Coefficient and kernel statistics are "
        "the terminal validation's 160 layer-heads. Kernel minima are over the scores observed on the fixed "
        "diagnostic sequence, relative to each head's constant, for even degrees (odd polynomials are unbounded "
        "below). Negative kernel values are the median layer's fraction of causal entries on that sequence.",
        "",
        "| m | Learned PPL | Fixed PPL | Change | Median c0 | Median min p(s)/c0 | Heads with negative kernel "
        "| Negative kernel values | Largest abs score (fixed) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        value = row["median_kernel_minimum_over_constant"]
        minimum = "—" if value == "" else f"{value:.3f}"
        lines.append(
            f"| {row['attention_degree']} | {row['terminal_validation_perplexity']:.4f} "
            f"| {row['fixed_control_validation_perplexity']:.4f} | {row['perplexity_change_vs_fixed']:+.4f} "
            f"| {row['median_constant']:.3f} | {minimum} "
            f"| {row['heads_with_negative_kernel']}/160 | {row['median_layer_negative_kernel_fraction']:.2f} "
            f"| {row['maximum_absolute_score']:.1f} ({row['fixed_control_maximum_absolute_score']:.1f}) |")
    lines += ["", "[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf) · "
              "[Kernel usage](kernel_usage.pdf) · [Learned kernels](learned_kernels.pdf)"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write outputs here instead of the record (regeneration checks)")
    args = parser.parse_args()
    verify_result_selection(Path(__file__).resolve().parent)
    output = args.output or OUTPUT
    table = rows()
    write_csv(output / "summary.csv", table, FIELDS)
    figures(output=output)
    readme = output / "README.md"
    if args.output is not None:
        readme.write_text((OUTPUT / "README.md").read_text())
    update_results_readme(readme, results_lines(table))
    print(output)


if __name__ == "__main__":
    main()
