"""Regenerate this record's summary, figures, and README results block from runs/.

Until every run has concluded (a completion record or a numerical-failure record),
the report writes nothing and prints ``pending``, so no derived output can be
fabricated from partial runs. Controls are read as data: the learned-coefficient
unnormalized Taylor runs without QK-norm and the QK-norm softmax baseline. Kernel, sign, and scale statistics come from the
per-layer, per-head arrays each validation records; a failed run reports its last
valid validation and its stopping update.
"""
from __future__ import annotations

import argparse
import json
import yaml
import math
from pathlib import Path

from functools import partial
from scaffolds.transformer.analysis import selected_views, selected_run_directory

from scaffolds.transformer.analysis import verify_result_selection

import numpy as np

from scaffolds.transformer.analysis import committed_records, save_figure, terminal_row, update_results_readme, write_csv

locate_run = partial(selected_run_directory, Path(__file__).resolve().parent)

OUTPUT = Path(__file__).resolve().parent
RECORD = OUTPUT.parents[1] / "experiments" / "qk_norm_learned_taylor"
EXPERIMENTS = RECORD.parent
DEGREES = (1, 2, 3, 4)
VARIANTS = ("unnormalized", "normalized")
RUNS = tuple(f"200m-qk-m{degree:03d}-{variant}-seed0" for degree in DEGREES for variant in VARIANTS)
LEARNED_CONTROLS = {degree: locate_run(EXPERIMENTS / "learned_unnormalized_taylor", f"200m-m{degree:03d}-seed0") for degree in DEGREES}
QK_SOFTMAX = locate_run(EXPERIMENTS / "qk_norm_attention", "200m-qk-softmax-seed0")
# Categorical slots 1-4 of the repository's figure palette, one per degree.
DEGREE_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COEFFICIENT_NAMES = ("constant", "linear", "quadratic", "cubic", "quartic")
FIELDS = ["run", "attention_degree", "normalization", "backend", "status", "parameters", "updates", "prediction_tokens",
          "terminal_validation_nll", "terminal_validation_perplexity", "best_step", "best_validation_perplexity",
          "last_valid_step", "last_valid_perplexity", "failure_step", "failure_error",
          "learned_no_qk_norm_control_perplexity", "change_vs_learned_no_qk_norm",
          "qk_norm_softmax_perplexity", "change_vs_qk_norm_softmax",
          "median_constant", "median_linear", "median_quadratic", "median_cubic", "median_quartic",
          "median_kernel_minimum_over_constant", "heads_with_negative_kernel",
          "median_layer_negative_kernel_fraction", "median_layer_negative_weight_fraction",
          "scale_min", "scale_median", "scale_max", "maximum_absolute_score",
          "source_commit", "slurm_job_id", "hardware"]


def run_name(degree: int, variant: str) -> str:
    return f"200m-qk-m{degree:03d}-{variant}-seed0"


def validations(run_directory: Path) -> list[dict]:
    return committed_records(Path(run_directory))


def failure_record(run_directory: Path) -> dict | None:
    files = sorted(Path(run_directory).glob("numerical_failure_step_*.json"))
    return json.loads(files[-1].read_text()) if files else None


def concluded(run_directory: Path) -> bool:
    """A run has concluded when it completed or stopped with a recorded numerical failure."""
    return (run_directory / "completion.json").exists() or failure_record(run_directory) is not None


def completed(record: Path = RECORD) -> bool:
    return all(concluded(locate_run(record, run)) for run in RUNS)


def coefficients(record: dict) -> np.ndarray:
    """Learned coefficients of one validation as [heads over all layers, m + 1]."""
    values = np.asarray(record["attention_diagnostics"]["coefficients"]["effective"], dtype=np.float64)
    return values.reshape(-1, values.shape[-1])


def scales(record: dict) -> np.ndarray:
    values = np.asarray(record["attention_diagnostics"]["qk_norm_scale"]["effective"], dtype=np.float64)
    return values.reshape(-1)


def score_range(record: dict) -> tuple[float, float]:
    layers = record["attention_diagnostics"]["layers"]
    return min(layer["score_min"] for layer in layers), max(layer["score_max"] for layer in layers)


def kernel_minimum_over_constant(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Each head's minimum of p(s) / c_0 over the scores observed on the diagnostic sequence."""
    grid = np.linspace(low, high, 20001)
    kernels = sum(values[:, [order]] * grid[None, :] ** order for order in range(values.shape[1]))
    return kernels.min(axis=1) / values[:, 0]


def median_layer_fraction(record: dict, key: str) -> float:
    fractions = [layer[key] for layer in record["attention_diagnostics"]["layers"] if key in layer]
    return float(np.median(fractions)) if fractions else float("nan")


def control_perplexity(directory: Path):
    completion = directory / "completion.json"
    return json.loads(completion.read_text())["validation_perplexity"] if completion.exists() else None


def _hardware(run_directory: Path, node: str) -> str:
    metadata = json.loads((run_directory / "metadata.json").read_text())
    kinds = {device["kind"] for device in metadata.get("execution_runtime", {}).get("devices", [])}
    gpu = next((name for name in ("B300", "H200", "H100", "A100", "A6000") if all(name in kind for kind in kinds)), "")
    return " ".join(part for part in (f"{len(kinds and metadata['execution_runtime']['devices'])}x{gpu}" if gpu else "", node) if part)


def rows(record: Path = RECORD, *, allow_pending: bool = False) -> list[dict]:
    manifest = selected_views(Path(__file__).resolve().parent, record)
    nodes = {run["name"]: (run.get("attempts") or [{}])[-1].get("node", "")
             for run in manifest["runs"]}
    qk_softmax = control_perplexity(QK_SOFTMAX)
    table = []
    for degree in DEGREES:
        for variant in VARIANTS:
            name = run_name(degree, variant)
            run_directory = locate_run(record, name)
            row = {field: "" for field in FIELDS}
            row.update(run=name, attention_degree=degree, normalization=variant)
            if not (run_directory / "config.yaml").exists():
                if allow_pending:
                    row["status"] = "queued"
                    table.append(row)
                    continue
                raise ValueError(f"{name} has not concluded")
            records = validations(run_directory)
            final = records[-1] if records else None
            failure = failure_record(run_directory)
            config = yaml.safe_load((run_directory / "config.yaml").read_text())
            row.update(backend=config["model"]["attention_backend"], hardware=_hardware(run_directory, nodes.get(name, "")))
            if (run_directory / "completion.json").exists():
                row.update(terminal_row(run_directory), status="complete")
                row["hardware"] = " ".join(part for part in (f"{row['gpu_count']}x{row['gpu_type']}", nodes.get(name, "")) if part)
            else:
                metadata = json.loads((run_directory / "metadata.json").read_text())
                if failure is not None:
                    row.update(status="numerical-failure", failure_step=failure["step"], updates=failure["step"],
                               failure_error=failure["error"].split(":")[0] + ": " + failure["error"].split(": ", 1)[-1][:60],
                               prediction_tokens=failure.get("training_tokens", ""))
                elif allow_pending:
                    row.update(status="running", updates=metadata.get("step", ""))
                else:
                    raise ValueError(f"{name} has not concluded")
                row.update(parameters=metadata.get("parameter_count", config["model"].get("parameter_count", "")),
                           source_commit=metadata.get("source_commit", ""), slurm_job_id=metadata.get("slurm_job_id", ""))
                if final is not None:
                    row.update(last_valid_step=final["step"], last_valid_perplexity=final["validation_perplexity"],
                               parameters=final["parameter_count"], source_commit=final["source_commit"])
                    best = min(records, key=lambda entry: (entry["validation_nll"], -entry["step"]))
                    row.update(best_step=best["step"], best_validation_perplexity=best["validation_perplexity"])
            terminal = row["terminal_validation_perplexity"]
            learned = control_perplexity(LEARNED_CONTROLS[degree])
            row.update(learned_no_qk_norm_control_perplexity=learned if learned is not None else "",
                       change_vs_learned_no_qk_norm=terminal - learned if terminal != "" and learned is not None else "",
                       qk_norm_softmax_perplexity=qk_softmax if qk_softmax is not None else "",
                       change_vs_qk_norm_softmax=terminal - qk_softmax if terminal != "" and qk_softmax is not None else "")
            if final is not None:
                values = coefficients(final)
                low, high = score_range(final)
                minimum = kernel_minimum_over_constant(values, low, high)
                medians = np.median(values, axis=0)
                alpha = scales(final)
                row.update(
                    **{f"median_{coefficient}": float(medians[order]) if order <= degree else ""
                       for order, coefficient in enumerate(COEFFICIENT_NAMES)},
                    median_kernel_minimum_over_constant=float(np.median(minimum)),
                    heads_with_negative_kernel=int(np.sum(minimum < 0)),
                    median_layer_negative_kernel_fraction=median_layer_fraction(final, "negative_kernel_fraction"),
                    median_layer_negative_weight_fraction=median_layer_fraction(final, "negative_weight_fraction"),
                    scale_min=float(alpha.min()), scale_median=float(np.median(alpha)), scale_max=float(alpha.max()),
                    maximum_absolute_score=max(abs(low), abs(high)),
                )
            table.append(row)
    return table


def _plain_log(axis) -> None:
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


def _series(record: Path) -> dict:
    return {(degree, variant): validations(locate_run(record, run_name(degree, variant)))
            for degree in DEGREES for variant in VARIANTS}


def _mark_end(axis, records, failed: bool, color: str, value) -> None:
    if not records:
        return
    step = records[-1]["step"] / 1000
    if failed:
        axis.plot([step], [value], "x", color=color, markersize=7, markeredgewidth=2)
    else:
        axis.plot([step], [value], "o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=1.5)


def figures(record: Path = RECORD, output: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "0", "path.simplify": False})
    directory = (output or OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    series = _series(record)
    failures = {key: failure_record(locate_run(record, run_name(*key))) for key in series}
    failed = {key: failure is not None for key, failure in failures.items()}
    line = dict(linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    softmax = validations(QK_SOFTMAX)
    titles = {"unnormalized": "Unnormalized: p(s) V", "normalized": "Normalized: p(s) V / rowsum"}

    # Learning curves, one panel per normalization, degrees by color; controls in gray/dotted.
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for axis, variant in zip(axes, VARIANTS):
        if softmax:
            axis.plot([entry["step"] / 1000 for entry in softmax], [entry["validation_perplexity"] for entry in softmax],
                      color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
        for degree in DEGREES:
            color = DEGREE_COLORS[degree]
            if variant == "unnormalized":
                control = validations(LEARNED_CONTROLS[degree])
                if control:
                    axis.plot([entry["step"] / 1000 for entry in control],
                              [entry["validation_perplexity"] for entry in control], color=color, linewidth=1.0,
                              linestyle=(0, (1, 2)), alpha=0.9)
            records = series[(degree, variant)]
            if records:
                axis.plot([entry["step"] / 1000 for entry in records], [entry["validation_perplexity"] for entry in records],
                          color=color, label=f"m = {degree}", **line)
                _mark_end(axis, records, failed[(degree, variant)], color, records[-1]["validation_perplexity"])
        _plain_log(axis)
        axis.set_title(titles[variant], color=INK, fontsize=10)
        axis.set_xlabel("Update (thousands)", color=INK, fontsize=9)
        axis.set_xlim(0, 19.5)
        _style(axis)
    axes[0].set_ylabel("Held-out perplexity (log scale)", color=INK, fontsize=9)
    handles = [Line2D([], [], color=DEGREE_COLORS[degree], linewidth=2, label=f"m = {degree}") for degree in DEGREES]
    handles += [Line2D([], [], color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), label="QK-norm softmax"),
                Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)), label="Learned, no QK-norm (unnormalized panel)"),
                Line2D([], [], color=MUTED, marker="x", linestyle="none", markersize=7, markeredgewidth=2, label="Numerical failure")]
    axes[1].legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK, loc="upper right")
    figure.tight_layout()
    save_figure(figure, directory / "learning_curves")
    plt.close(figure)

    # Learned kernels at the last validation of each run, relative to each head's constant.
    figure, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True, sharey=True)
    grid = np.linspace(-6, 6, 601)
    for row_index, variant in enumerate(VARIANTS):
        for column, degree in enumerate(DEGREES):
            axis = axes[row_index, column]
            records = series[(degree, variant)]
            color = DEGREE_COLORS[degree]
            if records:
                values = coefficients(records[-1])
                kernels = sum(values[:, [order]] * grid[None, :] ** order for order in range(degree + 1)) / values[:, [0]]
                for kernel in kernels:
                    axis.plot(grid, kernel, color=color, linewidth=0.6, alpha=0.3)
                low, high = score_range(records[-1])
                axis.axvspan(max(low, -6), min(high, 6), color=GRID, alpha=0.5, linewidth=0)
                status = (f"failed {failures[(degree, variant)]['step']:,}, shown {records[-1]['step']:,}"
                          if failed[(degree, variant)] else f"update {records[-1]['step']:,}")
                axis.set_title(f"m = {degree} {variant}: {status}", color=INK, fontsize=8.5)
            else:
                axis.set_title(f"m = {degree} {variant}: failed before a validation", color=INK, fontsize=8.5)
            taylor = sum(grid ** order / math.factorial(order) for order in range(degree + 1))
            axis.plot(grid, taylor, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
            axis.axhline(0, color=MUTED, linewidth=0.8)
            axis.set_ylim(-2, 8)
            _style(axis)
            if row_index == 1:
                axis.set_xlabel("Score s", color=INK, fontsize=9)
    for row_index in range(2):
        axes[row_index, 0].set_ylabel("Kernel p(s) / p(0)", color=INK, fontsize=9)
    axes[0, 0].legend(handles=[Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="Taylor, 1/r!"),
                               Line2D([], [], color=MUTED, linewidth=0.6, alpha=0.6, label="Learned heads (160)"),
                               plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.5, label="Observed score range")],
                      frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    save_figure(figure, directory / "learned_kernels")
    plt.close(figure)

    # Scales and kernel signs over training.
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    for column, variant in enumerate(VARIANTS):
        for degree in DEGREES:
            records = series[(degree, variant)]
            if not records:
                continue
            color = DEGREE_COLORS[degree]
            steps = [entry["step"] / 1000 for entry in records]
            medians = [float(np.median(scales(entry))) for entry in records]
            fractions = [median_layer_fraction(entry, "negative_kernel_fraction") for entry in records]
            axes[0, column].plot(steps, medians, color=color, label=f"m = {degree}", **line)
            axes[1, column].plot(steps, fractions, color=color, label=f"m = {degree}", **line)
            _mark_end(axes[0, column], records, failed[(degree, variant)], color, medians[-1])
            _mark_end(axes[1, column], records, failed[(degree, variant)], color, fractions[-1])
        if softmax:
            axes[0, column].plot([entry["step"] / 1000 for entry in softmax], [float(np.median(scales(entry))) for entry in softmax],
                                 color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), label="QK-norm softmax")
        axes[0, column].axhline(8.0, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
        axes[0, column].set_title(titles[variant], color=INK, fontsize=10)
        axes[1, column].set_xlabel("Update (thousands)", color=INK, fontsize=9)
        axes[1, column].set_ylim(-0.02, 0.75)
        for axis in axes[:, column]:
            _style(axis)
            axis.set_xlim(0, 19.5)
    axes[0, 0].set_ylabel("Median exp(γ) over heads", color=INK, fontsize=9)
    axes[1, 0].set_ylabel("Negative kernel values\n(median layer fraction)", color=INK, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    save_figure(figure, directory / "scales_and_signs")
    plt.close(figure)


def results_lines(table: list[dict]) -> list[str]:
    def number(value, form):
        return "—" if value == "" or value is None or (isinstance(value, float) and math.isnan(value)) else form.format(value)

    lines = [
        "Held-out perplexity at the terminal validation (19,074 updates, 10,000,269,312 prediction tokens; seed 0) "
        "or, for a numerical failure, the last valid validation and the stopping update. Controls: the "
        "learned-coefficient unnormalized Taylor run of the same degree without QK-norm (all-parameter decay) and "
        "the QK-norm softmax baseline (14.4574). Kernel statistics are the last validation's 160 layer-heads over "
        "the scores observed on the fixed diagnostic sequence; negative kernel values are the median layer's "
        "fraction of causal entries; scales are exp(γ) over the 160 heads.",
        "",
        "| Run | Status | Terminal PPL | Learned, no QK-norm | Δ | Δ vs QK-norm softmax | Median c0 | Heads with negative kernel "
        "| Negative kernel values | exp(γ) median (min–max) | Largest abs score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        status = row["status"]
        if status == "numerical-failure":
            status = f"failed at update {row['failure_step']:,}; last valid {row['last_valid_perplexity']:.2f} at {row['last_valid_step']:,}" \
                if row["last_valid_step"] != "" else f"failed at update {row['failure_step']:,} before the first validation"
        heads = number(row["heads_with_negative_kernel"], "{}")
        lines.append(
            f"| m={row['attention_degree']} {row['normalization']} | {status} "
            f"| {number(row['terminal_validation_perplexity'], '{:.4f}')} "
            f"| {number(row['learned_no_qk_norm_control_perplexity'], '{:.4f}')} "
            f"| {number(row['change_vs_learned_no_qk_norm'], '{:+.4f}')} | {number(row['change_vs_qk_norm_softmax'], '{:+.4f}')} "
            f"| {number(row['median_constant'], '{:.3f}')} | {heads if heads == '—' else heads + '/160'} "
            f"| {number(row['median_layer_negative_kernel_fraction'], '{:.2f}')} "
            f"| {number(row['scale_median'], '{:.1f}')} ({number(row['scale_min'], '{:.1f}')}–{number(row['scale_max'], '{:.1f}')}) "
            f"| {number(row['maximum_absolute_score'], '{:.1f}')} |")
    lines += ["", "[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf) · "
              "[Learned kernels](learned_kernels.pdf) · [Scales and signs](scales_and_signs.pdf)"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write outputs here instead of the record (regeneration checks)")
    args = parser.parse_args()
    verify_result_selection(Path(__file__).resolve().parent)
    if not completed():
        done = sum(concluded(locate_run(RECORD, run)) for run in RUNS)
        print(f"pending: {done}/{len(RUNS)} runs concluded; no summary, figures, or results block written")
        return
    output = args.output or OUTPUT
    output.mkdir(parents=True, exist_ok=True)
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
