"""Regenerate this record's summary, figures, and README results block from runs/.

Controls are read as data: the original-decay softmax control of
`key_dimension_taylor` and, once its record exists in this checkout, the
canonical-decay softmax control of `decay_policy_baseline`. Scale statistics come
from the per-layer, per-head exp(gamma) values each validation records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scaffolds.transformer.analysis import committed_records, save_figure, terminal_row, update_results_readme, write_csv

RECORD = Path(__file__).resolve().parents[2]
EXPERIMENTS = RECORD.parent
RUNS = ("200m-qk-softmax-seed0",)
CONTROLS = {
    "original_decay": ("Softmax, original decay", EXPERIMENTS / "key_dimension_taylor/runs/200m-softmax-seed0"),
    "canonical_decay": ("Softmax, canonical decay", EXPERIMENTS / "decay_policy_baseline/runs/200m-softmax-seed0"),
}
# Categorical slots 1-3 of the repository's figure palette; layer depth uses one sequential hue.
QK_NORM, ORIGINAL, CANONICAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONTROL_COLORS = {"original_decay": ORIGINAL, "canonical_decay": CANONICAL}
FIELDS = ["run", "status", "parameters", "updates", "prediction_tokens", "terminal_validation_nll",
          "terminal_validation_perplexity", "best_step", "best_validation_perplexity",
          "softmax_original_decay_perplexity", "change_vs_original_decay",
          "softmax_canonical_decay_perplexity", "change_vs_canonical_decay",
          "initial_scale", "terminal_scale_min", "terminal_scale_median", "terminal_scale_max",
          "terminal_layer_median_scales", "maximum_absolute_score", "source_commit", "slurm_job_id", "hardware"]


def validations(run_directory: Path) -> list[dict]:
    return committed_records(Path(run_directory))


def scales(record: dict) -> np.ndarray:
    """exp(gamma) of one validation as [layers, heads]."""
    values = np.asarray(record["attention_diagnostics"]["qk_norm_scale"]["effective"], dtype=np.float64)
    return values.reshape(values.shape[0], values.shape[1])


def score_range(record: dict) -> tuple[float, float]:
    layers = record["attention_diagnostics"]["layers"]
    return min(layer["score_min"] for layer in layers), max(layer["score_max"] for layer in layers)


def control_records(name: str) -> list[dict]:
    """Committed validations of a control, or none when its record is absent from this checkout."""
    directory = CONTROLS[name][1]
    return validations(directory) if (directory / "metrics.jsonl").exists() else []


def control_terminal_perplexity(name: str):
    """Terminal perplexity from a control's completion record; older records lack runtime fields."""
    directory = CONTROLS[name][1]
    if not (directory / "completion.json").exists():
        return None
    completion = json.loads((directory / "completion.json").read_text())
    return completion["validation_perplexity"]


def rows(record: Path = RECORD) -> list[dict]:
    manifest = json.loads((record / "provenance/manifest.json").read_text())
    nodes = {run["name"]: (run.get("execution", {}).get("attempts") or [{}])[-1].get("node", "")
             for run in manifest["runs"]}
    result = []
    for name in RUNS:
        run_directory = record / "runs" / name
        records = validations(run_directory)
        if not records:
            raise ValueError(f"No committed validations for {name}")
        final = records[-1]
        if (run_directory / "completion.json").exists():
            row = terminal_row(run_directory)
            row["status"] = "complete"
        else:
            # A run in progress summarizes its latest committed validation; completion replaces this.
            best = min(records, key=lambda entry: (entry["validation_nll"], -entry["step"]))
            row = {"status": "running", "parameters": final["parameter_count"], "updates": final["step"],
                   "prediction_tokens": final["training_tokens"], "terminal_validation_nll": "",
                   "terminal_validation_perplexity": "", "best_step": best["step"],
                   "best_validation_perplexity": best["validation_perplexity"],
                   "source_commit": final["source_commit"], "slurm_job_id": final["slurm_job_id"],
                   "gpu_count": "", "gpu_type": ""}
        terminal = scales(final)
        initial = final["qk_norm_initial_scale"]
        terminal_ppl = row["terminal_validation_perplexity"]
        controls = {key: control_terminal_perplexity(key) for key in CONTROLS}
        row.update(
            run=name,
            softmax_original_decay_perplexity=controls["original_decay"] if controls["original_decay"] is not None else "",
            change_vs_original_decay=(terminal_ppl - controls["original_decay"]
                                      if terminal_ppl != "" and controls["original_decay"] is not None else ""),
            softmax_canonical_decay_perplexity=controls["canonical_decay"] if controls["canonical_decay"] is not None else "",
            change_vs_canonical_decay=(terminal_ppl - controls["canonical_decay"]
                                       if terminal_ppl != "" and controls["canonical_decay"] is not None else ""),
            initial_scale=initial,
            terminal_scale_min=float(terminal.min()), terminal_scale_median=float(np.median(terminal)),
            terminal_scale_max=float(terminal.max()),
            terminal_layer_median_scales=" ".join(f"{value:.3f}" for value in np.median(terminal, axis=1)),
            maximum_absolute_score=max(abs(value) for value in score_range(final)),
            hardware=" ".join(part for part in (f"{row['gpu_count']}x{row['gpu_type']}" if row["gpu_type"] else "",
                                                nodes.get(name, "")) if part),
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


def figures(record: Path = RECORD, output: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "0", "path.simplify": False})
    directory = (output or record) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    line = dict(linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    series = [("QK-norm softmax", QK_NORM, validations(record / "runs" / RUNS[0]))]
    series += [(label, CONTROL_COLORS[key], control_records(key)) for key, (label, _) in CONTROLS.items()
               if control_records(key)]

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, color, records in series:
        steps = [entry["step"] / 1000 for entry in records]
        perplexities = [entry["validation_perplexity"] for entry in records]
        # Terminal values go in the legend: the curves end within 0.3 of one another.
        axes[0].plot(steps, perplexities, color=color, label=f"{label} ({perplexities[-1]:.2f})", **line)
        axes[0].plot(steps[-1:], perplexities[-1:], "o", color=color, markersize=5, markeredgecolor="white",
                     markeredgewidth=1.5)
    _plain_log(axes[0])
    axes[0].set_xlabel("Update (thousands)", color=INK, fontsize=9)
    axes[0].set_ylabel("Held-out perplexity (log scale)", color=INK, fontsize=9)
    axes[0].set_xlim(left=0)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    # The difference against the canonical-decay control, when that control is present.
    reference = next((records for label, _, records in series if label == CONTROLS["canonical_decay"][0]), None)
    if reference is not None:
        by_step = {entry["step"]: entry["validation_perplexity"] for entry in reference}
        for label, color, records in series:
            if records is reference:
                continue
            common = [entry for entry in records if entry["step"] in by_step]
            steps = [entry["step"] / 1000 for entry in common]
            differences = [entry["validation_perplexity"] - by_step[entry["step"]] for entry in common]
            axes[1].plot(steps, differences, color=color, label=label, **line)
            axes[1].annotate(f"{differences[-1]:+.2f}", (steps[-1], differences[-1]), xytext=(6, 0),
                             textcoords="offset points", color=INK, fontsize=8, va="center")
        axes[1].axhline(0, color=MUTED, linewidth=0.8)
        axes[1].set_ylim(-0.75, 0.75)
        axes[1].set_xlim(left=0)
        axes[1].set_xlabel("Update (thousands)", color=INK, fontsize=9)
        axes[1].set_ylabel("Perplexity minus canonical-decay softmax", color=INK, fontsize=9)
        axes[1].set_title("Difference from the canonical-decay control (update 1,000: +5.7, clipped)",
                          color=INK, fontsize=9)
        axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    else:
        axes[1].set_axis_off()
    for axis in axes:
        _style(axis)
    figure.tight_layout()
    save_figure(figure, directory / "learning_curves")
    plt.close(figure)

    records = series[0][2]
    trajectory = np.stack([scales(entry) for entry in records])  # [validations, layers, heads]
    steps = np.array([entry["step"] / 1000 for entry in records])
    layers = trajectory.shape[1]
    initial = records[0]["qk_norm_initial_scale"]
    depth = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, layers))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for layer in range(layers):
        # Every scale starts at the declared initial value before the first validation.
        mean = np.concatenate(([initial], trajectory[:, layer].mean(axis=1)))
        axes[0].plot(np.concatenate(([0.0], steps)), mean, color=depth[layer], linewidth=1.6, solid_capstyle="round")
        if layer in (0, layers - 1):
            axes[0].annotate(f"layer {layer + 1}", (steps[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                             color=INK, fontsize=8, va="center")
    axes[0].axhline(initial, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].annotate(f"initial {initial:g}", (0, initial), xytext=(4, -10), textcoords="offset points",
                     color=MUTED, fontsize=8)
    axes[0].set_xlabel("Update (thousands)", color=INK, fontsize=9)
    axes[0].set_ylabel("exp(γ), mean over heads", color=INK, fontsize=9)
    axes[0].set_title("Learned scale by layer (light: first, dark: last)", color=INK, fontsize=10)
    axes[0].set_xlim(left=0)
    rng = np.random.default_rng(0)
    for layer in range(layers):
        values = trajectory[-1, layer]
        jitter = rng.uniform(-0.18, 0.18, size=values.shape)
        axes[1].plot(layer + 1 + jitter, values, "o", color=QK_NORM, markersize=4.5, markeredgecolor="white",
                     markeredgewidth=1.0, alpha=0.9)
    axes[1].plot(np.arange(1, layers + 1), np.median(trajectory[-1], axis=1), "_", color=INK, markersize=16,
                 markeredgewidth=1.6, label="Layer median")
    axes[1].axhline(initial, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), label=f"Initial {initial:g}")
    axes[1].set_xticks(np.arange(1, layers + 1))
    axes[1].set_xlabel("Layer", color=INK, fontsize=9)
    axes[1].set_ylabel("exp(γ) per head", color=INK, fontsize=9)
    axes[1].set_title(f"Scales at update {records[-1]['step']:,} ({layers * trajectory.shape[2]} heads)",
                      color=INK, fontsize=10)
    for axis in axes:
        _style(axis)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    save_figure(figure, directory / "qk_norm_scales")
    plt.close(figure)


def results_lines(table: list[dict]) -> list[str]:
    def perplexity(value):
        return "pending" if value == "" else f"{value:.4f}"

    def change(value):
        return "—" if value == "" else f"{value:+.4f}"

    lines = [
        "Held-out perplexity at the terminal validation (19,074 updates, 10,000,269,312 prediction tokens; seed 0) "
        "against the softmax controls: the original-decay control trained with weight decay on every parameter, "
        "the canonical-decay control under the selective policy this run uses. Scales are the terminal "
        "validation's 160 layer-heads; the score bound is the largest exp(γ), and the largest |score| is over the "
        "fixed diagnostic sequence.",
        "",
        "| Run | Status | Terminal PPL | Best PPL (update) | Softmax, original decay | Δ | Softmax, canonical decay | Δ "
        "| exp(γ) min / median / max | Largest abs score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['run']} | {row['status']} | {perplexity(row['terminal_validation_perplexity'])} "
            f"| {row['best_validation_perplexity']:.4f} ({row['best_step']:,}) "
            f"| {perplexity(row['softmax_original_decay_perplexity'])} | {change(row['change_vs_original_decay'])} "
            f"| {perplexity(row['softmax_canonical_decay_perplexity'])} | {change(row['change_vs_canonical_decay'])} "
            f"| {row['terminal_scale_min']:.2f} / {row['terminal_scale_median']:.2f} / {row['terminal_scale_max']:.2f} "
            f"| {row['maximum_absolute_score']:.2f} |")
    lines += ["", "[Summary CSV](summary.csv) · [Learning curves](figures/learning_curves.pdf) · "
              "[Learned scales](figures/qk_norm_scales.pdf)"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write outputs here instead of the record (regeneration checks)")
    args = parser.parse_args()
    output = args.output or RECORD
    output.mkdir(parents=True, exist_ok=True)
    table = rows()
    write_csv(output / "summary.csv", table, FIELDS)
    figures(output=output)
    readme = output / "README.md"
    if args.output is not None:
        readme.write_text((RECORD / "README.md").read_text())
    update_results_readme(readme, results_lines(table))
    print(output)


if __name__ == "__main__":
    main()
