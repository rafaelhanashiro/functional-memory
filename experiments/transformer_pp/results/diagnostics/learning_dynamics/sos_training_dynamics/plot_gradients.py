"""Gradient stability figures from complete, non-overlapping trainer windows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


POST_STEP = 1500
MEAN_COLOR = "#2675A7"
MAX_COLOR = "#D87537"
INK = "#26333D"


def _load_windows(root: Path, entry: dict) -> list[dict]:
    study = Path(entry["study"])
    if study.parts[0] != "experiments":
        study = Path("experiments") / study
    source = root / study / "runs" / entry["run_id"] / "metrics.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    windows = sorted(
        (row for row in rows if "train_gradient_norm_count" in row),
        key=lambda row: row["step"],
    )
    if not windows:
        raise ValueError(f"No gradient windows in {source}")
    previous_end = 0
    for row in windows:
        start, end = int(row["training_window_start_step"]), int(row["step"])
        count = int(row["train_gradient_norm_count"])
        if start != previous_end or end <= start or count != end - start:
            raise ValueError(f"Gradient coverage gap/overlap at {entry['run_id']}:{end}")
        previous_end = end
        mean, maximum = float(row["train_gradient_norm_mean"]), float(row["train_gradient_norm_max"])
        fraction = float(row["train_gradient_clipped_fraction"])
        clipped = fraction * count
        if not (math.isfinite(mean) and math.isfinite(maximum) and 0 < mean <= maximum):
            raise ValueError(f"Invalid gradient norm at {entry['run_id']}:{end}")
        if not (0 <= fraction <= 1 and math.isclose(clipped, round(clipped), abs_tol=1e-6)):
            raise ValueError(f"Nonintegral clipping count at {entry['run_id']}:{end}")
        if int(row["attention_degree"]) != int(entry["degree"]):
            raise ValueError(f"Degree mismatch at {entry['run_id']}:{end}")
        for spike in row.get("train_gradient_spikes", []):
            if not (start < int(spike["step"]) <= end):
                raise ValueError(f"Spike outside its reporting window at {entry['run_id']}:{end}")
            if float(spike["gradient_norm"]) > maximum * (1 + 1e-6):
                raise ValueError(f"Spike exceeds window maximum at {entry['run_id']}:{end}")
    if POST_STEP not in [int(row["step"]) for row in windows]:
        raise ValueError(f"Post-{POST_STEP} summary requires an exact window boundary")
    if len({float(row["gradient_clip_norm"]) for row in windows}) != 1:
        raise ValueError("Changing gradient clipping thresholds need a different plot")
    if float(windows[0]["gradient_clip_norm"]) != 1:
        raise ValueError("This diagnostic expects the selected runs' clipping threshold of 1")
    record = json.loads(source.with_name("record.json").read_text())
    if record["status"] != "complete" or int(record["step"]) != previous_end:
        raise ValueError(f"Gradient windows do not cover completed training in {entry['run_id']}")
    for row in windows:
        rule = row["gradient_spike_rule"]
        if (rule["factor"], rule["history"], rule["minimum_history"]) != (5, 100, 20):
            raise ValueError("Unexpected spike detector settings")
    return windows


def _aggregate(windows: list[dict]) -> dict:
    count = sum(int(row["train_gradient_norm_count"]) for row in windows)
    clipped = sum(round(row["train_gradient_clipped_fraction"] * row["train_gradient_norm_count"]) for row in windows)
    maximum_row = max(windows, key=lambda row: row["train_gradient_norm_max"])
    return {
        "updates": count,
        "clipped_updates": clipped,
        "clipped_fraction": clipped / count,
        "gradient_mean": sum(row["train_gradient_norm_mean"] * row["train_gradient_norm_count"] for row in windows) / count,
        "gradient_max": maximum_row["train_gradient_norm_max"],
        "gradient_max_window_start_exclusive": maximum_row["training_window_start_step"],
        "gradient_max_window_end_inclusive": maximum_row["step"],
        "detected_spikes": sum(int(row["train_gradient_spike_count"]) for row in windows),
    }


def plot(root: Path, out: Path, selection: dict) -> list[dict]:
    """Save the gradient figure and summary; caller verifies selected source hashes."""
    entries = selection["runs"]
    if not entries:
        raise ValueError("Select at least one diagnostic run")
    loaded = [(entry, _load_windows(root, entry)) for entry in entries]
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    plot_settings = {
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
        "axes.labelsize": 11, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#BBC3CA",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(plot_settings):
        fig, axes = plt.subplots(
            2, len(loaded), figsize=(6.5 * len(loaded), 7.3), squeeze=False,
            sharex=True, sharey="row", gridspec_kw={"height_ratios": [3.1, 1.25]},
        )
        for col, (entry, windows) in enumerate(loaded):
            top, bottom = axes[:, col]
            starts = np.asarray([row["training_window_start_step"] for row in windows])
            ends = np.asarray([row["step"] for row in windows])
            centers = (starts + ends) / 2
            means = np.asarray([row["train_gradient_norm_mean"] for row in windows])
            maxima = np.asarray([row["train_gradient_norm_max"] for row in windows])
            fractions = np.asarray([row["train_gradient_clipped_fraction"] for row in windows])
            clip_norm = float(windows[0]["gradient_clip_norm"])
            top.hlines(means, starts, ends, color=MEAN_COLOR, linewidth=2.5, zorder=3)
            top.scatter(centers, means, color=MEAN_COLOR, s=15, zorder=4)
            top.hlines(maxima, starts, ends, color=MAX_COLOR, linewidth=1.6, zorder=3)
            top.axhline(clip_norm, color="#8A929B", linestyle="--", linewidth=1.3)
            top.set_yscale("log")
            top.set_ylim(0.09, 20)
            top.set_title(f"SoS polynomial · $m={entry['degree']}$", pad=13, weight="bold")
            top.grid(axis="y", color="#DFE4E8", linewidth=0.8)
            top.set_axisbelow(True)
            spikes = sorted(
                (spike for row in windows for spike in row.get("train_gradient_spikes", [])),
                key=lambda spike: spike["step"],
            )
            if spikes:
                top.scatter([s["step"] for s in spikes], [s["gradient_norm"] for s in spikes],
                            marker="D", color=INK, edgecolor="white", linewidth=0.6, s=42, zorder=5)
                # The exact update is known only for recorded spikes, not window maxima.
                largest = max(spikes, key=lambda spike: spike["gradient_norm"])
                top.annotate(
                    f"Step {largest['step']:,}: {largest['gradient_norm']:.2f}",
                    xy=(largest["step"], largest["gradient_norm"]), xytext=(-12, 25),
                    textcoords="offset points", ha="right", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
                )
            bottom.bar(starts, 100 * fractions, width=ends - starts, align="edge",
                       color=MEAN_COLOR, alpha=0.82, edgecolor="white", linewidth=0.4)
            # Tiny nonzero percentages would otherwise be invisible on the percent axis.
            post = [row for row in windows if row["training_window_start_step"] >= POST_STEP]
            all_stats, post_stats = _aggregate(windows), _aggregate(post)
            bottom.text(0.98, 0.92,
                        f"After step {POST_STEP:,}: {post_stats['clipped_updates']:,} / {post_stats['updates']:,} clipped\n"
                        f"({100 * post_stats['clipped_fraction']:.3f}%)",
                        transform=bottom.transAxes, ha="right", va="top", fontsize=10,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
            bottom.set_ylim(0, 105)
            bottom.set_yticks([0, 50, 100])
            bottom.grid(axis="y", color="#DFE4E8", linewidth=0.8)
            bottom.set_axisbelow(True)
            bottom.set_xlabel("Training update")
            bottom.xaxis.set_major_locator(MultipleLocator(1500))
            bottom.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            summary = {
                "run_id": entry["run_id"], "degree": entry["degree"],
                "reporting_windows": len(windows), "gradient_clip_norm": clip_norm,
                **{f"all_{key}": value for key, value in all_stats.items()},
                **{f"post_{POST_STEP}_{key}": value for key, value in post_stats.items()},
                "final_window_start_exclusive": int(starts[-1]),
                "final_window_end_inclusive": int(ends[-1]),
                "final_window_gradient_mean": float(means[-1]),
                "final_window_gradient_max": float(maxima[-1]),
                "recorded_spike_steps": ";".join(str(s["step"]) for s in spikes),
                "recorded_spike_gradient_norms": ";".join(format(s["gradient_norm"], ".12g") for s in spikes),
            }
            summaries.append(summary)
        max_step = max(windows[-1]["step"] for _, windows in loaded)
        for ax in axes.flat:
            ax.set_xlim(0, max_step + 70)
        axes[0, 0].set_ylabel("Global gradient norm before clipping\n(log scale)")
        axes[1, 0].set_ylabel("Updates clipped (%)")
        fig.suptitle("Transformer++ · SoS training dynamics", fontsize=20, weight="bold", y=0.98)
        handles = [
            Line2D([], [], color=MEAN_COLOR, lw=2.5, marker="o", markersize=4, label="Window mean"),
            Line2D([], [], color=MAX_COLOR, lw=1.6, label="Window maximum (time within span unknown)"),
            Line2D([], [], color=INK, marker="D", linestyle="none", markersize=5, label="Recorded spike at exact update"),
            Line2D([], [], color="#8A929B", lw=1.3, linestyle="--", label="Clipping threshold: 1"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.931),
                   ncol=2, frameon=False, fontsize=10, columnspacing=2.5)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.145, hspace=0.2, wspace=0.12)
        fig.text(0.075, 0.065,
                 "Horizontal spans and clipping bars cover every update: 100-update windows through 1,500, then 500 (final window: 70).",
                 fontsize=9.5, color="#67717D")
        fig.text(0.075, 0.035,
                 "Recorded spikes exceed 5× the preceding 100-update median, after at least 20 history updates. Window maxima need no spike threshold.",
                 fontsize=9.5, color="#67717D")
        fig.savefig(out / "gradient_norms.png", dpi=180, facecolor="white")
        fig.savefig(out / "gradient_norms.pdf", facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    with (out / "gradient_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries
