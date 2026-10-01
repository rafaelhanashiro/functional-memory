"""Gated DeltaNet state ranks: pretrained against random models under two initializations.

Each head is one curve at the selected layer, averaged over inputs (and seeds for
random models), as in the canonical state-rank figure. The per-layer summary gives
final-token values averaged over heads and inputs, with seed ranges. The pinned
numerical archives are ignored by Git and live on Kennel.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from results.state_rank.plot import GRID, HEAD_COLORS, INK, METRICS, MUTED, archives, head_curves

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parents[2]
FIELDS = ["layer", "condition", "participation_ratio", "participation_ratio_seed_min",
          "participation_ratio_seed_max", "threshold_rank", "threshold_rank_seed_min", "threshold_rank_seed_max"]


def summary(selection: dict, data: dict) -> list[dict]:
    rows = []
    layers = data[selection["analysis"]["conditions"][0]["roles"][0]]["layers"]
    for index, layer in enumerate(layers):
        for condition in selection["analysis"]["conditions"]:
            values = {metric: [float(data[role][metric][index, :, -1, :].mean()) for role in condition["roles"]]
                      for metric in METRICS}
            row = {"layer": int(layer), "condition": condition["key"]}
            for metric, seeds in values.items():
                row[metric] = f"{np.mean(seeds):.6g}"
                if len(seeds) > 1:
                    row.update({f"{metric}_seed_min": f"{min(seeds):.6g}", f"{metric}_seed_max": f"{max(seeds):.6g}"})
            rows.append({field: row.get(field, "") for field in FIELDS})
    return rows


def figure(selection: dict, data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = selection["analysis"]
    layer, conditions = analysis["layer"], analysis["conditions"]
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "path.simplify": False})
    fig, axes = plt.subplots(len(METRICS), len(conditions), figsize=(3.2 * len(conditions), 5.6), sharex=True)
    prefixes = data[conditions[0]["roles"][0]]["prefixes"]
    for column, condition in enumerate(conditions):
        seeds = len(condition["roles"])
        title = f"{condition['label']}" + (f"\n({seeds} seeds)" if seeds > 1 else "\n")
        for row, metric in enumerate(METRICS):
            axis = axes[row, column]
            curves = np.mean([head_curves(data[role], layer, metric) for role in condition["roles"]], axis=0)
            for head in range(curves.shape[1]):
                axis.plot(prefixes, curves[:, head], color=HEAD_COLORS[head], linewidth=1.5, label=f"Head {head}")
            if metric == "threshold_rank":
                axis.plot(prefixes, np.minimum(prefixes, analysis["state_width"]), color=MUTED, linewidth=1,
                          linestyle="--", label="Rank ceiling")
            axis.grid(True, color=GRID, linewidth=0.8)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
            if row == 0:
                axis.set_title(title, color=INK, fontsize=9)
            if row == len(METRICS) - 1:
                axis.set_xlabel("Prefix length (tokens)", color=INK, fontsize=9)
            if column == 0:
                axis.set_ylabel(METRICS[metric], color=INK, fontsize=9)
    for row in range(len(METRICS)):
        top = max(axis.get_ylim()[1] for axis in axes[row])
        for axis in axes[row]:
            axis.set_ylim(0, top)
    fig.suptitle(f"{analysis['model']}, layer {layer}", color=INK, fontsize=10)
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"state_rank_init.{suffix}", dpi=180,
                    metadata={"CreationDate": None} if suffix == "pdf" else None)
    plt.close(fig)


def main() -> None:
    selection = yaml.safe_load((RESULTS / "selection.yaml").read_text())
    data = archives(selection, ROOT)
    figure(selection, data, RESULTS)
    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary(selection, data))


if __name__ == "__main__":
    main()
