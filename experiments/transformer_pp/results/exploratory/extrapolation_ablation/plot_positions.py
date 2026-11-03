"""Regenerate the per-key weight figure and table from the position-weights run in this topic's selection."""
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

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]
INK, MUTED, GRID, SHADE = "#172334", "#5b6573", "#e4e7eb", "#eef0f3"
# (model, column title, color): the entity colors of the weight-profile figure.
MODELS = (("query_2k", "Trained at 2,048 tokens", "#eb6834"), ("query_4k", "Trained at 4,096 tokens", "#e87ba4"))
LAYERS = (0, 4, 9, 14)
FAR, CEILING, BLOCK = 2048, 8.0, 64
# Key-distance bands of the table, [low, high).
BANDS = ((0, 16), (16, 64), (64, 256), (256, 1024), (1024, 2048), (2048, 4096))


def weights() -> dict[tuple[str, int], dict]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    entry = next(entry for entry in selection["runs"] if entry["role"] == "position_weights")
    verify_pinned_runs(ROOT, [entry])
    path = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] / "metrics.jsonl"
    return {(row["model"], row["layer"]): row for row in map(json.loads, path.read_text().splitlines())}


def per_key(row: dict) -> np.ndarray:
    """Each key's |w| ÷ the uniform weight 1/(number of keys), in key-position order."""
    share = np.asarray(row["head_mean_share"], dtype=np.float64)
    return share * share.size


def main() -> None:
    data = weights()
    with (TOPIC / "position_weights.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "layer", "share_2048_or_more_back", "weight_key_0",
                         *[f"weight_distance_{low}_{high - 1}" for low, high in BANDS],
                         "far_peak_block_keys", "far_peak_block_weight", "weight_end_of_document_keys"])
        for model, _, _ in MODELS:
            for layer in sorted(layer for name, layer in data if name == model):
                ratio = per_key(data[(model, layer)])
                distance = ratio.size - 1 - np.arange(ratio.size)
                # The highest 64-key block (the figure's black line) among keys >= 2,048 back.
                blocks = ratio[:ratio.size - FAR].reshape(-1, BLOCK).mean(axis=1)
                peak = int(np.argmax(blocks))
                writer.writerow([model, layer + 1, f"{ratio[distance >= FAR].sum() / ratio.size:.4f}", f"{ratio[0]:.2f}",
                                 *[f"{ratio[(distance >= low) & (distance < high)].mean():.4f}" for low, high in BANDS],
                                 f"{peak * BLOCK}-{(peak + 1) * BLOCK - 1}", f"{blocks[peak]:.2f}",
                                 " ".join(f"{key}:{ratio[key]:.2f}" for key in data[(model, layer)]["eos_positions"])])
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "functional-memory",
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(len(LAYERS), 2, figsize=(11, 2.25 * len(LAYERS) + 1.3), sharex=True, sharey=True)
        for row_axes, layer in zip(axes, LAYERS):
            for axis, (model, title, color) in zip(row_axes, MODELS):
                row = data[(model, layer)]
                ratio = per_key(row)
                keys = ratio.size
                axis.axvspan(0, keys - FAR, color=SHADE, linewidth=0, zorder=0)
                axis.stairs(ratio, np.arange(keys + 1), fill=True, color=color, linewidth=0, zorder=2)
                axis.stairs(ratio.reshape(-1, BLOCK).mean(axis=1), np.arange(0, keys + 1, BLOCK), color=INK,
                            linewidth=1.3, zorder=3)
                axis.axhline(1, color=MUTED, linewidth=1, linestyle="--", zorder=3)
                axis.plot(row["eos_positions"], np.zeros(len(row["eos_positions"])), linestyle="none", marker="^",
                          markersize=5, color=INK, clip_on=False, zorder=4)
                clipped = ratio > CEILING
                far = 100 * ratio[:keys - FAR].sum() / keys
                axis.set_title(f"{far:.0f}% of weight ≥ 2,048 back · {clipped.sum()} keys above {CEILING:.0f}× "
                               f"(max {ratio.max():,.0f}×)", fontsize=8, color=MUTED, loc="right")
                if layer == LAYERS[0]:
                    axis.text(0, 1.2, title, transform=axis.transAxes, fontsize=10.5, color=INK, weight="bold")
                axis.set_ylim(0, CEILING)
                axis.set_xlim(0, keys)
                axis.set_yticks([0, 1, 2, 4, 6, 8])
                axis.tick_params(labelsize=8.5, colors=INK)
                axis.grid(True, axis="y", color=GRID, linewidth=0.8)
                axis.set_axisbelow(True)
                for spine in ("top", "right"):
                    axis.spines[spine].set_visible(False)
            row_axes[0].set_ylabel(f"Layer {layer + 1}\n|w| ÷ uniform", fontsize=9.5, color=INK)
        for axis in axes[-1]:
            axis.set_xticks([0, 1024, 2048, 3072, 4096], ["0", "1k", "2k", "3k", "4k"])
            axis.set_xlabel("Key position (the query is at 4,095)", fontsize=9.5, color=INK, labelpad=8)
        figure.suptitle("Where the query at position 4,095 puts its weight", x=0.07, ha="left", fontsize=13,
                        weight="bold", color=INK)
        figure.text(0.07, 0.948, "Query + W decay, m = 4; one 4,096-token validation row. Each bar is one key's share "
                    "of the query's |w|, averaged over the 14 heads,\n÷ the uniform share 1/4,096; the axis stops at 8×. "
                    "Black line: mean over each block of 64 keys. Dashed line: uniform.\nGray: keys ≥ 2,048 back, "
                    "distances the 2,048-token model never saw. ▲ end-of-document token.", fontsize=8.5, color=MUTED,
                    va="top", linespacing=1.4)
        figure.subplots_adjust(top=0.835, left=0.08, right=0.985, bottom=0.07, hspace=0.32, wspace=0.06)
        figure.savefig(TOPIC / "position_weights.png", dpi=180, metadata={"Software": "functional-memory"})
        figure.savefig(TOPIC / "position_weights.pdf", metadata={"Creator": "functional-memory", "Producer": None,
                                                                  "CreationDate": None})
        plt.close(figure)


if __name__ == "__main__":
    main()
