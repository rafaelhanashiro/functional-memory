"""Regenerate the per-key weight figures and tables from the position-weights runs in this topic's selection."""
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
FAR, BLOCK = 2048, 64
# Key-distance bands of the tables, [low, high).
BANDS = ((0, 16), (16, 64), (64, 256), (256, 1024), (1024, 2048), (2048, 4096))
# The two key stretches where the 2k model's far weight peaks, [low, high) in key positions.
STRETCHES = ((896, 1024), (1664, 1792))
SINGLE_NOTE = ("Query + W decay, m = 4; one 4,096-token validation row. Each bar is one key's share of the query's |w|, "
               "averaged over the 14 heads,\n÷ the uniform share 1/4,096; the axis stops at 8×. Black line: mean over each "
               "block of 64 keys. Dashed line: uniform.\nGray: keys ≥ 2,048 back, distances the 2,048-token model never "
               "saw. ▲ end-of-document token.")
BATCH_NOTE = ("Query + W decay, m = 4; the first 128 validation sequences of 4,096 tokens. Each bar is one key's share of "
              "the query's |w|, averaged over the 14 heads\nand the 128 sequences, ÷ the uniform share 1/4,096; the axis "
              "stops at {ceiling:.0f}×. Black line: mean over each block of 64 keys. Dashed line: uniform.\nGray: keys "
              "≥ 2,048 back, distances the 2,048-token model never saw. In parentheses: 10th–90th percentile of that "
              "share across sequences.")
BATCH_CEILING = 8.0


def metrics(role: str) -> dict[tuple[str, int], dict]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    entry = next(entry for entry in selection["runs"] if entry["role"] == role)
    verify_pinned_runs(ROOT, [entry])
    path = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] / "metrics.jsonl"
    return {(row["model"], row["layer"]): row for row in map(json.loads, path.read_text().splitlines())}


def per_key(share: list[float]) -> np.ndarray:
    """Each key's |w| share ÷ the uniform share 1/(number of keys), in key-position order."""
    share = np.asarray(share, dtype=np.float64)
    return share * share.size


def write_table(path: Path, data: dict, field: str, extra: dict) -> None:
    """One row per model and layer; `extra` maps column names to functions of the metrics row and per-key weights."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "layer", "share_2048_or_more_back", "weight_key_0",
                         *[f"weight_distance_{low}_{high - 1}" for low, high in BANDS],
                         "far_peak_block_keys", "far_peak_block_weight", *extra])
        for model, _, _ in MODELS:
            for layer in sorted(layer for name, layer in data if name == model):
                row = data[(model, layer)]
                ratio = per_key(row[field])
                distance = ratio.size - 1 - np.arange(ratio.size)
                # The highest 64-key block (the figure's black line) among keys >= 2,048 back.
                blocks = ratio[:ratio.size - FAR].reshape(-1, BLOCK).mean(axis=1)
                peak = int(np.argmax(blocks))
                writer.writerow([model, layer + 1, f"{ratio[distance >= FAR].sum() / ratio.size:.4f}", f"{ratio[0]:.2f}",
                                 *[f"{ratio[(distance >= low) & (distance < high)].mean():.4f}" for low, high in BANDS],
                                 f"{peak * BLOCK}-{(peak + 1) * BLOCK - 1}", f"{blocks[peak]:.2f}",
                                 *[column(row, ratio) for column in extra.values()]])


def draw(name: str, data: dict, field: str, *, title: str, note: str, ceiling: float, yticks: list[float],
         spread: bool) -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "functional-memory",
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(len(LAYERS), 2, figsize=(11, 2.25 * len(LAYERS) + 1.3), sharex=True, sharey=True)
        for row_axes, layer in zip(axes, LAYERS):
            for axis, (model, column_title, color) in zip(row_axes, MODELS):
                row = data[(model, layer)]
                ratio = per_key(row[field])
                keys = ratio.size
                axis.axvspan(0, keys - FAR, color=SHADE, linewidth=0, zorder=0)
                axis.stairs(ratio, np.arange(keys + 1), fill=True, color=color, linewidth=0, zorder=2)
                axis.stairs(ratio.reshape(-1, BLOCK).mean(axis=1), np.arange(0, keys + 1, BLOCK), color=INK,
                            linewidth=1.3, zorder=3)
                axis.axhline(1, color=MUTED, linewidth=1, linestyle="--", zorder=3)
                if not spread:
                    axis.plot(row["eos_positions"], np.zeros(len(row["eos_positions"])), linestyle="none", marker="^",
                              markersize=5, color=INK, clip_on=False, zorder=4)
                clipped = ratio > ceiling
                far = 100 * ratio[:keys - FAR].sum() / keys
                share = f"{far:.0f}% of weight ≥ 2,048 back"
                if spread:
                    low, high = 100 * np.percentile(row["share_beyond_2048_by_row"], [10, 90])
                    share += f" ({low:.0f}–{high:.0f}%)"
                count = int(clipped.sum())
                axis.set_title(f"{share} · {count} {'key' if count == 1 else 'keys'} above {ceiling:.0f}× "
                               f"(max {ratio.max():,.0f}×)",
                               fontsize=8, color=MUTED, loc="right")
                if layer == LAYERS[0]:
                    axis.text(0, 1.2, column_title, transform=axis.transAxes, fontsize=10.5, color=INK, weight="bold")
                axis.set_ylim(0, ceiling)
                axis.set_xlim(0, keys)
                axis.set_yticks(yticks)
                axis.tick_params(labelsize=8.5, colors=INK)
                axis.grid(True, axis="y", color=GRID, linewidth=0.8)
                axis.set_axisbelow(True)
                for spine in ("top", "right"):
                    axis.spines[spine].set_visible(False)
            row_axes[0].set_ylabel(f"Layer {layer + 1}\n|w| ÷ uniform", fontsize=9.5, color=INK)
        for axis in axes[-1]:
            axis.set_xticks([0, 1024, 2048, 3072, 4096], ["0", "1k", "2k", "3k", "4k"])
            axis.set_xlabel("Key position (the query is at 4,095)", fontsize=9.5, color=INK, labelpad=8)
        figure.suptitle(title, x=0.07, ha="left", fontsize=13, weight="bold", color=INK)
        figure.text(0.07, 0.948, note, fontsize=8.5, color=MUTED, va="top", linespacing=1.4)
        figure.subplots_adjust(top=0.835, left=0.08, right=0.985, bottom=0.07, hspace=0.32, wspace=0.06)
        figure.savefig(TOPIC / f"{name}.png", dpi=180, metadata={"Software": "functional-memory"})
        figure.savefig(TOPIC / f"{name}.pdf", metadata={"Creator": "functional-memory", "Producer": None,
                                                         "CreationDate": None})
        plt.close(figure)


def main() -> None:
    single = metrics("position_weights")
    write_table(TOPIC / "position_weights.csv", single, "head_mean_share", {
        "weight_end_of_document_keys": lambda row, ratio: " ".join(f"{key}:{ratio[key]:.2f}" for key in row["eos_positions"])})
    draw("position_weights", single, "head_mean_share", title="Where the query at position 4,095 puts its weight",
         note=SINGLE_NOTE, ceiling=8.0, yticks=[0, 1, 2, 4, 6, 8], spread=False)
    batch = metrics("position_weights_batch")
    write_table(TOPIC / "position_weights_batch.csv", batch, "mean_share", {
        "share_2048_or_more_back_p10": lambda row, _: f"{np.percentile(row['share_beyond_2048_by_row'], 10):.4f}",
        "share_2048_or_more_back_p90": lambda row, _: f"{np.percentile(row['share_beyond_2048_by_row'], 90):.4f}",
        "stretch_peak_block_weight": lambda _, ratio: f"{max(ratio[low:high].reshape(-1, BLOCK).mean(axis=1).max() for low, high in STRETCHES):.4f}"})
    draw("position_weights_batch", batch, "mean_share",
         title="Where the query at position 4,095 puts its weight, over 128 sequences",
         note=BATCH_NOTE.format(ceiling=BATCH_CEILING), ceiling=BATCH_CEILING, yticks=[0, 1, 2, 4, 6, 8], spread=True)


if __name__ == "__main__":
    main()
