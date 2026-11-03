"""Regenerate the weight-by-distance figure and table from the weight-profile run in this topic's selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]
INK, MUTED, GRID = "#172334", "#5b6573", "#e4e7eb"
# (case, query window end, label, color) in fixed categorical order: blue, orange, aqua, yellow, magenta.
SERIES = (("2k_model_at_2k", 2048, "trained at 2k, queries near 2k", "#2a78d6"),
          ("2k_model_at_4k", 4096, "trained at 2k, queries near 4k", "#eb6834"),
          ("2k_model_at_16k", 8192, "trained at 2k, queries near 8k", "#1baf7a"),
          ("2k_model_at_16k", 16384, "trained at 2k, queries near 16k", "#eda100"),
          ("4k_model_at_4k", 4096, "trained at 4k, queries near 4k", "#e87ba4"))


def profiles() -> tuple[list[int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    entry = next(entry for entry in selection["runs"] if entry["role"] == "weight_profile")
    path = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    edges, result = rows[0]["bin_edges"], {}
    for case, end, label, _ in SERIES:
        layers = [row for row in rows if row["case"] == case and row["query_end"] == end]
        share = np.mean([np.mean(row["weight_share_by_head"], axis=0) for row in layers], axis=0)
        result[label] = (share, np.asarray(layers[0]["key_share"]))
    return edges, result


def main() -> None:
    edges, data = profiles()
    with (TOPIC / "weight_profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series", "distance_bin", "weight_share", "key_share", "weight_per_key_over_uniform"])
        for label, (share, keys) in data.items():
            for (low, high), s, k in zip(zip(edges[:-1], edges[1:]), share, keys):
                if k > 0:
                    writer.writerow([label, f"{low}-{high}", f"{s:.4f}", f"{k:.4f}", f"{s / k:.3f}"])
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "functional-memory",
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.6))
        for (case, end, label, color) in SERIES:
            share, keys = data[label]
            present = keys > 0
            centers = np.sqrt(np.maximum(np.asarray(edges[:-1]), 1) * np.asarray(edges[1:]))[present]
            left.plot(centers, share[present] / keys[present], marker="o", markersize=5, linewidth=2, color=color, label=label)
            right.plot(centers, share[present] * 100, marker="o", markersize=5, linewidth=2, color=color, label=label)
        left.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        left.text(10, 1.08, "uniform", fontsize=8, color=MUTED)
        left.set_yscale("log")
        left.set_ylabel("Weight per key ÷ uniform (log)", fontsize=10, color=INK)
        right.set_ylabel("Share of the query's |weight| (%)", fontsize=10, color=INK)
        for axis in (left, right):
            axis.set_xscale("log")
            ticks = np.sqrt(np.maximum(np.asarray(edges[:-1]), 1) * np.asarray(edges[1:]))
            axis.set_xticks(ticks, ["0–16", "16–64", "64–256", "256–512", "0.5–1k", "1–2k", "2–4k", "4–8k", "8–16k"],
                            rotation=45, fontsize=8, color=INK)
            axis.minorticks_off()
            axis.set_xlabel("Key distance", fontsize=10, color=INK)
            axis.tick_params(axis="y", labelsize=9, colors=INK)
            axis.grid(True, which="major", color=GRID, linewidth=0.8)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
        right.legend(frameon=False, fontsize=8, loc="upper left")
        figure.suptitle("Where query attention puts its weight", x=0.06, ha="left", fontsize=13, weight="bold", color=INK)
        figure.text(0.06, 0.9, "Query + W decay, m = 4; last 256 queries before each position, all layers and heads.",
                    fontsize=8.5, color=MUTED)
        figure.subplots_adjust(top=0.84, left=0.07, right=0.98, bottom=0.2, wspace=0.25)
        figure.savefig(TOPIC / "weight_profile.png", dpi=180, metadata={"Software": "functional-memory"})
        figure.savefig(TOPIC / "weight_profile.pdf", metadata={"Creator": "functional-memory", "Producer": None,
                                                                "CreationDate": None})
        plt.close(figure)


if __name__ == "__main__":
    main()
