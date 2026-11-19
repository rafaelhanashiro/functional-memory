"""Regenerate the extrapolation-ablation figure and table from this topic's explicit selection."""
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

TOPIC = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
INK, MUTED, GRID = "#172334", "#5b6573", "#e4e7eb"
BLOCKS = ((0, 2048), (0, 1024), (1024, 2048), (2048, 2304), (2304, 3072), (3072, 4096), (4096, 8192), (8192, 12288), (12288, 16384))
# Categorical slots in fixed order: blue, orange, aqua, yellow, magenta.
SERIES = (("query, original attention", "#2a78d6", "-"), ("query, 2,048-key window", "#eb6834", "-"),
          ("query, RoPE clamped at 2,047", "#1baf7a", "-"), ("exp (reference)", "#eda100", "--"),
          ("FoX no RoPE + W decay (reference)", "#e87ba4", "--"))


def curves() -> dict[str, np.ndarray]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    entries = [entry for entry in selection["runs"] if entry["role"] in ("ablation_evaluation", "original_16k_evaluation")]
    verify_pinned_runs(ROOT, entries)
    runs = {entry["role"]: ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] for entry in entries}
    ablation = {(row["mode"], row["length"]): np.asarray(row["per_position_nll"], dtype=np.float64)
                for row in map(json.loads, (runs["ablation_evaluation"] / "metrics.jsonl").read_text().splitlines())}
    original = {row["role"]: np.asarray(row["per_position_nll"], dtype=np.float64)
                for row in map(json.loads, (runs["original_16k_evaluation"] / "metrics.jsonl").read_text().splitlines())
                if row["length"] == "evaluation"}
    return dict(zip([name for name, _, _ in SERIES],
                    (ablation[("full", "evaluation")], ablation[("window", "evaluation")], ablation[("clamp", "evaluation")],
                     original["exp"], original["forget_gate_no_rope_wdecay_m4"])))


def main() -> None:
    data = curves()
    with (TOPIC / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series", *[f"ppl_positions_{a + 1}_{b}" for a, b in BLOCKS], "ppl_positions_1_16384"])
        for name, curve in data.items():
            writer.writerow([name, *[f"{math.exp(curve[a:b].mean()):.4f}" for a, b in BLOCKS], f"{math.exp(curve.mean()):.4f}"])
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "functional-memory",
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure, axis = plt.subplots(figsize=(8.4, 4.9))
        positions = np.arange(1, 16385)
        bins = positions.reshape(-1, 64).mean(axis=1)
        for name, color, style in SERIES:
            binned = np.exp(data[name].reshape(-1, 64).mean(axis=1))
            axis.plot(bins, binned, color=color, linestyle=style, linewidth=2, label=name)
        axis.axvline(2048, color=MUTED, linewidth=1, linestyle=":")
        axis.text(2048, 1.02, " training length", transform=axis.get_xaxis_transform(), fontsize=8, color=MUTED,
                  va="bottom")
        axis.set_yscale("log")
        axis.set_xlim(0, 16384)
        axis.set_xticks([0, 2048, 4096, 8192, 12288, 16384], ["0", "2k", "4k", "8k", "12k", "16k"], fontsize=9, color=INK)
        axis.tick_params(axis="y", labelsize=9, colors=INK)
        axis.set_xlabel("Position in the 16,384-token row", fontsize=10, color=INK)
        axis.set_ylabel("Perplexity (64-position bins, log scale)", fontsize=10, color=INK)
        axis.grid(True, which="major", color=GRID, linewidth=0.8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.legend(frameon=False, fontsize=8.5, loc="upper left")
        figure.suptitle("Query attention past its training length", x=0.08, ha="left", fontsize=13, weight="bold", color=INK)
        figure.text(0.08, 0.905, "2,048-token checkpoints scored at 16,384 tokens; only the attention call changes.",
                    fontsize=8.5, color=MUTED)
        figure.subplots_adjust(top=0.85, left=0.09, right=0.98, bottom=0.12)
        figure.savefig(TOPIC / "per_position.png", dpi=180, metadata={"Software": "functional-memory"})
        figure.savefig(TOPIC / "per_position.pdf", metadata={"Creator": "functional-memory", "Producer": None,
                                                              "CreationDate": None})
        plt.close(figure)


if __name__ == "__main__":
    main()
