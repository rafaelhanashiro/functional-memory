"""Regenerate the long-document per-position figures and table from this topic's selection.

Three figures share one style: the 16k extrapolation of exp, query + W decay and FoX no RoPE +
W decay on a log axis; the same curves on a linear 10-30 perplexity axis; and the log-axis
figure with the query model's 2,048-key window and RoPE-clamp passes added. Every perplexity
is exp(mean NLL), including the 64-position bins and the blocks of the table.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]
BIN, TRAINED, LENGTH = 64, 2048, 16384
BLOCKS = ((0, 2048), (2048, 2304), (2304, 3072), (3072, 4096), (4096, 8192), (8192, 12288), (12288, 16384))
# (label, color, dashes): the context-length figure's colors; the two ablations add black and dark magenta.
SERIES = {
    "exp": ("exp", "#1f77b4", None),
    "query_wdecay_m4": ("query + W decay", "#d62728", None),
    "forget_gate_no_rope_wdecay_m4": ("FoX no RoPE + W decay", "#2ca02c", None),
    "window": ("query + W decay, 2,048-key sliding window", "#000000", (6, 2.5)),
    "clamp": ("query + W decay, RoPE clamped at 2,047", "#8b008b", (2.5, 1.5)),
}
BASELINE = ("exp", "query_wdecay_m4", "forget_gate_no_rope_wdecay_m4")


def curves() -> tuple[dict[str, np.ndarray], int]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    entries = {entry["role"]: entry for entry in selection["runs"]
               if entry["role"] in ("long_documents_evaluation", "long_documents_ablation")}
    verify_pinned_runs(ROOT, list(entries.values()))
    runs = {role: ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] for role, entry in entries.items()}
    data, sequences = {}, set()
    for row in map(json.loads, (runs["long_documents_evaluation"] / "metrics.jsonl").read_text().splitlines()):
        if row["length"] == "evaluation":
            data[row["role"]] = row
    for row in map(json.loads, (runs["long_documents_ablation"] / "metrics.jsonl").read_text().splitlines()):
        if row["length"] == "evaluation" and row["mode"] in ("window", "clamp"):
            data[row["mode"]] = row
    result = {}
    for key in SERIES:
        row = data[key]
        values = np.asarray(row["per_position_nll"], dtype=np.float64)
        if values.shape != (LENGTH,) or not np.isfinite(values).all() or abs(values.mean() - row["nll"]) > 1e-10:
            raise ValueError(f"Invalid per-position measurements for {key}")
        sequences.add((row["sequences"], row.get("start_sequence", 0)))
        result[key] = values
    if len(sequences) != 1:
        raise ValueError("Every curve must score the same long documents")
    return result, sequences.pop()[0]


def draw(name: str, data: dict[str, np.ndarray], keys: tuple[str, ...], sequences: int, *, zoom: bool,
         legend: str) -> None:
    positions = np.arange(1, LENGTH + 1)
    centers = np.arange(LENGTH // BIN) * BIN + (BIN + 1) / 2
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "svg.hashsalt": "functional-memory"}):
        figure, axis = plt.subplots(figsize=(13.5, 6.3), layout="constrained")
        for key in keys:
            axis.plot(positions, np.exp(data[key]), color=SERIES[key][1], lw=0.4, alpha=0.18)
        for key in keys:
            label, color, dashes = SERIES[key]
            line, = axis.plot(centers, np.exp(data[key].reshape(-1, BIN).mean(axis=1)), color=color, lw=2.4,
                              label=f"{label} (overall PPL {math.exp(data[key].mean()):.3f})")
            if dashes:
                line.set_dashes(dashes)
        axis.axvline(TRAINED, color="#555555", ls="--", lw=1.1)
        axis.grid(which="both", color="#cccccc", alpha=0.55)
        axis.set_axisbelow(True)
        offset, align = ((-6, "right") if zoom else (6, "left"))
        axis.annotate("training length", xy=(TRAINED, 1), xycoords=("data", "axes fraction"), xytext=(offset, -7),
                      textcoords="offset points", ha=align, va="top", color="#555555", fontsize=10)
        axis.set_title(f"Per-position perplexity on {sequences:,} long documents none of the models trained on\n"
                       f"2,048-token checkpoints · one document per row · bold: {BIN}-position bins · "
                       "light: every position", fontsize=14)
        if zoom:
            axis.set_ylim(10, 30)
        else:
            axis.set_yscale("log")
        axis.set_ylabel("PPL(t) = exp(mean NLL at position t)")
        axis.set_xlabel("Token position in the 16,384-token document")
        axis.set_xlim(1, LENGTH)
        axis.set_xticks(np.arange(TRAINED, LENGTH + 1, TRAINED))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.legend(loc=legend, fontsize=11, framealpha=0.95)
        figure.savefig(TOPIC / f"{name}.png", dpi=160, metadata={"Software": "functional-memory"})
        figure.savefig(TOPIC / f"{name}.pdf", metadata={"Creator": "functional-memory", "Producer": None,
                                                         "CreationDate": None})
        plt.close(figure)


def main() -> None:
    data, sequences = curves()
    with (TOPIC / "long_documents_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["series", *[f"ppl_positions_{a + 1}_{b}" for a, b in BLOCKS], "ppl_positions_1_16384"])
        for key, curve in data.items():
            writer.writerow([key, *[f"{math.exp(curve[a:b].mean()):.4f}" for a, b in BLOCKS],
                             f"{math.exp(curve.mean()):.4f}"])
    draw("long_documents_16k", data, BASELINE, sequences, zoom=False, legend="center right")
    draw("long_documents_16k_zoom", data, BASELINE, sequences, zoom=True, legend="upper right")
    draw("long_documents_16k_ablations", data, tuple(SERIES), sequences, zoom=False, legend="center right")


if __name__ == "__main__":
    main()
