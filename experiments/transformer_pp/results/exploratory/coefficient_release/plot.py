"""Coefficient release topic: perplexity table and u_r = r! c_r over training.

Reads only the runs pinned in selection.yaml. Static runs report the stored coefficient
u_r (raw, before the 1/r! scale); query runs report the generator bias b_r, its value at
zero modulation. Each curve is the mean over layers and heads at every evaluation.

    scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.coefficient_release.plot
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PANELS = (("static", "r9c69dd7cfc8d", "r1c72e8860ade", "r991a756da7a9"),
          ("query + W decay", "r25fffc16c2f3", "rd40b46f5dc9b", "rcc4f4e5d7abb"))
COLUMNS = ("control (no release)", "released at 455", "released by order, 455-1500")


def _selection():
    return yaml.safe_load((HERE / "selection.yaml").read_text())


def _rows(run_id: str) -> list[dict]:
    path = ROOT / "experiments/transformer_pp/runs" / run_id / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trajectory(run_id: str):
    steps, values = [], []
    for row in _rows(run_id):
        report = row["attention_diagnostics"]["coefficients"]
        if report.get("raw_parameter_names", [""])[0] == "c0":
            array = np.asarray(report["raw"], float)
        else:
            array = np.asarray(report["projection_bias"], float)
        steps.append(row["step"])
        values.append(array.reshape(-1, array.shape[-1]).mean(axis=0))
    return np.asarray(steps), np.asarray(values)


def summary() -> list[dict]:
    table = []
    for entry in _selection()["runs"]:
        rows = _rows(entry["run_id"])
        best = min(rows, key=lambda row: row["validation_nll"])
        terminal = max(rows, key=lambda row: row["step"])
        table.append({"run_id": entry["run_id"], "role": entry["role"], "label": entry["label"],
                      "best_ppl": f"{best['validation_perplexity']:.3f}", "best_step": best["step"],
                      "terminal_ppl": f"{terminal['validation_perplexity']:.3f}", "terminal_step": terminal["step"]})
    return table


def main():
    table = summary()
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    figure, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey="row")
    colors = plt.cm.viridis(np.linspace(0, 0.9, 9))
    for row, (name, *runs) in enumerate(PANELS):
        for column, run_id in enumerate(runs):
            axis = axes[row, column]
            steps, values = _trajectory(run_id)
            for order in range(values.shape[1]):
                axis.plot(steps, values[:, order], color=colors[order], linewidth=1.4, label=f"r = {order}")
            axis.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":")
            if column:
                axis.axvspan(0, 455, color="0.92", zorder=0)
            if column == 2:
                axis.axvspan(455, 1500, color="0.96", zorder=0)
            axis.set_title(f"{name}: {COLUMNS[column]}\n{run_id}", fontsize=9)
            axis.set_xscale("log")
            if column == 0:
                axis.set_ylabel("mean u_r" if row == 0 else "mean b_r")
            if row == 1:
                axis.set_xlabel("training step")
    axes[0, 2].legend(fontsize=7, ncol=3, loc="upper left", frameon=False)
    figure.suptitle("Degree-eight coefficients over training (mean over layers and heads; shaded: frozen)", fontsize=10)
    figure.tight_layout()
    figure.savefig(HERE / "coefficients.png", dpi=150)


if __name__ == "__main__":
    main()
