"""Quartic spline pretraining: perplexity against the degree-four controls and score coverage.

Reads only the runs pinned in selection.yaml. Coverage pools every layer and head of the
fixed evaluation sample, weighted by its causal score count; "beyond ±32" is the share of
scores evaluated by the end pieces' polynomial continuation.

    scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.spline_attention.plot_training
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
PAIRS = (("static", "static_spline", "control_static"), ("query + W decay", "query_spline", "control_query"))


def _runs() -> dict[str, str]:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    return {entry["role"]: entry["run_id"] for entry in selection["runs"]}


def _rows(run_id: str) -> list[dict]:
    path = ROOT / "experiments/transformer_pp/runs" / run_id / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _coverage(row: dict) -> dict:
    report = row["attention_diagnostics"]["spline_score_coverage"]
    count = np.asarray(report["causal_score_count"], float)
    pieces = np.asarray(report["piece_fraction"], float)
    beyond = np.asarray(report["above_range_fraction"], float) + np.asarray(report["below_range_fraction"], float)
    pooled = (pieces * count[..., None]).sum(axis=(0, 1)) / count.sum()
    return {"outer": float(pooled[0] + pooled[3]), "beyond": float((beyond * count).sum() / count.sum()),
            "worst_head_beyond": float(beyond.max()), "score_min": float(np.min(report["score_min"])),
            "score_max": float(np.max(report["score_max"]))}


def summary() -> list[dict]:
    runs, table = _runs(), []
    for name, spline_role, control_role in PAIRS:
        spline, control = _rows(runs[spline_role]), _rows(runs[control_role])
        best = min(spline, key=lambda row: row["validation_nll"])
        best_control = min(control, key=lambda row: row["validation_nll"])
        terminal, terminal_control = max(spline, key=lambda row: row["step"]), max(control, key=lambda row: row["step"])
        coverage = _coverage(terminal)
        table.append({
            "variant": name, "spline_run": runs[spline_role], "control_run": runs[control_role],
            "spline_best_ppl": f"{best['validation_perplexity']:.3f}", "control_best_ppl": f"{best_control['validation_perplexity']:.3f}",
            "spline_terminal_ppl": f"{terminal['validation_perplexity']:.3f}",
            "control_terminal_ppl": f"{terminal_control['validation_perplexity']:.3f}",
            "terminal_outer_piece_share": f"{coverage['outer']:.4f}", "terminal_beyond_range_share": f"{coverage['beyond']:.5f}",
            "terminal_worst_head_beyond_range": f"{coverage['worst_head_beyond']:.4f}",
            "terminal_score_min": f"{coverage['score_min']:.1f}", "terminal_score_max": f"{coverage['score_max']:.1f}",
        })
    return table


def main():
    table = summary()
    with (HERE / "training_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    runs = _runs()
    figure, (gap_axis, cover_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for (name, spline_role, control_role), color in zip(PAIRS, ("#2a6f97", "#c0392b")):
        spline, control = _rows(runs[spline_role]), _rows(runs[control_role])
        control_ppl = {row["step"]: row["validation_perplexity"] for row in control}
        steps = [row["step"] for row in spline if row["step"] in control_ppl and row["step"] >= 1000]
        ratio = [100 * (row["validation_perplexity"] / control_ppl[row["step"]] - 1) for row in spline if row["step"] in steps]
        gap_axis.plot(steps, ratio, color=color, linewidth=1.6, label=name)
        coverage = [(row["step"], _coverage(row)) for row in spline]
        cover_axis.plot([s for s, _ in coverage], [100 * c["outer"] for _, c in coverage], color=color, linewidth=1.6,
                        label=f"{name}: outer pieces")
        cover_axis.plot([s for s, _ in coverage], [100 * c["worst_head_beyond"] for _, c in coverage], color=color,
                        linewidth=1.2, linestyle="--", label=f"{name}: worst head beyond ±32")
    gap_axis.axhline(0, color="0.6", linewidth=0.8)
    gap_axis.set_xlabel("training step")
    gap_axis.set_ylabel("validation PPL vs degree-four control (%)")
    gap_axis.set_title("Spline against its m = 4 control (from step 1,000)", fontsize=10)
    gap_axis.legend(frameon=False, fontsize=8)
    cover_axis.set_xlabel("training step")
    cover_axis.set_ylabel("share of causal scores (%)")
    cover_axis.set_title("Scores in the outer pieces (|s| ≥ 6) and beyond ±32", fontsize=10)
    cover_axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(HERE / "training.png", dpi=150)


if __name__ == "__main__":
    main()
