"""Plot the selected m=4 batch fits and their differently scoped gradient norms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from audit_optimization import HERE, ROOT, ROLES
from scaffolds.records.selections import verify_pinned_runs

COLORS = {"query": "#2765ad", "free": "#12896b", "static": "#d38221"}
LABELS = {"query": "Query-only generator", "free": "Free per-sequence coefficients", "static": "Static coefficients"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    entries = {entry["role"]: entry for entry in selection["runs"] if entry["role"] in ROLES}
    if set(entries) != set(ROLES):
        raise ValueError("Selection must explicitly name the three m=4 batch fits")
    if not args.allow_running:
        verify_pinned_runs(ROOT, list(entries.values()))
    runs = {}
    for role, entry in entries.items():
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        payload = (run / "metrics.jsonl").read_bytes()
        if not payload.endswith(b"\n"):
            if not args.allow_running:
                raise ValueError("Incomplete final metric line")
            payload = payload[:payload.rfind(b"\n") + 1]
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        rows = [row for row in rows if row.get("event") == "train"]
        record = json.loads((run / "record.json").read_text())
        config = yaml.safe_load((run / "config.yaml").read_text())
        reductions = [b["step"] - 1 for a, b in zip(rows, rows[1:]) if b["learning_rate"] < a["learning_rate"]]
        runs[ROLES[role]] = dict(rows=rows, record=record, config=config, reductions=reductions)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True,
                         "grid.alpha": .18, "grid.linewidth": .7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle("m = 4, B = 256: coefficient-only fitting", fontsize=18, fontweight="bold")
    ax = axes[0, 0]
    for name in ("query", "free", "static"):
        run = runs[name]; rows = run["rows"]
        x = np.asarray([row["step"] for row in rows]); y = np.asarray([row["nll"] for row in rows])
        label = LABELS[name] + (" (running)" if run["record"]["status"] != "complete" else "")
        ax.plot(x, y, color=COLORS[name], label=label, lw=1.6)
        for reduction in run["reductions"]:
            row = next(row for row in rows if row["step"] == reduction)
            ax.plot(reduction, row["nll"], marker="v", color=COLORS[name], ms=7, zorder=5)
    ax.set(title="Fitting loss", xlabel="Optimizer updates", ylabel="Mean next-token NLL (nats)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    ax.text(.02, .025, "▼ LR reduced 10×\nDifferent frozen source backbones; fitting inputs only",
            transform=ax.transAxes, va="bottom", fontsize=8, color="#555555")
    for name, ax in zip(("query", "free", "static"), (axes[0, 1], axes[1, 0], axes[1, 1])):
        run = runs[name]; rows = run["rows"]
        x = np.asarray([row["step"] for row in rows]); norms = np.asarray([row["gradient_norm_max"] for row in rows])
        ax.plot(x, norms, color=COLORS[name], lw=.75, alpha=.8)
        threshold = float(run["config"]["oracle"]["clip_norm"])
        ax.axhline(threshold, color="#444444", ls="--", lw=1, label="Clipping threshold = 1")
        for i, reduction in enumerate(run["reductions"]):
            ax.axvline(reduction, color="#9a4040", ls=":", lw=1.2, label="LR reduced 10×" if i == 0 else None)
        ax.set_yscale("log")
        ax.set(title=LABELS[name], xlabel="Optimizer updates",
               ylabel="Max per-sequence gradient norm" if name == "free" else "Full-batch gradient norm")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.94, loc="upper right", fontsize=8)
        if name == "free":
            text = "256 independently clipped gradients; plotted maximum"
        else:
            fraction = np.mean(norms > threshold)
            text = f"One gradient after batch averaging; {100 * fraction:.1f}% of updates clipped"
        ax.text(.02, .025, text, transform=ax.transAxes, va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82))
    running = [(name, value) for name, value in runs.items() if value["record"]["status"] != "complete"]
    if running:
        status = "; ".join(f"{LABELS[name]} through update {value['rows'][-1]['step']:,}" for name, value in running)
        fig.supxlabel("PROVISIONAL — " + status, fontsize=10, color="#814b20")
    else:
        fig.supxlabel("Terminal traces. Gradient norms are measured before clipping; y-axis scales differ.", fontsize=10)
    fig.savefig(HERE / "optimization_curves.png", dpi=180, metadata={"Software": "functional-memory"})
    fig.savefig(HERE / "optimization_curves.pdf", metadata={"Creator": "functional-memory", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(HERE / "optimization_curves.png")


if __name__ == "__main__":
    main()
