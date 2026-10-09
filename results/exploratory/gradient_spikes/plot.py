"""Gradient-norm figure and spike table for the gradient_spikes topic.

Reads only each selected run's committed metrics.jsonl (pinned in selection.yaml):
per-validation-window mean and maximum of the pre-clip gradient norm over every
update, and, where recorded, each gradient spike's update, norm, running median,
and the parameter group with the largest gradient norm.

  scaffolds/transformer/.venv/bin/python results/exploratory/gradient_spikes/plot.py
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
Y_MIN, Y_MAX = 0.1, 1e4
ROOT = HERE.parents[2]


def load_runs() -> list[dict]:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    runs = []
    for entry in selection["runs"]:
        source = entry["sources"]["metrics.jsonl"]
        path = ROOT / source["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"{path} differs from its pinned hash; run scaffolds.records refresh")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        windows = [event for event in events if event.get("event") == "validation"]
        spikes = [spike for window in windows for spike in window.get("train_gradient_spikes", [])]
        runs.append({"run_id": entry["run_id"], "label": entry["label"],
                     "windows": windows, "spikes": spikes,
                     "spike_count": (sum(window["train_gradient_spike_count"] for window in windows)
                                     if all("train_gradient_spike_count" in window for window in windows) else None)})
    return runs


def dominant_group(spike: dict) -> str:
    norms = {key.split("/")[1]: value for key, value in spike["statistics"].items()
             if key.startswith("update/") and key.endswith("/gradient_norm") and isinstance(value, (int, float))}
    return max(norms, key=norms.get)


def table(runs: list[dict]) -> str:
    lines = ["| Run | Spikes | Largest (× median) | Groups of recorded spikes |", "| --- | ---: | ---: | --- |"]
    for run in runs:
        if run["spike_count"] is None:
            largest = max(window["train_gradient_norm_max"] for window in run["windows"][1:])
            lines.append(f"| {run['label']} | not recorded | window max {largest:.3g} | — |")
            continue
        ratios = [spike["gradient_norm"] / spike["running_median"] for spike in run["spikes"]]
        groups = Counter(dominant_group(spike) for spike in run["spikes"])
        largest = f"{max(ratios):.3g}" if ratios else "—"
        described = ", ".join(f"{group} {count}" for group, count in groups.most_common()) or "—"
        lines.append(f"| {run['label']} | {run['spike_count']} | {largest} | {described} |")
    return "\n".join(lines)


def figure(runs: list[dict]) -> None:
    columns = 4
    rows = -(-len(runs) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows), sharex=True, sharey=True,
                             constrained_layout=True)
    for axis, run in zip(axes.flat, runs, strict=False):
        steps = [window["step"] for window in run["windows"]]
        axis.plot(steps, [window["train_gradient_norm_mean"] for window in run["windows"]],
                  color="#2a6fb0", linewidth=2, label="window mean")
        axis.plot(steps, [window["train_gradient_norm_max"] for window in run["windows"]],
                  color="#9ab8d8", linewidth=2, label="window max")
        shown = [spike for spike in run["spikes"] if spike["gradient_norm"] <= Y_MAX]
        clipped = [spike for spike in run["spikes"] if spike["gradient_norm"] > Y_MAX]
        if shown:
            axis.scatter([spike["step"] for spike in shown], [spike["gradient_norm"] for spike in shown],
                         s=14, color="#c2452d", zorder=3, label="recorded spike")
        if clipped:
            # Off-scale spikes sit on the top edge, with their norms written out.
            axis.scatter([spike["step"] for spike in clipped], [Y_MAX * 0.8] * len(clipped), marker="^", s=40,
                         color="#c2452d", zorder=3, clip_on=False)
            axis.annotate("off scale: " + ", ".join(f"{spike['gradient_norm']:.1e}" for spike in clipped),
                          xy=(clipped[0]["step"], Y_MAX * 0.8), xytext=(8, -14), textcoords="offset points",
                          fontsize=8, color="#c2452d")
        axis.set_yscale("log")
        axis.set_ylim(Y_MIN, Y_MAX)
        axis.set_title(run["label"], fontsize=10)
        axis.grid(alpha=0.3, linewidth=0.5)
    for axis in axes.flat[len(runs):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("update")
    for axis in axes[:, 0]:
        axis.set_ylabel("pre-clip gradient norm")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Gradient norms at warmup 400 (windows of 1,000 updates; spikes > 5× running median)")
    fig.savefig(HERE / "gradient_norms.png", dpi=150)
    fig.savefig(HERE / "gradient_norms.pdf", metadata={"CreationDate": None})


def main() -> None:
    runs = load_runs()
    figure(runs)
    print(table(runs))


if __name__ == "__main__":
    main()
