"""Figures and summary for the extrapolation_scores diagnostic from its selected run."""
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
STYLE = {"exp": ("exp", "#1f77b4"), "fixed_taylor_m4": ("fixed Taylor (m = 4)", "#ff7f0e"),
         "query_wdecay_m4": ("query + W decay (m = 4)", "#d62728")}


def pooled(rows):
    """Pool every layer's per-bin score sums, extremes, and late-query weight shares."""
    count = np.sum([row["score_count"] for row in rows], axis=0)
    mean = np.sum([row["score_sum"] for row in rows], axis=0) / count
    std = np.sqrt(np.maximum(np.sum([row["score_sum_squares"] for row in rows], axis=0) / count - mean ** 2, 0.0))
    return {"count": count, "mean": mean, "std": std,
            "min": np.min([row["score_min"] for row in rows], axis=0), "max": np.max([row["score_max"] for row in rows], axis=0),
            "share": np.mean([row["late_query_weight_share"] for row in rows], axis=0),
            "far": np.array([row["head_far_weight_share"] for row in rows])}


def main():
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    run = next(entry for entry in selection["runs"] if entry["role"] == "score_diagnostic")
    rows = [json.loads(line) for line in open(ROOT / "experiments/transformer_pp/runs" / run["run_id"] / "metrics.jsonl")]
    width, limit = rows[0]["bin_width"], rows[0]["training_length"]
    data = {role: pooled([row for row in rows if row["role"] == role]) for role in STYLE}
    centers = (np.arange(len(rows[0]["score_count"])) + 0.5) * width
    seen = centers < limit

    fig, axes = plt.subplots(1, len(STYLE), figsize=(13, 4), sharey=True)
    for axis, (role, (label, color)) in zip(axes, STYLE.items()):
        values = data[role]
        axis.fill_between(centers, values["min"], values["max"], color=color, alpha=0.12, lw=0, label="min to max")
        axis.fill_between(centers, values["mean"] - values["std"], values["mean"] + values["std"], color=color, alpha=0.35, lw=0, label="mean ± std")
        axis.plot(centers, values["mean"], color=color, lw=2, label="mean")
        axis.axvline(limit, color="k", ls="--", lw=0.8)
        axis.set_title(label)
        axis.set_xlabel("key distance t − i (dashed: training length)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("score s = <q, k> / sqrt(d_K), all layers and heads")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "scores_by_distance.png", dpi=150)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for role, (label, color) in STYLE.items():
        axis.plot(centers, data[role]["share"], color=color, lw=2, label=f"{label}: {data[role]['far'].mean():.1%} beyond {limit:,}")
    axis.axvline(limit, color="k", ls="--", lw=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("key distance t − i (dashed: training length)")
    axis.set_ylabel(f"share of a late query's weight per {width}-token bin")
    axis.set_title(f"Where queries at positions ≥ {limit:,} put their weight (mean over layers and heads)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "weight_share_by_distance.png", dpi=150)

    with open(HERE / "summary.csv", "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["role", "score_std_seen", "score_std_unseen", "max_abs_score_seen", "max_abs_score_unseen",
                         "mean_far_weight_share", "max_head_far_weight_share"])
        for role in STYLE:
            values = data[role]
            def moments(mask):
                n = values["count"][mask].sum()
                mean = (values["mean"][mask] * values["count"][mask]).sum() / n
                second = ((values["std"][mask] ** 2 + values["mean"][mask] ** 2) * values["count"][mask]).sum() / n
                return np.sqrt(max(second - mean ** 2, 0.0))
            writer.writerow([role, f"{moments(seen):.3f}", f"{moments(~seen):.3f}",
                             f"{max(abs(values['min'][seen].min()), values['max'][seen].max()):.2f}",
                             f"{max(abs(values['min'][~seen].min()), values['max'][~seen].max()):.2f}",
                             f"{values['far'].mean():.4f}", f"{values['far'].max():.4f}"])


if __name__ == "__main__":
    main()
