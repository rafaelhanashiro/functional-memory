"""Summarize the pinned spline-knot calibration; every quantile is labeled by its sampling level.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.pretraining.alternative_attention.splines.spline_attention.summarize

Writes ``summary.csv``, ``coverage.csv`` and ``score_coverage.png``.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
STYLE = {"static": ("Static", "#1f77b4"), "query_wdecay": ("Query + W decay", "#d62728")}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".10g") if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})


def summarize(rows):
    count = np.asarray([row["score_count"] for row in rows], dtype=np.float64)
    means = np.asarray([row["score_mean"] for row in rows])
    stds = np.asarray([row["score_std"] for row in rows])
    mean = np.average(means, weights=count)
    variance = np.average(stds ** 2 + (means - mean) ** 2, weights=count)
    identity = {key: rows[0][key] for key in ("role", "state", "source_run_id", "checkpoint_step")}
    result = {**identity, "layer_head_count": len(rows), "score_count": int(count.sum()),
              "score_mean": mean, "score_std": np.sqrt(variance),
              "score_min": min(row["score_min"] for row in rows),
              "score_max": max(row["score_max"] for row in rows)}
    for source, label in (("absolute_score_quantiles", "absolute_score"),
                          ("query_absmax_quantiles", "query_absmax")):
        values = [row[source]["0.999"] for row in rows]
        result[f"head_median_{label}_q999"] = float(np.median(values))
        result[f"head_max_{label}_q999"] = float(np.max(values))
    result["zero_weight_query_count"] = sum(row.get("zero_weight_query_count", 0) for row in rows)
    coverage = []
    radii = [entry["radius"] for entry in rows[0]["radius_statistics"]]
    for index, radius in enumerate(radii):
        stats = [row["radius_statistics"][index] for row in rows]
        if any(item["radius"] != radius for item in stats):
            raise ValueError("Inconsistent radius grid")
        shares = np.asarray([item["mean_query_absolute_weight_share_outside"] for item in stats])
        queries = np.asarray([row["sampled_query_count"] for row in rows])
        coverage.append({**identity, "radius": radius,
            "pooled_causal_pair_fraction_outside": np.average(
                [item["causal_pair_fraction_outside"] for item in stats], weights=count),
            "head_mean_query_absolute_weight_share_outside": float(shares.mean()),
            "head_p95_mean_query_absolute_weight_share_outside": float(np.quantile(shares, 0.95)),
            "head_max_mean_query_absolute_weight_share_outside": float(shares.max()),
            "mean_query_peak_fraction_outside": np.average(
                [item["query_peak_fraction_outside"] for item in stats], weights=queries)})
    return result, coverage


def main(output: Path = HERE):
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    entries = [entry for entry in selection["runs"] if entry["role"] == "score_calibration"]
    if len(entries) != 1:
        raise ValueError("Select exactly one score calibration run")
    verify_pinned_runs(ROOT, entries)
    entry = entries[0]
    run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    if json.loads((run / "record.json").read_text())["status"] != "complete":
        raise ValueError("The selected diagnostic must be complete")
    metadata = json.loads((run / "metadata.json").read_text())
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    keys = [(snapshot["role"], snapshot["state"]) for snapshot in metadata["snapshots"]]
    if len(keys) != len(set(keys)) or set(keys) != {(row["role"], row["state"]) for row in rows}:
        raise ValueError("Snapshot identities are inconsistent")
    edges = np.asarray(metadata["histogram_edges"])
    summaries, coverage_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for snapshot, key in zip(metadata["snapshots"], keys):
        selected = sorted((row for row in rows if (row["role"], row["state"]) == key),
                          key=lambda row: (row["layer"], row["head"]))
        coordinates = {(row["layer"], row["head"]) for row in selected}
        expected = {(layer, head) for layer in range(snapshot["layers"]) for head in range(snapshot["heads"])}
        if coordinates != expected or len(selected) != len(expected):
            raise ValueError(f"Incomplete or duplicate layer/head measurements: {key}")
        summary, coverage = summarize(selected)
        summaries.append(summary)
        coverage_rows.extend(coverage)
        counts = np.sum([row["histogram_counts"] for row in selected], axis=0)
        accounted = int(counts.sum()) + sum(row["histogram_underflow"] + row["histogram_overflow"] for row in selected)
        if accounted != summary["score_count"]:
            raise ValueError("Histogram does not account for all sampled-row scores")
        label, color = STYLE[key[0]]
        state = "reconstructed init" if key[1] == "initialization" else "trained checkpoint"
        linestyle = "--" if key[1] == "initialization" else "-"
        axes[0].stairs(counts / summary["score_count"] / np.diff(edges), edges,
                       color=color, linestyle=linestyle, linewidth=1.8, label=f"{label}, {state}")
        axes[1].plot([row["radius"] for row in coverage],
                     [row["head_mean_query_absolute_weight_share_outside"] for row in coverage],
                     color=color, linestyle=linestyle, linewidth=1.8)
    axes[0].set(xlim=(-16, 16), xlabel="Attention score (central range)", ylabel="Score density")
    axes[0].legend(fontsize=8)
    axes[1].set(xscale="log", yscale="symlog", xlabel="Candidate knot-range radius R",
                ylabel="Mean absolute-weight share outside [−R, R]", ylim=(0, 1))
    axes[1].set_yscale("symlog", linthresh=1e-5)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "score_coverage.png", dpi=160, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "coverage.csv", coverage_rows)


if __name__ == "__main__":
    main()
