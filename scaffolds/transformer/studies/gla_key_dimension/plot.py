"""Summarize GLA terminal perplexity beside the dK × Taylor comparison points.

Writes ``results/gla_key_dimension/summary.csv`` and
``figures/perplexity_vs_key_dimension.{png,pdf}`` from ``runs/`` and the
checksummed comparison runs pinned in the provenance manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "results/gla_key_dimension"
DEGREES = (2, 4, 8, 16)
# Fixed series colors: GLA is the accent; softmax neutral; Taylor degree is a
# single-hue ramp because degree is ordered (light = low, dark = high).
COLORS = {"gla": "#c2410c", "softmax": "#374151",
          2: "#3b82f6", 4: "#2563eb", 8: "#1d4ed8", 16: "#1e3a8a"}
# Degree markers keep neighbouring ramp steps distinguishable without color.
MARKERS = {2: "o", 4: "^", 8: "v", 16: "P"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_table(manifest: dict) -> dict:
    """Terminal perplexity by (d_K, degree); degree None is softmax. Checksums are verified."""
    table = {}
    for run in manifest["comparisons"]["runs"]:
        for key in ("resolved_config", "metrics"):
            if sha256(ROOT / run[key]["path"]) != run[key]["sha256"]:
                raise ValueError(f"Comparison {run['name']} {key} differs from its pinned checksum")
        table[(run["qk_head_dim"], run["attention_degree"])] = run["terminal"]["validation_perplexity"]
    return table


def gla_rows(manifest: dict, record: Path) -> list[dict]:
    table = comparison_table(manifest)
    rows = []
    for run in manifest["runs"]:
        directory = record / "runs" / run["name"]
        completion = directory / "completion.json"
        row = {"run": run["name"], "qk_head_dim": run["qk_head_dim"], "parameters": run["parameter_count"],
               "state_elements_per_layer": run["state_elements_per_layer"], "status": "incomplete"}
        if completion.exists():
            terminal = json.loads(completion.read_text())
            best = json.loads((directory / "best_validation.json").read_text())
            if terminal["training_tokens"] != manifest["prediction_tokens"]:
                raise ValueError(f"{run['name']} completion does not cover the full training budget")
            softmax = table[(run["qk_head_dim"], None)]
            row.update(status="complete", updates=terminal["final_step"], prediction_tokens=terminal["training_tokens"],
                       terminal_validation_nll=terminal["validation_nll"],
                       terminal_validation_perplexity=terminal["validation_perplexity"],
                       best_step=best["step"], best_validation_perplexity=best["validation_perplexity"],
                       softmax_perplexity=softmax,
                       delta_perplexity_vs_softmax=terminal["validation_perplexity"] - softmax,
                       **{f"taylor_m{degree:03d}_perplexity": table[(run["qk_head_dim"], degree)] for degree in DEGREES})
        rows.append(row)
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    fields = ["run", "qk_head_dim", "parameters", "state_elements_per_layer", "status", "updates",
              "prediction_tokens", "terminal_validation_nll", "terminal_validation_perplexity", "best_step",
              "best_validation_perplexity", "softmax_perplexity", "delta_perplexity_vs_softmax",
              *[f"taylor_m{degree:03d}_perplexity" for degree in DEGREES]]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], manifest: dict, directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = comparison_table(manifest)
    widths = manifest["key_dimensions"]
    figure, axis = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for degree in DEGREES:
        axis.plot(widths, [table[(w, degree)] for w in widths], color=COLORS[degree], linewidth=1.5,
                  marker=MARKERS[degree], markersize=5, label=f"Taylor m={degree} (normalized)")
    axis.plot(widths, [table[(w, None)] for w in widths], color=COLORS["softmax"], linewidth=2,
              linestyle="--", marker="s", markersize=5, label="Softmax")
    complete = [row for row in rows if row["status"] == "complete"]
    if complete:
        axis.plot([row["qk_head_dim"] for row in complete],
                  [row["terminal_validation_perplexity"] for row in complete],
                  color=COLORS["gla"], linewidth=2.5, marker="D", markersize=7, label="GLA", zorder=5)
    axis.set_xscale("log", base=2)
    axis.set_xticks(widths, [str(w) for w in widths])
    axis.set_xlabel("Query/key head dimension $d_K$")
    axis.set_ylabel("Terminal validation perplexity (10.0B tokens)")
    axis.grid(True, color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda index: labels[index] != "GLA")  # GLA first, others as drawn.
    axis.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", frameon=False, fontsize=8)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = directory / f"perplexity_vs_key_dimension.{suffix}"
        # Omit the PDF creation timestamp so regenerated figures are byte-identical.
        figure.savefig(path, dpi=180, metadata={"CreationDate": None} if suffix == "pdf" else None)
        written.append(path)
    plt.close(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args()
    manifest = json.loads((args.record / "provenance/manifest.json").read_text())
    rows = gla_rows(manifest, args.record)
    write_summary(rows, args.record / "summary.csv")
    for path in plot(rows, manifest, args.record / "figures"):
        print(path)


if __name__ == "__main__":
    main()
