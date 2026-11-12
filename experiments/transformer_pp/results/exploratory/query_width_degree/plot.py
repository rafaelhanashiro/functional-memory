"""Regenerate the key-width-by-degree heatmap and summary from this topic's explicit selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]
INK, MUTED = "#172334", "#5b6573"


def collect() -> list[dict]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((run / "config.yaml").read_text())
        record = json.loads((run / "record.json").read_text())
        row = {"role": entry["role"], "run_id": entry["run_id"], "status": record["status"],
               "qk_head_dim": config["model"]["qk_head_dim"], "attention_degree": config["model"]["attention_degree"]}
        if record["status"] == "complete":
            best = json.loads((run / "best_validation.json").read_text())
            terminal = json.loads((run / "completion.json").read_text())
            row.update(best_step=best["step"], best_validation_nll=best["validation_nll"],
                       best_validation_perplexity=best["validation_perplexity"],
                       terminal_step=terminal["final_step"], terminal_validation_perplexity=terminal["validation_perplexity"])
        rows.append(row)
    return rows


def render(rows: list[dict]) -> None:
    widths = sorted({row["qk_head_dim"] for row in rows})
    degrees = sorted({row["attention_degree"] for row in rows})
    indexed = {(row["qk_head_dim"], row["attention_degree"]): row for row in rows}
    grid = np.full((len(widths), len(degrees)), np.nan)
    for i, width in enumerate(widths):
        for j, degree in enumerate(degrees):
            row = indexed.get((width, degree))
            if row and row["status"] == "complete":
                grid[i, j] = row["best_validation_perplexity"]
    values = grid[np.isfinite(grid)]
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    # One hue, light to dark with perplexity: as in the attention-scaling heatmap.
    cmap = matplotlib.colormaps["Blues"].resampled(256).with_extremes(bad=(1, 1, 1, 0))
    cell_w, cell_h, left, bottom = 1.05, 0.68, 1.15, 0.85
    axis_w, axis_h = len(degrees) * cell_w, len(widths) * cell_h
    figure_w, figure_h = left + axis_w + 1.75, bottom + axis_h + 1.5
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white", "svg.hashsalt": "functional-memory"}):
        figure = plt.figure(figsize=(figure_w, figure_h))
        axis = figure.add_axes((left / figure_w, bottom / figure_h, axis_w / figure_w, axis_h / figure_h))
        axis.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, origin="upper", aspect="auto",
                    interpolation="none", zorder=2)
        best_cell = np.unravel_index(np.nanargmin(grid), grid.shape)
        for i, width in enumerate(widths):
            for j, degree in enumerate(degrees):
                if not np.isfinite(grid[i, j]):
                    axis.text(j, i, "not run", ha="center", va="center", color=MUTED, fontsize=8.5, zorder=4)
                    continue
                shade = np.dot(cmap(norm(grid[i, j]))[:3], (0.2126, 0.7152, 0.0722))
                axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=10,
                          color="white" if shade < 0.5 else INK,
                          weight="bold" if (i, j) == best_cell else "normal", zorder=4)
        axis.set_xticks(range(len(degrees)), [f"m = {degree}" for degree in degrees], fontsize=9.5, color=INK)
        axis.set_yticks(range(len(widths)), [str(width) for width in widths], fontsize=9.5, color=INK)
        axis.set_xlabel("Polynomial degree", fontsize=10, color=INK, labelpad=8)
        axis.set_ylabel("Key dimension $d_K$", fontsize=10, color=INK, labelpad=10)
        axis.set_xticks(np.arange(-0.5, len(degrees), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(widths), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2, zorder=3)
        axis.tick_params(which="both", length=0, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        color_axis = figure.add_axes(((left + axis_w + 0.35) / figure_w, bottom / figure_h,
                                      0.2 / figure_w, axis_h / figure_h))
        colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
        colorbar.set_label("Validation-best perplexity\n(lower is better)", fontsize=9, color=INK, labelpad=8)
        colorbar.ax.tick_params(labelsize=8, length=3, colors=INK)
        colorbar.outline.set_linewidth(0.6)
        figure.text(0.3 / figure_w, 1 - 0.28 / figure_h, "Query coefficients across key width and degree",
                    fontsize=13, weight="bold", va="top", color=INK)
        figure.text(0.3 / figure_w, 1 - 0.62 / figure_h,
                    "Transformer++; tanh query coefficients with decayed W, 14 heads, value width 64.\n"
                    "Each cell is one run's validation-best perplexity.",
                    fontsize=8.5, va="top", color=MUTED, linespacing=1.35)
        figure.savefig(TOPIC / "heatmap.png", dpi=180, metadata={"Software": "functional-memory"})
        figure.savefig(TOPIC / "heatmap.pdf", metadata={"Creator": "functional-memory", "Producer": None,
                                                        "CreationDate": None})
        plt.close(figure)


def main() -> None:
    rows = collect()
    fields = ["role", "run_id", "status", "qk_head_dim", "attention_degree", "best_step", "best_validation_nll",
              "best_validation_perplexity", "terminal_step", "terminal_validation_perplexity"]
    with (TOPIC / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    render(rows)


if __name__ == "__main__":
    main()
