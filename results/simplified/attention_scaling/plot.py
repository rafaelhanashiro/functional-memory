"""Simplified heatmap and degree chart from the results/attention_scaling selections.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.simplified.attention_scaling.plot
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from results.attention_scaling.chart import _spread
from results.attention_scaling.collect import collect_rows
from results.attention_scaling.render import _build_figure

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SELECTIONS = ROOT / "results/attention_scaling"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILIES = [  # panel, label, color, marker
    ("unnormalized_fixed", "Unnormalized Taylor", "#2a78d6", "o"),
    ("unnormalized_fixed_hermite", "Unnormalized Hermite", "#eb6834", "^"),
    ("unnormalized_learned", "Unnormalized learned", "#1baf7a", "D"),
    ("unnormalized_learned_head_rms_norm", "Unnormalized learned, head RMSNorm", "#7a3fbf", "P"),
    ("normalized_nonnegative", "Normalized learned, nonnegative", "#eda100", "s"),
    ("normalized_fixed", "Normalized Taylor", "#e87ba4", "v"),
]
REFERENCES = {"softmax": "softmax", "gated_deltanet": "Gated DeltaNet", "unnormalized_exp": "unnormalized exp"}
STYLE = {"font.family": "DejaVu Sans", "pdf.fonttype": 42}


def _save(figure, name, **options):
    figure.savefig(OUT / f"{name}.png", metadata={"Software": "functional-memory"}, **options)
    figure.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": "functional-memory"},
                   **options)
    plt.close(figure)


def heatmap():
    """The combined heatmap without the plain-score softmax column, subtitle, or failure updates."""
    config = yaml.safe_load((SELECTIONS / "selection.yaml").read_text())
    rows = [row for row in collect_rows(ROOT, config) if row["column"] != "softmax_no_qk_norm"]
    for row in rows:
        row["failure_step"] = None
    for panel in config["panels"]:
        panel["columns"] = [column for column in panel["columns"] if column["id"] != "softmax_no_qk_norm"]
    figure = _build_figure(rows, config)
    height = figure.get_size_inches()[1]
    for text in list(figure.texts):
        if text.get_text().startswith("Terminal validation perplexity"):
            text.remove()
        elif text.get_text().startswith("Attention scaling"):
            text.set_y(text.get_position()[1] - 0.16 / height)
    for legend in figure.legends:
        for text in legend.get_texts():
            if text.get_text().startswith("Numerical failure"):
                text.set_text("Numerical failure")
    _save(figure, "heatmap", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.25)


def degree_chart():
    """Completed runs at d_K = 64 by degree, one solid line per family, with reference levels."""
    config = yaml.safe_load((SELECTIONS / "degree_chart.yaml").read_text())
    by_panel = {}
    for row in collect_rows(ROOT, config):
        if row["status"] == "complete":
            by_panel.setdefault(row["panel"], []).append(row)
    figure, axis = plt.subplots(figsize=(8.0, 5.6), facecolor="white")
    figure.subplots_adjust(left=0.1, right=0.78, top=0.9, bottom=0.2)
    values = {}
    for panel, label, color, marker in FAMILIES:
        points = sorted((int(row["column"][1:]), float(row["terminal_ppl"])) for row in by_panel.get(panel, []))
        xs, ys = zip(*points)
        axis.plot(xs, ys, color=color, linewidth=2.0, marker=marker, markersize=7, markeredgecolor="white",
                  markeredgewidth=0.8, label=label, zorder=3, solid_capstyle="round")
        for x, y in points:
            values.setdefault(x, []).append(y)
    references = [row for row in by_panel.get("references", []) if row["column"] in REFERENCES]
    for row in references:
        axis.axhline(float(row["terminal_ppl"]), color=MUTED, linewidth=1.0, alpha=0.85, zorder=1)
    every = [y for ys in values.values() for y in ys] + [float(row["terminal_ppl"]) for row in references]
    low, high = min(every), max(every)
    pad = 0.06 * (high - low)
    axis.set_ylim(low - pad, high + 2 * pad)
    axis.set_xlim(0.6, 4.5)
    gap = 10.5 * (high - low + 3 * pad) / (axis.get_window_extent().height * 72 / figure.dpi)
    for x, ys in values.items():
        for y, placed in zip(ys, _spread(ys, 0.85 * gap)):
            axis.annotate(f"{y:.2f}", (x, placed), xytext=(8, 0), textcoords="offset points", va="center",
                          fontsize=7.5, color=INK, zorder=4,
                          bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    levels = [float(row["terminal_ppl"]) for row in references]
    for row, placed in zip(references, _spread(levels, gap)):
        axis.annotate(f"{REFERENCES[row['column']]} {float(row['terminal_ppl']):.2f}", (1.0, placed),
                      xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=8, color=MUTED, annotation_clip=False)
    axis.set_xticks([1, 2, 3, 4], [f"m = {m}" for m in (1, 2, 3, 4)], fontsize=9.5, color=INK)
    axis.set_xlabel("Polynomial degree", fontsize=10, color=INK)
    axis.set_ylabel("Terminal held-out perplexity (lower is better)", fontsize=10, color=INK)
    axis.tick_params(colors=MUTED, labelcolor=INK, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    figure.text(0.1, 0.965, "Polynomial attention kernels across degree at d_K = 64",
                fontsize=13, weight="bold", color=INK, va="top")
    figure.legend(loc="lower left", bbox_to_anchor=(0.09, 0.0), ncol=2, frameon=False, fontsize=8.5,
                  handlelength=2.6, columnspacing=1.6)
    _save(figure, "degree_chart", dpi=200)


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        heatmap()
        degree_chart()
