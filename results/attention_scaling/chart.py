"""Render terminal perplexity against Taylor degree for the fixed-d_K polynomial families.

One line per kernel family on a shared linear perplexity axis, with horizontal
reference levels for the softmax and exponential mechanisms. Numerical failures
have no perplexity, so they are marked in a strip below the axis at their degree,
labeled with the failing update. Runs that predate the scaffold's current defaults
(legacy) are drawn hollow with dashed connectors and are replaced when rerun.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REFERENCE_PANEL = "references"
# Categorical slots of the validated reference palette, with distinct markers as secondary
# encoding. The Hermite line takes slot 7: slots 5 and 6 fail all-pairs against the orange
# line; slot 5 marks the strip-only normalized Hermite family.
FAMILY_STYLE = {
    "normalized_learned_taylor": ("#2a78d6", "o"),
    "normalized_nonnegative": ("#eb6834", "s"),
    "unnormalized_learned": ("#1baf7a", "D"),
    "unnormalized_fixed": ("#eda100", "^"),
    "normalized_fixed_hermite": ("#e87ba4", "v"),
    "unnormalized_fixed_hermite": ("#4a3aa7", "p"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
REFERENCE_STYLE = {"softmax": "-", "softmax_no_qk_norm": (0, (6, 3)), "unnormalized_exp": (0, (1, 2))}


def _degree(column_id: str) -> int:
    return int(column_id.removeprefix("m"))


def _by_panel(rows: list[dict], config: dict) -> dict[str, list[dict]]:
    columns = {panel["id"]: {column["id"] for column in panel["columns"]} for panel in config["panels"]}
    grouped: dict[str, list[dict]] = {panel["id"]: [] for panel in config["panels"]}
    for row in rows:
        if row["panel"] not in columns or row["column"] not in columns[row["panel"]]:
            raise ValueError(f"Unexpected chart coordinate: {(row['panel'], row['column'])}")
        grouped[row["panel"]].append(row)
    unknown = set(grouped) - set(FAMILY_STYLE) - {REFERENCE_PANEL}
    if unknown:
        raise ValueError(f"No style for chart families: {sorted(unknown)}")
    return grouped


def _spread(levels: list[float], gap: float) -> list[float]:
    """Nudge sorted label positions apart by at least `gap`, keeping their mean."""
    order = sorted(range(len(levels)), key=lambda index: levels[index])
    placed = [levels[index] for index in order]
    for _ in range(50):
        moved = False
        for index in range(1, len(placed)):
            if placed[index] - placed[index - 1] < gap:
                shift = (gap - (placed[index] - placed[index - 1])) / 2
                placed[index - 1] -= shift
                placed[index] += shift
                moved = True
        if not moved:
            break
    result = [0.0] * len(levels)
    for index, level in zip(order, placed):
        result[index] = level
    return result


def _build_figure(rows: list[dict], config: dict):
    """Build the figure; kept separate from saving for layout tests."""
    grouped = _by_panel(rows, config)
    families = [panel for panel in config["panels"] if panel["id"] != REFERENCE_PANEL]
    titles = {panel["id"]: panel["title"] for panel in config["panels"]}
    references = [row for row in grouped.get(REFERENCE_PANEL, []) if row["status"] == "complete"]
    reference_labels = {column["id"]: column["label"]
                        for panel in config["panels"] if panel["id"] == REFERENCE_PANEL for column in panel["columns"]}
    degrees = sorted({_degree(column["id"]) for panel in families for column in panel["columns"]})
    failures = [(panel["id"], row) for panel in families for row in grouped[panel["id"]]
                if row["status"] == "numerical_failure"]
    failure_families = [panel["id"] for panel in families
                        if any(family == panel["id"] for family, _ in failures)]

    values = [float(row["terminal_ppl"]) for row in rows if row["status"] == "complete"]
    lower, upper = (min(values), max(values)) if values else (0.0, 1.0)
    pad = 0.06 * (upper - lower or 1.0)

    strip_rows = max(len(failure_families), 1)
    figure = plt.figure(figsize=(8.6, 5.4 + 0.28 * strip_rows), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 1, height_ratios=(5.2, 0.34 * strip_rows + 0.35), hspace=0.08,
                               left=0.09, right=0.70, top=0.84, bottom=0.2)
    axis = figure.add_subplot(grid[0])
    axis.set_gid("chart:perplexity")
    strip = figure.add_subplot(grid[1], sharex=axis)
    strip.set_gid("chart:failures")

    margin_labels: list[tuple[float, str, str, str | None]] = []  # (level, text, text color, marker color)
    for row in references:
        level = float(row["terminal_ppl"])
        legacy = bool(row.get("legacy"))
        axis.axhline(level, color=MUTED, linewidth=1.1, linestyle=REFERENCE_STYLE.get(row["column"], "-"),
                     alpha=0.55 if legacy else 0.9, zorder=1, gid=f"reference:{row['column']}")
        label = f"{reference_labels.get(row['column'], row['column'])} {level:.2f}" + (" (legacy)" if legacy else "")
        margin_labels.append((level, label, MUTED, None))

    value_labels: dict[int, list[float]] = {}
    for panel in families:
        family = panel["id"]
        color, marker = FAMILY_STYLE[family]
        points = sorted(((_degree(row["column"]), float(row["terminal_ppl"]), bool(row.get("legacy")))
                         for row in grouped[family] if row["status"] == "complete"), key=lambda point: point[0])
        for (x0, y0, legacy0), (x1, y1, legacy1) in zip(points, points[1:]):
            axis.plot([x0, x1], [y0, y1], color=color, linewidth=2, zorder=2, solid_capstyle="round",
                      linestyle=(0, (4, 3)) if legacy0 or legacy1 else "-")
        for x, y, legacy in points:
            axis.plot([x], [y], marker=marker, markersize=8, color=color, markeredgecolor=color,
                      markerfacecolor=SURFACE if legacy else color, markeredgewidth=1.8, linestyle="none",
                      zorder=3, gid=f"point:{family}:m{x}:{'legacy' if legacy else 'current'}")
            value_labels.setdefault(x, []).append(y)
        if points:
            margin_labels.append((points[-1][1], titles[family], INK, color))

    for index, family in enumerate(failure_families):
        color, _ = FAMILY_STYLE[family]
        y = len(failure_families) - 1 - index
        strip.annotate(titles[family], (1.0, y), xycoords=("axes fraction", "data"), xytext=(8, 0),
                       textcoords="offset points", va="center", fontsize=8, color=INK)
        for failed_family, row in failures:
            if failed_family != family:
                continue
            x = _degree(row["column"])
            strip.plot([x], [y], marker="X", markersize=9, color=color, markeredgecolor=SURFACE,
                       markeredgewidth=0.8, linestyle="none", gid=f"failure:{family}:m{x}")
            step = row.get("failure_step")
            strip.annotate(f"@{step:,}" if step is not None else "failed", (x, y), xytext=(8, 0),
                           textcoords="offset points", va="center", fontsize=7.5, color=INK)

    axis.set_xlim(min(degrees) - 0.4, max(degrees) + 0.4)
    axis.set_ylim(lower - pad, upper + 2.5 * pad)
    height_points = axis.get_window_extent().height * 72 / figure.dpi
    gap = 11.5 * (axis.get_ylim()[1] - axis.get_ylim()[0]) / height_points
    # Each value sits to the right of its marker; values at one degree are nudged apart.
    for x, levels in value_labels.items():
        for level, placed in zip(levels, _spread(levels, 0.85 * gap)):
            axis.annotate(f"{level:.2f}", (x, placed), xytext=(8, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=7.5, color=INK, zorder=4,
                          bbox={"boxstyle": "round,pad=0.12", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})
    # Right-margin labels for every family and reference level, nudged apart to stay legible.
    for (level, text, text_color, marker_color), placed in zip(
            margin_labels, _spread([label[0] for label in margin_labels], gap)):
        if marker_color is not None:
            axis.plot([1.03], [placed], marker="s", markersize=6, color=marker_color, transform=axis.get_yaxis_transform(),
                      clip_on=False, linestyle="none")
        axis.annotate(text, (1.0, placed), xycoords=("axes fraction", "data"), xytext=(16, 0),
                      textcoords="offset points", va="center", fontsize=8.2, color=text_color, annotation_clip=False)
    axis.set_ylabel("Terminal held-out perplexity (lower is better)", fontsize=9.5, color=INK)
    axis.tick_params(labelbottom=False, colors=MUTED, labelcolor=INK, labelsize=8.5, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(MUTED)

    strip.set_ylim(-0.6, strip_rows - 0.4)
    strip.set_yticks([])
    strip.set_ylabel("numerical\nfailure", fontsize=8, color=MUTED, rotation=0, ha="right", va="center")
    strip.set_xticks(degrees, [f"m = {degree}" for degree in degrees], fontsize=9, color=INK)
    strip.set_xlabel("Polynomial degree", fontsize=9.5, color=INK)
    strip.tick_params(length=0)
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)
    strip.spines["bottom"].set_color(MUTED)
    strip.set_facecolor("#f6f6f4")
    if not failure_families:
        strip.annotate("none", (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                       fontsize=8, color=MUTED)

    figure.text(0.09, 0.955, config.get("title", "Polynomial attention kernels across degree"),
                fontsize=13, weight="bold", color=INK, va="top")
    any_legacy = any(row.get("legacy") for row in rows if row["status"] == "complete")
    figure.text(0.09, 0.915, "Terminal perplexity at 19,074 updates, seed 0; learned coefficients are per layer and head."
                + ("\nHollow points and dashed lines are legacy runs (weight decay on all parameters, no QK-norm),\n"
                   "replaced when rerun." if any_legacy else ""), fontsize=8.5, color=MUTED, va="top", linespacing=1.3)
    handles = [Line2D([], [], color=FAMILY_STYLE[panel["id"]][0], marker=FAMILY_STYLE[panel["id"]][1],
                      linewidth=2, markersize=7, label=panel["title"]) for panel in families]
    if any_legacy:
        handles.append(Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linewidth=1.5,
                              linestyle=(0, (4, 3)), markersize=7, label="legacy run (to be rerun)"))
    handles.append(Line2D([], [], color=MUTED, marker="X", linestyle="none", markersize=8,
                          label="numerical failure (update shown)"))
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.08, 0.005), ncol=3, frameon=False,
                  fontsize=8, columnspacing=1.6, handlelength=2.4)
    return figure


def render_degree_chart(rows: list[dict], config: dict, output_dir: Path) -> dict[str, Path]:
    """Write deterministic PNG and PDF copies of the degree chart."""
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {f"degree_chart_{extension}": figures_dir / f"degree_chart.{extension}" for extension in ("png", "pdf")}
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure = _build_figure(rows, config)
        try:
            figure.savefig(outputs["degree_chart_png"], dpi=180, metadata={"Software": "functional-memory"})
            figure.savefig(outputs["degree_chart_pdf"], dpi=180,
                           metadata={"CreationDate": None, "ModDate": None, "Creator": "functional-memory"})
        finally:
            plt.close(figure)
    return outputs

