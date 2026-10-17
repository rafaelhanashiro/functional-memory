"""Render terminal perplexity against polynomial degree for the fixed-d_K kernel families.

One line per kernel family on a shared linear perplexity axis. Hue and marker shape
identify the coefficient family (Taylor, Hermite, learned); line style and marker fill
identify the kernel variant (normalized, unnormalized, nonnegative coefficients).
Reference mechanisms are thin horizontal levels labeled in the right margin. Numerical
failures have no perplexity, so they are marked in a strip below the axis at their
degree, labeled with the failing update. Runs that predate the scaffold's current
defaults (legacy) are drawn faded and are replaced when rerun.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REFERENCE_PANEL = "references"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Coefficient families take the first three slots of the validated categorical palette,
# which pass all-pairs CVD separation; the marker shape repeats the hue as a second cue.
COEFFICIENTS = {"taylor": ("Taylor", "#2a78d6", "o"), "hermite": ("Hermite", "#eb6834", "^"),
                "learned": ("Learned", "#1baf7a", "D")}
# Kernel variants: line style, plus marker fill so that unconnected points carry it too.
VARIANTS = {"normalized": ("Normalized", "-", "full"),
            "unnormalized": ("Unnormalized", (0, (5, 2.5)), "none"),
            "nonnegative": ("Normalized, nonnegative coefficients", (0, (6, 2, 1, 2)), "bottom"),
            "unnormalized_head_rms_norm": ("Unnormalized, head RMSNorm", (0, (1.2, 1.8)), "top")}
FAMILY_STYLE = {
    "normalized_fixed": ("taylor", "normalized"),
    "unnormalized_fixed": ("taylor", "unnormalized"),
    "normalized_fixed_hermite": ("hermite", "normalized"),
    "unnormalized_fixed_hermite": ("hermite", "unnormalized"),
    "normalized_learned_taylor": ("learned", "normalized"),
    "unnormalized_learned": ("learned", "unnormalized"),
    "unnormalized_learned_head_rms_norm": ("learned", "unnormalized_head_rms_norm"),
    "normalized_nonnegative": ("learned", "nonnegative"),
}
LEGACY_ALPHA = 0.4


def _marker(family: str, legacy: bool = False) -> dict:
    """Marker keyword arguments: hue and shape by coefficients, fill by kernel variant."""
    coefficients, variant = FAMILY_STYLE[family]
    _, color, marker = COEFFICIENTS[coefficients]
    fill = VARIANTS[variant][2]
    return {"marker": marker, "color": color, "markeredgecolor": color, "markeredgewidth": 1.3,
            "markerfacecolor": SURFACE if fill == "none" else color, "fillstyle": "full" if fill == "none" else fill,
            "markerfacecoloralt": SURFACE, "alpha": LEGACY_ALPHA if legacy else 1.0}


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
    references = [row for row in grouped.get(REFERENCE_PANEL, []) if row["status"] == "complete"]
    reference_labels = {column["id"]: column["label"]
                        for panel in config["panels"] if panel["id"] == REFERENCE_PANEL for column in panel["columns"]}
    degrees = sorted({_degree(column["id"]) for panel in families for column in panel["columns"]})
    failure_families = [panel["id"] for panel in families
                        if any(row["status"] == "numerical_failure" for row in grouped[panel["id"]])]
    any_legacy = any(row.get("legacy") for row in rows if row["status"] == "complete")

    values = [float(row["terminal_ppl"]) for row in rows if row["status"] == "complete"]
    lower, upper = (min(values), max(values)) if values else (0.0, 1.0)
    pad = 0.06 * (upper - lower or 1.0)

    strip_rows = max(len(failure_families), 1)
    figure = plt.figure(figsize=(8.0, 5.5 + 0.22 * strip_rows), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 1, height_ratios=(5.2, 0.3 * strip_rows + 0.35), hspace=0.08,
                               left=0.1, right=0.78, top=0.86, bottom=0.2)
    axis = figure.add_subplot(grid[0])
    axis.set_gid("chart:perplexity")
    strip = figure.add_subplot(grid[1], sharex=axis)
    strip.set_gid("chart:failures")

    # Reference levels are one neutral style, identified by their labels in the right margin.
    reference_levels = []
    for row in references:
        level = float(row["terminal_ppl"])
        legacy = bool(row.get("legacy"))
        axis.axhline(level, color=MUTED, linewidth=1.0, alpha=LEGACY_ALPHA if legacy else 0.85,
                     zorder=1, gid=f"reference:{row['column']}")
        # A multi-line label carries its value on the first line.
        name, *rest = reference_labels.get(row["column"], row["column"]).split("\n")
        label = "\n".join([f"{name} {level:.2f}" + (" (legacy)" if legacy else ""), *rest])
        reference_levels.append((level, label))

    value_labels: dict[int, list[float]] = {}
    for panel in families:
        family = panel["id"]
        coefficients, variant = FAMILY_STYLE[family]
        color, dash = COEFFICIENTS[coefficients][1], VARIANTS[variant][1]
        points = sorted(((_degree(row["column"]), float(row["terminal_ppl"]), bool(row.get("legacy")))
                         for row in grouped[family] if row["status"] == "complete"), key=lambda point: point[0])
        declared = {_degree(column["id"]) for column in panel["columns"]}
        for (x0, y0, legacy0), (x1, y1, legacy1) in zip(points, points[1:]):
            if any(x0 < degree < x1 for degree in declared):  # never bridge a failed or missing degree
                continue
            axis.plot([x0, x1], [y0, y1], color=color, linewidth=1.6, zorder=2, linestyle=dash,
                      solid_capstyle="round", dash_capstyle="round",
                      alpha=LEGACY_ALPHA if legacy0 or legacy1 else 1.0)
        for x, y, legacy in points:
            axis.plot([x], [y], markersize=6.5, linestyle="none", zorder=3, **_marker(family, legacy),
                      gid=f"point:{family}:m{x}:{'legacy' if legacy else 'current'}")
            value_labels.setdefault(x, []).append(y)

    # Failures keep their family's marker and fill, so the legends identify each strip row.
    for index, family in enumerate(failure_families):
        y = len(failure_families) - 1 - index
        for row in grouped[family]:
            if row["status"] != "numerical_failure":
                continue
            x = _degree(row["column"])
            strip.plot([x], [y], markersize=6.5, linestyle="none", **_marker(family, bool(row.get("legacy"))),
                       gid=f"failure:{family}:m{x}")
            step = row.get("failure_step")
            strip.annotate(f"@{step:,}" if step is not None else "failed", (x, y), xytext=(7, 0),
                           textcoords="offset points", va="center", fontsize=7, color=INK)

    axis.set_xlim(min(degrees) - 0.4, max(degrees) + 0.5)
    axis.set_ylim(lower - pad, upper + 2.5 * pad)
    height_points = axis.get_window_extent().height * 72 / figure.dpi
    gap = 10.5 * (axis.get_ylim()[1] - axis.get_ylim()[0]) / height_points
    # Each value sits to the right of its marker; values at one degree are nudged apart.
    for x, levels in value_labels.items():
        for level, placed in zip(levels, _spread(levels, 0.85 * gap)):
            axis.annotate(f"{level:.2f}", (x, placed), xytext=(7, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=7, color=INK, zorder=4,
                          bbox={"boxstyle": "round,pad=0.12", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})
    levels = [level for level, _ in reference_levels]
    for (_, label), placed in zip(reference_levels, _spread(levels, gap)):
        # Multi-line labels hang below their level so that the first line sits on it.
        multiline = "\n" in label
        axis.annotate(label, (1.0, placed), xycoords=("axes fraction", "data"), xytext=(6, 4.5 if multiline else 0),
                      textcoords="offset points", ha="left", va="top" if multiline else "center", fontsize=7.5,
                      color=MUTED, annotation_clip=False)
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

    figure.text(0.1, 0.965, config.get("title", "Polynomial attention kernels across degree"),
                fontsize=13, weight="bold", color=INK, va="top")
    figure.text(0.1, 0.925, "Terminal perplexity at 19,074 updates, seed 0; learned coefficients are per layer and head."
                + (f"\n{config['caption']}" if config.get("caption") else "")
                + ("\nFaded points and lines are legacy runs, which predate the current defaults."
                   if any_legacy else ""), fontsize=8.5, color=MUTED, va="top", linespacing=1.3)

    # Two factored legends: hue for the coefficient family, line style and fill for the kernel.
    used = [FAMILY_STYLE[panel["id"]] for panel in families]
    coefficient_handles = [
        Line2D([], [], color=color, marker=marker, linewidth=1.6, markersize=6, label=name)
        for key, (name, color, marker) in COEFFICIENTS.items() if any(key == pair[0] for pair in used)]
    variant_handles = []
    for key, (name, dash, fill) in VARIANTS.items():
        if any(key == pair[1] for pair in used):
            variant_handles.append(Line2D(
                [], [], color=MUTED, linestyle=dash, linewidth=1.6, marker="o", markersize=6, markeredgecolor=MUTED,
                markerfacecolor=SURFACE if fill == "none" else MUTED, fillstyle="full" if fill == "none" else fill,
                markerfacecoloralt=SURFACE, label=name))
    if any_legacy:
        variant_handles.append(Line2D([], [], color=MUTED, marker="o", linewidth=1.6, markersize=6,
                                      alpha=LEGACY_ALPHA, label="legacy run (to be rerun)"))
    style = {"loc": "lower left", "frameon": False, "fontsize": 7.8, "title_fontsize": 8, "alignment": "left",
             "handlelength": 3.2}
    figure.legend(handles=coefficient_handles, title="Coefficients", bbox_to_anchor=(0.09, 0.005), **style)
    figure.legend(handles=variant_handles, title="Kernel", bbox_to_anchor=(0.3, 0.005), **style)
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

