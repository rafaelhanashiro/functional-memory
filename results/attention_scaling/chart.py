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
# Categorical slots 1-4 of the validated reference palette (adjacent pairlist, lines),
# with distinct markers as secondary encoding.
FAMILY_STYLE = {
    "normalized_learned_taylor": ("#2a78d6", "o"),
    "normalized_nonnegative": ("#eb6834", "s"),
    "unnormalized_learned": ("#1baf7a", "D"),
    "unnormalized_fixed": ("#eda100", "^"),
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

    completed = [(_degree(row["column"]), float(row["terminal_ppl"])) for panel in families
                 for row in grouped[panel["id"]] if row["status"] == "complete"]
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
            # The highest value at a degree is labeled above its marker, the others below.
            above = y >= max(level for degree, level in completed if degree == x)
            axis.annotate(f"{y:.2f}", (x, y), xytext=(0, 9 if above else -10), textcoords="offset points",
                          ha="center", va="bottom" if above else "top", fontsize=7.5, color=INK, zorder=4)
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
    # Right-margin labels for every family and reference level, nudged apart to stay legible.
    height_points = axis.get_window_extent().height * 72 / figure.dpi
    gap = 11.5 * (axis.get_ylim()[1] - axis.get_ylim()[0]) / height_points
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
    figure.text(0.09, 0.915, "Terminal perplexity at 19,074 updates, seed 0; learned coefficients are per layer and head.\n"
                "Hollow points and dashed lines are legacy runs (weight decay on all parameters, no QK-norm),\n"
                "replaced when rerun.", fontsize=8.5, color=MUTED, va="top", linespacing=1.3)
    handles = [Line2D([], [], color=FAMILY_STYLE[panel["id"]][0], marker=FAMILY_STYLE[panel["id"]][1],
                      linewidth=2, markersize=7, label=panel["title"]) for panel in families]
    handles += [Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linewidth=1.5,
                       linestyle=(0, (4, 3)), markersize=7, label="legacy run (to be rerun)"),
                Line2D([], [], color=MUTED, marker="X", linestyle="none", markersize=8,
                       label="numerical failure (update shown)")]
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


# Slots 1-2 of the reference palette, which validate all-pairs, for the fixed-coefficient comparison.
COEFFICIENT_STYLE = {"taylor": ("#2a78d6", "o"), "hermite": ("#eb6834", "s")}


def _build_facet_figure(rows: list[dict], config: dict):
    """Fixed Taylor and Hermite coefficients side by side, one facet per normalization."""
    panels = {panel["id"]: panel for panel in config["panels"]}
    grouped: dict[str, list[dict]] = {panel_id: [] for panel_id in panels}
    for row in rows:
        if row["panel"] not in panels:
            raise ValueError(f"Unexpected chart coordinate: {(row['panel'], row['column'])}")
        grouped[row["panel"]].append(row)
    families = [panel for panel in config["panels"] if panel["id"] != REFERENCE_PANEL]
    unknown = {panel.get("family") for panel in families} - set(COEFFICIENT_STYLE)
    if unknown:
        raise ValueError(f"No style for coefficient families: {sorted(map(str, unknown))}")
    facets = config["facets"]
    if {panel.get("facet") for panel in families} != {facet["id"] for facet in facets}:
        raise ValueError("Every family panel needs one declared facet, and every facet a family")
    references = [row for row in grouped.get(REFERENCE_PANEL, []) if row["status"] == "complete"]
    reference_labels = {column["id"]: column["label"] for column in panels.get(REFERENCE_PANEL, {}).get("columns", [])}
    degrees = sorted({_degree(column["id"]) for panel in families for column in panel["columns"]})
    order = list(COEFFICIENT_STYLE)  # failure-strip rows, top to bottom

    values = [float(row["terminal_ppl"]) for row in rows if row["status"] == "complete"]
    lower, upper = (min(values), max(values)) if values else (0.0, 1.0)
    pad = 0.06 * (upper - lower or 1.0)

    figure = plt.figure(figsize=(9.6, 6.0), facecolor=SURFACE)
    grid = figure.add_gridspec(2, len(facets), height_ratios=(5.2, 0.95), hspace=0.08, wspace=0.08,
                               left=0.09, right=0.97, top=0.8, bottom=0.17)
    first = None
    for index, facet in enumerate(facets):
        axis = figure.add_subplot(grid[0, index], sharey=first)
        first = first or axis
        axis.set_gid(f"chart:{facet['id']}")
        strip = figure.add_subplot(grid[1, index], sharex=axis)
        strip.set_gid(f"chart:{facet['id']}:failures")
        facet_panels = [panel for panel in families if panel["facet"] == facet["id"]]

        for row in references:
            level = float(row["terminal_ppl"])
            axis.axhline(level, color=MUTED, linewidth=1.1, linestyle=REFERENCE_STYLE.get(row["column"], "-"),
                         alpha=0.9, zorder=1, gid=f"reference:{facet['id']}:{row['column']}")
            axis.annotate(f"{reference_labels.get(row['column'], row['column'])} {level:.2f}", (0.0, level),
                          xycoords=("axes fraction", "data"), xytext=(4, -3), textcoords="offset points",
                          ha="left", va="top", fontsize=7.5, color=MUTED)

        completed = [(_degree(row["column"]), float(row["terminal_ppl"])) for panel in facet_panels
                     for row in grouped[panel["id"]] if row["status"] == "complete"]
        for panel in facet_panels:
            color, marker = COEFFICIENT_STYLE[panel["family"]]
            points = sorted((_degree(row["column"]), float(row["terminal_ppl"]))
                            for row in grouped[panel["id"]] if row["status"] == "complete")
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                if x1 - x0 == 1:  # never bridge a degree that failed or was not run
                    axis.plot([x0, x1], [y0, y1], color=color, linewidth=2, zorder=2, solid_capstyle="round")
            for x, y in points:
                axis.plot([x], [y], marker=marker, markersize=8, color=color, linestyle="none", zorder=3,
                          gid=f"point:{facet['id']}:{panel['family']}:m{x}")
                above = y >= max(level for degree, level in completed if degree == x)
                axis.annotate(f"{y:.2f}", (x, y), xytext=(0, 9 if above else -10), textcoords="offset points",
                              ha="center", va="bottom" if above else "top", fontsize=7.5, color=INK, zorder=4)
            if points:
                axis.annotate(panel["title"], points[-1], xytext=(10, 0), textcoords="offset points",
                              va="center", fontsize=8.2, color=INK)

        failed = False
        for row_index, family in enumerate(order):
            y = len(order) - 1 - row_index
            color, _ = COEFFICIENT_STYLE[family]
            if index == 0:
                strip.annotate(next(panel["title"] for panel in families if panel["family"] == family), (0.0, y),
                               xycoords=("axes fraction", "data"), xytext=(-6, 0), textcoords="offset points",
                               ha="right", va="center", fontsize=8, color=INK)
            for panel in facet_panels:
                if panel["family"] != family:
                    continue
                for row in grouped[panel["id"]]:
                    if row["status"] != "numerical_failure":
                        continue
                    failed = True
                    x = _degree(row["column"])
                    strip.plot([x], [y], marker="X", markersize=9, color=color, markeredgecolor=SURFACE,
                               markeredgewidth=0.8, linestyle="none", gid=f"failure:{facet['id']}:{family}:m{x}")
                    step = row.get("failure_step")
                    strip.annotate(f"@{step:,}" if step is not None else "failed", (x, y), xytext=(8, 0),
                                   textcoords="offset points", va="center", fontsize=7.5, color=INK)
        if not failed:
            strip.annotate("no failures", (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                           fontsize=8, color=MUTED)

        axis.set_title(facet["title"], loc="left", fontsize=10, color=INK, pad=6)
        axis.set_xlim(min(degrees) - 0.4, max(degrees) + 0.75)
        axis.set_ylim(lower - pad, upper + 2.5 * pad)
        if index == 0:
            axis.set_ylabel("Terminal held-out perplexity (lower is better)", fontsize=9.5, color=INK)
        axis.tick_params(labelbottom=False, labelleft=index == 0, colors=MUTED, labelcolor=INK,
                         labelsize=8.5, length=0)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            axis.spines[side].set_visible(False)
        axis.spines["left"].set_color(MUTED)
        strip.set_ylim(-0.6, len(order) - 0.4)
        strip.set_yticks([])
        strip.set_xticks(degrees, [f"m = {degree}" for degree in degrees], fontsize=9, color=INK)
        strip.set_xlabel("Polynomial degree", fontsize=9.5, color=INK)
        strip.tick_params(length=0)
        for side in ("top", "right", "left"):
            strip.spines[side].set_visible(False)
        strip.spines["bottom"].set_color(MUTED)
        strip.set_facecolor("#f6f6f4")

    figure.text(0.09, 0.965, config.get("title", "Fixed polynomial coefficients"),
                fontsize=13, weight="bold", color=INK, va="top")
    figure.text(0.09, 0.925, "Terminal perplexity at 19,074 updates, seed 0, QK-norm on every run. Taylor uses 1/r!; Hermite\n"
                "uses the fixed standard-normal projection of exp(s). A failed run has no perplexity; the strip\n"
                "below each panel marks the update at which it stopped.", fontsize=8.5, color=MUTED, va="top",
                linespacing=1.3)
    handles = [Line2D([], [], color=color, marker=marker, linewidth=2, markersize=7,
                      label=next(panel["title"] for panel in families if panel["family"] == family))
               for family, (color, marker) in COEFFICIENT_STYLE.items()]
    handles += [Line2D([], [], color=MUTED, linewidth=1.1, label=reference_labels.get(row["column"], row["column"]))
                for row in references[:1]]
    handles.append(Line2D([], [], color=MUTED, marker="X", linestyle="none", markersize=8,
                          label="numerical failure (update shown)"))
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.08, 0.005), ncol=len(handles), frameon=False,
                  fontsize=8, columnspacing=1.6, handlelength=2.4)
    return figure


def render_hermite_chart(rows: list[dict], config: dict, output_dir: Path) -> dict[str, Path]:
    """Write deterministic PNG and PDF copies of the Hermite–Taylor comparison."""
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {f"hermite_taylor_{extension}": figures_dir / f"hermite_taylor.{extension}" for extension in ("png", "pdf")}
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure = _build_facet_figure(rows, config)
        try:
            figure.savefig(outputs["hermite_taylor_png"], dpi=180, metadata={"Software": "functional-memory"})
            figure.savefig(outputs["hermite_taylor_pdf"], dpi=180,
                           metadata={"CreationDate": None, "ModDate": None, "Creator": "functional-memory"})
        finally:
            plt.close(figure)
    return outputs
