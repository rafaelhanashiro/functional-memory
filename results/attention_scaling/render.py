"""Render one comparable figure for the attention scaling experiments."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle


_STATUS_STYLE = {
    "numerical_failure": ("#f5c5bd", "#b44c40", "xx"),
    "not_run": ("#e5e8ec", "#929aa4", "//"),
    "incomplete": ("#f9e5ae", "#bb8b28", ".."),
}
# Runs that predate the current defaults keep their perplexity color and get this outline.
_LEGACY_EDGE, _LEGACY_WIDTH, _LEGACY_INSET = "#b45309", 2.2, 0.07
_LEGACY_LABEL = "Legacy run: decay on all parameters, no QK-norm (to be rerun)"


def _coordinate_rows(rows: list[dict], config: dict) -> dict[tuple[str, str, int], dict]:
    """Validate coordinates and fill any absent cell as an explicitly missing run."""
    widths = config["d_k"]
    panels = config["panels"]
    if not widths or len(widths) != len(set(widths)) or not panels:
        raise ValueError("Expected nonempty, unique d_k values and at least one panel")
    if len({panel["id"] for panel in panels}) != len(panels):
        raise ValueError("Panel IDs must be unique")
    expected = set()
    for panel in panels:
        columns = panel["columns"]
        if not columns or len({column["id"] for column in columns}) != len(columns):
            raise ValueError(f"Expected unique, nonempty columns in {panel['id']}")
        expected.update((panel["id"], column["id"], width) for column in columns for width in widths)

    indexed = {}
    for row in rows:
        key = (row["panel"], row["column"], row["d_k"])
        if key not in expected:
            raise ValueError(f"Unexpected heatmap coordinate: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate heatmap coordinate: {key}")
        status = row["status"]
        if status not in ("complete", *_STATUS_STYLE):
            raise ValueError(f"Unexpected run status: {status}")
        value = row.get("terminal_ppl")
        if status == "complete":
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"Complete run needs a finite, positive terminal PPL: {key}")
        elif value is not None:
            raise ValueError(f"Non-complete run cannot supply terminal PPL: {key}")
        if row.get("legacy") and status == "not_run":
            raise ValueError(f"An untrained coordinate cannot be a legacy run: {key}")
        indexed[key] = row
    for key in expected - indexed.keys():
        indexed[key] = {"panel": key[0], "column": key[1], "d_k": key[2],
                        "status": "not_run", "terminal_ppl": None, "failure_step": None}
    return indexed


def _annotation(row: dict) -> tuple[str, float]:
    status = row["status"]
    if status == "complete":
        return f"{float(row['terminal_ppl']):.3f}", 8.5
    if status == "numerical_failure":
        step = row.get("failure_step")
        return (f"FAIL\n@{step:,}" if step is not None else "FAIL"), 8.0
    if status == "not_run":
        return "NOT\nRUN", 7.8
    return "INCOMPLETE", 6.9


def _tick_label(label: str) -> str:
    return "\n".join(textwrap.fill(part, width=13, break_long_words=False,
                                    break_on_hyphens=False) for part in str(label).split("\n"))


def _build_figure(rows: list[dict], config: dict):
    """Build the figure; kept separate from saving for layout and scale tests."""
    indexed = _coordinate_rows(rows, config)
    widths = config["d_k"]
    panels = config["panels"]
    values = np.array([float(row["terminal_ppl"]) for row in indexed.values()
                       if row["status"] == "complete"], dtype=float)
    if len(values):
        lower, upper = float(values.min()), float(values.max())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
    else:
        lower, upper = 0.0, 1.0
    norm = Normalize(vmin=lower, vmax=upper)
    cmap = matplotlib.colormaps["Blues"].resampled(256).with_extremes(bad=(1, 1, 1, 0))

    # Cell sizes stay the same across families, including the narrower reference panel.
    cell_w, cell_h, gap = 0.84, 0.62, 0.38
    groups = [panels[:3], panels[3:]] if len(panels) > 3 else [panels]
    group_widths = [sum(len(panel["columns"]) * cell_w for panel in group)
                    + gap * (len(group) - 1) for group in groups]
    content_w = max(group_widths)
    left = 1.15
    figure_w = left + content_w + 1.55
    axis_h = len(widths) * cell_h
    if len(groups) == 2:
        bottom_y, top_y = 1.55, 1.55 + axis_h + 1.2
        group_y = [top_y, bottom_y]
    else:
        group_y = [1.55]
    figure_h = group_y[0] + axis_h + 1.2 + (0.25 if config.get("caption") else 0)  # a caption adds a subtitle line
    figure = plt.figure(figsize=(figure_w, figure_h), facecolor="white")

    for group, width, y in zip(groups, group_widths, group_y):
        x = left + (content_w - width) / 2
        for panel_index, panel in enumerate(group):
            columns = panel["columns"]
            axis_w = len(columns) * cell_w
            axis = figure.add_axes((x / figure_w, y / figure_h,
                                    axis_w / figure_w, axis_h / figure_h))
            axis.set_gid(f"panel:{panel['id']}")
            grid = np.full((len(widths), len(columns)), np.nan)
            for i, dk in enumerate(widths):
                for j, column in enumerate(columns):
                    row = indexed[(panel["id"], column["id"], dk)]
                    if row["status"] == "complete":
                        grid[i, j] = float(row["terminal_ppl"])
                    else:
                        face, edge, hatch = _STATUS_STYLE[row["status"]]
                        axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                 facecolor=face, edgecolor=edge,
                                                 linewidth=0, hatch=hatch, zorder=1))
            axis.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                        origin="upper", aspect="auto", interpolation="none", zorder=2)
            for i, dk in enumerate(widths):
                for j, column in enumerate(columns):
                    if indexed[(panel["id"], column["id"], dk)].get("legacy"):
                        axis.add_patch(Rectangle((j - 0.5 + _LEGACY_INSET, i - 0.5 + _LEGACY_INSET),
                                                 1 - 2 * _LEGACY_INSET, 1 - 2 * _LEGACY_INSET, fill=False,
                                                 edgecolor=_LEGACY_EDGE, linewidth=_LEGACY_WIDTH,
                                                 gid="legacy", zorder=3.5))
            for i, dk in enumerate(widths):
                for j, column in enumerate(columns):
                    row = indexed[(panel["id"], column["id"], dk)]
                    label, font_size = _annotation(row)
                    if row["status"] == "complete":
                        color = "white" if np.dot(cmap(norm(grid[i, j]))[:3],
                                                   (0.2126, 0.7152, 0.0722)) < 0.5 else "#172334"
                    else:
                        color = "#222b36"
                    axis.text(j, i, label, ha="center", va="center", color=color,
                              fontsize=font_size, weight="bold" if row["status"] != "complete" else "normal",
                              linespacing=0.95, zorder=4)
            axis.set_xticks(range(len(columns)), [_tick_label(column["label"]) for column in columns],
                            fontsize=8.7)
            axis.set_yticks(range(len(widths)), [str(dk) for dk in widths] if panel_index == 0 else [],
                            fontsize=9.5)
            if panel_index == 0:
                axis.set_ylabel("Key dimension $d_k$", fontsize=10, labelpad=10)
            axis.set_title(panel["title"], fontsize=11.5, weight="bold", pad=12, loc="left")
            axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(widths), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=1.5, zorder=3)
            axis.tick_params(which="both", length=0, pad=5)
            for spine in axis.spines.values():
                spine.set_visible(False)
            x += axis_w + gap

    color_axis = figure.add_axes(((left + content_w + 0.47) / figure_w,
                                  1.55 / figure_h, 0.22 / figure_w,
                                  (group_y[0] + axis_h - 1.55) / figure_h))
    color_axis.set_gid("colorbar")
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.set_label("Terminal held-out perplexity\n(lower is better)", fontsize=9.5, labelpad=10)
    colorbar.ax.tick_params(labelsize=8, length=3)
    colorbar.outline.set_linewidth(0.6)

    figure.text(left / figure_w, 1 - 0.30 / figure_h,
                config.get("title", "Attention scaling across key dimension and expansion degree"),
                fontsize=15, weight="bold", va="top", color="#172334")
    any_legacy = any(row.get("legacy") for row in indexed.values())
    figure.text(left / figure_w, 1 - 0.66 / figure_h,
                "Terminal validation perplexity; each cell is one requested configuration."
                + (" Outlined cells predate the current defaults and will be rerun." if any_legacy else "")
                + (f"\n{config['caption']}" if config.get("caption") else ""),
                fontsize=9.5, va="top", color="#475569", linespacing=1.3)
    legend = [Patch(facecolor=face, edgecolor=edge, hatch=hatch, label=label)
              for (face, edge, hatch), label in zip(
                  _STATUS_STYLE.values(),
                  ("Numerical failure (step shown)", "Not run", "Incomplete"))]
    if any_legacy:
        legend.append(Patch(facecolor="white", edgecolor=_LEGACY_EDGE, linewidth=_LEGACY_WIDTH, label=_LEGACY_LABEL))
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.48, 0.025),
                  ncol=4, frameon=False, fontsize=9, columnspacing=2.0, handlelength=1.8)
    return figure


def render_heatmap(rows: list[dict], config: dict, output_dir: Path) -> dict[str, Path]:
    """Write deterministic PNG and PDF copies of the combined terminal-PPL heatmap."""
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {f"combined_terminal_ppl_{extension}": figures_dir / f"combined_terminal_ppl.{extension}"
               for extension in ("png", "pdf")}
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False,
                         "pdf.fonttype": 42, "figure.facecolor": "white",
                         "savefig.facecolor": "white"}):
        figure = _build_figure(rows, config)
        try:
            figure.savefig(outputs["combined_terminal_ppl_png"], dpi=180,
                           metadata={"Software": "functional-memory"})
            figure.savefig(outputs["combined_terminal_ppl_pdf"], dpi=180,
                           metadata={"CreationDate": None, "ModDate": None,
                                     "Creator": "functional-memory"})
        finally:
            plt.close(figure)
    return outputs
