"""Regenerate the polynomial Gated DeltaNet comparison from its pinned runs.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.gdn_polynomial.plot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs
from scaffolds.transformer.models.initialization import REFERENCE


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
FIELDS = (
    "run_id", "role", "label", "mechanism", "d_k", "erase", "write", "degree", "control_run_id", "status",
    "last_step", "last_validation_perplexity", "terminal_validation_nll", "terminal_validation_perplexity",
    "best_validation_step", "best_validation_perplexity", "percent_vs_tied_gdn", "percent_vs_control",
    "parameter_count", "gradient_clipped_fraction", "gradient_norm_max", "gradient_spike_count",
    "metrics_sha256", "completion_sha256",
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Cells without a terminal value: face, edge, hatch, legend label.
STATUS_STYLE = {
    "numerical_failure": ("#f5c5bd", "#b44c40", "xx", "Numerical failure (update shown)"),
    "incomplete": ("#f9e5ae", "#bb8b28", "..", "Incomplete (last update shown)"),
    "cancelled": ("#e8e1e8", "#9a8295", "--", "Cancelled (last update when available)"),
    "pending": ("#dbe7f3", "#6f8fb0", "\\\\", "Pending (held or queued)"),
    "not_run": ("#e5e8ec", "#929aa4", "//", "Not run"),
}
# Hue follows the write (degree and linear activation): validated categorical slots 1, 2, 3, 7, 4.
WRITES = {(1, "sigmoid"): ("m = 1", "#2a78d6", "o"), (1, "tanh"): ("m = 1, tanh write", "#eb6834", "^"),
          (2, "sigmoid"): ("m = 2", "#1baf7a", "D"), (3, "sigmoid"): ("m = 3", "#4a3aa7", "s"),
          (4, "sigmoid"): ("m = 4", "#eda100", "v")}
# Line style follows the key width.
WIDTHS = {64: (0, ()), 16: (0, (5, 2.5)), 8: (0, (1.2, 1.8))}
ERASE_NAMES = {"tied": "Tied to the write gate", "independent": "Independent"}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _match(value: object, expected: object, description: str) -> None:
    if value != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {value!r}")


def _check_recipe(entry: dict, config: dict, label: str) -> None:
    """The recipe must be the declared mechanism, width, erase gate, write, and degree."""
    model = config["model"]
    _match(model.get("qk_head_dim"), entry["d_k"], f"{label} d_K")
    mechanism = model.get("attention_mechanism", "dot_product")
    _match(model.get("initialization"), REFERENCE[mechanism], f"{label} initialization")
    if entry["mechanism"] == "softmax":
        _match(mechanism, "dot_product", f"{label} mechanism")
        _match((model.get("attention_degree"), model.get("attention_normalization")), (None, True),
               f"{label} softmax kernel")
        _match(model.get("qk_norm"), True, f"{label} QK-norm")
        return
    _match(mechanism, entry["mechanism"], f"{label} mechanism")
    _match(model.get("gdn_erase_mode", "tied"), entry["erase"], f"{label} erase gate")
    _match(model.get("gdn_write_activation", "sigmoid"), entry["write"], f"{label} linear write activation")
    _match(model.get("gdn_polynomial_degree", 1), entry["degree"], f"{label} GDN degree")


def _gradients(validations: list[dict]) -> dict:
    """Clipped fraction, maximum norm, and spike count over every logged update."""
    windows = [record for record in validations if record.get("train_gradient_norm_count")]
    if not windows:
        return {"gradient_clipped_fraction": None, "gradient_norm_max": None, "gradient_spike_count": None}
    count = sum(record["train_gradient_norm_count"] for record in windows)
    return {"gradient_clipped_fraction": sum(record["train_gradient_clipped_fraction"]
                                             * record["train_gradient_norm_count"] for record in windows) / count,
            "gradient_norm_max": max(record["train_gradient_norm_max"] for record in windows),
            "gradient_spike_count": sum(record.get("train_gradient_spike_count", 0) for record in windows)}


def load_rows(root: Path, selection: dict) -> list[dict]:
    """One checked row per selected run: outcome, validation curve, terminal and best metrics."""
    study = Path(selection["study"]).name
    entries = selection["runs"]
    if len({entry["run_id"] for entry in entries}) != len(entries):
        raise ValueError("A run is selected twice")
    verify_pinned_runs(root, [dict(entry, study=study) for entry in entries])
    step, tokens = selection["terminal_step"], selection["training_tokens"]
    rows = []
    for entry in entries:
        name = entry["run_id"]
        label = f"{study}/{name}"
        directory = root / "experiments" / study / "runs" / name
        _check_recipe(entry, yaml.safe_load((directory / "config.yaml").read_text()), label)
        record = load_run(directory)
        recorded = record["status"]
        failures = sorted(directory.glob("numerical_failure_step_*.json"))
        if ((recorded == "complete") != (directory / "completion.json").exists()
                or (recorded == "numerical_failure") != bool(failures)):
            raise ValueError(f"Outcome files of {label} disagree with its recorded status {recorded!r}")
        metrics = directory / "metrics.jsonl"
        lines = metrics.read_text().splitlines() if metrics.exists() else []
        validations = [line for line in map(json.loads, filter(str.strip, lines)) if line.get("event") == "validation"]
        if validations:
            _match(validations[-1].get("source_commit"), record["source_commit"], f"{label} source commit")
            metadata = _json(directory / "metadata.json")
            for key, value in selection["data_identity"].items():
                _match(metadata.get("data_identity", {}).get(key), value, f"{label} data identity {key}")
        status = (recorded if recorded in ("complete", "numerical_failure", "cancelled")
                  else "incomplete" if validations else "pending")
        row = {key: entry.get(key) for key in ("role", "label", "mechanism", "d_k", "erase", "write", "degree")}
        row.update(run_id=name, control_run_id=entry.get("control"), status=status,
                   curve=[(line["step"], line["validation_perplexity"]) for line in validations],
                   last_step=validations[-1]["step"] if validations else None,
                   last_validation_perplexity=validations[-1]["validation_perplexity"] if validations else None,
                   terminal_validation_nll=None, terminal_validation_perplexity=None,
                   best_validation_step=None, best_validation_perplexity=None,
                   parameter_count=validations[-1].get("parameter_count") if validations else None,
                   failure_step=_json(failures[-1])["step"] if failures else None,
                   metrics_sha256=_sha256(metrics) if metrics.exists() else None, completion_sha256=None,
                   **_gradients(validations))
        if status == "complete":
            completion = _json(directory / "completion.json")
            _match((completion.get("final_step"), completion.get("training_tokens")), (step, tokens),
                   f"{label} terminal budget")
            last = validations[-1]
            _match((completion["validation_nll"], completion["validation_perplexity"]),
                   (last["validation_nll"], last["validation_perplexity"]), f"{label} terminal validation")
            best = _json(directory / "best_validation.json")
            row.update(terminal_validation_nll=completion["validation_nll"],
                       terminal_validation_perplexity=completion["validation_perplexity"],
                       best_validation_step=best["step"], best_validation_perplexity=best["validation_perplexity"],
                       completion_sha256=_sha256(directory / "completion.json"))
        rows.append(row)
    by_run = {row["run_id"]: row for row in rows}
    tied = {row["d_k"]: row for row in rows if row["role"] == "reference" and row["erase"] == "tied"}
    for row in rows:
        reference, control = tied.get(row["d_k"]), by_run.get(row["control_run_id"])
        row["percent_vs_tied_gdn"] = (_percent(row, reference)
                                      if row["mechanism"] == "gated_deltanet" and reference is not row else None)
        row["percent_vs_control"] = _percent(row, control) if control else None
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percent(row: dict, baseline: dict | None) -> float | None:
    if baseline is None or row["terminal_validation_perplexity"] is None \
            or baseline["terminal_validation_perplexity"] is None:
        return None
    return 100 * (row["terminal_validation_perplexity"] / baseline["terminal_validation_perplexity"] - 1)


def _grid(rows: list[dict], selection: dict) -> tuple[list[tuple[str, list[tuple[str, object]]]], dict]:
    """Panels of (column label, key) and the row at each (key, d_K); keys select one grid run each."""
    panels = [("Reference mechanisms", [("softmax", "softmax"), ("Gated DeltaNet", "tied")]),
              ("Gated DeltaNet · independent erase, polynomial", [(f"m={m}", m) for m in selection["degrees"]])]
    cells = {}
    for row in rows:
        key = ("softmax" if row["mechanism"] == "softmax" else "tied" if row["role"] == "reference"
               else row["degree"] if row["role"] == "primary" else None)
        if key is None:
            continue
        if (key, row["d_k"]) in cells:
            raise ValueError(f"Two runs at grid cell {(key, row['d_k'])}")
        cells[key, row["d_k"]] = row
    return panels, cells


def _cell_text(row: dict | None) -> tuple[str, str]:
    """Annotation and status for one grid cell."""
    if row is None:
        return "NOT\nRUN", "not_run"
    if row["status"] == "complete":
        return f"{row['terminal_validation_perplexity']:.3f}", "complete"
    if row["status"] == "numerical_failure":
        return f"FAIL\n@{row['failure_step']:,}", "numerical_failure"
    if row["status"] == "incomplete":
        return f"INCOMPLETE\n@{row['last_step']:,}", "incomplete"
    if row["status"] == "cancelled":
        return ("CANCELLED" + (f"\n@{row['last_step']:,}" if row["last_step"] is not None else ""), "cancelled")
    return "PENDING", "pending"


def heatmap(rows: list[dict], selection: dict):
    """Terminal perplexity by key width: references beside polynomial Gated DeltaNet."""
    panels, cells = _grid(rows, selection)
    widths = selection["d_k"]
    values = [row["terminal_validation_perplexity"] for row in cells.values() if row["status"] == "complete"]
    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = matplotlib.colormaps["Blues"].resampled(256)
    cell_w, cell_h, gap, left = 0.95, 0.62, 0.45, 1.15
    content_w = sum(len(columns) for _, columns in panels) * cell_w + gap * (len(panels) - 1)
    figure_w, axis_h = left + content_w + 1.6, len(widths) * cell_h
    bottom = 0.95
    figure_h = bottom + axis_h + 1.25
    figure = plt.figure(figsize=(figure_w, figure_h), facecolor="white")
    x = left
    for index, (title, columns) in enumerate(panels):
        axis_w = len(columns) * cell_w
        axis = figure.add_axes((x / figure_w, bottom / figure_h, axis_w / figure_w, axis_h / figure_h))
        axis.set_gid(f"panel:{index}")
        grid = np.full((len(widths), len(columns)), np.nan)
        for i, width in enumerate(widths):
            for j, (_, key) in enumerate(columns):
                row = cells.get((key, width))
                text, status = _cell_text(row)
                if status == "complete":
                    grid[i, j] = row["terminal_validation_perplexity"]
                    rgb = cmap(norm(grid[i, j]))[:3]
                    color, weight = ("white" if np.dot(rgb, (0.2126, 0.7152, 0.0722)) < 0.5 else "#172334"), "normal"
                else:
                    face, edge, hatch, _ = STATUS_STYLE[status]
                    axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor=edge, linewidth=0,
                                             hatch=hatch, zorder=1))
                    color, weight = "#222b36", "bold"
                axis.text(j, i, text, ha="center", va="center", color=color, weight=weight, linespacing=0.95,
                          fontsize=8.5 if status == "complete" else 6.9, zorder=4, gid=f"cell:{key}:{width}:{status}")
        axis.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, origin="upper", aspect="auto",
                    interpolation="none", zorder=2)
        axis.set_xticks(range(len(columns)), [name for name, _ in columns], fontsize=8.7)
        axis.set_yticks(range(len(widths)), [str(width) for width in widths] if index == 0 else [], fontsize=9.5)
        if index == 0:
            axis.set_ylabel("Key dimension $d_k$", fontsize=10, labelpad=10)
        axis.set_title(title, fontsize=11, weight="bold", pad=10, loc="left")
        axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(widths), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5, zorder=3)
        axis.tick_params(which="both", length=0, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        x += axis_w + gap
    color_axis = figure.add_axes(((left + content_w + 0.5) / figure_w, bottom / figure_h, 0.2 / figure_w,
                                  axis_h / figure_h))
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.set_label("Terminal held-out perplexity\n(lower is better)", fontsize=9, labelpad=8)
    colorbar.ax.tick_params(labelsize=8, length=3)
    colorbar.outline.set_linewidth(0.6)
    figure.text(left / figure_w, 1 - 0.3 / figure_h, "Polynomial Gated DeltaNet across key dimension",
                fontsize=14, weight="bold", va="top", color="#172334")
    figure.text(left / figure_w, 1 - 0.66 / figure_h,
                "Terminal validation perplexity at 19,074 updates, seed 0; each cell is one run.",
                fontsize=9.5, va="top", color="#475569")
    shown = {_cell_text(cells.get((key, width)))[1] for _, columns in panels for _, key in columns for width in widths}
    legend = [Patch(facecolor=face, edgecolor=edge, hatch=hatch, label=name)
              for status, (face, edge, hatch, name) in STATUS_STYLE.items() if status in shown]
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(legend), frameon=False,
                  fontsize=8.5, columnspacing=1.6, handlelength=1.8)
    return figure


def validation_gap(rows: list[dict]):
    """Validation perplexity relative to tied Gated DeltaNet at the same d_K and update."""
    tied = {row["d_k"]: dict(row["curve"]) for row in rows if row["role"] == "reference" and row["erase"] == "tied"}
    series = [row for row in rows if row["mechanism"] == "gated_deltanet" and row["role"] in ("primary", "variant")
              and row["curve"] and row["d_k"] in tied]
    series.sort(key=lambda row: (-row["d_k"], row["degree"], row["write"]))
    figure, axis = plt.subplots(figsize=(8.0, 5.2), facecolor="white")
    figure.subplots_adjust(left=0.11, right=0.84, top=0.85, bottom=0.27)
    axis.axhline(0, color=MUTED, linewidth=1.0, zorder=1)
    ends = []
    for row in series:
        name, color, marker = WRITES[row["degree"], row["write"]]
        reference = tied[row["d_k"]]
        points = [(step, 100 * (value / reference[step] - 1)) for step, value in row["curve"] if step in reference]
        steps, gaps = zip(*points)
        axis.plot(np.array(steps) / 1000, gaps, color=color, linestyle=WIDTHS[row["d_k"]], linewidth=1.8,
                  solid_capstyle="round", dash_capstyle="round", zorder=2, gid=f"curve:{row['run_id']}")
        axis.plot([steps[-1] / 1000], [gaps[-1]], marker=marker, color=color, markersize=6.5, linestyle="none",
                  markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ends.append((steps[-1] / 1000, gaps[-1], f"{gaps[-1]:+.1f}%"
                     + ("" if row["status"] == "complete" else f" @{steps[-1]:,}"), row["status"] == "complete"))
    for x, y, text, complete in ends:
        axis.annotate(text, (x, y), xytext=(7, 0) if complete else (4, -11), textcoords="offset points",
                      va="center", fontsize=7.5, color=INK,
                      zorder=4, bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none",
                                      "alpha": 0.85}, annotation_clip=False)
    axis.set_xlim(0, 20.2)
    axis.set_xticks(range(0, 21, 4))
    axis.set_xlabel("Update (thousands)", fontsize=9.5, color=INK)
    axis.set_ylabel("Validation perplexity vs tied Gated DeltaNet (%)", fontsize=9.5, color=INK)
    axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    figure.text(0.11, 0.965, "Polynomial Gated DeltaNet against tied Gated DeltaNet", fontsize=13, weight="bold",
                color=INK, va="top")
    figure.text(0.11, 0.915, "Validation perplexity relative to the tied reference at the same key width and update; "
                "seed 0.\n0 is the tied reference; each end value is the last validation, with its update if incomplete.",
                fontsize=8.5, color=MUTED, va="top", linespacing=1.3)
    style = {"loc": "lower left", "frameon": False, "fontsize": 7.8, "title_fontsize": 8, "alignment": "left",
             "handlelength": 3.2}
    used_writes = sorted({(row["degree"], row["write"]) for row in series}, key=list(WRITES).index)
    figure.legend(handles=[Line2D([], [], color=WRITES[key][1], marker=WRITES[key][2], linewidth=1.8, markersize=6,
                                  label=WRITES[key][0]) for key in used_writes],
                  title="Write (independent erase)", bbox_to_anchor=(0.1, 0.0), ncol=2, **style)
    used_widths = sorted({row["d_k"] for row in series}, reverse=True)
    figure.legend(handles=[Line2D([], [], color=MUTED, linestyle=WIDTHS[width], linewidth=1.8,
                                  label=f"$d_k$ = {width}") for width in used_widths],
                  title="Key width", bbox_to_anchor=(0.55, 0.0), **style)
    return figure


def _save(figure, name: str, output: Path) -> dict[str, Path]:
    paths = {extension: output / f"{name}.{extension}" for extension in ("png", "pdf")}
    try:
        figure.savefig(paths["png"], dpi=180, metadata={"Software": "functional-memory"})
        figure.savefig(paths["pdf"], dpi=180, metadata={"CreationDate": None, "ModDate": None,
                                                        "Creator": "functional-memory"})
    finally:
        plt.close(figure)
    return paths


def _value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _tables(rows: list[dict], selection: dict) -> str:
    panels, cells = _grid(rows, selection)
    columns = [column for _, panel_columns in panels for column in panel_columns]
    def cell(row):
        if row is None:
            return "—"
        if row["status"] == "complete":
            return f"{row['terminal_validation_perplexity']:.3f}"
        if row["status"] == "incomplete":
            return f"incomplete (update {row['last_step']:,}: {row['last_validation_perplexity']:.2f})"
        if row["status"] == "numerical_failure":
            return f"fail @{row['failure_step']:,}"
        if row["status"] == "cancelled":
            return (f"cancelled (update {row['last_step']:,}: {row['last_validation_perplexity']:.2f})"
                    if row["last_step"] is not None else "cancelled")
        return "pending"
    grid = ["| d_K | " + " | ".join(name.replace("m=", "GDN m = ") for name, _ in columns) + " |",
            "| ---: | " + " | ".join("---:" for _ in columns) + " |"]
    grid += [f"| {width} | " + " | ".join(cell(cells.get((key, width))) for _, key in columns) + " |"
             for width in selection["d_k"]]
    linear = [row for row in rows if row["mechanism"] == "gated_deltanet" and row["degree"] == 1
              and row["d_k"] == 64 and row["status"] == "complete"]
    variants = ["| Erase gate | Linear write | Run | Terminal PPL | vs tied reference |",
                "| --- | --- | --- | ---: | ---: |"]
    variants += [f"| {ERASE_NAMES[row['erase']]} | {row['write']} | `{row['run_id']}` | "
                 f"{row['terminal_validation_perplexity']:.3f} | "
                 + ("" if row["percent_vs_tied_gdn"] is None else f"{row['percent_vs_tied_gdn']:+.2f}%") + " |"
                 for row in linear]
    counts = {status: sum(_cell_text(cells.get((key, width)))[1] == status for _, key in columns
                          for width in selection["d_k"]) for status in ("complete", *STATUS_STYLE)}
    return ("### Terminal perplexity\n\n"
            f"{sum(counts.values())} cells: {counts['complete']} complete, {counts['incomplete']} incomplete, "
            f"{counts['pending']} pending, {counts['cancelled']} cancelled, {counts['not_run']} not run, "
            f"{counts['numerical_failure']} failed.\n\n"
            "![Terminal perplexity by key width](terminal_ppl.png)\n\n[PDF](terminal_ppl.pdf)\n\n"
            + "\n".join(grid) + "\n\n"
            "### Against tied Gated DeltaNet\n\n"
            "![Validation perplexity against tied Gated DeltaNet](validation_gap.png)\n\n"
            "[PDF](validation_gap.pdf) · [Per-run measurements](summary.csv)\n\n"
            "### Degree-one variants (d_K = 64)\n\n" + "\n".join(variants) + "\n")


def update_readme(path: Path, block: str) -> None:
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    authored = path.read_text()
    if authored.count(begin) != 1 or authored.count(end) != 1:
        raise ValueError("README needs exactly one generated-results marker pair")
    prefix, rest = authored.split(begin, 1)
    _, suffix = rest.split(end, 1)
    path.write_text(prefix + begin + "\n" + block + end + suffix)


def build(root: Path = ROOT, output: Path = DIRECTORY) -> list[dict]:
    """Check every pinned input, then write both figures, summary.csv, and the README tables."""
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    rows = load_rows(root, selection)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        _save(heatmap(rows, selection), "terminal_ppl", output)
        _save(validation_gap(rows), "validation_gap", output)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: _value(row.get(field)) for field in FIELDS} for row in rows)
    update_readme(output / "README.md", _tables(rows, selection))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY)
    build(ROOT, parser.parse_args().output)


if __name__ == "__main__":
    main()
