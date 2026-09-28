"""Report the joint query/key-width and Taylor-degree experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from functools import partial

from scaffolds.transformer.analysis import (
    committed_records, selected_best, selected_run_directory, selected_views, update_results_readme,
)

locate_run = partial(selected_run_directory, Path(__file__).resolve().parent)

EXPERIMENT = "key_dimension_taylor"
WIDTHS = (8, 16, 32, 64, 128)
DEGREES = (2, 4, 8, 16)
TERMINAL_STEP = 19_074


def summarize_run(run: dict) -> tuple[dict, list[dict]]:
    """Accept only committed, identity-matched terminal validation as an endpoint."""
    workdir = Path(run["workdir"])
    source, degree = run["source_commit"], run["attention_degree"]
    records = committed_records(workdir)
    for record in records:
        if (record.get("source_commit") != source or record.get("qk_head_dim") != run["qk_head_dim"]
                or record.get("parameter_count") != run["parameter_count"]
                or record.get("attention_degree") != degree):
            raise ValueError(f"Metric identity does not match joint sweep: {run['name']}")
    metadata_path = workdir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    raw_status = metadata.get("status", "incomplete" if records else "pending")
    error = metadata.get("error") or ""
    failure_kind = metadata.get("failure_kind", "")
    if run["failure"] is not None:
        failure_kind, raw_status = "numerical", "failed"
        error = run["failure"].get("error", "Numerical failure")
    if raw_status in ("failed", "ineligible"):
        failure_kind = failure_kind or "infrastructure"
        status = "failed" if failure_kind == "numerical" else "incomplete"
        if status == "incomplete":
            raw_status = "interrupted"
    else:
        status = "pending" if raw_status == "pending" else "incomplete"
    row = {
        "run": run["name"], "qk_head_dim": run["qk_head_dim"], "attention_degree": degree,
        "kernel": "softmax" if degree is None else f"m={degree}",
        "value_head_dim": run["value_head_dim"], "parameter_count": run["parameter_count"],
        "source_commit": source, "status": status, "execution_status": raw_status, "failure_kind": failure_kind, "error": error,
        "terminal_step": "", "training_tokens": "", "terminal_nll": "", "terminal_ppl": "",
        "delta_ppl_vs_softmax": "", "best_step": "", "best_ppl": "", "latest_tokens_per_second": "",
    }
    best = selected_best(run, records)
    if best is not None:
        row.update(best_step=best["step"], best_ppl=best["validation_perplexity"],
                   latest_tokens_per_second=records[-1].get("tokens_per_second", ""))
    completion_path = workdir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        final = next((record for record in records if record["step"] == TERMINAL_STEP), None)
        if (run["config"]["optim"]["total_steps"] != TERMINAL_STEP
                or completion.get("final_step") != TERMINAL_STEP
                or completion.get("training_tokens") != run["training_tokens"]
                or completion.get("source_commit") != source
                or completion.get("qk_head_dim") != run["qk_head_dim"]
                or completion.get("attention_degree") != degree
                or completion.get("parameter_count") != run["parameter_count"]
                or final is None or final["training_tokens"] != run["training_tokens"]
                or any(completion.get(key) != final[key] for key in ("validation_nll", "validation_perplexity"))):
            raise ValueError(f"Completion lacks matching terminal validation: {completion_path}")
        row.update(status="complete", execution_status="complete", failure_kind="", error="", terminal_step=TERMINAL_STEP,
                   training_tokens=final["training_tokens"], terminal_nll=final["validation_nll"],
                   terminal_ppl=final["validation_perplexity"])
    return row, records


def placeholder_row(width: int, degree: int | None) -> dict:
    """An untrained coordinate of a partial grid; never imputed."""
    return {"run": "", "qk_head_dim": width, "attention_degree": degree,
            "kernel": "softmax" if degree is None else f"m={degree}", "status": "not run",
            "terminal_ppl": "", "delta_ppl_vs_softmax": ""}


def grid_rows(rows: list[dict], *, widths=WIDTHS, degrees=DEGREES, allow_missing=False) -> list[list[dict]]:
    """Return ascending widths and degrees, with the softmax reference last."""
    widths, degrees = tuple(sorted(widths)), tuple(sorted(degrees))
    if not widths or not degrees or len(set(widths)) != len(widths) or len(set(degrees)) != len(degrees):
        raise ValueError("Grid dimensions must be nonempty and unique")
    cells = {}
    for row in rows:
        key = (row["qk_head_dim"], row["attention_degree"])
        if key in cells or key[0] not in widths or key[1] not in (*degrees, None):
            raise ValueError(f"Duplicate or unsupported grid coordinate: {key}")
        cells[key] = row
    expected = {(width, degree) for width in widths for degree in (*degrees, None)}
    if set(cells) != expected:
        if not allow_missing:
            raise ValueError(f"Joint grid is incomplete: {sorted(expected - set(cells), key=str)}")
        cells.update({key: placeholder_row(*key) for key in expected - set(cells)})
    for width in widths:
        reference = cells[width, None]
        for degree in degrees:
            row = cells[width, degree]
            if row["status"] == reference["status"] == "complete":
                row["delta_ppl_vs_softmax"] = row["terminal_ppl"] - reference["terminal_ppl"]
    return [[cells[width, degree] for degree in (*degrees, None)] for width in widths]


def aggregate(manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import numpy as np

    output_dir = output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir
    figures_dir.mkdir(exist_ok=True)
    all_runs = [*manifest["runs"], *manifest.get("controls", [])]
    summaries = [summarize_run(run) for run in all_runs]
    widths = tuple(sorted(manifest.get("key_dimensions", {run["qk_head_dim"] for run in manifest["runs"]})))
    degrees = tuple(sorted(manifest.get("degrees", {run["attention_degree"] for run in manifest["runs"]
                                                    if run["attention_degree"] is not None})))
    # The grid is every declared width crossed with every declared degree plus softmax.
    in_grid = [run["qk_head_dim"] in widths and (run["attention_degree"] is None or run["attention_degree"] in degrees)
               for run in all_runs]
    n_degrees = len(degrees)
    grid = grid_rows([row for (row, _), keep in zip(summaries, in_grid) if keep], widths=widths, degrees=degrees,
                     allow_missing=manifest.get("grid_coverage") == "partial")
    grid_members = [row for line in grid for row in line if row["run"]]
    grid_run_names = {row["run"] for row in grid_members}
    extras = [row for (row, _), keep in zip(summaries, in_grid) if not keep]
    references = {line[-1]["qk_head_dim"]: line[-1] for line in grid}
    for row in extras:
        reference = references.get(row["qk_head_dim"])
        if (row["attention_degree"] is not None and row["status"] == "complete"
                and reference is not None and reference["status"] == "complete"):
            row["delta_ppl_vs_softmax"] = row["terminal_ppl"] - reference["terminal_ppl"]
    rows = sorted(grid_members + extras, key=lambda row: (row["qk_head_dim"], 99 if row["attention_degree"] is None else row["attention_degree"], row["run"]))
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], "grid_cell"])
        writer.writeheader()
        writer.writerows([{**row, "grid_cell": row in grid_members} for row in rows])

    def save(figure, name):
        for extension in ("png", "pdf"):
            # PDFs omit their creation timestamp so regenerated figures are byte-identical.
            figure.savefig(figures_dir / f"{name}.{extension}", dpi=180, bbox_inches="tight",
                           metadata={"CreationDate": None} if extension == "pdf" else None)
        plt.close(figure)

    values = np.array([[row["terminal_ppl"] if row["status"] == "complete" else np.nan for row in line] for line in grid])
    finite = values[np.isfinite(values)]
    minimum, maximum = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    # A far outlier (degree 0) would flatten every other cell. Cap the color scale
    # at 1.5x the median; larger cells take the over-range color, the colorbar
    # shows the extension, and each cell still prints its exact value.
    capped = [value for value in finite if value <= 1.5 * float(np.median(finite))] if len(finite) else []
    top = max(capped) if capped else maximum
    if minimum == top:
        minimum, top = minimum - 0.5, top + 0.5
    norm = Normalize(minimum, top)
    from matplotlib.colors import ListedColormap
    # One hue, light to dark; the lightest step stays distinct from gray untrained cells.
    cmap = ListedColormap(plt.colormaps["Blues"](np.linspace(0.18, 0.92, 256)))
    cmap.set_over(plt.colormaps["Blues"](1.0))
    cmap.set_bad("#eeeeee")
    figure = plt.figure(figsize=(10, 6.5), layout="constrained")
    layout = figure.add_gridspec(1, 3, width_ratios=(n_degrees, 1, 0.13), wspace=0.10)
    axes = [figure.add_subplot(layout[0, 0]), figure.add_subplot(layout[0, 1])]
    for axis, columns in zip(axes, (range(n_degrees), range(n_degrees, n_degrees + 1))):
        columns = list(columns)
        image = axis.imshow(values[:, columns], norm=norm, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(columns)), [f"m={degrees[col]}" if col < n_degrees else "Softmax" for col in columns])
        axis.set_yticks(range(len(widths)), [str(width) for width in widths] if axis == axes[0] else [])
        for row_index, line in enumerate(grid):
            for x, column in enumerate(columns):
                row = line[column]
                value = values[row_index, column]
                annotation = f"{value:.4f}" if np.isfinite(value) else "—" if row["status"] == "not run" else row["status"].upper()
                color = "white" if np.isfinite(value) and norm(value) > 0.58 else "black"
                axis.text(x, row_index, annotation, ha="center", va="center", fontsize=9, color=color)
        axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(widths), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Query/key dimension per head (dK)")
    axes[0].set_title("Taylor degree")
    axes[1].set_title("Reference")
    figure.colorbar(image, cax=figure.add_subplot(layout[0, 2]), extend="max" if maximum > top else "neither",
                    label="Terminal held-out perplexity (linear scale" + (f"; ▲ above {top:.1f}" if maximum > top else "") + ")")
    figure.suptitle("Joint dK × Taylor degree · 19,074 updates · seed 0")
    save(figure, "terminal_ppl_heatmap")

    deltas = np.array([[row["delta_ppl_vs_softmax"] if row["delta_ppl_vs_softmax"] != "" else np.nan for row in line[:n_degrees]] for line in grid])
    # Scale deltas over the cells within the terminal-perplexity color range; the
    # same over-range outliers take the extreme colors and keep printed values.
    in_range = np.isfinite(deltas) & (values[:, :n_degrees] <= top)
    available = np.abs(deltas[in_range]) if in_range.any() else np.abs(deltas[np.isfinite(deltas)])
    extent = max(float(available.max()) if len(available) else 1.0, 1e-8)
    beyond = bool(np.any(np.abs(deltas[np.isfinite(deltas)]) > extent))
    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
    delta_cmap = plt.colormaps["RdBu_r"].copy()
    delta_cmap.set_over(plt.colormaps["RdBu_r"](1.0))
    delta_cmap.set_under(plt.colormaps["RdBu_r"](0.0))
    delta_cmap.set_bad("#eeeeee")
    image = axis.imshow(deltas, cmap=delta_cmap, norm=TwoSlopeNorm(0.0, -extent, extent), aspect="auto")
    for i, line in enumerate(grid):
        for j, row in enumerate(line[:n_degrees]):
            value = deltas[i, j]
            annotation = (f"{value:+.4f}" if np.isfinite(value) else "—" if row["status"] == "not run"
                          else row["status"].upper() if row["status"] != "complete" else "NO REF")
            axis.text(j, i, annotation, ha="center", va="center", fontsize=9,
                      color="white" if np.isfinite(value) and abs(value) > 0.65 * extent else "black")
    axis.set_xticks(range(n_degrees), [f"m={degree}" for degree in degrees])
    axis.set_yticks(range(len(widths)), [str(width) for width in widths])
    axis.set_ylabel("Query/key dimension per head (dK)")
    axis.set_title("Terminal perplexity − softmax at the same dK\nNegative is better")
    figure.colorbar(image, ax=axis, extend="both" if beyond else "neither",
                    label="Δ perplexity (linear scale" + (f"; arrows beyond ±{extent:.2f}" if beyond else "") + ")")
    save(figure, "delta_ppl_heatmap")

    curve_rows = (len(widths) + 3) // 3
    figure, axes = plt.subplots(curve_rows, 3, figsize=(14, 4 * curve_rows), layout="constrained", squeeze=False)
    for axis, width in zip(axes.flat, widths):
        for row, records in summaries:
            if row["run"] not in grid_run_names or row["qk_head_dim"] != width or not records:
                continue
            degree = row["attention_degree"]
            axis.plot([record["training_tokens"] / 1e9 for record in records], [record["validation_perplexity"] for record in records],
                      label=row["kernel"],
                      color="black" if degree is None else f"C{degrees.index(degree)}",
                      linestyle="-" if row["status"] == "complete" else "--")
        axis.set_title(f"dK={width}")
        axis.set_xlabel("Prediction tokens (billions)")
        axis.set_ylabel("Held-out perplexity")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(widths):]:
        axis.axis("off")
    if len(widths) < axes.size:
        axes.flat[len(widths)].text(0, 0.8, "Dashed curves: incomplete or failed\nEndpoints require 19,074 updates", va="top")
    save(figure, "learning_curves")

    count = {status: sum(row["status"] == status for row in rows) for status in ("complete", "pending", "incomplete", "failed")}
    header = "| dK | " + " | ".join(f"m={degree}" for degree in degrees) + " | Softmax |"
    table = [header, "|" + "---:|" * (n_degrees + 2)]
    for width, line in zip(widths, grid):
        cells = [f"{row['terminal_ppl']:.4f}" if row["status"] == "complete" else "—" if row["status"] == "not run" else row["status"].upper()
                 for row in line]
        table.append(f"| {width} | " + " | ".join(cells) + " |")
    references = {width: line[-1] for width, line in zip(widths, grid)}
    lines = [f"Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0. "
             f"{count['complete']} of {len(rows)} retained runs are complete"
             + (f"; {count['pending']} pending, {count['incomplete']} incomplete, {count['failed']} numerical failures" if len(rows) != count["complete"] else "")
             + f". The heatmap shows the {len(widths) * (n_degrees + 1)}-cell joint comparison;"
             + " other runs remain in the CSV."
             + (" A dash marks a coordinate that was never trained; nothing is imputed." if manifest.get("grid_coverage") == "partial" else ""),
             "", *table, ""]
    if extras:
        lines += ["Runs retained outside the grid: " + "; ".join(
            f"`{row['run']}` ({row['kernel']}, dK={row['qk_head_dim']}, PPL {row['terminal_ppl']:.4f})" for row in extras)
            + ". The grid uses one softmax reference per width"
            + (f"; at dK=64 it is `{references[64]['run']}` (PPL {references[64]['terminal_ppl']:.4f})" if 64 in references and references[64]["run"] else "")
            + ".", ""]
    lines += ["Each delta in the CSV and delta heatmap subtracts the softmax endpoint at the same width. "
              "FAILED means a recorded numerical failure; interrupted or unaudited runs remain INCOMPLETE. "
              f"Source commits, exact recipes, and artifact checksums are in each run’s `config.yaml` and `record.json` under `experiments/{EXPERIMENT}/runs/`.", ""]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["Recorded failures or interruptions:", "", *[f"- `{row['run']}` ({row['failure_kind'] or row['execution_status']}): {row['error']}" for row in failures], ""]
    unretained = [run["name"] for run in all_runs if not (run.get("selected_checkpoint") or {}).get("path")]
    if unretained:
        lines += [f"Checkpoints and logs are not retained for {len(unretained)} runs trained on another cluster "
                  f"({', '.join(f'`{name}`' for name in unretained)}); their metrics, completion records, metadata, and recipes are retained and checksum-verified.", ""]
    lines += ["[Summary CSV](summary.csv) · [Terminal heatmap](terminal_ppl_heatmap.pdf) · [Delta heatmap](delta_ppl_heatmap.pdf) · [Learning curves](learning_curves.pdf)", ""]
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/key_dimension_taylor/sweep.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(selected_views(Path(__file__).resolve().parent, args.manifest), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
