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
             + f". The table shows the {len(widths) * (n_degrees + 1)}-cell joint comparison;"
             + " other runs remain in the CSV."
             + (" A dash marks a coordinate that was never trained; nothing is imputed." if manifest.get("grid_coverage") == "partial" else ""),
             "", *table, ""]
    if extras:
        lines += ["Runs retained outside the grid: " + "; ".join(
            f"`{row['run']}` ({row['kernel']}, dK={row['qk_head_dim']}, PPL {row['terminal_ppl']:.4f})" for row in extras)
            + ". The grid uses one softmax reference per width"
            + (f"; at dK=64 it is `{references[64]['run']}` (PPL {references[64]['terminal_ppl']:.4f})" if 64 in references and references[64]["run"] else "")
            + ".", ""]
    lines += ["Each delta in the CSV subtracts the softmax endpoint at the same width. "
              "FAILED means a recorded numerical failure; interrupted or unaudited runs remain INCOMPLETE. "
              f"Source commits, exact recipes, and artifact checksums are in each run’s `config.yaml` and `record.json` under `experiments/attention_scaling/runs/`.", ""]
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["Recorded failures or interruptions:", "", *[f"- `{row['run']}` ({row['failure_kind'] or row['execution_status']}): {row['error']}" for row in failures], ""]
    unretained = [run["name"] for run in all_runs if not (run.get("selected_checkpoint") or {}).get("path")]
    if unretained:
        lines += [f"Checkpoints and logs are not retained for {len(unretained)} runs trained on another cluster "
                  f"({', '.join(f'`{name}`' for name in unretained)}); their metrics, completion records, metadata, and recipes are retained and checksum-verified.", ""]
    lines += ["[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf) · "
              "[Heatmap](../attention_scaling/combined_terminal_ppl.png) in the attention comparison", ""]
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/attention_scaling/sweep.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(selected_views(Path(__file__).resolve().parent, args.manifest), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
