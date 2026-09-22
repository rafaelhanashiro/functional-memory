"""Report the joint query/key-width and Taylor-degree experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scaffolds.transformer.studies.polynomial_attention.plot import committed_records
from scaffolds.transformer.studies.polynomial_attention.plot import default_manifest_path, update_results_readme
from scaffolds.transformer.studies.polynomial_attention.plot import artifact_path, checked_artifact, check_dependency_lock, retained_root

EXPERIMENT = "key_dimension_taylor"
WIDTHS = (8, 16, 32, 64, 128)
DEGREES = (2, 4, 8, 16)
TERMINAL_STEP = 19_074


def summarize_run(run: dict) -> tuple[dict, list[dict]]:
    """Accept only committed, identity-matched terminal validation as an endpoint."""
    workdir = Path(run["workdir"])
    if run.get("retained_completed"):
        if not run.get("provenance_sha256"):
            raise ValueError(f"Missing historical checksums: {run['name']}")
        for name, expected in run["provenance_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Historical artifact checksum changed: {name}")
    source = run.get("source_commit", run["config"]["train"]["source_commit"])
    degree = run["attention_degree"]
    records = committed_records(workdir, metrics_path=Path(run.get("metrics_path", workdir / "metrics.jsonl")))
    for record in records:
        if (record.get("source_commit") != source or record.get("qk_head_dim") != run["qk_head_dim"]
                or record.get("parameter_count") != run["parameter_count"]
                or record.get("attention_degree") != degree):
            raise ValueError(f"Metric identity does not match joint sweep: {run['name']}")
    metadata_path = workdir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    raw_status = metadata.get("status", "incomplete" if records or (workdir / "config.json").exists() else "pending")
    error = metadata.get("error") or ""
    failure_kind = metadata.get("failure_kind", "")
    failures = sorted(workdir.glob("numerical_failure_step_*.json"))
    preflight_failure = workdir / "preflight_failure.json"
    production_failure = workdir / "production_failure.json"
    if production_failure.exists():
        failure = json.loads(production_failure.read_text())
        failure_kind = failure.get("failure_kind", "numerical" if failure.get("numerical_failures") else "infrastructure")
        raw_status, error = "failed", failure.get("error", "Production interrupted")
    elif preflight_failure.exists():
        failure = json.loads(preflight_failure.read_text())
        failure_kind = failure.get("failure_kind", "numerical" if failure.get("reason") == "numerical" else "infrastructure")
        raw_status, error = "preflight-ineligible", failure.get("error", failure.get("reason", "Preflight failed"))
    elif failures:
        failure = json.loads(failures[-1].read_text())
        failure_kind = "numerical"
        raw_status, error = "failed", failure.get("error", "Numerical failure")
    if raw_status in ("failed", "preflight-ineligible", "ineligible"):
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
        "reused": bool(run.get("retained_completed")), "original_experiment": run.get("original_experiment", EXPERIMENT),
        "source_commit": source, "status": status, "execution_status": raw_status, "failure_kind": failure_kind, "error": error,
        "unavailable_historical_artifacts": json.dumps(run.get("unavailable_historical_artifacts", []), sort_keys=True),
        "terminal_step": "", "training_tokens": "", "terminal_nll": "", "terminal_ppl": "",
        "delta_ppl_vs_softmax": "", "best_step": "", "best_ppl": "", "latest_tokens_per_second": "",
    }
    if records:
        best = min(records, key=lambda record: (record["validation_nll"], -record["step"]))
        row.update(best_step=best["step"], best_ppl=best["validation_perplexity"],
                   latest_tokens_per_second=records[-1].get("tokens_per_second", ""))
        # A historical checkpoint's original selection remains authoritative on ties.
        selected = run.get("selected_checkpoint", run.get("retained_provenance", run.get("historical_provenance", {})).get("selected_checkpoint"))
        if run.get("retained_completed") and selected:
            chosen = next((record for record in records if record["step"] == selected["step"]), None)
            if chosen is None or chosen["validation_nll"] != best["validation_nll"]:
                raise ValueError(f"Historical checkpoint is not validation-best: {run['name']}")
            row.update(best_step=chosen["step"], best_ppl=chosen["validation_perplexity"])
    completion_path = Path(run.get("completion_path", workdir / "completion.json"))
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
    if run.get("retained_completed") and row["status"] != "complete":
        raise ValueError(f"Historical reference is incomplete: {run['name']}")
    return row, records


def grid_rows(rows: list[dict], *, widths=WIDTHS, degrees=DEGREES) -> list[list[dict]]:
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
        raise ValueError(f"Joint grid is incomplete: {sorted(expected - set(cells), key=str)}")
    for width in widths:
        reference = cells[width, None]
        for degree in degrees:
            row = cells[width, degree]
            if row["status"] == reference["status"] == "complete":
                row["delta_ppl_vs_softmax"] = row["terminal_ppl"] - reference["terminal_ppl"]
    return [[cells[width, degree] for degree in (*degrees, None)] for width in widths]


def summarize_retained_run(run: dict, root: Path) -> tuple[dict, list[dict]]:
    """Read a completed cell using portable, checksummed retained artifacts."""
    import yaml

    config = yaml.safe_load(checked_artifact(root, run["resolved_config"]).read_text())
    metrics = checked_artifact(root, run["metrics"])
    completion = checked_artifact(root, run["completion"])
    checked_artifact(root, run["metadata"])
    if (config["train"]["seed"] != run["seed"]
            or config["train"]["source_commit"] != run["source_commit"]
            or any(config["model"].get(key) != run[key]
                   for key in ("qk_head_dim", "value_head_dim", "attention_degree"))):
        raise ValueError(f"Retained configuration identity does not match: {run['name']}")
    tokens = config["optim"]["total_steps"] * config["data"]["batch_size"] * (config["data"]["sequence_length"] - 1)
    adapted = dict(run, config=config, workdir=str(metrics.parent), metrics_path=str(metrics),
                   completion_path=str(completion), training_tokens=tokens,
                   provenance_sha256={str(metrics): run["metrics"]["sha256"], str(completion): run["completion"]["sha256"]})
    row, records = summarize_run(adapted)
    terminal = run["terminal"]
    final = next((record for record in records if record["step"] == terminal["step"]), None)
    if row["status"] != "complete" or final is None or any(final[key] != value for key, value in terminal.items()):
        raise ValueError(f"Retained terminal metrics do not match: {run['name']}")
    selected = run["selected_checkpoint"]
    artifact_path(root, selected["path"])
    best = next((record for record in records if record["step"] == selected["step"]), None)
    if (best is None or best["validation_nll"] != min(record["validation_nll"] for record in records)
            or any(best[key] != selected[key] for key in ("validation_nll", "validation_perplexity"))):
        raise ValueError(f"Retained checkpoint is not validation-best: {run['name']}")
    row.update(best_step=selected["step"], best_ppl=selected["validation_perplexity"])
    return row, records


def aggregate(manifest: dict, output_dir: Path | None = None, *, manifest_path: Path | None = None) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import numpy as np

    retained = manifest.get("format") == "functional-memory-retained-experiment"
    root = retained_root(manifest, manifest_path) if retained else Path(manifest["project_root"])
    if retained:
        check_dependency_lock(manifest, root)
    output_dir = output_dir or root / "results" / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    summaries = [(summarize_retained_run(run, root) if retained else summarize_run(run))
                 for run in [*manifest["runs"], *manifest["controls"]]]
    widths = tuple(sorted(manifest.get("key_dimensions", {run["qk_head_dim"] for run in manifest["runs"]})))
    degrees = tuple(sorted(manifest.get("degrees", {run["attention_degree"] for run in manifest["runs"]})))
    n_degrees = len(degrees)
    grid = grid_rows([row for row, _ in summaries], widths=widths, degrees=degrees)
    rows = [row for line in grid for row in line]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def save(figure, name):
        for extension in ("png", "pdf"):
            figure.savefig(figures_dir / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    values = np.array([[row["terminal_ppl"] if row["status"] == "complete" else np.nan for row in line] for line in grid])
    finite = values[np.isfinite(values)]
    minimum, maximum = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if minimum == maximum:
        minimum, maximum = minimum - 0.5, maximum + 0.5
    norm = Normalize(minimum, maximum)
    cmap = plt.colormaps["viridis_r"].copy()
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
                annotation = f"{value:.4f}" if np.isfinite(value) else row["status"].upper()
                color = "white" if np.isfinite(value) and norm(value) > 0.58 else "black"
                axis.text(x, row_index, annotation, ha="center", va="center", fontsize=9, color=color)
        axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(widths), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Query/key dimension per head (dK)")
    axes[0].set_title("Taylor degree")
    axes[1].set_title("Reference")
    figure.colorbar(image, cax=figure.add_subplot(layout[0, 2]), label="Terminal held-out perplexity (linear scale)")
    figure.suptitle("Joint dK × Taylor degree · 19,074 updates · seed 0")
    save(figure, "terminal_ppl_heatmap")

    deltas = np.array([[row["delta_ppl_vs_softmax"] if row["delta_ppl_vs_softmax"] != "" else np.nan for row in line[:n_degrees]] for line in grid])
    available = np.abs(deltas[np.isfinite(deltas)])
    extent = max(float(available.max()) if len(available) else 1.0, 1e-8)
    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
    delta_cmap = plt.colormaps["RdBu_r"].copy()
    delta_cmap.set_bad("#eeeeee")
    image = axis.imshow(deltas, cmap=delta_cmap, norm=TwoSlopeNorm(0.0, -extent, extent), aspect="auto")
    for i, line in enumerate(grid):
        for j, row in enumerate(line[:n_degrees]):
            value = deltas[i, j]
            annotation = f"{value:+.4f}" if np.isfinite(value) else row["status"].upper() if row["status"] != "complete" else "NO REF"
            axis.text(j, i, annotation, ha="center", va="center", fontsize=9,
                      color="white" if np.isfinite(value) and abs(value) > 0.65 * extent else "black")
    axis.set_xticks(range(n_degrees), [f"m={degree}" for degree in degrees])
    axis.set_yticks(range(len(widths)), [str(width) for width in widths])
    axis.set_ylabel("Query/key dimension per head (dK)")
    axis.set_title("Terminal perplexity − softmax at the same dK\nNegative is better")
    figure.colorbar(image, ax=axis, label="Δ perplexity (linear scale)")
    save(figure, "delta_ppl_heatmap")

    curve_rows = (len(widths) + 3) // 3
    figure, axes = plt.subplots(curve_rows, 3, figsize=(14, 4 * curve_rows), layout="constrained", squeeze=False)
    for axis, width in zip(axes.flat, widths):
        for row, records in summaries:
            if row["qk_head_dim"] != width or not records:
                continue
            degree = row["attention_degree"]
            axis.plot([record["training_tokens"] / 1e9 for record in records], [record["validation_perplexity"] for record in records],
                      label=row["kernel"] + (" *" if row["reused"] else ""),
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
    axes.flat[len(widths)].text(0, 0.8, "* Reused historical run\nDashed curves: incomplete or failed\nEndpoints require 19,074 updates", va="top")
    save(figure, "learning_curves")

    lines = ["# Joint key dimension × Taylor degree", "", f"Source: `{manifest['source_commit']}`. Sweep identity: `{manifest['identity']}`.", "",
             f"Status: {sum(row['status'] == 'complete' for row in rows)} complete, {sum(row['status'] == 'pending' for row in rows)} pending, {sum(row['status'] == 'incomplete' for row in rows)} incomplete, and {sum(row['status'] == 'failed' for row in rows)} numerical failures. Of the incomplete cells, {sum(row['execution_status'] == 'interrupted' for row in rows)} have a recorded infrastructure or validation interruption.", "",
             "The primary endpoint is held-out NLL/perplexity at 19,074 updates (10,000,269,312 prediction tokens), independently of the retained validation-best checkpoint. All runs use seed 0; this comparison does not estimate seed variation.", "",
             f"The grid uses ascending dK rows ({', '.join(map(str, widths))}), Taylor degrees {', '.join(map(str, degrees))}, and a separate softmax column with the same linear color scale. Heatmaps omit reuse markers; asterisks in the table and learning curves mark reused results. Failed, pending, and incomplete cells have no imputed perplexity.", "",
             "FAILED means a recorded numerical failure. Infrastructure interruptions and failed restore/audit validation remain INCOMPLETE, with `execution_status=interrupted`, `failure_kind`, and error details in the CSV. They do not establish a scientific numerical failure. A validated completion supersedes earlier resolved interruptions.", "",
             "Reused dK=64 Taylor results come from `polynomial_attention`. Softmax references come from `key_dimension`; its dK=64 softmax endpoint is **14.689767074685026**. The separate `polynomial_attention` softmax result, **14.69417664861716**, is historical context and is not used in this grid or its deltas. Each delta subtracts the softmax endpoint at the same width.", "",
             f"Original source commits, resolved configurations, and artifact checksums remain in `{'provenance/manifest.json' if retained else 'sweep.json'}`. Historical hardware histories remain attached to their source provenance; throughput across these runs is not a matched hardware benchmark.", "",
             "| dK | Kernel | Origin | Status | Terminal NLL | Terminal PPL | Δ PPL vs softmax |", "|---:|---|---|---|---:|---:|---:|"]
    for row in rows:
        nll, ppl = (f"{row['terminal_nll']:.6f}", f"{row['terminal_ppl']:.4f}") if row["status"] == "complete" else ("—", "—")
        delta = f"{row['delta_ppl_vs_softmax']:+.4f}" if row["delta_ppl_vs_softmax"] != "" else "—"
        lines.append(f"| {row['qk_head_dim']} | {row['kernel']}{' *' if row['reused'] else ''} | {row['original_experiment']} | {row['status']} | {nll} | {ppl} | {delta} |")
    failures = [row for row in rows if row["error"]]
    if failures:
        lines += ["", "Recorded failures or interruptions:", "", *[f"- `{row['run']}` ({row['failure_kind'] or row['execution_status']}): {row['error']}" for row in failures]]
    unavailable = [(run, artifact) for run in [*manifest["runs"], *manifest["controls"]]
                   for artifact in run.get("unavailable_historical_artifacts", [])]
    if unavailable:
        lines += ["", "Unavailable historical artifacts:", "",
                  "The following historical artifacts are absent from this cluster. Their original paths and recorded hashes are preserved in the provenance; their contents have not been locally revalidated. The available scientific metrics and completion records were checksum-verified.", ""]
        lines += [f"- `{run['name']}`: {artifact.get('kind', 'artifact')} `{artifact['path']}`; recorded SHA256 `{artifact.get('sha256', artifact.get('metadata_sha256', 'not recorded'))}`."
                  for run, artifact in unavailable]
    lines += ["", "[Summary CSV](summary.csv) · [Terminal heatmap](figures/terminal_ppl_heatmap.pdf) · [Delta heatmap](figures/delta_ppl_heatmap.pdf) · [Learning curves](figures/learning_curves.pdf)", ""]
    update_results_readme(output_dir / "README.md", lines)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest_path(Path("results/key_dimension_taylor")))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
