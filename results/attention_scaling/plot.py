"""Regenerate the cross-mechanism d_K × Taylor-degree comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import yaml
from collections import Counter
from pathlib import Path

from .chart import render_degree_chart
from .collect import collect_rows
from .render import render_heatmap


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "attention_scaling"
FIELDS = (
    "figure", "panel", "column", "d_k", "status", "terminal_nll", "terminal_ppl", "failure_step",
    "source_experiment", "source_run", "source_commit", "metrics_path", "metrics_sha256",
    "completion_path", "completion_sha256", "failure_path", "failure_sha256",
    "weight_decay_policy", "qk_norm", "legacy", "legacy_reasons",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(root: Path, path: Path) -> dict[str, str]:
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": digest(path)}


def csv_row(row: dict, figure: str) -> dict:
    artifacts = row.get("artifacts") or {}
    def artifact_field(kind: str, field: str) -> str:
        return (artifacts.get(kind) or {}).get(field, "")

    return {
        "figure": figure, "panel": row["panel"], "column": row["column"], "d_k": row["d_k"],
        "status": row["status"], "terminal_nll": row.get("terminal_nll"),
        "terminal_ppl": row.get("terminal_ppl"), "failure_step": row.get("failure_step"),
        "source_experiment": row.get("experiment", ""), "source_run": row.get("run", ""),
        "source_commit": row.get("source_commit", ""),
        "metrics_path": artifact_field("metrics", "path"),
        "metrics_sha256": artifact_field("metrics", "sha256"),
        "completion_path": artifact_field("completion", "path"),
        "completion_sha256": artifact_field("completion", "sha256"),
        "failure_path": artifact_field("failure", "path"),
        "failure_sha256": artifact_field("failure", "sha256"),
        "weight_decay_policy": row.get("weight_decay_policy") or "",
        "qk_norm": "" if row.get("qk_norm") is None else str(bool(row["qk_norm"])).lower(),
        "legacy": str(bool(row.get("legacy"))).lower(),
        "legacy_reasons": "; ".join(row.get("legacy_reasons") or []),
    }


def _counts(rows: list[dict]) -> tuple[Counter, int, int]:
    trained = [row for row in rows if row["status"] != "not_run"]
    return Counter(row["status"] for row in rows), len(trained), sum(1 for row in trained if row.get("legacy"))


def update_readme(path: Path, rows: list[dict], chart_rows: list[dict]) -> None:
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    authored = path.read_text()
    if authored.count(begin) != 1 or authored.count(end) != 1:
        raise ValueError("README needs exactly one generated-results marker pair")
    prefix, rest = authored.split(begin, 1)
    _, suffix = rest.split(end, 1)
    counts, trained, legacy = _counts(rows)
    chart_counts, chart_trained, chart_legacy = _counts(chart_rows)
    block = (
        f"{len(rows)} displayed coordinates: {counts['complete']} complete, "
        f"{counts['numerical_failure']} numerical failures, "
        f"{counts['not_run']} not run, and {counts['incomplete']} incomplete. "
        "Only completed runs at the common 19,074-update endpoint receive a color "
        "and a terminal PPL. A failed run is labeled with its failure update; "
        "an untrained coordinate is marked NOT RUN. "
        f"{legacy} of the {trained} trained cells are outlined as legacy runs: they trained "
        "with weight decay on all parameters and, where it applies, without QK-norm, before those "
        "became the scaffold's defaults; each is replaced when its experiment is rerun. "
        f"{trained - legacy} cells already use the current defaults.\n\n"
        "![Combined terminal perplexity heatmap](combined_terminal_ppl.png)\n\n"
        "[PDF](combined_terminal_ppl.pdf)\n\n"
        "### Polynomial kernels at d_K = 64\n\n"
        f"{chart_trained} trained runs: {chart_counts['complete']} complete and "
        f"{chart_counts['numerical_failure']} numerical failures; {chart_legacy} are legacy runs, drawn "
        "hollow with dashed connectors. Numerical failures appear in the strip below the axis at "
        "their degree, labeled with the failing update. The nonnegative family exists only for even "
        "degrees, which are the only degrees at which a polynomial can be nonnegative on the real line.\n\n"
        "![Terminal perplexity against polynomial degree](degree_chart.png)\n\n"
        "[PDF](degree_chart.pdf) · [Cell-level data and source checksums](summary.csv) "
        "· [Provenance](manifest.json)\n"
    )
    path.write_text(prefix + begin + "\n" + block + end + suffix)


def build(repo_root: Path = REPO_ROOT, *, config_path: Path | None = None,
          chart_config_path: Path | None = None, output_dir: Path | None = None) -> list[dict]:
    """Regenerate both figures, the summary, the README block, and provenance; returns the heatmap rows."""
    repo_root = repo_root.resolve()
    output_dir = output_dir or repo_root / "results" / EXPERIMENT
    config_path = config_path or repo_root / "results" / EXPERIMENT / "selection.yaml"
    chart_config_path = chart_config_path or config_path.parent / "degree_chart.yaml"
    config = yaml.safe_load(config_path.read_text())
    chart_config = yaml.safe_load(chart_config_path.read_text())
    rows = collect_rows(repo_root, config)
    chart_rows = collect_rows(repo_root, chart_config)
    figure_paths = {**render_heatmap(rows, config, output_dir),
                    **render_degree_chart(chart_rows, chart_config, output_dir)}

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_row(row, "heatmap") for row in rows)
        writer.writerows(csv_row(row, "degree_chart") for row in chart_rows)

    readme_path = output_dir / "README.md"
    update_readme(readme_path, rows, chart_rows)

    source_rows = [
        {"figure": figure, "panel": row["panel"], "column": row["column"], "d_k": row["d_k"],
         "experiment": row["experiment"], "run": row["run"], "status": row["status"],
         "source_commit": row.get("source_commit"), "artifacts": row.get("artifacts", {}),
         "weight_decay_policy": row.get("weight_decay_policy"), "qk_norm": row.get("qk_norm"),
         "legacy": bool(row.get("legacy")), "legacy_reasons": row.get("legacy_reasons", [])}
        for figure, figure_rows in (("heatmap", rows), ("degree_chart", chart_rows))
        for row in figure_rows if row["status"] != "not_run"
    ]
    status_counts = {}
    for figure, figure_rows in (("heatmap", rows), ("degree_chart", chart_rows)):
        counts = dict(sorted(Counter(row["status"] for row in figure_rows).items()))
        counts["legacy"] = sum(1 for row in figure_rows if row.get("legacy"))
        status_counts[figure] = counts
    code = Path(__file__).resolve().parent
    lock = repo_root / "scaffolds" / "transformer" / "uv.lock"
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "kind": "derived_comparison_no_new_training",
        "home_cluster": "kennel",
        "artifact_root": "/data/cl/u/rafah/home/projects/functional-memory",
        "seed": config["seed"],
        "data_identity": config["data_identity"],
        "terminal_step": config["terminal_step"],
        "training_tokens": config["training_tokens"],
        "display_coordinates": {"heatmap": {"d_k": config["d_k"], "panels": config["panels"]},
                                "degree_chart": {"d_k": chart_config["d_k"], "panels": chart_config["panels"]}},
        "source_manifests": {**config["source_manifests"], **chart_config["source_manifests"]},
        "source_records": {**config.get("source_records", {}), **chart_config.get("source_records", {})},
        "source_runs": source_rows,
        "status_counts": status_counts,
        "analysis": {
            "config": reference(repo_root, config_path),
            "chart_config": reference(repo_root, chart_config_path),
            "dependency_lock": reference(repo_root, lock),
            "code": {name: reference(repo_root, code / name)
                     for name in ("collect.py", "render.py", "chart.py", "plot.py")},
        },
        "derived_artifacts": {
            "summary_csv": reference(repo_root, csv_path),
            "readme": reference(repo_root, readme_path),
            **{name: reference(repo_root, path) for name, path in sorted(figure_paths.items())},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(args.repo_root, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
