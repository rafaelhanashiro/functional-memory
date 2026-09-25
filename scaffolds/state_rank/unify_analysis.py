"""Read-only, common presentation of retained pretrained/random rank measurements."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from scaffolds.state_rank.reports import resolved_config_path, write_report_section
from scaffolds.state_rank.plot_comparison import RANK_METRICS
from scaffolds.state_rank.unified_metrics import (
    ModelAnalysis, comparison_statistics, load_analysis, write_summaries,
)
from scaffolds.state_rank.unified_plots import plot_analysis, plot_depth_comparison


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_comparison(analyses: list[ModelAnalysis]) -> dict:
    """Match articles, budgets and metric conventions without equating tokenizers."""
    if len(analyses) < 2 or len({a.key for a in analyses}) != len(analyses):
        raise ValueError("Need at least two architectures with unique keys")
    if any(Path(a.key).name != a.key or a.key in ("", ".", "..") for a in analyses):
        raise ValueError("Model keys must be single directory names")
    reference = analyses[0]
    article_fields = ("title", "article_index", "body_sha256")
    articles = [tuple(record[field] for field in article_fields) for record in reference.inputs]
    seeds = sorted(c.seed for c in reference.conditions if c.weights == "random")
    convention_fields = ("participation_ratio_convention", "threshold_rank_singular_tolerance")
    convention = {key: reference.conditions[0].metadata[key] for key in convention_fields}
    for analysis in analyses:
        other_articles = [tuple(record[field] for field in article_fields) for record in analysis.inputs]
        if other_articles != articles:
            raise ValueError(f"{analysis.key}: source articles/order differ")
        if not np.array_equal(analysis.prefixes, reference.prefixes):
            raise ValueError(f"{analysis.key}: token-position budget differs")
        if sorted(c.seed for c in analysis.conditions if c.weights == "random") != seeds:
            raise ValueError(f"{analysis.key}: random seed set differs")
        for condition in analysis.conditions:
            if {key: condition.metadata.get(key) for key in convention_fields} != convention:
                raise ValueError(f"{analysis.key}/{condition.name}: rank conventions differ")
    return {
        "same_source_articles_in_same_order": True,
        "same_token_position_budget": True,
        "seeds": seeds,
        "rank_conventions": convention,
        "identical_text_spans_across_architectures": all(
            [record["text"] for record in a.inputs] == [record["text"] for record in reference.inputs]
            for a in analyses[1:]
        ),
        "interpretation": "Matched source articles with model-native token prefixes; exact inputs within each architecture.",
    }


def source_hashes(analyses: list[ModelAnalysis]) -> dict[str, str]:
    paths = []
    for analysis in analyses:
        paths.extend(analysis.results / name for name in ("manifest.json", "inputs.json"))
        paths.append(resolved_config_path(analysis.results))
        paths.extend(analysis.results / c.name / name for c in analysis.conditions
                     for name in ("metadata.json", "metrics.npz"))
    return {str(path): sha256(path) for path in paths}


def validate_output(output: Path, analyses: list[ModelAnalysis]) -> None:
    for analysis in analyses:
        source = analysis.results.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must be separate from immutable source result directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a fresh --output directory: {output}")


def write_report(output: Path, analyses: list[ModelAnalysis], statistics: dict, comparison: dict) -> None:
    lines = [
        "# Unified state-rank comparison", "",
        "The same analysis is applied to pretrained and randomly initialized Llama and Gated DeltaNet. "
        "All curves reuse the retained measurements; no inference or training was rerun.", "",
        "## How to read the figures", "",
        "- **Per head, across tokens:** fix a layer; each colored curve preserves a head index and averages over sequences. "
        "The random panel additionally averages the five seeds at each head index. Heads remain separate.",
        "- **Per layer, across tokens:** each colored curve is one layer's rank averaged over sequences and heads, "
        "and additionally over seeds for random models, at each token position.",
        "- **Across depth:** at the final token, average over sequences and heads within each model. "
        "Random means and ranges then summarize the five independently initialized models.",
        "- Each architecture has one collection with pretrained and random panels side by side. "
        "No individual-sequence or individual-seed figures are retained in the collection.",
        "- Random per-head curves average the same numerical head index across independently initialized models. "
        "This describes an initialization ensemble, not matched learned features across seeds or with the pretrained model.",
        "- In per-head panels, pretrained shading is the sequence min/max; random shading is the min/max of the "
        "five sequence-averaged seed curves. Layer panels show means only. The retained depth overview also shows faint seed curves.",
        "- Compute rank separately for each state before averaging. No rank is computed from an averaged state or spectrum. "
        "Bands are descriptive ranges, not confidence intervals.", "",
        "All layer and head indices are zero-based. Token positions count the initial BOS token.", "",
        "| Model | Layers | Heads | State rank ceiling | Selected layer |", "|---|---:|---:|---:|---:|",
    ]
    for a in analyses:
        lines.append(f"| {a.label} | {len(a.layers)} | {len(a.head_indices)} | {a.width} | {a.layer} |")
    lines += [
        "", "[Terminal ranks versus layer](depth_counts.png) · [PDF](depth_counts.pdf)", "",
        "Llama measures its implicit softmax numerator (eight KV heads, width 64); Gated DeltaNet measures its "
        "native 256 × 256 recurrent state before output processing.", "",
        "The same four source articles are used in both architectures, with model-native 512-token prefixes. "
        "The text spans and token IDs differ across architectures; pretrained and random conditions within "
        "an architecture use exactly the same tokens.", "",
        "## Model collections", "",
        "Each figure has pretrained and random columns, with four rank measures and shared vertical scales. "
        "PNG links below also have sibling PDF files.", "",
        "| Model collection | Per head | Per layer |", "|---|---|---|",
    ]
    for a in analyses:
        lines.append(f"| {a.label} | [Layer {a.layer}]({a.key}/heads.png) "
                     f"| [All layers]({a.key}/layers.png) |")
    lines += [
        "", "## Terminal comparisons", "",
        "The table compares pretrained ranks with the mean of the five random models, after averaging "
        "over sequences and heads at token 512. It describes aggregate layers, not every individual head.", "",
        "| Model | Metric | Pretrained lower | Equal | Pretrained higher |",
        "|---|---|---:|---:|---:|",
    ]
    for a in analyses:
        for key, label in RANK_METRICS:
            stat = statistics[a.key][key]
            counts = [len(stat[f"pretrained_{word}_random_mean"]) for word in ("below", "equal", "above")]
            lines.append(f"| {a.label} | {label} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += [
        "", "Llama's pretrained layer averages are lower for all four rank measures at every layer. "
        "DeltaNet's participation ratio is lower at every layer, but its threshold and energy-rank comparisons are mixed. "
        "Thus the result depends on the rank definition; it does not support a blanket claim that pretraining reduces all ranks.", "",
        "These are cross-sectional comparisons with fresh reference initializations, not the actual initial checkpoints "
        "of the pretrained models or a measured trajectory through training.", "",
        "Participation ratio is `(sum(sigma^2))^2 / sum(sigma^4)`. Threshold rank counts singular values "
        "strictly above `1e-4 * sigma_max`. Energy ranks count modes needed for 90% and 99% of squared singular-value energy.", "",
        "## Data and provenance", "",
        "- [Per-head terminal summaries](per_head_terminal.csv): sequence mean/min/max at the selected layer; each seed remains separate.",
        "- [Per-layer terminal summaries](depth_summary.csv): sequence/head means and fractions of state width for each model and seed.",
        f"- [Detailed comparisons](comparison_statistics.json), [resolved configuration]({os.path.relpath(resolved_config_path(output), output)}), "
        "[provenance](provenance.json), and [postprocessing validation](validation.json).",
        "- Full token curves remain in the comparison runs' `metrics.npz` archives; no duplicate archives or checkpoints were created.",
        "- The comparisons' successful Slurm logs are recorded in provenance. This CPU postprocessing required no Slurm job.", "",
        "Reproduce from the repository root with a fresh output directory:", "", "```bash",
        "PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m scaffolds.state_rank.unify_analysis --config CONFIG --output /path/to/new-results",
        "```", "",
    ]
    write_report_section(output, "unified-comparison", "\n".join(lines))


def generate(config_path: Path, project_root: Path, output: Path | None = None) -> Path:
    config = json.loads(config_path.read_text())
    analyses = [load_analysis((project_root / entry["results"]).resolve(), key=entry["key"],
                             label=entry["label"], layer=entry.get("layer")) for entry in config["models"]]
    comparison = validate_comparison(analyses)
    output = (output if output is not None else project_root / config["output"]).resolve()
    validate_output(output, analyses)
    before = source_hashes(analyses)
    output.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "project_root": str(project_root.resolve()), "output": str(output),
                "plot_aggregation": {
                    "pretrained": "mean over sequences",
                    "random": "mean over sequences within each seed, then equal-weight seed mean",
                    "heads": "preserve numerical head indices; no matching of learned features",
                    "layers": "also average heads within each model; preserve token and layer axes",
                },
                "models": [{"key": a.key, "label": a.label, "results": str(a.results), "layer": a.layer}
                           for a in analyses]}
    resolved_config_path(output, writing=True).write_text(json.dumps(resolved, indent=2) + "\n")
    summary_counts = write_summaries(output, analyses)
    figures = []
    for analysis in analyses:
        figures.extend(plot_analysis(analysis, output))
    figures.extend(plot_depth_comparison(analyses, output))
    statistics = {a.key: comparison_statistics(a) for a in analyses}
    (output / "comparison_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    write_report(output, analyses, statistics, comparison)
    after = source_hashes(analyses)
    if before != after:
        raise RuntimeError("Source result files changed during postprocessing")
    validation = {"status": "passed", "complete_conditions": sum(len(a.conditions) for a in analyses),
                  "source_files_unchanged": True, "source_archive_checksums_verified": True,
                  "summary_counts": summary_counts,
                  "comparison": comparison, "figure_files": [str(path) for path in figures]}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()
    source_files = [Path(__file__), Path(__file__).with_name("unified_metrics.py"),
                    Path(__file__).with_name("unified_plots.py"), Path(__file__).with_name("plot_comparison.py"),
                    Path(__file__).with_name("artifacts.py"),
                    config_path, SOURCE_ROOT / "scaffolds/state_rank/uv.lock"]
    provenance = {
        "schema_version": 2, "kind": "derived-state-rank-comparison", "created_utc": datetime.now(timezone.utc).isoformat(),
        "plot_aggregation": resolved["plot_aggregation"],
        "generator_source_commit": source_commit,
        "generator_files_sha256": {str(path.relative_to(SOURCE_ROOT)) if path.is_relative_to(SOURCE_ROOT) else str(path): sha256(path)
                                   for path in source_files},
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
        "execution": "Local CPU postprocessing of retained measurements; no model run or new Slurm job",
        "input_files_sha256": before,
        "models": [{"key": a.key, "selected_layer": a.layer, "state_width": a.width,
                    "source_commit": a.manifest["source_commit"], "model_id": a.manifest["config"]["model_id"],
                    "conditions": [{"name": c.name, "seed": c.seed, "source_commit": c.metadata["source_commit"],
                                    "slurm_job_id": c.metadata["slurm_job_id"], "log_path": c.metadata["log_path"]}
                                   for c in a.conditions]} for a in analyses],
        "output_files_sha256": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(generate(args.config.resolve(), args.project_root.resolve(), args.output))


if __name__ == "__main__":
    main()
