"""Map state-rank execution workspaces onto the canonical experiment record.

Comparisons and the unified analysis execute unchanged in a workspace laid out
like the record (``<root>/experiments/state_rank/<comparison>/...``).
``retain`` copies each condition's measurements into ``runs/``, figures into
``figures/``, one merged ``summary.csv``, the generated README sections, and a
summary of execution records into ``provenance/manifest.json``. ``regenerate``
rebuilds such a workspace from the record alone and reruns the unchanged
plotting code, so every derived output is reproducible from ``runs/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RECORD = REPOSITORY_ROOT / "experiments/state_rank"
# Comparison directory -> unified-analysis model key.
COMPARISONS = {"random_vs_pretrained": "llama", "random_vs_pretrained_deltanet": "gated-deltanet"}
UNIFIED = "unified_comparison"
FIGURE_SUFFIXES = (".png", ".pdf")
# Pretrained single-model collections retained as runs; their figure is effective_rank.
PRETRAINED_COLLECTIONS = ("llama-3.2-1b-l8-kvall-t512", "gated-deltanet-1.3b-l12-hall-t512")
# Generated README sections kept in the record; the per-comparison sections repeat
# the authored aggregation conventions and describe workspace files.
README_SECTIONS = ("unified-comparison",)
# Workspace commands assume the comparison-directory layout; the record regenerates instead.
WORKSPACE_COMMAND = ("PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m scaffolds.state_rank.unify_analysis "
                     "--config CONFIG --output /path/to/new-results")
RECORD_COMMAND = "PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.state_rank.adapter.analysis.record regenerate"
# Execution records folded into the manifest; other workspace JSON is derived and regenerable.
EXECUTION_JSON = ("manifest.json", "provenance.json", "validation.json")
SECTION = re.compile(r"<!-- state-rank:([\w-]+):start -->\n(.*?)\n<!-- state-rank:\1:end -->", re.S)
# Per-comparison columns that the unified depth summary does not carry.
EXTRA_COLUMNS = ("final_log_trace_mean", "final_key_norms_mean", "final_value_norms_mean",
                 "final_key_l2_rms", "final_value_l2_rms")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def figure_name(relative: Path) -> str:
    """Flatten a workspace figure path into one record figure name."""
    return "-".join(Path(relative).parts)


def relocate(target: str) -> str | None:
    """Map a workspace-family-relative link to the record; None drops the link."""
    if "://" in target or target.startswith(("#", "/", "../")):
        return target
    path = Path(target)
    top = path.parts[0] if path.parts else ""
    if top in (*COMPARISONS, UNIFIED):
        if path.suffix in FIGURE_SUFFIXES:
            return f"figures/{figure_name(path)}"
        if path.name in ("summary.csv", "depth_summary.csv"):
            return "summary.csv"
        if path.name in EXECUTION_JSON:
            return "provenance/manifest.json"
        return None  # Other derived tables and statistics are regenerable, not retained.
    if top == "configs" and path.name.endswith(".resolved.json"):
        return "provenance/manifest.json"
    return target


def relocate_links(text: str) -> str:
    """Rewrite generated-section links; links to unretained tables become plain text."""
    out, position = [], 0
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        label, target = match.groups()
        new = relocate(target)
        out.append(text[position:match.start()])
        out.append(f"[{label}]({new})" if new else label)
        position = match.end()
    out.append(text[position:])
    return "".join(out)


def sections(document: str) -> dict[str, str]:
    return {key: body for key, body in SECTION.findall(document)}


def replace_sections(document: str, updates: dict[str, str]) -> str:
    for key, body in updates.items():
        start, end = f"<!-- state-rank:{key}:start -->", f"<!-- state-rank:{key}:end -->"
        block = f"{start}\n{body}\n{end}"
        if start in document:
            document = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, document, count=1, flags=re.S)
        else:
            document = document.rstrip() + "\n\n" + block + "\n"
    return document


def merged_summary(workspace: Path) -> list[dict]:
    """One row per model, condition, and layer: unified ranks plus comparison-only columns."""
    with (workspace / UNIFIED / "depth_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    extras = {}
    for comparison, key in COMPARISONS.items():
        with (workspace / comparison / "summary.csv").open() as handle:
            for row in csv.DictReader(handle):
                extras[(key, row["condition"], row["layer"])] = row
    merged = []
    for row in rows:
        source = extras.get((row["model"], row["condition"], row["layer"]))
        if source is None:
            raise ValueError(f"No comparison summary row for {row['model']}/{row['condition']}/layer {row['layer']}")
        merged.append({**row, **{column: source.get(column, "") for column in EXTRA_COLUMNS}})
    return merged


def retain_derived(workspace: Path, record: Path) -> dict:
    """Replace record figures, summary, and generated README sections from a workspace."""
    figures = record / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for old in figures.iterdir():
        if old.name.startswith(tuple(f"{name}-" for name in (*COMPARISONS, UNIFIED))):
            old.unlink()
    copied = []
    for top in (*COMPARISONS, UNIFIED):
        for path in sorted((workspace / top).rglob("*")):
            if path.is_file() and path.suffix in FIGURE_SUFFIXES:
                destination = figures / figure_name(path.relative_to(workspace))
                shutil.copyfile(path, destination)
                copied.append(destination.name)
    rows = merged_summary(workspace)
    with (record / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    generated = sections((workspace / "README.md").read_text())
    updates = {key: relocate_links(generated[key]).replace(WORKSPACE_COMMAND, RECORD_COMMAND) for key in README_SECTIONS}
    readme = record / "README.md"
    readme.write_text(replace_sections(readme.read_text(), updates))
    return {"figures": copied, "summary_rows": len(rows), "readme_sections": sorted(updates)}


def retain_runs(workspace: Path, record: Path) -> list[dict]:
    """Copy each complete condition's raw outputs into runs/<comparison>-<condition>/."""
    retained = []
    for comparison in COMPARISONS:
        manifest = json.loads((workspace / comparison / "manifest.json").read_text())
        for condition in manifest["conditions"]:
            source = workspace / comparison / condition["name"]
            metadata = json.loads((source / "metadata.json").read_text())
            if metadata.get("status") != "complete" or sha256(source / "metrics.npz") != metadata["metrics_sha256"]:
                raise ValueError(f"{comparison}/{condition['name']} is not a verified complete condition")
            destination = record / "runs" / f"{comparison}-{condition['name']}"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("metadata.json", "metrics.npz"):
                shutil.copyfile(source / name, destination / name)
            retained.append({"run": destination.name, "metrics_sha256": metadata["metrics_sha256"],
                             "slurm_job_id": metadata.get("slurm_job_id"), "log_path": metadata.get("log_path")})
    return retained


def execution_records(workspace: Path) -> dict:
    """Execution state that the record keeps only inside its provenance manifest."""
    comparisons = {}
    for comparison in COMPARISONS:
        directory = workspace / comparison
        comparisons[comparison] = {
            "manifest": json.loads((directory / "manifest.json").read_text()),
            "inputs_sha256": sha256(directory / "inputs.json"),
            "resolved_config": json.loads((workspace / "configs" / f"{comparison}.resolved.json").read_text()),
        }
    unified = workspace / UNIFIED
    return {"comparisons": comparisons, "unified_analysis": {
        "resolved_config": json.loads((workspace / "configs" / f"{UNIFIED}.resolved.json").read_text()),
        **{name: json.loads((unified / f"{name}.json").read_text()) for name in ("provenance", "validation")},
    }}


def reconstitute(record: Path, root: Path) -> Path:
    """Build a read-only workspace from the record; condition directories are symlinks."""
    provenance = json.loads((record / "provenance/manifest.json").read_text())
    workspace = root / "experiments/state_rank"
    (workspace / "configs").mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("# State rank\n")
    for comparison, entry in provenance["comparisons"].items():
        directory = workspace / comparison
        directory.mkdir(parents=True)
        manifest = entry["manifest"]
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        inputs = record / "runs" / Path(manifest["baseline_directory"]).name / "inputs.json"
        if sha256(inputs) != manifest["inputs_sha256"]:
            raise ValueError(f"{comparison}: retained inputs differ from the execution manifest")
        shutil.copyfile(inputs, directory / "inputs.json")
        (workspace / "configs" / f"{comparison}.resolved.json").write_text(json.dumps(entry["resolved_config"], indent=2) + "\n")
        for condition in manifest["conditions"]:
            (directory / condition["name"]).symlink_to(record / "runs" / f"{comparison}-{condition['name']}", target_is_directory=True)
    return workspace


def regenerate_collection_figures(record: Path, scratch: Path) -> list[str]:
    """Redraw each pretrained collection's aggregate figure from its saved spectra."""
    import numpy as np
    from scaffolds.state_rank.spectrum import plot_aggregate

    written = []
    for name in PRETRAINED_COLLECTIONS:
        run = record / "runs" / name
        metadata = json.loads((run / "metadata.json").read_text())
        with np.load(run / "spectra.npz", allow_pickle=False) as archive:
            results = {key: archive[key] for key in archive.files}
        output = scratch / name
        output.mkdir(parents=True)
        label = "Llama" if metadata["model"].startswith("meta-llama/") else "Gated DeltaNet"
        plot_aggregate(results, output, layer=metadata["layer"], model_name=label)
        for suffix in FIGURE_SUFFIXES:
            destination = record / "figures" / f"{name}-effective_rank{suffix}"
            shutil.copyfile(output / f"effective_rank{suffix}", destination)
            written.append(destination.name)
    return written


def regenerate(record: Path = RECORD) -> dict:
    """Recompute figures, summary, and README sections from the record's runs."""
    from scaffolds.state_rank.plot_comparison import plot_comparison
    from scaffolds.state_rank.unify_analysis import generate

    with tempfile.TemporaryDirectory(prefix="state-rank-regenerate-") as temporary:
        root = Path(temporary)
        workspace = reconstitute(record, root)
        for comparison in COMPARISONS:
            plot_comparison(workspace / comparison)
        generate(record / "configs/unified_rank_analysis.json", root, workspace / UNIFIED)
        result = retain_derived(workspace, record)
        result["collection_figures"] = regenerate_collection_figures(record, root / "collections")
        return result


def retain(workspace: Path, record: Path = RECORD) -> dict:
    """Retain a completed execution workspace: runs, derived outputs, and provenance."""
    runs = retain_runs(workspace, record)
    records = execution_records(workspace)
    manifest_path = record / "provenance/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(records)
    manifest["retained_runs"] = runs
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"runs": len(runs), **retain_derived(workspace, record)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    retain_parser = commands.add_parser("retain", help="Retain a completed workspace family directory")
    retain_parser.add_argument("workspace", type=Path, help="<root>/experiments/state_rank of the execution workspace")
    commands.add_parser("regenerate", help="Recompute derived outputs from the record")
    args = parser.parse_args()
    result = retain(args.workspace.resolve()) if args.command == "retain" else regenerate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
