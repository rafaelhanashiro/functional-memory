"""Regenerate state-rank results from selected canonical run records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import yaml
from scaffolds.records import load_study, run_directory, validate_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECORD = REPOSITORY_ROOT / "experiments/state_rank"
RESULTS = REPOSITORY_ROOT / "results/state_rank"
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
RECORD_COMMAND = "PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m results.state_rank.record regenerate"
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
            return figure_name(path)
        if path.name in ("summary.csv", "depth_summary.csv"):
            return "summary.csv"
        if path.name in EXECUTION_JSON:
            return "../../experiments/state_rank/sweep.yaml"
        return None  # Other derived tables and statistics are regenerable, not retained.
    if top == "configs" and path.name.endswith(".resolved.json"):
        return "../../experiments/state_rank/sweep.yaml"
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
    figures = record
    figures.mkdir(parents=True, exist_ok=True)
    for old in figures.iterdir():
        if old.suffix in FIGURE_SUFFIXES and old.name.startswith(tuple(f"{name}-" for name in (*COMPARISONS, UNIFIED))):
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


def reconstitute(record: Path, root: Path, selection: dict | None = None) -> Path:
    """Build a read-only workspace from the record; condition directories are symlinks."""
    provenance = load_study(record)
    workspace = root / "experiments/state_rank"
    (workspace / "configs").mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("# State rank\n")
    for comparison, entry in provenance["comparisons"].items():
        selected = selection["comparisons"][comparison] if selection else entry
        directory = workspace / comparison
        directory.mkdir(parents=True)
        manifest = entry["manifest"]
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        inputs = run_directory(record, selected["baseline_run_id"]) / "inputs.json"
        if sha256(inputs) != manifest["inputs_sha256"]:
            raise ValueError(f"{comparison}: retained inputs differ from the execution manifest")
        shutil.copyfile(inputs, directory / "inputs.json")
        (workspace / "configs" / f"{comparison}.resolved.json").write_text(json.dumps(entry["resolved_config"], indent=2) + "\n")
        for condition in manifest["conditions"]:
            (directory / condition["name"]).symlink_to(run_directory(record, selected["condition_run_ids"][condition["name"]]), target_is_directory=True)
    return workspace


def regenerate_collection_figures(record: Path, scratch: Path, results: Path, collection_ids: dict) -> list[str]:
    """Redraw each pretrained collection's aggregate figure from its saved spectra."""
    import numpy as np
    from scaffolds.state_rank.spectrum import plot_aggregate

    written = []
    for name, run_id in collection_ids.items():
        run = run_directory(record, run_id)
        metadata = json.loads((run / "metadata.json").read_text())
        with np.load(run / "spectra.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        output = scratch / name
        output.mkdir(parents=True)
        label = "Llama" if metadata["model"].startswith("meta-llama/") else "Gated DeltaNet"
        plot_aggregate(arrays, output, layer=metadata["layer"], model_name=label)
        for suffix in FIGURE_SUFFIXES:
            destination = results / f"{name}-effective_rank{suffix}"
            shutil.copyfile(output / f"effective_rank{suffix}", destination)
            written.append(destination.name)
    return written


def regenerate(record: Path = RECORD, results: Path = RESULTS) -> dict:
    """Recompute figures, summary, and README sections from the record's runs."""
    from scaffolds.state_rank.plot_comparison import plot_comparison
    from scaffolds.state_rank.unify_analysis import generate

    selection = yaml.safe_load((results / "selection.yaml").read_text())
    pinned = {entry["run_id"] for entry in selection["runs"]}
    selected = set(selection["collections"].values())
    for comparison in selection["comparisons"].values():
        selected.add(comparison["baseline_run_id"])
        selected.update(comparison["condition_run_ids"].values())
    if pinned != selected:
        raise ValueError("Selected comparison runs must exactly match the pinned input set")
    for entry in selection["runs"]:
        run = run_directory(record, entry["run_id"])
        validate_record(run)
        if sha256(run / "config.yaml") != entry["config_sha256"]:
            raise ValueError(f"Selection config hash changed: {run.name}")
        for artifact in entry["inputs"].values():
            path = run / Path(artifact["path"]).name
            if sha256(path) != artifact["sha256"]:
                raise ValueError(f"Selection input hash changed: {path}")
    with tempfile.TemporaryDirectory(prefix="state-rank-regenerate-") as temporary:
        root = Path(temporary)
        workspace = reconstitute(record, root, selection)
        for comparison in COMPARISONS:
            plot_comparison(workspace / comparison)
        config_path = root / "analysis.json"
        config_path.write_text(json.dumps(selection["analysis"]))
        generate(config_path, root, workspace / UNIFIED)
        result = retain_derived(workspace, results)
        result["collection_figures"] = regenerate_collection_figures(record, root / "collections", results, selection["collections"])
        return result



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["regenerate"])
    parser.parse_args()
    print(json.dumps(regenerate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
