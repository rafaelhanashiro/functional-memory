"""Fingerprints that tie a result's figures to the runs they plot.

A result's selection files pin each plotted run to the SHA256 of the files that
determine its numbers: its recipe, measurements, completion, status metadata,
validation-best record, and any numerical-failure record. Bookkeeping such as
``record.json`` is not pinned, so provenance edits never invalidate a figure, while
any change to a scientific input does until the selection is refreshed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from . import _sha256

SCIENTIFIC_FILES = ("config.yaml", "metrics.jsonl", "completion.json", "metadata.json", "best_validation.json")
FAILURE_PATTERN = "numerical_failure_step_*.json"
# Selection files are written with one of these; refreshing keeps the file's own style.
_DUMP_OPTIONS = ({"sort_keys": False}, {"sort_keys": False, "allow_unicode": True})


def run_sources(root: str | Path, study: str, run_id: str) -> dict[str, dict[str, str]]:
    """Fingerprints of one run's scientific files, keyed by file name."""
    run = Path(root) / "experiments" / study / "runs" / run_id
    if not (run / "config.yaml").is_file():
        raise ValueError(f"Unknown run: {study}/{run_id}")
    names = [name for name in SCIENTIFIC_FILES if (run / name).is_file()]
    names += sorted(path.name for path in run.glob(FAILURE_PATTERN))
    return {name: {"path": f"experiments/{study}/runs/{run_id}/{name}", "sha256": _sha256(run / name)}
            for name in names}


def verify_pinned_runs(root: str | Path, runs: list[dict[str, Any]]) -> None:
    """Check that each pinned run's scientific files are exactly the pinned ones, unchanged."""
    for entry in runs:
        current = run_sources(root, entry["study"], entry["run_id"])
        pinned = entry["sources"]
        if set(pinned) != set(current) or any(pinned[name] != current[name] for name in current):
            changed = sorted(name for name in set(pinned) | set(current) if pinned.get(name) != current.get(name))
            raise ValueError(f"Scientific inputs of {entry['study']}/{entry['run_id']} differ from the selection "
                             f"({', '.join(changed)}); run `python -m scaffolds.records refresh` on the result")


def _fingerprints(entry: dict[str, Any]) -> dict[str, str | None]:
    if "config_sha256" in entry:
        return {"config.yaml": entry["config_sha256"],
                **{Path(value["path"]).name: value["sha256"] for value in entry.get("inputs", {}).values()}}
    return {name: value["sha256"] for name, value in entry.get("sources", {}).items()}


def _pinned_runs(entry: dict[str, Any]) -> set[str]:
    """Run IDs that an entry's existing pins point into."""
    pins = entry.get("inputs") if "config_sha256" in entry else entry.get("sources")
    return {Path(value["path"]).parent.name for value in (pins or {}).values()}


def _refreshed(root: Path, study: str, entry: dict[str, Any]) -> dict[str, Any]:
    run = root / "experiments" / study / "runs" / entry["run_id"]
    if "config_sha256" not in entry:
        return dict(entry, sources=run_sources(root, study, entry["run_id"]))
    # Analyses of retained arrays pin the inputs their record captured; an archive kept
    # only on the home cluster keeps the hash its record holds.
    inputs = json.loads((run / "record.json").read_text())["artifacts"]
    inputs = {name: dict(value, sha256=_sha256(root / value["path"])) if (root / value["path"]).is_file() else value
              for name, value in inputs.items()}
    return dict(entry, config_sha256=_sha256(run / "config.yaml"), inputs=inputs)


def _array_roles(selection: dict[str, Any]) -> dict[str, str]:
    """Run IDs and display labels selected by collection and comparison roles."""
    roles = {run_id: label for label, run_id in selection.get("collections", {}).items()}
    for label, comparison in selection.get("comparisons", {}).items():
        roles.setdefault(comparison["baseline_run_id"], f"{label}-baseline")
        for condition, run_id in comparison["condition_run_ids"].items():
            roles.setdefault(run_id, f"{label}-{condition}")
    return roles


def _repository_root(directory: Path) -> Path:
    """Resolve root or study-local results, or a study's staged results, without requiring Git metadata."""
    for parent in (directory, *directory.parents):
        if parent.name == "results":
            # A self-contained setup keeps its results beside runs, under
            # experiments/<study>/results[/<analysis>]. Scientific pins remain
            # repository-relative, including when this directory itself is used.
            if parent.parent.parent.name == "experiments":
                return parent.parent.parent.parent
            return parent.parent
        if parent.name == "staging" and parent.parent.parent.name == "experiments":
            # Finished results wait under experiments/<study>/staging/<topic> until placed in results/.
            return parent.parent.parent.parent
    raise ValueError("A result collection must live under results/, experiments/<study>/results/ or "
                     f"experiments/<study>/staging/: {directory}")


def refresh_selection(directory: str | Path) -> list[str]:
    """Re-pin a result's selection files to their runs' current scientific files.

    A direct ``runs`` list keeps each role; pointing its label at a new ``run_id``
    and refreshing pins the new run. A grid's ``cells`` and an array analysis's
    ``collections``/``comparisons`` pin exactly the runs those roles name.
    Returns one line per changed fingerprint, empty when current.
    """
    directory = Path(directory).resolve()
    root = _repository_root(directory)
    changes = []
    for path in sorted(directory.glob("*.yaml")):
        original = path.read_text()
        selection = yaml.safe_load(original)
        if not isinstance(selection, dict) or not ("cells" in selection or isinstance(selection.get("runs"), list)):
            continue
        style = next((option for option in _DUMP_OPTIONS if yaml.safe_dump(selection, **option) == original),
                     _DUMP_OPTIONS[-1])
        default_study = Path(selection["study"]).name if "study" in selection else None
        previous_entries = selection.get("runs", [])
        previous = {(entry.get("study", default_study), entry["run_id"]): entry for entry in previous_entries}
        if "cells" in selection:
            wanted = list({(cell["experiment"], cell["run"]): {
                "study": cell["experiment"], "run_id": cell["run"], "label": cell["label"]}
                for cell in selection["cells"]}.values())
        elif "collections" in selection or "comparisons" in selection:
            roles = _array_roles(selection)
            # Keep established labels and ordering for retained runs. New roles
            # get the same array-input pin schema, even before their first refresh.
            wanted = [entry for key, entry in previous.items() if key[0] == default_study
                      and entry["run_id"] in roles]
            wanted += [{"run_id": run_id, "label": label, "config_sha256": None, "inputs": {}}
                       for run_id, label in roles.items() if (default_study, run_id) not in previous]
        else:
            wanted = previous_entries
        refreshed = []
        wanted_keys = set()
        for entry in wanted:
            key = (entry.get("study", default_study), entry["run_id"])
            wanted_keys.add(key)
            replaced = _pinned_runs(entry) - {key[1]}
            entry = _refreshed(root, key[0], entry)
            before, after = _fingerprints(previous.get(key, {})), _fingerprints(entry)
            prefix = f"{path.name}: {'/'.join(key)}"
            if replaced:
                changes.append(f"{prefix} pinned for {entry['label']}, replacing {', '.join(sorted(replaced))}")
            elif not before:
                changes.append(f"{prefix} pinned")
            else:
                for name in sorted(set(before) | set(after)):
                    if before.get(name) != after.get(name):
                        change = "pinned" if name not in before else "removed" if name not in after else "changed"
                        changes.append(f"{prefix} {name} {change}")
            refreshed.append(entry)
        changes += [f"{path.name}: {'/'.join(key)} unpinned" for key in previous if key not in wanted_keys]
        selection["runs"] = refreshed
        text = yaml.safe_dump(selection, **style)
        if text != original:
            path.write_text(text)
    return changes
