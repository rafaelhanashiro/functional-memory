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


def refresh_selection(directory: str | Path) -> list[str]:
    """Re-pin a result's selection files to their runs' current scientific files.

    Each selection keeps its ``runs`` list; pointing a label at a new ``run_id`` and
    refreshing pins the new run. A grid that declares ``cells`` pins exactly the runs
    its cells name. Returns one line per changed fingerprint, empty when current.
    """
    directory = Path(directory).resolve()
    root = directory.parents[1]
    changes = []
    for path in sorted(directory.glob("*.yaml")):
        original = path.read_text()
        selection = yaml.safe_load(original)
        if not isinstance(selection, dict) or not ("cells" in selection or isinstance(selection.get("runs"), list)):
            continue
        style = next((option for option in _DUMP_OPTIONS if yaml.safe_dump(selection, **option) == original),
                     _DUMP_OPTIONS[-1])
        default_study = Path(selection["study"]).name if "study" in selection else None
        previous = {(entry.get("study", default_study), entry["run_id"]): entry for entry in selection.get("runs", [])}
        if "cells" in selection:
            wanted = {(cell["experiment"], cell["run"]): {"study": cell["experiment"], "run_id": cell["run"],
                                                         "label": cell["label"]}
                      for cell in selection["cells"]}
        else:
            wanted = previous
        refreshed = []
        for key, entry in wanted.items():
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
        changes += [f"{path.name}: {'/'.join(key)} unpinned" for key in previous if key not in wanted]
        selection["runs"] = refreshed
        text = yaml.safe_dump(selection, **style)
        if text != original:
            path.write_text(text)
    return changes
