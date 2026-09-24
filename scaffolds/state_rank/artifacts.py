"""Locate retained state-rank recipes and update their single scientific README."""
from __future__ import annotations

import os
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "results/state_rank/configs"
HISTORICAL_COLLECTIONS = {
    "results/llama-3.2-1b-l8-kvall-t512",
    "results/gated-deltanet-1.3b-l12-hall-t512",
}


def retained_path(project_root: Path, configured_path: str) -> Path:
    """Resolve relocated baselines without modifying exact historical recipes.

    The two pretrained collections are retained as runs of the state-rank record.
    """
    if configured_path in HISTORICAL_COLLECTIONS:
        return project_root / "results/state_rank/runs" / Path(configured_path).name
    return project_root / configured_path


def family_directory(output: Path) -> Path:
    """Use the canonical family document; independent exports own their README."""
    output = output.resolve()
    for candidate in (output, *output.parents):
        if candidate.name == "state_rank" and candidate.parent.name == "results":
            return candidate
    return output


def resolved_config_path(output: Path, *, writing: bool = False) -> Path:
    family = family_directory(output)
    if family == output.resolve():
        return output / "config.json"
    path = family / "configs" / f"{output.name}.resolved.json"
    if writing:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_report_section(output: Path, key: str, text: str) -> Path:
    """Replace one generated section while preserving methods and other results."""
    output = output.resolve()
    family = family_directory(output)
    path = family / "README.md"
    start, end = f"<!-- state-rank:{key}:start -->", f"<!-- state-rank:{key}:end -->"

    def relocate_link(match):
        destination = match.group(1)
        if "://" in destination or destination.startswith(("#", "/")):
            return match.group(0)
        relocated = os.path.relpath(output / destination, family)
        return f"]({relocated})"

    text = re.sub(r"\]\(([^)]+)\)", relocate_link, text)
    # The experiment owns the top-level heading; generated sections sit below it.
    text = re.sub(r"(?m)^(#{1,5}) ", r"#\1 ", text)
    section = f"{start}\n{text.rstrip()}\n{end}"
    document = path.read_text() if path.exists() else "# State rank\n"
    if start in document:
        if end not in document:
            raise ValueError(f"Incomplete generated section in {path}: {key}")
        document = re.sub(re.escape(start) + r".*?" + re.escape(end),
                          lambda _: section, document, count=1, flags=re.DOTALL)
    else:
        document = document.rstrip() + "\n\n" + section + "\n"
    path.write_text(document)
    return path
