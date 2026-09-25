"""The retained comparison is regenerated from its pinned source runs."""

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from experiments.attention_scaling_comparison.adapter.analysis.plot import build


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "experiments" / "attention_scaling_comparison"


def hashes(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob("*") if path.is_file()}


def test_combined_figure_is_source_checked_and_byte_reproducible(tmp_path):
    output = tmp_path / "comparison"
    output.mkdir()
    shutil.copyfile(EXPERIMENT / "README.md", output / "README.md")
    config = EXPERIMENT / "configs" / "heatmap.json"

    rows = build(ROOT, config_path=config, output_dir=output)
    assert len(rows) == 125
    assert Counter(row["status"] for row in rows) == {
        "complete": 41, "numerical_failure": 4, "not_run": 80}
    by_cell = {(r["panel"], r["column"], r["d_k"]): r for r in rows}
    assert by_cell["normalized_fixed", "m1", 64]["failure_step"] == 9
    assert by_cell["normalized_fixed", "m3", 64]["failure_step"] == 17
    assert by_cell["normalized_learned", "unrestricted_global", 64]["failure_step"] == 2324
    assert by_cell["normalized_learned", "unrestricted_per_head", 64]["failure_step"] == 4187
    assert by_cell["normalized_learned", "positive_global", 64]["terminal_ppl"] == pytest.approx(16.189034521301647)
    assert by_cell["references", "gla", 128]["terminal_ppl"] == pytest.approx(15.436583304097534)
    assert by_cell["unnormalized_fixed", "m8", 64]["status"] == "not_run"
    assert by_cell["unnormalized_fixed", "m8", 64]["terminal_ppl"] is None

    csv_rows = list(csv.DictReader((output / "summary.csv").open()))
    assert len(csv_rows) == 125
    failed = [row for row in csv_rows if row["status"] == "numerical_failure"]
    assert all(row["terminal_ppl"] == "" for row in failed)
    provenance = json.loads((output / "provenance" / "manifest.json").read_text())
    assert len(provenance["source_manifests"]) == 7
    assert len(provenance["source_runs"]) == 45
    assert (output / "figures" / "combined_terminal_ppl.pdf").stat().st_size > 1000
    first = hashes(output)
    build(ROOT, config_path=config, output_dir=output)
    assert hashes(output) == first
