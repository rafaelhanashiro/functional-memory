"""The retained comparison is regenerated from its pinned source runs."""

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from results.attention_scaling.plot import build


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "results" / "attention_scaling"


def hashes(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob("*") if path.is_file()}


def test_combined_figures_are_source_checked_and_byte_reproducible(tmp_path):
    output = tmp_path / "comparison"
    output.mkdir()
    shutil.copyfile(EXPERIMENT / "README.md", output / "README.md")
    config = EXPERIMENT / "selection.yaml"

    rows = build(ROOT, config_path=config, output_dir=output)
    assert len(rows) == 45
    assert Counter(row["status"] for row in rows) == {"complete": 31, "numerical_failure": 2, "not_run": 12}
    assert not any(row["legacy"] for row in rows)
    by_cell = {(r["panel"], r["column"], r["d_k"]): r for r in rows}
    assert by_cell["normalized_fixed", "m1", 64]["failure_step"] == 67
    assert by_cell["normalized_fixed", "m3", 64]["failure_step"] == 218
    assert by_cell["references", "gated_deltanet", 128]["terminal_ppl"] == pytest.approx(13.326871698095397)
    assert by_cell["references", "softmax", 64]["terminal_ppl"] == pytest.approx(13.731335348660728)
    assert by_cell["references", "softmax_no_qk_norm", 64]["terminal_ppl"] == pytest.approx(14.183944824873869)
    assert by_cell["references", "softmax", 8]["terminal_ppl"] == pytest.approx(15.732218014487678)
    assert by_cell["normalized_fixed", "m2", 16]["terminal_ppl"] == pytest.approx(17.27905787246534)
    assert by_cell["normalized_fixed", "m16", 64]["terminal_ppl"] == pytest.approx(13.73957880890914)

    csv_rows = list(csv.DictReader((output / "summary.csv").open()))
    assert Counter(row["figure"] for row in csv_rows) == {"heatmap": 45, "degree_chart": 30}
    chart = {(row["panel"], row["column"]): row for row in csv_rows if row["figure"] == "degree_chart"}
    assert float(chart["unnormalized_learned", "m3"]["terminal_ppl"]) == pytest.approx(14.630474939069227)
    assert chart["normalized_learned_taylor", "m2"]["failure_step"] == "937"
    assert chart["normalized_nonnegative", "m2"]["legacy"] == "false"
    assert float(chart["normalized_nonnegative", "m4"]["terminal_ppl"]) == pytest.approx(13.815325885598568)
    assert float(chart["unnormalized_fixed_hermite", "m2"]["terminal_ppl"]) == pytest.approx(15.624292998941916)
    assert float(chart["unnormalized_fixed", "m2"]["terminal_ppl"]) == pytest.approx(18.099043285164523)
    assert float(chart["references", "unnormalized_exp"]["terminal_ppl"]) == pytest.approx(15.662688497541682)
    assert chart["normalized_fixed_hermite", "m2"]["failure_step"] == "2875"
    assert all(row["terminal_ppl"] == "" for row in csv_rows if row["status"] == "numerical_failure")
    provenance = json.loads((output / "manifest.json").read_text())
    assert "source_manifests" not in provenance and "source_records" not in provenance
    assert Counter(row["figure"] for row in provenance["source_runs"]) == {"heatmap": 33, "degree_chart": 30}
    assert provenance["status_counts"]["heatmap"]["legacy"] == 0
    assert provenance["status_counts"]["degree_chart"] == {"complete": 20, "legacy": 0, "numerical_failure": 10}
    for name in ("combined_terminal_ppl", "degree_chart"):
        assert (output / f"{name}.pdf").stat().st_size > 1000
    first = hashes(output)
    build(ROOT, config_path=config, output_dir=output)
    assert hashes(output) == first
