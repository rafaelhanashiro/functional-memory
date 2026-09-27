"""The committed summary and figures regenerate byte-identically from runs/."""
import json
from pathlib import Path

from scaffolds.records import load_study

from results.gla_key_dimension import plot
from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / RECORD.name


def test_regeneration_reproduces_committed_outputs(tmp_path):
    manifest = load_study(STUDY)
    rows = plot.gla_rows(manifest, STUDY)
    plot.write_summary(rows, tmp_path / "summary.csv")
    plot.plot(rows, manifest, tmp_path)
    assert regeneration_differences(RECORD, tmp_path) == []
