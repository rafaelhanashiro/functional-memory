"""The committed summary and figures regenerate byte-identically from runs/."""
import json
from pathlib import Path

from experiments.gla_key_dimension.adapter.analysis import plot
from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[2]


def test_regeneration_reproduces_committed_outputs(tmp_path):
    manifest = json.loads((RECORD / "provenance/manifest.json").read_text())
    rows = plot.gla_rows(manifest, RECORD)
    plot.write_summary(rows, tmp_path / "summary.csv")
    plot.plot(rows, manifest, tmp_path / "figures")
    assert regeneration_differences(RECORD, tmp_path) == []
