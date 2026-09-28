"""The committed summary and figures regenerate byte-identically from runs/."""
from pathlib import Path

from results.gla_key_dimension import plot
from scaffolds.transformer.analysis import selected_views, regeneration_differences

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / "attention_scaling"


def test_regeneration_reproduces_committed_outputs(tmp_path):
    manifest = selected_views(RECORD, STUDY)
    rows = plot.gla_rows(manifest, STUDY)
    plot.write_summary(rows, tmp_path / "summary.csv")
    plot.plot(rows, manifest, tmp_path)
    assert regeneration_differences(RECORD, tmp_path) == []
