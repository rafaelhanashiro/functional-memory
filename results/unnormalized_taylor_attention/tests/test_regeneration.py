"""The committed terminal summary regenerates byte-identically from runs/ and the manifest."""
from pathlib import Path

from results.unnormalized_taylor_attention import summary
from scaffolds.transformer.analysis import regeneration_differences, write_csv

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / "attention_scaling"


def test_regeneration_reproduces_committed_summary(tmp_path):
    write_csv(tmp_path / "summary.csv", summary.rows(STUDY), summary.FIELDS)
    assert regeneration_differences(RECORD, tmp_path) == []
