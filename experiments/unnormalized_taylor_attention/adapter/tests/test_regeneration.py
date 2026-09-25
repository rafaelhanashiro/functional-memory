"""The committed terminal summary regenerates byte-identically from runs/ and the manifest."""
from pathlib import Path

from experiments.unnormalized_taylor_attention.adapter.analysis import summary
from scaffolds.transformer.analysis import regeneration_differences, write_csv

RECORD = Path(__file__).resolve().parents[2]


def test_regeneration_reproduces_committed_summary(tmp_path):
    write_csv(tmp_path / "summary.csv", summary.rows(RECORD), summary.FIELDS)
    assert regeneration_differences(RECORD, tmp_path) == []
