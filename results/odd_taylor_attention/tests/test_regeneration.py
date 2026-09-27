"""The committed summary, figures, and results block regenerate byte-identically from runs/."""
import json
import shutil
from pathlib import Path

from scaffolds.records import load_study

from results.odd_taylor_attention import plot
from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / RECORD.name


def test_regeneration_reproduces_committed_outputs(tmp_path):
    manifest_path = STUDY / "sweep.yaml"
    shutil.copy(RECORD / "README.md", tmp_path / "README.md")
    plot.aggregate(load_study(manifest_path), tmp_path, manifest_path=manifest_path)
    assert regeneration_differences(RECORD, tmp_path) == []
