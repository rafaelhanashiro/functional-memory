"""The committed summary, figures, and results block regenerate byte-identically from runs/."""
import json
import shutil
from pathlib import Path

from experiments.learned_quadratic_attention.adapter.analysis import plot
from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[2]


def test_regeneration_reproduces_committed_outputs(tmp_path):
    manifest_path = RECORD / "provenance/manifest.json"
    shutil.copy(RECORD / "README.md", tmp_path / "README.md")
    plot.aggregate(json.loads(manifest_path.read_text()), tmp_path, manifest_path=manifest_path)
    assert regeneration_differences(RECORD, tmp_path) == []
