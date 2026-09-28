"""The committed summary, figures, and README results block regenerate byte-identically from runs/."""
from pathlib import Path
import subprocess
import sys

import pytest

from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / "attention_scaling"
ROOT = RECORD.parents[1]


def test_regeneration_reproduces_committed_outputs(tmp_path):
    if not (RECORD / "summary.csv").exists():
        pytest.skip("No committed results yet: the run has not completed")
    subprocess.run([sys.executable, "-m", "results.qk_norm_attention.report",
                    "--output", str(tmp_path)], cwd=ROOT, check=True, env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
    assert regeneration_differences(RECORD, tmp_path) == []
