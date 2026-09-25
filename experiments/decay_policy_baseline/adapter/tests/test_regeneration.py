"""Once the run completes, the committed summary, figure, and README results block regenerate byte-identically."""
from pathlib import Path
import subprocess
import sys

import pytest

from scaffolds.transformer.analysis import regeneration_differences
from experiments.decay_policy_baseline.adapter.analysis import report

RECORD = Path(__file__).resolve().parents[2]
ROOT = RECORD.parents[1]


def test_regeneration_reproduces_committed_outputs(tmp_path):
    if not report.completed(RECORD):
        pytest.skip("the run has not completed yet")
    subprocess.run([sys.executable, "-m", "experiments.decay_policy_baseline.adapter.analysis.report",
                    "--output", str(tmp_path)], cwd=ROOT, check=True, env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
    assert regeneration_differences(RECORD, tmp_path) == []
