"""The committed summary, figures, and README results block regenerate byte-identically from runs/."""
from pathlib import Path
import subprocess
import sys

from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[2]
ROOT = RECORD.parents[1]


def test_regeneration_reproduces_committed_outputs(tmp_path):
    subprocess.run([sys.executable, "-m", "experiments.learned_unnormalized_taylor.adapter.analysis.report",
                    "--output", str(tmp_path)], cwd=ROOT, check=True, env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
    assert regeneration_differences(RECORD, tmp_path) == []
