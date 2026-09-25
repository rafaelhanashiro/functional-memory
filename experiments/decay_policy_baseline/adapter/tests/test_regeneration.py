"""Before completion the report writes nothing; afterwards the committed outputs regenerate byte-identically."""
from pathlib import Path
import subprocess
import sys

from scaffolds.transformer.analysis import regeneration_differences
from experiments.decay_policy_baseline.adapter.analysis import report

RECORD = Path(__file__).resolve().parents[2]
ROOT = RECORD.parents[1]


def regenerate(output: Path) -> str:
    result = subprocess.run([sys.executable, "-m", "experiments.decay_policy_baseline.adapter.analysis.report",
                             "--output", str(output)], cwd=ROOT, check=True, text=True, capture_output=True,
                            env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
    return result.stdout


def test_regeneration_reproduces_committed_outputs_or_waits_for_the_run(tmp_path):
    stdout = regenerate(tmp_path)
    if report.completed(RECORD):
        assert regeneration_differences(RECORD, tmp_path) == []
    else:
        # No summary, figure, or README block may be fabricated from a pending run.
        assert stdout.startswith("pending:")
        assert not list(tmp_path.rglob("*"))
