"""Before every run concludes the report writes nothing; afterwards the committed outputs regenerate byte-identically."""
from pathlib import Path
import subprocess
import sys

from scaffolds.transformer.analysis import regeneration_differences
from results.nonnegative_learned_taylor import report

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / "attention_scaling"
ROOT = RECORD.parents[1]


def test_regeneration_reproduces_committed_outputs_or_waits_for_the_runs(tmp_path):
    result = subprocess.run([sys.executable, "-m", "results.nonnegative_learned_taylor.report",
                             "--output", str(tmp_path)], cwd=ROOT, check=True, text=True, capture_output=True,
                            env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
    if report.completed(STUDY):
        assert regeneration_differences(RECORD, tmp_path) == []
    else:
        assert result.stdout.startswith("pending:")
        assert not list(tmp_path.rglob("*"))
