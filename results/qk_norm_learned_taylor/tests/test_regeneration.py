"""The committed summary, figures, and README results block regenerate byte-identically from runs/.

Acceptance runs these tests with no skips allowed, so before the sweep concludes the
test checks that the report fabricates nothing instead of skipping.
"""
from pathlib import Path
import subprocess
import sys

from scaffolds.transformer.analysis import regeneration_differences
from results.qk_norm_learned_taylor import report

RECORD = Path(__file__).resolve().parents[1]
STUDY = RECORD.parents[1] / "experiments" / "attention_scaling"
ROOT = RECORD.parents[1]


def regenerate(output: Path) -> str:
    result = subprocess.run([sys.executable, "-m", "results.qk_norm_learned_taylor.report",
                             "--output", str(output)], cwd=ROOT, check=True, text=True, capture_output=True,
                            env={"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
    return result.stdout


def test_regeneration_reproduces_committed_outputs_or_waits_for_the_sweep(tmp_path):
    stdout = regenerate(tmp_path)
    if report.completed(STUDY):
        assert regeneration_differences(RECORD, tmp_path) == []
    else:
        # No summary, figure, or README block may be fabricated from pending runs.
        assert stdout.startswith("pending:")
        assert not list(tmp_path.rglob("*"))
        assert not (RECORD / "summary.csv").exists() and not (RECORD / "learning_curves.pdf").exists()
