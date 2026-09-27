"""Exercise launcher EXIT traps without allocating hardware or running workers."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from scaffolds.records import create_run, load_run, update_record


ROOT = Path(__file__).resolve().parents[3]
LAUNCHERS = [
    "scaffolds/transformer/launchers/kennel.sbatch",
    "experiments/state_rank/launch_comparison.sbatch",
]


@pytest.fixture(params=LAUNCHERS)
def launcher_exit_script(request):
    launcher = (ROOT / request.param).read_text()
    trap = re.search(r"^finish_record\(\) \{\n.*?^trap finish_record EXIT$", launcher, re.M | re.S)
    assert trap is not None, f"Missing EXIT trap in {request.param}"
    return 'set -euo pipefail\n' + trap.group() + '\nprintf "worker finished\\n"\n(exit "$WORKER_EXIT")\n'


@pytest.mark.parametrize("worker_exit", [0, 7])
@pytest.mark.parametrize("finalizer_exit", [0, 9])
def test_finalization_preserves_worker_failure(
    tmp_path, launcher_exit_script, worker_exit, finalizer_exit
):
    finalizer = tmp_path / "finalizer"
    finalizer.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$FINALIZER_ARGS"\nexit "$FINALIZER_EXIT"\n'
    )
    finalizer.chmod(0o755)
    args_path = tmp_path / "args"
    run_root = tmp_path / "run"
    env = {
        **os.environ, "PYTHON": str(finalizer), "RUN_ROOT": str(run_root),
        "WORKER_EXIT": str(worker_exit), "FINALIZER_EXIT": str(finalizer_exit),
        "FINALIZER_ARGS": str(args_path),
    }
    result = subprocess.run(["bash", "-c", launcher_exit_script], env=env, text=True, capture_output=True)
    assert result.returncode == (worker_exit or finalizer_exit)
    assert args_path.read_text().splitlines() == [
        "-m", "scaffolds.launch.finalize", "--run", str(run_root), "--exit-code", str(worker_exit),
    ]
    assert result.stdout == "worker finished\n"
    assert result.stderr == ""


@pytest.mark.parametrize("worker_exit", [0, 7])
def test_successful_finalization_leaves_the_recorded_log_unchanged(
    tmp_path, launcher_exit_script, worker_exit
):
    run = create_run(tmp_path / "study", {"seed": 0})
    update_record(run, {
        "status": "running" if worker_exit else "complete",
        "attempts": [{"slurm_job_id": "123"}],
    })
    log = run / "training.log"
    env = {
        **os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(ROOT),
        "RUN_ROOT": str(run), "WORKER_EXIT": str(worker_exit), "SLURM_JOB_ID": "123",
    }
    with log.open("w") as stream:
        result = subprocess.run(
            ["bash", "-c", launcher_exit_script], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    assert result.returncode == worker_exit
    record = load_run(run)
    assert record["attempts"][-1]["exit_code"] == worker_exit
    assert record["status"] == ("failed" if worker_exit else "complete")
    assert log.read_text() == "worker finished\n"
    assert record["log"]["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
