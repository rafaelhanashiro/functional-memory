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
    "scaffolds/transformer/launchers/orcd.sbatch",
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


def test_orcd_launcher_time_limit_fits_both_partitions():
    launcher = (ROOT / "scaffolds/transformer/launchers/orcd.sbatch").read_text()

    def hours(option):
        h, m, s = re.search(rf"^#SBATCH --{option}=(\d+):(\d\d):(\d\d)$", launcher, re.M).groups()
        return int(h) + int(m) / 60 + int(s) / 3600

    # Backfill trims --time to mit_normal_gpu's 6-hour maximum only when --time-min fits it.
    assert hours("time-min") <= 6 < hours("time") <= 48


def test_orcd_launcher_requeues_its_worker_before_the_time_limit(tmp_path):
    launcher = (ROOT / "scaffolds/transformer/launchers/orcd.sbatch").read_text()
    assert re.search(r"^#SBATCH --signal=B:USR1@\d+$", launcher, re.M)
    assert re.search(r"^#SBATCH --requeue$", launcher, re.M)
    requeue = re.search(r"^requeue_before_limit\(\) \{\n.*?^run_step\(\) .*?$", launcher, re.M | re.S)
    assert requeue is not None, "Missing the time-limit requeue trap"
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SCONTROL_ARGS"\n')
    scontrol.chmod(0o755)
    args_path, worker = tmp_path / "args", tmp_path / "worker.pid"
    # The signal arrives while a long worker runs; the trap must not wait for it to finish.
    # Slurm kills the worker on requeue; here it is detached from the output and killed below.
    script = ('set -euo pipefail\n' + requeue.group()
              + '\n(sleep 1; kill -USR1 $$) &\n'
              + 'run_step bash -c \'echo $$ > "$WORKER_PID"; exec sleep 60\' >/dev/null 2>&1\n'
              + 'printf "worker finished\\n"\n')
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
           "SLURM_JOB_ID": "123", "SCONTROL_ARGS": str(args_path), "WORKER_PID": str(worker)}
    try:
        result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=30)
    finally:
        if worker.exists():
            subprocess.run(["kill", worker.read_text().strip()], capture_output=True)
    assert result.returncode == 128 + 10
    assert args_path.read_text().splitlines() == ["requeue", "123"]
    assert "worker finished" not in result.stdout
