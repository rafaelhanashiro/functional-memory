"""Exercise launcher EXIT traps without allocating hardware or running workers."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from scaffolds.launch.finalize import (
    STALL_REQUEUES, STALLED_EXIT_CODE, TERMINATED_EXIT_CODE, finalize_attempt, main as finalize_main,
)
from scaffolds.records import create_run, load_run, update_record


ROOT = Path(__file__).resolve().parents[3]
WATCHDOG = ROOT / "scaffolds/launch/watchdog.sh"
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
    return (f'set -euo pipefail\nsource {WATCHDOG}\n' + trap.group()
            + '\nprintf "worker finished\\n"\n(exit "$WORKER_EXIT")\n')


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


@pytest.mark.parametrize("launcher", LAUNCHERS[:2])
@pytest.mark.parametrize("override", (None, "", "2", "4"))
def test_transformer_gpu_defaults_match_allocation_and_allow_explicit_overrides(launcher, override):
    source = (ROOT / launcher).read_text()
    requested = int(re.search(r"^#SBATCH --gres=gpu(?::[A-Za-z0-9_]+)?:(\d+)$", source, re.M).group(1))
    # Execute argument parsing without entering the worktree or touching hardware.
    parsing = source[source.index("set -euo pipefail"):source.index("EXECUTION_ROOT=")]
    args = ["acceptance.module", "config.yaml", "launch-commit"]
    if override is not None:
        args.append(override)
    result = subprocess.run(
        ["bash", "-c", parsing + '\nprintf "%s\\n" "$GPU_COUNT"\n', "launcher", *args],
        check=True, text=True, capture_output=True,
    )
    expected = int(override) if override else requested
    assert requested == 1
    assert int(result.stdout.strip()) == expected


def test_orcd_launcher_requeues_its_worker_before_the_time_limit(tmp_path):
    launcher = (ROOT / "scaffolds/transformer/launchers/orcd.sbatch").read_text()
    assert re.search(r"^#SBATCH --signal=B:USR1@\d+$", launcher, re.M)
    assert re.search(r"^#SBATCH --requeue$", launcher, re.M)
    requeue = re.search(r"^requeue_before_limit\(\) \{\n.*?^trap requeue_before_limit USR1$", launcher, re.M | re.S)
    assert requeue is not None, "Missing the time-limit requeue trap"
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SCONTROL_ARGS"\n')
    scontrol.chmod(0o755)
    args_path, worker = tmp_path / "args", tmp_path / "worker.pid"
    # The signal arrives while a long worker runs; the trap must not wait for it to finish.
    # Slurm kills the worker on requeue; here it is detached from the output and killed below.
    script = (f'set -euo pipefail\nsource {WATCHDOG}\n' + requeue.group()
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


@pytest.mark.parametrize("launcher", LAUNCHERS[:2])
def test_transformer_launchers_watch_their_log_and_run_long_steps_under_it(launcher):
    text = (ROOT / launcher).read_text()
    assert re.search(r"^source scaffolds/launch/watchdog\.sh$", text, re.M)
    assert re.search(r'^start_watchdog "\$RUN_ROOT/training\.log"$', text, re.M)
    assert re.search(r"^  local result=\$\?\n  local finalizer_result=0\n  trap - EXIT\n  stop_watchdog$", text, re.M)
    # A foreground step would defer the watchdog's signal until it finished.
    steps = re.findall(r'^(.*)"\$PYTHON" -m ("\$ADAPTER" "\$CONFIG" --directory|scaffolds\.transformer\.\w+)', text, re.M)
    assert [module for _, module in steps] == ['"$ADAPTER" "$CONFIG" --directory', "scaffolds.transformer.train",
                                               "scaffolds.transformer.retention"]
    assert all(prefix.strip() == "run_step" for prefix, _ in steps)


def test_the_finalizer_and_watchdog_agree_on_the_stall_exit_code():
    assert re.search(rf"^STALLED_EXIT_CODE={STALLED_EXIT_CODE}$", WATCHDOG.read_text(), re.M)


def _watched(tmp_path, worker):
    """Run a worker under the watchdog with a 2-second stall limit; return the result and its PIDs."""
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/bash\necho gpu-state\n")
    nvidia_smi.chmod(0o755)
    log, pids = tmp_path / "training.log", tmp_path / "pids"
    log.touch()
    script = (f"set -euo pipefail\nsource {WATCHDOG}\nSTALL_SECONDS=2 STALL_POLL_SECONDS=1\n"
              'start_watchdog "$LOG"\n'
              f"run_step bash -c 'echo $$ >> \"$PIDS\"; {worker}'\n"
              # The finalizer runs after the watchdog stops, as in the launchers' EXIT trap.
              'stop_watchdog\nsleep 1\nprintf "worker finished\\n"\n')
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "SLURM_JOB_ID": "123",
           "LOG": str(log), "PIDS": str(pids)}
    try:
        result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=30)
    finally:
        subprocess.run(["kill", "-KILL", *pids.read_text().split()], capture_output=True)
    return result, pids.read_text().split()


def _ended(pid):
    # A killed process stays a zombie until its new parent reaps it.
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def test_the_watchdog_ends_a_step_whose_log_stops_growing(tmp_path):
    # The worker's child keeps running, as a data-parallel peer waiting in a collective would.
    result, pids = _watched(tmp_path, 'sleep 60 & echo $! >> "$PIDS"; wait')
    assert result.returncode == STALLED_EXIT_CODE
    assert "ending stalled job 123" in result.stdout and "gpu-state" in result.stdout
    assert "worker finished" not in result.stdout
    assert len(pids) == 2 and all(map(_ended, pids))


def test_the_watchdog_leaves_a_step_that_keeps_logging(tmp_path):
    result, _ = _watched(tmp_path, 'for i in 1 2 3 4 5 6; do echo step=$i >> "$LOG"; sleep 0.5; done')
    assert result.returncode == 0
    # Stopping the watchdog leaves no job-termination notice in the run's log.
    assert result.stdout == "worker finished\n" and result.stderr == ""


def test_a_stalled_run_is_requeued_until_its_stall_limit_then_fails(tmp_path, monkeypatch):
    run = create_run(tmp_path / "study", {"seed": 0})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    attempts = []
    for stall in range(STALL_REQUEUES + 1):
        # A requeued job keeps its ID; each restart appends an attempt.
        attempts = [*load_run(run).get("attempts", []), {"slurm_job_id": "123", "slurm_restart_count": str(stall)}]
        update_record(run, {"status": "running", "attempts": attempts})
        requeue = finalize_attempt(run, STALLED_EXIT_CODE)
        record = load_run(run)
        assert record["attempts"][-1]["exit_code"] == STALLED_EXIT_CODE
        if stall < STALL_REQUEUES:
            assert requeue and record["status"] == "submitted" and "error" not in record
    assert not requeue
    assert record["status"] == "failed" and record["error"].startswith("Stalled:")
    assert len(record["attempts"]) == STALL_REQUEUES + 1


def test_an_ordinary_failure_is_never_requeued(tmp_path, monkeypatch):
    run = create_run(tmp_path / "study", {"seed": 0})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    update_record(run, {"status": "running", "attempts": [{"slurm_job_id": "123"}]})
    assert not finalize_attempt(run, 1)
    assert load_run(run)["status"] == "failed"


@pytest.mark.parametrize("status", ["acceptance", "running"])
def test_a_slurm_termination_is_recorded_without_failing_the_run(tmp_path, monkeypatch, status):
    run = create_run(tmp_path / "study", {"seed": 0})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    update_record(run, {"status": status, "attempts": [{"slurm_job_id": "123"}]})
    # Slurm requeues a preempted job itself; the finalizer neither requeues nor fails it.
    assert not finalize_attempt(run, TERMINATED_EXIT_CODE)
    record = load_run(run)
    assert record["attempts"][-1]["exit_code"] == TERMINATED_EXIT_CODE
    assert record["status"] == status and "error" not in record


def test_the_kennel_launcher_records_slurms_termination_of_a_running_step(tmp_path):
    launcher = (ROOT / "scaffolds/transformer/launchers/kennel.sbatch").read_text()
    traps = re.search(r"^finish_record\(\) \{\n.*?^trap finish_record EXIT\n(?:#.*\n)*trap 'exit (\d+)' TERM$",
                      launcher, re.M | re.S)
    assert traps is not None, "Missing the TERM trap after the EXIT trap"
    assert int(traps.group(1)) == TERMINATED_EXIT_CODE
    run = create_run(tmp_path / "study", {"seed": 0})
    update_record(run, {"status": "acceptance", "attempts": [{"slurm_job_id": "123"}]})
    worker = tmp_path / "worker.pid"
    # Slurm signals the whole job; here only the launcher, which must not wait for the step to finish.
    script = (f"set -euo pipefail\nsource {WATCHDOG}\n" + traps.group()
              + '\n(sleep 1; kill -TERM $$) &\n'
              + 'run_step bash -c \'echo $$ > "$WORKER_PID"; exec sleep 60\'\n'
              + 'printf "worker finished\\n"\n')
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(ROOT), "RUN_ROOT": str(run),
           "SLURM_JOB_ID": "123", "WORKER_PID": str(worker)}
    log = run / "training.log"
    try:
        with log.open("w") as stream:
            result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=30)
    finally:
        if worker.exists():
            subprocess.run(["kill", worker.read_text().strip()], capture_output=True)
    assert result.returncode == TERMINATED_EXIT_CODE
    record = load_run(run)
    assert record["attempts"][-1]["exit_code"] == TERMINATED_EXIT_CODE
    assert "finished_utc" in record["attempts"][-1]
    assert record["status"] == "acceptance" and "error" not in record
    assert "worker finished" not in log.read_text()
    assert record["log"]["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()


@pytest.mark.parametrize("refused", [False, True])
def test_the_finalizer_requeues_a_stalled_job_through_slurm(tmp_path, monkeypatch, refused):
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SCONTROL_ARGS"\n'
                        + ('echo "Requested operation not permitted" >&2\nexit 1\n' if refused else ""))
    scontrol.chmod(0o755)
    args_path = tmp_path / "args"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SCONTROL_ARGS", str(args_path))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    run = create_run(tmp_path / "study", {"seed": 0})
    update_record(run, {"status": "running", "attempts": [{"slurm_job_id": "123"}]})
    argv = ["--run", str(run), "--exit-code", str(STALLED_EXIT_CODE)]
    if refused:
        with pytest.raises(SystemExit):
            finalize_main(argv)
    else:
        finalize_main(argv)
    assert args_path.read_text().splitlines() == ["requeue", "123"]
    record = load_run(run)
    if refused:
        assert record["status"] == "failed" and "not permitted" in record["error"]
    else:
        assert record["status"] == "submitted"
