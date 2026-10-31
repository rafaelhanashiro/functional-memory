"""CPU checks for scientific outcome separation and checkpoint-safe Slurm exits."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import pytest

from scaffolds.transformer.native_fla import launch


def test_other_study_is_supported_and_paths_cannot_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    value = "experiments/another_study/runs/r0123456789ab"
    assert launch.resolve_run(value) == tmp_path / value
    with pytest.raises(ValueError, match="inside this worktree"):
        launch.resolve_run(str(tmp_path.parent / "outside"))


@pytest.mark.parametrize("acceptance,partition,limit", [(False, "standard", "24:00:00"),
                                                      (True, "debug", "02:00:00")])
def test_submission_uses_qualified_gpu_and_advance_signal(tmp_path, monkeypatch, acceptance, partition, limit):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    run = tmp_path / "experiments/study/runs/r0123456789ab"
    command = launch.submission_command(run, "a" * 40, partition=partition, time_limit=limit, acceptance=acceptance)
    for flag in ("--hold", "--gres=gpu:b300:1", "--signal=B:USR1@600", f"--partition={partition}", f"--time={limit}"):
        assert flag in command
    assert not any("dependency" in arg or "nice" in arg for arg in command)
    log = run / ".runtime/acceptance.log" if acceptance else run / "training.log"
    assert f"--output={log}" in command
    assert command[-1] == ("acceptance" if acceptance else "production")


@pytest.fixture
def record_updates(monkeypatch):
    record = {"status": "running", "attempts": [{"phase": "production", "slurm_job_id": "123",
                                                   "slurm_restart_count": "0"}]}
    updates = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setattr(launch, "load_run", lambda _: record)
    monkeypatch.setattr(launch, "capture_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(launch, "update_record", lambda _, fields: updates.append(fields))
    return record, updates


def test_checkpointed_requeue_is_submitted_not_failed(tmp_path, record_updates):
    _, updates = record_updates
    launch.finish_attempt(tmp_path, 75, False)
    assert updates[-1]["status"] == "submitted"
    assert updates[-1]["attempts"][-1]["exit_code"] == 75


def test_acceptance_failure_does_not_replace_production_outcome(tmp_path, record_updates):
    _, updates = record_updates
    launch.finish_attempt(tmp_path, 2, True)
    assert "status" not in updates[-1]
    assert "attempts" not in updates[-1]
    assert updates[-1]["acceptance"]["status"] == "failed"


def test_numerical_failure_remains_a_scientific_failure(tmp_path, record_updates):
    record, updates = record_updates
    record["status"] = "numerical_failure"
    launch.finish_attempt(tmp_path, 1, False)
    assert "status" not in updates[-1]


def test_early_restart_failure_does_not_rewrite_previous_attempt(tmp_path, record_updates, monkeypatch):
    _, updates = record_updates
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    launch.finish_attempt(tmp_path, 1, False)
    assert "attempts" not in updates[-1]


def test_completed_restart_keeps_completion_status(tmp_path, monkeypatch, record_updates):
    record, updates = record_updates
    record.update(status="complete")
    run = tmp_path / "experiments/study/runs/r0123456789ab"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("frozen recipe")
    calls = []
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "check_inputs", lambda *args, **kwargs: calls.append(kwargs) or record)
    monkeypatch.setattr(launch, "frozen_hash", lambda *args: "frozen")
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *args, **kwargs: "allocation")
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", "86400")
    launch.start_attempt(run, "a" * 40, False)
    assert calls == [{"allow_complete": True}]
    assert updates[-1]["status"] == "complete"


def _shell_functions():
    source = Path(launch.__file__).with_name("launch.sh").read_text()
    return source[source.index('TRAINER_PID=""'):source.index("finish() {")]


def test_advance_signal_waits_for_durable_child_checkpoint(tmp_path):
    worker = tmp_path / "worker.py"
    ready, saved = tmp_path / "ready", tmp_path / "saved"
    worker.write_text("""import signal, sys, time
from pathlib import Path
def checkpoint(*_):
    time.sleep(0.15)
    Path(sys.argv[2]).write_text('durable')
    raise SystemExit(75)
signal.signal(signal.SIGUSR1, checkpoint)
Path(sys.argv[1]).write_text('ready')
while True: time.sleep(0.01)
""")
    script = "set -euo pipefail\n" + _shell_functions() + '\nrun_trainer "$@"\n'
    process = subprocess.Popen(["bash", "-c", script, "launcher", sys.executable, str(worker), str(ready), str(saved)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.kill(process.pid, signal.SIGUSR1)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 75, stderr
        assert saved.read_text() == "durable"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_signal_before_trainer_start_requeues_without_starting_worker(tmp_path):
    forbidden = tmp_path / "worker_started"
    script = ("set -euo pipefail\n" + _shell_functions()
              + '\nrequest_checkpoint\nrun_trainer touch "$1"\n')
    result = subprocess.run(["bash", "-c", script, "launcher", str(forbidden)], capture_output=True, text=True)
    assert result.returncode == 75, result.stderr
    assert not forbidden.exists()
