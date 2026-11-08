"""Run monitoring must distinguish failure, scheduler outages and delivery uncertainty."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess

import pytest

from scaffolds.launch import run_monitor as monitor


def iso(seconds):
    return datetime.fromtimestamp(seconds, UTC).isoformat()


@pytest.fixture
def setup(tmp_path):
    run = tmp_path / "r123456789abc"
    run.mkdir()
    record = {
        "status": "acceptance", "submission": {"slurm_job_id": "22", "submitted_utc": iso(100)},
        "attempts": [{"slurm_job_id": "22", "started_utc": iso(200), "slurm_restart_count": "0"}],
    }
    (run / "record.json").write_text(json.dumps(record))
    (run / "training.log").write_text("Previous attempt: CUDA_ERROR_ILLEGAL_ADDRESS\n")
    os.utime(run / "training.log", (900, 900))
    config = {"thread_id": "test-thread", "state_path": str(tmp_path / "state.json"),
              "codex": "test-codex", "end_utc": iso(100000), "runs": [{"path": str(run)}]}
    return config, run, record


class Commands:
    def __init__(self, scheduler="RUNNING", delivery="ok", outage=False):
        self.scheduler, self.delivery, self.outage = scheduler, delivery, outage
        self.messages = []

    def __call__(self, args, **kwargs):
        if args[0] in {"squeue", "sacct"}:
            if self.outage:
                return subprocess.CompletedProcess(args, 1, "", "controller unavailable")
            text = f"22|{self.scheduler}\n" if args[0] == "squeue" else ""
            return subprocess.CompletedProcess(args, 0, text, "")
        assert args[:3] == ["test-codex", "queue", "--thread"]
        assert args[-1] == "--resume"
        self.messages.append(args[args.index("--message") + 1])
        if self.delivery == "error":
            return subprocess.CompletedProcess(args, 1, "", "queue connection failed")
        warning = "Warning: queued, but failed to resume thread" if self.delivery == "warning" else ""
        return subprocess.CompletedProcess(args, 0, "Queued message", warning)


def rewrite(run, record, **changes):
    record.update(changes)
    (run / "record.json").write_text(json.dumps(record))


def test_new_pending_submission_wins_over_failed_attempt(setup):
    config, run, record = setup
    record["submission"] = {"slurm_job_id": "22", "submitted_utc": iso(300)}
    record["attempts"] = [{"slurm_job_id": "11", "started_utc": iso(100), "exit_code": 1}]
    rewrite(run, record, status="submitted")
    assert monitor.current_job(record) == ("22", {})
    state, code = monitor.poll(config, {}, now=1000, command=Commands(scheduler="PENDING"))
    assert code == 0
    assert state["snapshots"][0]["phase"] == "queued"


def test_notifications_on_transitions_not_every_poll_or_old_log_errors(setup):
    config, run, record = setup
    command = Commands()
    state, _ = monitor.poll(config, {}, now=1000, command=command)
    assert state["snapshots"][0]["phase"] == "acceptance"
    state, _ = monitor.poll(config, state, now=1300, command=command)
    assert len(command.messages) == 1
    rewrite(run, record, status="running")
    state, _ = monitor.poll(config, state, now=1600, command=command)
    assert len(command.messages) == 2
    assert "acceptance \\u2192 running" in command.messages[-1]
    state, _ = monitor.poll(config, state, now=1900, command=command)
    assert len(command.messages) == 2


def test_failures_notify_promptly_and_monitor_keeps_watching_retries(setup):
    config, run, record = setup
    command = Commands()
    state, _ = monitor.poll(config, {}, now=1000, command=command)
    command.scheduler = "FAILED"
    rewrite(run, record, status="acceptance_failed", error="current acceptance GPU failure")
    state, _ = monitor.poll(config, state, now=1300, command=command)
    assert len(command.messages) == 2
    assert "current acceptance GPU failure" in command.messages[-1]
    assert not state["disabled"]
    state, _ = monitor.poll(config, state, now=1600, command=command)
    assert len(command.messages) == 2
    command.scheduler = "PENDING"
    rewrite(run, record, status="submitted")
    state, _ = monitor.poll(config, state, now=1900, command=command)
    assert len(command.messages) == 3


def test_scheduler_outage_is_not_run_failure(setup):
    config, _, _ = setup
    state, code = monitor.poll(config, {}, now=1000, command=Commands(outage=True))
    assert code == 0
    assert state["snapshots"][0]["phase"] == "acceptance"
    assert state["snapshots"][0]["scheduler_state"] == "UNKNOWN"
    assert len(state["scheduler_errors"]) == 2
    assert not state["disabled"]


def test_stall_is_deduplicated_and_progress_recovery_notifies(setup):
    config, run, record = setup
    rewrite(run, record, status="running")
    command = Commands()
    state, _ = monitor.poll(config, {}, now=1000, command=command)
    state, _ = monitor.poll(config, state, now=2200, command=command)
    assert state["snapshots"][0]["stalled"]
    assert len(command.messages) == 2
    state, _ = monitor.poll(config, state, now=2500, command=command)
    assert len(command.messages) == 2
    os.utime(run / "training.log", (2600, 2600))
    state, _ = monitor.poll(config, state, now=2700, command=command)
    assert not state["snapshots"][0]["stalled"]
    assert "progress resumed" in command.messages[-1]


def test_acceptance_has_longer_stall_threshold(setup):
    config, _, _ = setup
    command = Commands()
    state, _ = monitor.poll(config, {}, now=2200, command=command)
    assert not state["snapshots"][0]["stalled"]
    state, _ = monitor.poll(config, state, now=4600, command=command)
    assert state["snapshots"][0]["stalled"]


def test_resume_warning_is_not_confirmed_delivery_or_duplicate_enqueue(setup):
    config, run, record = setup
    command = Commands(delivery="warning")
    state, code = monitor.poll(config, {}, now=1000, command=command)
    assert code == 2
    assert state["delivery"]["status"] == "queued_resume_unconfirmed"
    assert not state.get("last_sent")
    (run / "metrics.jsonl").write_text('{"step":10}\n')
    state, code = monitor.poll(config, state, now=1300, command=command)
    assert code == 2
    assert len(command.messages) == 1


def test_failed_queue_retries_then_deduplicates(setup):
    config, _, _ = setup
    command = Commands(delivery="error")
    state, code = monitor.poll(config, {}, now=1000, command=command)
    assert code == 2
    command.delivery = "ok"
    state, code = monitor.poll(config, state, now=1300, command=command)
    assert code == 0 and not state.get("pending")
    state, _ = monitor.poll(config, state, now=1600, command=command)
    assert len(command.messages) == 2


def test_complete_disables_only_after_delivery_and_expiry_notifies(setup):
    config, run, record = setup
    rewrite(run, record, status="complete")
    command = Commands(scheduler="COMPLETED")
    state, _ = monitor.poll(config, {}, now=1000, command=command)
    assert state["disabled"]
    monitor.poll(config, state, now=1300, command=command)
    assert len(command.messages) == 1
    rewrite(run, record, status="acceptance")
    state, _ = monitor.poll(config, {}, now=100001, command=Commands())
    assert state["disabled"] and state["expired"]


def test_dry_run_does_not_send_or_write_state(setup, capsys):
    config, _, _ = setup
    command = Commands()
    state, code = monitor.poll(config, {}, now=1000, dry_run=True, command=command)
    assert code == 0 and state == {}
    assert not command.messages
    assert not Path(config["state_path"]).exists()
    assert "Monitoring activated" in capsys.readouterr().out


def test_last_metric_ignores_partial_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":10}\n{"step":20}\n{"step":')
    assert monitor.last_metric(path) == {"step": 20}


def test_latest_validation_is_preserved_between_evaluations(setup):
    config, run, _ = setup
    (run / "metrics.jsonl").write_text(
        '{"step":500,"validation_perplexity":14.5}\n{"step":510,"loss":2.6}\n')
    state, _ = monitor.poll(config, {}, now=1000, command=Commands())
    row = state["snapshots"][0]
    assert row["step"] == 510
    assert row["validation_step"] == 500 and row["validation_ppl"] == 14.5


def test_completion_reports_actual_best_and_terminal_schema(setup):
    config, run, record = setup
    rewrite(run, record, status="complete")
    (run / "best_validation.json").write_text('{"step":7500,"validation_perplexity":13.8}')
    (run / "completion.json").write_text('{"final_step":7570,"validation_perplexity":13.9}')
    state, _ = monitor.poll(config, {}, now=1000, command=Commands(scheduler="COMPLETED"))
    row = state["snapshots"][0]
    assert row["best_validation_ppl"] == 13.8 and row["terminal_ppl"] == 13.9


def test_requeue_restart_counter_notifies(setup):
    config, run, record = setup
    command = Commands()
    state, _ = monitor.poll(config, {}, now=1000, command=command)
    record["attempts"][-1]["slurm_restart_count"] = "1"
    rewrite(run, record)
    state, _ = monitor.poll(config, state, now=1300, command=command)
    assert len(command.messages) == 2
    assert state["snapshots"][0]["restart"] == "1"
