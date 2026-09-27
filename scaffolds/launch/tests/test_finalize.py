"""Final records include the complete log after the worker has exited."""
import hashlib
import json
from datetime import UTC, datetime

import pytest

from scaffolds.launch.finalize import finalize_attempt
from scaffolds.records import create_run, load_run, update_record, validate_record


def test_finalization_hashes_closed_outputs_and_records_attempt_exit(tmp_path, monkeypatch):
    run = create_run(tmp_path / "study", {"seed": 0})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    update_record(run, {"status": "complete", "attempts": [{"slurm_job_id": "123"}]})
    (run / "metrics.jsonl").write_text('{"step":4}\n')
    log = run / "training.log"
    log.write_text("training complete\nW&B flushed\n")
    finalize_attempt(run, 0)
    record = load_run(run)
    assert record["log"]["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert record["metrics"]["sha256"] == hashlib.sha256((run / "metrics.jsonl").read_bytes()).hexdigest()
    assert record["attempts"][-1]["exit_code"] == 0
    assert record["status"] == "complete"
    assert not validate_record(run)["unavailable"]


def test_aborted_worker_records_failure_but_preserves_numerical_classification(tmp_path):
    run = create_run(tmp_path / "study", {"seed": 0})
    update_record(run, {"status": "running"})
    finalize_attempt(run, 7)
    assert load_run(run)["status"] == "failed"
    update_record(run, {"status": "numerical_failure", "error": "nonfinite gradient"})
    finalize_attempt(run, 1)
    assert load_run(run)["status"] == "numerical_failure"
    assert load_run(run)["error"] == "nonfinite gradient"


@pytest.mark.parametrize("numerical", [False, True])
def test_acceptance_failure_records_stage_and_diagnostics(tmp_path, monkeypatch, numerical):
    run = create_run(tmp_path / "study", {"seed": 0})
    directory = run / ".runtime/acceptance/H100-4"
    (directory / "pilot").mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    attempt = {"slurm_job_id": "123", "started_utc": datetime.now(UTC).isoformat(),
               "acceptance_directory": str(directory)}
    update_record(run, {"status": "acceptance", "attempts": [attempt]})
    (directory / "failure.json").write_text(json.dumps({
        "slurm_job_id": "123", "stage": "pilot" if numerical else "gpu-tests", "error": "qualification failed",
        "failed_utc": datetime.now(UTC).isoformat(),
    }))
    if numerical:
        diagnostic = directory / "pilot/numerical_failure_step_00000003.json"
        diagnostic.write_text(json.dumps({"step": 3, "error": "nonfinite gradient", "timestamp_utc": datetime.now(UTC).isoformat()}))
        (directory / "pilot/metadata.json").write_text(json.dumps({
            "numerical_failure_path": str(diagnostic), "updated_utc": datetime.now(UTC).isoformat(),
        }))
    (run / "training.log").write_text("worker traceback\n")
    finalize_attempt(run, 1)
    record = load_run(run)
    assert record["status"] == ("acceptance_numerical_failure" if numerical else "acceptance_failed")
    assert record["acceptance"]["stage"] == ("pilot" if numerical else "gpu-tests")
    assert record["attempts"][-1]["exit_code"] == 1
    assert not (run / "completion.json").exists()
    if numerical:
        assert record["acceptance"]["numerical_failure"]["diagnostics"]["step"] == 3
    assert not validate_record(run)["unavailable"]


def test_requeue_does_not_reuse_an_old_acceptance_failure(tmp_path, monkeypatch):
    run = create_run(tmp_path / "study", {"seed": 0})
    directory = run / ".runtime/acceptance/H100-4"
    directory.mkdir(parents=True)
    failure = directory / "failure.json"
    failure.write_text(json.dumps({"slurm_job_id": "123", "stage": "gpu-tests", "error": "old failure",
                                   "failed_utc": "2000-01-01T00:00:00+00:00"}))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    update_record(run, {"status": "acceptance", "attempts": [{
        "slurm_job_id": "123", "started_utc": datetime.now(UTC).isoformat(), "acceptance_directory": str(directory),
    }]})
    finalize_attempt(run, 1)
    assert load_run(run)["status"] == "failed"
    assert "acceptance" not in load_run(run)


def test_requeue_does_not_attribute_stale_pilot_numerics_to_new_resume_failure(tmp_path, monkeypatch):
    run = create_run(tmp_path / "study", {"seed": 0})
    directory = run / ".runtime/acceptance/H100-4"
    (directory / "pilot").mkdir(parents=True)
    diagnostic = directory / "pilot/numerical_failure_step_00000003.json"
    diagnostic.write_text(json.dumps({"step": 3, "timestamp_utc": "2000-01-01T00:00:00+00:00"}))
    (directory / "pilot/metadata.json").write_text(json.dumps({
        "numerical_failure_path": str(diagnostic), "updated_utc": "2000-01-01T00:00:00+00:00",
    }))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    update_record(run, {"status": "acceptance", "attempts": [{
        "slurm_job_id": "123", "started_utc": datetime.now(UTC).isoformat(), "acceptance_directory": str(directory),
    }]})
    (directory / "failure.json").write_text(json.dumps({
        "slurm_job_id": "123", "stage": "resume", "error": "checkpoint unavailable",
        "failed_utc": datetime.now(UTC).isoformat(),
    }))
    finalize_attempt(run, 1)
    record = load_run(run)
    assert record["status"] == "acceptance_failed"
    assert "numerical_failure" not in record["acceptance"]
