"""Final records include the complete log after the worker has exited."""
import hashlib

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
