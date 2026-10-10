"""Seal a job's log and final artifact hashes after its child process exits."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path

from scaffolds.records import capture_artifacts, load_run, update_record

STALLED_EXIT_CODE = 124
"""Exit code with which a launcher's watchdog (``watchdog.sh``) ends a job whose log stopped growing."""


def _acceptance_failure(attempt: dict) -> dict:
    """Retain qualification failures separately from production outcomes."""
    if not attempt.get("acceptance_directory") or not attempt.get("started_utc"):
        return {}
    directory = Path(attempt["acceptance_directory"])
    failure_path = directory / "failure.json"
    started = datetime.fromisoformat(attempt["started_utc"])
    if not failure_path.is_file():
        return {}
    failure = json.loads(failure_path.read_text())
    def current(timestamp: str | None) -> bool:
        return timestamp is not None and datetime.fromisoformat(timestamp) >= started

    if (failure.get("slurm_job_id") != attempt.get("slurm_job_id")
            or not current(failure.get("failed_utc"))):
        # Use worker timestamps, not NFS mtimes, to distinguish requeues.
        return {}

    def evidence(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    summary = {**failure, "status": "failed", "failure": evidence(failure_path)}
    status = "acceptance_failed"
    metadata_path = directory / "pilot" / "metadata.json"
    if failure.get("stage") in {"pilot", "resume"} and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        diagnostic = metadata.get("numerical_failure_path")
        if current(metadata.get("updated_utc")):
            summary["pilot_metadata"] = evidence(metadata_path)
        if current(metadata.get("updated_utc")) and diagnostic and Path(diagnostic).is_file():
            path = Path(diagnostic)
            diagnostics = json.loads(path.read_text())
            if current(diagnostics.get("timestamp_utc")):
                summary["status"] = "numerical_failure"
                summary["numerical_failure"] = {"artifact": evidence(path), "diagnostics": diagnostics}
                status = "acceptance_numerical_failure"
    return {"status": status, "acceptance": summary, "error": failure["error"]}


def finalize_attempt(run: Path, exit_code: int) -> None:
    run = Path(run)
    if not (run / "record.json").exists():
        return
    record = load_run(run)
    changes = capture_artifacts(run, include_log=True)
    attempts = record.get("attempts", [])
    if attempts and attempts[-1].get("slurm_job_id") == os.environ.get("SLURM_JOB_ID"):
        changes["attempts"] = [*attempts[:-1], {**attempts[-1], "exit_code": exit_code,
                                "finished_utc": datetime.now(UTC).isoformat()}]
    if exit_code and record.get("status") in {"planned", "submitted", "acceptance", "running"}:
        changes["status"] = "failed"
        changes["error"] = (f"Stalled: the log stopped growing and the launcher ended the job (exit code {exit_code}); "
                            "see training.log" if exit_code == STALLED_EXIT_CODE
                            else f"Job process exited with code {exit_code}; see training.log")
        if (record.get("status") == "acceptance" and attempts
                and attempts[-1].get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")):
            changes.update(_acceptance_failure(attempts[-1]))
    update_record(run, changes)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    finalize_attempt(args.run, args.exit_code)


if __name__ == "__main__":
    main()
