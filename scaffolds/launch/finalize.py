"""Seal a job's log and final artifact hashes after its child process exits."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path

from scaffolds.records import capture_artifacts, load_run, update_record


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
        changes["error"] = f"Job process exited with code {exit_code}; see training.log"
    update_record(run, changes)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    finalize_attempt(args.run, args.exit_code)


if __name__ == "__main__":
    main()
