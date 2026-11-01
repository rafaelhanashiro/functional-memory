"""Submit and record isolated native FLA jobs on Kennel.

The scheduler and CUDA toolkit are execution settings; neither changes the
upstream model or scientific recipe. Qualification has its own log and outcome.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.native_fla.config import load_config


ROOT = Path(__file__).resolve().parents[3]
REQUEUE_EXIT_CODE = 75


def resolve_run(value: str) -> Path:
    path = Path(value)
    if re.fullmatch(r"r[0-9a-f]{12}", value):
        path = ROOT / "experiments" / "transformer_pp" / "runs" / value
    elif not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("The run must be inside this worktree") from error
    if not re.fullmatch(r"experiments/[A-Za-z0-9_]+/runs/r[0-9a-f]{12}", relative):
        raise ValueError("Expected experiments/<study>/runs/<run_id>")
    return path


def check_inputs(run: Path, launch_commit: str, *, allow_complete: bool = False) -> dict:
    study = run.parent.parent.name
    changed = changes(ROOT, launch_commit, study, "transformer")
    if changed:
        raise ValueError("Frozen execution set changed since launch: " + ", ".join(changed))
    subprocess.run(["git", "cat-file", "-e", f"{launch_commit}:{(run / 'config.yaml').relative_to(ROOT)}"],
                   cwd=ROOT, check=True, capture_output=True)
    config = load_config(run / "config.yaml")
    for field, expected in (("workdir", run), ("checkpoint_dir", run / "checkpoints")):
        actual = Path(config["train"][field])
        if not actual.is_absolute():
            actual = ROOT / actual
        if actual.resolve() != expected:
            raise ValueError(f"train.{field} must remain inside its canonical run")
    validate_record(run, verify_hashes=False)
    record = load_run(run)
    if record.get("home_cluster") != "kennel":
        raise ValueError("This launcher requires a run whose recorded home cluster is kennel")
    actual_hash = hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest()
    if record["config"]["sha256"] != actual_hash:
        raise ValueError("Canonical recipe checksum differs from the run record")
    if record.get("status") == "complete" and not allow_complete:
        raise ValueError("This run is already complete")
    return record


def submission_command(run: Path, launch_commit: str, *, partition: str, time_limit: str,
                       acceptance: bool = False) -> list[str]:
    log = run / ".runtime" / "acceptance.log" if acceptance else run / "training.log"
    return [
        "sbatch", "--parsable", "--hold", f"--partition={partition}", f"--time={time_limit}",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--mem=128G", "--gres=gpu:b300:1",
        "--requeue", "--signal=B:USR1@600", "--open-mode=append",
        f"--job-name={'accept-' if acceptance else ''}{run.name}",
        f"--chdir={ROOT}", f"--output={log}", f"--error={log}",
        str(ROOT / "scaffolds/transformer/native_fla/launch.sh"),
        str(run.relative_to(ROOT)), launch_commit, "acceptance" if acceptance else "production",
    ]


def submit(run: Path, launch_commit: str, *, partition: str, time_limit: str,
           acceptance: bool = False, dry_run: bool = False) -> str | None:
    check_inputs(run, launch_commit)
    command = submission_command(run, launch_commit, partition=partition, time_limit=time_limit,
                                 acceptance=acceptance)
    if dry_run:
        print(shlex.join(command))
        return None
    for name in (run.name, "accept-" + run.name):
        queued = subprocess.check_output(["squeue", "-h", "--user", str(os.getuid()),
                                          "--name", name, "-o", "%A"], text=True).strip()
        if queued:
            raise ValueError(f"Run already has a queued allocation: {queued}")
    (run / ".runtime").mkdir(exist_ok=True)
    reply = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    job_id = reply.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {reply!r}")
    phase = "acceptance" if acceptance else "production"
    try:
        fields = {
            "source_commit": launch_commit, "home_cluster": "kennel",
            "frozen_execution_sha256": frozen_hash(ROOT, launch_commit, run.parent.parent.name, "transformer"),
            "submission": {"phase": phase, "slurm_job_id": job_id, "partition": partition,
                           "time_limit": time_limit, "command": command,
                           "submitted_utc": datetime.now(UTC).isoformat()},
        }
        if not acceptance:
            fields["status"] = "submitted"
        update_record(run, fields)
    except Exception:
        # The new job is still held and has executed nothing.
        subprocess.run(["scancel", job_id], check=True, capture_output=True)
        raise
    subprocess.run(["scontrol", "release", job_id], check=True, capture_output=True)
    print(job_id)
    return job_id


def start_attempt(run: Path, launch_commit: str, acceptance: bool) -> None:
    record = check_inputs(run, launch_commit, allow_complete=True)
    attempt = {
        "phase": "acceptance" if acceptance else "production", "launch_commit": launch_commit,
        "config_path": str((run / "config.yaml").relative_to(ROOT)),
        "config_sha256": hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "allocation_seconds": int(os.environ["TRANSFORMER_ALLOCATION_SECONDS"]),
        "started_utc": datetime.now(UTC).isoformat(), "home_cluster": "kennel",
        "gpu_type": "B300", "gpu_count": 1, "cuda_environment": "native_fla_cuda13",
        "node": os.environ.get("SLURMD_NODENAME"),
        "allocation": subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                                              text=True).strip(),
    }
    fields = {"attempts": [*record.get("attempts", []), attempt], "source_commit": launch_commit,
              "frozen_execution_sha256": frozen_hash(ROOT, launch_commit, run.parent.parent.name, "transformer")}
    if acceptance:
        fields["acceptance"] = {"status": "running", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                                "directory": str(run / ".runtime" / "acceptance")}
    else:
        fields.update(status=record.get("status") if record.get("status") == "complete" else "running",
                      log_path=str(run / "training.log"))
    update_record(run, fields)


def finish_attempt(run: Path, exit_code: int, acceptance: bool) -> None:
    record = load_run(run)
    attempts = record.get("attempts", [])
    fields = {}
    if (attempts and attempts[-1].get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
            and str(attempts[-1].get("slurm_restart_count", "0")) == os.environ.get("SLURM_RESTART_COUNT", "0")
            and attempts[-1].get("phase") == ("acceptance" if acceptance else "production")):
        fields["attempts"] = [*attempts[:-1], {**attempts[-1], "exit_code": exit_code,
                                               "finished_utc": datetime.now(UTC).isoformat()}]
    if acceptance:
        log = run / ".runtime" / "acceptance.log"
        outcome = {"status": "passed" if exit_code == 0 else "failed", "exit_code": exit_code}
        if log.is_file():
            outcome["log"] = {"path": str(log), "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        success = run / ".runtime" / "acceptance" / "success.json"
        if exit_code == 0 and success.is_file():
            outcome["summary"] = json.loads(success.read_text())
            outcome["sha256"] = hashlib.sha256(success.read_bytes()).hexdigest()
        fields["acceptance"] = outcome
    else:
        fields.update(capture_artifacts(run, include_log=True))
        if exit_code == REQUEUE_EXIT_CODE:
            fields["status"] = "submitted"
        elif exit_code and record.get("status") in {"planned", "submitted", "running"}:
            fields.update(status="failed", error=f"Native FLA process exited with code {exit_code}; see training.log")
    update_record(run, fields)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--partition", help="Default: standard; debug with --acceptance")
    parser.add_argument("--time", dest="time_limit", help="Default: 24:00:00; 02:00:00 with --acceptance")
    parser.add_argument("--launch-commit")
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--event", choices=("submit", "start", "finish"), default="submit", help=argparse.SUPPRESS)
    parser.add_argument("--exit-code", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run = resolve_run(args.run)
    if args.event == "finish":
        if args.exit_code is None:
            parser.error("--exit-code is required for finish")
        finish_attempt(run, args.exit_code, args.acceptance)
        return
    launch_commit = args.launch_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.event == "start":
        start_attempt(run, launch_commit, args.acceptance)
        return
    submit(run, launch_commit, partition=args.partition or ("debug" if args.acceptance else "standard"),
           time_limit=args.time_limit or ("02:00:00" if args.acceptance else "24:00:00"),
           acceptance=args.acceptance, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
