"""Execute qualified allocations while retaining the immutable training source.

Run this file directly. It orchestrates frozen production workers in training_root;
no model, optimizer, schedule, data, or checkpoint implementation is replaced here.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import socket
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

try:
    from . import common
except ImportError:  # Direct script entry avoids importing orchestration as training code.
    import common


class ExistingNumericalFailure(FloatingPointError):
    """A scientifically failed coordinate must not be automatically retrained."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _run(policy: dict, manifest: dict, name: str) -> dict:
    candidates = [run for run in manifest["runs"] if run["name"] == name and not run.get("retained_completed", False)]
    if len(candidates) != 1 or name not in policy["assignments"]:
        raise ValueError("Production requires a known active coordinate assigned by this policy")
    if policy["assignments"][name] not in ("h100", "b300", "a6000"):
        raise ValueError("Unsupported assigned production profile")
    return candidates[0]


def _old_helpers(policy: dict):
    root = Path(policy["training_root"]).resolve()
    sys.path.insert(0, str(root))
    production = importlib.import_module("experiments.key_dimension_taylor.production")
    retention = importlib.import_module("experiments.key_dimension_taylor.retention")
    for module in (production, retention):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Production imported training helpers outside the immutable checkout")
    return SimpleNamespace(validate_segment_restart=production.validate_segment_restart,
                           finalize_run=retention.finalize_run)


def _segment(policy: dict, manifest: dict, run: dict, index: int) -> tuple[dict, dict]:
    """Accept only this profile's qualified, checkpoint-aligned 80% budget."""
    profile_name = policy["assignments"][run["name"]]
    if profile_name == "h100":
        root = Path(manifest["preflight_dir"])
        shared = _read(root / "success.json")
        qualification = _read(root / "checks" / f"{run['name']}.json")
        expected = {"identity": manifest["identity"], "source_commit": manifest["source_commit"], "status": "passed"}
        if any(shared.get(key) != value or qualification.get(key) != value for key, value in expected.items()):
            raise RuntimeError("H100 production lacks its matching original preflight")
    elif profile_name == "a6000":
        qualification = common.load_a6000_qualification(policy, manifest, run)
    else:
        root = Path(policy["qualification_root"]) / profile_name
        kernels = _read(root / "kernels.json")
        qualification = _read(root / "checks" / f"{run['name']}.json")
        expected = {"identity": policy["identity"], "policy_identity": policy["identity"],
                    "source_commit": manifest["source_commit"], "orchestrator_commit": policy["orchestrator_commit"],
                    "profile": profile_name, "scheduler": policy["profiles"][profile_name], "status": "passed"}
        if any(kernels.get(key) != value or qualification.get(key) != value for key, value in expected.items()):
            raise RuntimeError("B300 production lacks its matching kernel and coordinate qualification")
        if kernels.get("tests_passed") != 105:
            raise RuntimeError("B300 production requires all 105 immutable kernel checks")
    if any(qualification.get(key) != run[key] for key in ("qk_head_dim", "attention_degree")):
        raise RuntimeError("Qualification belongs to another scientific coordinate")
    runtime = qualification["runtime"]
    # Every authorized profile has eight-hour allocations and 20% headroom.
    if (runtime.get("limit_seconds") != 8 * 3600 or runtime.get("allocation_fraction") != 0.8
            or runtime.get("usable_seconds") != 0.8 * 8 * 3600):
        raise RuntimeError("Qualification changed the authorized wall-time budget")
    segments = runtime["segments"]
    total = run["config"]["optim"]["total_steps"]
    cursor = 0
    for segment in segments:
        start, end, seconds = segment["start_step"], segment["end_step"], segment["projected_seconds"]
        if (type(start) is not int or type(end) is not int or start != cursor or not start < end <= total
                or (end != total and end % 500 != 0) or not math.isfinite(seconds)
                or not 0 < seconds <= runtime["usable_seconds"]):
            raise RuntimeError("Qualification contains an invalid or oversized checkpoint segment")
        cursor = end
    if cursor != total or type(index) is not int or not 0 <= index < len(segments):
        raise ValueError("Unknown production segment or incomplete qualified schedule")
    return segments[index], qualification


def _base(policy: dict, manifest: dict, run: dict, index: int) -> dict:
    return {"identity": manifest["identity"], "sweep_identity": manifest["identity"],
            "policy_identity": policy["identity"], "source_commit": manifest["source_commit"],
            "orchestrator_commit": policy["orchestrator_commit"], "training_root": policy["training_root"],
            "profile": policy["assignments"][run["name"]], "run_name": run["name"], "segment_index": index,
            "qk_head_dim": run["qk_head_dim"], "attention_degree": run["attention_degree"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "actual_node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
            "actual_node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "actual_partition": os.environ.get("SLURM_JOB_PARTITION"),
            **common.execution_provenance(policy),
            **({"qualification_reference": policy["a6000_qualifications"][run["name"]]}
               if policy["assignments"][run["name"]] == "a6000" else {})}


def _lock_fd(handle) -> int:
    descriptor = handle if isinstance(handle, int) else handle.fileno()
    if descriptor < 0:
        raise RuntimeError("Production lock has no inheritable descriptor")
    return descriptor


def _hardware_worker(policy_path: Path, policy: dict, run: dict, output: Path, lock_fd: int) -> dict:
    # Release the hardware probe's JAX contexts before starting a training process.
    command = [sys.executable, str(Path(__file__).resolve()), "--policy", str(policy_path),
               "--run-name", run["name"], "--worker", "hardware", "--output", str(output)]
    subprocess.run(command, cwd=policy["orchestrator_root"], env=common.training_env(policy),
                   pass_fds=(lock_fd,), check=True)
    return _read(output)


def launch_worker(policy: dict, run: dict, worker: str, end_step: int, output: Path, lock_fd: int) -> dict:
    command = [sys.executable, "-m", "experiments.key_dimension_taylor.production",
               "--manifest", str(policy["training_manifest"]), "--run-name", run["name"],
               "--worker", worker, "--end-step", str(end_step), "--output", str(output)]
    # Inherit both the launcher's process group and the open flock descriptor.
    # The latter keeps a replacement out even if this parent exits before its child.
    process = subprocess.run(command, cwd=policy.get("runtime_root", policy["training_root"]), env=common.training_env(policy),
                             pass_fds=(lock_fd,), check=False)
    if process.returncode:
        failure = _read(output) if output.exists() else {}
        if failure.get("numerical_failure") is True:
            raise FloatingPointError(failure.get("error", "Frozen training worker failed numerically"))
        raise subprocess.CalledProcessError(process.returncode, command)
    return _read(output)


def _checkpoint_exists(run: dict) -> bool:
    root = Path(run["checkpoint_dir"])
    return root.exists() and any(path.is_dir() and path.name.isdigit() for path in root.iterdir())


def _has_numerical_failure(root: Path) -> bool:
    return bool(list(root.glob("numerical_failure_step_*.json"))) or (
        (root / "production_failure.json").exists()
        and _read(root / "production_failure.json").get("failure_kind") == "numerical")


def _record_launch(root: Path, base: dict, policy: dict, manifest: dict, hardware: dict, attempt: str) -> None:
    """Append execution provenance while keeping original submission history intact."""
    log_path = (Path(manifest["project_root"]) / "results" / "logs" / "key_dimension_taylor"
                / base["run_name"] / f"slurm-{base['slurm_job_id']}.out")
    event = {**base, "attempt": attempt, "hardware": hardware, "log_path": str(log_path),
             "started_utc": datetime.now(UTC).isoformat()}
    with common.locked(root / ".launch-metadata.lock"):
        path = root / "launch.json"
        launch = _read(path) if path.exists() else {}
        if launch.get("source_commit", manifest["source_commit"]) != manifest["source_commit"]:
            raise RuntimeError("Canonical launch record changed its scientific source")
        launch.setdefault("execution_history", []).append(event)
        launch.update(source_commit=manifest["source_commit"], sweep_identity=manifest["identity"],
                      log_path=str(log_path), active_execution=event)
        common.write_json(path, launch)
    metadata_path = root / "metadata.json"
    metadata = _read(metadata_path) if metadata_path.exists() else {}
    metadata.update(source_commit=manifest["source_commit"], execution_policy_identity=policy["identity"],
                    orchestrator_commit=policy["orchestrator_commit"], execution_profile=base["profile"],
                    execution_hardware=hardware, actual_node=base["actual_node"],
                    actual_partition=base["actual_partition"], slurm_job_id=base["slurm_job_id"], log_path=str(log_path))
    metadata.update(common.execution_provenance(policy))
    if "qualification_reference" in base:
        metadata["qualification_reference"] = base["qualification_reference"]
    common.write_json(metadata_path, metadata)


def _record_failure(root: Path, attempt_root: Path, base: dict, stage: str, error: BaseException,
                    *, hardware: dict | None, segment: dict | None) -> dict:
    numerical = [ _read(path) for path in sorted(root.glob("numerical_failure_step_*.json")) ]
    record = {**base, **(segment or {}), "stage": stage,
              "status": "suppressed" if isinstance(error, ExistingNumericalFailure) else "failed",
              "failure_kind": "numerical" if numerical or isinstance(error, FloatingPointError) else "infrastructure",
              "hardware": hardware, "numerical_failures": numerical,
              "error": f"{type(error).__name__}: {error}", "failed_utc": datetime.now(UTC).isoformat()}
    common.write_json(attempt_root / "failure.json", record)
    common.write_json(root / "failure_history" / f"segment-{base['segment_index']:03d}-{attempt_root.name}.json", record)
    if not (isinstance(error, ExistingNumericalFailure) and (root / "production_failure.json").exists()):
        common.write_json(root / "production_failure.json", record)
    common.write_json(attempt_root / "attempt.json", record)
    return record


def run_segment(policy: dict, manifest: dict, run_name: str, index: int, *, policy_path: Path) -> dict:
    common.verify_policy(policy, manifest)
    run = _run(policy, manifest, run_name)
    root = Path(run["workdir"])
    # The scheduler also excludes duplicate jobs. This lock covers the actual
    # execution and is inherited by children, including after parent termination.
    with common.locked(root / ".concurrent-production.lock") as handle:
        common.assert_legacy_jobs_terminal(policy, manifest, run_name)
        common.verify_policy(policy, manifest)
        base = _base(policy, manifest, run, index)
        attempt = f"{base['slurm_job_id'] or 'local'}-{base['slurm_restart_count']}-{uuid.uuid4().hex[:8]}"
        attempt_root = root / "execution_attempts" / attempt
        attempt_root.mkdir(parents=True, exist_ok=False)
        stage, hardware, segment = "qualification", None, None
        common.write_json(attempt_root / "attempt.json", {**base, "status": "running", "started_utc": datetime.now(UTC).isoformat()})
        try:
            if not (root / "completion.json").exists() and _has_numerical_failure(root):
                raise ExistingNumericalFailure("This coordinate already failed numerically; automatic repeat is suppressed")
            segment, qualification = _segment(policy, manifest, run, index)
            start, end = segment["start_step"], segment["end_step"]
            stage = "hardware"
            hardware = _hardware_worker(policy_path, policy, run, attempt_root / "hardware.json", _lock_fd(handle))
            helpers = _old_helpers(policy)
            completed_before = (root / "completion.json").exists()
            if not completed_before:
                _record_launch(root, base, policy, manifest, hardware, attempt)
            segment_root = root / "segments"
            segment_root.mkdir(exist_ok=True)
            restored = None
            if completed_before:
                stage = "validate_completion"
                result = launch_worker(policy, run, "train", run["config"]["optim"]["total_steps"],
                                       attempt_root / "train.json", _lock_fd(handle))
                if result.get("final_step") != run["config"]["optim"]["total_steps"]:
                    raise RuntimeError("Frozen trainer rejected the completed coordinate")
            else:
                if index or _checkpoint_exists(run):
                    stage = "restore"
                    restored = launch_worker(policy, run, "restore", end, attempt_root / "restore.json", _lock_fd(handle))
                    current = restored.get("restored_step")
                    if type(current) is not int or not start <= current <= end:
                        raise RuntimeError("Restored checkpoint is outside the qualified segment")
                    if index and current == start:
                        previous = _read(segment_root / f"segment-{index - 1:03d}.json")
                        helpers.validate_segment_restart(previous, restored, start)
                stage = "train"
                result = launch_worker(policy, run, "train", end, attempt_root / "train.json", _lock_fd(handle))
                if result.get("final_step") != end or not start <= result.get("start_step", -1) <= end:
                    raise RuntimeError("Frozen training worker changed its qualified update interval")
                audit = result.get("checkpoint_audit", {})
                if (not audit.get("final_state_sha256")
                        or audit.get("final_state_sha256") != audit.get("checkpoint_state_sha256")):
                    raise RuntimeError("Segment lacks an exact live/durable checkpoint audit")
                if restored is not None:
                    if (result.get("start_step") != restored["restored_step"]
                            or audit.get("start_state_sha256") != restored["state_sha256"]
                            or (result["start_step"] < end and audit.get("first_update") != restored["next_update"])):
                        raise RuntimeError("Training differs from its independently audited checkpoint restart")
                elif result.get("start_step") != 0 or result.get("restored_step") is not None:
                    raise RuntimeError("Fresh coordinate unexpectedly resumed an unaudited checkpoint")
                common.write_json(segment_root / f"segment-{index:03d}.json", result)
            if (root / "completion.json").exists():
                stage = "retention"
                helpers.finalize_run(manifest, run_name)
            (root / "production_failure.json").unlink(missing_ok=True)
            stage = "report"
            common.render_report(policy, manifest)
            receipt = {**base, **segment, "format": "key-dimension-taylor-concurrent-segment",
                       "index": index, "hardware": hardware, "status": "complete",
                       "completed_before_attempt": completed_before, "final_step": result["final_step"],
                       "qualification_identity": qualification.get("identity", qualification.get("original_policy_identity")),
                       "completed_utc": datetime.now(UTC).isoformat()}
            common.write_json(attempt_root / "receipt.json", receipt)
            common.write_json(attempt_root / "attempt.json", receipt)
            common.write_json(root / "receipts" / f"segment-{index:03d}-{attempt}.json", receipt)
            if not completed_before:
                common.write_json(segment_root / f"receipt-{index:03d}.json", receipt)
            return result
        except BaseException as error:
            _record_failure(root, attempt_root, base, stage, error, hardware=hardware, segment=segment)
            try:
                common.render_report(policy, manifest)
            except Exception as report_error:
                common.write_json(attempt_root / "report_failure.json", {"error": f"{type(report_error).__name__}: {report_error}"})
            raise


def record_interruption(policy: dict, manifest: dict, run_name: str, index: int, exit_code: int) -> dict:
    """Bounded post-process fallback; preserve any more specific numerical error."""
    run = _run(policy, manifest, run_name)
    root = Path(run["workdir"])
    with common.locked(root / ".concurrent-production.lock"):
        common.assert_legacy_jobs_terminal(policy, manifest, run_name)
        base = _base(policy, manifest, run, index)
        failure_path = root / "production_failure.json"
        if failure_path.exists():
            previous = _read(failure_path)
            if (previous.get("failure_kind") == "numerical"
                    or (previous.get("slurm_job_id") == base["slurm_job_id"]
                        and previous.get("slurm_restart_count", 0) == base["slurm_restart_count"])):
                return previous
        attempt = f"launcher-{base['slurm_job_id'] or 'local'}-{base['slurm_restart_count']}-{uuid.uuid4().hex[:8]}"
        attempt_root = root / "execution_attempts" / attempt
        attempt_root.mkdir(parents=True, exist_ok=False)
        record = _record_failure(root, attempt_root, base, "launcher_exit_or_signal",
                                 RuntimeError(f"Allocation launcher exited with status {exit_code}"), hardware=None, segment=None)
        record["exit_code"] = exit_code
        common.write_json(failure_path, record)
        common.write_json(attempt_root / "failure.json", record)
        common.write_json(attempt_root / "attempt.json", record)
        common.write_json(root / "failure_history" / f"segment-{index:03d}-{attempt}.json", record)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--segment", type=int)
    parser.add_argument("--record-interruption", type=int)
    parser.add_argument("--worker", choices=("hardware",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy, manifest = common.load_policy(args.policy)
    if args.worker:
        if args.output is None:
            parser.error("Hardware worker requires --output")
        common.verify_policy(policy, manifest)
        run = _run(policy, manifest, args.run_name)
        result = common.check_hardware(policy["profiles"][policy["assignments"][run["name"]]])
        common.write_json(args.output, result)
    elif args.segment is None:
        parser.error("Production requires --segment")
    elif args.record_interruption is not None:
        result = record_interruption(policy, manifest, args.run_name, args.segment, args.record_interruption)
    else:
        result = run_segment(policy, manifest, args.run_name, args.segment, policy_path=args.policy)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
