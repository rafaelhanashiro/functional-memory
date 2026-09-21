"""Qualify a new allocation using the immutable training checkout and exact audits.

Run this file directly. The orchestration checkout supplies only policy handling;
all kernels, model code, training workers, and audit helpers come from training_root.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

try:
    from . import common
except ImportError:  # Direct script entry keeps orchestration outside experiments imports.
    import common

KERNEL_CASE_COUNT = 105
PILOT_STEPS = 32


class NumericalQualificationFailure(RuntimeError):
    """The assigned coordinate failed numerically on this hardware profile."""


def _load_original_helpers(policy: dict):
    training_root = Path(policy["training_root"]).resolve()
    sys.path.insert(0, str(training_root))
    original = importlib.import_module("experiments.key_dimension_taylor.preflight")
    if not Path(original.__file__).resolve().is_relative_to(training_root):
        raise RuntimeError("Qualification imported training helpers outside the immutable checkout")
    metrics = importlib.import_module("experiments.polynomial_attention.plot")
    if not Path(metrics.__file__).resolve().is_relative_to(training_root):
        raise RuntimeError("Qualification imported metric helpers outside the immutable checkout")
    return SimpleNamespace(validate_resume_audit=original.validate_resume_audit,
                           projected_runtime=original.projected_runtime,
                           RuntimeLimitExceeded=original.RuntimeLimitExceeded,
                           require_passing_tests=original.require_passing_tests,
                           committed_records=metrics.committed_records)


def _identity(policy: dict, profile_name: str) -> dict:
    return {"identity": policy["identity"], "policy_identity": policy["identity"],
            "source_commit": policy["source_commit"], "orchestrator_commit": policy["orchestrator_commit"],
            "profile": profile_name, "scheduler": policy["profiles"][profile_name]}


def _matches(record: dict, policy: dict, profile_name: str, statuses: tuple[str, ...]) -> bool:
    return record.get("status") in statuses and all(record.get(key) == value
        for key, value in _identity(policy, profile_name).items())


def _read_matching(path: Path, policy: dict, profile_name: str, statuses: tuple[str, ...]) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if not _matches(record, policy, profile_name, statuses):
        raise RuntimeError(f"Qualification evidence has a different policy or invalid status: {path}")
    return record


def assigned_runs(policy: dict, manifest: dict, profile_name: str) -> list[dict]:
    if profile_name not in policy["profiles"]:
        raise ValueError(f"Unknown execution profile: {profile_name}")
    assigned = {name for name, profile in policy["assignments"].items() if profile == profile_name}
    runs = [run for run in manifest["runs"] if run["name"] in assigned and not run.get("retained_completed", False)]
    if not runs or {run["name"] for run in runs} != assigned:
        raise ValueError("Hardware qualification requires known active assigned coordinates")
    return runs


def _hardware_worker(policy_path: Path, policy: dict, manifest: dict, profile_name: str, output: Path) -> dict:
    # Exit the JAX process before pytest/pilots start, releasing all GPU contexts.
    command = [sys.executable, str(Path(__file__).resolve()), "--policy", str(policy_path),
               "--profile", profile_name, "--worker", "hardware", "--output", str(output)]
    subprocess.run(command, cwd=policy["orchestrator_root"], env=common.training_env(policy), check=True)
    return json.loads(output.read_text())


def _launch_pilot(policy: dict, manifest: dict, run: dict, workdir: Path, output: Path, stop: int) -> dict:
    command = [sys.executable, "-m", "experiments.key_dimension_taylor.preflight",
               str(policy["training_manifest"]), "--worker", "pilot", "--run-name", run["name"],
               "--workdir", str(workdir), "--output", str(output), "--stop-after-step", str(stop)]
    process = subprocess.run(command, cwd=policy["training_root"], env=common.training_env(policy), check=False)
    if process.returncode:
        failure = json.loads(output.read_text()) if output.exists() else {}
        if failure.get("numerical_failure") is True:
            raise NumericalQualificationFailure(failure.get("error", "Numerical pilot failure"))
        raise subprocess.CalledProcessError(process.returncode, command)
    return json.loads(output.read_text())


def _validate_progress(run: dict, profile: dict, initial: dict, resumed: dict) -> None:
    if (initial.get("start_step") != 0 or initial.get("final_step") != PILOT_STEPS
            or initial.get("data_start_sequence") != 0
            or initial.get("restored_step") is not None
            or initial.get("data_parallel_devices") != profile["gpu_count"]):
        raise RuntimeError("Qualification pilot changed its from-scratch budget, cursor, or allocation")
    if (resumed.get("start_step") != PILOT_STEPS or resumed.get("restored_step") != PILOT_STEPS
            or resumed.get("final_step") != PILOT_STEPS + 1
            or resumed.get("data_start_sequence") != PILOT_STEPS * run["config"]["data"]["batch_size"]
            or resumed.get("data_parallel_devices") != profile["gpu_count"]):
        raise RuntimeError("Qualification restart changed its optimizer progress, data cursor, or allocation")
    for outcome in (initial, resumed):
        if outcome.get("attention_degree") != run["attention_degree"]:
            raise RuntimeError("Qualification pilot degree differs from its assigned coordinate")


def run_qualification(policy_path: Path, policy: dict, manifest: dict, profile_name: str) -> dict:
    common.verify_policy(policy, manifest)
    runs = assigned_runs(policy, manifest, profile_name)
    helpers = _load_original_helpers(policy)
    profile = policy["profiles"][profile_name]
    root = Path(policy["qualification_root"]) / profile_name
    attempt = root / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    shared = _identity(policy, profile_name)
    stage = "hardware"
    hardware = None
    try:
        hardware = _hardware_worker(policy_path, policy, manifest, profile_name, attempt / "hardware.json")
        stage = "kernels"
        kernels_path = root / "kernels.json"
        if _read_matching(kernels_path, policy, profile_name, ("passed",)) is None:
            junit = attempt / "kernels.xml"
            subprocess.run([sys.executable, "-m", "pytest", "-c", "experiments/polynomial_attention/pyproject.toml",
                            "experiments/key_dimension_taylor/tests/test_kernels.py", "-m", "gpu", "-q",
                            f"--junitxml={junit}"], cwd=policy["training_root"],
                           env=common.training_env(policy), check=True)
            passed = helpers.require_passing_tests(junit, KERNEL_CASE_COUNT)
            if passed != KERNEL_CASE_COUNT:
                raise RuntimeError(f"Immutable kernel suite changed: expected {KERNEL_CASE_COUNT}, got {passed}")
            common.write_json(kernels_path, {**shared, "status": "passed", "tests_passed": passed,
                              "hardware": hardware, "junit_path": str(junit), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                              "completed_utc": datetime.now(UTC).isoformat()})
        outcomes = {}
        for run in runs:
            stage = run["name"]
            path = root / "checks" / f"{stage}.json"
            previous = _read_matching(path, policy, profile_name, ("passed", "ineligible"))
            if previous is not None:
                if previous.get("qk_head_dim") != run["qk_head_dim"] or previous.get("attention_degree") != run["attention_degree"]:
                    raise RuntimeError("Cached qualification coordinate identity changed")
                outcomes[stage] = previous["status"]
                continue
            workdir = attempt / stage
            coordinate = {**shared, "name": stage, "qk_head_dim": run["qk_head_dim"],
                          "attention_degree": run["attention_degree"], "hardware": hardware,
                          "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
            try:
                initial = _launch_pilot(policy, manifest, run, workdir, attempt / f"{stage}-initial.json", PILOT_STEPS)
                validation = next((row for row in helpers.committed_records(workdir) if row["step"] == PILOT_STEPS), None)
                if validation is None:
                    raise RuntimeError("Qualification pilot lacks its committed step-32 validation")
                if (validation.get("source_commit") != policy["source_commit"]
                        or validation.get("qk_head_dim") != run["qk_head_dim"]
                        or validation.get("attention_degree") != run["attention_degree"]
                        or validation.get("parameter_count") != run["parameter_count"]):
                    raise RuntimeError("Qualification validation identity differs from its assigned coordinate")
                if not all(math.isfinite(validation[key]) for key in ("validation_nll", "validation_perplexity")):
                    raise NumericalQualificationFailure("Qualification validation is nonfinite")
                resumed = _launch_pilot(policy, manifest, run, workdir, attempt / f"{stage}-resume.json", PILOT_STEPS + 1)
                _validate_progress(run, profile, initial, resumed)
                helpers.validate_resume_audit(initial, resumed)
                runtime = helpers.projected_runtime(run, initial, resumed, profile)
                record = {**coordinate, "status": "passed", "initial": initial, "resumed": resumed,
                          "runtime": runtime, "validation": validation,
                          "completed_utc": datetime.now(UTC).isoformat()}
                common.write_json(path, record)
                outcomes[stage] = "passed"
                if (workdir / "checkpoints").exists():
                    shutil.rmtree(workdir / "checkpoints")
            except (NumericalQualificationFailure, helpers.RuntimeLimitExceeded) as error:
                failures = [{"pilot": workdir.name, "record": json.loads(item.read_text())}
                            for item in sorted(workdir.glob("numerical_failure_step_*.json"))]
                numerical = isinstance(error, NumericalQualificationFailure)
                record = {**coordinate, "status": "ineligible", "reason": "numerical" if numerical else "runtime",
                          "failure_kind": "numerical" if numerical else "runtime", "error": f"{type(error).__name__}: {error}",
                          "numerical_failures": failures, "completed_utc": datetime.now(UTC).isoformat()}
                common.write_json(path, record)
                outcomes[stage] = "ineligible"
                if (workdir / "checkpoints").exists():
                    shutil.rmtree(workdir / "checkpoints")
        result = {**shared, "status": "passed", "hardware": hardware, "outcomes": outcomes,
                  "eligible_runs": sorted(name for name, status in outcomes.items() if status == "passed"),
                  "ineligible_runs": sorted(name for name, status in outcomes.items() if status == "ineligible"),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "completed_utc": datetime.now(UTC).isoformat()}
        common.write_json(root / "success.json", result)
        (root / "failure.json").unlink(missing_ok=True)
        print(json.dumps({"qualification": str(root), "profile": profile_name, "eligible_runs": result["eligible_runs"],
                          "ineligible_runs": result["ineligible_runs"]}, sort_keys=True), flush=True)
        return result
    except BaseException as error:
        failure = {**shared, "status": "failed", "stage": stage, "failure_kind": "infrastructure",
                   "hardware": hardware, "error": f"{type(error).__name__}: {error}",
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "failed_utc": datetime.now(UTC).isoformat()}
        common.write_json(attempt / "failure.json", failure)
        common.write_json(root / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--profile", default="b300")
    parser.add_argument("--worker", choices=("hardware",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy, manifest = common.load_policy(args.policy)
    common.verify_policy(policy, manifest)
    if args.worker:
        if args.output is None:
            parser.error("Hardware worker requires --output")
        common.write_json(args.output, common.check_hardware(policy["profiles"][args.profile]))
    else:
        run_qualification(args.policy, policy, manifest, args.profile)


if __name__ == "__main__":
    main()
