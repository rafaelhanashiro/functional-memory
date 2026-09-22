"""Bounded hardware qualification; never submit or execute production training.

Run this script directly. Scientific imports and GPU subprocesses use the original
immutable training checkout. Kernels need one GPU; full-model pilots need four.
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
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

try:
    from . import common
except ImportError:
    import common

KERNEL_CASE_COUNT = 105
PILOT_STEPS = 32
FINAL_STATUSES = {"passed", "partial", "kernel_failure", "numerical_failure", "memory_failure", "runtime_ineligible"}
MEMORY_MARKERS = ("out of memory", "out_of_memory", "resource_exhausted", "resource exhausted",
                  "oom-kill", "memory allocation failed", "failed to allocate", "std::bad_alloc",
                  "outofresources", "out of resource", "shared memory allocation", "shared memory limit")


class CheckFailure(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def failure_status(error: BaseException | str) -> str:
    message = str(error).lower()
    if isinstance(error, MemoryError) or any(marker in message for marker in MEMORY_MARKERS):
        return "memory_failure"
    if isinstance(error, FloatingPointError):
        return "numerical_failure"
    return "infrastructure_failure"


def _helpers(policy: dict):
    root = Path(policy["training_root"]).resolve()
    sys.path.insert(0, str(root))
    original = importlib.import_module("experiments.key_dimension_taylor.preflight")
    metrics = importlib.import_module("experiments.polynomial_attention.plot")
    for module in (original, metrics):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Scientific helper import escaped the frozen training checkout")
    return SimpleNamespace(validate_resume_audit=original.validate_resume_audit,
                           projected_runtime=original.projected_runtime,
                           RuntimeLimitExceeded=original.RuntimeLimitExceeded,
                           require_passing_tests=original.require_passing_tests,
                           committed_records=metrics.committed_records)


def _identity(policy: dict, profile_name: str) -> dict:
    return {"identity": policy["identity"], "source_commit": policy["source_commit"],
            "orchestrator_commit": policy["orchestrator_commit"], "profile": profile_name,
            "scheduler": policy["profiles"][profile_name], "validation_only": True}


def _cached(path: Path, policy: dict, profile_name: str) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if any(record.get(key) != value for key, value in _identity(policy, profile_name).items()):
        raise RuntimeError(f"Cached evidence belongs to a different frozen policy: {path}")
    return record if record.get("status") in FINAL_STATUSES else None


def _attempt(root: Path, stage: str) -> Path:
    job = os.environ.get("SLURM_JOB_ID", "local")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    result = root / "attempts" / f"{stage}-{job}-{restart}-{uuid.uuid4().hex[:8]}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def _run_child(command: list[str], policy: dict, output: Path, lock_fd: int,
               *, record: Path | None = None) -> dict:
    # The child inherits ownership even if a supervisor exits during preemption.
    # JAX is initialized only here, never in the long-lived stage supervisor.
    with output.open("w") as log:
        process = subprocess.run(command, cwd=policy["training_root"], env=common.training_env(policy),
                                 stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,), check=False)
    result = json.loads(record.read_text()) if record is not None and record.exists() else {}
    if process.returncode:
        tail = output.read_text(errors="replace")[-12000:]
        status = result.get("status")
        if status not in FINAL_STATUSES - {"passed"} | {"infrastructure_failure"}:
            status = failure_status(tail)
        raise CheckFailure(status, result.get("error") or f"Child exit {process.returncode}; {tail}")
    if record is not None and not result:
        raise CheckFailure("infrastructure_failure", f"Successful child omitted audit: {record}")
    return result


def _hardware(policy_path: Path, policy: dict, profile_name: str, stage: str,
              attempt: Path, lock_fd: int) -> dict:
    output = attempt / "hardware.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--policy", str(policy_path),
               "--profile", profile_name, "--stage", stage, "--worker", "hardware", "--output", str(output)]
    return _run_child(command, policy, attempt / "hardware.log", lock_fd, record=output)


def _check_reference(record: dict, policy: dict, profile_name: str, *, run: dict | None = None) -> None:
    hardware = record.get("hardware", {})
    kinds = hardware.get("device_kinds", [])
    if (record.get("source_commit") != policy["source_commit"] or record.get("status") != "passed"
            or not kinds or any(policy["profiles"][profile_name]["gpu_type"].lower() not in kind.lower()
                                for kind in kinds)):
        raise ValueError("Historical qualification has incompatible source, status, or GPU model")
    if run is not None:
        for key in ("name", "qk_head_dim", "attention_degree"):
            if record.get(key) != run[key]:
                raise ValueError(f"Historical qualification coordinate mismatch: {key}")
        _validate_validation(record.get("validation", {}), policy, run)
        _validate_progress(run, policy["profiles"][profile_name], record["initial"], record["resumed"])
    elif record.get("tests_passed") != KERNEL_CASE_COUNT:
        raise ValueError("Historical kernel qualification did not pass all 105 cases")


def _reused_kernel(policy: dict, profile_name: str) -> dict | None:
    reference = policy.get("reused_kernels", {}).get(profile_name)
    if reference is None:
        return None
    original = common.load_reference(reference)
    _check_reference(original, policy, profile_name)
    return {**_identity(policy, profile_name), "status": "passed", "reused": True,
            "reference": reference, "hardware": original["hardware"], "tests_passed": KERNEL_CASE_COUNT,
            "original_identity": original.get("identity"), "completed_utc": datetime.now(UTC).isoformat()}


def _kernel_failure(attempt: Path, error: BaseException) -> str:
    if isinstance(error, CheckFailure):
        return error.status
    junit = attempt / "kernels.xml"
    if junit.exists():
        try:
            tree = ET.parse(junit)
        except ET.ParseError:
            return "infrastructure_failure"
        messages = "\n".join(ET.tostring(item, encoding="unicode") for tag in ("failure", "error")
                             for item in tree.iter(tag))
        if failure_status(messages) == "memory_failure":
            return "memory_failure"
        if list(tree.iter("failure")):
            return "numerical_failure"
    return "infrastructure_failure"


def _kernel_coordinates(junit: Path, policy: dict, manifest: dict) -> dict:
    """A failed coordinate cannot invalidate independently completed tests."""
    cases = list(ET.parse(junit).iter("testcase"))
    by_name = {case.attrib.get("name"): case for case in cases}
    expected, coordinates = set(), {}
    for width in policy["key_dimensions"]:
        for degree in policy["degrees"]:
            run = common.grid_run(policy, manifest, width, degree)
            suffix = f"dk{width:03d}-m{degree:03d}"
            names = [f"test_gpu_all_pairs_fp64_partial_tiles_padding_and_causality[{dtype}-{suffix}]"
                     for dtype in ("fp32", "bf16")]
            names.extend(f"test_gpu_all_pairs_{kind}[{suffix}]" for kind in (
                "negative_score_arithmetic_matches_fp64", "large_finite_kernel_and_values", "full_production_shape_bf16"))
            if degree == 16:
                names.append(f"test_gpu_m16_broad_bf16_scores_and_small_cotangent[dk{width:03d}]")
            expected.update(names)
            coordinates[run["name"]] = names
    if (len(cases) != KERNEL_CASE_COUNT or len(by_name) != KERNEL_CASE_COUNT or set(by_name) != expected):
        raise CheckFailure("infrastructure_failure", "Kernel suite aborted or omitted/duplicated a required case; expected exactly 105 named cases")
    eligible, blocked, passed = [], {}, 0
    failures = {}
    for name, case in by_name.items():
        issues = [child for child in case if child.tag in ("failure", "error", "skipped")]
        if not issues:
            passed += 1
            continue
        message = "\n".join(ET.tostring(item, encoding="unicode") for item in issues)
        status = failure_status(message)
        if status != "memory_failure" and any(child.tag == "failure" for child in issues):
            status = "numerical_failure"
        failures[name] = {"test": name, "status": status, "detail": message[-6000:]}
    for name, names in coordinates.items():
        issues = [failures[test] for test in names if test in failures]
        if issues:
            statuses = {item["status"] for item in issues}
            status = ("memory_failure" if "memory_failure" in statuses else "infrastructure_failure"
                      if "infrastructure_failure" in statuses else "numerical_failure")
            blocked[name] = {"status": status, "failures": issues}
        else:
            eligible.append(name)
    return {"status": "passed" if not blocked else "partial", "tests_passed": passed,
            "tests_completed": len(cases), "eligible_runs": eligible, "blocked_runs": blocked}


def run_kernels(policy_path: Path, policy: dict, manifest: dict, profile_name: str) -> dict:
    common.verify_policy(policy, manifest)
    root = Path(policy["output_root"]) / profile_name
    root.mkdir(parents=True, exist_ok=True)
    path = root / "kernels.json"
    with common.locked(root / ".kernels.lock") as lock:
        previous = _cached(path, policy, profile_name)
        if previous is not None:
            return previous
        reused = _reused_kernel(policy, profile_name)
        if reused is not None:
            common.write_json(path, reused)
            return reused
        attempt = _attempt(root, "kernels")
        hardware = None
        try:
            hardware = _hardware(policy_path, policy, profile_name, "kernels", attempt, lock.fileno())
            junit = attempt / "kernels.xml"
            command = [sys.executable, "-m", "pytest", "-c", "experiments/polynomial_attention/pyproject.toml",
                       "experiments/key_dimension_taylor/tests/test_kernels.py", "-m", "gpu", "-q",
                       f"--junitxml={junit}"]
            child_failure = None
            try:
                _run_child(command, policy, attempt / "kernels.log", lock.fileno())
            except CheckFailure as error:
                child_failure = error
            eligibility = _kernel_coordinates(junit, policy, manifest)
            if eligibility["status"] == "passed" and child_failure is not None:
                raise child_failure
            result = {**_identity(policy, profile_name), **eligibility, "reused": False,
                      "hardware": hardware, "junit_path": str(junit)}
        except Exception as error:
            result = {**_identity(policy, profile_name), "status": _kernel_failure(attempt, error),
                      "hardware": hardware, "error": f"{type(error).__name__}: {error}"}
        result.update(slurm_job_id=os.environ.get("SLURM_JOB_ID"), completed_utc=datetime.now(UTC).isoformat(),
                      attempt_path=str(attempt))
        common.write_json(attempt / "result.json", result)
        common.write_json(path, result)
        print(json.dumps({"stage": "kernels", "profile": profile_name, "status": result["status"]}), flush=True)
        return result


def _validate_progress(run: dict, profile: dict, initial: dict, resumed: dict) -> None:
    if (initial.get("start_step") != 0 or initial.get("final_step") != PILOT_STEPS
            or initial.get("data_start_sequence") != 0 or initial.get("restored_step") is not None
            or initial.get("data_parallel_devices") != profile["gpu_count"]):
        raise RuntimeError("Pilot changed its from-scratch update budget, cursor, or device count")
    if (resumed.get("start_step") != PILOT_STEPS or resumed.get("restored_step") != PILOT_STEPS
            or resumed.get("final_step") != PILOT_STEPS + 1
            or resumed.get("data_start_sequence") != PILOT_STEPS * run["config"]["data"]["batch_size"]
            or resumed.get("data_parallel_devices") != profile["gpu_count"]):
        raise RuntimeError("Restart changed the optimizer progress, data cursor, or device count")
    if any(outcome.get("attention_degree") != run["attention_degree"] for outcome in (initial, resumed)):
        raise RuntimeError("Pilot attention degree differs from its coordinate")


def _validate_validation(validation: dict, policy: dict, run: dict) -> None:
    expected = {"step": PILOT_STEPS, "source_commit": policy["source_commit"],
                "qk_head_dim": run["qk_head_dim"], "attention_degree": run["attention_degree"],
                "parameter_count": run["parameter_count"]}
    if any(validation.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Pilot validation identity or parameter count differs from its coordinate")
    if not all(math.isfinite(validation.get(key, float("nan"))) for key in ("validation_nll", "validation_perplexity")):
        raise CheckFailure("numerical_failure", "Pilot validation has nonfinite loss or perplexity")


def _launch_pilot(policy_path: Path, policy: dict, profile_name: str, run: dict, workdir: Path,
                  attempt: Path, stop: int, lock_fd: int) -> dict:
    output = attempt / f"{run['name']}-{stop}.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--policy", str(policy_path),
               "--profile", profile_name, "--stage", "pilots", "--worker", "pilot",
               "--width", str(run["qk_head_dim"]), "--degree", str(run["attention_degree"]),
               "--workdir", str(workdir), "--stop", str(stop), "--output", str(output)]
    return _run_child(command, policy, output.with_suffix(".log"), lock_fd, record=output)


def _shard_runs(policy: dict, manifest: dict, profile_name: str, shard: int) -> list[dict]:
    shards = policy["shards"][profile_name]
    if not 0 <= shard < len(shards):
        raise ValueError(f"Unknown {profile_name} pilot shard: {shard}")
    resolved = [common.grid_run(policy, manifest, width, degree)
                for width in policy["key_dimensions"] for degree in policy["degrees"]]
    runs = {run["name"]: run for run in resolved}
    return [runs[name] for name in shards[shard]]


def _cleanup_checkpoints(workdir: Path) -> None:
    # Workdir is a freshly created isolated attempt, never a production run.
    checkpoints = workdir / "checkpoints"
    if checkpoints.is_symlink():
        raise RuntimeError("Refusing to clean a checkpoint symlink outside the diagnostic workdir")
    if checkpoints.exists():
        shutil.rmtree(checkpoints)


def run_pilots(policy_path: Path, policy: dict, manifest: dict, profile_name: str, shard: int = 0) -> dict:
    common.verify_policy(policy, manifest)
    root = Path(policy["output_root"]) / profile_name
    root.mkdir(parents=True, exist_ok=True)
    helpers = _helpers(policy)
    runs = _shard_runs(policy, manifest, profile_name, shard)
    with common.locked(root / f".pilots-shard{shard}.lock") as lock:
        kernels = _cached(root / "kernels.json", policy, profile_name)
        if kernels is None:
            kernels = _reused_kernel(policy, profile_name)
            if kernels is not None:
                common.write_json(root / "kernels.json", kernels)
        if kernels is None or kernels["status"] not in ("passed", "partial"):
            result = {**_identity(policy, profile_name), "status": "blocked", "shard": shard,
                      "reason": "Full-grid kernel correctness has not passed on this GPU model",
                      "kernels": kernels, "completed_utc": datetime.now(UTC).isoformat()}
            common.write_json(root / f"pilots-shard{shard}.json", result)
            return result
        attempt = _attempt(root, f"pilots-shard{shard}")
        try:
            hardware = _hardware(policy_path, policy, profile_name, "pilots", attempt, lock.fileno())
        except Exception as error:
            result = {**_identity(policy, profile_name), "status": failure_status(error), "shard": shard,
                      "stage": "hardware", "error": f"{type(error).__name__}: {error}",
                      "completed_utc": datetime.now(UTC).isoformat()}
            common.write_json(attempt / "result.json", result)
            common.write_json(root / f"pilots-shard{shard}.json", result)
            return result
        outcomes = {}
        for run in runs:
            path = root / "checks" / f"{run['name']}.json"
            previous = _cached(path, policy, profile_name)
            if previous is not None:
                if any(previous.get(key) != run[key] for key in ("name", "qk_head_dim", "attention_degree", "parameter_count")):
                    raise RuntimeError("Cached pilot coordinate changed")
                outcomes[run["name"]] = previous["status"]
                continue
            coordinate = {**_identity(policy, profile_name), **{key: run[key] for key in
                           ("name", "qk_head_dim", "attention_degree", "parameter_count")},
                          "hardware": hardware, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "shard": shard}
            if run["name"] in kernels.get("blocked_runs", {}):
                record = {**coordinate, "status": "kernel_failure", "kernel_failure": kernels["blocked_runs"][run["name"]],
                          "completed_utc": datetime.now(UTC).isoformat()}
                common.write_json(path, record)
                outcomes[run["name"]] = record["status"]
                continue
            if kernels["status"] == "partial" and run["name"] not in kernels.get("eligible_runs", []):
                raise RuntimeError("Partial kernel result omitted coordinate eligibility")
            workdir = attempt / run["name"]
            try:
                reference = policy.get("reused_checks", {}).get(profile_name, {}).get(run["name"])
                if reference is not None:
                    original = common.load_reference(reference)
                    _check_reference(original, policy, profile_name, run=run)
                    helpers.validate_resume_audit(original["initial"], original["resumed"])
                    record = {**coordinate, "status": "passed", "reused": True, "reference": reference,
                              "original_identity": original.get("identity"), "hardware": original["hardware"],
                              "runtime": original["runtime"]}
                else:
                    initial = _launch_pilot(policy_path, policy, profile_name, run, workdir, attempt, PILOT_STEPS, lock.fileno())
                    validation = next((row for row in helpers.committed_records(workdir) if row["step"] == PILOT_STEPS), {})
                    _validate_validation(validation, policy, run)
                    resumed = _launch_pilot(policy_path, policy, profile_name, run, workdir, attempt, PILOT_STEPS + 1, lock.fileno())
                    _validate_progress(run, policy["profiles"][profile_name], initial, resumed)
                    helpers.validate_resume_audit(initial, resumed)
                    runtime = helpers.projected_runtime(run, initial, resumed, policy["profiles"][profile_name])
                    record = {**coordinate, "status": "passed", "reused": False, "initial": initial,
                              "resumed": resumed, "validation": validation, "runtime": runtime}
            except Exception as error:
                status = (error.status if isinstance(error, CheckFailure) else "runtime_ineligible"
                          if isinstance(error, helpers.RuntimeLimitExceeded) else failure_status(error))
                record = {**coordinate, "status": status, "error": f"{type(error).__name__}: {error}",
                          "numerical_failures": [json.loads(item.read_text())
                                                 for item in sorted(workdir.glob("numerical_failure_step_*.json"))]}
            finally:
                _cleanup_checkpoints(workdir)
            record.update(completed_utc=datetime.now(UTC).isoformat(), attempt_path=str(attempt))
            common.write_json(attempt / f"{run['name']}-result.json", record)
            common.write_json(path, record)
            outcomes[run["name"]] = record["status"]
            print(json.dumps({"profile": profile_name, "run": run["name"], "status": record["status"]}), flush=True)
        result = {**_identity(policy, profile_name), "status": "completed", "shard": shard,
                  "outcomes": outcomes, "hardware": hardware, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "completed_utc": datetime.now(UTC).isoformat()}
        common.write_json(attempt / "result.json", result)
        common.write_json(root / f"pilots-shard{shard}.json", result)
        return result


def _pilot_worker(policy: dict, manifest: dict, profile_name: str, width: int, degree: int,
                  workdir: Path, stop: int) -> dict:
    if stop not in (PILOT_STEPS, PILOT_STEPS + 1):
        raise ValueError("Hardware checks are limited to 32 updates and one restored update")
    isolated_root = Path(policy["output_root"]).resolve() / profile_name / "attempts"
    if not workdir.resolve().is_relative_to(isolated_root) or workdir.is_symlink():
        raise ValueError("Pilot workdir must be an isolated diagnostic attempt")
    common.check_hardware(policy["profiles"][profile_name], count=4)
    run = common.grid_run(policy, manifest, width, degree)
    root = Path(policy["training_root"]).resolve()
    sys.path.insert(0, str(root))
    configs = importlib.import_module("experiments.polynomial_attention.configs")
    loop = importlib.import_module("experiments.polynomial_attention.training.loop")
    if any(not Path(module.__file__).resolve().is_relative_to(root) for module in (configs, loop)):
        raise RuntimeError("Pilot scientific imports escaped the frozen training checkout")
    config = configs.load_config(run["config_path"])
    config.train.workdir, config.train.wandb_mode = workdir, "disabled"
    config.train.eval_every, config.train.eval_batches = PILOT_STEPS, 1
    if config.optim.total_steps != 19074:
        raise ValueError("Pilot must preserve the full production learning-rate schedule")
    return loop.run_training(config, resume=True, stop_after_step=stop, audit_checkpoint=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("b300", "a100", "a6000"))
    parser.add_argument("--stage", choices=("kernels", "pilots"), default="pilots")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--worker", choices=("hardware", "pilot"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--stop", type=int, choices=(PILOT_STEPS, PILOT_STEPS + 1), default=PILOT_STEPS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy, manifest = common.load_policy(args.policy)
    common.verify_policy(policy, manifest)
    if args.worker:
        if args.output is None:
            parser.error("Workers require --output")
        try:
            if args.worker == "hardware":
                profile = policy["profiles"][args.profile]
                count = profile.get("kernel_gpu_count", 1) if args.stage == "kernels" else 4
                result = common.check_hardware(profile, count=count)
            else:
                if args.width is None or args.degree is None or args.workdir is None:
                    parser.error("Pilot workers require --width, --degree, and --workdir")
                result = _pilot_worker(policy, manifest, args.profile, args.width, args.degree, args.workdir, args.stop)
            common.write_json(args.output, result)
        except Exception as error:
            common.write_json(args.output, {"status": failure_status(error), "error": f"{type(error).__name__}: {error}"})
            raise
    elif args.stage == "kernels":
        result = run_kernels(args.policy, policy, manifest, args.profile)
        if result["status"] not in ("passed", "partial"):
            raise SystemExit(1)
    else:
        run_pilots(args.policy, policy, manifest, args.profile, args.shard)


if __name__ == "__main__":
    main()
