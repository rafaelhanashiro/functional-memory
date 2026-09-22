"""Per-variant GPU eligibility with shared correctness and immutable-source gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.studies.polynomial_attention.preflight import passed_check, record_success, require_passing_tests
from scaffolds.transformer.studies.polynomial_attention.sweep import write_json
from scaffolds.transformer.studies.odd_taylor_attention.sweep import (
    CONTROL_NAMES, VARIANTS, STABILITY_STEPS, identity_payload, manifest_path, parameter_count,
    scientific_recipe, scheduler_profile, sha256, validate_control,
)

SHARED_CHECKS = ("attention-cpu", "attention-distributed-cpu", "attention-fused-gpu", "initialization")


class NumericalPilotFailure(RuntimeError):
    """A correctly implemented variant became numerically invalid in its pilot."""


class RuntimeLimitExceeded(RuntimeError):
    """The variant cannot complete within the selected allocation walltime."""


def verify_manifest(manifest: dict) -> None:
    actual = hashlib.sha256(json.dumps(identity_payload(manifest), sort_keys=True).encode()).hexdigest()
    if actual != manifest["identity"]:
        raise ValueError("Sweep manifest identity does not match its contents")
    if manifest["scheduler"] != scheduler_profile():
        raise ValueError("Odd Taylor sweep requires the four-H100 dogo allocation")
    variants = [run["attention_degree"] for run in manifest["runs"]]
    if variants != list(VARIANTS) or manifest["stability_steps"] != STABILITY_STEPS:
        raise ValueError("Preflight requires degrees one and three in canonical order")
    if [run["name"] for run in manifest["controls"]] != list(CONTROL_NAMES):
        raise ValueError("Preflight requires both historical controls")
    execution = Path(manifest["execution_root"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution, text=True).strip()
    if commit != manifest["source_commit"]:
        raise ValueError("Frozen execution source commit changed")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution, text=True):
        raise ValueError("Frozen execution checkout is no longer clean")
    if sha256(execution / manifest.get("dependency_lock_path", "scaffolds/transformer/uv.lock")) != manifest["dependency_lock_sha256"]:
        raise ValueError("Dependency lock changed")
    for data in manifest["data_identity"].values():
        if sha256(Path(data["path"])) != data["sha256"]:
            raise ValueError(f"Data manifest changed: {data['path']}")
    for control in manifest["controls"]:
        validate_control(control)
    recipe = scientific_recipe(manifest["controls"][0]["config"])
    for run in manifest["runs"]:
        config = load_config(run["config_path"])
        if not config_dicts_equal(config_to_dict(config), run["config"]) or scientific_recipe(run["config"]) != recipe:
            raise ValueError(f"Run configuration changed: {run['name']}")
        if (config.model.quadratic_mode, config.model.quadratic_sharing) != (run["quadratic_mode"], run["quadratic_sharing"]):
            raise ValueError(f"Run family/sharing does not match configuration: {run['name']}")
        if (config.model.attention_degree != run["attention_degree"] or config.model.attention_backend != "fused"
                or config.model.quadratic_mode != "fixed"):
            raise ValueError("Odd Taylor production requires fused fixed odd-degree attention")
        if config.train.source_commit != manifest["source_commit"] or parameter_count(config.model) != run["parameter_count"]:
            raise ValueError(f"Run source or parameter count changed: {run['name']}")
        expected_root = Path(manifest["project_root"]) / "checkpoints/odd_taylor_attention" / run["name"]
        link = Path(run["workdir"]) / "checkpoints"
        if Path(run["checkpoint_dir"]) != expected_root or not link.is_symlink() or link.resolve() != expected_root.resolve():
            raise ValueError(f"Run checkpoint storage is not canonical: {run['name']}")


def eligible_runs(manifest: dict) -> set[str]:
    root = Path(manifest["preflight_dir"])
    if not passed_check(root / "success.json", manifest["identity"]):
        return set()
    if not all(passed_check(root / "checks" / f"{name}.json", manifest["identity"]) for name in SHARED_CHECKS):
        return set()
    return {run["name"] for run in manifest["runs"]
            if passed_check(root / "checks" / f"{run['name']}.json", manifest["identity"])}


def preflight_complete(manifest: dict) -> bool:
    root = Path(manifest["preflight_dir"])
    if not passed_check(root / "success.json", manifest["identity"]):
        return False
    if not all(passed_check(root / "checks" / f"{name}.json", manifest["identity"]) for name in SHARED_CHECKS):
        return False
    # A completed preflight may contain numerical/runtime rejections. Both
    # outcomes must exist, while correctness/infrastructure failures never pass.
    for run in manifest["runs"]:
        path = root / "checks" / f"{run['name']}.json"
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        if record.get("identity") != manifest["identity"] or record.get("status") not in ("passed", "ineligible"):
            return False
    return True


def launch_worker(manifest: dict, worker: str, output: Path, *, name: str = "",
                  workdir: Path | None = None, stop: int = 2, cpu: bool = False) -> dict:
    command = [sys.executable, "-m", "scaffolds.transformer.studies.odd_taylor_attention.preflight",
               str(manifest_path(Path(manifest["project_root"]))), "--worker", worker,
               "--output", str(output), "--run-name", name, "--stop-after-step", str(stop)]
    if workdir is not None:
        command += ["--workdir", str(workdir)]
    result = subprocess.run(command, cwd=manifest["execution_root"],
                            env=dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda"), check=False)
    if result.returncode:
        failure = json.loads(output.read_text()) if output.exists() else {}
        if worker == "pilot" and failure.get("numerical_failure") is True:
            raise NumericalPilotFailure(failure["error"])
        raise subprocess.CalledProcessError(result.returncode, command)
    return json.loads(output.read_text())


def walltime_seconds(value: str) -> int:
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = map(int, clock.split(":"))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def projected_runtime(run: dict, outcome: dict, scheduler: dict) -> dict:
    timing, config = outcome["timing"], run["config"]
    steady = timing["train_step_seconds"][1:]
    evaluation, checkpoints = timing["evaluation_seconds"], timing["checkpoint_seconds"]
    values = [timing["runtime_seconds"], *steady, *evaluation, *checkpoints]
    if not steady or not evaluation or not checkpoints or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError(f"Pilot has no valid steady training/evaluation/checkpoint timings: {timing}")
    steps = config["optim"]["total_steps"]
    eval_steps = set(range(config["train"]["eval_every"], steps + 1, config["train"]["eval_every"])) | {steps}
    ckpt_steps = set(range(config["train"]["ckpt_every"], steps + 1, config["train"]["ckpt_every"])) | eval_steps
    estimate = {
        "startup_seconds": 3 * timing["runtime_seconds"], "steady_train_step_seconds": max(steady),
        "training_seconds": max(steady) * steps,
        "evaluation_seconds": max(evaluation) * config["train"]["eval_batches"] * len(eval_steps),
        "checkpoint_seconds": max(checkpoints) * len(ckpt_steps),
        "limit_seconds": walltime_seconds(scheduler["production_time"]),
    }
    estimate["projected_total_seconds"] = sum(estimate[key] for key in
        ("startup_seconds", "training_seconds", "evaluation_seconds", "checkpoint_seconds"))
    # Conservatively allow startup for the stability, restoration, and continuation processes.
    # Leave ten percent headroom for shared-filesystem and allocation variation.
    estimate["required_seconds_with_margin"] = estimate["projected_total_seconds"] * 1.1
    if estimate["required_seconds_with_margin"] >= estimate["limit_seconds"]:
        raise RuntimeLimitExceeded(f"{run['name']} projects beyond production walltime with 10% margin: {estimate}")
    return estimate


def validate_initializations(initializations: dict, manifest: dict) -> None:
    reference = initializations["fixed"]
    if reference["coefficient_count"] != 0:
        raise RuntimeError("Fixed reference unexpectedly has learned coefficients")
    for run in manifest["runs"]:
        result = initializations[run["name"]]
        if (result["backbone_sha256"] != reference["backbone_sha256"] or
                result["backbone_parameter_count"] != reference["backbone_parameter_count"] or
                result["coefficient_count"] != 0 or result["coefficient_shape"] is not None or
                result["parameter_count"] != run["parameter_count"]):
            raise RuntimeError(f"Backbone initialization or parameter count changed: {run['name']}")


def validate_resume_audit(initial: dict, resumed: dict) -> None:
    """Require exact checkpoint state, next input, and RNG across the restart.

    Independent GPU trajectories need not be bitwise reproducible. CPU tests
    separately compare uninterrupted and resumed trajectories for all variants.
    This gate isolates checkpoint fidelity without imposing float tolerances.
    """
    for outcome in (initial, resumed):
        audit = outcome.get("checkpoint_audit")
        if not isinstance(audit, dict):
            raise RuntimeError("Pilot is missing its checkpoint audit")
        hashes = [audit.get(key) for key in
                  ("start_state_sha256", "final_state_sha256", "checkpoint_state_sha256")]
        for key, step in (("first_update", outcome["start_step"] + 1),
                          ("next_update", outcome["final_step"] + 1)):
            update = audit.get(key)
            if not isinstance(update, dict) or update.get("step") != step:
                raise RuntimeError(f"Checkpoint audit has an invalid {key} step")
            hashes.extend(update.get(key) for key in ("batch_sha256", "rng_sha256"))
        if any(not isinstance(value, str) or len(value) != 64 or
               any(char not in "0123456789abcdef" for char in value) for value in hashes):
            raise RuntimeError("Checkpoint audit contains an invalid SHA256")
        if audit["final_state_sha256"] != audit["checkpoint_state_sha256"]:
            raise RuntimeError("Durable checkpoint differs from the live training state")
    before, after = initial["checkpoint_audit"], resumed["checkpoint_audit"]
    if before["final_state_sha256"] != after["start_state_sha256"]:
        raise RuntimeError("Restarted state differs from the saved live training state")
    if before["next_update"] != after["first_update"]:
        raise RuntimeError("Restarted training changed the next batch or RNG")


def retain_preflight_evidence(manifest: dict) -> None:
    """Keep small reproducibility evidence after temporary pilot caches are gone."""
    source = Path(manifest["preflight_dir"])
    destination = Path(manifest["project_root"]) / "results/odd_taylor_attention/preflight" / manifest["identity"]
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*SHARED_CHECKS, *[run["name"] for run in manifest["runs"]]):
        record = json.loads((source / "checks" / f"{name}.json").read_text())
        if record.get("junit_path"):
            xml_destination = destination / f"{name}.xml"
            shutil.copyfile(record["junit_path"], xml_destination)
            record["junit_path"] = str(xml_destination)
        write_json(destination / "checks" / f"{name}.json", record)
    write_json(destination / "success.json", json.loads((source / "success.json").read_text()))
    write_json(destination / "provenance.json", {
        "identity": manifest["identity"], "source_commit": manifest["source_commit"],
        "execution_root": manifest["execution_root"], "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "data_identity": manifest["data_identity"], "scheduler": manifest["scheduler"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "log_path": manifest.get("preflight_log_path"),
        "eligible_runs": sorted(eligible_runs(manifest)),
    })


def run_preflight(manifest: dict, *, check_only: bool = False, check_hardware: bool = False,
                  run_name: str | None = None) -> None:
    verify_manifest(manifest)
    root = Path(manifest["preflight_dir"])
    if check_only and not check_hardware:
        if not preflight_complete(manifest):
            raise RuntimeError("This sweep has no successful matching complete preflight")
        if run_name is not None and run_name not in eligible_runs(manifest):
            raise RuntimeError(f"Run did not pass its own pilot: {run_name}")
        return
    attempt = root / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True)
    stage = "hardware"
    try:
        hardware = launch_worker(manifest, "hardware", attempt / "hardware.json")
        if preflight_complete(manifest):
            if run_name is not None and run_name not in eligible_runs(manifest):
                raise RuntimeError(f"Run did not pass its own pilot: {run_name}")
            if not check_only:
                retain_preflight_evidence(manifest)
            return
        if check_only:
            raise RuntimeError("This sweep has no successful matching complete preflight")
        for stage, test_files, cpu, selection, minimum_tests in (
            ("attention-cpu", ("test_odd_attention.py", "test_odd_training.py"), True,
             ("-m", "not gpu", "-k", "not distributed"), 4),
            ("attention-distributed-cpu", ("test_odd_training.py",), True,
             ("-m", "not gpu", "-k", "distributed"), 2),
            ("attention-fused-gpu", ("test_odd_attention.py",), False, ("-m", "gpu"), 4),
        ):
            check_path = root / "checks" / f"{stage}.json"
            if passed_check(check_path, manifest["identity"]):
                continue
            junit = attempt / f"{stage}.xml"
            test_paths = [f"scaffolds/transformer/tests/{test_file}" for test_file in test_files]
            environment = dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda")
            if stage == "attention-distributed-cpu":
                environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
            subprocess.run([sys.executable, "-m", "pytest", "-c", "scaffolds/transformer/pyproject.toml",
                            *test_paths, "-q", *selection, f"--junitxml={junit}"],
                           cwd=manifest["execution_root"], env=environment, check=True)
            record_success(check_path, manifest, tests_passed=require_passing_tests(junit, minimum_tests), junit_path=str(junit))
        stage = "initialization"
        path = root / "checks/initialization.json"
        if not passed_check(path, manifest["identity"]):
            initializations = {name: launch_worker(manifest, "initialization", attempt / f"init-{name}.json", name=name, cpu=True)
                               for name in ["fixed", *[run["name"] for run in manifest["runs"]]]}
            validate_initializations(initializations, manifest)
            record_success(path, manifest, initializations=initializations)
        for run in manifest["runs"]:
            stage = run["name"]
            path = root / "checks" / f"{stage}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get("identity") == manifest["identity"] and previous.get("status") in ("passed", "ineligible"):
                    continue
            workdir = attempt / stage
            try:
                initial = launch_worker(manifest, "pilot", attempt / f"{stage}-step2.json", name=stage, workdir=workdir)
                if initial["final_step"] != 2 or initial["data_parallel_devices"] != manifest["scheduler"]["gpu_count"]:
                    raise RuntimeError(f"Pilot did not complete two updates on the required devices: {initial}")
                runtime = projected_runtime(run, initial, manifest["scheduler"])
                from scaffolds.transformer.studies.polynomial_attention.plot import committed_records
                validation = next((row for row in committed_records(workdir) if row["step"] == 2), None)
                if validation is None:
                    raise RuntimeError("Pilot is missing committed step-two validation")
                if not all(math.isfinite(validation[key]) for key in ("validation_nll", "validation_perplexity")):
                    raise NumericalPilotFailure("Pilot validation is nonfinite")
                resumed = launch_worker(manifest, "pilot", attempt / f"{stage}-resume.json", name=stage, workdir=workdir, stop=3)
                if (resumed["restored_step"] != 2 or resumed["final_step"] != 3 or
                        resumed["data_start_sequence"] != 2 * run["config"]["data"]["batch_size"] or
                        resumed["data_parallel_devices"] != manifest["scheduler"]["gpu_count"]):
                    raise RuntimeError(f"Resume lost optimizer/data progress: {resumed}")
                validate_resume_audit(initial, resumed)
                record_success(path, manifest, outcome=initial, resumed=resumed, runtime=runtime,
                               resume_validation="exact_live_durable_restart_state_and_next_batch_rng",
                               attention_degree=run["attention_degree"],
                               validation_nll=validation["validation_nll"], validation_perplexity=validation["validation_perplexity"])
                checkpoints = workdir / "checkpoints"
                if checkpoints.exists():
                    shutil.rmtree(checkpoints)
            except (NumericalPilotFailure, RuntimeLimitExceeded) as error:
                details = []
                for failure_path in sorted(workdir.glob("numerical_failure_step_*.json")):
                    details.append({"pilot": workdir.name, "record": json.loads(failure_path.read_text())})
                record = {"identity": manifest["identity"], "source_commit": manifest["source_commit"],
                          "status": "ineligible", "reason": "numerical" if isinstance(error, NumericalPilotFailure) else "runtime",
                          "error": f"{type(error).__name__}: {error}", "completed_utc": datetime.now(UTC).isoformat(),
                          "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "numerical_failures": details}
                write_json(path, record)
                write_json(Path(run["workdir"]) / "preflight_failure.json", record)
                print(f"preflight_ineligible={stage} reason={record['reason']} error={error}", flush=True)
        record_success(root / "success.json", manifest, hardware=hardware,
                       checks=[*SHARED_CHECKS, *[run["name"] for run in manifest["runs"]]])
        retain_preflight_evidence(manifest)
    except BaseException as error:
        write_json(attempt / "failure.json", {"identity": manifest["identity"], "status": "failed", "stage": stage,
                                             "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise
    print(f"preflight_complete={root / 'success.json'} eligible={sorted(eligible_runs(manifest))}", flush=True)


def worker_result(manifest: dict, worker: str, name: str, workdir: Path | None, stop: int) -> dict:
    import jax
    import numpy as np
    if worker == "hardware":
        from scaffolds.transformer.studies.polynomial_attention.preflight import worker_result as historical_worker
        return historical_worker(manifest, worker, name, workdir, stop)
    if worker == "initialization":
        from flax import nnx
        from scaffolds.transformer.models import build_model
        config = load_config(manifest["controls"][0]["config_path"] if name == "fixed" else
                             next(run["config_path"] for run in manifest["runs"] if run["name"] == name))
        rng = jax.random.PRNGKey(config.train.seed)
        model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        digest, count, coeff = hashlib.sha256(), 0, None
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]:
            array = np.asarray(leaf)
            if array.dtype != np.float32:
                raise ValueError("Every initial parameter must be FP32")
            if "quadratic_coefficients" in str(path):
                if coeff is not None:
                    raise ValueError("Expected exactly one model-level coefficient parameter array")
                coeff = array
                continue
            digest.update(str(path).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            count += array.size
        coefficient_count = 0 if coeff is None else coeff.size
        if coeff is not None and not np.all(coeff.reshape(-1, 2) == coeff.reshape(-1, 2)[0]):
            raise ValueError("Coefficient initialization differs across heads/layers")
        return {"backbone_sha256": digest.hexdigest(), "backbone_parameter_count": count,
                "parameter_count": count + coefficient_count, "coefficient_count": coefficient_count,
                "coefficient_shape": None if coeff is None else list(coeff.shape),
                "initial_coefficient_pair": None if coeff is None else coeff.reshape(-1, 2)[0].tolist()}
    if worker == "pilot":
        if workdir is None:
            raise ValueError("Pilot requires an isolated workdir")
        run = next(run for run in manifest["runs"] if run["name"] == name)
        config = load_config(run["config_path"])
        config.train.workdir, config.train.wandb_mode = workdir, "disabled"
        config.train.checkpoint_dir = None  # Isolate pilot artifacts from production.
        config.train.eval_every, config.train.eval_batches = 2, 1
        from scaffolds.transformer.training.loop import run_training
        return run_training(config, resume=True, stop_after_step=stop, audit_checkpoint=True)
    raise ValueError(f"Unknown worker: {worker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-hardware", action="store_true")
    parser.add_argument("--worker", choices=("hardware", "initialization", "pilot"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        verify_manifest(manifest)
        try:
            write_json(args.output, worker_result(manifest, args.worker, args.run_name, args.workdir, args.stop_after_step))
        except BaseException as error:
            write_json(args.output, {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                    "numerical_failure": isinstance(error, FloatingPointError)})
            raise
    else:
        run_preflight(manifest, check_only=args.check_only, check_hardware=args.check_hardware,
                      run_name=args.run_name or None)


if __name__ == "__main__":
    main()
