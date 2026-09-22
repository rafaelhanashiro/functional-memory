"""Frozen-source, kernel, and full-model gates with per-coordinate eligibility."""
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

from experiments.shared_transformer.configs import config_to_dict, load_config
from experiments.shared_transformer.configs.loader import config_dicts_equal
from experiments.polynomial_attention.preflight import passed_check, record_success, require_passing_tests
from experiments.key_dimension_taylor.sweep import (
    ALLOCATION_FRACTION, CHECKPOINT_BOUNDARY, EXPERIMENT, PILOT_STEPS, active_runs,
    identity_payload, manifest_path, parameter_count, run_name, scheduler_profile,
    scientific_recipe, sha256, validate_retained_run, write_json,
)

SHARED_CHECKS = ("kernels-cpu", "training-audit-cpu", "training-distributed-cpu", "kernels-gpu", "initialization")


class NumericalPilotFailure(RuntimeError):
    """A coordinate failed numerically despite shared kernel correctness checks."""


class RuntimeLimitExceeded(RuntimeError):
    """Even the smallest allowed checkpoint segment exceeds the time allowance."""


def verify_manifest(manifest: dict) -> None:
    actual = hashlib.sha256(json.dumps(identity_payload(manifest), sort_keys=True).encode()).hexdigest()
    if actual != manifest["identity"] or manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Sweep identity does not match its frozen contents")
    if manifest["scheduler"] != scheduler_profile():
        raise ValueError("Sweep requires four H100s on dogo with the recorded walltimes")
    expected = [(width, degree) for width in manifest["key_dimensions"] for degree in manifest["degrees"]]
    if [(run["qk_head_dim"], run["attention_degree"]) for run in manifest["runs"]] != expected:
        raise ValueError("Manifest must contain the complete selected Cartesian grid")
    if manifest["execution_policy"] != {"pilot_steps": PILOT_STEPS, "checkpoint_boundary": CHECKPOINT_BOUNDARY,
                                        "allocation_fraction": ALLOCATION_FRACTION}:
        raise ValueError("Pilot or allocation segmentation policy changed")
    execution = Path(manifest["execution_root"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution, text=True).strip() != manifest["source_commit"]:
        raise ValueError("Frozen execution source changed")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution, text=True):
        raise ValueError("Frozen execution checkout is no longer clean")
    if sha256(execution / manifest.get("dependency_lock_path", "experiments/polynomial_attention/uv.lock")) != manifest["dependency_lock_sha256"]:
        raise ValueError("Dependency lock changed")
    for data in manifest["data_identity"].values():
        if sha256(Path(data["path"])) != data["sha256"]:
            raise ValueError("Data manifest changed")
    references = [run for run in manifest["runs"] if run.get("retained_completed")]
    references += [*manifest["controls"], manifest["alternative_softmax64"]]
    for run in references:
        validate_retained_run(run)
    recipe = scientific_recipe(manifest["controls"][0]["config"], manifest["controls"][0]["data_identity"])
    for run in active_runs(manifest):
        config = load_config(run["config_path"])
        if (not config_dicts_equal(config_to_dict(config), run["config"])
                or scientific_recipe(run["config"], manifest["data_identity"]) != recipe
                or config.model.qk_head_dim != run["qk_head_dim"]
                or config.model.attention_degree != run["attention_degree"]
                or config.model.attention_backend != "fused"
                or parameter_count(config.model) != run["parameter_count"]
                or config.train.source_commit != manifest["source_commit"]):
            raise ValueError(f"Run configuration or grid identity changed: {run['name']}")
        root = Path(manifest["project_root"])
        expected_run = root / "results" / EXPERIMENT / run["name"]
        expected_checkpoint = root / "experiments" / EXPERIMENT / "checkpoints" / run["name"]
        link = expected_run / "checkpoints"
        if (Path(run["workdir"]) != expected_run or Path(run["checkpoint_dir"]) != expected_checkpoint
                or not link.is_symlink() or link.resolve() != expected_checkpoint.resolve()):
            raise ValueError("Run artifacts do not use canonical retained storage")


def eligible_runs(manifest: dict) -> set[str]:
    root = Path(manifest["preflight_dir"])
    if not passed_check(root / "success.json", manifest["identity"]):
        return set()
    if not all(passed_check(root / "checks" / f"{name}.json", manifest["identity"]) for name in SHARED_CHECKS):
        return set()
    return {run["name"] for run in active_runs(manifest)
            if passed_check(root / "checks" / f"{run['name']}.json", manifest["identity"])}


def preflight_complete(manifest: dict) -> bool:
    root = Path(manifest["preflight_dir"])
    if not passed_check(root / "success.json", manifest["identity"]):
        return False
    if not all(passed_check(root / "checks" / f"{name}.json", manifest["identity"]) for name in SHARED_CHECKS):
        return False
    for run in active_runs(manifest):
        path = root / "checks" / f"{run['name']}.json"
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        if record.get("identity") != manifest["identity"] or record.get("status") not in ("passed", "ineligible"):
            return False
    return True


def launch_worker(manifest: dict, worker: str, output: Path, *, name="", workdir=None,
                  stop=PILOT_STEPS, cpu=False) -> dict:
    command = [sys.executable, "-m", "experiments.key_dimension_taylor.preflight",
               str(manifest_path(Path(manifest["project_root"]))), "--worker", worker,
               "--output", str(output), "--run-name", name, "--stop-after-step", str(stop)]
    if workdir is not None:
        command += ["--workdir", str(workdir)]
    result = subprocess.run(command, cwd=manifest["execution_root"],
                            env=dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda"), check=False)
    if result.returncode:
        failure = json.loads(output.read_text()) if output.exists() else {}
        if worker == "pilot" and failure.get("numerical_failure"):
            raise NumericalPilotFailure(failure["error"])
        raise subprocess.CalledProcessError(result.returncode, command)
    return json.loads(output.read_text())


def projected_runtime(run: dict, initial: dict, resumed: dict, scheduler: dict) -> dict:
    timing, config = initial["timing"], run["config"]
    steady, evaluation, checkpoints = timing["train_step_seconds"][1:], timing["evaluation_seconds"], timing["checkpoint_seconds"]
    # Each continuation separately restores/audits state before entering the
    # training worker. Charge two complete measured startups conservatively.
    startup = 2 * max(timing["runtime_seconds"], resumed["timing"]["runtime_seconds"])
    values = [startup, *steady, *evaluation, *checkpoints]
    if not steady or not evaluation or not checkpoints or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("Pilot lacks valid synchronized training/evaluation/checkpoint timings")
    hours, minutes, seconds = map(int, scheduler["production_time"].split(":"))
    limit = hours * 3600 + minutes * 60 + seconds
    usable = ALLOCATION_FRACTION * limit
    total = config["optim"]["total_steps"]
    eval_every, ckpt_every = config["train"]["eval_every"], config["train"]["ckpt_every"]
    if ckpt_every != CHECKPOINT_BOUNDARY:
        raise ValueError("Production segments require the original 500-update checkpoint cadence")
    def estimate(start, end):
        eval_steps = {step for step in range(eval_every, total + 1, eval_every) if start < step <= end}
        if end == total:
            eval_steps.add(total)
        ckpt_steps = {step for step in range(ckpt_every, total + 1, ckpt_every) if start < step <= end} | eval_steps | {end}
        return (startup + (end - start) * max(steady)
                + len(eval_steps) * config["train"]["eval_batches"] * max(evaluation)
                + len(ckpt_steps) * max(checkpoints))
    segments, start = [], 0
    boundaries = [*range(CHECKPOINT_BOUNDARY, total, CHECKPOINT_BOUNDARY), total]
    while start < total:
        fitting = [end for end in boundaries if end > start and estimate(start, end) <= usable]
        if not fitting:
            raise RuntimeLimitExceeded(f"{run['name']} cannot fit its smallest checkpoint segment within 80% of walltime")
        end = max(fitting)
        segments.append({"start_step": start, "end_step": end, "projected_seconds": estimate(start, end)})
        start = end
    return {"startup_seconds": startup, "steady_train_step_seconds": max(steady),
            "evaluation_batch_seconds": max(evaluation), "checkpoint_seconds": max(checkpoints),
            "limit_seconds": limit, "usable_seconds": usable, "allocation_fraction": ALLOCATION_FRACTION,
            "segments": segments, "projected_total_seconds": sum(item["projected_seconds"] for item in segments)}

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
    source = Path(manifest["preflight_dir"])
    destination = Path(manifest["project_root"]) / "results" / EXPERIMENT / "preflight" / manifest["identity"]
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*SHARED_CHECKS, *[run["name"] for run in active_runs(manifest)]):
        record = json.loads((source / "checks" / f"{name}.json").read_text())
        if record.get("junit_path"):
            target = destination / f"{name}.xml"
            shutil.copyfile(record["junit_path"], target)
            record["junit_path"] = str(target)
        write_json(destination / "checks" / f"{name}.json", record)
    write_json(destination / "success.json", json.loads((source / "success.json").read_text()))
    write_json(destination / "provenance.json", {"identity": manifest["identity"], "source_commit": manifest["source_commit"],
               "data_identity": manifest["data_identity"], "scheduler": manifest["scheduler"],
               "log_path": manifest.get("preflight_log_path"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "eligible_runs": sorted(eligible_runs(manifest))})


def run_preflight(manifest: dict, *, check_only=False, check_hardware=False, run_name=None) -> None:
    verify_manifest(manifest)
    root = Path(manifest["preflight_dir"])
    if check_only and not check_hardware:
        if not preflight_complete(manifest):
            raise RuntimeError("No complete matching preflight")
        if run_name is not None and run_name not in eligible_runs(manifest):
            raise RuntimeError("Run did not pass its own pilot")
        return
    attempt = root / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True)
    stage = "hardware"
    try:
        hardware = launch_worker(manifest, "hardware", attempt / "hardware.json")
        if preflight_complete(manifest):
            if run_name is not None and run_name not in eligible_runs(manifest):
                raise RuntimeError("Run did not pass its own pilot")
            return
        if check_only:
            raise RuntimeError("No complete matching preflight")
        for stage, test_file, cpu, selection, minimum in (
            ("kernels-cpu", "experiments/key_dimension_taylor/tests/test_kernels.py", True, ("-m", "not gpu"), 20),
            ("training-audit-cpu", "experiments/key_dimension_taylor/tests/test_training_audit.py", True, ("-m", "not gpu"), 29),
            ("training-distributed-cpu", "experiments/shared_transformer/tests/test_training.py", True, ("-k", "distributed"), 1),
            ("kernels-gpu", "experiments/key_dimension_taylor/tests/test_kernels.py", False, ("-m", "gpu"), 105),
        ):
            path = root / "checks" / f"{stage}.json"
            if passed_check(path, manifest["identity"]):
                continue
            environment = dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda")
            if stage == "training-distributed-cpu":
                environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
            junit = attempt / f"{stage}.xml"
            subprocess.run([sys.executable, "-m", "pytest", "-c", "experiments/shared_transformer/pyproject.toml",
                            test_file, "-q", *selection, f"--junitxml={junit}"],
                           cwd=manifest["execution_root"], env=environment, check=True)
            record_success(path, manifest, tests_passed=require_passing_tests(junit, minimum), junit_path=str(junit))
        stage = "initialization"
        path = root / "checks/initialization.json"
        if not passed_check(path, manifest["identity"]):
            widths = sorted({run["qk_head_dim"] for run in active_runs(manifest)})
            references = {width: launch_worker(manifest, "initialization", attempt / f"init-reference-dk{width}.json",
                           name=f"reference-dk{width}", cpu=True) for width in widths}
            results = {}
            for run in active_runs(manifest):
                result = launch_worker(manifest, "initialization", attempt / f"init-{run['name']}.json", name=run["name"], cpu=True)
                if result != references[run["qk_head_dim"]] or result["parameter_count"] != run["parameter_count"]:
                    raise RuntimeError(f"Backbone initialization changed for {run['name']}")
                results[run["name"]] = result
            record_success(path, manifest, references=references, results=results)
        for run in active_runs(manifest):
            stage = run["name"]
            path = root / "checks" / f"{stage}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("identity") == manifest["identity"] and record.get("status") in ("passed", "ineligible"):
                    continue
            workdir = attempt / stage
            try:
                initial = launch_worker(manifest, "pilot", attempt / f"{stage}-initial.json", name=stage, workdir=workdir)
                if initial["final_step"] != PILOT_STEPS or initial["data_parallel_devices"] != manifest["scheduler"]["gpu_count"]:
                    raise RuntimeError("Pilot lost full update budget or allocation")
                from experiments.polynomial_attention.plot import committed_records
                validation = next((row for row in committed_records(workdir) if row["step"] == PILOT_STEPS), None)
                if validation is None or not all(math.isfinite(validation[key]) for key in ("validation_nll", "validation_perplexity")):
                    raise NumericalPilotFailure("Pilot lacks finite committed validation")
                resumed = launch_worker(manifest, "pilot", attempt / f"{stage}-resume.json", name=stage, workdir=workdir, stop=PILOT_STEPS + 1)
                if (resumed["restored_step"] != PILOT_STEPS or resumed["final_step"] != PILOT_STEPS + 1
                        or resumed["data_start_sequence"] != PILOT_STEPS * run["config"]["data"]["batch_size"]
                        or resumed["data_parallel_devices"] != manifest["scheduler"]["gpu_count"]):
                    raise RuntimeError("Pilot restart changed optimizer, data cursor, or allocation")
                validate_resume_audit(initial, resumed)
                runtime = projected_runtime(run, initial, resumed, manifest["scheduler"])
                record_success(path, manifest, initial=initial, resumed=resumed, runtime=runtime,
                               qk_head_dim=run["qk_head_dim"], attention_degree=run["attention_degree"], validation=validation)
                if (workdir / "checkpoints").exists():
                    shutil.rmtree(workdir / "checkpoints")
            except (NumericalPilotFailure, RuntimeLimitExceeded) as error:
                failures = [{"pilot": workdir.name, "record": json.loads(item.read_text())}
                            for item in sorted(workdir.glob("numerical_failure_step_*.json"))]
                record = {"identity": manifest["identity"], "source_commit": manifest["source_commit"], "status": "ineligible",
                          "reason": "numerical" if isinstance(error, NumericalPilotFailure) else "runtime", "error": str(error),
                          "qk_head_dim": run["qk_head_dim"], "attention_degree": run["attention_degree"],
                          "numerical_failures": failures, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
                write_json(path, record)
                write_json(Path(run["workdir"]) / "preflight_failure.json", record)
                if (workdir / "checkpoints").exists():
                    shutil.rmtree(workdir / "checkpoints")
        record_success(root / "success.json", manifest, hardware=hardware)
        retain_preflight_evidence(manifest)
    except BaseException as error:
        write_json(attempt / "failure.json", {"identity": manifest["identity"], "status": "failed", "stage": stage,
                                             "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise


def worker_result(manifest: dict, worker: str, name: str, workdir: Path | None, stop: int) -> dict:
    import jax
    if worker == "hardware":
        from experiments.polynomial_attention.preflight import worker_result as original_worker
        return original_worker(manifest, worker, name, workdir, stop)
    if worker == "initialization":
        import numpy as np
        from flax import nnx
        if name.startswith("reference-dk"):
            width = int(name.removeprefix("reference-dk"))
            from experiments.key_dimension.configs import load_config as original_config
            from experiments.key_dimension.models import build_model
            config = original_config(next(run["config_path"] for run in manifest["controls"] if run["qk_head_dim"] == width))
        else:
            from experiments.shared_transformer.models import build_model
            config = load_config(next(run["config_path"] for run in active_runs(manifest) if run["name"] == name))
        rng = jax.random.PRNGKey(config.train.seed)
        model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        digest, count = hashlib.sha256(), 0
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]:
            array = np.asarray(leaf)
            if array.dtype != np.float32:
                raise ValueError("Initial model parameters must be FP32")
            digest.update(str(path).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            count += array.size
        return {"parameter_count": count, "parameter_sha256": digest.hexdigest()}
    if worker == "pilot":
        if workdir is None:
            raise ValueError("Pilots require an isolated workdir")
        config = load_config(next(run["config_path"] for run in active_runs(manifest) if run["name"] == name))
        config.train.workdir, config.train.wandb_mode = workdir, "disabled"
        config.train.eval_every, config.train.eval_batches = PILOT_STEPS, 1
        from experiments.shared_transformer.training.loop import run_training
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
    parser.add_argument("--stop-after-step", type=int, default=PILOT_STEPS)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.worker:
        if args.output is None:
            parser.error("Workers require --output")
        verify_manifest(manifest)
        try:
            write_json(args.output, worker_result(manifest, args.worker, args.run_name, args.workdir, args.stop_after_step))
        except BaseException as error:
            write_json(args.output, {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                    "numerical_failure": isinstance(error, FloatingPointError)})
            raise
    else:
        run_preflight(manifest, check_only=args.check_only, check_hardware=args.check_hardware, run_name=args.run_name or None)


if __name__ == "__main__":
    main()
