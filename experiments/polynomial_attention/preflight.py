"""Retry-safe CPU and two-H200 checks that gate an exact Taylor sweep identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from experiments.polynomial_attention.configs import config_to_dict, load_config
from experiments.polynomial_attention.sweep import ATTENTION_DEGREES, run_name, write_json

PREFLIGHT_CHECKS = ("attention-cpu", "attention-gpu", "initialization", *[run_name(m) for m in ATTENTION_DEGREES])


def verify_manifest(manifest: dict) -> None:
    payload = {
        "source_commit": manifest["source_commit"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "data_identity": manifest["data_identity"],
        "configs": [run["config"] for run in manifest["runs"]],
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if identity != manifest["identity"]:
        raise ValueError("Sweep manifest identity does not match its contents")
    if [run["attention_degree"] for run in manifest["runs"]] != list(ATTENTION_DEGREES):
        raise ValueError("Preflight requires every degree and exact-softmax control in canonical order")
    execution_root = Path(manifest["execution_root"])
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution_root, text=True).strip()
    if actual_commit != manifest["source_commit"]:
        raise ValueError("Execution source commit changed since sweep resolution")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution_root, text=True)
    if dirty:
        raise ValueError("Frozen execution checkout is no longer clean")
    lock_sha = hashlib.sha256((execution_root / "experiments/polynomial_attention/uv.lock").read_bytes()).hexdigest()
    if lock_sha != manifest["dependency_lock_sha256"]:
        raise ValueError("Dependency lock changed since sweep resolution")
    for identity in manifest["data_identity"].values():
        if hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError(f"Data manifest changed: {identity['path']}")
    for run in manifest["runs"]:
        if config_to_dict(load_config(run["config_path"])) != run["config"]:
            raise ValueError(f"Run config changed: {run['config_path']}")
        if run["config"]["model"]["attention_degree"] != run["attention_degree"]:
            raise ValueError(f"Run degree does not match resolved config: {run['name']}")


def passed_check(path: Path, identity: str) -> bool:
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    return record.get("identity") == identity and record.get("status") == "passed"


def record_success(path: Path, manifest: dict, **fields) -> None:
    write_json(path, {
        "identity": manifest["identity"],
        "source_commit": manifest["source_commit"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "status": "passed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        **fields,
    })


def preflight_complete(manifest: dict) -> bool:
    root = Path(manifest["preflight_dir"])
    if not passed_check(root / "success.json", manifest["identity"]):
        return False
    return all(passed_check(root / "checks" / f"{name}.json", manifest["identity"]) for name in PREFLIGHT_CHECKS)


def require_passing_tests(junit_path: Path, minimum: int) -> int:
    testcases = list(ET.parse(junit_path).iter("testcase"))
    passed = [case for case in testcases if not any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))]
    if len(passed) < minimum or len(passed) != len(testcases):
        raise RuntimeError(f"Expected at least {minimum} passing checks and no skipped tests; got {len(passed)}/{len(testcases)}")
    return len(passed)


def launch_worker(manifest: dict, worker: str, output: Path, *, name: str = "", workdir: Path | None = None, stop: int = 2, cpu: bool = False) -> dict:
    """Every worker exits before the next starts, releasing JAX/GPU caches."""
    manifest_path = Path(manifest["project_root"]) / "results/polynomial_attention/sweep.json"
    command = [sys.executable, "-m", "experiments.polynomial_attention.preflight", str(manifest_path),
               "--worker", worker, "--output", str(output), "--run-name", name, "--stop-after-step", str(stop)]
    if workdir is not None:
        command += ["--workdir", str(workdir)]
    environment = dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda")
    subprocess.run(command, cwd=manifest["execution_root"], env=environment, check=True)
    return json.loads(output.read_text())


def run_preflight(manifest: dict, *, check_only: bool = False) -> None:
    verify_manifest(manifest)
    root = Path(manifest["preflight_dir"])
    success_path = root / "success.json"
    if preflight_complete(manifest):
        print(f"preflight_success_reused={success_path}", flush=True)
        return
    if check_only:
        raise RuntimeError(f"This sweep has no successful matching complete preflight: {success_path}")
    attempt = root / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True)
    stage = "hardware"
    try:
        hardware = launch_worker(manifest, "hardware", attempt / "hardware.json")
        for marker, minimum in (("cpu", len(ATTENTION_DEGREES)), ("gpu", len(ATTENTION_DEGREES) + 1)):
            stage = f"attention-{marker}"
            check_path = root / "checks" / f"{stage}.json"
            if passed_check(check_path, manifest["identity"]):
                continue
            junit_path = attempt / f"{stage}.xml"
            subprocess.run([
                sys.executable, "-m", "pytest", "-c", "experiments/polynomial_attention/pyproject.toml",
                "experiments/polynomial_attention/tests/test_attention.py", "-q", "-m", "not gpu" if marker == "cpu" else "gpu",
                f"--junitxml={junit_path}",
            ], cwd=manifest["execution_root"], env=dict(os.environ, JAX_PLATFORMS="cpu" if marker == "cpu" else "cuda"), check=True)
            passed = require_passing_tests(junit_path, minimum)
            record_success(check_path, manifest, tests_passed=passed, junit_path=str(junit_path))
        stage = "initialization"
        initial_check = root / "checks/initialization.json"
        if not passed_check(initial_check, manifest["identity"]):
            initializations = {
                name: launch_worker(manifest, "initialization", attempt / f"init-{name}.json", name=name, cpu=True)
                for name in ["original", *[run["name"] for run in manifest["runs"]]]
            }
            reference = initializations["original"]
            if any(value != reference for value in initializations.values()):
                raise RuntimeError(f"Initial parameters differ across degrees or from original dK64 recipe: {initializations}")
            record_success(initial_check, manifest, initializations=initializations)
        for run in manifest["runs"]:
            stage = run["name"]
            check_path = root / "checks" / f"{stage}.json"
            if passed_check(check_path, manifest["identity"]):
                print(f"preflight_check_reused={stage}", flush=True)
                continue
            workdir = attempt / stage
            outcome = launch_worker(manifest, "pilot", attempt / f"{stage}-step2.json", name=stage, workdir=workdir)
            if outcome["final_step"] != 2 or outcome["data_parallel_devices"] != 2:
                raise RuntimeError(f"Preflight {stage} did not finish two updates on two GPUs: {outcome}")
            from experiments.polynomial_attention.plot import committed_records
            records = committed_records(workdir)
            validation = next((record for record in records if record["step"] == 2), None)
            if validation is None or not math.isfinite(validation["validation_nll"]) or not math.isfinite(validation["validation_perplexity"]):
                raise RuntimeError(f"Preflight {stage} has no finite committed validation at step 2")
            if run["attention_degree"] in (16, None):
                outcome = launch_worker(manifest, "pilot", attempt / f"{stage}-resume.json", name=stage, workdir=workdir, stop=3)
                if outcome["restored_step"] != 2 or outcome["final_step"] != 3 or outcome["data_start_sequence"] != 2 * run["config"]["data"]["batch_size"]:
                    raise RuntimeError(f"Preflight {stage} resume did not preserve optimizer/data progress: {outcome}")
            record_success(check_path, manifest, attention_degree=run["attention_degree"], outcome=outcome,
                           validation_nll=validation["validation_nll"], validation_perplexity=validation["validation_perplexity"], workdir=str(workdir))
        record_success(success_path, manifest, checks=list(PREFLIGHT_CHECKS), hardware=hardware)
    except BaseException as error:
        write_json(attempt / "failure.json", {"identity": manifest["identity"], "status": "failed", "stage": stage,
                                             "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise
    print(f"preflight_success={success_path}", flush=True)


def worker_result(manifest: dict, worker: str, name: str, workdir: Path | None, stop: int) -> dict:
    import jax
    if worker == "hardware":
        devices = jax.local_devices()
        if jax.default_backend() != "gpu" or len(devices) != 2 or any("H200" not in d.device_kind.upper() for d in devices):
            raise RuntimeError(f"Preflight requires two H200 GPUs; found {devices!r}")
        return {"backend": "gpu", "devices": [str(d) for d in devices], "device_kinds": [d.device_kind for d in devices]}
    if worker == "initialization":
        import numpy as np
        from flax import nnx
        if name == "original":
            from experiments.key_dimension.configs import load_config as load_original
            from experiments.key_dimension.models import build_model
            config = load_original(Path(manifest["execution_root"]) / "experiments/key_dimension/configs/base.yaml")
        else:
            from experiments.polynomial_attention.models import build_model
            config = load_config(next(run["config_path"] for run in manifest["runs"] if run["name"] == name))
        rng = jax.random.PRNGKey(config.train.seed)
        model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        digest = hashlib.sha256()
        count = 0
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]:
            array = np.asarray(leaf)
            if array.dtype != np.float32:
                raise ValueError("Every initial parameter must be FP32")
            digest.update(str(path).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            count += array.size
        return {"parameter_count": count, "parameter_sha256": digest.hexdigest()}
    if worker == "pilot":
        if workdir is None:
            raise ValueError("Pilot requires an isolated workdir")
        run = next(run for run in manifest["runs"] if run["name"] == name)
        config = load_config(run["config_path"])
        config.train.workdir = workdir
        config.train.wandb_mode = "disabled"
        config.train.eval_every = 2
        config.train.eval_batches = 1
        from experiments.polynomial_attention.training.loop import run_training
        return run_training(config, resume=True, stop_after_step=stop)
    raise ValueError(f"Unknown preflight worker: {worker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-only", action="store_true")
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
        write_json(args.output, worker_result(manifest, args.worker, args.run_name, args.workdir, args.stop_after_step))
    else:
        run_preflight(manifest, check_only=args.check_only)


if __name__ == "__main__":
    main()
