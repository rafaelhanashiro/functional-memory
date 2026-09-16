"""Retry-safe accelerator checks that gate this exact sweep identity."""
from __future__ import annotations

import argparse
import copy
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

from experiments.key_dimension.configs import config_to_dict, load_config
from experiments.key_dimension.sweep import write_json

PREFLIGHT_DIMENSIONS = (2, 4, 64, 128)


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
    execution_root = Path(manifest["execution_root"])
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution_root, text=True).strip()
    if actual_commit != manifest["source_commit"]:
        raise ValueError("Execution source commit changed since sweep resolution")
    lock_sha = hashlib.sha256((execution_root / "experiments/key_dimension/uv.lock").read_bytes()).hexdigest()
    if lock_sha != manifest["dependency_lock_sha256"]:
        raise ValueError("Dependency lock changed since sweep resolution")
    for identity in manifest["data_identity"].values():
        if hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError(f"Data manifest changed: {identity['path']}")
    for run in manifest["runs"]:
        if config_to_dict(load_config(run["config_path"])) != run["config"]:
            raise ValueError(f"Run config changed: {run['config_path']}")


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


def run_preflight(manifest: dict, *, check_only: bool = False) -> None:
    verify_manifest(manifest)
    root = Path(manifest["preflight_dir"])
    success_path = root / "success.json"
    if passed_check(success_path, manifest["identity"]):
        print(f"preflight_success_reused={success_path}", flush=True)
        return
    if check_only:
        raise RuntimeError(f"This sweep has no successful matching preflight: {success_path}")

    import jax

    backend = jax.default_backend()
    local_devices = jax.local_devices()
    if backend != "gpu" or len(local_devices) != 2:
        raise RuntimeError(
            "Production preflight requires the same two-GPU layout as production; "
            f"backend={backend!r}, local_devices={local_devices!r}, "
            f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')!r}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    attempt = root / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True)
    parity_record = root / "checks/attention-parity.json"
    if not passed_check(parity_record, manifest["identity"]):
        junit_path = attempt / "attention-parity.xml"
        subprocess.run([
            sys.executable, "-m", "pytest",
            "-c", "experiments/key_dimension/pyproject.toml",
            "experiments/key_dimension/tests/test_attention.py", "-q", "-k", "cudnn",
            f"--junitxml={junit_path}",
        ], cwd=manifest["execution_root"], check=True)
        testcases = list(ET.parse(junit_path).iter("testcase"))
        passed = [case for case in testcases if not any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))]
        if len(passed) != 8 or len(passed) != len(testcases):
            raise RuntimeError(f"Expected eight passing accelerator checks; got {len(passed)}/{len(testcases)}")
        record_success(parity_record, manifest, check="padded_cudnn_forward_and_gradients", tests_passed=len(passed))

    from experiments.key_dimension.training.loop import run_training

    for key_dim in PREFLIGHT_DIMENSIONS:
        check_path = root / "checks" / f"dk{key_dim:03d}.json"
        if passed_check(check_path, manifest["identity"]):
            print(f"preflight_check_reused=dk{key_dim:03d}", flush=True)
            continue
        run = next(run for run in manifest["runs"] if run["qk_head_dim"] == key_dim)
        config = copy.deepcopy(load_config(run["config_path"]))
        config.train.workdir = attempt / f"dk{key_dim:03d}"
        config.train.wandb_mode = "disabled"
        config.train.eval_every = 2
        config.train.eval_batches = 1
        outcome = run_training(config, resume=True, stop_after_step=2)
        if outcome["final_step"] != 2:
            raise RuntimeError(f"Preflight dk={key_dim} did not finish two updates: {outcome}")
        from experiments.key_dimension.plot import committed_records
        records = committed_records(config.train.workdir)
        validation = next((record for record in records if record["step"] == 2), None)
        if validation is None or not math.isfinite(validation["validation_nll"]) or not math.isfinite(validation["validation_perplexity"]):
            raise RuntimeError(f"Preflight dk={key_dim} has no finite committed validation at step 2")
        if key_dim == 64:
            outcome = run_training(config, resume=True, stop_after_step=3)
            if outcome["restored_step"] != 2 or outcome["final_step"] != 3:
                raise RuntimeError(f"Preflight resume did not preserve optimizer/data progress: {outcome}")
        record_success(check_path, manifest, qk_head_dim=key_dim, outcome=outcome, validation_nll=validation["validation_nll"], validation_perplexity=validation["validation_perplexity"], workdir=str(config.train.workdir))
    record_success(success_path, manifest, checks=["attention-parity", *[f"dk{dim:03d}" for dim in PREFLIGHT_DIMENSIONS]])
    print(f"preflight_success={success_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run_preflight(json.loads(args.manifest.read_text()), check_only=args.check_only)


if __name__ == "__main__":
    main()
