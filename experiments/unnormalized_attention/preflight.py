"""Identity-checked tests and full-recipe GPU pilot before production."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import UTC, datetime
import xml.etree.ElementTree as ET

from experiments.unnormalized_attention.configs import load_config, config_to_dict


ROOT = Path(__file__).resolve().parents[2]
FAMILY = "experiments.unnormalized_attention"
CHECKS = ("cpu-tests", "gpu-tests", "initialization", "pilot", "resume")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(config_path, gpu_count, gpu_type):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Preflight requires a clean frozen experiment checkout")
    config = load_config(config_path)
    if config.train.source_commit not in ("auto", source):
        raise ValueError("Configuration source differs from execution checkout")
    config.train.source_commit = source
    if config.model.attention_normalization:
        raise ValueError("This preflight is for the unnormalized experiment")
    payload = {
        "source_commit": source,
        "config": config_to_dict(config),
        "config_sha256": sha256(config_path),
        "dependency_lock_sha256": sha256(ROOT / "experiments/unnormalized_attention/uv.lock"),
        "data_manifests": {
            split: {"path": path, "sha256": sha256(path)}
            for split, path in (("train", config.data.train_manifest), ("eval", config.data.eval_manifest))
        },
        "gpu_count": gpu_count, "gpu_type": gpu_type.upper(),
    }
    payload["identity"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def passed(path, experiment):
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return value.get("identity") == experiment["identity"] and value.get("status") == "passed"


def record(path, experiment, **fields):
    write_json(path, {
        "identity": experiment["identity"], "source_commit": experiment["source_commit"],
        "status": "passed", "completed_utc": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"), **fields,
    })


def require_passing_tests(path):
    cases = list(ET.parse(path).iter("testcase"))
    failed = [case for case in cases if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    if not cases or failed:
        raise RuntimeError(f"Acceptance tests must all pass with no skips: {len(cases)} cases, {len(failed)} failures/skips")
    return len(cases)


def run_worker(args, name, output, *, stop=20, cpu=False):
    command = [sys.executable, "-m", FAMILY + ".preflight", str(args.config),
               "--directory", str(args.directory), "--gpu-count", str(args.gpu_count),
               "--gpu-type", args.gpu_type, "--worker", name, "--output", str(output),
               "--stop-after-step", str(stop)]
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda"), check=True)
    return json.loads(output.read_text())


def worker(args, experiment):
    if args.worker == "hardware":
        import jax
        devices = jax.local_devices()
        if (jax.default_backend() != "gpu" or len(devices) != args.gpu_count
                or any(args.gpu_type.upper() not in d.device_kind.upper() for d in devices)):
            raise RuntimeError(f"Expected {args.gpu_count} {args.gpu_type} GPUs; got {devices}")
        from importlib.metadata import version
        return {"devices": [str(d) for d in devices], "device_kinds": [d.device_kind for d in devices],
                "versions": {name: version(name) for name in ("jax", "flax", "optax", "orbax-checkpoint")}}
    if args.worker == "pilot":
        from experiments.unnormalized_attention.training.loop import run_training
        config = load_config(args.config)
        config.train.source_commit = experiment["source_commit"]
        config.train.workdir = args.directory / "pilot"
        config.train.wandb_mode = "disabled"
        config.train.eval_every = 20
        config.train.eval_batches = 1
        # Preserve model, context, batch, seed, LR, warmup, and 21,935-step horizon.
        return run_training(config, resume=True, stop_after_step=args.stop_after_step)
    raise ValueError(args.worker)


def run(args, experiment):
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "identity.json", experiment)
    hardware = run_worker(args, "hardware", directory / "hardware.json")
    complete = passed(directory / "success.json", experiment) and all(
        passed(directory / (name + ".json"), experiment) for name in CHECKS
    )
    if complete:
        print(f"matching_preflight_passed={directory / 'success.json'}", flush=True)
        return
    if args.check_only:
        raise RuntimeError("Production requires all matching preflight checks")
    stage = "cpu-tests"
    try:
        for stage in ("cpu-tests", "gpu-tests"):
            path = directory / (stage + ".json")
            if passed(path, experiment):
                continue
            junit = directory / (stage + ".xml")
            env = dict(os.environ, JAX_PLATFORMS="cpu" if stage == "cpu-tests" else "cuda")
            subprocess.run([
                sys.executable, "-m", "pytest", "-c", "experiments/unnormalized_attention/pyproject.toml",
                "experiments/unnormalized_attention/tests", "-q", "-m", "not gpu" if stage == "cpu-tests" else "gpu",
                f"--junitxml={junit}",
            ], cwd=ROOT, env=env, check=True)
            record(path, experiment, tests_passed=require_passing_tests(junit))
        stage = "initialization"
        path = directory / (stage + ".json")
        if not passed(path, experiment):
            output = directory / "initialization-result.json"
            subprocess.run([sys.executable, "-m", FAMILY + ".verify_initialization", str(args.config), "--output", str(output)],
                           cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu"), check=True)
            record(path, experiment, result=json.loads(output.read_text()))
        stage = "pilot"
        path = directory / (stage + ".json")
        if not passed(path, experiment):
            # A preempted disposable pilot may have saved step 20 without its
            # timing result. Restart incomplete pilot checks to measure a full
            # compilation/steady-step window; already passed checks survive.
            if (directory / "pilot").exists():
                shutil.rmtree(directory / "pilot")
            outcome = run_worker(args, "pilot", directory / "pilot-result.json")
            if outcome["final_step"] != 20 or outcome["data_parallel_devices"] != args.gpu_count:
                raise RuntimeError(f"Incomplete full-recipe pilot: {outcome}")
            records = [json.loads(line) for line in (directory / "pilot/metrics.jsonl").read_text().splitlines()]
            validation = next(row for row in records if row["step"] == 20)
            if not math.isfinite(validation["validation_nll"]) or not validation["attention_diagnostics"]["valid"]:
                raise RuntimeError("Pilot validation or attention diagnostics failed")
            timing = outcome["timing"]
            steady = timing["train_step_seconds"][1:]
            if not steady or any(not math.isfinite(x) or x <= 0 for x in steady):
                raise RuntimeError("Pilot has no valid synchronized steady update timing")
            config = experiment["config"]
            # Include compilation once plus conservative evaluation/checkpoint costs.
            projected = (timing["runtime_seconds"] + max(steady) * config["optim"]["total_steps"]
                         + max(timing["evaluation_seconds"]) * config["train"]["eval_batches"] * 23
                         + max(timing["checkpoint_seconds"]) * 44)
            if projected > 46 * 3600:
                raise RuntimeError(f"Pilot projects beyond a 48h allocation: {projected / 3600:.2f} hours")
            record(path, experiment, outcome=outcome, validation=validation, projected_hours=projected / 3600)
        stage = "resume"
        path = directory / (stage + ".json")
        if not passed(path, experiment):
            outcome = run_worker(args, "pilot", directory / "resume-result.json", stop=21)
            # A retry after checkpoint 21 was committed can already be at 21.
            if (outcome["restored_step"] not in (20, 21) or outcome["final_step"] != 21
                    or outcome["data_start_sequence"] != outcome["restored_step"] * experiment["config"]["data"]["batch_size"]):
                raise RuntimeError(f"Pilot resume did not preserve optimizer and data position: {outcome}")
            record(path, experiment, outcome=outcome)
        record(directory / "success.json", experiment, checks=list(CHECKS), hardware=hardware)
        # Checkpoints from this disposable pilot are no longer needed after resume verification.
        shutil.rmtree(directory / "pilot")
        print(f"preflight_success={directory / 'success.json'}", flush=True)
    except BaseException as error:
        write_json(directory / "failure.json", {"identity": experiment["identity"], "stage": stage,
                   "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", choices=("hardware", "pilot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int, default=20)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run tests, pilots, and training inside a Slurm allocation")
    args.directory = args.directory.resolve()
    args.config = args.config.resolve()
    experiment = identity(args.config, args.gpu_count, args.gpu_type)
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        write_json(args.output, worker(args, experiment))
    else:
        run(args, experiment)


if __name__ == "__main__":
    main()
