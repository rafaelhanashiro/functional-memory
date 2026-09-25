"""Identity-checked tests and full-recipe GPU pilot before GLA production."""
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

from scaffolds.transformer.configs import load_config, config_to_dict


ROOT = Path(__file__).resolve().parents[4]
FAMILY = "scaffolds.transformer.studies.gla_key_dimension"
CHECKS = ("cpu-tests", "gpu-tests", "initialization", "pilot", "resume")
# CPU acceptance covers the recurrence oracle, gradients, bridge tracing,
# configuration, and this study's recipes; GPU acceptance compares the fused
# Triton kernels with the sequential recurrence for every production width.
CPU_TESTS = ("scaffolds/transformer/tests/test_gla.py", "scaffolds/transformer/tests/test_gla_bridge.py",
             "scaffolds/transformer/tests/test_gla_integration.py", "scaffolds/transformer/studies/gla_key_dimension/tests")
GPU_TESTS = ("scaffolds/transformer/tests/test_gla_kernel.py",)
PILOT_STEPS = 20
ALLOCATION_HOURS = 48


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    """Qualification follows execution choices separately from the recipe."""
    host_staging = os.environ.get("TRANSFORMER_HOST_STAGING", "0")
    if host_staging not in ("0", "1"):
        raise ValueError("TRANSFORMER_HOST_STAGING must be resolved to 0 or 1 before preflight")
    return {
        "TRANSFORMER_HOST_STAGING": host_staging,
        **{key: os.environ.get(key) for key in (
            "NCCL_P2P_DISABLE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_FLAGS", "CUDA_HOME",
        )},
    }


def identity(config_path, gpu_count, gpu_type):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Preflight requires a clean frozen experiment checkout")
    config = load_config(config_path)
    if config.train.source_commit not in ("auto", source):
        raise ValueError("Configuration source differs from execution checkout")
    config.train.source_commit = source
    if config.model.attention_mechanism != "gla":
        raise ValueError("This preflight is for the GLA experiment")
    payload = {
        "source_commit": source,
        "config": config_to_dict(config),
        "config_sha256": sha256(config_path),
        "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "data_manifests": {
            split: {"path": path, "sha256": sha256(path)}
            for split, path in (("train", config.data.train_manifest), ("eval", config.data.eval_manifest))
        },
        "gpu_count": gpu_count, "gpu_type": gpu_type.upper(),
        "runtime": runtime_identity(),
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


def validate_resume_audit(initial, resumed):
    """Require exact checkpoint state, next batch, and RNG across a fresh-process restart."""
    for outcome in (initial, resumed):
        audit = outcome.get("checkpoint_audit")
        if not isinstance(audit, dict) or audit.get("final_state_sha256") != audit.get("checkpoint_state_sha256"):
            raise RuntimeError("Durable checkpoint differs from the live training state")
    before, after = initial["checkpoint_audit"], resumed["checkpoint_audit"]
    if before["final_state_sha256"] != after["start_state_sha256"]:
        raise RuntimeError("Restarted state differs from the saved live training state")
    if before.get("next_update") is None or before["next_update"] != after.get("first_update"):
        raise RuntimeError("Restarted training changed the next batch or RNG")


def run_worker(args, name, output, *, stop=PILOT_STEPS, cpu=False):
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
                "versions": {name: version(name) for name in ("jax", "flax", "optax", "orbax-checkpoint", "triton")}}
    if args.worker == "initialization":
        import jax
        from flax import nnx
        from scaffolds.transformer.models import build_model, model_parameter_count
        from scaffolds.transformer.training.checkpointing import state_sha256
        config = load_config(args.config)
        config.model.attention_backend = "blocked"  # Initialization does not depend on the backend.
        # Same derivation as training/state.py create_state, so this hashes the trained initialization.
        rng = jax.random.PRNGKey(config.train.seed)
        params = nnx.state(build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0))), nnx.Param)
        count = sum(leaf.size for leaf in jax.tree.leaves(params))
        if count != model_parameter_count(config.model):
            raise RuntimeError(f"Initialized parameter count {count} differs from the model formula")
        return {"parameter_count": count, "parameter_sha256": state_sha256(params)}
    if args.worker == "pilot":
        from scaffolds.transformer.training.loop import run_training
        config = load_config(args.config)
        config.train.source_commit = experiment["source_commit"]
        config.train.workdir = args.directory / "pilot"
        config.train.checkpoint_dir = None
        config.train.wandb_mode = "disabled"
        config.train.eval_every = PILOT_STEPS
        config.train.eval_batches = 1
        # Preserve model, context, batch, seed, LR, warmup, and full step horizon.
        return run_training(config, resume=True, stop_after_step=args.stop_after_step, audit_checkpoint=True)
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
        for stage, paths in (("cpu-tests", CPU_TESTS), ("gpu-tests", GPU_TESTS)):
            path = directory / (stage + ".json")
            if passed(path, experiment):
                continue
            junit = directory / (stage + ".xml")
            env = dict(os.environ, JAX_PLATFORMS="cpu" if stage == "cpu-tests" else "cuda")
            if stage == "gpu-tests":
                # Kernel tests are single-device operations; the pilot below uses every GPU.
                env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            subprocess.run([
                sys.executable, "-m", "pytest", "-c", "scaffolds/transformer/pyproject.toml", *paths,
                "-q", "--import-mode=importlib", "-m", "not gpu" if stage == "cpu-tests" else "gpu",
                f"--junitxml={junit}",
            ], cwd=ROOT, env=env, check=True)
            record(path, experiment, tests_passed=require_passing_tests(junit))
        stage = "initialization"
        path = directory / (stage + ".json")
        if not passed(path, experiment):
            result = run_worker(args, "initialization", directory / "initialization-result.json", cpu=True)
            record(path, experiment, result=result)
        stage = "pilot"
        path = directory / (stage + ".json")
        if not passed(path, experiment):
            # A preempted disposable pilot may have saved its checkpoint without a
            # timing result; restart incomplete pilots. Passed checks survive.
            if (directory / "pilot").exists():
                shutil.rmtree(directory / "pilot")
            outcome = run_worker(args, "pilot", directory / "pilot-result.json")
            if outcome["final_step"] != PILOT_STEPS or outcome["data_parallel_devices"] != args.gpu_count:
                raise RuntimeError(f"Incomplete full-recipe pilot: {outcome}")
            if outcome["attention_backend"] != "fused":
                raise RuntimeError(f"Production requires the fused GLA kernel; resolved {outcome['attention_backend']}")
            records = [json.loads(line) for line in (directory / "pilot/metrics.jsonl").read_text().splitlines()]
            validation = next(row for row in records if row["step"] == PILOT_STEPS)
            if not math.isfinite(validation["validation_nll"]) or not validation["attention_diagnostics"]["valid"]:
                raise RuntimeError("Pilot validation or attention diagnostics failed")
            timing = outcome["timing"]
            steady = timing["train_step_seconds"][1:]
            if not steady or any(not math.isfinite(x) or x <= 0 for x in steady):
                raise RuntimeError("Pilot has no valid synchronized steady update timing")
            config = experiment["config"]
            total_steps = config["optim"]["total_steps"]
            evaluations = total_steps // config["train"]["eval_every"] + 1
            checkpoints = total_steps // config["train"]["ckpt_every"] + 1
            # Include compilation once plus conservative evaluation/checkpoint costs.
            projected = (timing["runtime_seconds"] + max(steady) * total_steps
                         + max(timing["evaluation_seconds"]) * config["train"]["eval_batches"] * evaluations
                         + max(timing["checkpoint_seconds"]) * checkpoints)
            if projected > (ALLOCATION_HOURS - 2) * 3600:
                raise RuntimeError(f"Pilot projects beyond a {ALLOCATION_HOURS}h allocation: {projected / 3600:.2f} hours")
            record(path, experiment, outcome=outcome, validation=validation, projected_hours=projected / 3600)
        stage = "resume"
        path = directory / (stage + ".json")
        if not passed(path, experiment):
            outcome = run_worker(args, "pilot", directory / "resume-result.json", stop=PILOT_STEPS + 1)
            # A retry after the resumed checkpoint was committed can already be at the next step.
            if (outcome["restored_step"] not in (PILOT_STEPS, PILOT_STEPS + 1) or outcome["final_step"] != PILOT_STEPS + 1
                    or outcome["data_start_sequence"] != outcome["restored_step"] * experiment["config"]["data"]["batch_size"]):
                raise RuntimeError(f"Pilot resume did not preserve optimizer and data position: {outcome}")
            if outcome["restored_step"] == PILOT_STEPS:
                validate_resume_audit(json.loads((directory / "pilot.json").read_text())["outcome"], outcome)
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
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", choices=("hardware", "initialization", "pilot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int, default=PILOT_STEPS)
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
