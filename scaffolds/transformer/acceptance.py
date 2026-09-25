"""Identity-keyed acceptance of one production allocation before training.

An experiment adapter describes its checks in an :class:`AcceptanceSpec` and
calls :func:`main` from its own execution module. Acceptance then runs, in the
production allocation and keyed to the source, recipe, dependency lock, data
manifests, GPU type and count, and runtime settings:

1. CPU tests and GPU tests (every case must pass; skips fail acceptance);
2. an initialization check (parameter count and hash, plus any adapter check);
3. a full-recipe pilot with validation and valid attention diagnostics;
4. an exact fresh-process resume (state, next batch, and RNG hashes).

Matching passed stages are reused on requeue. Worker processes re-invoke the
adapter module by name, so scaffold code never imports experiment code.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from scaffolds.transformer.configs import config_to_dict, load_config

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ("cpu-tests", "gpu-tests", "initialization", "pilot", "resume")
RUNTIME_KEYS = ("NCCL_P2P_DISABLE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_FLAGS", "CUDA_HOME")


@dataclass(frozen=True)
class AcceptanceSpec:
    """What one experiment requires of an allocation before production."""

    entry_module: str
    """Adapter module whose ``main()`` calls :func:`main`; workers re-invoke it."""
    cpu_tests: tuple[str, ...]
    gpu_tests: tuple[str, ...] = ()
    check_config: Callable | None = None
    """Raises when a recipe is not one this experiment runs."""
    required_backend: str | None = None
    """Resolved attention backend the pilot must use (for example ``"fused"``)."""
    check_initialization: Callable | None = None
    """Called with the loaded config after the count/hash check; returns extra facts."""
    runtime_env: dict[str, str] = field(default_factory=dict)
    """Environment the launcher exports for this experiment (for example XLA flags)."""
    pilot_steps: int = 20
    allocation_hours: int = 48


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity() -> dict:
    """Execution choices that qualification depends on, separate from the recipe."""
    host_staging = os.environ.get("TRANSFORMER_HOST_STAGING", "0")
    if host_staging not in ("0", "1"):
        raise ValueError("TRANSFORMER_HOST_STAGING must be resolved to 0 or 1 before acceptance")
    return {"TRANSFORMER_HOST_STAGING": host_staging, **{key: os.environ.get(key) for key in RUNTIME_KEYS}}


def identity(spec: AcceptanceSpec, config_path: Path, gpu_count: int, gpu_type: str) -> dict:
    # The launcher exports its launch commit; analysis commits may have advanced HEAD since.
    source = os.environ.get("SHARED_TRANSFORMER_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = load_config(config_path)
    if config.train.source_commit not in ("auto", source):
        raise ValueError("Configuration source differs from the execution checkout")
    config.train.source_commit = source
    if spec.check_config is not None:
        spec.check_config(config)
    payload = {
        "source_commit": source,
        "config": config_to_dict(config),
        "config_sha256": sha256(config_path),
        "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "data_manifests": {split: {"path": path, "sha256": sha256(path)} for split, path in
                           (("train", config.data.train_manifest), ("eval", config.data.eval_manifest))},
        "gpu_count": gpu_count, "gpu_type": gpu_type.upper(),
        "runtime": runtime_identity(),
    }
    payload["identity"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def passed(path: Path, experiment: dict) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return value.get("identity") == experiment["identity"] and value.get("status") == "passed"


def record(path: Path, experiment: dict, **fields) -> None:
    write_json(path, {
        "identity": experiment["identity"], "source_commit": experiment["source_commit"],
        "status": "passed", "completed_utc": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"), **fields,
    })


def require_passing_tests(path: Path) -> int:
    cases = list(ET.parse(path).iter("testcase"))
    failed = [case for case in cases if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    if not cases or failed:
        raise RuntimeError(f"Acceptance tests must all pass with no skips: {len(cases)} cases, {len(failed)} failures/skips")
    return len(cases)


def validate_resume_audit(initial: dict, resumed: dict) -> None:
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


def initialization_facts(config) -> dict:
    """Parameter count and hash of the initialization training will use."""
    import jax
    from flax import nnx

    from scaffolds.transformer.models import build_model, model_parameter_count
    from scaffolds.transformer.training.checkpointing import state_sha256

    config.model.attention_backend = "blocked"  # Initialization does not depend on the backend.
    rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
    params = nnx.state(build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0))), nnx.Param)
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    if count != model_parameter_count(config.model):
        raise RuntimeError(f"Initialized parameter count {count} differs from the model formula")
    return {"parameter_count": count, "parameter_sha256": state_sha256(params)}


def matches_softmax_initialization(config) -> dict:
    """Require the initialization of the recipe's exact-softmax counterpart.

    Attention kernels and normalization own no parameters, so a mechanism that
    changes only the kernel must start from the softmax control's exact weights.
    """
    from dataclasses import replace

    actual = initialization_facts(config)
    control = replace(config, model=replace(config.model, attention_degree=None, attention_normalization=True))
    expected = initialization_facts(control)
    if actual != expected:
        raise RuntimeError(f"Initialization differs from the exact-softmax counterpart: {actual} != {expected}")
    return {"softmax_counterpart_parameter_sha256": expected["parameter_sha256"]}


def _run_worker(spec, args, name, output, *, stop, cpu=False) -> dict:
    command = [sys.executable, "-m", spec.entry_module, str(args.config), "--directory", str(args.directory),
               "--gpu-count", str(args.gpu_count), "--gpu-type", args.gpu_type, "--worker", name,
               "--output", str(output), "--stop-after-step", str(stop)]
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda"), check=True)
    return json.loads(Path(output).read_text())


def _worker(spec: AcceptanceSpec, args, experiment: dict) -> dict:
    if args.worker == "hardware":
        import jax
        from importlib.metadata import version
        devices = jax.local_devices()
        if (jax.default_backend() != "gpu" or len(devices) != args.gpu_count
                or any(args.gpu_type.upper() not in d.device_kind.upper() for d in devices)):
            raise RuntimeError(f"Expected {args.gpu_count} {args.gpu_type} GPUs; got {devices}")
        packages = ("jax", "flax", "optax", "orbax-checkpoint")
        return {"devices": [str(d) for d in devices], "device_kinds": [d.device_kind for d in devices],
                "versions": {name: version(name) for name in packages}}
    if args.worker == "initialization":
        config = load_config(args.config)
        facts = initialization_facts(config)
        if spec.check_initialization is not None:
            facts["experiment_check"] = spec.check_initialization(load_config(args.config))
        return facts
    if args.worker == "pilot":
        from scaffolds.transformer.training.loop import run_training
        config = load_config(args.config)
        config.train.source_commit = experiment["source_commit"]
        config.train.workdir = args.directory / "pilot"
        config.train.checkpoint_dir = None
        config.train.wandb_mode = "disabled"
        config.train.eval_every = spec.pilot_steps
        config.train.eval_batches = 1
        # Model, context, batch, seed, LR schedule, and step horizon are the production recipe's.
        return run_training(config, resume=True, stop_after_step=args.stop_after_step, audit_checkpoint=True)
    raise ValueError(args.worker)


def _run(spec: AcceptanceSpec, args, experiment: dict) -> None:
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "identity.json", experiment)
    hardware = _run_worker(spec, args, "hardware", directory / "hardware.json", stop=spec.pilot_steps)
    if passed(directory / "success.json", experiment) and all(passed(directory / f"{name}.json", experiment) for name in CHECKS):
        print(f"matching_acceptance_passed={directory / 'success.json'}", flush=True)
        return
    stage = "cpu-tests"
    try:
        for stage, paths in (("cpu-tests", spec.cpu_tests), ("gpu-tests", spec.gpu_tests)):
            path = directory / f"{stage}.json"
            if passed(path, experiment):
                continue
            if not paths:
                record(path, experiment, tests_passed=0, note="no tests declared")
                continue
            junit = directory / f"{stage}.xml"
            env = dict(os.environ, JAX_PLATFORMS="cpu" if stage == "cpu-tests" else "cuda")
            if stage == "gpu-tests":
                # Kernel tests are single-device operations; the pilot uses every GPU.
                env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            subprocess.run([sys.executable, "-m", "pytest", "-c", "scaffolds/transformer/pyproject.toml", *paths,
                            "-q", "--import-mode=importlib", "-m", "not gpu" if stage == "cpu-tests" else "gpu",
                            f"--junitxml={junit}"], cwd=ROOT, env=env, check=True)
            record(path, experiment, tests_passed=require_passing_tests(junit))
        stage = "initialization"
        path = directory / f"{stage}.json"
        if not passed(path, experiment):
            result = _run_worker(spec, args, stage, directory / "initialization-result.json", stop=spec.pilot_steps, cpu=True)
            record(path, experiment, result=result)
        stage = "pilot"
        path = directory / f"{stage}.json"
        if not passed(path, experiment):
            # A preempted pilot may have saved a checkpoint without timing; restart it.
            if (directory / "pilot").exists():
                shutil.rmtree(directory / "pilot")
            outcome = _run_worker(spec, args, "pilot", directory / "pilot-result.json", stop=spec.pilot_steps)
            if outcome["final_step"] != spec.pilot_steps or outcome["data_parallel_devices"] != args.gpu_count:
                raise RuntimeError(f"Incomplete full-recipe pilot: {outcome}")
            if spec.required_backend is not None and outcome["attention_backend"] != spec.required_backend:
                raise RuntimeError(f"Production requires backend {spec.required_backend!r}; resolved {outcome['attention_backend']!r}")
            records = [json.loads(line) for line in (directory / "pilot/metrics.jsonl").read_text().splitlines()]
            validation = next(row for row in records if row["step"] == spec.pilot_steps)
            if not math.isfinite(validation["validation_nll"]) or not validation["attention_diagnostics"]["valid"]:
                raise RuntimeError("Pilot validation or attention diagnostics failed")
            timing = outcome["timing"]
            steady = timing["train_step_seconds"][1:]
            if not steady or any(not math.isfinite(x) or x <= 0 for x in steady):
                raise RuntimeError("Pilot has no valid synchronized steady update timing")
            config = experiment["config"]
            total = config["optim"]["total_steps"]
            evaluations = total // config["train"]["eval_every"] + 1
            checkpoints = total // config["train"]["ckpt_every"] + 1
            # Compilation once plus conservative evaluation and checkpoint costs.
            projected = (timing["runtime_seconds"] + max(steady) * total
                         + max(timing["evaluation_seconds"]) * config["train"]["eval_batches"] * evaluations
                         + max(timing["checkpoint_seconds"]) * checkpoints)
            if projected > (spec.allocation_hours - 2) * 3600:
                raise RuntimeError(f"Pilot projects beyond a {spec.allocation_hours}h allocation: {projected / 3600:.2f} hours")
            record(path, experiment, outcome=outcome, validation=validation, projected_hours=projected / 3600)
        stage = "resume"
        path = directory / f"{stage}.json"
        if not passed(path, experiment):
            outcome = _run_worker(spec, args, "pilot", directory / "resume-result.json", stop=spec.pilot_steps + 1)
            restored = outcome["restored_step"]
            # A retry after the resumed checkpoint was committed can already be at the next step.
            if (restored not in (spec.pilot_steps, spec.pilot_steps + 1) or outcome["final_step"] != spec.pilot_steps + 1
                    or outcome["data_start_sequence"] != restored * experiment["config"]["data"]["batch_size"]):
                raise RuntimeError(f"Pilot resume did not preserve optimizer and data position: {outcome}")
            if restored == spec.pilot_steps:
                validate_resume_audit(json.loads((directory / "pilot.json").read_text())["outcome"], outcome)
            record(path, experiment, outcome=outcome)
        record(directory / "success.json", experiment, checks=list(CHECKS), hardware=hardware)
        shutil.rmtree(directory / "pilot")  # The disposable pilot checkpoint is no longer needed.
        print(f"acceptance_success={directory / 'success.json'}", flush=True)
    except BaseException as error:
        write_json(directory / "failure.json", {"identity": experiment["identity"], "stage": stage,
                   "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise


def main(spec: AcceptanceSpec, argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--worker", choices=("hardware", "initialization", "pilot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int, default=spec.pilot_steps)
    parser.add_argument("--runtime-env", action="store_true", help="Print shell exports for the launcher and exit")
    args = parser.parse_args(argv)
    if args.runtime_env:
        for key, value in spec.runtime_env.items():
            print(f"export {key}={json.dumps(value)}")
        return
    if args.config is None or args.directory is None:
        parser.error("config and --directory are required")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run acceptance inside a Slurm allocation")
    args.config, args.directory = args.config.resolve(), args.directory.resolve()
    experiment = identity(spec, args.config, args.gpu_count, args.gpu_type)
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        write_json(args.output, _worker(spec, args, experiment))
    else:
        _run(spec, args, experiment)
