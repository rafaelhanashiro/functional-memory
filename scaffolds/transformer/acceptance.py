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
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from scaffolds.transformer.configs import config_to_dict, load_config, validate_config

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ("cpu-tests", "cpu-distributed-tests", "gpu-tests", "initialization", "pilot", "resume")
# Test stages: name, pytest selection, and whether the tests run on CPU. Distributed
# CPU tests need several JAX devices, so only that stage forces two host devices.
TEST_STAGES = (
    ("cpu-tests", ("-m", "not gpu", "-k", "not distributed"), True),
    ("cpu-distributed-tests", ("-m", "not gpu", "-k", "distributed"), True),
    ("gpu-tests", ("-m", "gpu"), False),
)
NO_TESTS_COLLECTED = 5
RUNTIME_KEYS = ("NCCL_P2P_DISABLE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_FLAGS", "CUDA_HOME")
NORMALIZED_POLYNOMIAL_TESTS = "scaffolds/transformer/tests/test_normalized_polynomial_fused_attention.py"


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
    select_tests: Callable | None = None
    """Optional recipe/backend selector called as ``(config, resolved_backend, cpu)``."""


def test_paths(spec: AcceptanceSpec, config: dict, *, resolved_backend: str, cpu: bool) -> tuple[str, ...]:
    """Include the resolved mechanism's required checks even if an adapter omits them."""
    paths = list(spec.cpu_tests if cpu else spec.gpu_tests)
    if spec.select_tests is not None:
        paths.extend(spec.select_tests(config, resolved_backend, cpu))
    model = config["model"]
    if (resolved_backend == "fused" and model.get("attention_mechanism", "dot_product") == "dot_product"
            and model.get("attention_normalization", True)
            and (model.get("attention_coefficients") == "learned"
                 or model.get("attention_fixed_coefficients") is not None
                 or model.get("attention_fixed_square_factors") is not None)):
        paths.append(NORMALIZED_POLYNOMIAL_TESTS)
    return tuple(dict.fromkeys(paths))


def attention_test_paths(config: dict, resolved_backend: str, cpu: bool) -> tuple[str, ...]:
    """Qualify the recipe's mechanism without depending on study names or past sweeps."""
    model = config["model"]
    fused = resolved_backend == "fused"
    tests = []
    if model.get("attention_mechanism", "dot_product") == "gla":
        tests = ["gla", "gla_bridge", "gla_integration"] if cpu else (["gla_kernel"] if fused else [])
    else:
        degree = model.get("attention_degree")
        normalized = model.get("attention_normalization", True)
        learned = model.get("attention_coefficients", "fixed") == "learned"
        quadratic = model.get("quadratic_mode", "fixed") != "fixed"
        fixed = (model.get("attention_fixed_coefficients") is not None
                 or model.get("attention_fixed_square_factors") is not None)
        square = model.get("attention_coefficient_parameterization", "direct") == "sum_of_squares"
        if cpu:
            tests.append("attention")
        if model.get("qk_norm") and (cpu or resolved_backend in ("fused", "cudnn")):
            tests.append("qk_norm")
        if quadratic:
            tests.extend(["learned_quadratic", "learned_training"] if cpu else
                         (["learned_fused_attention"] if fused else []))
        elif learned:
            if cpu:
                tests.extend(["learned_polynomial", "learned_polynomial_training"])
            if normalized:
                if cpu or resolved_backend == "blocked":
                    tests.append("learned_polynomial_normalized")
                    if square:
                        tests.append("learned_polynomial_nonnegative")
                if fused:
                    tests.append("normalized_polynomial_fused_attention")
            elif fused:
                tests.append("learned_polynomial_fused_attention")
        elif fixed:
            if cpu:
                tests.extend(["fixed_polynomial", "fixed_polynomial_config"])
                if square:
                    tests.append("fixed_square_factors")
            elif fused:
                tests.append("fixed_square_factors_fused" if square else "fixed_polynomial_fused")
            if model.get("qk_norm") and (cpu or fused):
                tests.append("fixed_polynomial_qk_norm")
        elif normalized:
            if degree is None:
                if cpu:
                    tests.append("softmax_backends")
                elif resolved_backend in ("fused", "cudnn"):
                    # This check exercises the production width without requiring
                    # unrelated width-256 cuDNN support from the allocation.
                    tests.append("attention.py::test_gpu_exact_control_matches_original_cudnn")
            elif degree % 2:
                tests.extend(["odd_attention", "odd_training"] if cpu else (["odd_attention"] if fused else []))
            elif fused:
                tests.extend(["fused_attention", "taylor_kernel_widths"])
        if not normalized:
            tests.extend(["unnormalized_attention", "unnormalized_training"] if cpu else
                         (["unnormalized_fused_attention"] if fused and not learned and not fixed else []))
    return tuple(f"scaffolds/transformer/tests/test_{name}" + ("" if ".py::" in name else ".py") for name in tests)


def attention_spec(config, *, entry_module: str) -> AcceptanceSpec:
    """Shared acceptance for a Transformer recipe across all attention families."""
    validate_config(config)
    runtime_env = {}
    if config.model.attention_mechanism == "gla":
        # The custom-call bridge executes outside XLA command buffers.
        flags = os.environ.get("XLA_FLAGS", "").split()
        flags = [flag for flag in flags if not flag.startswith("--xla_gpu_enable_command_buffer=")]
        runtime_env = {"XLA_FLAGS": " ".join([*flags, "--xla_gpu_enable_command_buffer="]),
                       "EJKERNEL_TRITON_CACHE_DIR": ".cache/transformer/triton"}
    request = config.model.attention_backend
    return AcceptanceSpec(
        entry_module=entry_module,
        cpu_tests=tuple(f"scaffolds/transformer/tests/test_{name}.py" for name in (
            "model", "training", "checkpoint_audit", "parameter_policy")),
        check_config=validate_config,
        required_backend=None if request == "auto" else request,
        check_initialization=(None if config.model.attention_mechanism == "gla"
                              else matches_softmax_backbone_initialization),
        runtime_env=runtime_env,
        select_tests=attention_test_paths,
    )


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
    control = replace(config, model=replace(config.model, attention_degree=None, attention_normalization=True,
                                           attention_fixed_coefficients=None, attention_fixed_square_factors=None,
                                           attention_coefficient_parameterization="direct"))
    expected = initialization_facts(control)
    if actual != expected:
        raise RuntimeError(f"Initialization differs from the exact-softmax counterpart: {actual} != {expected}")
    return {"softmax_counterpart_parameter_sha256": expected["parameter_sha256"]}


def matches_softmax_backbone_initialization(config) -> dict:
    """Require the exact-softmax control's backbone weights plus declared initial mechanism parameters.

    Mechanisms that add trainable attention arrays register them without drawing
    random numbers, so every other parameter must equal the softmax control's and
    each added array must hold its declared initial value.
    """
    from dataclasses import replace

    import jax
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.attention.parameters import attention_parameter_specs
    from scaffolds.transformer.models import build_model

    def flat_parameters(current):
        current.model.attention_backend = "blocked"
        rng = jax.random.PRNGKey(current.train.seed)
        model = build_model(current.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}

    specs = attention_parameter_specs(config.model)
    control = replace(config, model=replace(
        config.model, attention_degree=None, attention_normalization=True, quadratic_mode="fixed",
        attention_coefficients="fixed", attention_fixed_coefficients=None,
        attention_fixed_square_factors=None,
        attention_coefficient_parameterization="direct", qk_norm=False))
    actual, expected = flat_parameters(replace(config, model=replace(config.model))), flat_parameters(control)
    added = {(spec.name,) for spec in specs}
    if set(actual) - set(expected) != added or set(expected) - set(actual):
        raise RuntimeError(f"Unexpected parameter names beyond the softmax control: {sorted(set(actual) ^ set(expected))}")
    for path, value in expected.items():
        if not np.array_equal(actual[path], value):
            raise RuntimeError(f"Backbone parameter {path} differs from the exact-softmax control")
    for spec in specs:
        initial = np.asarray(spec.initial, dtype=np.float32)
        if not np.array_equal(actual[(spec.name,)], np.broadcast_to(initial, actual[(spec.name,)].shape)):
            raise RuntimeError(f"{spec.name} does not hold its declared initial value")
    return {
        "softmax_counterpart_parameter_sha256": initialization_facts(control)["parameter_sha256"],
        "mechanism_parameters": {spec.name: {"shape": list(actual[(spec.name,)].shape), "initial": list(spec.initial),
                                             "weight_decay": spec.weight_decay} for spec in specs},
    }


def entry_module(spec: AcceptanceSpec) -> str:
    """The adapter's importable module name, also when it runs as `python -m` (`__name__ == "__main__"`)."""
    if spec.entry_module != "__main__":
        return spec.entry_module
    main_spec = getattr(sys.modules["__main__"], "__spec__", None)
    if main_spec is None:
        raise RuntimeError("Run the adapter's acceptance with python -m <module>")
    return main_spec.name


def _run_worker(spec, args, name, output, *, stop, cpu=False) -> dict:
    command = [sys.executable, "-m", entry_module(spec), str(args.config), "--directory", str(args.directory),
               "--gpu-count", str(args.gpu_count), "--gpu-type", args.gpu_type, "--worker", name,
               "--output", str(output), "--stop-after-step", str(stop)]
    subprocess.run(command, cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda"), check=True)
    return json.loads(Path(output).read_text())


def _worker(spec: AcceptanceSpec, args, experiment: dict) -> dict:
    if args.worker == "hardware":
        import jax
        from importlib.metadata import version
        from scaffolds.transformer.attention.backends.selection import resolve_attention_backend

        devices = jax.local_devices()
        if (jax.default_backend() != "gpu" or len(devices) != args.gpu_count
                or any(args.gpu_type.upper() not in d.device_kind.upper() for d in devices)):
            raise RuntimeError(f"Expected {args.gpu_count} {args.gpu_type} GPUs; got {devices}")
        packages = ("jax", "flax", "optax", "orbax-checkpoint")
        config = load_config(args.config)
        return {"devices": [str(d) for d in devices], "device_kinds": [d.device_kind for d in devices],
                "versions": {name: version(name) for name in packages},
                "attention_backend": resolve_attention_backend(
                    config.model, dtype=config.train.dtype, sequence_length=config.data.sequence_length - 1)}
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
        for stage, selection, cpu in TEST_STAGES:
            path = directory / f"{stage}.json"
            if passed(path, experiment):
                continue
            paths = test_paths(spec, experiment["config"], resolved_backend=hardware["attention_backend"], cpu=cpu)
            if not paths:
                record(path, experiment, tests_passed=0, note="no tests declared")
                continue
            junit = directory / f"{stage}.xml"
            env = dict(os.environ, JAX_PLATFORMS="cpu" if cpu else "cuda")
            if stage == "cpu-distributed-tests":
                env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=2".strip()
            if not cpu:
                # Kernel tests are single-device operations; the pilot uses every GPU.
                env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            result = subprocess.run([sys.executable, "-m", "pytest", "-c", "scaffolds/transformer/pyproject.toml", *paths,
                                     "-q", "--import-mode=importlib", *selection, f"--junitxml={junit}"], cwd=ROOT, env=env)
            if result.returncode == NO_TESTS_COLLECTED and stage == "cpu-distributed-tests":
                record(path, experiment, tests_passed=0, note="no distributed tests among the declared CPU tests")
                continue
            if result.returncode != 0:
                raise RuntimeError(f"Acceptance {stage} failed with pytest exit code {result.returncode}")
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
                   "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "failed_utc": datetime.now(UTC).isoformat()})
        raise


def main(spec: AcceptanceSpec | Callable, argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--worker", choices=("hardware", "initialization", "pilot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--runtime-env", action="store_true", help="Print shell exports for the launcher and exit")
    args = parser.parse_args(argv)
    if not isinstance(spec, AcceptanceSpec):
        if args.config is None:
            parser.error("config is required to select attention acceptance checks and runtime settings")
        spec = spec(load_config(args.config))
    if args.stop_after_step is None:
        args.stop_after_step = spec.pilot_steps
    if args.runtime_env:
        for key, value in spec.runtime_env.items():
            print(f"export {key}={shlex.quote(value)}")
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
