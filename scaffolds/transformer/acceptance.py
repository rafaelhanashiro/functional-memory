"""Identity-keyed acceptance of one production allocation before training.

An experiment adapter describes its checks in an :class:`AcceptanceSpec` and
calls :func:`main` from its own execution module. Acceptance then runs, in the
production allocation and keyed to the source, recipe, dependency lock, data
manifests, GPU model and count, and runtime settings:

1. CPU tests and GPU tests (every case must pass; skips fail acceptance);
2. an initialization check (parameter count and hash, plus any adapter check);
3. a full-recipe pilot with validation and valid attention diagnostics;
4. an exact fresh-process resume (state, next batch, and RNG hashes).

Matching passed stages are reused on requeue; a requeue onto another GPU model
(for example H100 NVL after H100 80GB HBM3, both the launcher's ``H100``) runs
them again. With ``--test-cache``, a passed test stage is also shared with other
runs: the recipe only selects which tests run, so runs whose selected tests,
source, dependency lock, runtime settings, and (for GPU tests) GPU model match
reuse one result. Each run still checks its own initialization, pilot, and
resume. Worker processes re-invoke the adapter module by name, so scaffold code
never imports experiment code.
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
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.training.cadence import evaluation_due, scheduled_work_counts

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
                "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_FLAGS", "CUDA_HOME", "TRANSFORMER_CUDA_ENVIRONMENT")
NORMALIZED_POLYNOMIAL_TESTS = "scaffolds/transformer/tests/test_normalized_polynomial_fused_attention.py"
INPUT_DEPENDENT_COEFFICIENT_TESTS = "scaffolds/transformer/tests/test_input_dependent_coefficients.py"
QUERY_TANH_POLYNOMIAL_TESTS = "scaffolds/transformer/tests/test_query_tanh_polynomial.py"
DENSE_QUERY_READOUT_TESTS = "scaffolds/transformer/tests/test_dense_query_readout.py"
SIGMOID_COEFFICIENT_TESTS = "scaffolds/transformer/tests/test_sigmoid_coefficients.py"
COEFFICIENT_FORM_TESTS = "scaffolds/transformer/tests/test_coefficient_forms.py"
DENSE_MIXING_SCALE_TESTS = "scaffolds/transformer/tests/test_dense_mixing_scale.py"
QK_HEAD_SCALE_TESTS = "scaffolds/transformer/tests/test_qk_head_scale.py"
QK_HEAD_SCALE_BACKEND_TESTS = (
    "scaffolds/transformer/tests/test_qk_rms_attention_backends.py::test_full_context_qk_rms_head_scale_backend")
QUERY_KEY_HEAD_SCALE_DEGREES = (4, 8, 12)
COEFFICIENT_SCALE_TESTS = "scaffolds/transformer/tests/test_coefficient_scale.py"
PATH_ATTENTION_TESTS = "scaffolds/transformer/tests/test_path_attention.py"
PATH_ATTENTION_FUSED_TESTS = "scaffolds/transformer/tests/test_path_attention_fused.py"
TRITON_PORT_TESTS = "scaffolds/transformer/tests/test_triton_port.py"
QUERY_KEY_POLYNOMIAL_TESTS = "scaffolds/transformer/tests/test_query_key_polynomial.py"
QUERY_KEY_TILING_TESTS = "scaffolds/transformer/tests/test_query_key_tiling.py"
TANH_WRITE_GDN_TESTS = "scaffolds/transformer/tests/test_gdn_tanh_write.py"
TANH_WRITE_GDN_KERNEL_TESTS = "scaffolds/transformer/tests/test_gdn_tanh_write_kernel.py"
POLYNOMIAL_GDN_TESTS = "scaffolds/transformer/tests/test_polynomial_gdn.py"
POLYNOMIAL_GDN_INTEGRATION_TESTS = "scaffolds/transformer/tests/test_polynomial_gdn_integration.py"
POLYNOMIAL_GDN_KERNEL_TESTS = "scaffolds/transformer/tests/test_polynomial_gdn_kernel.py"
CUBIC_GDN_TESTS = "scaffolds/transformer/tests/test_cubic_gdn.py"
CUBIC_GDN_KERNEL_TESTS = "scaffolds/transformer/tests/test_cubic_gdn_kernel.py"


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
    """Qualification horizon; launchers that require one allocation can override it at runtime."""
    select_tests: Callable | None = None
    """Optional recipe/backend selector called as ``(config, resolved_backend, cpu)``."""


def test_paths(spec: AcceptanceSpec, config: dict, *, resolved_backend: str, cpu: bool,
               runtime_test_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Include the resolved mechanism's required checks even if an adapter omits them."""
    paths = list(spec.cpu_tests if cpu else spec.gpu_tests)
    if spec.select_tests is not None:
        paths.extend(spec.select_tests(config, resolved_backend, cpu))
    if not cpu:
        paths.extend(runtime_test_paths)
    if cpu and config.get("continuation", {}).get("parent_checkpoint"):
        paths.extend(("scaffolds/transformer/tests/test_continuation.py",
                      "scaffolds/transformer/tests/test_continuation_acceptance.py"))
    model = config["model"]
    if cpu and model.get("positional_encoding", "rope") == "none":
        paths.append("scaffolds/transformer/tests/test_positional_encoding_none.py")
    if (cpu and model.get("attention_forget_gate", False)
            and model.get("attention_forget_gate_form", "input") == "constant"):
        paths.append("scaffolds/transformer/tests/test_constant_decay.py")
    if (model.get("attention_coefficient_readout", "diagonal") == "dense"
            and (model.get("qk_norm_head_scale", False) or model.get("attention_coefficient_readout_scale", 1.0) != 1.0)
            and (cpu or resolved_backend == "mosaic")):
        paths.append(DENSE_MIXING_SCALE_TESTS)
    if cpu and model.get("qk_norm_head_scale", False):
        paths.append(QK_HEAD_SCALE_TESTS)
    if (not cpu and model.get("qk_norm_head_scale", False)
            and model.get("attention_mechanism", "dot_product") == "dot_product"
            and not model.get("attention_normalization", True)
            and model.get("attention_coefficients", "fixed") == "fixed"
            and model.get("attention_fixed_coefficients") is None
            and model.get("attention_fixed_square_factors") is None
            and (model.get("attention_degree"), resolved_backend) in ((None, "fused"), (4, "mosaic"))):
        degree = model.get("attention_degree")
        family = "exp" if degree is None else "fixed"
        paths.append(f"{QK_HEAD_SCALE_BACKEND_TESTS}[{family}-{degree or 0}-{resolved_backend}]")
    if (not cpu and model.get("qk_norm_head_scale", False)
            and model.get("attention_mechanism", "dot_product") == "dot_product"
            and not model.get("attention_normalization", True)
            and model.get("attention_coefficients", "fixed") == "learned"
            and model.get("attention_coefficient_conditioning", "static") == "query_key"
            and model.get("attention_coefficient_activation", "identity") == "tanh"
            and resolved_backend == "fused"):
        # Every qualified degree runs together, so runs at different degrees share one
        # GPU stage and a failure surfaces before any of them trains. A degree without
        # a parametrized case has no node id, so acceptance fails closed.
        degrees = sorted({*QUERY_KEY_HEAD_SCALE_DEGREES, int(model["attention_degree"])})
        paths.extend(f"{QK_HEAD_SCALE_BACKEND_TESTS}[query_key-{degree}-fused]" for degree in degrees)
    if model.get("positional_encoding", "rope") == "path":
        paths.append(PATH_ATTENTION_TESTS)
        if resolved_backend == "fused":
            paths.extend((PATH_ATTENTION_FUSED_TESTS, *((TRITON_PORT_TESTS,) if cpu else ())))
    if (model.get("attention_mechanism") == "gated_deltanet"
            and model.get("gdn_write_activation", "sigmoid") == "tanh"):
        if cpu:
            paths.append(TANH_WRITE_GDN_TESTS)
        elif resolved_backend == "fused":
            paths.append(TANH_WRITE_GDN_KERNEL_TESTS)
    if (model.get("attention_mechanism") == "gated_deltanet"
            and model.get("gdn_polynomial_degree", 1) in (2, 3)):
        if cpu:
            paths.extend((POLYNOMIAL_GDN_TESTS, POLYNOMIAL_GDN_INTEGRATION_TESTS))
        elif resolved_backend == "fused":
            paths.append(POLYNOMIAL_GDN_KERNEL_TESTS)
        if model.get("gdn_polynomial_degree") == 3:
            if cpu:
                paths.append(CUBIC_GDN_TESTS)
            elif resolved_backend == "fused":
                paths.append(CUBIC_GDN_KERNEL_TESTS)
    conditioning = model.get("attention_coefficient_conditioning", "static")
    if conditioning in ("query", "query_key"):
        paths.append(INPUT_DEPENDENT_COEFFICIENT_TESTS)
        activation = model.get("attention_coefficient_activation", "identity")
        if activation in ("tanh", "sigmoid", "pure_tanh", "affine_sigmoid"):
            # Every nonlinear generator runs on the same query-coefficient kernels as tanh.
            paths.append(QUERY_TANH_POLYNOMIAL_TESTS)
        if activation == "tanh" and model.get("attention_coefficient_readout", "diagonal") == "dense":
            paths.append(DENSE_QUERY_READOUT_TESTS)
        if activation == "sigmoid":
            paths.append(SIGMOID_COEFFICIENT_TESTS)
        if activation in ("pure_tanh", "affine_sigmoid") or model.get("attention_forget_gate", False):
            paths.append(COEFFICIENT_FORM_TESTS)
        if conditioning == "query_key":
            paths.append(QUERY_KEY_POLYNOMIAL_TESTS)
            if resolved_backend == "fused" and cpu:
                paths.append(QUERY_KEY_TILING_TESTS)
    if model.get("attention_coefficient_scale", "none") != "none":
        paths.append(COEFFICIENT_SCALE_TESTS)
    if (resolved_backend == "fused" and model.get("attention_mechanism", "dot_product") == "dot_product"
            and model.get("attention_normalization", True)
            and (model.get("attention_coefficients") == "learned"
                 or model.get("attention_fixed_coefficients") is not None
                 or model.get("attention_fixed_square_factors") is not None)):
        paths.append(NORMALIZED_POLYNOMIAL_TESTS)
    return tuple(dict.fromkeys(paths))


def _runtime_test_paths(config, resolved_backend: str) -> tuple[str, ...]:
    """Require implementation-specific GPU checks only when that implementation is selected.

    The backend owns device and shape eligibility; acceptance does not duplicate
    its hardware policy. CPU selector checks are always included by test_paths.
    """
    if (resolved_backend == "fused" and config.model.attention_mechanism == "dot_product"
            and config.model.attention_coefficient_conditioning == "query_key"):
        import jax.numpy as jnp
        from scaffolds.transformer.attention.backends.fused_query_key_polynomial import _backward_tile_sizes

        tiles = _backward_tile_sizes(
            config.model.qk_head_dim, config.model.value_head_dim,
            degree=config.model.attention_degree, time=config.data.sequence_length - 1,
            dtype=jnp.dtype(config.train.dtype), interpret=False)
        if tiles == (32, 32):
            return (QUERY_KEY_TILING_TESTS,)
    return ()


def attention_test_paths(config: dict, resolved_backend: str, cpu: bool) -> tuple[str, ...]:
    """Qualify the recipe's mechanism without depending on study names or past sweeps."""
    model = config["model"]
    fused = resolved_backend == "fused"
    tests = []
    if model.get("attention_mechanism", "dot_product") == "gla":
        tests = ["gla", "gla_bridge", "gla_integration"] if cpu else (["gla_kernel"] if fused else [])
    elif model.get("attention_mechanism", "dot_product") == "gated_deltanet":
        tests = ["gated_deltanet"] if cpu else (["gated_deltanet_kernel"] if fused else [])
        if cpu and fused:
            tests.append("triton_bridge_precision")
    else:
        degree = model.get("attention_degree")
        normalized = model.get("attention_normalization", True)
        learned = model.get("attention_coefficients", "fixed") == "learned"
        quadratic = model.get("quadratic_mode", "fixed") != "fixed"
        fixed = (model.get("attention_fixed_coefficients") is not None
                 or model.get("attention_fixed_square_factors") is not None)
        parameterization = model.get("attention_coefficient_parameterization", "direct")
        square = parameterization == "sum_of_squares"
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
                    if parameterization == "rational":
                        tests.append("learned_polynomial_rational")
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
        if model.get("attention_output_scaling", "none") != "none" and cpu:
            tests.append("attention_output_scaling")
        if resolved_backend == "mosaic" and not cpu:
            query = model.get("attention_coefficient_conditioning", "static") in ("query", "query_key")
            kernel = "mosaic_query" if query else "mosaic_taylor" if normalized else "mosaic_unnormalized"
            tests.extend([kernel, "mosaic_repeatability"])
    if model.get("initialization", "flax_default") != "flax_default" and cpu:
        tests.append("initialization")
    return tuple(f"scaffolds/transformer/tests/test_{name}" + ("" if ".py::" in name else ".py") for name in tests)


def mosaic_test_options(config: dict, resolved_backend: str) -> tuple[str, ...]:
    """GPU tests of a Mosaic run cover only the weight precision the run uses."""
    model = config["model"]
    if resolved_backend != "mosaic" or model.get("attention_degree") is None:
        return ()
    from scaffolds.transformer.attention.backends.mosaic.api import default_precision

    family = "taylor" if model.get("attention_normalization", True) else "query"
    return ("--mosaic-precision", default_precision(family, model["attention_degree"]))


def attention_spec(config, *, entry_module: str) -> AcceptanceSpec:
    """Shared acceptance for a Transformer recipe across all attention families."""
    validate_config(config)
    runtime_env = {}
    if (config.model.attention_mechanism in ("gla", "gated_deltanet")
            or (config.model.positional_encoding == "path" and config.model.attention_backend in ("auto", "fused"))):
        # The custom-call bridge executes outside XLA command buffers.
        flags = os.environ.get("XLA_FLAGS", "").split()
        flags = [flag for flag in flags if not flag.startswith("--xla_gpu_enable_command_buffer=")]
        runtime_env = {"XLA_FLAGS": " ".join([*flags, "--xla_gpu_enable_command_buffer="]),
                       "EJKERNEL_TRITON_CACHE_DIR": ".cache/transformer/triton"}
    request = config.model.attention_backend
    return AcceptanceSpec(
        entry_module=entry_module,
        cpu_tests=tuple(f"scaffolds/transformer/tests/test_{name}.py" for name in (
            "model", "training", "telemetry", "checkpoint_audit", "parameter_policy",
            "query_coefficient_learning_rate", "runtime_timing")),
        check_config=validate_config,
        required_backend=None if request == "auto" else request,
        check_initialization=(None if (config.model.attention_mechanism in ("gla", "gated_deltanet")
                                      or continuation_start_step(config))
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


def allocation_seconds(spec: AcceptanceSpec) -> int:
    """Resolve the launcher's granted wall-clock budget without changing the recipe."""
    value = os.environ.get("TRANSFORMER_ALLOCATION_SECONDS")
    if value is not None:
        if not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise ValueError("TRANSFORMER_ALLOCATION_SECONDS must be a positive integer")
        return int(value)
    seconds = spec.allocation_hours * 3600
    if not math.isfinite(seconds) or seconds <= 0 or int(seconds) != seconds:
        raise ValueError("AcceptanceSpec.allocation_hours must yield a positive finite integer number of seconds")
    return int(seconds)


def require_allocation_budget(projected_seconds: float, limit_seconds: int) -> None:
    """Keep the existing two-hour allowance for acceptance and runtime overhead."""
    if not math.isfinite(projected_seconds) or projected_seconds < 0:
        raise RuntimeError("Pilot has no valid projected runtime")
    if projected_seconds > limit_seconds - 2 * 3600:
        raise RuntimeError(f"Pilot projects beyond a {limit_seconds / 3600:g}h allocation "
                           f"with a 2h reserve: {projected_seconds / 3600:.2f} hours")


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
        "allocation_seconds": allocation_seconds(spec),
    }
    return _sealed(payload)


def _sealed(payload: dict) -> dict:
    payload = {key: value for key, value in payload.items() if key != "identity"}
    return {**payload, "identity": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def on_devices(experiment: dict, device_kinds) -> dict:
    """Key acceptance to the allocated GPU model, not only the launcher's GPU type.

    One launcher GPU type (for example ``H100``) can cover models with different memory
    and speed (SXM and NVL), so a recipe that fits one may not fit the other.
    """
    return _sealed({**experiment, "device_kinds": sorted(set(device_kinds))})


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
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "allocation_seconds": experiment.get("allocation_seconds"), **fields,
    })


def stage_identity(experiment: dict, stage: str, paths, selection, *, cpu: bool) -> str:
    """What a test stage's outcome depends on; the recipe only chooses the tests.

    GPU tests also key on the allocated GPU model (see :func:`on_devices`).
    """
    payload = {
        "stage": stage, "tests": list(paths), "selection": list(selection),
        "source_commit": experiment["source_commit"],
        "dependency_lock_sha256": experiment["dependency_lock_sha256"],
        "runtime": experiment["runtime"], "gpu_type": None if cpu else experiment["gpu_type"],
        "device_kinds": None if cpu else experiment["device_kinds"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _replace_shared(path: Path, write) -> None:
    # Concurrent runs may publish the same key; a per-process temporary keeps each write whole.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(temporary)
    temporary.replace(path)


def share_tests(cache: Path, key: str, junit: Path, directory: Path) -> None:
    """Publish a passed stage's report for runs with the same test identity."""
    cache.mkdir(parents=True, exist_ok=True)
    _replace_shared(cache / f"{key}.xml", lambda path: shutil.copyfile(junit, path))
    entry = {"test_identity": key, "junit_sha256": sha256(junit), "acceptance_directory": str(directory),
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "completed_utc": datetime.now(UTC).isoformat()}
    _replace_shared(cache / f"{key}.json",
                    lambda path: path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n"))


def reuse_tests(cache: Path, key: str, junit: Path) -> dict | None:
    """Copy a matching passed report into this run's evidence and return its origin."""
    entry, report = cache / f"{key}.json", cache / f"{key}.xml"
    try:
        origin = json.loads(entry.read_text())
        shutil.copyfile(report, junit)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # A report replaced by a concurrent publisher no longer matches its entry; rerun instead.
    if origin.get("test_identity") != key or sha256(junit) != origin.get("junit_sha256"):
        return None
    return origin


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


def continuation_start_step(config) -> int:
    """The absolute update at which a fresh child starts; ordinary runs start at zero."""
    continuation = config.get("continuation", {}) if isinstance(config, dict) else config.continuation
    if isinstance(continuation, dict):
        return int(continuation["parent_step"]) if continuation.get("parent_checkpoint") else 0
    return int(continuation.parent_step) if continuation.parent_checkpoint else 0


def remaining_work_counts(config) -> tuple[int, int]:
    """Count the child's baseline and scheduled evaluations/checkpoint writes."""
    start = continuation_start_step(config)
    if start == 0:
        return scheduled_work_counts(config)
    total = config.optim.total_steps
    evaluations = {step for step in range(start + 1, total + 1) if evaluation_due(config, step)}
    checkpoints = evaluations | {
        step for step in range(start + 1, total + 1) if step % config.train.ckpt_every == 0
    } | {total}
    # A fresh continuation first evaluates and saves its restored baseline.
    return len(evaluations) + 1, len(checkpoints) + 1


def project_runtime(timing: dict, config) -> dict:
    """Project a full execution from separately warmed pilot measurements.

    The disposable pilot (including startup, warmups, and its training updates)
    is charged once. It does not replace any production updates. Diagnostics
    recur per validation; only evaluation batches scale with ``eval_batches``.
    """
    def samples(name: str) -> list[float]:
        values = timing.get(name)
        if (not isinstance(values, list) or not values
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in values)):
            raise RuntimeError(f"Pilot requires positive finite {name}; rerun its timing profile")
        return values

    pilot = timing.get("runtime_seconds")
    if (isinstance(pilot, bool) or not isinstance(pilot, (int, float))
            or not math.isfinite(pilot) or pilot <= 0):
        raise RuntimeError("Pilot requires a positive finite runtime_seconds")
    updates = samples("train_step_seconds")[1:]  # First update includes compilation.
    if not updates:
        raise RuntimeError("Pilot has no synchronized warm training updates")
    batches = samples("evaluation_batch_seconds")
    if len(batches) < 3:
        raise RuntimeError("Pilot requires at least three warm evaluation_batch_seconds samples")
    rates = {
        "training_update": statistics.median(updates),
        "diagnostics": statistics.median(samples("diagnostic_seconds")),
        "evaluation_batch": statistics.median(batches),
        "checkpoint": max(samples("checkpoint_seconds")),
    }
    evaluations, checkpoints = remaining_work_counts(config)
    counts = {
        "training_updates": config.optim.total_steps - continuation_start_step(config),
        "validation_events": evaluations,
        "batches_per_validation": config.train.eval_batches,
        "checkpoint_events": checkpoints,
    }
    if counts["training_updates"] <= 0:
        raise RuntimeError("Pilot runtime projection requires remaining training updates")
    components = {
        "pilot_overhead": float(pilot),
        "training": rates["training_update"] * counts["training_updates"],
        "diagnostics": rates["diagnostics"] * evaluations,
        "evaluation": rates["evaluation_batch"] * config.train.eval_batches * evaluations,
        "checkpointing": rates["checkpoint"] * checkpoints,
    }
    projected = sum(components.values())
    if not math.isfinite(projected):
        raise RuntimeError("Pilot has no finite projected runtime")
    return {"method": "warm_components_v1", "counts": counts, "seconds_per_unit": rates,
            "components_seconds": components, "projected_seconds": projected,
            "reserve_seconds": 2 * 3600}


def initialization_facts(config) -> dict:
    """Parameter count and hash of the initialization training will use."""
    import jax
    from flax import nnx

    from scaffolds.transformer.models import build_model, model_parameter_count
    from scaffolds.transformer.training.checkpointing import state_sha256

    config.model.attention_backend = "blocked"  # Initialization does not depend on the backend.
    continuation = None
    if continuation_start_step(config):
        from scaffolds.transformer.training.continuation import initialize_continuation_state

        state, continuation = initialize_continuation_state(config)
        params = state.params
        if int(state.step) != continuation_start_step(config):
            raise RuntimeError("Continuation initialization did not preserve the parent step")
        for kind in ("parameters", "optimizer"):
            if continuation[f"parent_{kind}_sha256"] != continuation[f"backbone_{kind}_sha256"]:
                raise RuntimeError(f"Continuation changed the parent's backbone {kind}")
        from scaffolds.transformer.training.optim import continuation_introduces_parameters

        if (continuation_introduces_parameters(config)
                and continuation.get("new_coefficient_optimizer_count") != 0):
            raise RuntimeError("New parameters must start with a fresh optimizer clock")
    else:
        rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
        params = nnx.state(build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0))), nnx.Param)
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    if count != model_parameter_count(config.model):
        raise RuntimeError(f"Initialized parameter count {count} differs from the model formula")
    return {"parameter_count": count, "parameter_sha256": state_sha256(params),
            **({"continuation": continuation} if continuation is not None else {})}


def matches_softmax_initialization(config) -> dict:
    """Require the initialization of the recipe's exact-softmax counterpart.

    Attention kernels and normalization own no parameters, so a mechanism that
    changes only the kernel must start from the softmax control's exact weights.
    """
    from dataclasses import replace

    actual = initialization_facts(config)
    # The control is built on the blocked reference: initialization does not depend on the
    # backend, and an explicit kernel request (mosaic) need not support exact softmax.
    control = replace(config, model=replace(config.model, attention_backend="blocked",
                                           attention_degree=None, attention_normalization=True,
                                           attention_output_scaling="none",
                                           attention_fixed_coefficients=None, attention_fixed_square_factors=None,
                                           attention_coefficient_parameterization="direct",
                                           attention_coefficient_conditioning="static", attention_coefficient_readout="diagonal",
                                           attention_coefficient_feature_count=None,
                                           attention_coefficient_readout_scale=1.0,
                                           attention_coefficient_activation="identity",
                                           attention_coefficient_amplitude_initial=1.0,
                                           attention_coefficient_projection_initial_std=0.0,
                                           attention_coefficient_tanh_bias=False,
                                           attention_coefficient_kernel_weight_decay=False,
                                           attention_forget_gate=False,
                                           attention_forget_gate_form="input",
                                           qk_norm_token_scale="none",
                                           positional_encoding="rope",
                                           attention_coefficient_amplitude_scale="none",
                                           attention_coefficient_scale="none",
                                           attention_coefficient_initialization="taylor",
                                           attention_denominator_degree=None,
                                           attention_denominator_parameterization="direct",
                                           attention_denominator_constant="fixed"))
    expected = initialization_facts(control)
    if actual != expected:
        raise RuntimeError(f"Initialization differs from the exact-softmax counterpart: {actual} != {expected}")
    return {"softmax_counterpart_parameter_sha256": expected["parameter_sha256"]}


def matches_softmax_backbone_initialization(config) -> dict:
    """Require the exact-softmax control's backbone weights plus declared initial mechanism parameters.

    Mechanisms that add trainable attention arrays register them without drawing
    model random numbers, so every other parameter must equal the softmax control's and
    each added array must hold its declared initial value, regenerated from the
    mechanism stream where it is random.
    """
    from dataclasses import replace

    import jax
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.models import build_model

    def flat_parameters(current):
        current.model.attention_backend = "blocked"
        rng = jax.random.PRNGKey(current.train.seed)
        model = build_model(current.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}

    specs = attention_parameter_specs(config.model)
    control = replace(config, model=replace(
        config.model, attention_backend="blocked",  # as in matches_softmax_initialization
        attention_degree=None, attention_normalization=True, attention_output_scaling="none",
        quadratic_mode="fixed", attention_coefficients="fixed", attention_fixed_coefficients=None,
        attention_fixed_square_factors=None,
        attention_coefficient_parameterization="direct", attention_coefficient_initialization="taylor",
        attention_coefficient_conditioning="static", attention_coefficient_readout="diagonal",
        attention_coefficient_feature_count=None, attention_coefficient_readout_scale=1.0,
        attention_coefficient_activation="identity", attention_coefficient_amplitude_initial=1.0,
        attention_coefficient_projection_initial_std=0.0, attention_coefficient_tanh_bias=False,
        attention_coefficient_kernel_weight_decay=False, attention_forget_gate=False, attention_forget_gate_form="input",
        qk_norm_token_scale="none", positional_encoding="rope",
        attention_coefficient_amplitude_scale="none", attention_coefficient_scale="none",
        attention_denominator_degree=None, attention_denominator_parameterization="direct",
        attention_denominator_constant="fixed", qk_norm=False, qk_norm_head_scale=False))
    actual, expected = flat_parameters(replace(config, model=replace(config.model))), flat_parameters(control)
    added = {(spec.name,) for spec in specs}
    if set(actual) - set(expected) != added or set(expected) - set(actual):
        raise RuntimeError(f"Unexpected parameter names beyond the softmax control: {sorted(set(actual) ^ set(expected))}")
    for path, value in expected.items():
        if not np.array_equal(actual[path], value):
            raise RuntimeError(f"Backbone parameter {path} differs from the exact-softmax control")
    declared = initial_attention_parameter_values(
        specs, num_layers=config.model.num_layers, num_heads=config.model.num_heads,
        key=mechanism_key(jax.random.PRNGKey(config.train.seed)))
    for spec in specs:
        if not np.array_equal(actual[(spec.name,)], np.asarray(declared[spec.name])):
            raise RuntimeError(f"{spec.name} does not hold its declared initial value")
    return {
        "softmax_counterpart_parameter_sha256": initialization_facts(control)["parameter_sha256"],
        "mechanism_parameters": {spec.name: {"shape": list(actual[(spec.name,)].shape), "initial": list(spec.initial),
                                             **({"initial_std": spec.initial_std} if spec.initial_std else {}),
                                             **({"initializer": spec.initializer} if spec.initializer else {}),
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
        backend = resolve_attention_backend(
            config.model, dtype=config.train.dtype, sequence_length=config.data.sequence_length - 1)
        return {"devices": [str(d) for d in devices], "device_kinds": [d.device_kind for d in devices],
                "versions": {name: version(name) for name in packages},
                "attention_backend": backend, "runtime_test_paths": _runtime_test_paths(config, backend)}
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
        config.train.eval_every = continuation_start_step(config) + spec.pilot_steps
        # The scratch worker needs validation at its own stop, regardless of a
        # production recipe's denser early cadence. Never edit the frozen recipe.
        config.train.eval_early_every = None
        config.train.eval_early_until = 0
        config.train.eval_batches = 1
        # Model, context, batch, seed, LR schedule, and step horizon are the production recipe's.
        return run_training(config, resume=True, stop_after_step=args.stop_after_step,
                            audit_checkpoint=True, profile_timing=True)
    raise ValueError(args.worker)


def _run(spec: AcceptanceSpec, args, experiment: dict) -> None:
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    start_step = continuation_start_step(experiment["config"])
    pilot_stop = start_step + spec.pilot_steps
    resume_stop = pilot_stop + 1
    if start_step and resume_stop > experiment["config"]["optim"]["total_steps"]:
        raise ValueError("The remaining training budget must include the pilot and its resume update")
    hardware = _run_worker(spec, args, "hardware", directory / "hardware.json", stop=spec.pilot_steps)
    experiment = on_devices(experiment, hardware["device_kinds"])
    write_json(directory / "identity.json", experiment)
    if passed(directory / "success.json", experiment) and all(passed(directory / f"{name}.json", experiment) for name in CHECKS):
        print(f"matching_acceptance_passed={directory / 'success.json'}", flush=True)
        return
    stage = "cpu-tests"
    try:
        for stage, selection, cpu in TEST_STAGES:
            path = directory / f"{stage}.json"
            if passed(path, experiment):
                continue
            paths = test_paths(spec, experiment["config"], resolved_backend=hardware["attention_backend"], cpu=cpu,
                               runtime_test_paths=tuple(hardware.get("runtime_test_paths", ())))
            if not cpu:
                selection = (*selection, *mosaic_test_options(experiment["config"], hardware["attention_backend"]))
            if not paths:
                record(path, experiment, tests_passed=0, note="no tests declared")
                continue
            junit = directory / f"{stage}.xml"
            cache = getattr(args, "test_cache", None)
            key = stage_identity(experiment, stage, paths, selection, cpu=cpu) if cache else None
            origin = reuse_tests(cache, key, junit) if cache else None
            if origin is not None:
                record(path, experiment, tests_passed=require_passing_tests(junit), reused_from=origin)
                print(f"acceptance_reused={stage} from={origin['acceptance_directory']}", flush=True)
                continue
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
            if cache:
                share_tests(cache, key, junit, directory)
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
            outcome = _run_worker(spec, args, "pilot", directory / "pilot-result.json", stop=pilot_stop)
            if (outcome["final_step"] != pilot_stop or outcome["data_parallel_devices"] != args.gpu_count
                    or outcome.get("start_step", 0) != start_step):
                raise RuntimeError(f"Incomplete full-recipe pilot: {outcome}")
            if spec.required_backend is not None and outcome["attention_backend"] != spec.required_backend:
                raise RuntimeError(f"Production requires backend {spec.required_backend!r}; resolved {outcome['attention_backend']!r}")
            records = [json.loads(line) for line in (directory / "pilot/metrics.jsonl").read_text().splitlines()]
            validation = next(row for row in records if row["step"] == pilot_stop)
            if not math.isfinite(validation["validation_nll"]) or not validation["attention_diagnostics"]["valid"]:
                raise RuntimeError("Pilot validation or attention diagnostics failed")
            projection = project_runtime(outcome["timing"], load_config(args.config))
            # Keep the component costs even when the budget gate rejects the allocation.
            write_json(directory / "runtime-projection.json", projection)
            print(f"acceptance_runtime_projection={json.dumps(projection, sort_keys=True)}", flush=True)
            projected = projection["projected_seconds"]
            require_allocation_budget(projected, experiment["allocation_seconds"])
            record(path, experiment, outcome=outcome, validation=validation, projected_hours=projected / 3600,
                   runtime_projection=projection)
        stage = "resume"
        path = directory / f"{stage}.json"
        if not passed(path, experiment):
            outcome = _run_worker(spec, args, "pilot", directory / "resume-result.json", stop=resume_stop)
            restored = outcome["restored_step"]
            # A retry after the resumed checkpoint was committed can already be at the next step.
            if (restored not in (pilot_stop, resume_stop) or outcome["final_step"] != resume_stop
                    or outcome["data_start_sequence"] != restored * experiment["config"]["data"]["batch_size"]):
                raise RuntimeError(f"Pilot resume did not preserve optimizer and data position: {outcome}")
            if restored == pilot_stop:
                validate_resume_audit(json.loads((directory / "pilot.json").read_text())["outcome"], outcome)
            record(path, experiment, outcome=outcome)
        reused = {name: json.loads((directory / f"{name}.json").read_text()).get("reused_from")
                  for name, _, _ in TEST_STAGES}
        reused = {name: origin for name, origin in reused.items() if origin}
        record(directory / "success.json", experiment, checks=list(CHECKS), hardware=hardware,
               **({"reused_tests": reused} if reused else {}))
        shutil.rmtree(directory / "pilot")  # The disposable pilot checkpoint is no longer needed.
        print(f"acceptance_success={directory / 'success.json'}", flush=True)
    except BaseException as error:
        write_json(directory / "failure.json", {"identity": experiment["identity"], "stage": stage,
                   "error": f"{type(error).__name__}: {error}", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "allocation_seconds": experiment.get("allocation_seconds"),
                   "failed_utc": datetime.now(UTC).isoformat()})
        raise


def main(spec: AcceptanceSpec | Callable, argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--worker", choices=("hardware", "initialization", "pilot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--runtime-env", action="store_true", help="Print shell exports for the launcher and exit")
    parser.add_argument("--test-cache", type=Path,
                        help="Directory sharing passed test stages between runs with the same test identity")
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
    if args.test_cache is not None:
        args.test_cache = args.test_cache.resolve()
    experiment = identity(spec, args.config, args.gpu_count, args.gpu_type)
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        write_json(args.output, _worker(spec, args, experiment))
    else:
        _run(spec, args, experiment)
