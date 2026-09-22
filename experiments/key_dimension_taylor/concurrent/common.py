"""Execution amendments around an unchanged, separately frozen training source."""
from __future__ import annotations

import contextlib
import copy
import fcntl
import getpass
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


A6000_RUNTIME_ENVIRONMENT = {
    "NCCL_P2P_DISABLE": "1", "POLYNOMIAL_ATTENTION_HOST_STAGING": "1",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


def _verify_checkout(root: Path, commit: str) -> None:
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != commit
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True)):
        raise ValueError(f"Frozen execution checkout changed: {root}")


def _reference_bytes(reference: dict) -> bytes:
    contents = Path(reference["path"]).read_bytes()
    if hashlib.sha256(contents).hexdigest() != reference["sha256"]:
        raise ValueError(f"Qualification reference checksum changed: {reference['path']}")
    return contents


def _reference_json(reference: dict) -> dict:
    return json.loads(_reference_bytes(reference))


def execution_provenance(policy: dict) -> dict:
    """Keep the immutable scientific/config source distinct from executed code."""
    return {"original_scientific_source_commit": policy["source_commit"],
            "runtime_root": policy.get("runtime_root", policy["training_root"]),
            "runtime_commit": policy.get("runtime_commit", policy["source_commit"]),
            "runtime_environment": policy.get("runtime_environment", {})}


def validate_runtime_segments(runtime: dict, total: int) -> None:
    if (runtime.get("limit_seconds") != 28800 or runtime.get("allocation_fraction") != 0.8
            or runtime.get("usable_seconds") != 23040):
        raise RuntimeError("Qualification changed the authorized wall-time budget")
    cursor = 0
    for segment in runtime["segments"]:
        start, end, seconds = segment["start_step"], segment["end_step"], segment["projected_seconds"]
        if (type(start) is not int or type(end) is not int or start != cursor or not start < end <= total
                or (end != total and end % 500 != 0) or not math.isfinite(seconds)
                or not 0 < seconds <= runtime["usable_seconds"]):
            raise RuntimeError("Qualification contains an invalid or oversized checkpoint segment")
        cursor = end
    if cursor != total:
        raise ValueError("Qualification has an incomplete segment schedule")


def _original_preflight(policy: dict):
    root = Path(policy["training_root"]).resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("experiments.key_dimension_taylor.preflight")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError("Scientific imports escaped the frozen training checkout")
    return module


def _hardware_policy(policy: dict, manifest: dict) -> dict:
    original = _reference_json(policy["hardware_policy_reference"])
    if (policy_identity(original) != original.get("identity")
            or original.get("production_submission") is not False
            or original.get("training_identity") != manifest["identity"]
            or original.get("source_commit") != manifest["source_commit"]
            or original.get("training_manifest") != policy["training_manifest"]
            or original.get("training_root") != policy["training_root"]
            or original.get("dependency_lock_sha256") != manifest["dependency_lock_sha256"]
            or original.get("data_identity") != manifest["data_identity"]):
        raise ValueError("Hardware qualification policy differs from the original sweep")
    _verify_checkout(Path(original["orchestrator_root"]), original["orchestrator_commit"])
    _verify_checkout(Path(policy["training_root"]), manifest["source_commit"])
    lock = Path(policy["training_root"]) / "experiments/polynomial_attention/uv.lock"
    if hashlib.sha256(lock.read_bytes()).hexdigest() != manifest["dependency_lock_sha256"]:
        raise ValueError("Original qualified runtime dependency lock changed")
    for reference in original.get("reference_files", []):
        _reference_bytes(reference)
    return original


def _b300_hardware(hardware: dict, profile: dict) -> None:
    kinds = hardware.get("device_kinds", [])
    if (hardware.get("gpu_count") != 4 or len(kinds) != 4
            or any("b300" not in kind.lower() for kind in kinds)
            or hardware.get("node") not in profile["allowed_nodes"]
            or hardware.get("partition") != profile["partition"]):
        raise ValueError("B300 qualification used incompatible hardware")


def load_b300_qualification(policy: dict, manifest: dict, run: dict) -> dict:
    """Read a completed hardware-check record, including its original provenance.

    The original policy is immutable. Its future per-coordinate output can be
    absent while short pilots run; capture its SHA256 only after all gates pass.
    """
    original = _hardware_policy(policy, manifest)
    profile = original["profiles"]["b300"]
    root = Path(original["output_root"]) / "b300"
    path = root / "checks" / f"{run['name']}.json"
    contents = path.read_bytes()
    record = json.loads(contents)
    expected = {"identity": original["identity"], "source_commit": manifest["source_commit"],
                "orchestrator_commit": original["orchestrator_commit"], "profile": "b300",
                "scheduler": profile, "validation_only": True, "status": "passed"}
    if (any(record.get(key) != value for key, value in expected.items())
            or any(record.get(key) != run[key] for key in ("name", "qk_head_dim", "attention_degree", "parameter_count"))):
        raise ValueError("B300 coordinate qualification is absent, failed, or belongs to another source")
    candidates = [item for item in original["runs"] if item["name"] == run["name"]]
    if len(candidates) != 1:
        raise ValueError("Hardware policy omitted or duplicated the B300 coordinate")
    validation_run = candidates[0]
    if any(validation_run.get(key) != run[key] for key in ("qk_head_dim", "attention_degree", "parameter_count")):
        raise ValueError("B300 configuration coordinate differs")
    _reference_bytes({"path": validation_run["config_path"], "sha256": validation_run["config_sha256"]})
    preflight = _original_preflight(policy)
    if preflight.scientific_recipe(validation_run["config"], manifest["data_identity"]) != preflight.scientific_recipe(run["config"], manifest["data_identity"]):
        raise ValueError("B300 pilot changed the scientific recipe")
    kernel_path = root / "kernels.json"
    kernel_bytes = kernel_path.read_bytes()
    kernels = json.loads(kernel_bytes)
    if any(kernels.get(key) != value for key, value in expected.items()) or kernels.get("tests_passed") != 105:
        raise ValueError("B300 requires its matching 105-case kernel qualification")
    _b300_hardware(kernels["hardware"], profile)
    if kernels.get("reused"):
        source = _reference_json(kernels["reference"])
        if (source.get("status") != "passed" or source.get("tests_passed") != 105
                or source.get("source_commit") != manifest["source_commit"] or source.get("profile") != "b300"):
            raise ValueError("Reused B300 kernel qualification has incompatible provenance")
        _b300_hardware(source["hardware"], profile)
    # The three migrating coordinates are fresh full-grid checks. Supporting
    # prior reused checks preserves their original fields rather than relabeling.
    proof = _reference_json(record["reference"]) if record.get("reused") else record
    if (proof.get("status") != "passed" or proof.get("source_commit") != manifest["source_commit"]
            or proof.get("profile") != "b300"
            or any(proof.get(key) != run[key] for key in ("name", "qk_head_dim", "attention_degree"))):
        raise ValueError("B300 pilot reference has incompatible provenance")
    _b300_hardware(record["hardware"], profile)
    _b300_hardware(proof["hardware"], profile)
    initial, resumed = proof["initial"], proof["resumed"]
    for outcome, start, end in ((initial, 0, 32), (resumed, 32, 33)):
        if (outcome.get("start_step") != start or outcome.get("final_step") != end
                or outcome.get("restored_step") != (start or None)
                or outcome.get("data_start_sequence") != start * run["config"]["data"]["batch_size"]
                or outcome.get("data_parallel_devices") != 4 or outcome.get("attention_degree") != run["attention_degree"]):
            raise ValueError("B300 pilot changed its update budget, checkpoint, or cursor")
    preflight.validate_resume_audit(initial, resumed)
    validation = proof["validation"]
    if (validation.get("step") != 32 or validation.get("source_commit") != manifest["source_commit"]
            or any(validation.get(key) != run[key] for key in ("qk_head_dim", "attention_degree", "parameter_count"))
            or not all(math.isfinite(validation.get(key, math.nan)) for key in ("validation_nll", "validation_perplexity"))
            or validation.get("attention_diagnostics", {}).get("valid") is not True):
        raise ValueError("B300 pilot lacks matching finite evaluation and diagnostics")
    if record["runtime"] != preflight.projected_runtime(run, initial, resumed, profile):
        raise ValueError("B300 runtime differs from its synchronized pilot timings")
    validate_runtime_segments(record["runtime"], run["config"]["optim"]["total_steps"])
    return {**record, "qualification_reference": {"path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest()},
            "kernel_qualification_reference": {"path": str(kernel_path.resolve()), "sha256": hashlib.sha256(kernel_bytes).hexdigest()}}


def load_a6000_qualification(policy: dict, manifest: dict, run: dict) -> dict:
    """Validate original host-staging evidence without copying or relabeling it."""
    if policy["assignments"].get(run["name"]) != "a6000":
        raise ValueError("A6000 qualification requires an A6000-assigned coordinate")
    record = _reference_json(policy["a6000_qualifications"][run["name"]])
    expected = {"format": "key-dimension-taylor-a6000-host-staging", "status": "passed",
                "validation_only": True, "production_submission": False,
                "runtime_commit": policy["runtime_commit"], "runtime_root": policy["runtime_root"],
                "original_scientific_source_commit": manifest["source_commit"],
                **{key: run[key] for key in ("name", "qk_head_dim", "attention_degree", "parameter_count")}}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("A6000 qualification source, coordinate, or status differs")
    if policy.get("runtime_environment") != A6000_RUNTIME_ENVIRONMENT:
        raise ValueError("A6000 execution environment differs from the qualified workaround")
    environment = record.get("environment", {})
    if (any(environment.get(key) != value for key, value in A6000_RUNTIME_ENVIRONMENT.items())
            or environment.get("POLYNOMIAL_ATTENTION_SOURCE_COMMIT") != policy["runtime_commit"]
            or environment.get("PYTHONPATH") != policy["runtime_root"]
            or environment.get("JAX_PLATFORMS") != "cuda" or environment.get("XLA_FLAGS") not in (None, "")):
        raise ValueError("A6000 qualification used a different execution environment")
    original = _reference_json(record["original_policy"])
    if (policy_identity(original) != original.get("identity")
            or original.get("identity") != record.get("original_policy_identity")
            or original.get("production_submission") is not False
            or original.get("training_identity") != manifest["identity"]
            or original.get("source_commit") != manifest["source_commit"]
            or original.get("training_manifest") != policy["training_manifest"]
            or original.get("training_root") != policy["training_root"]
            or original.get("dependency_lock_sha256") != manifest["dependency_lock_sha256"]
            or original.get("data_identity") != manifest["data_identity"]):
        raise ValueError("Original validation policy does not match this scientific sweep")
    _verify_checkout(Path(original["orchestrator_root"]), original["orchestrator_commit"])
    candidates = [item for item in original["runs"] if item["name"] == run["name"]]
    if len(candidates) != 1:
        raise ValueError("Original validation policy omitted or duplicated this coordinate")
    validation_run = candidates[0]
    if (any(validation_run.get(key) != run[key] for key in ("qk_head_dim", "attention_degree", "parameter_count"))
            or record.get("resolved_original_config") != validation_run["config"]
            or record.get("data_identity") != manifest["data_identity"]
            or record.get("dependency_lock_sha256") != manifest["dependency_lock_sha256"]
            or record["original_config"] != {"path": validation_run["config_path"], "sha256": validation_run["config_sha256"]}):
        raise ValueError("A6000 qualification changed its configuration or data")
    _reference_bytes(record["original_config"])
    for reference in original.get("reference_files", []):
        _reference_bytes(reference)
    kernels = _reference_json(record["kernel_reference"])
    if (kernels.get("status") != "passed" or kernels.get("tests_passed") != 105
            or kernels.get("identity") != original["identity"] or kernels.get("profile") != "a6000"
            or kernels.get("source_commit") != manifest["source_commit"]
            or kernels.get("orchestrator_commit") != original["orchestrator_commit"]
            or run["name"] not in kernels.get("eligible_runs", [])):
        raise ValueError("A6000 requires its matching 105-case kernel qualification")
    runtime_root, original_root = Path(policy["runtime_root"]), Path(policy["training_root"])
    _verify_checkout(runtime_root, policy["runtime_commit"])
    for relative in ("experiments/polynomial_attention/models/components/attention.py",
                     "experiments/polynomial_attention/models/components/fused_attention.py"):
        if (runtime_root / relative).read_bytes() != (original_root / relative).read_bytes():
            raise ValueError("Runtime kernel changed since the original kernel qualification")
    if hashlib.sha256((runtime_root / "experiments/polynomial_attention/uv.lock").read_bytes()).hexdigest() != manifest["dependency_lock_sha256"]:
        raise ValueError("A6000 runtime dependency lock differs")
    preflight = _original_preflight(policy)
    if preflight.scientific_recipe(validation_run["config"], manifest["data_identity"]) != preflight.scientific_recipe(run["config"], manifest["data_identity"]):
        raise ValueError("A6000 pilot changed the scientific recipe")
    initial, resumed = record["initial"], record["resumed"]
    expected_config = copy.deepcopy(validation_run["config"])
    expected_config["train"].update(source_commit=policy["runtime_commit"],
                                    workdir=str(Path(record["attempt_path"]) / "workdir"),
                                    wandb_mode="disabled", eval_every=32, eval_batches=1)
    for outcome, start, end in ((initial, 0, 32), (resumed, 32, 33)):
        if (outcome.get("start_step") != start or outcome.get("final_step") != end
                or outcome.get("restored_step") != (start or None)
                or outcome.get("data_start_sequence") != start * run["config"]["data"]["batch_size"]
                or outcome.get("data_parallel_devices") != 4 or outcome.get("attention_degree") != run["attention_degree"]):
            raise ValueError("A6000 pilot changed its update budget, restart, or data cursor")
        if outcome.get("resolved_config") != expected_config:
            raise ValueError("A6000 worker changed its resolved scientific configuration")
        hardware = outcome.get("hardware", {})
        kinds = hardware.get("device_kinds", [])
        if (hardware.get("gpu_count") != 4 or len(kinds) != 4 or any("a6000" not in kind.lower() for kind in kinds)
                or hardware.get("node") not in policy["profiles"]["a6000"]["allowed_nodes"]
                or hardware.get("partition") != policy["profiles"]["a6000"]["partition"]):
            raise ValueError("A6000 pilot ran on incompatible hardware")
    preflight.validate_resume_audit(initial, resumed)
    validation = record["initial_last_committed_validation"]
    if (validation != record.get("resumed_last_committed_validation")
            or validation.get("step") != 32 or validation.get("source_commit") != policy["runtime_commit"]
            or any(validation.get(key) != run[key] for key in ("qk_head_dim", "attention_degree", "parameter_count"))
            or not all(math.isfinite(validation.get(key, math.nan)) for key in ("validation_nll", "validation_perplexity"))
            or validation.get("attention_diagnostics", {}).get("valid") is not True):
        raise ValueError("A6000 pilot lacks its matching finite step-32 evaluation")
    parity = record["h100_initialization_and_input_parity"]
    reference = _reference_json(parity["reference"])
    if (parity.get("status") != "passed" or reference.get("status") != "passed"
            or reference.get("identity") != manifest["identity"] or reference.get("source_commit") != manifest["source_commit"]
            or any(initial["checkpoint_audit"][key] != reference["initial"]["checkpoint_audit"][key]
                   or parity.get(key) != initial["checkpoint_audit"][key] for key in ("start_state_sha256", "first_update"))):
        raise ValueError("A6000 initialization or inputs differ from original H100 qualification")
    if record["runtime"] != preflight.projected_runtime(run, initial, resumed, policy["profiles"]["a6000"]):
        raise ValueError("A6000 qualified runtime no longer matches its synchronized pilot timings")
    validate_runtime_segments(record["runtime"], run["config"]["optim"]["total_steps"])
    return record


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            # Closing the descriptor preserves an inherited child's lock if its
            # supervisor dies. An explicit LOCK_UN would release it for both.
            pass


def policy_identity(policy: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in policy.items() if k != "identity"},
                                    sort_keys=True).encode()).hexdigest()


def load_policy(path: Path) -> tuple[dict, dict]:
    policy = json.loads(Path(path).read_text())
    manifest = json.loads(Path(policy["training_manifest"]).read_text())
    sys.path.insert(0, policy["training_root"])
    return policy, manifest


def verify_policy(policy: dict, manifest: dict) -> None:
    if policy_identity(policy) != policy["identity"]:
        raise ValueError("Execution amendment identity changed")
    if (policy["training_identity"] != manifest["identity"]
            or policy["source_commit"] != manifest["source_commit"]
            or policy["training_root"] != manifest["execution_root"]):
        raise ValueError("Execution amendment does not reference the original training source")
    root = Path(policy["orchestrator_root"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True)
    if head != policy["orchestrator_commit"] or dirty:
        raise ValueError("Frozen orchestration checkout changed")
    sys.path.insert(0, policy["training_root"])
    original = importlib.import_module("experiments.key_dimension_taylor.preflight")
    if not Path(original.__file__).resolve().is_relative_to(Path(policy["training_root"]).resolve()):
        raise ValueError("Scientific imports escaped the frozen training checkout")
    original.verify_manifest(manifest)
    active = {run["name"] for run in manifest["runs"] if not run.get("retained_completed")}
    for name, profile in policy["assignments"].items():
        if name not in active or profile not in policy["profiles"]:
            raise ValueError("Unknown coordinate or hardware assignment")
    for profile in policy["profiles"].values():
        if profile["gpu_count"] != 4 or profile["production_time"] != "08:00:00":
            raise ValueError("Execution amendment changed the four-device/eight-hour recipe")
    if "a6000" in policy["assignments"].values():
        if (set(policy["assignments"].values()) != {"a6000"}
                or policy.get("runtime_environment") != A6000_RUNTIME_ENVIRONMENT
                or set(policy.get("a6000_qualifications", {})) != set(policy["assignments"])
                or policy["profiles"]["a6000"].get("gpu_type") != "a6000"
                or policy["profiles"]["a6000"].get("partition") != "sweep"
                or not set(policy["profiles"]["a6000"].get("allowed_nodes", []))
                or not set(policy["profiles"]["a6000"]["allowed_nodes"]) <= {"doodle", "terrier"}):
            raise ValueError("A6000 requires a separate qualified runtime-only amendment")
        _verify_checkout(Path(policy["runtime_root"]), policy["runtime_commit"])
    elif any(key in policy for key in ("runtime_root", "runtime_commit", "runtime_environment", "a6000_qualifications")):
        raise ValueError("Host-staging runtime is reserved for the A6000 amendment")
    if "h100_or_better" in policy["assignments"].values():
        flexible = policy["profiles"]["h100_or_better"]
        if (set(policy["assignments"].values()) != {"h100_or_better"}
                or flexible.get("gpu_type") != "h100_or_better"
                or set(flexible.get("allowed_gpu_types", [])) != {"h100", "b300"}
                or set(flexible.get("allowed_nodes", [])) != {"akita", "dogo", "mastiff"}
                or flexible.get("partition") != "sweep"):
            raise ValueError("Flexible production requires only qualified H100/B300 allocations")
        for family, nodes in (("h100", {"dogo", "mastiff"}), ("b300", {"akita"})):
            profile = policy["profiles"][family]
            if (profile.get("gpu_type") != family or set(profile.get("allowed_nodes", [])) != nodes
                    or profile.get("partition") != "sweep"):
                raise ValueError("Flexible concrete hardware profile changed")
        minimum = policy.get("minimum_checkpoint_step")
        if "minimum_checkpoint_step" in policy and (type(minimum) is not int or minimum <= 0 or minimum % 500):
            raise ValueError("Checkpoint handoff minimum must be a positive 500-update boundary")
        if minimum is not None and any(minimum > run["config"]["optim"]["total_steps"]
                                       for run in manifest["runs"] if run["name"] in policy["assignments"]):
            raise ValueError("Checkpoint handoff minimum exceeds the training budget")
        _hardware_policy(policy, manifest)


def training_env(policy: dict) -> dict:
    root = Path(policy.get("runtime_root", policy["training_root"]))
    manifest = json.loads(Path(policy["training_manifest"]).read_text())
    cache = Path(manifest["project_root"]) / ".cache/key_dimension_taylor/concurrent"
    for suffix in ("jax", "xdg", "tmp"):
        (cache / suffix).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cuda",
                POLYNOMIAL_ATTENTION_SOURCE_COMMIT=policy.get("runtime_commit", manifest["source_commit"]),
                JAX_COMPILATION_CACHE_DIR=str(cache / "jax"), XDG_CACHE_HOME=str(cache / "xdg"),
                TMPDIR=str(cache / "tmp"), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "32"),
                MKL_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "32"), MPLBACKEND="Agg")
    if "runtime_root" in policy:
        if policy.get("runtime_environment") != A6000_RUNTIME_ENVIRONMENT:
            raise ValueError("A6000 runtime environment differs from qualification")
        if env.get("XLA_FLAGS"):
            raise ValueError("A6000 qualification did not use custom XLA_FLAGS")
        env.update(policy["runtime_environment"])
    elif "h100_or_better" in policy["assignments"].values():
        # Submission may originate in an A6000 execution environment. The
        # original fast-GPU runtime was qualified without those workarounds.
        for key in ("NCCL_P2P_DISABLE", "POLYNOMIAL_ATTENTION_HOST_STAGING", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
            env.pop(key, None)
        if env.get("XLA_FLAGS"):
            raise ValueError("Flexible production was not qualified with custom XLA_FLAGS")
    return env


def check_hardware(profile: dict) -> dict:
    import jax
    devices = jax.devices()
    allowed = profile.get("allowed_gpu_types", [profile["gpu_type"]])
    families = [{kind for kind in allowed if kind.lower() in device.device_kind.lower()} for device in devices]
    if (len(devices) != profile["gpu_count"] or any(device.platform != "gpu" for device in devices)
            or any(len(matches) != 1 for matches in families)
            or len({next(iter(matches)) for matches in families if matches}) != 1):
        raise RuntimeError(f"Allocation does not match hardware profile: {devices}")
    actual_profile = next(iter(families[0]))
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_JOB_NODELIST")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    allowed_nodes = profile.get("allowed_nodes", [profile.get("node")])
    if node not in allowed_nodes or partition != profile["partition"]:
        raise RuntimeError(f"Unexpected node/partition: {node}/{partition}")
    return {"devices": [str(device) for device in devices],
            "device_kinds": [device.device_kind for device in devices],
            "node": node, "partition": partition, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu_count": len(devices), "jax_version": jax.__version__, "actual_profile": actual_profile}


def job_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    output = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
                                      "--format=JobIDRaw,State"], text=True)
    result = {parts[0]: parts[1].split()[0].split("+")[0]
              for line in output.splitlines() if len(parts := line.split("|")) >= 2
              and parts[0] in job_ids}
    # squeue is authoritative for an active/requeued job; accounting can lag.
    output = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T"], text=True)
    result.update(dict(line.split("|", 1) for line in output.splitlines()
                       if "|" in line and line.split("|", 1)[0] in job_ids))
    return result


TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}


def assert_legacy_jobs_terminal(policy: dict, manifest: dict, run_name: str) -> None:
    jobs = policy["superseded_jobs"].get(run_name, [])
    states = job_states(jobs)
    if any(states.get(job) not in TERMINAL for job in jobs):
        raise RuntimeError(f"Legacy coordinate owner is still active or unaccounted for: {states}")


def render_report(policy: dict, manifest: dict) -> None:
    """Publish only when legacy launchers cannot race this renderer."""
    root = Path(manifest["project_root"]) / "results/key_dimension_taylor"
    output = root / "execution_reports" / policy["identity"]
    with locked(root / ".report.lock"):
        sys.path.insert(0, policy["training_root"])
        from experiments.key_dimension_taylor.plot import aggregate
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".render-", dir=output.parent) as temporary:
            staging = Path(temporary)
            aggregate(manifest, output_dir=staging)
            report = staging / "README.md"
            appendix = ["", "## Concurrent execution amendment", "",
                        "Remaining coordinates run independently on their qualified allocations. "
                        "The scientific configuration is unchanged. "
                        "Hardware can affect floating-point trajectories and throughput.", "",
                        f"Original scientific/config source: `{policy['source_commit']}`. "
                        f"Executed runtime source: `{policy.get('runtime_commit', policy['source_commit'])}`. "
                        f"Orchestration source: `{policy['orchestrator_commit']}`. "
                        f"Execution policy: `{policy['identity']}`.", "",
                        "| Coordinate | Assigned GPUs | Node | Partition |",
                        "|---|---|---|---|"]
            for name, profile_name in policy["assignments"].items():
                profile = policy["profiles"][profile_name]
                node = profile.get("node") or ", ".join(profile.get("allowed_nodes", []))
                appendix.append(f"| {name} | 4 {profile['gpu_type'].upper()} | {node} | {profile['partition']} |")
            if "runtime_root" in policy:
                appendix += ["", "A6000 workers execute the separately frozen host-staging runtime. "
                             "Canonical configurations and metric source fields identify the original scientific recipe; "
                             "the actual executed source, environment, and hashed qualification references are preserved "
                             "in each allocation's execution provenance.", "",
                             f"Runtime environment: `{json.dumps(policy['runtime_environment'], sort_keys=True)}`."]
            if "h100_or_better" in policy["assignments"].values():
                appendix += ["", "Flexible allocations admit H100 and B300 GPUs only. Each worker detects its "
                             "actual GPU family and validates that coordinate's matching qualification before training. "
                             "Existing checkpoints resume with the original frozen trainer; earlier hardware/runtime "
                             "history remains in the allocation records. Actual hardware can change subsequent "
                             "floating-point trajectories."]
            appendix += ["", "Actual allocations, restart counts, and qualification evidence are retained "
                         "in each run's `execution_attempts/` and the separate execution-policy ledger. "
                         "Completed and historical reference runs retain their original provenance.", ""]
            report.write_text(report.read_text() + "\n".join(appendix))
            for item in staging.iterdir():
                os.replace(item, output / item.name)
        states = job_states(policy.get("legacy_renderer_jobs", []))
        if all(states.get(job) in TERMINAL for job in policy.get("legacy_renderer_jobs", [])):
            for item in output.iterdir():
                descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=root)
                os.close(descriptor)
                shutil.copyfile(item, temporary)
                os.replace(temporary, root / item.name)
