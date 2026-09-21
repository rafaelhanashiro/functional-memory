"""Execution amendments around an unchanged, separately frozen training source."""
from __future__ import annotations

import contextlib
import fcntl
import getpass
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def training_env(policy: dict) -> dict:
    root = Path(policy["training_root"])
    manifest = json.loads(Path(policy["training_manifest"]).read_text())
    cache = Path(manifest["project_root"]) / ".cache/key_dimension_taylor/concurrent"
    for suffix in ("jax", "xdg", "tmp"):
        (cache / suffix).mkdir(parents=True, exist_ok=True)
    return dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cuda",
                POLYNOMIAL_ATTENTION_SOURCE_COMMIT=manifest["source_commit"],
                JAX_COMPILATION_CACHE_DIR=str(cache / "jax"), XDG_CACHE_HOME=str(cache / "xdg"),
                TMPDIR=str(cache / "tmp"), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "32"),
                MKL_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "32"), MPLBACKEND="Agg")


def check_hardware(profile: dict) -> dict:
    import jax
    devices = jax.devices()
    if len(devices) != profile["gpu_count"] or any(
            device.platform != "gpu" or profile["gpu_type"].lower() not in device.device_kind.lower()
            for device in devices):
        raise RuntimeError(f"Allocation does not match hardware profile: {devices}")
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_JOB_NODELIST")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if node != profile["node"] or partition != profile["partition"]:
        raise RuntimeError(f"Unexpected node/partition: {node}/{partition}")
    return {"devices": [str(device) for device in devices],
            "device_kinds": [device.device_kind for device in devices],
            "node": node, "partition": partition, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu_count": len(devices), "jax_version": jax.__version__}


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
                        "Remaining coordinates run independently on H100 and B300 allocations. "
                        "The scientific configuration and frozen training code are unchanged. "
                        "Hardware can affect floating-point trajectories and throughput.", "",
                        f"Training source: `{policy['source_commit']}`. "
                        f"Orchestration source: `{policy['orchestrator_commit']}`. "
                        f"Execution policy: `{policy['identity']}`.", "",
                        "| Coordinate | Assigned GPUs | Node | Partition |",
                        "|---|---|---|---|"]
            for name, profile_name in policy["assignments"].items():
                profile = policy["profiles"][profile_name]
                appendix.append(f"| {name} | 4 {profile['gpu_type'].upper()} | {profile['node']} | {profile['partition']} |")
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
