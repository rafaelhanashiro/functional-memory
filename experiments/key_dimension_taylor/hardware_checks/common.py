"""Isolated validation policy; no production scheduling or scientific output writes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_execution_common", Path(__file__).parents[1] / "concurrent/common.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
write_json, locked, policy_identity = _base.write_json, _base.locked, _base.policy_identity
training_env = _base.training_env


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load_reference(ref: dict) -> dict:
    path = Path(ref["path"])
    if sha256(path) != ref["sha256"]:
        raise ValueError(f"Referenced qualification changed: {path}")
    return json.loads(path.read_text())


def load_policy(path: Path) -> tuple[dict, dict]:
    policy = json.loads(Path(path).read_text())
    manifest = json.loads(Path(policy["training_manifest"]).read_text())
    sys.path.insert(0, policy["training_root"])
    return policy, manifest


def verify_policy(policy: dict, manifest: dict) -> None:
    if policy_identity(policy) != policy["identity"] or policy["production_submission"] is not False:
        raise ValueError("Validation-only policy identity changed")
    if (policy["source_commit"] != manifest["source_commit"]
            or policy["training_identity"] != manifest["identity"]
            or policy["training_root"] != manifest["execution_root"]):
        raise ValueError("Validation must reference the unchanged scientific source")
    root = Path(policy["orchestrator_root"])
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != policy["orchestrator_commit"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True)):
        raise ValueError("Frozen validation checkout changed")
    sys.path.insert(0, policy["training_root"])
    from experiments.key_dimension_taylor import preflight
    from experiments.polynomial_attention.configs import config_to_dict, load_config
    if not Path(preflight.__file__).resolve().is_relative_to(Path(policy["training_root"]).resolve()):
        raise ValueError("Scientific code imported outside its frozen checkout")
    preflight.verify_manifest(manifest)
    expected = {(width, degree) for width in manifest["key_dimensions"] for degree in manifest["degrees"]}
    if {(run["qk_head_dim"], run["attention_degree"]) for run in policy["runs"]} != expected:
        raise ValueError("Validation does not cover the full Cartesian grid")
    from experiments.key_dimension_taylor.sweep import scientific_recipe, parameter_count
    baseline = scientific_recipe(next(run["config"] for run in manifest["runs"] if not run.get("retained_completed")),
                                 manifest["data_identity"])
    output = Path(policy["output_root"]).resolve()
    canonical = Path(manifest["project_root"]) / "results/key_dimension_taylor"
    if output != canonical / "hardware_checks":
        raise ValueError("Hardware checks must use their isolated results directory")
    names = {run["name"] for run in policy["runs"]}
    if len(names) != len(expected):
        raise ValueError("Duplicate validation coordinates")
    for run in policy["runs"]:
        path = Path(run["config_path"])
        config = load_config(path)
        if (sha256(path) != run["config_sha256"] or config_to_dict(config) != run["config"]
                or scientific_recipe(run["config"], manifest["data_identity"]) != baseline
                or config.model.qk_head_dim != run["qk_head_dim"]
                or config.model.attention_degree != run["attention_degree"]
                or parameter_count(config.model) != run["parameter_count"]
                or config.train.source_commit != manifest["source_commit"]
                or not Path(config.train.workdir).resolve().is_relative_to(output)):
            raise ValueError(f"Validation configuration changed or escaped isolation: {run['name']}")
    for name, profile in policy["profiles"].items():
        if profile["gpu_count"] != 4 or profile["partition"] != "sweep" or profile["production_time"] != "08:00:00":
            raise ValueError("Validation must preserve four-device training and use the sweep partition")
        shards = [run for shard in policy["shards"][name] for run in shard]
        if len(shards) != len(names) or set(shards) != names:
            raise ValueError("Validation shards omit or duplicate coordinates")
    for profile, records in policy["reused_checks"].items():
        for name, ref in records.items():
            record = load_reference(ref)
            run = next(run for run in policy["runs"] if run["name"] == name)
            if (record.get("status") != "passed" or record.get("source_commit") != manifest["source_commit"]
                    or record.get("profile") != profile
                    or record.get("qk_head_dim") != run["qk_head_dim"]
                    or record.get("attention_degree") != run["attention_degree"]
                    or record["validation"].get("parameter_count") != run["parameter_count"]):
                raise ValueError("Reused full-model qualification has incompatible provenance")
    for profile, ref in policy["reused_kernels"].items():
        record = load_reference(ref)
        if (record.get("status") != "passed" or record.get("source_commit") != manifest["source_commit"]
                or record.get("profile") != profile or record.get("tests_passed") != 105):
            raise ValueError("Reused kernel qualification has incompatible provenance")
    for ref in policy["reference_files"]:
        if sha256(Path(ref["path"])) != ref["sha256"]:
            raise ValueError(f"Referenced provenance changed: {ref['path']}")


def grid_run(policy: dict, manifest: dict, width: int, degree: int) -> dict:
    return next(run for run in policy["runs"] if (run["qk_head_dim"], run["attention_degree"]) == (width, degree))


def check_hardware(profile: dict, count: int) -> dict:
    import jax
    devices = jax.devices()
    if len(devices) != count or any(device.platform != "gpu" or profile["gpu_type"].lower() not in device.device_kind.lower()
                                    for device in devices):
        raise RuntimeError(f"Unexpected hardware: {devices}")
    node = os.environ.get("SLURMD_NODENAME")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if node not in profile["allowed_nodes"] or partition != profile["partition"]:
        raise RuntimeError(f"Unexpected validation allocation: {node}/{partition}")
    return {"device_kinds": [device.device_kind for device in devices], "gpu_count": count,
            "node": node, "partition": partition, "jax_version": jax.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "memory_stats": [device.memory_stats() for device in devices],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
