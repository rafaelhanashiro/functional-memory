"""Flexible GPU dispatch keeps qualifications and durable progress intact."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor.tests.test_a6000_production_core import evidence
from experiments.key_dimension_taylor.tests.test_concurrent_production import (
    mock_workers, read, run, setup_production,
)


@pytest.fixture
def b300(evidence, tmp_path):
    e = evidence
    common = e.common
    profile = {"gpu_type": "b300", "gpu_count": 4, "allowed_nodes": ["akita"],
               "production_time": "08:00:00", "partition": "sweep"}
    hardware = {"gpu_count": 4, "device_kinds": ["NVIDIA B300 SXM6 AC"] * 4,
                "node": "akita", "partition": "sweep"}
    original = read(e.record["original_policy"]["path"])
    original.update(output_root=str(tmp_path / "hardware"), profiles={"b300": profile})
    original["identity"] = common.policy_identity(original)
    original_path = Path(e.record["original_policy"]["path"])
    common.write_json(original_path, original)
    e.policy["hardware_policy_reference"] = e.reference(original_path)
    for key in ("runtime_root", "runtime_commit", "runtime_environment", "a6000_qualifications"):
        e.policy.pop(key, None)
    e.policy["assignments"][e.run["name"]] = "h100_or_better"
    e.policy["profiles"] = {"b300": profile}
    base = {"identity": original["identity"], "source_commit": e.manifest["source_commit"],
            "orchestrator_commit": original["orchestrator_commit"], "profile": "b300",
            "scheduler": profile, "validation_only": True, "status": "passed", "hardware": hardware}
    source = {"status": "passed", "source_commit": e.manifest["source_commit"], "profile": "b300",
              "tests_passed": 105, "hardware": hardware}
    source_path = tmp_path / "original-b300-kernels.json"
    common.write_json(source_path, source)
    kernel = {**base, "reused": True, "tests_passed": 105, "reference": e.reference(source_path)}
    kernel_path = Path(original["output_root"]) / "b300/kernels.json"
    common.write_json(kernel_path, kernel)
    validation = copy.deepcopy(e.record["initial_last_committed_validation"])
    validation["source_commit"] = e.manifest["source_commit"]
    native = {**base, **{key: e.run[key] for key in ("name", "qk_head_dim", "attention_degree", "parameter_count")},
              "initial": copy.deepcopy(e.record["initial"]), "resumed": copy.deepcopy(e.record["resumed"]),
              "validation": validation, "runtime": copy.deepcopy(e.record["runtime"]), "reused": False}
    path = Path(original["output_root"]) / "b300/checks" / f"{e.run['name']}.json"
    common.write_json(path, native)
    return SimpleNamespace(e=e, common=common, policy=e.policy, manifest=e.manifest, run=e.run,
                           native=native, path=path, kernel_path=kernel_path, source_path=source_path)


def test_native_b300_proof_keeps_identity_and_captures_checksum(b300):
    actual = b300.common.load_b300_qualification(b300.policy, b300.manifest, b300.run)
    assert actual["identity"] == b300.native["identity"]
    assert actual["qualification_reference"] == {
        "path": str(b300.path), "sha256": hashlib.sha256(b300.path.read_bytes()).hexdigest()}


def test_missing_b300_proof_is_pending(b300):
    b300.path.unlink()
    with pytest.raises(FileNotFoundError):
        b300.common.load_b300_qualification(b300.policy, b300.manifest, b300.run)


@pytest.mark.parametrize("case", ["status", "identity", "source", "coordinate", "family", "cursor", "state", "rng", "validation", "runtime"])
def test_invalid_native_b300_proof_is_rejected(b300, case):
    r = b300.native
    if case == "status": r["status"] = "numerical_failure"
    elif case == "identity": r["identity"] = "other"
    elif case == "source": r["source_commit"] = "f" * 40
    elif case == "coordinate": r["attention_degree"] = 2
    elif case == "family": r["hardware"]["device_kinds"] = ["NVIDIA RTX A6000"] * 4
    elif case == "cursor": r["resumed"]["data_start_sequence"] = 0
    elif case == "state": r["resumed"]["checkpoint_audit"]["start_state_sha256"] = "f" * 64
    elif case == "rng": r["resumed"]["checkpoint_audit"]["first_update"]["rng_sha256"] = "f" * 64
    elif case == "validation": r["validation"]["validation_nll"] = float("nan")
    elif case == "runtime": r["runtime"]["segments"][0]["projected_seconds"] += 1
    b300.common.write_json(b300.path, r)
    with pytest.raises((ValueError, RuntimeError)):
        b300.common.load_b300_qualification(b300.policy, b300.manifest, b300.run)


def test_b300_reused_kernel_reference_must_remain_unchanged(b300):
    b300.source_path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        b300.common.load_b300_qualification(b300.policy, b300.manifest, b300.run)


def test_fast_runtime_clears_a6000_environment(b300, monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    for key in ("NCCL_P2P_DISABLE", "POLYNOMIAL_ATTENTION_HOST_STAGING", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        monkeypatch.setenv(key, "1")
    env = b300.common.training_env(b300.policy)
    assert env["PYTHONPATH"] == b300.policy["training_root"]
    assert env["POLYNOMIAL_ATTENTION_SOURCE_COMMIT"] == b300.manifest["source_commit"]
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert not {"NCCL_P2P_DISABLE", "POLYNOMIAL_ATTENTION_HOST_STAGING", "XLA_PYTHON_CLIENT_MEM_FRACTION"} & env.keys()


@pytest.mark.parametrize("names,expected", [(["NVIDIA H100"] * 4, "h100"), (["NVIDIA B300 SXM6"] * 4, "b300"),
                                          (["NVIDIA RTX A6000"] * 4, None), (["NVIDIA H100"] * 2 + ["NVIDIA B300"] * 2, None)])
def test_actual_gpu_detection_rejects_weaker_and_mixed_allocations(b300, monkeypatch, names, expected):
    import jax
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(device_kind=name, platform="gpu") for name in names])
    monkeypatch.setenv("SLURMD_NODENAME", "dogo" if expected == "h100" else "akita")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "sweep")
    profile = {"gpu_type": "h100_or_better", "allowed_gpu_types": ["h100", "b300"],
               "allowed_nodes": ["dogo", "mastiff", "akita"], "gpu_count": 4, "partition": "sweep"}
    if expected:
        assert b300.common.check_hardware(profile)["actual_profile"] == expected
    else:
        with pytest.raises(RuntimeError, match="hardware profile"):
            b300.common.check_hardware(profile)


@pytest.mark.parametrize("family", ["h100", "b300"])
def test_flexible_production_restores_existing_progress_and_records_actual_family(setup_production, monkeypatch, family):
    f = setup_production
    f.policy["assignments"][f.run["name"]] = "h100_or_better"
    f.policy["profiles"]["h100_or_better"] = {"gpu_type": "h100_or_better", "gpu_count": 4,
        "allowed_gpu_types": ["h100", "b300"], "allowed_nodes": ["dogo", "mastiff", "akita"],
        "partition": "sweep", "production_time": "08:00:00"}
    path = Path(f.manifest["preflight_dir"]) / "checks" / f"{f.run['name']}.json"
    qualified = read(path)
    qualified["runtime"]["segments"] = [{"start_step": 0, "end_step": 1000, "projected_seconds": 20000.}]
    f.module.common.write_json(path, qualified)
    b300_proof = {**qualified, "identity": "original-hardware-policy", "qualification_reference": {"path": "/check", "sha256": "a" * 64},
                  "kernel_qualification_reference": {"path": "/kernel", "sha256": "b" * 64}}
    monkeypatch.setattr(f.module.common, "load_b300_qualification", lambda *_: b300_proof)
    (Path(f.run["checkpoint_dir"]) / "500").mkdir(parents=True)
    f.policy["minimum_checkpoint_step"] = 500
    handoff = {"format": "key-dimension-taylor-checkpoint-handoff", "policy_identity": f.policy["identity"],
               "source_commit": f.manifest["source_commit"], "run_name": f.run["name"], "checkpoint_step": 500,
               "checkpoint_path": str(Path(f.run["checkpoint_dir"]) / "500"), "checkpoint_preserved": True}
    f.module.common.write_json(f.root / "handoff.json", handoff)
    old_event = {"profile": "a6000", "runtime_commit": "c" * 40}
    f.module.common.write_json(f.root / "launch.json", {"source_commit": f.manifest["source_commit"], "execution_history": [old_event]})
    monkeypatch.setattr(f.module, "_hardware_worker", lambda *_: {"actual_profile": family, "device_kinds": [family] * 4, "gpu_count": 4})
    calls = mock_workers(f, monkeypatch)
    result = run(f, 0)
    assert result["start_step"] == 500 and result["final_step"] == 1000
    assert [args[args.index("--worker") + 1] for args, _ in calls] == ["restore", "train"]
    assert all(kwargs["cwd"] == f.policy["training_root"] for _, kwargs in calls)
    launch = read(f.root / "launch.json")
    assert launch["execution_history"][0] == old_event
    assert launch["active_execution"]["profile"] == family
    assert launch["active_execution"]["requested_profile"] == "h100_or_better"
    receipt = read(f.root / "segments/receipt-000.json")
    assert receipt["profile"] == family and receipt["requested_profile"] == "h100_or_better"
    assert receipt["runtime_commit"] == f.manifest["source_commit"]
    assert receipt["handoff_reference"]["sha256"] == hashlib.sha256((f.root / "handoff.json").read_bytes()).hexdigest()


def test_flexible_dispatch_requires_actual_family(setup_production):
    f = setup_production
    f.policy["assignments"][f.run["name"]] = "h100_or_better"
    with pytest.raises(RuntimeError, match="detected GPU family"):
        f.module._segment(f.policy, f.manifest, f.run, 0)


@pytest.mark.parametrize("case", ["missing_checkpoint", "wrong_policy", "rollback"])
def test_handoff_cannot_silently_restart_or_roll_back(setup_production, monkeypatch, case):
    f = setup_production
    f.policy["assignments"][f.run["name"]] = "h100_or_better"
    f.policy["minimum_checkpoint_step"] = 500
    handoff = {"format": "key-dimension-taylor-checkpoint-handoff", "policy_identity": f.policy["identity"],
               "source_commit": f.manifest["source_commit"], "run_name": f.run["name"], "checkpoint_step": 500,
               "checkpoint_path": str(Path(f.run["checkpoint_dir"]) / "500"), "checkpoint_preserved": True}
    if case == "wrong_policy":
        handoff["policy_identity"] = "another-policy"
    if case != "missing_checkpoint":
        (Path(f.run["checkpoint_dir"]) / ("250" if case == "rollback" else "500")).mkdir(parents=True)
    f.module.common.write_json(f.root / "handoff.json", handoff)
    path = Path(f.manifest["preflight_dir"]) / "checks" / f"{f.run['name']}.json"
    qualified = read(path)
    qualified["runtime"]["segments"] = [{"start_step": 0, "end_step": 1000, "projected_seconds": 20000.}]
    f.module.common.write_json(path, qualified)
    monkeypatch.setattr(f.module, "_hardware_worker", lambda *_: {"actual_profile": "h100"})
    calls = mock_workers(f, monkeypatch)
    with pytest.raises(RuntimeError, match="checkpoint|handoff"):
        run(f, 0)
    assert all(args[args.index("--worker") + 1] != "train" for args, _ in calls)
    if case == "rollback":
        assert [args[args.index("--worker") + 1] for args, _ in calls] == ["restore"]
