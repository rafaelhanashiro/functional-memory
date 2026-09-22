"""A6000 evidence must bind the original recipe to its qualified runtime."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor import preflight
from experiments.key_dimension_taylor.tests.test_concurrent_production import (
    mock_workers, outcome, read, run, setup_production,
)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "concurrent/common.py"
    spec = importlib.util.spec_from_file_location("a6000_evidence_common", path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    monkeypatch.setattr(common, "_verify_checkout", lambda *_: None)
    monkeypatch.setattr(common, "_original_preflight", lambda *_: preflight)
    original_root, runtime_root = tmp_path / "original", tmp_path / "runtime"
    for root in (original_root, runtime_root):
        for name, body in (("models/components/attention.py", "blocked"),
                           ("models/components/fused_attention.py", "fused"), ("uv.lock", "lock")):
            target = root / "experiments/polynomial_attention" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)

    def reference(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(name, value):
        path = tmp_path / name
        common.write_json(path, value)
        return reference(path)

    config = {"model": {"qk_head_dim": 128, "attention_degree": 16, "attention_backend": "fused"},
              "data": {"batch_size": 256, "train_manifest": "/train", "eval_manifest": "/eval"},
              "train": {"source_commit": "a" * 40, "workdir": str(tmp_path / "canonical"),
                        "eval_every": 1000, "eval_batches": 7, "ckpt_every": 500, "wandb_mode": "disabled"},
              "optim": {"total_steps": 19074}}
    coordinate = {"name": "200m-dk128-m016-seed0", "qk_head_dim": 128,
                  "attention_degree": 16, "parameter_count": 123456, "config": config}
    data = {key: {"sha256": key} for key in ("train_manifest", "eval_manifest")}
    manifest = {"identity": "original-scientific-identity", "source_commit": "a" * 40,
                "data_identity": data, "dependency_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
                "project_root": str(tmp_path), "runs": [coordinate]}
    manifest_path = write("manifest.json", manifest)["path"]
    config_ref = write("original-config.json", config)
    validation_run = {**coordinate, "config_path": config_ref["path"], "config_sha256": config_ref["sha256"]}
    profile = {"gpu_type": "a6000", "gpu_count": 4, "allowed_nodes": ["doodle", "terrier"],
               "partition": "sweep", "production_time": "08:00:00"}
    original_policy = {"production_submission": False, "training_identity": manifest["identity"],
                       "source_commit": manifest["source_commit"], "training_manifest": manifest_path,
                       "training_root": str(original_root), "orchestrator_root": str(tmp_path / "old_orchestrator"),
                       "orchestrator_commit": "b" * 40, "runs": [validation_run],
                       "dependency_lock_sha256": manifest["dependency_lock_sha256"], "data_identity": data}
    original_policy["identity"] = common.policy_identity(original_policy)
    kernels = {"status": "passed", "tests_passed": 105, "identity": original_policy["identity"],
               "profile": "a6000", "source_commit": manifest["source_commit"],
               "orchestrator_commit": original_policy["orchestrator_commit"], "eligible_runs": [coordinate["name"]]}
    policy = {"source_commit": manifest["source_commit"], "training_manifest": manifest_path,
              "training_root": str(original_root), "runtime_root": str(runtime_root), "runtime_commit": "c" * 40,
              "runtime_environment": common.A6000_RUNTIME_ENVIRONMENT.copy(),
              "assignments": {coordinate["name"]: "a6000"}, "profiles": {"a6000": profile}}
    attempt = tmp_path / "qualification-attempt"
    resolved = copy.deepcopy(config)
    resolved["train"].update(source_commit=policy["runtime_commit"], workdir=str(attempt / "workdir"),
                             eval_every=32, eval_batches=1)
    hardware = {"gpu_count": 4, "device_kinds": ["NVIDIA RTX A6000"] * 4, "node": "doodle", "partition": "sweep"}
    initial, resumed = outcome(0, 32), outcome(32, 33)
    for result, start in ((initial, 0), (resumed, 32)):
        result.update(data_start_sequence=start * 256, data_parallel_devices=4, attention_degree=16,
                      hardware=hardware, resolved_config=copy.deepcopy(resolved),
                      timing={"runtime_seconds": 50., "train_step_seconds": [20., 1., 1.],
                              "evaluation_seconds": [3.], "checkpoint_seconds": [1.]})
    h100 = {"status": "passed", "identity": manifest["identity"], "source_commit": manifest["source_commit"],
            "initial": {"checkpoint_audit": copy.deepcopy(initial["checkpoint_audit"])}}
    validation = {"step": 32, "source_commit": policy["runtime_commit"], "qk_head_dim": 128,
                  "attention_degree": 16, "parameter_count": 123456, "validation_nll": 4.,
                  "validation_perplexity": 54.598, "attention_diagnostics": {"valid": True}}
    record = {"format": "key-dimension-taylor-a6000-host-staging", "status": "passed", "validation_only": True,
              "production_submission": False, "runtime_commit": policy["runtime_commit"],
              "runtime_root": policy["runtime_root"], "original_scientific_source_commit": manifest["source_commit"],
              **{key: coordinate[key] for key in ("name", "qk_head_dim", "attention_degree", "parameter_count")},
              "environment": {**common.A6000_RUNTIME_ENVIRONMENT, "POLYNOMIAL_ATTENTION_SOURCE_COMMIT": policy["runtime_commit"],
                              "PYTHONPATH": policy["runtime_root"], "JAX_PLATFORMS": "cuda", "XLA_FLAGS": None},
              "original_policy": write("old-policy.json", original_policy),
              "original_policy_identity": original_policy["identity"], "resolved_original_config": config,
              "original_config": config_ref, "kernel_reference": write("kernels.json", kernels),
              "data_identity": data, "dependency_lock_sha256": manifest["dependency_lock_sha256"],
              "attempt_path": str(attempt), "initial": initial, "resumed": resumed,
              "initial_last_committed_validation": validation, "resumed_last_committed_validation": copy.deepcopy(validation),
              "h100_initialization_and_input_parity": {
                  "status": "passed", "reference": write("h100.json", h100),
                  **{key: initial["checkpoint_audit"][key] for key in ("start_state_sha256", "first_update")}},
              "runtime": preflight.projected_runtime(coordinate, initial, resumed, profile)}

    def seal():
        policy["a6000_qualifications"] = {coordinate["name"]: write("qualification.json", record)}

    seal()
    return SimpleNamespace(common=common, policy=policy, manifest=manifest, run=coordinate,
                           record=record, seal=seal, reference=reference)


def load(e):
    return e.common.load_a6000_qualification(e.policy, e.manifest, e.run)


def test_original_evidence_is_accepted_without_relabeling(evidence):
    assert load(evidence) == evidence.record
    assert "identity" not in load(evidence)


@pytest.mark.parametrize("reference", ["original_policy", "original_config", "kernel_reference"])
def test_changed_nested_reference_is_rejected(evidence, reference):
    Path(evidence.record[reference]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load(evidence)


def test_changed_qualification_reference_is_rejected(evidence):
    Path(evidence.policy["a6000_qualifications"][evidence.run["name"]]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load(evidence)


@pytest.mark.parametrize("kind", ["runtime", "coordinate", "status", "environment", "data", "lock",
                                  "restore", "cursor", "state", "rng", "worker_config", "hardware",
                                  "evaluation", "diagnostics", "h100", "runtime_segments", "timing"])
def test_bad_qualification_details_are_rejected(evidence, kind):
    r = evidence.record
    if kind == "runtime": r["runtime_commit"] = "d" * 40
    elif kind == "coordinate": r["qk_head_dim"] = 32
    elif kind == "status": r["status"] = "numerical_failure"
    elif kind == "environment": r["environment"]["NCCL_P2P_DISABLE"] = "0"
    elif kind == "data": r["data_identity"] = {}
    elif kind == "lock": r["dependency_lock_sha256"] = "e" * 64
    elif kind == "restore": r["resumed"]["restored_step"] = 0
    elif kind == "cursor": r["resumed"]["data_start_sequence"] = 0
    elif kind == "state": r["resumed"]["checkpoint_audit"]["start_state_sha256"] = "f" * 64
    elif kind == "rng": r["resumed"]["checkpoint_audit"]["first_update"]["rng_sha256"] = "f" * 64
    elif kind == "worker_config": r["resumed"]["resolved_config"]["optim"]["total_steps"] = 33
    elif kind == "hardware": r["resumed"]["hardware"]["gpu_count"] = 1
    elif kind == "evaluation": r["initial_last_committed_validation"]["validation_nll"] = float("nan")
    elif kind == "diagnostics": r["initial_last_committed_validation"]["attention_diagnostics"]["valid"] = False
    elif kind == "h100": r["h100_initialization_and_input_parity"]["first_update"] = {}
    elif kind == "runtime_segments": r["runtime"]["segments"][0]["end_step"] = 19000
    elif kind == "timing": r["initial"]["timing"]["train_step_seconds"][1] = 2.
    evidence.seal()
    with pytest.raises((ValueError, RuntimeError)):
        load(evidence)


@pytest.mark.parametrize("relative", ["models/components/fused_attention.py", "uv.lock"])
def test_changed_runtime_kernel_or_lock_is_rejected(evidence, relative):
    (Path(evidence.policy["runtime_root"]) / "experiments/polynomial_attention" / relative).write_text("changed")
    with pytest.raises(ValueError, match="changed|differs"):
        load(evidence)


def test_runtime_environment_and_provenance_are_explicit(evidence, monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    env = evidence.common.training_env(evidence.policy)
    assert env["PYTHONPATH"] == evidence.policy["runtime_root"]
    assert env["POLYNOMIAL_ATTENTION_SOURCE_COMMIT"] == evidence.policy["runtime_commit"]
    assert all(env[key] == value for key, value in evidence.common.A6000_RUNTIME_ENVIRONMENT.items())
    proof = evidence.common.execution_provenance(evidence.policy)
    assert proof["original_scientific_source_commit"] == evidence.manifest["source_commit"]
    assert proof["runtime_commit"] != proof["original_scientific_source_commit"]
    monkeypatch.setenv("XLA_FLAGS", "--changed")
    with pytest.raises(ValueError, match="XLA_FLAGS"):
        evidence.common.training_env(evidence.policy)


def test_a6000_production_keeps_existing_locks_and_resume_audits(setup_production, monkeypatch):
    f = setup_production
    f.policy.update(runtime_root=str(f.root / "qualified-runtime"), runtime_commit="c" * 40,
                    runtime_environment=f.module.common.A6000_RUNTIME_ENVIRONMENT.copy(),
                    a6000_qualifications={f.run["name"]: {"path": "/original/check.json", "sha256": "d" * 64}})
    f.policy["assignments"][f.run["name"]] = "a6000"
    f.policy["profiles"]["a6000"] = {"gpu_type": "a6000", "gpu_count": 4, "partition": "sweep",
                                          "allowed_nodes": ["doodle", "terrier"], "production_time": "08:00:00"}
    qualification = read(Path(f.manifest["preflight_dir"]) / "checks" / f"{f.run['name']}.json")
    qualification.pop("identity")
    qualification["original_policy_identity"] = "original-validation-identity"
    monkeypatch.setattr(f.module.common, "load_a6000_qualification", lambda *_: qualification)
    monkeypatch.setenv("SLURMD_NODENAME", "doodle")
    calls = mock_workers(f, monkeypatch)
    assert run(f, 0)["final_step"] == 500
    assert run(f, 1)["final_step"] == 1000
    assert len(f.restarts) == 1
    for command, kwargs in calls:
        if command[command.index("--worker") + 1] != "hardware":
            assert kwargs["cwd"] == f.policy["runtime_root"]
    metadata = read(f.root / "metadata.json")
    assert metadata["source_commit"] == f.manifest["source_commit"]
    assert metadata["runtime_commit"] == f.policy["runtime_commit"]
    assert metadata["qualification_reference"] == f.policy["a6000_qualifications"][f.run["name"]]
    receipt = read(f.root / "segments/receipt-001.json")
    assert receipt["qualification_identity"] == "original-validation-identity"
    assert receipt["runtime_commit"] == f.policy["runtime_commit"]
