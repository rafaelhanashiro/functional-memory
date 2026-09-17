import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.polynomial_attention import preflight


def fake_jax(monkeypatch, kinds, backend="gpu"):
    devices = [SimpleNamespace(device_kind=kind) for kind in kinds]
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(
        local_devices=lambda: devices, default_backend=lambda: backend,
    ))


@pytest.mark.parametrize("count", [2, 4])
def test_hardware_matches_selected_gpu_profile(monkeypatch, count):
    fake_jax(monkeypatch, ["NVIDIA H100 80GB HBM3"] * count)
    result = preflight.worker_result({"scheduler": {"gpu_type": "h100", "gpu_count": count}}, "hardware", "", None, 2)
    assert len(result["devices"]) == count


@pytest.mark.parametrize("kinds,backend", [
    (["NVIDIA H200"] * 2, "gpu"),
    (["NVIDIA H100"] * 4, "gpu"),
    (["NVIDIA H100"] * 2, "cpu"),
])
@pytest.mark.parametrize("check_only", [False, True])
def test_cached_success_cannot_hide_wrong_allocation(tmp_path, monkeypatch, kinds, backend, check_only):
    manifest = {
        "scheduler": {"gpu_type": "h100", "gpu_count": 2},
        "preflight_dir": str(tmp_path), "identity": "test",
    }
    fake_jax(monkeypatch, kinds, backend)
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    monkeypatch.setattr(preflight, "preflight_complete", lambda manifest: True)
    monkeypatch.setattr(preflight, "launch_worker", lambda manifest, worker, output:
                        preflight.worker_result(manifest, worker, "", None, 2))
    with pytest.raises(RuntimeError, match="requires 2 H100"):
        preflight.run_preflight(manifest, check_only=check_only, check_hardware=True)
    failure = json.loads(next((tmp_path / "attempts").glob("*/failure.json")).read_text())
    assert failure["stage"] == "hardware"


def test_login_submission_checks_cached_success_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    monkeypatch.setattr(preflight, "preflight_complete", lambda manifest: True)
    monkeypatch.setattr(preflight, "launch_worker", lambda *args, **kwargs: pytest.fail("Must not probe GPUs on login node"))
    preflight.run_preflight({"preflight_dir": str(tmp_path)}, check_only=True)
    assert not (tmp_path / "attempts").exists()


def test_modified_scheduler_invalidates_manifest_identity():
    manifest = {
        "source_commit": "a" * 40, "dependency_lock_sha256": "lock",
        "data_identity": {}, "scheduler": {"gpu_type": "h100", "gpu_count": 2},
        "runs": [{"config": {}}],
    }
    payload = {key: value for key, value in manifest.items() if key != "runs"}
    payload["configs"] = [{}]
    manifest["identity"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    manifest["scheduler"]["gpu_count"] = 4
    with pytest.raises(ValueError, match="identity does not match"):
        preflight.verify_manifest(manifest)


def runtime_inputs():
    run = {"name": "pilot", "config": {
        "optim": {"total_steps": 100},
        "train": {"eval_every": 40, "eval_batches": 7, "ckpt_every": 25},
    }}
    outcome = {"timing": {
        "runtime_seconds": 120.0, "train_step_seconds": [100.0, 2.0],
        "evaluation_seconds": [3.0], "checkpoint_seconds": [4.0],
    }}
    return run, outcome


def test_runtime_projection_excludes_compile_and_counts_terminal_overheads():
    run, outcome = runtime_inputs()
    result = preflight.projected_runtime(run, outcome)
    assert result["training_seconds"] == 200.0
    assert result["evaluation_seconds"] == 63.0  # Evaluations at 40, 80, 100.
    assert result["checkpoint_seconds"] == 24.0  # Saves at 25, 40, 50, 75, 80, 100.
    assert result["projected_total_seconds"] == 407.0


def test_runtime_gate_rejects_runs_exceeding_six_days():
    run, outcome = runtime_inputs()
    outcome["timing"]["train_step_seconds"][1] = preflight.MAX_PROJECTED_RUNTIME_SECONDS
    with pytest.raises(RuntimeError, match="beyond the six-day"):
        preflight.projected_runtime(run, outcome)


@pytest.mark.parametrize("field,value", [
    ("train_step_seconds", [100.0]),
    ("train_step_seconds", [100.0, float("nan")]),
    ("evaluation_seconds", []),
    ("checkpoint_seconds", [0.0]),
])
def test_runtime_gate_requires_valid_measurements(field, value):
    run, outcome = runtime_inputs()
    outcome["timing"][field] = value
    with pytest.raises(RuntimeError, match="no valid steady"):
        preflight.projected_runtime(run, outcome)
