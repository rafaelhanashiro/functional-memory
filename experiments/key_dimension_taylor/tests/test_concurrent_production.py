"""Exercise allocation routing and exact restarts without launching training."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor.production import validate_segment_restart

SCRIPT = Path(__file__).resolve().parents[1] / "concurrent/production.py"


def read(path):
    return json.loads(Path(path).read_text())


def update(step):
    return {"step": step, "batch_sha256": f"{step:064x}", "rng_sha256": f"{step + 100:064x}"}


def outcome(start, end):
    return {"start_step": start, "restored_step": start or None, "final_step": end,
            "checkpoint_audit": {"start_state_sha256": f"{start:064x}", "final_state_sha256": f"{end:064x}",
                                 "checkpoint_state_sha256": f"{end:064x}", "first_update": update(start + 1),
                                 "next_update": update(end + 1)}}


@pytest.fixture
def setup_production(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("production_common_fixture", SCRIPT.with_name("common.py"))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    monkeypatch.setitem(sys.modules, "common", common)
    spec = importlib.util.spec_from_file_location("production_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = {"name": "200m-dk128-m016-seed0", "qk_head_dim": 128, "attention_degree": 16,
           "workdir": str(tmp_path / "scientific"), "checkpoint_dir": str(tmp_path / "checkpoints"),
           "config": {"optim": {"total_steps": 1000}}}
    manifest = {"identity": "scientific-identity", "source_commit": "a" * 40, "runs": [run],
                "project_root": str(tmp_path), "preflight_dir": str(tmp_path / "preflight")}
    policy = {"identity": "execution-identity", "source_commit": manifest["source_commit"],
              "orchestrator_commit": "b" * 40, "training_root": str(tmp_path / "immutable"),
              "training_manifest": str(tmp_path / "original-manifest.json"),
              "orchestrator_root": str(tmp_path / "orchestrator"),
              "qualification_root": str(tmp_path / "qualification"),
              "assignments": {run["name"]: "h100"},
              "profiles": {kind: {"gpu_type": kind, "gpu_count": 4, "node": node,
                                  "partition": "sweep", "production_time": "08:00:00"}
                           for kind, node in (("h100", "dogo"), ("b300", "akita"))}}
    policy_path = tmp_path / "policy.json"
    common.write_json(policy_path, policy)
    common.write_json(Path(policy["training_manifest"]), manifest)
    runtime = {"limit_seconds": 28800, "usable_seconds": 23040., "allocation_fraction": .8,
               "segments": [{"start_step": 0, "end_step": 500, "projected_seconds": 20000.},
                            {"start_step": 500, "end_step": 1000, "projected_seconds": 20000.}]}
    qualified = {"identity": manifest["identity"], "source_commit": manifest["source_commit"], "status": "passed",
                 "qk_head_dim": 128, "attention_degree": 16, "runtime": runtime}
    common.write_json(Path(manifest["preflight_dir"]) / "success.json", qualified)
    common.write_json(Path(manifest["preflight_dir"]) / "checks" / f"{run['name']}.json", qualified)
    b300 = {**qualified, "identity": policy["identity"], "policy_identity": policy["identity"],
            "orchestrator_commit": policy["orchestrator_commit"], "profile": "b300", "scheduler": policy["profiles"]["b300"]}
    common.write_json(Path(policy["qualification_root"]) / "b300/kernels.json", {**b300, "tests_passed": 105})
    common.write_json(Path(policy["qualification_root"]) / "b300/checks" / f"{run['name']}.json", b300)
    monkeypatch.setattr(common, "verify_policy", lambda *_: None)
    monkeypatch.setattr(common, "assert_legacy_jobs_terminal", lambda *_: None)
    monkeypatch.setattr(common, "training_env", lambda policy: {"PYTHONPATH": policy["training_root"],
                       "POLYNOMIAL_ATTENTION_SOURCE_COMMIT": policy["source_commit"]})
    # Any in-parent device initialization must fail this fixture.
    monkeypatch.setattr(common, "check_hardware", lambda *_: pytest.fail("Hardware must be checked in a subprocess"))
    reports, retained, restarts = [], [], []
    monkeypatch.setattr(common, "render_report", lambda *_: reports.append((Path(run["workdir"]) / "production_failure.json").exists()))
    def validate(previous, restored, expected):
        restarts.append((previous, restored, expected))
        validate_segment_restart(previous, restored, expected)
    monkeypatch.setattr(module, "_old_helpers", lambda *_: SimpleNamespace(validate_segment_restart=validate,
                        finalize_run=lambda *args: retained.append(args)))
    monkeypatch.setenv("SLURM_JOB_ID", "210000")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "dogo")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "sweep")
    return SimpleNamespace(module=module, policy=policy, manifest=manifest, run=run, policy_path=policy_path,
                           root=Path(run["workdir"]), reports=reports, retained=retained, restarts=restarts)


def mock_workers(fixture, monkeypatch, *, fail=None, corrupt=None):
    calls = []
    def execute(args, **kwargs):
        # The real lock descriptor must remain open through every subprocess.
        assert len(kwargs["pass_fds"]) == 1
        os.fstat(kwargs["pass_fds"][0])
        assert "start_new_session" not in kwargs
        calls.append((args, kwargs))
        worker = args[args.index("--worker") + 1]
        output = Path(args[args.index("--output") + 1])
        if worker == "hardware":
            result = {"node": "dogo", "partition": "sweep", "gpu_count": 4, "device_kinds": ["NVIDIA H100"] * 4}
        else:
            if fail and worker == "train":
                fixture.module.common.write_json(output, {"error": "fixture failure", "numerical_failure": fail == "numerical"})
                return subprocess.CompletedProcess(args, 1)
            checkpoints = Path(fixture.run["checkpoint_dir"])
            current = max((int(path.name) for path in checkpoints.glob("*") if path.is_dir()), default=0)
            if worker == "restore":
                result = {"restored_step": current, "state_sha256": f"{current:064x}", "next_update": update(current + 1)}
            elif (fixture.root / "completion.json").exists():
                result = read(fixture.root / "completion.json")
            else:
                end = int(args[args.index("--end-step") + 1])
                result = outcome(current, end)
                (checkpoints / str(end)).mkdir(parents=True, exist_ok=True)
                if end == fixture.run["config"]["optim"]["total_steps"]:
                    fixture.module.common.write_json(fixture.root / "completion.json", result)
            if corrupt:
                corrupt(worker, result)
        fixture.module.common.write_json(output, result)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(fixture.module.subprocess, "run", execute)
    return calls


def run(fixture, index=0):
    return fixture.module.run_segment(fixture.policy, fixture.manifest, fixture.run["name"], index,
                                      policy_path=fixture.policy_path)


def test_routing_provenance_exact_restart_and_completed_idempotence(setup_production, monkeypatch):
    f = setup_production
    original = Path(f.policy["training_manifest"]).read_bytes()
    old_launch = {"source_commit": f.manifest["source_commit"], "segments": [{"job_id": "200515"}]}
    f.module.common.write_json(f.root / "launch.json", old_launch)
    calls = mock_workers(f, monkeypatch)
    assert run(f)["final_step"] == 500
    assert run(f, 1)["final_step"] == 1000
    completed = (f.root / "completion.json").read_bytes()
    launch_before = (f.root / "launch.json").read_bytes()
    assert run(f, 1)["final_step"] == 1000
    assert len(f.restarts) == 1 and len(f.retained) == 2
    assert [args[args.index("--worker") + 1] for args, _ in calls] == ["hardware", "train", "hardware", "restore", "train", "hardware", "train"]
    for args, opts in calls:
        if "hardware" in args:
            assert args[1] == str(SCRIPT) and opts["cwd"] == f.policy["orchestrator_root"]
        else:
            assert args[1:3] == ["-m", "experiments.key_dimension_taylor.production"]
            assert args[args.index("--manifest") + 1] == f.policy["training_manifest"]
            assert opts["cwd"] == f.policy["training_root"]
        assert opts["env"]["PYTHONPATH"] == f.policy["training_root"]
        assert opts["env"]["POLYNOMIAL_ATTENTION_SOURCE_COMMIT"] == f.manifest["source_commit"]
    launch = read(f.root / "launch.json")
    assert launch["segments"] == old_launch["segments"] and len(launch["execution_history"]) == 2
    assert launch["log_path"].endswith("slurm-210000.out")
    receipt = read(f.root / "segments/receipt-001.json")
    assert receipt["source_commit"] == f.manifest["source_commit"]
    assert receipt["orchestrator_commit"] == f.policy["orchestrator_commit"]
    assert receipt["actual_node"] == "dogo" and receipt["actual_partition"] == "sweep"
    assert receipt["hardware"]["gpu_count"] == 4
    assert len(list((f.root / "receipts").glob("*.json"))) == 3
    assert (f.root / "completion.json").read_bytes() == completed
    assert (f.root / "launch.json").read_bytes() == launch_before
    assert Path(f.policy["training_manifest"]).read_bytes() == original


def test_partial_requeue_restores_inside_segment_and_clears_failure_before_report(setup_production, monkeypatch):
    f = setup_production
    (Path(f.run["checkpoint_dir"]) / "750").mkdir(parents=True)
    f.module.common.write_json(f.root / "production_failure.json", {"failure_kind": "infrastructure"})
    calls = mock_workers(f, monkeypatch)
    result = run(f, 1)
    assert result["start_step"] == 750 and not f.restarts
    assert [args[args.index("--worker") + 1] for args, _ in calls] == ["hardware", "restore", "train"]
    assert f.reports == [False] and not (f.root / "production_failure.json").exists()


@pytest.mark.parametrize("fault", ["state", "next_batch", "durable", "restored_hash", "first_update"])
def test_exact_audits_reject_changed_state_or_cursor(setup_production, monkeypatch, fault):
    f = setup_production
    (Path(f.run["checkpoint_dir"]) / "500").mkdir(parents=True)
    previous = outcome(0, 500)
    if fault == "state":
        previous["checkpoint_audit"]["final_state_sha256"] = "f" * 64
    if fault == "next_batch":
        previous["checkpoint_audit"]["next_update"]["batch_sha256"] = "f" * 64
    f.module.common.write_json(f.root / "segments/segment-000.json", previous)
    def corrupt(worker, result):
        if worker == "train" and fault in ("durable", "restored_hash", "first_update"):
            audit = result["checkpoint_audit"]
            if fault == "durable":
                audit["checkpoint_state_sha256"] = "f" * 64
            elif fault == "restored_hash":
                audit["start_state_sha256"] = "f" * 64
            else:
                audit["first_update"]["rng_sha256"] = "f" * 64
    calls = mock_workers(f, monkeypatch, corrupt=corrupt)
    with pytest.raises(RuntimeError):
        run(f, 1)
    failure = read(f.root / "production_failure.json")
    assert failure["failure_kind"] == "infrastructure" and not f.retained
    if fault in ("state", "next_batch"):
        assert not any("train" in args for args, _ in calls)


@pytest.mark.parametrize("failure_kind", ["numerical", "infrastructure"])
def test_worker_failure_classification_and_launcher_preservation(setup_production, monkeypatch, failure_kind):
    f = setup_production
    mock_workers(f, monkeypatch, fail=failure_kind)
    with pytest.raises((FloatingPointError, subprocess.CalledProcessError)):
        run(f)
    failure = read(f.root / "production_failure.json")
    assert failure["failure_kind"] == failure_kind and failure["stage"] == "train"
    preserved = f.module.record_interruption(f.policy, f.manifest, f.run["name"], 0, 1)
    assert preserved == failure
    assert len(list((f.root / "failure_history").glob("*.json"))) == 1


def test_numerically_failed_coordinate_is_not_retried(setup_production, monkeypatch):
    f = setup_production
    prior = {"failure_kind": "numerical", "slurm_job_id": "200515"}
    f.module.common.write_json(f.root / "production_failure.json", prior)
    calls = mock_workers(f, monkeypatch)
    with pytest.raises(f.module.ExistingNumericalFailure):
        run(f)
    assert calls == [] and read(f.root / "production_failure.json") == prior
    assert read(next((f.root / "failure_history").glob("*.json")))["status"] == "suppressed"


def test_launcher_only_failure_records_signal_and_attempt(setup_production):
    f = setup_production
    result = f.module.record_interruption(f.policy, f.manifest, f.run["name"], 0, 143)
    assert result["exit_code"] == 143 and result["failure_kind"] == "infrastructure"
    assert read(next((f.root / "execution_attempts").glob("*/attempt.json"))) == result


def test_live_legacy_owner_prevents_execution_or_provenance_writes(setup_production, monkeypatch):
    f = setup_production
    def reject(*_):
        raise RuntimeError("Legacy job still running")
    monkeypatch.setattr(f.module.common, "assert_legacy_jobs_terminal", reject)
    calls = mock_workers(f, monkeypatch)
    with pytest.raises(RuntimeError, match="Legacy job"):
        run(f)
    assert not calls and not (f.root / "execution_attempts").exists()
    assert not (f.root / "launch.json").exists()


def test_b300_requires_individual_qualification_but_not_whole_profile_completion(setup_production):
    f = setup_production
    f.policy["assignments"][f.run["name"]] = "b300"
    segment, qualified = f.module._segment(f.policy, f.manifest, f.run, 0)
    assert segment["end_step"] == 500 and qualified["profile"] == "b300"
    assert not (Path(f.policy["qualification_root"]) / "b300/success.json").exists()


@pytest.mark.parametrize("fault", ["coordinate", "identity", "kernels", "budget", "seconds", "alignment", "gap"])
def test_b300_rejects_unqualified_or_oversized_segments(setup_production, fault):
    f = setup_production
    f.policy["assignments"][f.run["name"]] = "b300"
    path = Path(f.policy["qualification_root"]) / "b300/checks" / f"{f.run['name']}.json"
    record = read(path)
    if fault == "coordinate":
        record["qk_head_dim"] = 64
    elif fault == "identity":
        record["identity"] = "another-policy"
    elif fault == "kernels":
        kernels_path = path.parent.parent / "kernels.json"
        kernels = read(kernels_path)
        kernels["tests_passed"] = 104
        f.module.common.write_json(kernels_path, kernels)
    elif fault == "budget":
        record["runtime"]["allocation_fraction"] = .9
    elif fault == "seconds":
        record["runtime"]["segments"][0]["projected_seconds"] = 23041.
    elif fault == "alignment":
        record["runtime"]["segments"][0]["end_step"] = 499
    else:
        record["runtime"]["segments"][1]["start_step"] = 501
    f.module.common.write_json(path, record)
    with pytest.raises(RuntimeError):
        f.module._segment(f.policy, f.manifest, f.run, 0)
