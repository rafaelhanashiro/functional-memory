"""Qualification routing and isolation checks use synthetic subprocess results."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor import preflight as original_preflight

SCRIPT = Path(__file__).resolve().parents[1] / "concurrent/qualification.py"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup_qualification(tmp_path, monkeypatch):
    common = SimpleNamespace(
        verify_policy=lambda policy, manifest: None,
        training_env=lambda policy: {"JAX_PLATFORMS": "cuda", "PYTHONPATH": policy["training_root"]},
        write_json=write_json,
    )
    monkeypatch.setitem(sys.modules, "common", common)
    spec = importlib.util.spec_from_file_location("qualification_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    training = tmp_path / "immutable"
    orchestration = tmp_path / "orchestration"
    training.mkdir()
    orchestration.mkdir()
    runs = []
    for degree in (2, 16):
        name = f"200m-dk128-m{degree:03d}-seed0"
        runs.append({"name": name, "qk_head_dim": 128, "attention_degree": degree, "parameter_count": 220877824,
                     "retained_completed": False, "workdir": str(tmp_path / "scientific" / name),
                     "config": {"data": {"batch_size": 256}, "optim": {"total_steps": 19074},
                                "train": {"eval_every": 1000, "eval_batches": 7, "ckpt_every": 500}}})
    manifest = {"execution_root": str(training), "source_commit": "a" * 40, "runs": runs}
    manifest_path = tmp_path / "original-manifest.json"
    write_json(manifest_path, manifest)
    policy = {"identity": "policy-identity", "source_commit": "a" * 40, "orchestrator_commit": "b" * 40,
              "training_manifest": str(manifest_path), "training_root": str(training),
              "orchestrator_root": str(orchestration), "qualification_root": str(tmp_path / "qualification"),
              "profiles": {"b300": {"gpu_type": "b300", "gpu_count": 4, "node": "akita", "partition": "sweep", "production_time": "08:00:00"}},
              "assignments": {run["name"]: "b300" for run in runs}}
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    helpers = SimpleNamespace(
        validate_resume_audit=original_preflight.validate_resume_audit,
        projected_runtime=original_preflight.projected_runtime,
        RuntimeLimitExceeded=original_preflight.RuntimeLimitExceeded,
        require_passing_tests=original_preflight.require_passing_tests,
        committed_records=lambda root: [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()],
    )
    monkeypatch.setattr(module, "_load_original_helpers", lambda policy: helpers)
    return module, policy_path, policy, manifest


def audit(start, final):
    def update(step):
        return {"step": step, "batch_sha256": f"{step:064x}", "rng_sha256": f"{step + 100:064x}"}
    return {"start_state_sha256": f"{start:064x}", "final_state_sha256": f"{final:064x}",
            "checkpoint_state_sha256": f"{final:064x}", "first_update": update(start + 1), "next_update": update(final + 1)}


def mock_commands(monkeypatch, fixture, *, numerical_run=None, infrastructure_run=None, wrong_resume=False,
                  kernel_count=105, kernel_skip=False, step_seconds=.5):
    module, _, policy, manifest = fixture
    calls = []
    hardware = {"gpu_type": "b300", "gpu_count": 4, "node": "akita", "device_kinds": ["NVIDIA B300"] * 4}
    def command(args, **kwargs):
        calls.append((args, kwargs))
        if "--worker" in args and args[args.index("--worker") + 1] == "hardware":
            write_json(Path(args[args.index("--output") + 1]), hardware)
            return subprocess.CompletedProcess(args, 0)
        if "pytest" in args:
            junit = Path(next(arg.removeprefix("--junitxml=") for arg in args if arg.startswith("--junitxml=")))
            testcase = '<testcase name="case"/>'
            junit.write_text('<testsuite>' + testcase * (kernel_count - int(kernel_skip))
                             + ('<testcase name="skipped"><skipped/></testcase>' if kernel_skip else '') + '</testsuite>')
            return subprocess.CompletedProcess(args, 0)
        name = args[args.index("--run-name") + 1]
        run = next(run for run in manifest["runs"] if run["name"] == name)
        stop = int(args[args.index("--stop-after-step") + 1])
        workdir, output = Path(args[args.index("--workdir") + 1]), Path(args[args.index("--output") + 1])
        (workdir / "checkpoints" / str(stop)).mkdir(parents=True, exist_ok=True)
        if name in (numerical_run, infrastructure_run):
            numerical = name == numerical_run
            write_json(output, {"status": "failed", "numerical_failure": numerical, "error": "fixture failure"})
            if numerical:
                write_json(workdir / "numerical_failure_step_00000002.json", {"step": 2, "error": "nonfinite"})
            return subprocess.CompletedProcess(args, 1)
        if stop == 32:
            record = {"step": 32, "source_commit": policy["source_commit"], "qk_head_dim": run["qk_head_dim"],
                      "attention_degree": run["attention_degree"], "parameter_count": run["parameter_count"],
                      "validation_nll": 3., "validation_perplexity": 20.}
            (workdir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
        start = 32 if stop == 33 else 0
        outcome = {"start_step": start, "final_step": stop, "restored_step": 32 if start else None,
                   "data_start_sequence": start * 256, "data_parallel_devices": 4,
                   "attention_degree": run["attention_degree"], "checkpoint_audit": audit(start, stop),
                   "timing": {"runtime_seconds": 100., "train_step_seconds": [50., step_seconds],
                              "evaluation_seconds": [5.], "checkpoint_seconds": [10.]}}
        if wrong_resume and stop == 33:
            outcome["checkpoint_audit"]["start_state_sha256"] = "f" * 64
        write_json(output, outcome)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(module.subprocess, "run", command)
    return calls


def test_qualification_runs_only_frozen_kernels_and_workers(setup_qualification, monkeypatch):
    module, policy_path, policy, manifest = setup_qualification
    original_bytes = Path(policy["training_manifest"]).read_bytes()
    calls = mock_commands(monkeypatch, setup_qualification)
    result = module.run_qualification(policy_path, policy, manifest, "b300")
    assert result["eligible_runs"] == sorted(policy["assignments"])
    assert result["ineligible_runs"] == []
    kernels = next((args, opts) for args, opts in calls if "pytest" in args)
    assert kernels[1]["cwd"] == policy["training_root"]
    assert "experiments/key_dimension_taylor/tests/test_kernels.py" in kernels[0]
    assert kernels[0][kernels[0].index("-m", 3) + 1] == "gpu"
    pilots = [(args, opts) for args, opts in calls if "pilot" in args]
    assert len(pilots) == 4
    assert [int(args[args.index("--stop-after-step") + 1]) for args, _ in pilots] == [32, 33, 32, 33]
    for args, opts in pilots:
        assert opts["cwd"] == policy["training_root"]
        assert opts["env"]["PYTHONPATH"] == policy["training_root"]
        assert args[args.index("experiments.key_dimension_taylor.preflight") + 1] == policy["training_manifest"]
        assert "--worker" in args and "--workdir" in args
    hardware_args, hardware_opts = calls[0]
    assert hardware_opts["cwd"] == policy["orchestrator_root"] and hardware_args[1] == str(SCRIPT)
    root = Path(policy["qualification_root"]) / "b300"
    assert json.loads((root / "kernels.json").read_text())["tests_passed"] == 105
    assert (root / "success.json").exists()
    assert not list(root.glob("attempts/*/*/checkpoints"))
    assert Path(policy["training_manifest"]).read_bytes() == original_bytes
    assert all(not Path(run["workdir"]).exists() for run in manifest["runs"])


def test_numerical_failure_does_not_suppress_other_assigned_coordinates(setup_qualification, monkeypatch):
    module, policy_path, policy, manifest = setup_qualification
    rejected = manifest["runs"][0]["name"]
    calls = mock_commands(monkeypatch, setup_qualification, numerical_run=rejected)
    result = module.run_qualification(policy_path, policy, manifest, "b300")
    assert result["ineligible_runs"] == [rejected]
    assert result["eligible_runs"] == [manifest["runs"][1]["name"]]
    record = json.loads((Path(policy["qualification_root"]) / "b300/checks" / f"{rejected}.json").read_text())
    assert record["reason"] == "numerical"
    assert record["numerical_failures"][0]["record"]["step"] == 2
    assert record["source_commit"] == policy["source_commit"] and record["hardware"]["node"] == "akita"
    assert not list(Path(policy["qualification_root"]).glob("b300/attempts/*/*/checkpoints"))
    assert len([args for args, _ in calls if "pilot" in args]) == 3


@pytest.mark.parametrize("kernel_count,kernel_skip", [(104, False), (106, False), (105, True)])
def test_exact_immutable_kernel_suite_is_mandatory(setup_qualification, monkeypatch, kernel_count, kernel_skip):
    module, policy_path, policy, manifest = setup_qualification
    calls = mock_commands(monkeypatch, setup_qualification, kernel_count=kernel_count, kernel_skip=kernel_skip)
    with pytest.raises(RuntimeError):
        module.run_qualification(policy_path, policy, manifest, "b300")
    root = Path(policy["qualification_root"]) / "b300"
    assert not (root / "success.json").exists()
    assert json.loads((root / "failure.json").read_text())["stage"] == "kernels"
    assert not any("pilot" in args for args, _ in calls)


@pytest.mark.parametrize("wrong_resume", [False, True])
def test_infrastructure_or_exact_resume_failure_blocks_profile(setup_qualification, monkeypatch, wrong_resume):
    module, policy_path, policy, manifest = setup_qualification
    mock_commands(monkeypatch, setup_qualification, wrong_resume=wrong_resume,
                  infrastructure_run=None if wrong_resume else manifest["runs"][0]["name"])
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        module.run_qualification(policy_path, policy, manifest, "b300")
    root = Path(policy["qualification_root"]) / "b300"
    assert not (root / "success.json").exists()
    assert json.loads((root / "failure.json").read_text())["failure_kind"] == "infrastructure"
    # Keep infrastructure/audit failure checkpoints for diagnosis.
    assert list(root.glob("attempts/*/*/checkpoints"))


def test_runtime_ineligibility_is_recorded_for_each_coordinate(setup_qualification, monkeypatch):
    module, policy_path, policy, manifest = setup_qualification
    mock_commands(monkeypatch, setup_qualification, step_seconds=100.)
    result = module.run_qualification(policy_path, policy, manifest, "b300")
    assert result["eligible_runs"] == []
    assert result["ineligible_runs"] == sorted(policy["assignments"])
    for run in manifest["runs"]:
        record = json.loads((Path(policy["qualification_root"]) / "b300/checks" / f"{run['name']}.json").read_text())
        assert record["reason"] == "runtime"


def test_completed_policy_evidence_is_reused_but_hardware_rechecked(setup_qualification, monkeypatch):
    module, policy_path, policy, manifest = setup_qualification
    calls = mock_commands(monkeypatch, setup_qualification)
    first = module.run_qualification(policy_path, policy, manifest, "b300")
    count = len(calls)
    second = module.run_qualification(policy_path, policy, manifest, "b300")
    assert first["eligible_runs"] == second["eligible_runs"]
    assert len(calls) == count + 1
    assert "hardware" in calls[-1][0]
    path = Path(policy["qualification_root"]) / "b300/kernels.json"
    cached = json.loads(path.read_text())
    cached["source_commit"] = "changed"
    write_json(path, cached)
    with pytest.raises(RuntimeError, match="different policy"):
        module.run_qualification(policy_path, policy, manifest, "b300")


def test_profile_assignment_must_name_active_coordinates(setup_qualification):
    module, _, policy, manifest = setup_qualification
    invalid = copy.deepcopy(policy)
    invalid["assignments"]["missing"] = "b300"
    with pytest.raises(ValueError, match="known active"):
        module.assigned_runs(invalid, manifest, "b300")
    with pytest.raises(ValueError, match="Unknown"):
        module.assigned_runs(policy, manifest, "h100")
