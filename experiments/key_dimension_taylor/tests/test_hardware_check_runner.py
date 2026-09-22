"""Validation-only routing, isolation, independent eligibility, and resume gates."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor import preflight as original

SCRIPT = Path(__file__).resolve().parents[1] / "hardware_checks/runner.py"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        yield handle


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    common = SimpleNamespace(verify_policy=lambda *args: None,
                             training_env=lambda policy: {"PYTHONPATH": policy["training_root"], "JAX_PLATFORMS": "cuda"},
                             locked=locked, write_json=write_json)
    monkeypatch.setitem(sys.modules, "common", common)
    spec = importlib.util.spec_from_file_location("hardware_runner_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = {"identity": "policy", "source_commit": "a" * 40, "orchestrator_commit": "b" * 40,
              "training_root": str(tmp_path / "frozen"), "orchestrator_root": str(tmp_path / "orchestrator"),
              "output_root": str(tmp_path / "isolated"), "key_dimensions": [8], "degrees": [2, 4, 8, 16],
              "profiles": {"b300": {"gpu_type": "b300", "gpu_count": 4, "kernel_gpu_count": 4,
                                    "production_time": "08:00:00", "partition": "sweep"}},
              "reused_checks": {}, "reused_kernels": {}}
    runs = {}
    for width in policy["key_dimensions"]:
        for degree in policy["degrees"]:
            name = f"200m-dk{width:03d}-m{degree:03d}-seed0"
            runs[name] = {"name": name, "qk_head_dim": width, "attention_degree": degree,
                          "parameter_count": 181556224, "config_path": str(tmp_path / f"{name}.yaml"),
                          "config": {"data": {"batch_size": 256}, "optim": {"total_steps": 19074},
                                     "train": {"eval_every": 1000, "eval_batches": 7, "ckpt_every": 500}}}
    policy["shards"] = {"b300": [list(runs)]}
    manifest = {"runs": list(runs.values())}
    common.grid_run = lambda policy, manifest, width, degree: runs[f"200m-dk{width:03d}-m{degree:03d}-seed0"]
    common.load_reference = lambda ref: json.loads(Path(ref["path"]).read_text())
    helpers = SimpleNamespace(validate_resume_audit=original.validate_resume_audit,
                             projected_runtime=original.projected_runtime,
                             RuntimeLimitExceeded=original.RuntimeLimitExceeded,
                             require_passing_tests=original.require_passing_tests,
                             committed_records=lambda path: [json.loads((path / "metrics.json").read_text())])
    monkeypatch.setattr(module, "_helpers", lambda policy: helpers)
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    hardware = {"device_kinds": ["NVIDIA B300 SXM6 AC"] * 4, "gpu_count": 4, "node": "akita"}
    monkeypatch.setattr(module, "_hardware", lambda *args: hardware)
    write_json(Path(policy["output_root"]) / "b300/kernels.json",
               {**module._identity(policy, "b300"), "status": "passed", "tests_passed": 105, "hardware": hardware})
    return module, policy_path, policy, manifest, hardware


def audit(start, final):
    def update(step):
        return {"step": step, "batch_sha256": f"{step:064x}", "rng_sha256": f"{step + 100:064x}"}
    return {"start_state_sha256": f"{start:064x}", "final_state_sha256": f"{final:064x}",
            "checkpoint_state_sha256": f"{final:064x}", "first_update": update(start + 1), "next_update": update(final + 1)}


def outcome(run, stop):
    start = 0 if stop == 32 else 32
    return {"start_step": start, "final_step": stop, "restored_step": None if start == 0 else start,
            "data_start_sequence": start * 256, "data_parallel_devices": 4,
            "attention_degree": run["attention_degree"], "checkpoint_audit": audit(start, stop),
            "timing": {"runtime_seconds": 40, "train_step_seconds": [10] + [0.5] * 7,
                       "evaluation_seconds": [1], "checkpoint_seconds": [2]}}


def validation(run, policy):
    return {"step": 32, "source_commit": policy["source_commit"], "qk_head_dim": run["qk_head_dim"],
            "attention_degree": run["attention_degree"], "parameter_count": run["parameter_count"],
            "validation_nll": 4.0, "validation_perplexity": 54.6}


def pilot_stub(monkeypatch, setup, failures=None, wrong_restore=None):
    module, _, policy, _, _ = setup
    calls = []
    failures = failures or {}
    def launch(policy_path, policy, profile, run, workdir, attempt, stop, lock_fd):
        calls.append((run["name"], stop, str(workdir), lock_fd))
        (workdir / "checkpoints/32").mkdir(parents=True, exist_ok=True)
        if run["attention_degree"] in failures:
            raise failures[run["attention_degree"]]
        write_json(workdir / "metrics.json", validation(run, policy))
        result = outcome(run, stop)
        if wrong_restore == run["attention_degree"] and stop == 33:
            result["checkpoint_audit"]["start_state_sha256"] = "0" * 64
        return result
    monkeypatch.setattr(module, "_launch_pilot", launch)
    return calls


@pytest.mark.parametrize("error,status", [(MemoryError("allocation"), "memory_failure"),
    (RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"), "memory_failure"),
    (RuntimeError("RESOURCE_EXHAUSTED: insufficient GPU memory"), "memory_failure"),
    (RuntimeError("OutOfResources: out of resource: shared memory. Required: 180224 Hardware limit: 101376"), "memory_failure"),
    (FloatingPointError("Nonfinite denominator"), "numerical_failure"),
    (RuntimeError("Node disappeared"), "infrastructure_failure")])
def test_failure_classification_preserves_memory_vs_numerical(setup_runner, error, status):
    assert setup_runner[0].failure_status(error) == status


def test_independent_coordinate_checks_cleanup_and_preserve_schedule(setup_runner, monkeypatch):
    module, path, policy, manifest, _ = setup_runner
    calls = pilot_stub(monkeypatch, setup_runner, failures={
        2: module.CheckFailure("memory_failure", "OOM"),
        4: module.CheckFailure("numerical_failure", "Nonfinite"),
        8: original.RuntimeLimitExceeded("Does not fit a 500-update allocation"),
    })
    result = module.run_pilots(path, policy, manifest, "b300")
    assert list(result["outcomes"].values()) == ["memory_failure", "numerical_failure", "runtime_ineligible", "passed"]
    assert [stop for name, stop, _, _ in calls if "m016" in name] == [32, 33]
    assert all(fd >= 0 for _, _, _, fd in calls)
    assert not list(Path(policy["output_root"]).rglob("checkpoints"))
    assert all(run["config"]["optim"]["total_steps"] == 19074 for run in manifest["runs"])
    successful = json.loads((Path(policy["output_root"]) / "b300/checks" / f"{manifest['runs'][-1]['name']}.json").read_text())
    assert successful["initial"]["checkpoint_audit"]["final_state_sha256"] == successful["resumed"]["checkpoint_audit"]["start_state_sha256"]
    assert successful["runtime"]["segments"][-1]["end_step"] == 19074
    before = len(calls)
    module.run_pilots(path, policy, manifest, "b300")
    assert len(calls) == before


def test_resume_state_mismatch_fails_one_coordinate_only(setup_runner, monkeypatch):
    module, path, policy, manifest, _ = setup_runner
    calls = pilot_stub(monkeypatch, setup_runner, wrong_restore=2)
    result = module.run_pilots(path, policy, manifest, "b300")
    assert list(result["outcomes"].values()) == ["infrastructure_failure", "passed", "passed", "passed"]
    assert len(calls) == 8
    assert not list(Path(policy["output_root"]).rglob("checkpoints"))


def test_failed_kernels_block_full_model_pilots(setup_runner, monkeypatch):
    module, path, policy, manifest, _ = setup_runner
    kernel_path = Path(policy["output_root"]) / "b300/kernels.json"
    record = json.loads(kernel_path.read_text())
    record["status"] = "numerical_failure"
    write_json(kernel_path, record)
    calls = pilot_stub(monkeypatch, setup_runner)
    assert module.run_pilots(path, policy, manifest, "b300")["status"] == "blocked"
    assert calls == []


def test_reused_pilot_keeps_original_provenance_without_retraining(setup_runner, monkeypatch, tmp_path):
    module, path, policy, manifest, hardware = setup_runner
    run = manifest["runs"][0]
    reference_path = tmp_path / "original.json"
    record = {"source_commit": policy["source_commit"], "status": "passed", "hardware": hardware,
              "identity": "original-policy", "name": run["name"], "qk_head_dim": 8, "attention_degree": 2,
              "validation": validation(run, policy), "initial": outcome(run, 32), "resumed": outcome(run, 33),
              "runtime": {"original": True}}
    write_json(reference_path, record)
    original_bytes = reference_path.read_bytes()
    policy["reused_checks"] = {"b300": {run["name"]: {"path": str(reference_path), "sha256": "verified-by-common"}}}
    calls = pilot_stub(monkeypatch, setup_runner)
    result = module.run_pilots(path, policy, manifest, "b300")
    assert result["outcomes"][run["name"]] == "passed"
    assert all(name != run["name"] for name, *_ in calls)
    retained = json.loads((Path(policy["output_root"]) / "b300/checks" / f"{run['name']}.json").read_text())
    assert retained["reused"] is True and retained["original_identity"] == "original-policy"
    assert retained["reference"]["path"] == str(reference_path)
    assert reference_path.read_bytes() == original_bytes


def test_reference_with_wrong_parameter_count_is_rejected(setup_runner):
    module, _, policy, manifest, hardware = setup_runner
    run = manifest["runs"][0]
    record = {"source_commit": policy["source_commit"], "status": "passed", "hardware": hardware,
              "name": run["name"], "qk_head_dim": 8, "attention_degree": 2,
              "validation": {**validation(run, policy), "parameter_count": -1}}
    with pytest.raises(RuntimeError, match="parameter count"):
        module._check_reference(record, policy, "b300", run=run)


def test_child_routes_to_frozen_source_and_inherits_lock(setup_runner, monkeypatch, tmp_path):
    module, _, policy, _, _ = setup_runner
    commands = []
    record = tmp_path / "child.json"
    def subprocess_run(command, **kwargs):
        commands.append((command, kwargs))
        write_json(record, {"final_step": 32})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", subprocess_run)
    with (tmp_path / "lock").open("a+") as lock:
        result = module._run_child(["python", "runner.py"], policy, tmp_path / "child.log", lock.fileno(), record=record)
        assert commands[0][1]["pass_fds"] == (lock.fileno(),)
    assert result["final_step"] == 32
    assert commands[0][1]["cwd"] == policy["training_root"]
    assert commands[0][1]["env"]["PYTHONPATH"] == policy["training_root"]


def test_unstructured_memory_failure_uses_subprocess_log(setup_runner, monkeypatch, tmp_path):
    module, _, policy, _, _ = setup_runner
    def subprocess_run(command, **kwargs):
        kwargs["stdout"].write("RESOURCE_EXHAUSTED: GPU memory allocation failed")
        return SimpleNamespace(returncode=-9)
    monkeypatch.setattr(module.subprocess, "run", subprocess_run)
    with (tmp_path / "lock").open("a+") as lock:
        with pytest.raises(module.CheckFailure) as caught:
            module._run_child(["python"], policy, tmp_path / "child.log", lock.fileno())
    assert caught.value.status == "memory_failure"


@pytest.mark.parametrize("stop,inside", [(19074, True), (33, False)])
def test_worker_rejects_full_training_and_production_directories(setup_runner, tmp_path, stop, inside):
    module, _, policy, manifest, _ = setup_runner
    workdir = Path(policy["output_root"]) / "b300/attempts/test/run" if inside else tmp_path / "production"
    with pytest.raises(ValueError):
        module._pilot_worker(policy, manifest, "b300", 8, 2, workdir, stop)


def test_four_gpu_combined_allocation_kernel_probe(setup_runner, monkeypatch, tmp_path):
    module, path, policy, manifest, _ = setup_runner
    counts = []
    module.common.load_policy = lambda path: (policy, manifest)
    module.common.check_hardware = lambda profile, count: counts.append(count) or {"gpu_count": count}
    monkeypatch.setattr(sys, "argv", ["runner.py", "--policy", str(path), "--profile", "b300", "--stage", "kernels",
                                    "--worker", "hardware", "--output", str(tmp_path / "hardware.json")])
    module.main()
    assert counts == [4]


def test_kernel_suite_rejects_missing_or_skipped_cases(setup_runner, monkeypatch):
    module, path, policy, manifest, _ = setup_runner
    (Path(policy["output_root"]) / "b300/kernels.json").unlink()
    def child(command, policy, output, lock_fd, record=None):
        junit = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        junit.write_text('<testsuites><testsuite tests="105" failures="0" errors="0" skipped="1" /></testsuites>')
        return {}
    monkeypatch.setattr(module, "_run_child", child)
    result = module.run_kernels(path, policy, manifest, "b300")
    assert result["status"] == "infrastructure_failure"
    assert "passed" not in result or result.get("tests_passed") != 105


def test_shards_cover_reused_width64_in_full_grid(setup_runner):
    module, _, policy, manifest, _ = setup_runner
    policy["key_dimensions"] = [8, 16, 32, 64, 128]
    def grid(policy, manifest, width, degree):
        return {"name": f"200m-dk{width:03d}-m{degree:03d}-seed0", "qk_head_dim": width, "attention_degree": degree}
    module.common.grid_run = grid
    names = [grid(policy, manifest, width, degree)["name"] for width in policy["key_dimensions"] for degree in policy["degrees"]]
    policy["shards"]["b300"] = [names[:10], names[10:]]
    runs = [*module._shard_runs(policy, manifest, "b300", 0), *module._shard_runs(policy, manifest, "b300", 1)]
    assert len(runs) == 20 and len({run["name"] for run in runs}) == 20
    assert sum(run["qk_head_dim"] == 64 for run in runs) == 4


def test_one_ampere_kernel_resource_failure_allows_other_coordinates(setup_runner, monkeypatch, tmp_path):
    module, path, policy, manifest, _ = setup_runner
    policy["key_dimensions"] = [8, 16, 32, 64, 128]
    baseline = manifest["runs"][0]
    module.common.grid_run = lambda policy, manifest, width, degree: {
        **baseline, "name": f"200m-dk{width:03d}-m{degree:03d}-seed0", "qk_head_dim": width, "attention_degree": degree}
    suite = ET.Element("testsuite")
    for width in policy["key_dimensions"]:
        for degree in policy["degrees"]:
            suffix = f"dk{width:03d}-m{degree:03d}"
            names = [f"test_gpu_all_pairs_fp64_partial_tiles_padding_and_causality[{dtype}-{suffix}]"
                     for dtype in ("fp32", "bf16")]
            names += [f"test_gpu_all_pairs_{kind}[{suffix}]" for kind in
                      ("negative_score_arithmetic_matches_fp64", "large_finite_kernel_and_values", "full_production_shape_bf16")]
            if degree == 16:
                names.append(f"test_gpu_m16_broad_bf16_scores_and_small_cotangent[dk{width:03d}]")
            for name in names:
                case = ET.SubElement(suite, "testcase", name=name)
                if name == "test_gpu_all_pairs_full_production_shape_bf16[dk008-m002]":
                    ET.SubElement(case, "failure").text = "OutOfResources: out of resource: shared memory"
    junit = tmp_path / "partial.xml"
    ET.ElementTree(suite).write(junit)
    kernel = module._kernel_coordinates(junit, policy, manifest)
    assert kernel["status"] == "partial" and kernel["tests_completed"] == 105 and kernel["tests_passed"] == 104
    assert len(kernel["eligible_runs"]) == 19
    assert list(kernel["blocked_runs"]) == [baseline["name"]]
    assert kernel["blocked_runs"][baseline["name"]]["status"] == "memory_failure"
    write_json(Path(policy["output_root"]) / "b300/kernels.json", {**module._identity(policy, "b300"), **kernel})
    calls = pilot_stub(monkeypatch, setup_runner)
    result = module.run_pilots(path, policy, manifest, "b300")
    assert list(result["outcomes"].values()) == ["kernel_failure", "passed", "passed", "passed"]
    assert all(name != baseline["name"] for name, *_ in calls)
    assert len(calls) == 6
