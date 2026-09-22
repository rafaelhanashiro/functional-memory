"""Bounded hardware probes preserve science and cannot schedule training."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.key_dimension_taylor.hardware_checks import common
from experiments.key_dimension_taylor.sweep import parameter_count, scientific_recipe
from experiments.polynomial_attention.configs import config_to_dict, load_config, save_config


@pytest.fixture
def manager(monkeypatch):
    # Match the direct-script launch while leaving other orchestration modules alone.
    monkeypatch.setitem(sys.modules, "common", common)
    path = Path(common.__file__).with_name("manage.py")
    spec = importlib.util.spec_from_file_location("_hardware_check_manage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy_fixture(tmp_path, monkeypatch, manager):
    project = tmp_path / "project"
    root = Path(__file__).resolve().parents[3]
    config = load_config(root / "experiments/polynomial_attention/configs/runs/200m-m002-seed0.yaml")
    config.model.qk_head_dim = 8
    config.train.source_commit = "a" * 40
    config_path = project / "base.yaml"
    save_config(config, config_path)
    manifest = {"identity": "immutable-grid", "source_commit": config.train.source_commit,
                "execution_root": str(root), "project_root": str(project),
                "dependency_lock_sha256": "b" * 64,
                "data_identity": {name: {"sha256": name} for name in ("train_manifest", "eval_manifest")},
                "key_dimensions": [8, 16, 32, 64, 128], "degrees": [2, 4, 8, 16],
                "runs": [{"config_path": str(config_path), "config": config_to_dict(config),
                          "retained_completed": False}]}
    manifest_path = project / "results/key_dimension_taylor/sweep.json"
    common.write_json(manifest_path, manifest)
    evidence = project / "existing-checks/b300"
    junit = evidence / "kernels.xml"
    junit.parent.mkdir(parents=True)
    junit.write_text('<testsuite tests="105" failures="0"/>')
    common.write_json(evidence / "kernels.json", {"status": "passed", "profile": "b300",
                      "source_commit": manifest["source_commit"], "tests_passed": 105,
                      "junit_path": str(junit)})
    for width, degree in ((32, 2), (32, 4), (32, 8), (128, 2), (128, 4)):
        model = copy.deepcopy(config.model)
        model.qk_head_dim = width
        name = f"200m-dk{width:03d}-m{degree:03d}-seed0"
        common.write_json(evidence / "checks" / f"{name}.json", {
            "name": name, "status": "passed", "profile": "b300", "source_commit": manifest["source_commit"],
            "qk_head_dim": width, "attention_degree": degree,
            "validation": {"parameter_count": parameter_count(model)}})
    previous = {"training_identity": manifest["identity"], "source_commit": manifest["source_commit"],
                "qualification_root": str(evidence.parent)}
    previous["identity"] = common.policy_identity(previous)
    previous_path = project / "existing-policy.json"
    common.write_json(previous_path, previous)
    monkeypatch.setattr(common.subprocess, "check_output", lambda command, **kwargs:
                        "c" * 40 if command[:2] == ["git", "rev-parse"] else "")
    # Scientific-source verification has independent coverage; these fixtures do
    # not require the production dataset or historical checkpoints to be present.
    from experiments.key_dimension_taylor import preflight
    monkeypatch.setattr(preflight, "verify_manifest", lambda value: None)
    path = project / "hardware-policy.json"
    policy = manager.prepare(path, manifest_path, previous_path)
    return path, policy, manifest, config_path


def test_referenced_evidence_is_immutable(tmp_path):
    path = tmp_path / "check.json"
    common.write_json(path, {"status": "passed"})
    ref = common.reference(path)
    assert common.load_reference(ref) == {"status": "passed"}
    common.write_json(path, {"status": "failed"})
    with pytest.raises(ValueError, match="Referenced qualification changed"):
        common.load_reference(ref)


def test_preparation_covers_twenty_pairs_without_mutating_original_config(policy_fixture):
    _, policy, manifest, original_path = policy_fixture
    original = load_config(original_path)
    baseline = scientific_recipe(config_to_dict(original), manifest["data_identity"])
    assert len(policy["runs"]) == 20
    assert {(run["qk_head_dim"], run["attention_degree"]) for run in policy["runs"]} == {
        (width, degree) for width in (8, 16, 32, 64, 128) for degree in (2, 4, 8, 16)}
    for run in policy["runs"]:
        config = load_config(run["config_path"])
        assert scientific_recipe(run["config"], manifest["data_identity"]) == baseline
        assert config.optim.total_steps == 19074 and config.optim.warmup_steps == 2000
        assert config.train.micro_batch_size == 64 and config.data.batch_size == 256
        assert config.model.max_seq_len == 2048 and config.train.eval_batches == 7
        assert config.train.source_commit == manifest["source_commit"]
        assert Path(config.train.workdir).is_relative_to(Path(policy["output_root"]))
    assert config_to_dict(load_config(original_path)) == manifest["runs"][0]["config"]
    assert policy["production_submission"] is False


def test_profiles_keep_four_gpu_production_shapes_and_full_grid_shards(policy_fixture):
    _, policy, _, _ = policy_fixture
    assert set(policy["profiles"]) == {"b300", "a100", "a6000"}
    for name, profile in policy["profiles"].items():
        assert profile["gpu_count"] == profile["kernel_gpu_count"] == 4
        assert profile["partition"] == "sweep"
        assert [run for shard in policy["shards"][name] for run in shard] == [run["name"] for run in policy["runs"]]
    assert len(policy["reused_checks"]["b300"]) == 5


@pytest.mark.parametrize("change", ["production", "shortened_schedule", "escaped_output", "missing_pair"])
def test_rehashed_policy_rejects_invalid_science_or_validation_scope(policy_fixture, change):
    _, policy, manifest, _ = policy_fixture
    policy = copy.deepcopy(policy)
    if change == "production":
        policy["production_submission"] = True
    elif change == "missing_pair":
        policy["runs"].pop()
    else:
        run = policy["runs"][0]
        config = load_config(run["config_path"])
        if change == "shortened_schedule":
            config.optim.total_steps = 33
            config.optim.warmup_steps = 2
        else:
            config.train.workdir = Path(manifest["project_root"]) / "results/key_dimension_taylor/production"
        save_config(config, Path(run["config_path"]))
        run["config"] = config_to_dict(config)
        run["config_sha256"] = common.sha256(Path(run["config_path"]))
    policy["identity"] = common.policy_identity(policy)
    with pytest.raises(ValueError):
        common.verify_policy(policy, manifest)


class SlurmFixture:
    def __init__(self, monkeypatch, manager, path):
        self.path, self.calls, self.comments, self.priority = path, [], {}, {}
        self.next_job = 700
        monkeypatch.setattr(manager.subprocess, "check_output", self.output)
        monkeypatch.setattr(manager.subprocess, "run", self.run)

    def output(self, command, **kwargs):
        self.calls.append(command[:])
        if command[0] == "squeue":
            return "".join(f"{job}|{comment}\n" for job, comment in self.comments.items())
        if command[0] == "sbatch":
            assert "--hold" in command
            self.next_job += 1
            job = str(self.next_job)
            self.priority[job] = 0
            self.comments[job] = next(arg.removeprefix("--comment=") for arg in command if arg.startswith("--comment="))
            return job + ";test-cluster\n"
        if command[:3] == ["scontrol", "show", "job"]:
            job = command[-1]
            return f"JobId={job} JobState=PENDING Priority={self.priority[job]} "
        raise AssertionError(f"Unexpected external command: {command}")

    def run(self, command, **kwargs):
        self.calls.append(command[:])
        assert command[:2] == ["scontrol", "release"]
        job = command[-1]
        durable = json.loads(self.path.with_suffix(".jobs.json").read_text())
        assert any(entry["slurm_job_id"] == job for entry in durable["jobs"].values())
        self.priority[job] = 1
        return subprocess.CompletedProcess(command, 0)


def test_submission_only_queues_typed_sweep_validation_jobs_after_durable_record(policy_fixture, manager, monkeypatch):
    path, policy, _, _ = policy_fixture
    slurm = SlurmFixture(monkeypatch, manager, path)
    result = manager.submit(path, policy)
    assert len(result["jobs"]) == 3
    commands = [command for command in slurm.calls if command[0] == "sbatch"]
    for name, command in zip(policy["profiles"], commands, strict=True):
        assert f"--gres=gpu:{name}:4" in command and "--partition=sweep" in command
        assert not any(arg.startswith(("--nodelist", "--exclude", "--dependency")) for arg in command)
        assert command[-3].endswith("hardware_checks/launch.sbatch")
        assert command[-2:] == [str(path), name]
        assert "--requeue" in command
        assert not any("production.py" in arg or "concurrent/launch.sbatch" in arg for arg in command)
    manager.submit(path, policy)
    assert len([command for command in slurm.calls if command[0] == "sbatch"]) == 3
    assert len([command for command in slurm.calls if command[:2] == ["scontrol", "release"]]) == 3


def test_orphan_held_validation_job_is_recovered_without_duplicate(policy_fixture, manager, monkeypatch):
    path, policy, _, _ = policy_fixture
    slurm = SlurmFixture(monkeypatch, manager, path)
    slurm.comments["699"] = f"dkt-hardware:{policy['identity']}:b300"
    slurm.priority["699"] = 0
    ledger = manager.submit(path, policy)
    assert ledger["jobs"]["b300"]["slurm_job_id"] == "699"
    assert slurm.priority["699"] == 1
    assert len([command for command in slurm.calls if command[0] == "sbatch"]) == 2


def test_report_has_sixty_status_cells_and_never_invents_terminal_metrics(policy_fixture, manager):
    _, policy, _, _ = policy_fixture
    root = Path(policy["output_root"])
    name = policy["runs"][0]["name"]
    common.write_json(root / "a6000/checks" / f"{name}.json", {
        "status": "memory_failure", "error": "fixture OOM"})
    manager.report(policy)
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["rows"]
    assert len(rows) == 60
    assert len({(row["profile"], row["qk_head_dim"], row["attention_degree"]) for row in rows}) == 60
    assert sum(row["status"] == "passed (existing evidence)" for row in rows) == 5
    assert sum(row["status"] == "pending" for row in rows) == 54
    failure = next(row for row in rows if row["status"] == "memory_failure")
    assert failure["error"] == "fixture OOM"
    assert all(not any("perplexity" in key or "ppl" in key or "nll" in key for key in row) for row in rows)
    report = (root / "README.md").read_text()
    assert "No production jobs are submitted" in report
    assert "19,074" in report and "memory_failure" in report
