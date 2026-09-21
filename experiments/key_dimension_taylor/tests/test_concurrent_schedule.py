"""Exercise amendment provenance and Slurm ownership transitions without jobs."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor.concurrent import common, schedule


@pytest.fixture
def amendment(tmp_path):
    project = tmp_path / "project"
    training = tmp_path / "frozen-training"
    orchestration = tmp_path / "frozen-orchestration"
    training.mkdir()
    orchestration.mkdir()
    runs = []
    for degree in (2, 16):
        name = f"200m-dk128-m{degree:03d}-seed0"
        workdir = project / "results/key_dimension_taylor" / name
        checkpoints = project / "experiments/key_dimension_taylor/checkpoints" / name
        workdir.mkdir(parents=True)
        checkpoints.mkdir(parents=True)
        runs.append({"name": name, "workdir": str(workdir), "checkpoint_dir": str(checkpoints),
                     "completion_path": str(workdir / "completion.json"), "retained_completed": False})
    manifest = {"identity": "immutable-science", "source_commit": "a" * 40,
                "execution_root": str(training), "project_root": str(project), "runs": runs,
                "jobs": {"preflight": "90", runs[0]["name"]: ["100"], runs[1]["name"]: ["101"]},
                "preflight_dir": str(project / "original-preflight")}
    training_manifest = project / "results/key_dimension_taylor/sweep.json"
    common.write_json(training_manifest, manifest)
    policy = {"training_identity": manifest["identity"], "source_commit": manifest["source_commit"],
              "training_root": str(training), "training_manifest": str(training_manifest),
              "orchestrator_root": str(orchestration), "orchestrator_commit": "b" * 40,
              "qualification_root": str(project / "qualifications"),
              "profiles": {"b300": {"gpu_type": "b300", "gpu_count": 4, "node": "akita", "partition": "sweep",
                                     "production_time": "08:00:00", "cpus_per_task": 32, "memory": "256G"}},
              "assignments": {run["name"]: "b300" for run in runs},
              "superseded_jobs": {run["name"]: manifest["jobs"][run["name"]][:] for run in runs},
              "legacy_renderer_jobs": []}
    policy["identity"] = common.policy_identity(policy)
    policy_path = project / "amendment.json"
    common.write_json(policy_path, policy)
    ledger = {"identity": policy["identity"], "jobs": {}, "migration": {}}
    common.write_json(schedule.ledger_path(policy_path), ledger)
    return policy_path, policy, manifest, ledger


class SlurmFixture:
    def __init__(self, monkeypatch, amendment):
        self.path, self.policy, self.manifest, self.ledger = amendment
        self.states = {"100": "PENDING", "101": "PENDING"}
        self.priorities = {}
        self.comments = {}
        self.calls = []
        self.next_job = 200
        self.cancel_fail_on = None
        self.cancel_count = 0
        self.require_durable_release = True
        monkeypatch.setattr(common, "job_states", lambda jobs: {job: self.states[job] for job in jobs if job in self.states})
        monkeypatch.setattr(schedule.subprocess, "check_output", self.check_output)
        monkeypatch.setattr(schedule.subprocess, "run", self.run)

    def check_output(self, command, **kwargs):
        self.calls.append(command[:])
        if command[0] == "sbatch":
            assert "--hold" in command and "--requeue" in command
            self.next_job += 1
            job = str(self.next_job)
            self.states[job], self.priorities[job] = "PENDING", 0
            self.comments[job] = next(value.removeprefix("--comment=") for value in command if value.startswith("--comment="))
            return job + ";fixture-cluster\n"
        if command[:3] == ["scontrol", "show", "job"]:
            job = command[-1]
            return f"JobId={job} JobState={self.states[job]} Priority={self.priorities.get(job, 1)} Comment={self.comments.get(job, '')} "
        if command[0] == "squeue":
            return "".join(f"{job}|{comment}\n" for job, comment in self.comments.items()
                           if self.states.get(job) in {"PENDING", "RUNNING"})
        raise AssertionError(f"Unexpected external command: {command}")

    def run(self, command, **kwargs):
        self.calls.append(command[:])
        if command[:2] == ["scontrol", "hold"]:
            assert self.states[command[-1]] == "PENDING"
            self.priorities[command[-1]] = 0
        elif command[0] == "scancel":
            self.cancel_count += 1
            if self.cancel_count == self.cancel_fail_on:
                raise subprocess.CalledProcessError(1, command)
            self.states[command[-1]] = "CANCELLED"
        elif command[:2] == ["scontrol", "release"]:
            job = command[-1]
            if self.require_durable_release:
                durable_ledger = json.loads(schedule.ledger_path(self.path).read_text())
                entry = next(entry for entry in durable_ledger["jobs"].values() if entry["slurm_job_id"] == job)
                if entry["run_name"] != "b300-preflight":
                    durable_manifest = json.loads(Path(self.policy["training_manifest"]).read_text())
                    assert job in durable_manifest["jobs"][entry["run_name"]]
                    run = next(run for run in self.manifest["runs"] if run["name"] == entry["run_name"])
                    launch = json.loads((Path(run["workdir"]) / "launch.json").read_text())
                    assert len([row for row in launch["concurrent_segments"] if row["slurm_job_id"] == job]) == 1
                    assert self.policy["identity"] in launch["execution_amendments"]
            self.priorities[job] = 1
        else:
            raise AssertionError(f"Unexpected external command: {command}")
        return subprocess.CompletedProcess(command, 0)


@pytest.mark.parametrize("mutate", ["profile", "assignment", "source", "checkpoint_recipe"])
def test_policy_hash_rejects_any_unrecorded_execution_change(amendment, monkeypatch, mutate):
    _, policy, manifest, _ = amendment
    calls = []
    monkeypatch.setattr(common.subprocess, "check_output", lambda *args, **kwargs: calls.append(args) or "")
    changed = copy.deepcopy(policy)
    if mutate == "profile":
        changed["profiles"]["b300"]["partition"] = "standard"
    elif mutate == "assignment":
        changed["assignments"][manifest["runs"][0]["name"]] = "h100"
    elif mutate == "source":
        changed["source_commit"] = "c" * 40
    else:
        changed["profiles"]["b300"]["gpu_count"] = 2
    with pytest.raises(ValueError, match="identity changed"):
        common.verify_policy(changed, manifest)
    assert not calls


def test_rehashed_policy_still_requires_frozen_original_identity_and_imports(amendment, monkeypatch):
    _, policy, manifest, _ = amendment
    original_calls = []
    original = SimpleNamespace(__file__=str(Path(policy["training_root"]) / "experiments/key_dimension_taylor/preflight.py"),
                               verify_manifest=lambda value: original_calls.append(value))
    monkeypatch.setattr(common.importlib, "import_module", lambda name: original)
    monkeypatch.setattr(common.subprocess, "check_output", lambda command, **kwargs: policy["orchestrator_commit"] if "rev-parse" in command else "")
    common.verify_policy(policy, manifest)
    assert original_calls == [manifest]
    wrong = copy.deepcopy(policy)
    wrong["training_identity"] = "wrong"
    wrong["identity"] = common.policy_identity(wrong)
    with pytest.raises(ValueError, match="original training source"):
        common.verify_policy(wrong, manifest)
    original.__file__ = str(Path(policy["orchestrator_root"]) / "experiments/key_dimension_taylor/preflight.py")
    with pytest.raises(ValueError, match="imports escaped"):
        common.verify_policy(policy, manifest)


def test_job_states_prefers_live_requeue_and_handles_purged_terminal_job(monkeypatch):
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        if command[0] == "sacct":
            return "100|CANCELLED by 12345|\n101|COMPLETED|\n102|FAILED+|\n"
        return "100|RUNNING\n999|PENDING\n"
    monkeypatch.setattr(common.subprocess, "check_output", output)
    assert common.job_states(["100", "101", "102", "103"]) == {"100": "RUNNING", "101": "COMPLETED", "102": "FAILED"}
    assert calls[1][0] == "squeue" and "-u" in calls[1] and "-j" not in calls[1]
    assert common.job_states([]) == {}
    assert len(calls) == 2


@pytest.mark.parametrize("state", ["RUNNING", "CONFIGURING", "COMPLETING", "UNKNOWN"])
def test_migration_refuses_nonpending_owner_without_cancelling(amendment, monkeypatch, state):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    slurm.states["100"] = state
    before = Path(policy["training_manifest"]).read_bytes()
    with pytest.raises(RuntimeError, match="non-pending"):
        schedule.migrate(path, policy, manifest, ledger)
    assert not any(command[0] == "scancel" for command in slurm.calls)
    assert Path(policy["training_manifest"]).read_bytes() == before


def test_migration_holds_all_before_cancel_and_recovers_partial_cancellation(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    slurm.cancel_fail_on = 2
    original = Path(policy["training_manifest"]).read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        schedule.migrate(path, policy, manifest, ledger)
    assert slurm.calls[:2] == [["scontrol", "hold", "100"], ["scontrol", "hold", "101"]]
    assert slurm.states == {"100": "PENDING", "101": "CANCELLED"}
    assert json.loads(schedule.ledger_path(path).read_text())["migration"]["status"] == "held"
    assert Path(policy["training_manifest"]).read_bytes() == original
    slurm.cancel_fail_on = None
    recovered = json.loads(schedule.ledger_path(path).read_text())
    schedule.migrate(path, policy, manifest, recovered)
    assert all(state == "CANCELLED" for state in slurm.states.values())
    assert recovered["migration"]["status"] == "complete"
    assert all(not manifest["jobs"][name] for name in policy["assignments"])
    count = len(slurm.calls)
    schedule.migrate(path, policy, manifest, recovered)
    assert len(slurm.calls) == count


def test_migration_refuses_checkpoint_or_metadata_started_since_snapshot(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    checkpoint = Path(manifest["runs"][0]["checkpoint_dir"]) / "500"
    checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="existing checkpoints"):
        schedule.migrate(path, policy, manifest, ledger)
    assert not any(command[0] == "scancel" for command in slurm.calls)


def test_held_job_durable_ownership_precedes_release_and_repeat_is_idempotent(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    name = manifest["runs"][0]["name"]
    first = schedule.submit_held(path, policy, manifest, ledger, key=f"{name}:0", profile_name="b300", run_name=name)
    assert first == "201"
    # SlurmFixture checks ledger + launch + manifest during each release call.
    assert ledger["jobs"][f"{name}:0"]["released_utc"]
    schedule.submit_held(path, policy, manifest, ledger, key=f"{name}:0", profile_name="b300", run_name=name)
    assert len([command for command in slurm.calls if command[0] == "sbatch"]) == 1
    assert len([command for command in slurm.calls if command[:2] == ["scontrol", "release"]]) == 1
    launch = json.loads((Path(manifest["runs"][0]["workdir"]) / "launch.json").read_text())
    assert len(launch["concurrent_segments"]) == 1


def test_crash_after_ledger_write_repairs_launch_manifest_before_release(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    name = manifest["runs"][0]["name"]
    real_write = common.write_json
    failed = False
    def write(path, value):
        nonlocal failed
        if Path(path).name == "launch.json" and not failed:
            failed = True
            raise OSError("simulated process interruption before launch ownership")
        real_write(path, value)
    monkeypatch.setattr(common, "write_json", write)
    with pytest.raises(OSError):
        schedule.submit_held(path, policy, manifest, ledger, key=f"{name}:0", profile_name="b300", run_name=name)
    assert slurm.priorities["201"] == 0
    assert not any(command[:2] == ["scontrol", "release"] for command in slurm.calls)
    recovered_ledger = json.loads(schedule.ledger_path(path).read_text())
    recovered_manifest = json.loads(Path(policy["training_manifest"]).read_text())
    job = schedule.submit_held(path, policy, recovered_manifest, recovered_ledger,
                               key=f"{name}:0", profile_name="b300", run_name=name)
    assert job == "201" and slurm.priorities[job] == 1
    assert len([command for command in slurm.calls if command[0] == "sbatch"]) == 1
    assert job in json.loads(Path(policy["training_manifest"]).read_text())["jobs"][name]


def test_orphan_held_job_is_found_by_policy_comment_without_duplicate_submission(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    name = manifest["runs"][0]["name"]
    slurm.states["299"], slurm.priorities["299"] = "PENDING", 0
    slurm.comments["299"] = f"dkt:{policy['identity']}:{name}:0"
    assert schedule.submit_held(path, policy, manifest, ledger, key=f"{name}:0", profile_name="b300", run_name=name) == "299"
    assert not any(command[0] == "sbatch" for command in slurm.calls)
    assert slurm.priorities["299"] == 1


def test_production_dependencies_exist_only_inside_same_coordinate(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    slurm.states = {job: "CANCELLED" for job in slurm.states}
    ledger["migration"] = {"status": "complete"}
    qualification = Path(policy["qualification_root"]) / "b300"
    common.write_json(qualification / "kernels.json", {"identity": policy["identity"], "status": "passed", "tests_passed": 105})
    for run in manifest["runs"]:
        common.write_json(qualification / "checks" / f"{run['name']}.json", {
            "identity": policy["identity"], "status": "passed", "runtime": {"segments": [{}, {}]}})
    assert schedule.submit_profile(path, policy, manifest, ledger, "b300") == ["201", "202", "203", "204"]
    submissions = [command for command in slurm.calls if command[0] == "sbatch"]
    dependencies = [[value for value in command if value.startswith("--dependency=")] for command in submissions]
    assert dependencies == [[], ["--dependency=afterok:201"], [], ["--dependency=afterok:203"]]
    assert "--kill-on-invalid-dep=yes" in submissions[1] and "--kill-on-invalid-dep=yes" in submissions[3]
    assert all("--partition=sweep" in command and "--gres=gpu:b300:4" in command for command in submissions)


def test_ineligible_qualification_is_visible_without_submitting_coordinate(amendment, monkeypatch):
    path, policy, manifest, ledger = amendment
    slurm = SlurmFixture(monkeypatch, amendment)
    slurm.states = {job: "CANCELLED" for job in slurm.states}
    ledger["migration"] = {"status": "complete"}
    root = Path(policy["qualification_root"]) / "b300/checks"
    for run in manifest["runs"]:
        common.write_json(root / f"{run['name']}.json", {"identity": policy["identity"], "status": "ineligible",
                          "source_commit": policy["source_commit"], "reason": "numerical", "error": "nonfinite pilot"})
    assert schedule.submit_profile(path, policy, manifest, ledger, "b300") == []
    assert not any(command[0] == "sbatch" for command in slurm.calls)
    for run in manifest["runs"]:
        record = json.loads((Path(run["workdir"]) / "preflight_failure.json").read_text())
        assert record["policy_identity"] == policy["identity"] and record["stage"] == "hardware_qualification"
