from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.key_dimension_taylor.concurrent import common, fast_queue, production, schedule
from experiments.key_dimension_taylor.tests.test_concurrent_schedule import amendment, SlurmFixture


WEAKER = ["collie", "doodle", "samoyed", "terrier"]


def setup_fast(amendment):
    path, policy, manifest, ledger = amendment
    template = policy["profiles"]["b300"]
    policy["profiles"]["h100_or_better"] = {**template, "gpu_type": "h100_or_better",
        "allowed_gpu_types": ["h100", "b300"], "allowed_nodes": ["akita", "dogo", "mastiff"],
        "excluded_nodes": WEAKER}
    policy["profiles"]["h100_or_better"].pop("node")
    policy["profiles"]["b300"]["allowed_nodes"] = ["akita"]
    policy["assignments"] = {run["name"]: "h100_or_better" for run in manifest["runs"]}
    for run in manifest["runs"]:
        run["config"] = {"optim": {"total_steps": 19074}}
    return path, policy, manifest, ledger


def test_generic_request_excludes_slower_nodes_without_type_or_node_pin(amendment, monkeypatch):
    path, policy, manifest, ledger = setup_fast(amendment)
    slurm = SlurmFixture(monkeypatch, amendment)
    schedule.submit_held(path, policy, manifest, ledger, key=manifest["runs"][0]["name"] + ":0",
                         profile_name="h100_or_better", run_name=manifest["runs"][0]["name"], excluded_nodes=WEAKER)
    command = next(command for command in slurm.calls if command[0] == "sbatch")
    assert "--gres=gpu:4" in command
    assert f"--exclude={','.join(WEAKER)}" in command
    assert not any(arg.startswith(("--nodelist=", "--dependency=", "--begin=")) for arg in command)


def test_orphan_held_job_reconciles_exclusions_after_proof_arrives(amendment, monkeypatch):
    path, policy, manifest, ledger = setup_fast(amendment)
    slurm = SlurmFixture(monkeypatch, amendment)
    name = manifest["runs"][0]["name"]
    key = name + ":0"
    slurm.states["201"], slurm.priorities["201"] = "PENDING", 0
    slurm.comments["201"] = f"dkt:{policy['identity']}:{key}"
    original_run = slurm.run
    def external(command, **kwargs):
        if command[:2] == ["scontrol", "update"]:
            slurm.calls.append(command)
            return SimpleNamespace(returncode=0)
        return original_run(command, **kwargs)
    monkeypatch.setattr(schedule.subprocess, "run", external)
    job = schedule.submit_held(path, policy, manifest, ledger, key=key,
                              profile_name="h100_or_better", run_name=name, excluded_nodes=WEAKER)
    assert job == "201" and ledger["jobs"][key]["excluded_nodes"] == WEAKER
    assert not any(command[0] == "sbatch" for command in slurm.calls)
    update = ["scontrol", "update", "JobId=201", f"ExcNodeList={','.join(WEAKER)}"]
    assert slurm.calls.index(update) < slurm.calls.index(["scontrol", "release", "201"])


@pytest.mark.parametrize("qualification", ["pending", "failed", "passed", "too_long"])
def test_b300_admission_requires_its_coordinate_proof(amendment, monkeypatch, qualification):
    _, policy, manifest, _ = setup_fast(amendment)
    def load(*args):
        if qualification == "pending":
            raise FileNotFoundError("Not run yet")
        if qualification == "failed":
            raise ValueError("Pilot failed")
        return {"qualification_reference": {"path": "proof", "sha256": "a" * 64}}
    monkeypatch.setattr(common, "load_b300_qualification", load)
    monkeypatch.setattr(production, "_segment", lambda *args, **kwargs:
                        ({"end_step": 10000 if qualification == "too_long" else 19074}, {}))
    excluded, evidence = fast_queue.exclusions(policy, manifest, manifest["runs"][0])
    assert set(excluded) == set(WEAKER + ([] if qualification == "passed" else ["akita"]))
    assert evidence["b300"] == ("passed" if qualification == "passed" else "pending" if qualification == "pending" else "not_qualified")


def test_refresh_widens_only_pending_job_without_waiting_for_other_coordinates(amendment, monkeypatch):
    path, policy, manifest, ledger = setup_fast(amendment)
    for i, run in enumerate(manifest["runs"]):
        ledger["jobs"][f"{run['name']}:0"] = {"profile": "h100_or_better", "run_name": run["name"],
            "slurm_job_id": str(201 + i), "excluded_nodes": sorted(WEAKER + ["akita"])}
    monkeypatch.setattr(common, "job_states", lambda jobs: {"201": "PENDING", "202": "RUNNING"})
    checked, calls = [], []
    def qualify(p, m, run):
        checked.append(run["name"])
        return WEAKER[:], {"b300": "passed", "qualification_reference": {"path": "proof", "sha256": "a" * 64}}
    monkeypatch.setattr(fast_queue, "exclusions", qualify)
    monkeypatch.setattr(fast_queue.subprocess, "check_output", lambda *args, **kwargs: "JobState=PENDING")
    monkeypatch.setattr(fast_queue.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0, stderr=""))
    assert fast_queue.refresh(path, policy, manifest, ledger) is True
    assert checked == [manifest["runs"][0]["name"]]
    assert calls == [["scontrol", "update", "JobId=201", f"ExcNodeList={','.join(WEAKER)}"]]
    second = ledger["jobs"][manifest["runs"][1]["name"] + ":0"]
    assert "akita" in second["excluded_nodes"]
    # Preemption can return the active H100-only job to the queue later.
    monkeypatch.setattr(common, "job_states", lambda jobs: {"201": "RUNNING", "202": "PENDING"})
    assert fast_queue.refresh(path, policy, manifest, ledger) is False
    assert second["excluded_nodes"] == WEAKER
    assert calls[-1] == ["scontrol", "update", "JobId=202", f"ExcNodeList={','.join(WEAKER)}"]


def test_refresh_tolerates_job_starting_during_update(amendment, monkeypatch):
    path, policy, manifest, ledger = setup_fast(amendment)
    run = manifest["runs"][0]
    ledger["jobs"] = {"one": {"profile": "h100_or_better", "run_name": run["name"],
                               "slurm_job_id": "201", "excluded_nodes": sorted(WEAKER + ["akita"])}}
    states = iter([{"201": "PENDING"}, {"201": "RUNNING"}])
    monkeypatch.setattr(common, "job_states", lambda jobs: next(states))
    monkeypatch.setattr(fast_queue, "exclusions", lambda *args: (WEAKER[:], {"b300": "passed"}))
    monkeypatch.setattr(fast_queue.subprocess, "check_output", lambda *args, **kwargs: "JobState=PENDING")
    monkeypatch.setattr(fast_queue.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr="Job no longer pending"))
    assert fast_queue.refresh(path, policy, manifest, ledger) is True


def test_unknown_scheduler_state_keeps_controller_alive(amendment, monkeypatch):
    path, policy, manifest, ledger = setup_fast(amendment)
    ledger["jobs"] = {"one": {"profile": "h100_or_better", "run_name": manifest["runs"][0]["name"], "slurm_job_id": "201"}}
    monkeypatch.setattr(common, "job_states", lambda jobs: {})
    monkeypatch.setattr(fast_queue, "exclusions", lambda *args: pytest.fail("Do not change an unknown job"))
    assert fast_queue.refresh(path, policy, manifest, ledger) is True
