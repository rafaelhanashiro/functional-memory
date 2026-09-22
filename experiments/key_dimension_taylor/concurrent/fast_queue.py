"""Move checkpointed runs to independent jobs eligible for H100 or B300."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

try:
    from . import common, production, schedule, fast_handoff
except ImportError:
    import common
    import production
    import schedule
    import fast_handoff


def reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(path: Path, manifest_path: Path, previous_path: Path) -> dict:
    if path.exists():
        raise ValueError("Use a new immutable execution policy path")
    previous, manifest = common.load_policy(previous_path)
    common.verify_policy(previous, manifest)
    if Path(previous["training_manifest"]).resolve() != manifest_path.resolve():
        raise ValueError("Handoff must retain its scientific manifest")
    if set(previous["assignments"].values()) != {"a6000"}:
        raise ValueError("This handoff applies only to the existing A6000 amendment")
    old_ledger = json.loads(schedule.ledger_path(previous_path).read_text())
    if old_ledger["identity"] != previous["identity"]:
        raise ValueError("A6000 job ledger identity changed")
    by_name = {run["name"]: run for run in manifest["runs"]}
    names = sorted(previous["assignments"], key=lambda name: -json.loads(
        (Path(manifest["preflight_dir"]) / "checks" / f"{name}.json").read_text()
    )["runtime"]["projected_total_seconds"])
    owners = {name: [entry["slurm_job_id"] for entry in old_ledger["jobs"].values()
                     if entry["run_name"] == name] for name in names}
    if any(not jobs or Path(by_name[name]["completion_path"]).exists() for name, jobs in owners.items()):
        raise ValueError("Expected an unfinished A6000 chain for every migrated coordinate")
    all_jobs = [str(job) for key, jobs in manifest["jobs"].items() if key != "preflight" for job in jobs]
    states = common.job_states(all_jobs)
    base = dict(manifest["scheduler"])
    base.pop("node", None)
    base.update(partition="sweep")
    root = Path(__file__).resolve().parents[3]
    policy = {
        "format": "key-dimension-taylor-execution-amendment", "version": 3,
        "created_utc": schedule.utc(), "training_manifest": str(manifest_path.resolve()),
        "training_identity": manifest["identity"], "source_commit": manifest["source_commit"],
        "training_root": manifest["execution_root"], "orchestrator_root": str(root),
        "orchestrator_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "profiles": {
            "h100_or_better": dict(base, gpu_type="h100_or_better", allowed_gpu_types=["h100", "b300"],
                                   allowed_nodes=["akita", "dogo", "mastiff"],
                                   excluded_nodes=["collie", "doodle", "samoyed", "terrier"]),
            "h100": dict(base, gpu_type="h100", allowed_nodes=["dogo", "mastiff"]),
            "b300": dict(base, gpu_type="b300", allowed_nodes=["akita"]),
        },
        "assignments": {name: "h100_or_better" for name in names},
        "submission_order": names, "superseded_jobs": owners,
        "supersedes_policy": {**reference(previous_path), "identity": previous["identity"]},
        "hardware_policy_reference": reference(Path(manifest["project_root"]) / "results/key_dimension_taylor/hardware_checks/policy.json"),
        "legacy_renderer_jobs": [job for job in all_jobs if states.get(job) in {"RUNNING", "COMPLETING"}
                                 and job not in {old for jobs in owners.values() for old in jobs}],
        "minimum_checkpoint_step": 500, "cross_coordinate_dependencies": False, "scientific_changes": [],
        "authorization": "User requested cancelling A6000 allocations and queuing each run on any H100-or-better hardware on 2026-09-22",
        "eligibility_rule": "H100 immediately; admit B300 for each pending job only after its native full-model qualification passes",
        "inventory_note": "Current eligible GPU families are H100 and B300; no H200 is installed. Mastiff H100 remains eligible if restored to service.",
    }
    policy["identity"] = common.policy_identity(policy)
    common.verify_policy(policy, manifest)
    for name in names:
        segment, _ = production._segment(policy, manifest, by_name[name], 0, actual_profile="h100")
        if segment["end_step"] != by_name[name]["config"]["optim"]["total_steps"]:
            raise ValueError("Flexible jobs currently require one qualified full-budget fast-GPU allocation")
    common.write_json(path, policy)
    common.write_json(schedule.ledger_path(path), {"identity": policy["identity"], "jobs": {}, "migration": {},
                                                 "eligibility": {}, "created_utc": schedule.utc()})
    return policy


def exclusions(policy: dict, manifest: dict, run: dict) -> tuple[list[str], dict]:
    excluded = list(policy["profiles"]["h100_or_better"]["excluded_nodes"])
    try:
        record = common.load_b300_qualification(policy, manifest, run)
        segment, _ = production._segment(policy, manifest, run, 0, actual_profile="b300")
        if segment["end_step"] != run["config"]["optim"]["total_steps"]:
            raise ValueError("B300 qualification requires multiple allocations")
        evidence = {"b300": "passed", "qualification_reference": record["qualification_reference"]}
    except FileNotFoundError:
        excluded.extend(policy["profiles"]["b300"]["allowed_nodes"])
        evidence = {"b300": "pending"}
    except (ValueError, RuntimeError) as error:
        excluded.extend(policy["profiles"]["b300"]["allowed_nodes"])
        evidence = {"b300": "not_qualified", "error": f"{type(error).__name__}: {error}"}
    return sorted(set(excluded)), evidence


def submit(path: Path, policy: dict, manifest: dict, ledger: dict) -> list[str]:
    if ledger.get("migration", {}).get("status") != "complete":
        raise RuntimeError("Finish checkpoint handoff before releasing fast-GPU jobs")
    by_name = {run["name"]: run for run in manifest["runs"]}
    jobs = []
    for name in policy["submission_order"]:
        common.assert_legacy_jobs_terminal(policy, manifest, name)
        segment, _ = production._segment(policy, manifest, by_name[name], 0, actual_profile="h100")
        if segment["end_step"] != by_name[name]["config"]["optim"]["total_steps"]:
            raise ValueError("H100 qualification requires multiple allocations")
        excluded, evidence = exclusions(policy, manifest, by_name[name])
        ledger.setdefault("eligibility", {})[name] = {**evidence, "updated_utc": schedule.utc()}
        job = schedule.submit_held(path, policy, manifest, ledger, key=f"{name}:0", profile_name="h100_or_better",
                                   run_name=name, segment=0, excluded_nodes=excluded)
        jobs.append(job)
    common.write_json(schedule.ledger_path(path), ledger)
    return jobs


def refresh(path: Path, policy: dict, manifest: dict, ledger: dict) -> bool:
    """Broaden pending jobs as proofs arrive; running jobs are never touched."""
    by_name = {run["name"]: run for run in manifest["runs"]}
    entries = [entry for entry in ledger["jobs"].values() if entry["profile"] == "h100_or_better"]
    states = common.job_states([entry["slurm_job_id"] for entry in entries])
    waiting = False
    for entry in entries:
        job, name = entry["slurm_job_id"], entry["run_name"]
        state = states.get(job)
        if state is None:
            waiting = True
            continue
        if state != "PENDING":
            # Sweep jobs can be preempted and requeued later. Retain the
            # controller while an active H100-only reservation may still need
            # its B300 exclusion removed on a future pending transition.
            waiting |= state not in common.TERMINAL and any(
                node in entry.get("excluded_nodes", []) for node in policy["profiles"]["b300"]["allowed_nodes"])
            continue
        excluded, evidence = exclusions(policy, manifest, by_name[name])
        old_evidence = ledger.setdefault("eligibility", {}).get(name, {})
        if any(old_evidence.get(key) != value for key, value in evidence.items()):
            ledger["eligibility"][name] = {**evidence, "updated_utc": schedule.utc()}
        if excluded != entry.get("excluded_nodes"):
            if not set(excluded) <= set(entry.get("excluded_nodes", [])):
                raise RuntimeError("Qualification changed after admission; refusing silent reclassification")
            detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
            if "JobState=PENDING" not in detail:
                waiting = True
                continue
            # No dependency or future start time: either admitted GPU type can start immediately.
            command = ["scontrol", "update", f"JobId={job}", f"ExcNodeList={','.join(excluded)}"]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                if common.job_states([job]).get(job) != "PENDING":
                    waiting = True
                    continue  # It started between the read and update; leave its allocation alone.
                raise RuntimeError(f"Failed to widen GPU eligibility for {job}: {result.stderr.strip()}")
            entry.setdefault("eligibility_history", []).append({"at_utc": schedule.utc(),
                "previous_excluded_nodes": entry.get("excluded_nodes"), "excluded_nodes": excluded,
                "qualification_reference": evidence.get("qualification_reference"), "command": command})
            entry["excluded_nodes"] = excluded
        waiting |= evidence["b300"] == "pending"
    common.write_json(schedule.ledger_path(path), ledger)
    return waiting


def watch(path: Path, *, interval: float = 30, max_seconds: float = 86400) -> None:
    """Temporary CPU-only eligibility controller; exits once no pending proof remains."""
    started = time.monotonic()
    with common.locked(path.with_suffix(".watch.lock")):
        while True:
            with common.locked(path.with_suffix(".scheduler.lock")):
                policy, manifest = common.load_policy(path)
                common.verify_policy(policy, manifest)
                ledger = json.loads(schedule.ledger_path(path).read_text())
                if ledger["identity"] != policy["identity"]:
                    raise ValueError("Scheduler ledger identity changed")
                waiting = refresh(path, policy, manifest, ledger)
                timed_out = time.monotonic() - started >= max_seconds
                common.write_json(path.with_suffix(".watch.json"), {
                    "policy_identity": policy["identity"], "pid": os.getpid(), "updated_utc": schedule.utc(),
                    "status": "complete" if not waiting else "timed_out" if timed_out else "watching",
                    "pending_jobs_keep_h100_eligibility": True,
                })
            if not waiting or timed_out:
                return
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, metavar="MANIFEST")
    parser.add_argument("--from-policy", type=Path)
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            raise ValueError("Eligibility controller must run with JAX_PLATFORMS=cpu")
        watch(args.policy)
        return
    with common.locked(args.policy.with_suffix(".scheduler.lock")):
        if args.prepare:
            if args.from_policy is None:
                parser.error("Preparation requires --from-policy")
            print(json.dumps(prepare(args.policy, args.prepare, args.from_policy), indent=2))
            return
        policy, manifest = common.load_policy(args.policy)
        common.verify_policy(policy, manifest)
        ledger = json.loads(schedule.ledger_path(args.policy).read_text())
        if ledger["identity"] != policy["identity"]:
            raise ValueError("Scheduler ledger identity changed")
        if args.migrate:
            fast_handoff.migrate(args.policy, policy, manifest, ledger)
        if args.submit:
            print(json.dumps(submit(args.policy, policy, manifest, ledger)))
        if args.refresh:
            print(json.dumps({"pending_qualification": refresh(args.policy, policy, manifest, ledger)}))


if __name__ == "__main__":
    main()
