"""Replace pending serial jobs and submit independent hardware-qualified runs."""
from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from . import common
except ImportError:
    import common


def utc() -> str:
    return datetime.now(UTC).isoformat()


def ledger_path(path: Path) -> Path:
    return path.with_suffix(".jobs.json")


def prepare(path: Path, manifest_path: Path, b300_names: set[str]) -> dict:
    if path.exists():
        raise ValueError("Execution policy already exists; use its frozen identity")
    manifest = json.loads(manifest_path.read_text())
    root = Path(__file__).resolve().parents[3]
    assignments, superseded, legacy_renderers = {}, {}, []
    all_jobs = [str(job) for name, jobs in manifest["jobs"].items() if name != "preflight" for job in jobs]
    states = common.job_states(all_jobs)
    for run in manifest["runs"]:
        if run.get("retained_completed") or Path(run["completion_path"]).exists():
            continue
        jobs = [str(job) for job in manifest["jobs"].get(run["name"], [])]
        if jobs and all(states.get(job) == "PENDING" for job in jobs):
            assignments[run["name"]] = "b300" if run["name"] in b300_names else "h100"
            superseded[run["name"]] = jobs
        elif jobs and any(states.get(job) == "RUNNING" for job in jobs):
            legacy_renderers.extend(jobs)
        else:
            raise RuntimeError(f"Coordinate has an unexpected migration state: {run['name']}")
    if not b300_names or not b300_names <= assignments.keys():
        raise ValueError("B300 migration must select pending, unstarted coordinates")
    profile = dict(manifest["scheduler"])
    policy = {"format": "key-dimension-taylor-execution-amendment", "version": 1,
              "created_utc": utc(), "training_manifest": str(manifest_path.resolve()),
              "training_identity": manifest["identity"], "source_commit": manifest["source_commit"],
              "training_root": manifest["execution_root"], "orchestrator_root": str(root),
              "orchestrator_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "profiles": {"h100": profile, "b300": dict(profile, gpu_type="b300", node="akita", partition="sweep")},
              "assignments": assignments, "superseded_jobs": superseded,
              "legacy_renderer_jobs": legacy_renderers,
              "qualification_root": str(path.parent / "execution_validation" / path.stem),
              "authorization": "User requested concurrent B300 training alongside H100s on 2026-09-21",
              "scientific_changes": [], "cross_coordinate_dependencies": False}
    policy["identity"] = common.policy_identity(policy)
    common.verify_policy(policy, manifest)
    common.write_json(path, policy)
    common.write_json(ledger_path(path), {"identity": policy["identity"], "jobs": {}, "migration": {}, "created_utc": utc()})
    return policy


def migrate(path: Path, policy: dict, manifest: dict, ledger: dict) -> None:
    if ledger.get("migration", {}).get("status") == "complete":
        return
    all_jobs = [job for jobs in policy["superseded_jobs"].values() for job in jobs]
    states = common.job_states(all_jobs)
    # Hold the entire old dependency chain before cancelling anything. Refuse
    # running jobs even if they started after the original policy snapshot.
    for job in all_jobs:
        if states.get(job) == "PENDING":
            subprocess.run(["scontrol", "hold", job], check=True)
        elif states.get(job) != "CANCELLED":
            raise RuntimeError(f"Refusing migration of non-pending legacy job {job}: {states.get(job)}")
    states = common.job_states(all_jobs)
    if any(states.get(job) not in {"PENDING", "CANCELLED"} for job in all_jobs):
        raise RuntimeError(f"Legacy ownership changed while holding jobs: {states}")
    for run in manifest["runs"]:
        if run["name"] not in policy["assignments"]:
            continue
        if Path(run["completion_path"]).exists() or (Path(run["workdir"]) / "metadata.json").exists():
            raise RuntimeError(f"Refusing migration of a coordinate with training artifacts: {run['name']}")
        if any(Path(run["checkpoint_dir"]).iterdir()):
            raise RuntimeError(f"Refusing migration with existing checkpoints: {run['name']}")
    snapshot = {"status": "held", "at_utc": utc(), "jobs": states,
                "original_manifest_jobs": manifest["jobs"]}
    ledger["migration"] = snapshot
    common.write_json(ledger_path(path), ledger)
    for job in reversed(all_jobs):
        if states[job] == "PENDING":
            subprocess.run(["scancel", job], check=True)
    final = common.job_states(all_jobs)
    if any(final.get(job) != "CANCELLED" for job in all_jobs):
        raise RuntimeError(f"Legacy cancellation has not settled; rerun migration safely: {final}")
    ledger["migration"] = {**snapshot, "status": "complete", "completed_utc": utc(), "final_states": final}
    for name in policy["assignments"]:
        manifest["jobs"][name] = []
    common.write_json(Path(policy["training_manifest"]), manifest)
    common.write_json(ledger_path(path), ledger)


def _existing_comment(comment: str) -> str | None:
    output = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%k"], text=True)
    matches = [line.split("|", 1)[0] for line in output.splitlines() if line.split("|", 1)[-1] == comment]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate ownership for Slurm comment {comment}")
    return matches[0] if matches else None


def submit_held(path: Path, policy: dict, manifest: dict, ledger: dict, *, key: str,
                profile_name: str, run_name: str, segment: int = 0, dependency: str | None = None) -> str:
    comment = f"dkt:{policy['identity']}:{key}"
    previous_entry = ledger["jobs"].get(key)
    job = previous_entry["slurm_job_id"] if previous_entry else _existing_comment(comment)
    profile = policy["profiles"][profile_name]
    mode = "qualify" if key == "qualification:b300" else "train"
    log_dir = Path(manifest["project_root"]) / "results/logs/key_dimension_taylor" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    if job is None:
        command = ["sbatch", "--parsable", "--hold", "--nodes=1", "--ntasks=1", "--requeue",
                   "--signal=B:TERM@120", f"--partition={profile['partition']}", f"--nodelist={profile['node']}",
                   f"--gres=gpu:{profile['gpu_type']}:4", f"--time={profile['production_time']}",
                   f"--cpus-per-task={profile['cpus_per_task']}", f"--mem={profile['memory']}",
                   f"--comment={comment}", f"--job-name=dkt-{profile_name}-{run_name}-s{segment}",
                   f"--output={log_dir}/slurm-%j.out", f"--error={log_dir}/slurm-%j.out"]
        if dependency:
            command += [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
        command += [str(Path(policy["orchestrator_root"]) / "experiments/key_dimension_taylor/concurrent/launch.sbatch"),
                    mode, str(path), run_name, str(segment)]
        job = subprocess.check_output(command, cwd=policy["orchestrator_root"], text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Invalid sbatch response: {job}")
    entry = previous_entry or {"slurm_job_id": job, "profile": profile_name, "run_name": run_name, "segment": segment,
             "dependency_job_id": dependency, "comment": comment, "submitted_utc": utc(),
             "log_path": str(log_dir / f"slurm-{job}.out")}
    ledger["jobs"][key] = entry
    common.write_json(ledger_path(path), ledger)
    if mode == "train":
        manifest["jobs"].setdefault(run_name, [])
        if job not in manifest["jobs"][run_name]:
            manifest["jobs"][run_name].append(job)
        run = next(run for run in manifest["runs"] if run["name"] == run_name)
        launch_path = Path(run["workdir"]) / "launch.json"
        with common.locked(Path(run["workdir"]) / ".launch-metadata.lock"):
            launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
            launch.setdefault("execution_amendments", {})[policy["identity"]] = {
                "policy_path": str(path), "orchestrator_commit": policy["orchestrator_commit"],
                "superseded_jobs": policy["superseded_jobs"][run_name]}
            entries = launch.setdefault("concurrent_segments", [])
            if not any(item["slurm_job_id"] == job for item in entries):
                entries.append(entry.copy())
            common.write_json(launch_path, launch)
        common.write_json(Path(policy["training_manifest"]), manifest)
    if not entry.get("released_utc"):
        state = common.job_states([job]).get(job)
        if state == "PENDING":
            detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
            if "JobState=PENDING" in detail and "Priority=0 " in detail:
                subprocess.run(["scontrol", "release", job], check=True)
        elif state is None:
            raise RuntimeError(f"Submitted job has no scheduler/accounting record: {job}")
        entry["released_utc"] = utc()
        common.write_json(ledger_path(path), ledger)
    return job


def submit_profile(path: Path, policy: dict, manifest: dict, ledger: dict, profile_name: str) -> list[str]:
    jobs = []
    if ledger.get("migration", {}).get("status") != "complete":
        raise RuntimeError("Complete legacy migration before submitting production")
    for name, assigned in policy["assignments"].items():
        if assigned != profile_name:
            continue
        common.assert_legacy_jobs_terminal(policy, manifest, name)
        if profile_name == "h100":
            from experiments.key_dimension_taylor.preflight import eligible_runs
            if name not in eligible_runs(manifest):
                raise RuntimeError(f"Original H100 qualification is missing: {name}")
            check_path = Path(manifest["preflight_dir"]) / "checks" / f"{name}.json"
            expected_identity = manifest["identity"]
        else:
            check_path = Path(policy["qualification_root"]) / profile_name / "checks" / f"{name}.json"
            expected_identity = policy["identity"]
        if not check_path.exists():
            continue
        check = json.loads(check_path.read_text())
        if check.get("identity") != expected_identity:
            raise ValueError(f"Qualification identity mismatch: {name}")
        if check.get("status") == "ineligible":
            run = next(run for run in manifest["runs"] if run["name"] == name)
            common.write_json(Path(run["workdir"]) / "preflight_failure.json",
                              {**check, "policy_identity": policy["identity"],
                               "stage": "hardware_qualification", "qualification_path": str(check_path)})
            continue
        if check.get("status") != "passed":
            raise RuntimeError(f"Coordinate is not qualified: {name}")
        if profile_name == "b300":
            kernel = json.loads((check_path.parent.parent / "kernels.json").read_text())
            if kernel.get("identity") != expected_identity or kernel.get("status") != "passed":
                raise RuntimeError("B300 kernel validation has not passed")
        previous = None
        for index, _ in enumerate(check["runtime"]["segments"]):
            previous = submit_held(path, policy, manifest, ledger, key=f"{name}:{index}",
                                   profile_name=profile_name, run_name=name, segment=index, dependency=previous)
            jobs.append(previous)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, metavar="MANIFEST")
    parser.add_argument("--b300-names", default="")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--submit", choices=("h100", "b300", "qualification"))
    args = parser.parse_args()
    with common.locked(args.policy.with_suffix(".scheduler.lock")):
        if args.prepare:
            print(json.dumps(prepare(args.policy, args.prepare, set(args.b300_names.split(","))), indent=2))
            return
        policy, manifest = common.load_policy(args.policy)
        common.verify_policy(policy, manifest)
        ledger = json.loads(ledger_path(args.policy).read_text())
        if ledger["identity"] != policy["identity"]:
            raise ValueError("Scheduler ledger identity changed")
        if args.migrate:
            migrate(args.policy, policy, manifest, ledger)
        if args.submit == "qualification":
            result = submit_held(args.policy, policy, manifest, ledger, key="qualification:b300",
                                  profile_name="b300", run_name="b300-preflight")
        elif args.submit:
            result = submit_profile(args.policy, policy, manifest, ledger, args.submit)
        else:
            result = ledger
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
