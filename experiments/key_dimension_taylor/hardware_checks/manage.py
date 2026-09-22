"""Prepare, queue, and summarize short GPU validation; never launch production."""
from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import common


def prepare(path: Path, training_manifest: Path, reused_policy: Path) -> dict:
    if path.exists():
        raise ValueError("Validation policy already exists")
    manifest = json.loads(training_manifest.read_text())
    old = json.loads(reused_policy.read_text())
    if (common.policy_identity(old) != old["identity"] or old["training_identity"] != manifest["identity"]
            or old["source_commit"] != manifest["source_commit"]):
        raise ValueError("Existing B300 evidence belongs to a different training recipe")
    root = Path(__file__).resolve().parents[3]
    output = Path(manifest["project_root"]) / "results/key_dimension_taylor/hardware_checks"
    sys.path.insert(0, manifest["execution_root"])
    from experiments.polynomial_attention.configs import config_to_dict, load_config, save_config
    from experiments.key_dimension_taylor.sweep import resolve_config, parameter_count, run_name
    base = load_config(next(run["config_path"] for run in manifest["runs"] if not run.get("retained_completed")))
    runs = []
    for width in manifest["key_dimensions"]:
        for degree in manifest["degrees"]:
            name = run_name(width, degree)
            config = resolve_config(base, Path(manifest["project_root"]), manifest["source_commit"], width, degree)
            config.train.workdir = output / "templates" / name
            config.train.wandb_mode = "disabled"
            config_path = output / "configs" / f"{name}.yaml"
            save_config(config, config_path)
            runs.append({"name": name, "qk_head_dim": width, "attention_degree": degree,
                         "config_path": str(config_path), "config_sha256": common.sha256(config_path),
                         "parameter_count": parameter_count(config.model), "config": config_to_dict(config)})
    profiles = {name: {"gpu_type": name, "gpu_count": 4, "kernel_gpu_count": 4,
                       "allowed_nodes": nodes, "partition": "sweep", "production_time": "08:00:00",
                       "validation_time": "04:00:00", "cpus_per_task": 32, "memory": "256G"}
                for name, nodes in {"b300": ["akita"], "a100": ["samoyed", "collie"],
                                    "a6000": ["doodle", "terrier"]}.items()}
    evidence = Path(old["qualification_root"]) / "b300"
    reused = {p.stem: common.reference(p) for p in sorted((evidence / "checks").glob("*.json"))
              if json.loads(p.read_text()).get("status") == "passed"}
    kernel = json.loads((evidence / "kernels.json").read_text())
    policy = {"format": "key-dimension-taylor-hardware-checks", "version": 1,
              "created_utc": datetime.now(UTC).isoformat(), "production_submission": False,
              "training_manifest": str(training_manifest.resolve()), "training_root": manifest["execution_root"],
              "training_identity": manifest["identity"], "source_commit": manifest["source_commit"],
              "dependency_lock_sha256": manifest["dependency_lock_sha256"], "data_identity": manifest["data_identity"],
              "orchestrator_root": str(root),
              "orchestrator_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "output_root": str(output), "profiles": profiles, "runs": runs,
              "key_dimensions": manifest["key_dimensions"], "degrees": manifest["degrees"],
              "shards": {profile: [[run["name"] for run in runs]] for profile in profiles},
              "reused_checks": {"b300": reused},
              "reused_kernels": {"b300": common.reference(evidence / "kernels.json")},
              "reference_files": [common.reference(reused_policy), common.reference(Path(kernel["junit_path"]))],
              "pilot_steps": 32, "continuation_step": 33}
    policy["identity"] = common.policy_identity(policy)
    common.verify_policy(policy, manifest)
    common.write_json(path, policy)
    common.write_json(path.with_suffix(".jobs.json"), {"identity": policy["identity"], "jobs": {}})
    return policy


def submit(path: Path, policy: dict) -> dict:
    ledger_path = path.with_suffix(".jobs.json")
    with common.locked(path.with_suffix(".submit.lock")):
        ledger = json.loads(ledger_path.read_text())
        if ledger["identity"] != policy["identity"]:
            raise ValueError("Validation submission ledger changed")
        queued = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%k"], text=True)
        for name, profile in policy["profiles"].items():
            comment = f"dkt-hardware:{policy['identity']}:{name}"
            entry = ledger["jobs"].get(name)
            if entry and entry.get("released_utc"):
                continue
            matches = [line.split("|", 1)[0] for line in queued.splitlines() if line.split("|", 1)[-1] == comment]
            if len(matches) > 1:
                raise RuntimeError("Duplicate validation submission")
            job = entry["slurm_job_id"] if entry else (matches[0] if matches else None)
            log = Path(policy["output_root"]).parent.parent / "logs/key_dimension_taylor" / f"hardware-{name}"
            log.mkdir(parents=True, exist_ok=True)
            if job is None:
                command = ["sbatch", "--parsable", "--hold", "--partition=sweep", "--nodes=1", "--ntasks=1",
                           f"--gres=gpu:{name}:4", "--cpus-per-task=32", "--mem=256G", "--time=04:00:00",
                           "--requeue", "--signal=B:TERM@120", f"--comment={comment}",
                           f"--job-name=dkt-check-{name}", f"--output={log}/slurm-%j.out", f"--error={log}/slurm-%j.out",
                           str(Path(policy["orchestrator_root"]) / "experiments/key_dimension_taylor/hardware_checks/launch.sbatch"),
                           str(path), name]
                job = subprocess.check_output(command, text=True, cwd=policy["orchestrator_root"]).strip().split(";")[0]
                if not job.isdigit():
                    raise RuntimeError(f"Invalid scheduler response: {job}")
            entry = entry or {"slurm_job_id": job, "profile": name, "comment": comment,
                              "log_path": str(log / f"slurm-{job}.out"), "submitted_utc": datetime.now(UTC).isoformat()}
            ledger["jobs"][name] = entry
            common.write_json(ledger_path, ledger)
            detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
            if "JobState=PENDING" in detail and "Priority=0 " in detail:
                subprocess.run(["scontrol", "release", job], check=True)
            entry["released_utc"] = datetime.now(UTC).isoformat()
            common.write_json(ledger_path, ledger)
        return ledger


def report(policy: dict) -> None:
    root = Path(policy["output_root"])
    with common.locked(root / ".report.lock"):
        rows = []
        kernels = {profile: (json.loads(path.read_text()) if (path := root / profile / "kernels.json").exists() else {})
                   for profile in policy["profiles"]}
        lines = ["# Blackwell and Ampere preliminary checks", "",
                 "Validation only: each new pilot performs 32 updates, evaluation and checkpoint save, then "
                 "a fresh-process continuation to update 33. No production jobs are submitted. "
                 "The full 19,074-update learning-rate schedule and production tensor/batch shapes are preserved.", "",
                 f"Scientific source: `{policy['source_commit']}`. Validation source: `{policy['orchestrator_commit']}`.",
                 "B300's existing kernel suite and five full-model pilots are referenced by checksum. "
                 "The four dK64 pilots test the current recipe without changing historical results or provenance.", "",
                 "Kernel suites: " + "; ".join(f"{profile}: {record.get('status', 'existing passed evidence' if profile in policy.get('reused_kernels', {}) else 'pending')}"
                                                 for profile, record in kernels.items()) + ".", "",
                 "| dK | m | B300 | A100 | RTX A6000 |", "|---:|---:|---|---|---|"]
        for run in policy["runs"]:
            states = []
            for profile in ("b300", "a100", "a6000"):
                path = root / profile / "checks" / f"{run['name']}.json"
                record = json.loads(path.read_text()) if path.exists() else {}
                reused = policy.get("reused_checks", {}).get(profile, {}).get(run["name"])
                status = record.get("status", "passed (existing evidence)" if reused else "pending")
                kernel_status = kernels[profile].get("status")
                if status == "pending" and kernel_status and kernel_status not in ("passed", "partial"):
                    status = "blocked (kernel checks)"
                states.append(status)
                rows.append({"profile": profile, "qk_head_dim": run["qk_head_dim"],
                             "attention_degree": run["attention_degree"], "status": status,
                             "record_path": str(path) if path.exists() else None,
                             "existing_evidence": reused, "error": record.get("error")})
            lines.append(f"| {run['qk_head_dim']} | {run['attention_degree']} | {' | '.join(states)} |")
        lines += ["", "These checks establish short-run compatibility, memory fit, resume fidelity, and timing. "
                  "They do not establish full-training stability or terminal perplexity. Pilot checkpoints are temporary "
                  "and removed after results/audits are recorded. Existing production jobs and their configurations are untouched.", ""]
        common.write_json(root / "summary.json", {"identity": policy["identity"], "rows": rows})
        (root / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--reuse-policy", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        policy = prepare(args.policy, args.prepare, args.reuse_policy)
    else:
        policy, manifest = common.load_policy(args.policy)
        common.verify_policy(policy, manifest)
    if args.submit:
        print(json.dumps(submit(args.policy, policy), indent=2))
    report(policy)


if __name__ == "__main__":
    main()
