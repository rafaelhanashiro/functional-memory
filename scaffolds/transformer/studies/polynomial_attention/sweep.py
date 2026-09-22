"""Resolve the Taylor-degree sweep and submit pilots before production jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scaffolds.transformer.studies.polynomial_attention.configs import ATTENTION_DEGREES, config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.studies.polynomial_attention.configs.loader import config_dicts_equal


def scheduler_profile(gpu_count: int = 2, *, node: str | None = None) -> dict:
    """The supported single-node Kennel allocations for this fixed recipe."""
    if type(gpu_count) is not int or gpu_count not in (2, 4):
        raise ValueError("Kennel GPU count must be 2 or 4")
    if node not in (None, "dogo"):
        raise ValueError("The supported SXM node is dogo")
    profile = {
        "partition": "long",
        "gpu_type": "h100",
        "gpu_count": gpu_count,
        "cpus_per_task": 32,
        "memory": "256G",
        "preflight_time": "08:00:00",
        "production_time": "7-00:00:00",
    }
    if node is not None:
        profile.update(node=node, partition="standard", preflight_time="02:00:00", production_time="08:00:00")
    return profile


def validate_scheduler_profile(profile: dict) -> None:
    if not isinstance(profile, dict) or profile != scheduler_profile(profile.get("gpu_count"), node=profile.get("node")):
        raise ValueError("Unsupported Kennel scheduler profile")


def resolve_scheduler_profile(project_root: Path, gpu_count: int | None = None) -> dict:
    """Reuse the persisted allocation when a follow-up submission omits it."""
    manifest_path = project_root / "results/polynomial_attention/sweep.json"
    if not manifest_path.exists():
        return scheduler_profile(2 if gpu_count is None else gpu_count)
    previous = json.loads(manifest_path.read_text())
    profile = previous.get("scheduler")
    if profile is None:
        raise ValueError(f"Existing sweep predates the Kennel scheduler profile: {manifest_path}")
    validate_scheduler_profile(profile)
    if gpu_count is not None and gpu_count != profile["gpu_count"]:
        raise ValueError(f"GPU count differs from the persisted sweep: {manifest_path}")
    return profile.copy()


def scheduler_options(profile: dict, *, preflight: bool) -> list[str]:
    validate_scheduler_profile(profile)
    walltime = profile["preflight_time" if preflight else "production_time"]
    options = [
        "--requeue", "--nodes=1", "--ntasks=1",
        f"--partition={profile['partition']}",
        f"--gres=gpu:{profile['gpu_type']}:{profile['gpu_count']}",
        f"--cpus-per-task={profile['cpus_per_task']}",
        f"--mem={profile['memory']}", f"--time={walltime}",
    ]
    if profile.get("node"):
        options.append(f"--nodelist={profile['node']}")
    return options


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_name(degree: int | None, seed: int = 0) -> str:
    kernel = "softmax" if degree is None else f"m{degree:03d}"
    return f"200m-{kernel}-seed{seed}"


def resolve_runs(base, project_root: Path, source_commit: str) -> list[tuple[str, object]]:
    runs = []
    for degree in ATTENTION_DEGREES:
        config = copy.deepcopy(base)
        config.model.attention_degree = degree
        config.train.seed = 0
        config.train.source_commit = source_commit
        name = run_name(degree)
        config.train.wandb_run_name = name
        config.train.workdir = project_root.resolve() / "results" / "polynomial_attention" / "runs" / name
        config.train.checkpoint_dir = project_root.resolve() / "checkpoints" / "polynomial_attention" / name
        validate_config(config)
        runs.append((name, config))
    return runs


def parameter_count(model) -> int:
    embedding = model.vocab_size * model.hidden_size * (1 if model.tie_embeddings else 2)
    attention = 2 * model.hidden_size * model.num_heads * (model.qk_head_dim + model.value_head_dim)
    block = attention + 3 * model.hidden_size * model.intermediate_size + 2 * model.hidden_size
    return embedding + model.num_layers * block + model.hidden_size


def manifest_identity_payload(manifest: dict) -> dict:
    payload = {key: manifest[key] for key in (
        "source_commit", "dependency_lock_sha256", "data_identity", "scheduler",
    )}
    if "dependency_lock_path" in manifest:
        payload["dependency_lock_path"] = manifest["dependency_lock_path"]
    payload["configs"] = [run["config"] for run in manifest["runs"]]
    if manifest.get("schema_version", 2) >= 3:
        payload["retained_runs"] = [
            {"name": run["name"], "completion_sha256": run["retained_completion_sha256"]}
            for run in manifest["runs"] if run.get("retained_completed")
        ]
    return payload


def validate_retained_run(run: dict) -> None:
    """Accept the existing completed uniform-attention control without relabeling it."""
    if not run.get("retained_completed") or run["attention_degree"] != 0:
        raise ValueError("Only the completed degree-zero control can be retained")
    if not config_dicts_equal(config_to_dict(load_config(run["config_path"])), run["config"]):
        raise ValueError(f"Retained run config changed: {run['name']}")
    completion = Path(run["workdir"]) / "completion.json"
    if hashlib.sha256(completion.read_bytes()).hexdigest() != run["retained_completion_sha256"]:
        raise ValueError(f"Retained completion changed: {completion}")
    from scaffolds.transformer.studies.polynomial_attention.plot import summarize_run
    row, _ = summarize_run(run, run["config"]["train"]["source_commit"])
    if row["status"] != "complete":
        raise ValueError(f"Retained run is not complete: {run['name']}")


def make_manifest(base, project_root: Path, execution_root: Path, source_commit: str, *, scheduler: dict | None = None,
                  retained_runs: list[dict] | None = None) -> dict:
    scheduler = scheduler_profile() if scheduler is None else scheduler.copy()
    validate_scheduler_profile(scheduler)
    runs = resolve_runs(base, project_root, source_commit)
    data_identity = {
        name: {
            "path": str(Path(getattr(base.data, name)).resolve()),
            "sha256": hashlib.sha256(Path(getattr(base.data, name)).read_bytes()).hexdigest(),
        }
        for name in ("train_manifest", "eval_manifest")
    }
    lock_sha256 = hashlib.sha256((execution_root / "scaffolds/transformer/uv.lock").read_bytes()).hexdigest()
    identity_payload = {
        "source_commit": source_commit,
        "dependency_lock_path": "scaffolds/transformer/uv.lock",
        "dependency_lock_sha256": lock_sha256,
        "data_identity": data_identity,
        "scheduler": scheduler,
        "configs": [config_to_dict(config) for _, config in runs],
    }
    fingerprint = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": 2,
        "source_commit": source_commit,
        "dependency_lock_path": "scaffolds/transformer/uv.lock",
        "dependency_lock_sha256": lock_sha256,
        "project_root": str(project_root.resolve()),
        "execution_root": str(execution_root.resolve()),
        "identity": fingerprint,
        "data_identity": data_identity,
        "scheduler": scheduler,
        "preflight_dir": str(project_root.resolve() / ".cache/polynomial_attention/preflight" / fingerprint),
        "runs": [
            {
                "name": name,
                "attention_degree": config.model.attention_degree,
                "qk_head_dim": config.model.qk_head_dim,
                "value_head_dim": config.model.value_head_dim,
                "parameter_count": parameter_count(config.model),
                "workdir": str(config.train.workdir),
                "config_path": str(project_root.resolve() / "results/polynomial_attention/configs" / f"{name}.yaml"),
                "checkpoint_dir": str(config.train.checkpoint_dir),
                "training_tokens": config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1),
                "config": config_to_dict(config),
            }
            for name, config in runs
        ],
        "jobs": {},
    }
    if retained_runs:
        if len(retained_runs) != 1:
            raise ValueError("Expected only one retained degree-zero control")
        retained = copy.deepcopy(retained_runs[0])
        validate_retained_run(retained)
        candidate = manifest["runs"][0]
        if retained["name"] != candidate["name"] or retained["workdir"] != candidate["workdir"]:
            raise ValueError("Retained control must use the canonical degree-zero workdir")
        # Only execution provenance/backend can differ from the requested recipe.
        original, current = copy.deepcopy(retained["config"]), copy.deepcopy(candidate["config"])
        for config in (original, current):
            config["model"].pop("attention_backend")
            config["train"].pop("source_commit")
        if not config_dicts_equal(original, current):
            raise ValueError("Retained degree-zero control has a different scientific recipe")
        manifest["schema_version"] = 3
        manifest["runs"][0] = retained
        manifest["identity"] = hashlib.sha256(json.dumps(manifest_identity_payload(manifest), sort_keys=True).encode()).hexdigest()
        manifest["preflight_dir"] = str(project_root.resolve() / ".cache/polynomial_attention/preflight" / manifest["identity"])
    return manifest


def submit_job(command: list[str], execution_root: Path, dependency: str | None = None, dependency_type: str = "afterok") -> str:
    output = subprocess.check_output(command, cwd=execution_root, text=True).strip()
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch job id: {output!r}")
    details = subprocess.check_output(["scontrol", "show", "job", "--oneliner", job_id], text=True)
    if "Requeue=1" not in details:
        raise RuntimeError(f"Submitted job {job_id} did not enable requeue: {details}")
    if dependency and f"{dependency_type}:{dependency}" not in details:
        dependency_details = subprocess.check_output(
            ["scontrol", "show", "job", "--oneliner", dependency], text=True,
        )
        terminal_states = ("COMPLETED",) if dependency_type == "afterok" else (
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED",
        )
        if not any(f"JobState={state}" in dependency_details for state in terminal_states):
            raise RuntimeError(f"Submitted job {job_id} lost its {dependency_type} dependency: {details}")
    return job_id


def scheduler_log_options(project_root: Path, name: str) -> list[str]:
    log_dir = project_root / "results/logs/polynomial_attention" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_pattern = log_dir / "slurm-%j.out"
    return [f"--output={log_pattern}", f"--error={log_pattern}", "--open-mode=append"]


def prepare_sweep(manifest: dict, base) -> dict:
    """Persist exact resolved configurations, preserving any existing job identities."""
    validate_scheduler_profile(manifest["scheduler"])
    project_root = Path(manifest["project_root"])
    manifest_path = project_root / "results/polynomial_attention/sweep.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity") != manifest["identity"]:
            raise ValueError(f"Existing sweep has a different identity: {manifest_path}")
        if previous.get("execution_root") != manifest["execution_root"]:
            raise ValueError(f"Existing sweep has a different frozen execution checkout: {manifest_path}")
        manifest = previous
    for (_, config), run in zip(resolve_runs(base, project_root, manifest["source_commit"]), manifest["runs"], strict=True):
        if run.get("retained_completed"):
            validate_retained_run(run)
            continue
        if not config_dicts_equal(config_to_dict(config), run["config"]):
            raise ValueError(f"Resolved config differs from persisted sweep: {run['name']}")
        config_path = Path(run["config_path"])
        if config_path.exists() and not config_dicts_equal(config_to_dict(load_config(config_path)), run["config"]):
            raise ValueError(f"Refusing to overwrite a different run config: {config_path}")
        save_config(config, config_path)
        config.train.workdir.mkdir(parents=True, exist_ok=True)
        config.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_json(manifest_path, manifest)
    return manifest


def submit_preflight(manifest: dict, base, *, launch: bool = False) -> dict:
    manifest = prepare_sweep(manifest, base)
    if manifest["jobs"].get("preflight"):
        raise ValueError("Sweep already has a preflight job; inspect or requeue that job before retrying")
    project_root = Path(manifest["project_root"])
    execution_root = Path(manifest["execution_root"])
    launcher = execution_root / "scaffolds/transformer/studies/polynomial_attention/launch.sbatch"
    manifest_path = project_root / "results/polynomial_attention/sweep.json"
    command = ["sbatch", "--parsable", *scheduler_options(manifest["scheduler"], preflight=True)]
    command += scheduler_log_options(project_root, "preflight")
    command += ["--job-name=pa-preflight", str(launcher), "preflight-and-submit" if launch else "preflight", str(manifest_path)]
    job_id = submit_job(command, execution_root)
    manifest["jobs"]["preflight"] = job_id
    manifest["launch_after_preflight"] = launch
    manifest["preflight_submitted_utc"] = datetime.now(UTC).isoformat()
    manifest["preflight_log_path"] = str(project_root / "results/logs/polynomial_attention/preflight" / f"slurm-{job_id}.out")
    write_json(manifest_path, manifest)
    return manifest


def submit_sweep(manifest: dict, base, *, queue_after_preflight: bool = False) -> dict:
    """Submit checked runs, or queue them behind the matching preflight job.

    Every queued run depends on successful preflight, including later runs in
    the serial chain. The train launcher also rechecks the exact success record.
    """
    manifest = prepare_sweep(manifest, base)
    from scaffolds.transformer.studies.polynomial_attention.preflight import run_preflight, verify_manifest
    if queue_after_preflight:
        verify_manifest(manifest)
        if not manifest["jobs"].get("preflight"):
            raise ValueError("Queueing production requires a submitted matching preflight job")
    else:
        run_preflight(manifest, check_only=True)
    project_root = Path(manifest["project_root"])
    execution_root = Path(manifest["execution_root"])
    manifest_path = project_root / "results/polynomial_attention/sweep.json"
    launcher = execution_root / "scaffolds/transformer/studies/polynomial_attention/launch.sbatch"
    common = ["sbatch", "--parsable", *scheduler_options(manifest["scheduler"], preflight=False)]
    previous = manifest["jobs"].get("preflight")
    dependency_type = "afterok"
    for run in manifest["runs"]:
        if run.get("retained_completed"):
            validate_retained_run(run)
            continue
        if run["name"] in manifest["jobs"]:
            previous = manifest["jobs"][run["name"]]
            dependency_type = "afterany"
            continue
        dependencies = [f"{dependency_type}:{previous}"] if previous else []
        preflight_job = manifest["jobs"].get("preflight")
        if queue_after_preflight and previous != preflight_job:
            dependencies.insert(0, f"afterok:{preflight_job}")
        dependency = [f"--dependency={','.join(dependencies)}"] if dependencies else []
        label = "softmax" if run["attention_degree"] is None else f"m{run['attention_degree']:03d}"
        command = common + scheduler_log_options(project_root, run["name"]) + dependency
        command += [f"--job-name=pa-{label}", str(launcher), "train", str(manifest_path), run["name"]]
        job_id = submit_job(command, execution_root, previous, dependency_type)
        manifest["jobs"][run["name"]] = job_id
        write_json(Path(run["workdir"]) / "launch.json", {
            "source_commit": manifest["source_commit"],
            "sweep_identity": manifest["identity"],
            "slurm_job_id": job_id,
            "dependency_job_id": previous,
            "dependency_type": dependency_type if previous else None,
            "preflight_dependency_job_id": preflight_job if queue_after_preflight else None,
            "log_path": str(project_root / "results/logs/polynomial_attention" / run["name"] / f"slurm-{job_id}.out"),
            "data_identity": manifest["data_identity"],
            "scheduler": manifest["scheduler"],
        })
        write_json(manifest_path, manifest)
        previous = job_id
        dependency_type = "afterany"
    manifest["production_submitted_utc"] = datetime.now(UTC).isoformat()
    write_json(manifest_path, manifest)
    from scaffolds.transformer.studies.polynomial_attention.plot import aggregate
    aggregate(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("results/polynomial_attention/configs/base.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-count", type=int, choices=(2, 4),
                        help="H100 GPUs per node; defaults to the persisted allocation, or 2 for a new sweep.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print resolved jobs; this is the default.")
    mode.add_argument("--preflight-only", action="store_true", help="Persist all configs and submit only the six-variant GPU preflight.")
    mode.add_argument("--launch", action="store_true", help="Submit preflight now; its launcher submits production only after all checks pass.")
    mode.add_argument("--submit", action="store_true", help="Submit production only if this exact sweep has passed preflight.")
    mode.add_argument("--queue", action="store_true", help="Queue preflight and all production jobs with success dependencies.")
    args = parser.parse_args()
    execution_root = args.execution_root.resolve()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution_root, text=True).strip()
    if args.submit or args.preflight_only or args.launch or args.queue:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution_root, text=True)
        if dirty:
            raise ValueError(f"Execution checkout must be clean before submission: {execution_root}")
    base = load_config(args.base_config)
    scheduler = resolve_scheduler_profile(args.project_root, args.gpu_count)
    persisted_path = args.project_root / "results/polynomial_attention/sweep.json"
    retained = ([run for run in json.loads(persisted_path.read_text())["runs"] if run.get("retained_completed")]
                if persisted_path.exists() else None)
    manifest = make_manifest(base, args.project_root, execution_root, source_commit, scheduler=scheduler, retained_runs=retained)
    if args.queue:
        manifest = submit_preflight(manifest, base)
        manifest = submit_sweep(manifest, base, queue_after_preflight=True)
    elif args.preflight_only or args.launch:
        manifest = submit_preflight(manifest, base, launch=args.launch)
    elif args.submit:
        manifest = submit_sweep(manifest, base)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
