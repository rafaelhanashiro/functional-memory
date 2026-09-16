"""Resolve the Taylor-degree sweep and submit pilots before production jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from experiments.polynomial_attention.configs import ATTENTION_DEGREES, config_to_dict, load_config, save_config, validate_config


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
        config.model.attention_backend = "blocked"
        config.train.seed = 0
        config.train.source_commit = source_commit
        name = run_name(degree)
        config.train.wandb_run_name = name
        config.train.workdir = project_root.resolve() / "results" / "polynomial_attention" / name
        validate_config(config)
        runs.append((name, config))
    return runs


def parameter_count(model) -> int:
    embedding = model.vocab_size * model.hidden_size * (1 if model.tie_embeddings else 2)
    attention = 2 * model.hidden_size * model.num_heads * (model.qk_head_dim + model.value_head_dim)
    block = attention + 3 * model.hidden_size * model.intermediate_size + 2 * model.hidden_size
    return embedding + model.num_layers * block + model.hidden_size


def make_manifest(base, project_root: Path, execution_root: Path, source_commit: str) -> dict:
    runs = resolve_runs(base, project_root, source_commit)
    data_identity = {
        name: {
            "path": str(Path(getattr(base.data, name)).resolve()),
            "sha256": hashlib.sha256(Path(getattr(base.data, name)).read_bytes()).hexdigest(),
        }
        for name in ("train_manifest", "eval_manifest")
    }
    lock_sha256 = hashlib.sha256((execution_root / "experiments/polynomial_attention/uv.lock").read_bytes()).hexdigest()
    identity_payload = {
        "source_commit": source_commit,
        "dependency_lock_sha256": lock_sha256,
        "data_identity": data_identity,
        "configs": [config_to_dict(config) for _, config in runs],
    }
    fingerprint = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "dependency_lock_sha256": lock_sha256,
        "project_root": str(project_root.resolve()),
        "execution_root": str(execution_root.resolve()),
        "identity": fingerprint,
        "data_identity": data_identity,
        "preflight_dir": str(project_root.resolve() / ".cache/polynomial_attention/preflight" / fingerprint),
        "runs": [
            {
                "name": name,
                "attention_degree": config.model.attention_degree,
                "qk_head_dim": config.model.qk_head_dim,
                "value_head_dim": config.model.value_head_dim,
                "parameter_count": parameter_count(config.model),
                "workdir": str(config.train.workdir),
                "config_path": str(config.train.workdir / "config.yaml"),
                "training_tokens": config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1),
                "config": config_to_dict(config),
            }
            for name, config in runs
        ],
        "jobs": {},
    }


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
        if config_to_dict(config) != run["config"]:
            raise ValueError(f"Resolved config differs from persisted sweep: {run['name']}")
        config_path = Path(run["config_path"])
        if config_path.exists() and config_to_dict(load_config(config_path)) != run["config"]:
            raise ValueError(f"Refusing to overwrite a different run config: {config_path}")
        save_config(config, config_path)
        report_path = config_path.parent / "REPORT.md"
        if not report_path.exists():
            degree = run["attention_degree"]
            label = "exact softmax" if degree is None else f"Taylor degree {degree}"
            report_path.write_text(
                f"# {run['name']}\n\nStatus: awaiting successful preflight.\n\n"
                f"- Kernel: {label}; query/key and value dimensions: 64.\n"
                f"- Parameters: {run['parameter_count']:,}; seed: 0.\n"
                f"- Prediction-token budget: {run['training_tokens']:,}.\n"
                f"- Source: `{manifest['source_commit']}`; resolved config: `config.yaml`.\n"
                + ("- Degree zero gives causal prefix averaging; Q/K projections receive no attention-derived learning signal.\n" if degree == 0 else "")
            )
    write_json(manifest_path, manifest)
    return manifest


def submit_preflight(manifest: dict, base, *, launch: bool = False) -> dict:
    manifest = prepare_sweep(manifest, base)
    if manifest["jobs"].get("preflight"):
        raise ValueError("Sweep already has a preflight job; inspect or requeue that job before retrying")
    project_root = Path(manifest["project_root"])
    execution_root = Path(manifest["execution_root"])
    launcher = execution_root / "experiments/polynomial_attention/launch.sbatch"
    manifest_path = project_root / "results/polynomial_attention/sweep.json"
    command = ["sbatch", "--parsable", "--requeue", "--gres=gpu:h200:2"]
    command += scheduler_log_options(project_root, "preflight")
    command += ["--time=08:00:00", "--job-name=pa-preflight", str(launcher), "preflight-and-submit" if launch else "preflight", str(manifest_path)]
    job_id = submit_job(command, execution_root)
    manifest["jobs"]["preflight"] = job_id
    manifest["launch_after_preflight"] = launch
    manifest["preflight_submitted_utc"] = datetime.now(UTC).isoformat()
    manifest["preflight_log_path"] = str(project_root / "results/logs/polynomial_attention/preflight" / f"slurm-{job_id}.out")
    write_json(manifest_path, manifest)
    return manifest


def submit_sweep(manifest: dict, base) -> dict:
    """Submit production only after *all* matching pilots have actually passed."""
    manifest = prepare_sweep(manifest, base)
    from experiments.polynomial_attention.preflight import run_preflight
    run_preflight(manifest, check_only=True)
    project_root = Path(manifest["project_root"])
    execution_root = Path(manifest["execution_root"])
    manifest_path = project_root / "results/polynomial_attention/sweep.json"
    launcher = execution_root / "experiments/polynomial_attention/launch.sbatch"
    common = ["sbatch", "--parsable", "--requeue", "--gres=gpu:h200:2"]
    previous = manifest["jobs"].get("preflight")
    dependency_type = "afterok"
    for run in manifest["runs"]:
        if run["name"] in manifest["jobs"]:
            previous = manifest["jobs"][run["name"]]
            dependency_type = "afterany"
            continue
        dependency = [f"--dependency={dependency_type}:{previous}"] if previous else []
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
            "log_path": str(project_root / "results/logs/polynomial_attention" / run["name"] / f"slurm-{job_id}.out"),
            "data_identity": manifest["data_identity"],
        })
        write_json(manifest_path, manifest)
        previous = job_id
        dependency_type = "afterany"
    manifest["production_submitted_utc"] = datetime.now(UTC).isoformat()
    write_json(manifest_path, manifest)
    from experiments.polynomial_attention.plot import aggregate
    aggregate(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("experiments/polynomial_attention/configs/base.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print resolved jobs; this is the default.")
    mode.add_argument("--preflight-only", action="store_true", help="Persist all configs and submit only the six-variant GPU preflight.")
    mode.add_argument("--launch", action="store_true", help="Submit preflight now; its launcher submits production only after all checks pass.")
    mode.add_argument("--submit", action="store_true", help="Submit production only if this exact sweep has passed preflight.")
    args = parser.parse_args()
    execution_root = args.execution_root.resolve()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution_root, text=True).strip()
    if args.submit or args.preflight_only or args.launch:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution_root, text=True)
        if dirty:
            raise ValueError(f"Execution checkout must be clean before submission: {execution_root}")
    base = load_config(args.base_config)
    manifest = make_manifest(base, args.project_root, execution_root, source_commit)
    if args.preflight_only or args.launch:
        manifest = submit_preflight(manifest, base, launch=args.launch)
    elif args.submit:
        manifest = submit_sweep(manifest, base)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
