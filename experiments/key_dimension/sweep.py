"""Resolve and optionally submit the fixed-backbone key-dimension sweep."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from experiments.key_dimension.configs import config_to_dict, load_config, save_config, validate_config

KEY_DIMENSIONS = (2, 4, 8, 16, 32, 64, 128)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_name(key_dim: int, seed: int = 0) -> str:
    return f"200m-dk{key_dim:03d}-seed{seed}"


def resolve_runs(base, project_root: Path, source_commit: str) -> list[tuple[str, object]]:
    runs = []
    for key_dim in KEY_DIMENSIONS:
        config = copy.deepcopy(base)
        config.model.qk_head_dim = key_dim
        config.train.seed = 0
        config.train.source_commit = source_commit
        config.train.wandb_mode = "disabled"
        name = run_name(key_dim)
        config.train.wandb_run_name = name
        config.train.workdir = project_root.resolve() / "results" / "key_dimension" / name
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
    lock_sha256 = hashlib.sha256((execution_root / "experiments/key_dimension/uv.lock").read_bytes()).hexdigest()
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
        "preflight_dir": str(project_root.resolve() / ".cache/key_dimension/preflight" / fingerprint),
        "runs": [
            {
                "name": name,
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


def submit_job(command: list[str], execution_root: Path, dependency: str | None = None) -> str:
    output = subprocess.check_output(command, cwd=execution_root, text=True).strip()
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch job id: {output!r}")
    details = subprocess.check_output(["scontrol", "show", "job", "--oneliner", job_id], text=True)
    if "Requeue=1" not in details:
        raise RuntimeError(f"Submitted job {job_id} did not enable requeue: {details}")
    if dependency and f"afterok:{dependency}" not in details:
        dependency_details = subprocess.check_output(
            ["scontrol", "show", "job", "--oneliner", dependency], text=True,
        )
        if "JobState=COMPLETED" not in dependency_details:
            raise RuntimeError(f"Submitted job {job_id} lost its afterok dependency: {details}")
    return job_id


def scheduler_log_options(project_root: Path, name: str) -> list[str]:
    log_dir = project_root / "results/logs/key_dimension" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_pattern = log_dir / "slurm-%j.out"
    return [f"--output={log_pattern}", f"--error={log_pattern}", "--open-mode=append"]


def submit_sweep(manifest: dict, base) -> dict:
    project_root = Path(manifest["project_root"])
    execution_root = Path(manifest["execution_root"])
    manifest_path = project_root / "results/key_dimension/sweep.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("jobs"):
            raise ValueError(f"Sweep already has submitted jobs; inspect/resume those jobs: {manifest_path}")
        if previous.get("identity") != manifest["identity"]:
            raise ValueError(f"Existing sweep has a different identity: {manifest_path}")
    for (_, config), run in zip(resolve_runs(base, project_root, manifest["source_commit"]), manifest["runs"], strict=True):
        config_path = Path(run["config_path"])
        if config_path.exists() and config_to_dict(load_config(config_path)) != run["config"]:
            raise ValueError(f"Refusing to overwrite a different run config: {config_path}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, config_path)
        report_path = config_path.parent / "REPORT.md"
        if not report_path.exists():
            report_path.write_text(
                f"# {run['name']}\n\nStatus: queued for preflight and training.\n\n"
                f"- Query/key dimension: {run['qk_head_dim']}; value dimension: {run['value_head_dim']}.\n"
                f"- Parameters: {run['parameter_count']:,}; seed: 0.\n"
                f"- Prediction-token budget: {run['training_tokens']:,}.\n"
                f"- Source: `{manifest['source_commit']}`; resolved config: `config.yaml`.\n"
            )
    manifest["submitted_utc"] = datetime.now(UTC).isoformat()
    write_json(manifest_path, manifest)
    launcher = execution_root / "experiments/key_dimension/launch.sbatch"
    common = ["sbatch", "--parsable", "--requeue", "--gpus=h200:2"]
    previous = submit_job(common + scheduler_log_options(project_root, "preflight") + ["--time=01:00:00", "--job-name=kd-preflight", str(launcher), "preflight", str(manifest_path)], execution_root)
    manifest["jobs"]["preflight"] = previous
    write_json(manifest_path, manifest)
    for run in manifest["runs"]:
        job_id = submit_job(common + scheduler_log_options(project_root, run["name"]) + [f"--dependency=afterok:{previous}", f"--job-name=kd-{run['qk_head_dim']:03d}", str(launcher), "train", str(manifest_path), run["name"]], execution_root, previous)
        manifest["jobs"][run["name"]] = job_id
        write_json(Path(run["workdir"]) / "launch.json", {
            "source_commit": manifest["source_commit"],
            "sweep_identity": manifest["identity"],
            "slurm_job_id": job_id,
            "dependency_job_id": previous,
            "log_path": str(project_root / "results/logs/key_dimension" / run["name"] / f"slurm-{job_id}.out"),
            "data_identity": manifest["data_identity"],
        })
        write_json(manifest_path, manifest)
        previous = job_id
    from experiments.key_dimension.plot import aggregate
    aggregate(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("experiments/key_dimension/configs/base.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print resolved jobs; this is the default.")
    mode.add_argument("--submit", action="store_true", help="Write runtime configs and submit the gated sweep.")
    args = parser.parse_args()
    execution_root = args.execution_root.resolve()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution_root, text=True).strip()
    if args.submit:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution_root, text=True)
        if dirty:
            raise ValueError(f"Execution checkout must be clean before submission: {execution_root}")
    base = load_config(args.base_config)
    manifest = make_manifest(base, args.project_root, execution_root, source_commit)
    if args.submit:
        manifest = submit_sweep(manifest, base)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
