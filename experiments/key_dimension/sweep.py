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


def resolve_runs(base, project_root: Path, source_commit: str, key_dimensions=KEY_DIMENSIONS) -> list[tuple[str, object]]:
    if not key_dimensions or len(set(key_dimensions)) != len(key_dimensions):
        raise ValueError("Choose at least one distinct query/key dimension")
    runs = []
    for key_dim in key_dimensions:
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


def make_manifest(base, project_root: Path, execution_root: Path, source_commit: str, key_dimensions=KEY_DIMENSIONS) -> dict:
    runs = resolve_runs(base, project_root, source_commit, key_dimensions)
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


def submit_sweep(manifest: dict, base, manifest_path: Path | None = None) -> dict:
    project_root = Path(manifest["project_root"])
    execution_root = Path(manifest["execution_root"])
    manifest_path = (manifest_path or project_root / "results/key_dimension/sweep.json").resolve()
    if manifest_path.parent != project_root.resolve() / "results/key_dimension" or manifest_path.suffix != ".json":
        raise ValueError("Sweep manifest must be a JSON file directly under results/key_dimension")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("jobs"):
            raise ValueError(f"Sweep already has submitted jobs; inspect/resume those jobs: {manifest_path}")
        if previous.get("identity") != manifest["identity"]:
            raise ValueError(f"Existing sweep has a different identity: {manifest_path}")
    dimensions = [run["qk_head_dim"] for run in manifest["runs"]]
    for (_, config), run in zip(resolve_runs(base, project_root, manifest["source_commit"], dimensions), manifest["runs"], strict=True):
        config_path = Path(run["config_path"])
        if config_path.exists() and config_to_dict(load_config(config_path)) != run["config"]:
            raise ValueError(f"Refusing to overwrite a different run config: {config_path}")
        if any((config_path.parent / name).exists() for name in ("launch.json", "metadata.json", "completion.json")):
            raise ValueError(f"Run already has launch or training records; inspect/resume it: {config_path.parent}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, config_path)
        # Keep large artifacts outside scientific results and the execution checkout.
        checkpoint_dir = project_root / "experiments/key_dimension/checkpoints" / run["name"]
        checkpoint_link = config_path.parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_link.is_symlink():
            if checkpoint_link.resolve() != checkpoint_dir.resolve():
                raise ValueError(f"Run checkpoints point to an unexpected location: {checkpoint_link}")
        elif checkpoint_link.exists():
            raise ValueError(f"Refusing to replace an existing checkpoint directory: {checkpoint_link}")
        else:
            checkpoint_link.symlink_to(checkpoint_dir.resolve(), target_is_directory=True)
        run["checkpoint_dir"] = str(checkpoint_dir.resolve())
        report_path = config_path.parent / "REPORT.md"
        if not report_path.exists():
            report_path.write_text(
                f"# {run['name']}\n\nStatus: queued for preflight and training.\n\n"
                f"- Query/key dimension: {run['qk_head_dim']}; value dimension: {run['value_head_dim']}.\n"
                f"- Parameters: {run['parameter_count']:,}; seed: 0.\n"
                f"- Prediction-token budget: {run['training_tokens']:,}.\n"
                f"- Source: `{manifest['source_commit']}`; resolved config: `config.yaml`.\n"
            )
    # Validate comparison provenance and recipe compatibility before launching GPUs.
    from experiments.key_dimension.plot import aggregate
    aggregate(manifest)
    manifest["submitted_utc"] = datetime.now(UTC).isoformat()
    write_json(manifest_path, manifest)
    launcher = execution_root / "experiments/key_dimension/launch.sbatch"
    common = ["sbatch", "--parsable", "--requeue", "--gres=gpu:h200:2"]
    preflight_name = "preflight" if manifest_path.name == "sweep.json" else f"preflight-{manifest_path.stem.removeprefix('sweep-')}"
    previous = submit_job(common + scheduler_log_options(project_root, preflight_name) + ["--time=01:00:00", f"--job-name=kd-{preflight_name}", str(launcher), "preflight", str(manifest_path), preflight_name], execution_root)
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
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("experiments/key_dimension/configs/base.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument("--key-dimensions", type=int, nargs="+", default=KEY_DIMENSIONS,
                        choices=(*KEY_DIMENSIONS, 256), help="Submit only these dimensions; default: original seven-run sweep.")
    parser.add_argument("--manifest", type=Path, help="Separate manifest under results/key_dimension, e.g. sweep-dk256.json.")
    parser.add_argument("--reference-manifest", type=Path, action="append", default=[],
                        help="Existing sweep to include in comparison plots without resubmitting its runs.")
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
    manifest = make_manifest(base, args.project_root, execution_root, source_commit, args.key_dimensions)
    if args.reference_manifest:
        manifest["reference_manifests"] = [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in args.reference_manifest
        ]
    if args.submit:
        manifest = submit_sweep(manifest, base, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
