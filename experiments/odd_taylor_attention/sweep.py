"""Resolve two matched odd-Taylor runs and submit gated Slurm jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from experiments.polynomial_attention.configs import config_to_dict, load_config, save_config, validate_config
from experiments.polynomial_attention.sweep import (
    scheduler_options, scheduler_profile as historical_scheduler_profile,
    submit_job, write_json,
)

EXPERIMENT = "odd_taylor_attention"
VARIANTS = (1, 3)
STABILITY_STEPS = 5000
CONTROL_NAMES = ("200m-m002-seed0", "200m-softmax-seed0")


def scheduler_profile() -> dict:
    return historical_scheduler_profile(4, node="dogo")


def run_name(degree: int, seed: int = 0) -> str:
    if degree not in VARIANTS:
        raise ValueError(f"Unknown odd Taylor degree: {degree}")
    return f"200m-m{degree:03d}-seed{seed}"


def manifest_path(project_root: Path) -> Path:
    return project_root / "results" / EXPERIMENT / "sweep.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficient_count(model) -> int:
    return 0


def parameter_count(model) -> int:
    # Count independently of implementation details in the legacy sweep helper.
    embedding = model.vocab_size * model.hidden_size * (1 if model.tie_embeddings else 2)
    attention = 2 * model.hidden_size * model.num_heads * (model.qk_head_dim + model.value_head_dim)
    block = attention + 3 * model.hidden_size * model.intermediate_size + 2 * model.hidden_size
    return embedding + model.num_layers * block + model.hidden_size + coefficient_count(model)


# Historical adapters are read-only and preserve original provenance/checksums.
from experiments.learned_quadratic_attention.sweep import (
    scientific_recipe, historical_controls, validate_control,
)


def resolve_runs(base, project_root: Path, source_commit: str) -> list[tuple[str, object]]:
    runs = []
    for degree in VARIANTS:
        config = copy.deepcopy(base)
        config.model.attention_degree = degree
        config.model.attention_backend = "fused"
        config.model.quadratic_mode = "fixed"
        config.model.quadratic_sharing = "global"
        config.train.seed = 0
        config.train.source_commit = source_commit
        name = run_name(degree)
        config.train.workdir = project_root.resolve() / "results" / EXPERIMENT / name
        config.train.wandb_project = "functional-memory-odd-taylor-attention"
        config.train.wandb_run_name = name
        validate_config(config)
        runs.append((name, config))
    return runs


def identity_payload(manifest: dict) -> dict:
    return {
        **{key: manifest[key] for key in ("source_commit", "dependency_lock_sha256", "data_identity", "scheduler")},
        "configs": [run["config"] for run in manifest["runs"]],
        "controls": manifest["controls"],
        "stability_steps": manifest["stability_steps"],
    }


def make_manifest(base, project_root: Path, execution_root: Path, source_commit: str,
                  *, controls: list[dict] | None = None) -> dict:
    project_root, execution_root = project_root.resolve(), execution_root.resolve()
    controls = historical_controls(project_root) if controls is None else copy.deepcopy(controls)
    if [run["name"] for run in controls] != list(CONTROL_NAMES):
        raise ValueError("Both historical controls are required in canonical order")
    for control in controls:
        validate_control(control)
    resolved = resolve_runs(base, project_root, source_commit)
    baseline_recipe = scientific_recipe(controls[0]["config"])
    if any(scientific_recipe(config_to_dict(config)) != baseline_recipe for _, config in resolved):
        raise ValueError("Odd Taylor runs must preserve the completed quadratic control's training recipe")
    if any(scientific_recipe(control["config"]) != baseline_recipe for control in controls):
        raise ValueError("Historical controls use different scientific recipes")
    manifest = {
        "schema_version": 1, "experiment": EXPERIMENT, "stability_steps": STABILITY_STEPS,
        "source_commit": source_commit,
        "dependency_lock_sha256": sha256(execution_root / "experiments/polynomial_attention/uv.lock"),
        "project_root": str(project_root), "execution_root": str(execution_root),
        "scheduler": scheduler_profile(),
        "data_identity": {name: {"path": str(Path(getattr(base.data, name)).resolve()),
                                 "sha256": sha256(Path(getattr(base.data, name)))}
                          for name in ("train_manifest", "eval_manifest")},
        "controls": controls,
        "runs": [{
            "name": name, "attention_degree": config.model.attention_degree,
            "quadratic_mode": config.model.quadratic_mode,
            "quadratic_sharing": config.model.quadratic_sharing,
            "qk_head_dim": config.model.qk_head_dim, "value_head_dim": config.model.value_head_dim,
            "parameter_count": parameter_count(config.model),
            "coefficient_count": coefficient_count(config.model),
            "workdir": str(config.train.workdir),
            "checkpoint_dir": str(project_root / "experiments" / EXPERIMENT / "checkpoints" / name),
            "config_path": str(project_root / "experiments" / EXPERIMENT / "configs" / f"{name}.yaml"),
            "training_tokens": config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1),
            "config": config_to_dict(config),
        } for name, config in resolved],
        "jobs": {},
    }
    manifest["identity"] = hashlib.sha256(json.dumps(identity_payload(manifest), sort_keys=True).encode()).hexdigest()
    manifest["preflight_dir"] = str(project_root / ".cache" / EXPERIMENT / "preflight" / manifest["identity"])
    return manifest


def prepare_sweep(manifest: dict, base) -> dict:
    path = manifest_path(Path(manifest["project_root"]))
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != manifest["identity"] or previous["execution_root"] != manifest["execution_root"]:
            raise ValueError("Existing odd-Taylor sweep has a different identity or frozen checkout")
        manifest = previous
    for (_, config), run in zip(resolve_runs(base, Path(manifest["project_root"]), manifest["source_commit"]), manifest["runs"], strict=True):
        if config_to_dict(config) != run["config"]:
            raise ValueError(f"Resolved config changed: {run['name']}")
        config_path = Path(run["config_path"])
        if config_path.exists() and config_to_dict(load_config(config_path)) != run["config"]:
            raise ValueError(f"Refusing to overwrite a different configuration: {config_path}")
        save_config(config, config_path)
        workdir, checkpoints = Path(run["workdir"]), Path(run["checkpoint_dir"])
        workdir.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        link = workdir / "checkpoints"
        if link.is_symlink():
            if link.resolve() != checkpoints.resolve():
                raise ValueError(f"Checkpoint symlink points elsewhere: {link}")
        elif link.exists():
            raise ValueError(f"Expected canonical checkpoint symlink, found a directory: {link}")
        else:
            link.symlink_to(checkpoints, target_is_directory=True)
    write_json(path, manifest)
    return manifest


def scheduler_log_options(project_root: Path, name: str) -> list[str]:
    log_dir = project_root / "results/logs" / EXPERIMENT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    return [f"--output={log_dir}/slurm-%j.out", f"--error={log_dir}/slurm-%j.out", "--open-mode=append"]


def submit_preflight(manifest: dict, base, *, launch: bool = False) -> dict:
    manifest = prepare_sweep(manifest, base)
    if manifest["jobs"].get("preflight"):
        raise ValueError("Sweep already has a preflight job; inspect/requeue it before retrying")
    root, execution = Path(manifest["project_root"]), Path(manifest["execution_root"])
    command = ["sbatch", "--parsable", *scheduler_options(manifest["scheduler"], preflight=True),
               *scheduler_log_options(root, "preflight"), "--job-name=odd-preflight",
               str(execution / "experiments" / EXPERIMENT / "launch.sbatch"),
               "preflight-and-submit" if launch else "preflight", str(manifest_path(root))]
    job_id = submit_job(command, execution)
    manifest["jobs"]["preflight"] = job_id
    manifest["launch_after_preflight"] = launch
    manifest["preflight_submitted_utc"] = datetime.now(UTC).isoformat()
    manifest["preflight_log_path"] = str(root / "results/logs" / EXPERIMENT / "preflight" / f"slurm-{job_id}.out")
    write_json(manifest_path(root), manifest)
    return manifest


def submit_sweep(manifest: dict, base) -> dict:
    from experiments.odd_taylor_attention.preflight import eligible_runs, run_preflight
    manifest = prepare_sweep(manifest, base)
    run_preflight(manifest, check_only=True)
    eligible = eligible_runs(manifest)
    root, execution = Path(manifest["project_root"]), Path(manifest["execution_root"])
    previous = manifest["jobs"].get("preflight")
    dependency_type = "afterok"
    for run in manifest["runs"]:
        if run["name"] not in eligible:
            continue
        if run["name"] in manifest["jobs"]:
            previous, dependency_type = manifest["jobs"][run["name"]], "afterany"
            continue
        dependencies = [f"--dependency={dependency_type}:{previous}"] if previous else []
        command = ["sbatch", "--parsable", *scheduler_options(manifest["scheduler"], preflight=False),
                   *scheduler_log_options(root, run["name"]), *dependencies,
                   f"--job-name=odd-m{run['attention_degree']}",
                   str(execution / "experiments" / EXPERIMENT / "launch.sbatch"),
                   "train", str(manifest_path(root)), run["name"]]
        job_id = submit_job(command, execution, previous, dependency_type)
        manifest["jobs"][run["name"]] = job_id
        write_json(Path(run["workdir"]) / "launch.json", {
            "source_commit": manifest["source_commit"], "sweep_identity": manifest["identity"],
            "attention_degree": run["attention_degree"], "stability_steps": STABILITY_STEPS,
            "slurm_job_id": job_id, "dependency_job_id": previous,
            "dependency_type": dependency_type if previous else None,
            "log_path": str(root / "results/logs" / EXPERIMENT / run["name"] / f"slurm-{job_id}.out"),
            "data_identity": manifest["data_identity"], "scheduler": manifest["scheduler"],
        })
        write_json(manifest_path(root), manifest)
        previous, dependency_type = job_id, "afterany"
    manifest["production_submitted_utc"] = datetime.now(UTC).isoformat()
    write_json(manifest_path(root), manifest)
    from experiments.odd_taylor_attention.plot import aggregate
    aggregate(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path,
                        help="Defaults to the immutable completed fixed-quadratic control config.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.execution_root, text=True).strip()
    if args.preflight_only or args.launch or args.submit:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=args.execution_root, text=True)
        if dirty:
            raise ValueError("Execution checkout must be clean before submission")
        if args.project_root.resolve() == args.execution_root.resolve():
            raise ValueError("Use a separate frozen execution checkout and retained project root")
    controls = historical_controls(args.project_root.resolve())
    base_path = args.base_config or Path(controls[0]["config_path"])
    base = load_config(base_path)
    manifest = make_manifest(base, args.project_root, args.execution_root, source, controls=controls)
    if args.preflight_only or args.launch:
        manifest = submit_preflight(manifest, base, launch=args.launch)
    elif args.submit:
        manifest = submit_sweep(manifest, base)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
