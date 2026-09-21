"""Resolve a joint key-width/Taylor grid and submit independently gated segments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from experiments.polynomial_attention.configs import config_to_dict, load_config, save_config, validate_config
from experiments.polynomial_attention.sweep import (
    scheduler_options as historical_scheduler_options, scheduler_profile as historical_scheduler_profile, submit_job, write_json,
)

EXPERIMENT = "key_dimension_taylor"
KEY_DIMENSIONS = (8, 16, 32, 64, 128)
DEGREES = (2, 4, 8, 16)
CHECKPOINT_BOUNDARY = 500
ALLOCATION_FRACTION = 0.8
PILOT_STEPS = 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_name(key_dimension: int, degree: int, seed: int = 0) -> str:
    return f"200m-dk{key_dimension:03d}-m{degree:03d}-seed{seed}"


def manifest_path(project_root: Path) -> Path:
    return project_root / "results" / EXPERIMENT / "sweep.json"


def scheduler_profile() -> dict:
    result = historical_scheduler_profile(4, node="dogo")
    # All sixteen full-model pilots, their audits, and kernel compilation share
    # one allocation. Production segmentation preserves 20% of every allocation.
    result["preflight_time"] = "08:00:00"
    return result


def scheduler_options(profile: dict, *, preflight: bool) -> list[str]:
    if profile != scheduler_profile():
        raise ValueError("Unsupported joint-sweep scheduler profile")
    original = historical_scheduler_profile(4, node="dogo")
    options = historical_scheduler_options(original, preflight=preflight)
    if preflight:
        options = [f"--time={profile['preflight_time']}" if item.startswith("--time=") else item for item in options]
    return options


def parameter_count(model) -> int:
    embedding = model.vocab_size * model.hidden_size * (1 if model.tie_embeddings else 2)
    attention = 2 * model.hidden_size * model.num_heads * (model.qk_head_dim + model.value_head_dim)
    block = attention + 3 * model.hidden_size * model.intermediate_size + 2 * model.hidden_size
    return embedding + model.num_layers * block + model.hidden_size


def active_runs(manifest: dict) -> list[dict]:
    return [run for run in manifest["runs"] if not run.get("retained_completed", False)]


def selected_values(values, allowed, label):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(value not in allowed for value in values):
        raise ValueError(f"{label} must be distinct values drawn from {allowed}")
    return tuple(value for value in allowed if value in values)


def scientific_recipe(config: dict, data_identity: dict) -> dict:
    recipe = copy.deepcopy(config)
    for key in ("attention_degree", "attention_backend", "qk_head_dim", "quadratic_mode", "quadratic_sharing"):
        recipe["model"].pop(key, None)
    for key in ("source_commit", "workdir", "wandb_project", "wandb_run_name", "wandb_entity", "wandb_mode"):
        recipe["train"].pop(key, None)
    for key in ("train_manifest", "eval_manifest"):
        recipe["data"][key] = data_identity[key]["sha256"]
    return recipe


def artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Historical artifact path must be repository-relative: {value}")
    return root / path


def retained_runs(project_root: Path, experiment: str, names: list[str]) -> list[dict]:
    """Load original retained artifacts without changing their paths or bytes."""
    path = project_root / "results" / experiment / "provenance.json"
    provenance = json.loads(path.read_text())
    if (provenance.get("format") != "functional-memory-retained-experiment"
            or provenance.get("schema_version", provenance.get("version")) != 1 or provenance.get("experiment") != experiment):
        raise ValueError(f"Unsupported retained provenance: {path}")
    launch_paths = []
    for record in provenance.get("historical_launch_manifests", []):
        launch = artifact_path(project_root, record["path"])
        if sha256(launch) != record["sha256"]:
            raise ValueError(f"Historical launch manifest checksum changed: {launch}")
        launch_paths.append(launch)
    result = []
    for name in names:
        retained = copy.deepcopy(next(run for run in provenance["runs"] if run["name"] == name))
        paths = [path, *launch_paths]
        unavailable = []
        for key in ("resolved_config", "metrics", "slurm_log"):
            if key not in retained:
                raise ValueError(f"Historical run lacks {key}: {name}")
            item = retained[key]
            actual = artifact_path(project_root, item["path"])
            if key == "slurm_log" and not actual.exists():
                unavailable.append({"kind": key, **item})
                continue
            if sha256(actual) != item["sha256"]:
                raise ValueError(f"Historical {key} checksum changed: {name}")
            paths.append(actual)
        config_path = artifact_path(project_root, retained["resolved_config"]["path"])
        metrics_path = artifact_path(project_root, retained["metrics"]["path"])
        completion_path = artifact_path(project_root, retained["completion"])
        config = yaml.safe_load(config_path.read_text())
        degree = config["model"].get("attention_degree")
        if (config["train"]["source_commit"] != retained["source_commit"]
                or config["train"]["seed"] != retained["seed"]
                or config["model"]["qk_head_dim"] != retained["qk_head_dim"]
                or config["model"]["value_head_dim"] != retained["value_head_dim"]
                or degree != retained.get("attention_degree")):
            raise ValueError(f"Historical configuration identity changed: {name}")
        if metrics_path.parent != completion_path.parent or metrics_path.name != "metrics.jsonl" or completion_path.name != "completion.json":
            raise ValueError("Historical results must preserve canonical metric/completion filenames")
        paths.append(completion_path)
        selected = retained["selected_checkpoint"]
        checkpoint_metadata = artifact_path(project_root, selected["path"]) / "_CHECKPOINT_METADATA"
        if checkpoint_metadata.exists():
            if sha256(checkpoint_metadata) != selected["metadata_sha256"]:
                raise ValueError(f"Historical checkpoint metadata checksum changed: {name}")
            paths.append(checkpoint_metadata)
        else:
            unavailable.append({"kind": "selected_checkpoint", "path": selected["path"],
                                "metadata_sha256": selected["metadata_sha256"]})
        for filename in ("metadata.json", "best_validation.json"):
            extra = metrics_path.parent / filename
            if extra.exists():
                paths.append(extra)
        run = {**retained, "attention_degree": degree, "config": config,
               "config_path": str(config_path), "workdir": str(metrics_path.parent),
               "metrics_path": str(metrics_path), "completion_path": str(completion_path),
               "checkpoint_dir": str(artifact_path(project_root, retained["selected_checkpoint"]["path"]).parent),
               "training_tokens": retained["terminal"]["training_tokens"], "retained_completed": True,
               "historical_name": name, "original_experiment": experiment,
               "grid_name": run_name(retained["qk_head_dim"], degree) if degree is not None else f"200m-dk{retained['qk_head_dim']:03d}-softmax-seed0",
               "data_identity": copy.deepcopy(provenance["data_identity"]),
               "historical_dependency_lock_sha256": provenance["dependency_lock_sha256"],
               "historical_provenance": retained, "unavailable_historical_artifacts": unavailable,
               "provenance_sha256": {str(item): sha256(item) for item in paths}}
        validate_retained_run(run)
        result.append(run)
    return result


def validate_retained_run(run: dict) -> None:
    if not run.get("retained_completed"):
        raise ValueError("Expected a retained completed run")
    required = [run[key] for key in ("config_path", "metrics_path", "completion_path")]
    if any(path not in run["provenance_sha256"] for path in required):
        raise ValueError("Retained run is missing required provenance checksums")
    for path, expected in run["provenance_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Historical artifact changed: {path}")
    if yaml.safe_load(Path(run["config_path"]).read_text()) != run["config"]:
        raise ValueError("Historical resolved configuration changed")
    records = [json.loads(line) for line in Path(run["metrics_path"]).read_text().splitlines() if line.strip()]
    completion = json.loads(Path(run["completion_path"]).read_text())
    expected_step = run["config"]["optim"]["total_steps"]
    terminal = next((record for record in records if record["step"] == expected_step), None)
    if terminal is None or completion.get("final_step") != expected_step:
        raise ValueError("Historical run is not complete")
    for record in (*records, completion):
        if (record.get("source_commit") != run["source_commit"]
                or record.get("qk_head_dim") != run["qk_head_dim"]
                or record.get("attention_degree") != run["attention_degree"]
                or record.get("parameter_count") != run["parameter_count"]):
            raise ValueError("Historical metrics/completion identity changed")
    retained = run["historical_provenance"]
    if any(terminal[key] != value for key, value in retained["terminal"].items()):
        raise ValueError("Historical terminal metrics changed")
    if any(completion[key] != terminal[key] for key in ("training_tokens", "validation_nll", "validation_perplexity")):
        raise ValueError("Historical completion differs from terminal metrics")
    selected = retained["selected_checkpoint"]
    best = next((record for record in records if record["step"] == selected["step"]), None)
    if best is None or best["validation_nll"] != min(record["validation_nll"] for record in records):
        raise ValueError("Historical selected checkpoint is not validation-best")
    if any(best[key] != selected[key] for key in ("validation_nll", "validation_perplexity")):
        raise ValueError("Historical best-checkpoint metrics changed")


def historical_references(project_root: Path, key_dimensions=KEY_DIMENSIONS, degrees=DEGREES) -> tuple[list[dict], list[dict], dict]:
    reused = retained_runs(project_root, "polynomial_attention", [f"200m-m{degree:03d}-seed0" for degree in degrees]) if 64 in key_dimensions else []
    controls = retained_runs(project_root, "key_dimension", [f"200m-dk{width:03d}-seed0" for width in key_dimensions])
    alternative = retained_runs(project_root, "polynomial_attention", ["200m-softmax-seed0"])[0]
    return reused, controls, alternative


def resolve_config(base, root: Path, source: str, width: int, degree: int):
    config = copy.deepcopy(base)
    config.model.qk_head_dim, config.model.attention_degree = width, degree
    config.model.attention_backend = "fused"
    name = run_name(width, degree)
    config.train.seed, config.train.source_commit = 0, source
    config.train.workdir = root / "results" / EXPERIMENT / name
    config.train.wandb_project = "functional-memory-key-dimension-taylor"
    config.train.wandb_run_name = name
    validate_config(config)
    return config


def identity_payload(manifest: dict) -> dict:
    return {key: manifest[key] for key in ("source_commit", "dependency_lock_sha256", "data_identity", "scheduler",
            "key_dimensions", "degrees", "runs", "controls", "alternative_softmax64", "execution_policy")}


def make_manifest(base, project_root: Path, execution_root: Path, source_commit: str,
                  key_dimensions=KEY_DIMENSIONS, degrees=DEGREES, *, references=None) -> dict:
    root, execution = project_root.resolve(), execution_root.resolve()
    widths = selected_values(key_dimensions, KEY_DIMENSIONS, "Key dimensions")
    degrees = selected_values(degrees, DEGREES, "Degrees")
    reused, controls, alternative = historical_references(root, widths, degrees) if references is None else copy.deepcopy(references)
    data_identity = {key: {"path": str(Path(getattr(base.data, key)).resolve()), "sha256": sha256(Path(getattr(base.data, key)))}
                     for key in ("train_manifest", "eval_manifest")}
    baseline_recipe = scientific_recipe(config_to_dict(base), data_identity)
    for reference in [*reused, *controls, alternative]:
        validate_retained_run(reference)
        if scientific_recipe(reference["config"], reference["data_identity"]) != baseline_recipe:
            raise ValueError("Historical references do not match the scientific recipe and data checksums")
    if [(run["qk_head_dim"], run["attention_degree"]) for run in reused] != ([(64, degree) for degree in degrees] if 64 in widths else []):
        raise ValueError("Missing or unexpected reused Taylor grid coordinates")
    if [(run["qk_head_dim"], run["attention_degree"], run["original_experiment"]) for run in controls] != [(width, None, "key_dimension") for width in widths]:
        raise ValueError("Controls must be the matching historical key-dimension softmax sweep")
    runs = []
    for width in widths:
        for degree in degrees:
            if width == 64:
                runs.append(next(run for run in reused if run["attention_degree"] == degree))
                continue
            config = resolve_config(base, root, source_commit, width, degree)
            name = run_name(width, degree)
            runs.append({"name": name, "grid_name": name, "qk_head_dim": width, "attention_degree": degree,
                         "value_head_dim": config.model.value_head_dim, "parameter_count": parameter_count(config.model),
                         "source_commit": source_commit, "original_experiment": EXPERIMENT, "retained_completed": False,
                         "workdir": str(config.train.workdir), "checkpoint_dir": str(root / "experiments" / EXPERIMENT / "checkpoints" / name),
                         "config_path": str(root / "experiments" / EXPERIMENT / "configs" / f"{name}.yaml"),
                         "metrics_path": str(config.train.workdir / "metrics.jsonl"),
                         "completion_path": str(config.train.workdir / "completion.json"),
                         "training_tokens": config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1),
                         "config": config_to_dict(config)})
    manifest = {"format": "key-dimension-taylor-sweep", "schema_version": 1, "experiment": EXPERIMENT,
                "source_commit": source_commit, "project_root": str(root), "execution_root": str(execution),
                "dependency_lock_sha256": sha256(execution / "experiments/polynomial_attention/uv.lock"),
                "data_identity": data_identity, "scheduler": scheduler_profile(), "key_dimensions": list(widths),
                "degrees": list(degrees), "runs": runs, "controls": controls, "alternative_softmax64": alternative,
                "execution_policy": {"pilot_steps": PILOT_STEPS, "checkpoint_boundary": CHECKPOINT_BOUNDARY,
                                     "allocation_fraction": ALLOCATION_FRACTION}, "jobs": {}}
    manifest["identity"] = hashlib.sha256(json.dumps(identity_payload(manifest), sort_keys=True).encode()).hexdigest()
    manifest["preflight_dir"] = str(root / ".cache" / EXPERIMENT / "preflight" / manifest["identity"])
    return manifest


def prepare_sweep(manifest: dict) -> dict:
    path = manifest_path(Path(manifest["project_root"]))
    if path.exists():
        old = json.loads(path.read_text())
        if old["identity"] != manifest["identity"] or old["execution_root"] != manifest["execution_root"]:
            raise ValueError("Existing sweep has a different identity or frozen checkout")
        manifest = old
    for run in active_runs(manifest):
        config_path = Path(run["config_path"])
        if config_path.exists() and yaml.safe_load(config_path.read_text()) != run["config"]:
            raise ValueError("Refusing to overwrite a different resolved configuration")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(run["config"], sort_keys=False))
        workdir, checkpoints = Path(run["workdir"]), Path(run["checkpoint_dir"])
        workdir.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        link = workdir / "checkpoints"
        if link.is_symlink():
            if link.resolve() != checkpoints.resolve():
                raise ValueError("Checkpoint symlink points elsewhere")
        elif link.exists():
            raise ValueError("Canonical checkpoints require a symlink from the results directory")
        else:
            link.symlink_to(checkpoints, target_is_directory=True)
    write_json(path, manifest)
    return manifest


def scheduler_log_options(root: Path, name: str) -> list[str]:
    log_dir = root / "results/logs" / EXPERIMENT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    return [f"--output={log_dir}/slurm-%j.out", f"--error={log_dir}/slurm-%j.out", "--open-mode=append"]


def submit_preflight(manifest: dict, *, launch: bool = False) -> dict:
    manifest = prepare_sweep(manifest)
    if manifest["jobs"].get("preflight"):
        raise ValueError("Sweep already has a preflight job")
    root, execution = Path(manifest["project_root"]), Path(manifest["execution_root"])
    command = ["sbatch", "--parsable", *scheduler_options(manifest["scheduler"], preflight=True),
               *scheduler_log_options(root, "preflight"), "--job-name=dkt-preflight",
               str(execution / "experiments" / EXPERIMENT / "launch.sbatch"),
               "preflight-and-submit" if launch else "preflight", str(manifest_path(root))]
    job = submit_job(command, execution)
    manifest["jobs"]["preflight"] = job
    manifest["launch_after_preflight"] = launch
    manifest["preflight_submitted_utc"] = datetime.now(UTC).isoformat()
    manifest["preflight_log_path"] = str(root / "results/logs" / EXPERIMENT / "preflight" / f"slurm-{job}.out")
    write_json(manifest_path(root), manifest)
    return manifest


def submit_sweep(manifest: dict) -> dict:
    from experiments.key_dimension_taylor.preflight import eligible_runs, run_preflight
    manifest = prepare_sweep(manifest)
    run_preflight(manifest, check_only=True)
    eligible = eligible_runs(manifest)
    root, execution = Path(manifest["project_root"]), Path(manifest["execution_root"])
    previous, dependency_type = manifest["jobs"].get("preflight"), "afterok"
    for run in active_runs(manifest):
        if run["name"] not in eligible:
            continue
        pilot = json.loads((Path(manifest["preflight_dir"]) / "checks" / f"{run['name']}.json").read_text())
        segments = pilot["runtime"]["segments"]
        jobs = manifest["jobs"].setdefault(run["name"], [])
        for index, segment in enumerate(segments):
            if index < len(jobs):
                previous, dependency_type = jobs[index], "afterok"
                continue
            dependency = [f"--dependency={dependency_type}:{previous}"] if previous else []
            command = ["sbatch", "--parsable", *scheduler_options(manifest["scheduler"], preflight=False),
                       *scheduler_log_options(root, run["name"]), *dependency, "--kill-on-invalid-dep=yes",
                       f"--job-name=dkt-dk{run['qk_head_dim']}-m{run['attention_degree']}-s{index}",
                       str(execution / "experiments" / EXPERIMENT / "launch.sbatch"), "train",
                       str(manifest_path(root)), run["name"], str(index)]
            job = submit_job(command, execution, previous, dependency_type)
            jobs.append(job)
            launch_path = Path(run["workdir"]) / "launch.json"
            launches = json.loads(launch_path.read_text()) if launch_path.exists() else {"source_commit": manifest["source_commit"], "sweep_identity": manifest["identity"], "segments": []}
            launches["segments"].append({**segment, "index": index, "slurm_job_id": job,
                "dependency_job_id": previous, "dependency_type": dependency_type if previous else None,
                "log_path": str(root / "results/logs" / EXPERIMENT / run["name"] / f"slurm-{job}.out")})
            write_json(launch_path, launches)
            write_json(manifest_path(root), manifest)
            previous, dependency_type = job, "afterok"
        dependency_type = "afterany"
    manifest["production_submitted_utc"] = datetime.now(UTC).isoformat()
    write_json(manifest_path(root), manifest)
    from experiments.key_dimension_taylor.plot import aggregate
    aggregate(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument("--key-dimensions", nargs="+", type=int, default=KEY_DIMENSIONS)
    parser.add_argument("--degrees", nargs="+", type=int, default=DEGREES)
    group = parser.add_mutually_exclusive_group()
    for name in ("dry-run", "preflight-only", "launch", "submit"):
        group.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.submit:
        manifest = json.loads(manifest_path(args.project_root.resolve()).read_text())
        print(json.dumps(submit_sweep(manifest), indent=2, sort_keys=True))
        return
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.execution_root, text=True).strip()
    if args.preflight_only or args.launch:
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=args.execution_root, text=True):
            raise ValueError("Execution checkout must be clean before submission")
        if args.project_root.resolve() == args.execution_root.resolve():
            raise ValueError("Use a separate frozen execution checkout")
    references = historical_references(args.project_root.resolve(), selected_values(args.key_dimensions, KEY_DIMENSIONS, "Key dimensions"), selected_values(args.degrees, DEGREES, "Degrees"))
    # Read a retained polynomial configuration; never materialize old softmax paths.
    base_path = args.project_root / "experiments/polynomial_attention/configs/runs/200m-m002-seed0.yaml"
    manifest = make_manifest(load_config(base_path), args.project_root, args.execution_root, source,
                             args.key_dimensions, args.degrees, references=references)
    if args.preflight_only or args.launch:
        manifest = submit_preflight(manifest, launch=args.launch)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
