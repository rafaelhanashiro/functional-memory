"""Allocate canonical comparison runs, commit their recipes, then submit them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml

from scaffolds.records import create_run, load_run, update_record
from scaffolds.launch.frozen import frozen_hash
from scaffolds.state_rank.compare_random import (
    condition_directory, manifest_identity, prepare_manifest, sha256_file, source_commit,
    utc_now, validate_manifest, write_json,
)

RECIPES = {"llama": "random_vs_pretrained", "gated-deltanet": "random_vs_pretrained_deltanet"}
EXECUTION_ROOT = Path(__file__).resolve().parents[2]


def prepare_runs(model, project_root, study, *, config=None):
    """Validate data first, then allocate one immutable recipe per condition."""
    study = Path(study).resolve()
    study_metadata = yaml.safe_load((study / "sweep.yaml").read_text())
    recipe = config or study_metadata["recipes"][RECIPES[model]]
    prepare = prepare_manifest
    if model == "gated-deltanet":
        from scaffolds.state_rank import compare_deltanet
        prepare = compare_deltanet.prepare_manifest
    with tempfile.TemporaryDirectory(prefix="state-rank-prepare-") as temporary:
        prepared = prepare(recipe, project_root, Path(temporary) / "comparison")
        manifest = json.loads(prepared.read_text())
        directories = {}
        for condition in manifest["conditions"]:
            resolved = {**recipe, "condition": condition, "weights": condition["weights"], "seed": condition["seed"]}
            resolved.setdefault("output_directory", "allocated-by-create-run")
            run = create_run(study, resolved, label=f"{RECIPES[model]}/{condition['name']}")
            directories[condition["name"]] = str(run)
        workspace = Path(next(iter(directories.values()))) / ".runtime/comparison"
        shutil.copytree(prepared.parent, workspace)
    manifest.update(
        execution_root=str(EXECUTION_ROOT), output_directory=str(workspace),
        run_directories=directories,
        run_config_sha256={name: sha256_file(Path(path) / "config.yaml") for name, path in directories.items()},
    )
    manifest["identity"] = manifest_identity(manifest)
    path = workspace / "manifest.json"
    write_json(path, manifest)
    for directory in directories.values():
        update_record(directory, {"comparison": RECIPES[model], "status": "planned",
                                  "home_cluster": study_metadata.get("home_cluster"),
                                  "inputs": {"path": str(Path(manifest["baseline_directory"]) / "inputs.json"),
                                             "sha256": manifest["inputs_sha256"]},
                                  "execution": {"comparison_manifest": str(path)},
                                  "checkpoint_retention": "none"})
    return path


def submit(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    execution = Path(manifest["execution_root"])
    if manifest["jobs"]:
        raise ValueError("This comparison already has submitted jobs; inspect them before resubmitting")
    experiment = manifest.get("experiment", "state_rank/random_vs_pretrained")
    if experiment not in ("state_rank/random_vs_pretrained", "state_rank/random_vs_pretrained_deltanet"):
        raise ValueError(f"Unsupported comparison experiment: {experiment}")
    comparison = experiment.split("/", 1)[1]
    deltanet = comparison.endswith("_deltanet")
    if "run_directories" not in manifest:
        raise ValueError("Prepare canonical per-condition runs before submission")
    # Recipes are prepared first and committed by the caller. Bind execution to
    # that new commit without rewriting any recipe or creating a source-hash cycle.
    candidate = {**manifest, "source_commit": source_commit()}
    candidate["identity"] = manifest_identity(candidate)
    validate_manifest(candidate)
    subprocess.run(["python3", "scaffolds/launch/frozen.py", "check", "--launch-commit", candidate["source_commit"],
                    "--experiment", "state_rank", "--scaffold", "state_rank"], cwd=execution, check=True)
    for name in candidate["run_directories"]:
        recipe_path = condition_directory(candidate, name) / "config.yaml"
        relative = recipe_path.relative_to(execution)
        subprocess.run(["git", "cat-file", "-e", f"{candidate['source_commit']}:{relative}"], cwd=execution, check=True)
    execution_hash = frozen_hash(execution, candidate["source_commit"], "state_rank", "state_rank")
    manifest = candidate
    write_json(manifest_path, manifest)
    previous = None
    # The first two conditions double as full-context GPU preflights. Successful
    # afterok dependencies gate every later condition on both having passed.
    for condition in manifest["conditions"]:
        name = condition["name"]
        run = condition_directory(manifest, name)
        log = run / "training.log"
        command = ["sbatch", "--parsable", f"--job-name=rank-{name}",
                   f"--output={log}", f"--error={log}"]
        if previous:
            command += [f"--dependency=afterok:{previous}"]
        if deltanet:
            command += ["--time=02:00:00", "--cpus-per-task=8", "--mem=48G"]
        command += [str(execution / "experiments/state_rank/launch_comparison.sbatch"),
                    str(execution), manifest["project_root"], str(manifest_path), name, manifest["source_commit"]]
        if deltanet:
            command += ["compare_deltanet"]
        result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=execution)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {result.stdout!r}")
        manifest["jobs"][name] = job_id
        manifest["submitted_utc"] = utc_now()
        write_json(manifest_path, manifest)
        record = load_run(run)
        update_record(run, {
            "status": "submitted", "source_commit": manifest["source_commit"],
            "frozen_execution_sha256": execution_hash,
            "log_path": str(log),
            "attempts": [*record.get("attempts", []), {"slurm_job_id": job_id,
                         "submitted_utc": manifest["submitted_utc"], "source_commit": manifest["source_commit"]}],
        })
        print(f"{name}: {job_id}", flush=True)
        previous = job_id
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(RECIPES), default="llama")
    parser.add_argument("--config", type=Path, help="Optional complete recipe override; defaults to sweep.yaml recipes")
    parser.add_argument("--study", type=Path, default=Path("experiments/state_rank"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Checkout holding the pretrained collections' archives under experiments/state_rank/runs/")
    parser.add_argument("--manifest", type=Path, help="Prepared comparison manifest; submit after committing the new run recipes")
    parser.add_argument("--submit", action="store_true", help="Submit six serial Slurm jobs; first two gate the remaining four")
    args = parser.parse_args()
    if args.manifest is not None:
        if not args.submit:
            parser.error("--manifest requires --submit")
        submit(args.manifest)
        return
    config = yaml.safe_load(args.config.read_text()) if args.config else None
    manifest = prepare_runs(args.model, args.project_root, args.study, config=config)
    print(f"manifest={manifest}", flush=True)
    if args.submit:
        submit(manifest)
    else:
        print("Commit sweep.yaml and the new run config.yaml files, then submit with --manifest <path> --submit.", flush=True)


if __name__ == "__main__":
    main()
