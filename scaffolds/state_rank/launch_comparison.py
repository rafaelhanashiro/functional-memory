"""Freeze provenance and submit serial one-GPU random/pretrained rank conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scaffolds.state_rank.compare_random import DEFAULT_CONFIG, prepare_manifest, source_commit, utc_now, write_json


def submit(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    execution = Path(manifest["execution_root"])
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=execution, text=True)
    if dirty or source_commit() != manifest["source_commit"]:
        raise ValueError("Commit the tested scaffold before submitting from its frozen checkout")
    if manifest["jobs"]:
        raise ValueError("This comparison already has submitted jobs; inspect them before resubmitting")
    experiment = manifest.get("experiment", "state_rank/random_vs_pretrained")
    if experiment not in ("state_rank/random_vs_pretrained", "state_rank/random_vs_pretrained_deltanet"):
        raise ValueError(f"Unsupported comparison experiment: {experiment}")
    deltanet = experiment.endswith("_deltanet")
    previous = None
    # The first two conditions double as full-context GPU preflights. Successful
    # afterok dependencies gate every later condition on both having passed.
    for condition in manifest["conditions"]:
        name = condition["name"]
        log_directory = Path(manifest["project_root"]) / "results/logs" / experiment / name
        log_directory.mkdir(parents=True, exist_ok=True)
        command = ["sbatch", "--parsable", f"--job-name=rank-{name}",
                   f"--output={log_directory}/slurm-%j.out", f"--error={log_directory}/slurm-%j.out"]
        if previous:
            command += [f"--dependency=afterok:{previous}"]
        if deltanet:
            command += ["--time=02:00:00", "--cpus-per-task=8", "--mem=48G"]
        command += [str(execution / "scaffolds/state_rank/launch_comparison.sbatch"),
                    str(execution), manifest["project_root"], str(manifest_path), name, manifest["source_commit"]]
        if deltanet:
            command += ["compare_deltanet"]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {result.stdout!r}")
        manifest["jobs"][name] = job_id
        manifest["submitted_utc"] = utc_now()
        write_json(manifest_path, manifest)
        print(f"{name}: {job_id}", flush=True)
        previous = job_id
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama", "gated-deltanet"), default="llama")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true", help="Submit six serial Slurm jobs; first two gate the remaining four")
    args = parser.parse_args()
    prepare, default_config = prepare_manifest, DEFAULT_CONFIG
    if args.model == "gated-deltanet":
        from scaffolds.state_rank import compare_deltanet
        prepare, default_config = compare_deltanet.prepare_manifest, compare_deltanet.DEFAULT_CONFIG
    manifest = prepare(args.config or default_config, args.project_root, args.output)
    print(f"manifest={manifest}", flush=True)
    if args.submit:
        submit(manifest)


if __name__ == "__main__":
    main()
