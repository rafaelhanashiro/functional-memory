"""Prepare a random/pretrained comparison workspace and submit its serial one-GPU conditions.

Run from the experiment worktree root. The workspace mirrors the record's
`experiments/state_rank/<comparison>/` layout; `record retain` maps it into runs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scaffolds.state_rank.compare_random import prepare_manifest, source_commit, utc_now, write_json

RECIPES = {"llama": "experiments/state_rank/configs/random_vs_pretrained.json",
           "gated-deltanet": "experiments/state_rank/configs/random_vs_pretrained_deltanet.json"}


def submit(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    execution = Path(manifest["execution_root"])
    # Only the frozen execution set must match the launch commit.
    subprocess.run(["python3", "scaffolds/launch/frozen.py", "check", "--launch-commit", manifest["source_commit"],
                    "--experiment", "state_rank", "--scaffold", "state_rank"], cwd=execution, check=True)
    if manifest["jobs"]:
        raise ValueError("This comparison already has submitted jobs; inspect them before resubmitting")
    experiment = manifest.get("experiment", "state_rank/random_vs_pretrained")
    if experiment not in ("state_rank/random_vs_pretrained", "state_rank/random_vs_pretrained_deltanet"):
        raise ValueError(f"Unsupported comparison experiment: {experiment}")
    comparison = experiment.split("/", 1)[1]
    deltanet = comparison.endswith("_deltanet")
    previous = None
    # The first two conditions double as full-context GPU preflights. Successful
    # afterok dependencies gate every later condition on both having passed.
    for condition in manifest["conditions"]:
        name = condition["name"]
        log_directory = execution / "experiments/state_rank/logs" / comparison / name
        log_directory.mkdir(parents=True, exist_ok=True)
        command = ["sbatch", "--parsable", f"--job-name=rank-{name}",
                   f"--output={log_directory}/slurm-%j.out", f"--error={log_directory}/slurm-%j.out"]
        if previous:
            command += [f"--dependency=afterok:{previous}"]
        if deltanet:
            command += ["--time=02:00:00", "--cpus-per-task=8", "--mem=48G"]
        command += [str(execution / "experiments/state_rank/adapter/execution/launch_comparison.sbatch"),
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
        print(f"{name}: {job_id}", flush=True)
        previous = job_id
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(RECIPES), default="llama")
    parser.add_argument("--config", type=Path, help="Defaults to the experiment's recipe for --model")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Checkout holding the pretrained collections' archives under experiments/state_rank/runs/")
    parser.add_argument("--output", type=Path, required=True,
                        help="Workspace comparison directory, e.g. .cache/state_rank/<rerun>/experiments/state_rank/random_vs_pretrained")
    parser.add_argument("--submit", action="store_true", help="Submit six serial Slurm jobs; first two gate the remaining four")
    args = parser.parse_args()
    prepare = prepare_manifest
    if args.model == "gated-deltanet":
        from scaffolds.state_rank import compare_deltanet
        prepare = compare_deltanet.prepare_manifest
    manifest = prepare(args.config or Path(RECIPES[args.model]), args.project_root, args.output)
    print(f"manifest={manifest}", flush=True)
    if args.submit:
        submit(manifest)


if __name__ == "__main__":
    main()
