"""Allocate and submit fixed-input, coefficient-only fits of a pretrained Transformer."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import create_run, load_run, regenerate_index, update_record, validate_record
from scaffolds.records.selections import refresh_selection

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/attention_scaling"
TOPIC = ROOT / "results/exploratory/coefficient_oracle"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parent):
    parent = Path(parent).resolve()
    record = load_run(parent)
    validate_record(parent, checkpoint_hashes=False)
    if record["home_cluster"] != "kennel" or record["status"] != "complete":
        raise ValueError("The parent must be a completed Kennel run")
    source = yaml.safe_load((parent / "config.yaml").read_text())
    model = source["model"]
    if (model["attention_degree"] != 8 or model["attention_coefficient_conditioning"] != "query"
            or model["initialization"] != "transformer_reference"):
        raise ValueError("Expected the reference-initialized degree-eight query-only parent")
    best = json.loads((parent / "best_validation.json").read_text())
    checkpoint = parent / "checkpoints" / str(best["step"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if record["selected_checkpoint"]["step"] != best["step"]:
        raise ValueError("Parent selected checkpoint and validation best disagree")
    if TOPIC.joinpath("selection.yaml").exists():
        raise ValueError("This worktree already has an allocated coefficient-oracle selection")
    runs = []
    for count in (1, source["data"]["batch_size"]):
        recipe = deepcopy(source)
        # This historical, irrelevant GDN field is absent from the current dot-product schema.
        if recipe["model"].get("gdn_polynomial_degree") == 1:
            recipe["model"].pop("gdn_polynomial_degree")
        label = f"200m-m008-query-coefficient-oracle-sequences{count}-seed0"
        recipe.update(schema_version=1, experiment_kind="fixed_input_coefficient_oracle",
                      condition=f"coefficient_oracle/sequences_{count}")
        recipe["data"]["batch_size"] = count
        recipe["optim"].update(learning_rate=0.01, min_learning_rate=0.001,
                               weight_decay=0.0, warmup_steps=0, total_steps=10000)
        recipe["train"].update(source_commit="auto", micro_batch_size=min(count, 8),
                               log_every=1, eval_every=1, eval_batches=1, ckpt_every=100,
                               wandb_run_name=label, wandb_group="attention_scaling", wandb_dir=None)
        recipe["oracle"] = {
            "parent_run_id": parent.name, "parent_config": str(parent / "config.yaml"),
            "parent_config_sha256": sha256(parent / "config.yaml"),
            "parent_checkpoint": str(checkpoint), "parent_checkpoint_step": best["step"],
            "parent_checkpoint_sha256": record["selected_checkpoint"]["sha256"],
            "eval_manifest_sha256": sha256(source["data"]["eval_manifest"]),
            "sequence_count": count, "microbatch_size": min(count, 8), "data_start": 0,
            "data_split": "eval", "data_role": "fitting inputs; no generalization evaluation",
            "parameterization": "free_per_example_query_coefficients",
            "coefficient_storage": "effective monomial coefficients; exact captured initialization",
            "optimizer_coordinates": "factorial_scaled; gradients and updates divided by r!",
            "frozen_parameters": "all pretrained model parameters",
            "initialization": "capture actual coefficients on the original forward pass",
            "max_updates": 10000, "initial_updates": 2000, "plateau_window": 200,
            "plateau_tolerance": 0.001, "confirmation_updates": 500,
            "learning_rate": 0.01, "learning_rate_decay": 0.1,
            "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8, "clip_norm": 1.0,
            "gradient_scaling": "per-sequence mean NLL, clipped independently per sequence",
            "checkpoint_every": 100,
            "checkpoint_selection": "lowest aggregate fitting NLL; no independent validation",
        }
        run = create_run(STUDY, recipe, label=label)
        update_record(run, {
            "home_cluster": "kennel", "seed": 0, "parent_run_id": parent.name,
            "coefficient_sharing": "per_example_layer_head_query",
            "coefficient_parameterization": "free_per_example_query_coefficients",
            "checkpoint_retention": "best fitting coefficients, required to reproduce the fitted oracle",
            "source_checkpoint": {**record["selected_checkpoint"], "path": str(checkpoint)},
        })
        runs.append(run)
    TOPIC.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": 1, "opened": "2026-10-23",
        "question": "How much can free query polynomial coefficients improve a fixed pretrained model on fixed inputs?",
        "settings": {"metric": "fit_nll", "parent_run_id": parent.name,
                     "same_first_sequence": True, "generalization_claim": False},
        "runs": [{"study": "attention_scaling", "run_id": run.name,
                  "role": "single_sequence" if index == 0 else "full_batch", "sources": {}}
                 for index, run in enumerate(runs)],
    }
    (TOPIC / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(TOPIC)
    regenerate_index(STUDY)
    print(" ".join(run.name for run in runs), flush=True)


def submit(run_ids, *, partition, qualify_only=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = changes(ROOT, commit, "attention_scaling", "transformer")
    if dirty:
        raise ValueError(f"Commit the frozen inputs before submitting: {dirty}")
    for run_id in run_ids:
        run = STUDY / "runs" / run_id
        record = load_run(run)
        validate_record(run, checkpoint_hashes=False)
        if record["status"] == "complete":
            raise ValueError(f"Run {run_id} is already complete")
        previous = record.get("submission", {}).get("slurm_job_id")
        if previous:
            live = subprocess.run(["squeue", "-h", "-j", str(previous), "-o", "%T"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            if live:
                raise ValueError(f"Run {run_id} already has live job {previous}: {live}")
        log = run / "training.log"
        command = ["sbatch", "--parsable", "--nice=0", f"--partition={partition}",
                   "--gres=gpu:1", "--exclude=collie,samoyed,doodle,terrier,mastiff",
                   "--cpus-per-task=8", "--mem=96G",
                   "--time=00:30:00" if qualify_only else "--time=12:00:00",
                   f"--job-name=coef-oracle-{run_id}", f"--output={log}", f"--error={log}",
                   str(STUDY / "coefficient_oracle.sbatch"), str(ROOT), commit, run_id]
        if qualify_only:
            command.append("--qualify-only")
        job = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {job}")
        update_record(run, {
            "status": "submitted", "source_commit": commit,
            "frozen_execution_sha256": frozen_hash(ROOT, commit, "attention_scaling", "transformer"),
            "submission": {"slurm_job_id": job, "partition": partition, "command": command,
                           "qualify_only": qualify_only, "submitted_utc": datetime.now(UTC).isoformat()},
            "log_path": str(run / "training.log"),
        })
        print(f"{run_id} job_id={job}", flush=True)
    regenerate_index(STUDY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--parent", type=Path, required=True)
    launch = sub.add_parser("submit")
    launch.add_argument("run_ids", nargs="+")
    launch.add_argument("--partition", default="standard,long")
    launch.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent)
    else:
        submit(args.run_ids, partition=args.partition, qualify_only=args.qualify_only)


if __name__ == "__main__":
    main()
