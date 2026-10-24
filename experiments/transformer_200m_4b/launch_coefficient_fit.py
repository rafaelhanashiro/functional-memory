"""Allocate and submit a fixed-sequence fit of a pretrained query coefficient generator."""
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
STUDY = ROOT / "experiments/transformer_200m_4b"
TOPIC = STUDY / "results/exploratory/coefficient_oracle"
ROLE = "query_generator_single_sequence"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parent: Path):
    parent = parent.resolve()
    record = load_run(parent)
    validate_record(parent, checkpoint_hashes=False)
    if record["status"] != "complete" or record["home_cluster"] != "kennel":
        raise ValueError("The parent must be a completed Kennel run")
    source = yaml.safe_load((parent / "config.yaml").read_text())
    model = source["model"]
    if (model["attention_degree"] != 8 or model["attention_coefficient_conditioning"] != "query"
            or model["attention_coefficients"] != "learned" or model["attention_normalization"]
            or model["attention_coefficient_activation"] != "tanh"
            or model["attention_coefficient_scale"] != "factorial"
            or model["attention_coefficient_amplitude_scale"] != "factorial"
            or model["initialization"] != "transformer_truncated_reference"
            or not model["qk_norm"] or model["attention_backend"] != "mosaic"):
        raise ValueError("Expected the 4B degree-eight factorial-scaled query-only parent")
    best = json.loads((parent / "best_validation.json").read_text())
    descriptor = record["selected_checkpoint"]
    checkpoint = Path(descriptor["path"])
    if not checkpoint.is_absolute():
        checkpoint = parent.parents[3] / checkpoint
    if not checkpoint.is_dir() or descriptor["step"] != best["step"]:
        raise ValueError("Parent retained checkpoint and validation best disagree")
    selection_path = TOPIC / "selection.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    if any(entry["role"] == ROLE for entry in selection["runs"]):
        raise ValueError("A query-generator fit is already allocated in this topic")
    reference_run = STUDY / "runs/r3c0f21ab7dce"
    sequence_identity = json.loads((reference_run / "data_identity.json").read_text())
    if sequence_identity["manifest_sha256"] != sha256(source["data"]["eval_manifest"]):
        raise ValueError("The fitting sequence's manifest changed")
    recipe = deepcopy(source)
    recipe["model"].pop("gdn_polynomial_degree", None)
    label = "query-generator-m8-single-sequence"
    recipe.update(schema_version=1, experiment_kind="fixed_input_coefficient_fit",
                  condition="coefficient_oracle/query_generator_single_sequence")
    recipe["data"]["batch_size"] = 1
    recipe["optim"].update(learning_rate=0.001, min_learning_rate=0.0001,
                           weight_decay=0.0, warmup_steps=0, total_steps=10000)
    recipe["train"].update(source_commit="auto", micro_batch_size=1,
                           log_every=1, eval_every=1, eval_batches=1, ckpt_every=100,
                           wandb_run_name=label, wandb_group=STUDY.name, wandb_dir=None)
    recipe["oracle"] = {
        "parent_run_id": parent.name, "parent_study": parent.parent.parent.name,
        "parent_config": str(parent / "config.yaml"),
        "parent_config_sha256": sha256(parent / "config.yaml"),
        "parent_checkpoint": str(checkpoint), "parent_checkpoint_step": best["step"],
        "parent_checkpoint_sha256": descriptor["sha256"],
        "eval_manifest_sha256": sequence_identity["manifest_sha256"],
        "batch_sha256": sequence_identity["batch_sha256"],
        "sequence_count": 1, "microbatch_size": 1, "data_start": 0,
        "data_split": "eval", "data_role": "fitting inputs; no generalization evaluation",
        "parameterization": "query_coefficient_generator",
        "trainable_parameters": ["query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel"],
        "coefficient_storage": "native trained generator parameters",
        "optimizer_coordinates": "native stored generator parameters; factorial scaling remains in the model",
        "frozen_parameters": "all backbone parameters outside the query coefficient generator",
        "initialization": "restore trained query coefficient generator without resetting parameters",
        "max_updates": 10000, "initial_updates": 2000, "plateau_window": 200,
        "plateau_tolerance": 0.001, "confirmation_updates": 500,
        "learning_rate": 0.001, "learning_rate_decay": 0.1,
        "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8, "clip_norm": 1.0,
        "gradient_scaling": "mean next-token NLL on the one fixed sequence",
        "checkpoint_every": 100,
        "checkpoint_selection": "lowest fitting NLL; no independent validation",
    }
    run = create_run(STUDY, recipe, label=label)
    update_record(run, {
        "home_cluster": "kennel", "seed": source["train"]["seed"],
        "parent_run_id": parent.name, "parent_study": parent.parent.parent.name,
        "coefficient_sharing": "shared query-conditioned generator per layer and head",
        "coefficient_parameterization": "query_coefficient_generator",
        "checkpoint_retention": "best fitting generator parameters; backbone referenced by source checkpoint",
        "source_checkpoint": {**descriptor, "path": str(checkpoint)},
    })
    previous_parent = selection["settings"].pop("parent_run_id", None)
    for entry in selection["runs"]:
        if previous_parent:
            entry.setdefault("parent_run_id", previous_parent)
    selection["question"] = "How much can polynomial coefficients fit a fixed sequence with a frozen 4B backbone?"
    selection["settings"]["comparison"] = "Different source backbones; compare fitting capacity, not a matched-backbone parameterization ablation"
    selection["runs"].append({"study": STUDY.name, "run_id": run.name, "role": ROLE,
                              "parent_run_id": parent.name, "sources": {}})
    selection_path.write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(TOPIC)
    regenerate_index(STUDY)
    print(run.name, flush=True)


def submit(run_id, *, partition="debug,standard,long", time_limit="02:00:00"):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = changes(ROOT, commit, STUDY.name, "transformer")
    if dirty:
        raise ValueError(f"Commit the frozen inputs before submitting: {dirty}")
    run = STUDY / "runs" / run_id
    record = load_run(run)
    validate_record(run, checkpoint_hashes=False)
    if record["status"] == "complete":
        raise ValueError("The run is already complete")
    previous = record.get("submission", {}).get("slurm_job_id")
    if previous:
        live = subprocess.check_output(["squeue", "-h", "-j", str(previous), "-o", "%T"], text=True).strip()
        if live:
            raise ValueError(f"Run already has live job {previous}: {live}")
    log = run / "training.log"
    command = ["sbatch", "--parsable", "--nice=0", f"--partition={partition}",
               "--gres=gpu:1", "--exclude=collie,samoyed,doodle,terrier,mastiff",
               "--cpus-per-task=8", "--mem=96G", f"--time={time_limit}",
               f"--job-name=query-fit-{run_id}", f"--output={log}", f"--error={log}",
               str(STUDY / "coefficient_fit.sbatch"), str(ROOT), commit, run_id]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {job}")
    update_record(run, {"status": "submitted", "source_commit": commit,
        "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
        "submission": {"slurm_job_id": job, "partition": partition, "command": command,
                       "submitted_utc": datetime.now(UTC).isoformat()}, "log_path": str(log)})
    regenerate_index(STUDY)
    print(f"{run_id} job_id={job}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--parent", type=Path, required=True)
    launch = sub.add_parser("submit")
    launch.add_argument("run_id")
    launch.add_argument("--partition", default="debug,standard,long")
    launch.add_argument("--time-limit", default="02:00:00")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent)
    else:
        submit(args.run_id, partition=args.partition, time_limit=args.time_limit)


if __name__ == "__main__":
    main()
