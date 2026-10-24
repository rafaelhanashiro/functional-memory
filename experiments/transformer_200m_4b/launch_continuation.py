"""Allocate and submit matched fixed/learned continuations of one Taylor checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import yaml

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import create_run, load_run, regenerate_index, update_record, validate_record
from scaffolds.records.selections import refresh_selection

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/transformer_200m_4b"
UPDATES = 500
LEARNING_RATE = 0.0001
DEFAULT_TIME_LIMIT = "06:00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topic_path(conditioning: str) -> Path:
    if conditioning not in ("static", "query"):
        raise ValueError("Continuation conditioning must be static or query")
    return STUDY / f"results/exploratory/taylor_{conditioning}_continuation"


def recipes(source: dict, continuation: dict, *, conditioning: str = "static") -> dict[str, dict]:
    """Match additional training while enabling the requested coefficient family."""
    topic_path(conditioning)
    result = {}
    degree = source["model"]["attention_degree"]
    for role in ("fixed_taylor", f"learned_{conditioning}"):
        recipe = deepcopy(source)
        recipe["optim"].update(learning_rate=LEARNING_RATE, min_learning_rate=LEARNING_RATE,
                               warmup_steps=0, total_steps=continuation["parent_step"] + UPDATES)
        recipe["train"].update(source_commit="auto", micro_batch_size=8, log_every=10, eval_every=100,
                               eval_early_every=None, eval_early_until=0, ckpt_every=100,
                               wandb_run_name=f"taylor-m{degree}-continuation-{role}",
                               wandb_group=STUDY.name, wandb_dir=None, wandb_mode="online")
        recipe["continuation"] = deepcopy(continuation)
        if role != "fixed_taylor":
            recipe["model"].update(attention_coefficients="learned",
                                   attention_coefficient_sharing="per_head",
                                   attention_coefficient_scale="factorial")
            if conditioning == "query":
                recipe["model"].update(
                    attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh",
                    attention_coefficient_amplitude_initial=0.0,
                    attention_coefficient_projection_initial_std=1 / math.sqrt(source["model"]["hidden_size"]),
                    attention_coefficient_tanh_bias=False,
                    attention_coefficient_amplitude_scale="factorial",
                )
        result[role] = recipe
    return result


def prepare(parent: Path, *, conditioning: str = "static") -> dict[str, str]:
    parent = parent.resolve()
    topic = topic_path(conditioning)
    if (topic / "selection.yaml").exists():
        raise ValueError("This continuation pair is already allocated; reuse its permanent run IDs")
    record = load_run(parent)
    validate_record(parent, checkpoint_hashes=True)
    if record["status"] != "complete" or record["home_cluster"] != "kennel":
        raise ValueError("The source must be a completed Kennel run")
    source = yaml.safe_load((parent / "config.yaml").read_text())
    model = source["model"]
    expected = {"attention_coefficients": "fixed",
                "attention_coefficient_conditioning": "static", "attention_normalization": False,
                "attention_output_scaling": "head_rms_norm", "qk_norm": True,
                "initialization": "transformer_truncated_reference", "attention_backend": "mosaic"}
    if (not isinstance(model.get("attention_degree"), int) or model["attention_degree"] < 1
            or any(model.get(key) != value for key, value in expected.items())):
        raise ValueError("Expected a 4B fixed Taylor parent with QK/head RMSNorm")
    if source["optim"]["total_steps"] != 7570 or source["data"]["batch_size"] != 256:
        raise ValueError("Expected the 4B recipe and its 256-sequence scientific batch")
    descriptor = record["selected_checkpoint"]
    checkpoint = Path(descriptor["path"])
    if not checkpoint.is_absolute():
        checkpoint = parent.parents[3] / checkpoint
    best = json.loads((parent / "best_validation.json").read_text())
    if not checkpoint.is_dir() or descriptor["step"] != best["step"] or best["step"] != 7500:
        raise ValueError("Expected the validation-best Taylor checkpoint at update 7500")
    continuation = {
        "parent_config": str(parent / "config.yaml"),
        "parent_config_sha256": sha256(parent / "config.yaml"),
        "parent_data_identity_sha256": sha256(parent / "data_identity.json"),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": descriptor["sha256"],
        "parent_step": int(best["step"]),
    }
    run_ids = {}
    for role, recipe in recipes(source, continuation, conditioning=conditioning).items():
        run = create_run(STUDY, recipe, label=f"taylor-m{model['attention_degree']}-continuation-{role}")
        run_ids[role] = run.name
        update_record(run, {
            "home_cluster": "kennel", "seed": source["train"]["seed"],
            "experiment_kind": "matched_checkpoint_continuation", "condition": role,
            "parent_run_id": parent.name, "parent_study": parent.parent.parent.name,
            "source_checkpoint": {**descriptor, "path": str(checkpoint)},
            "source_data_identity": {"path": str(parent / "data_identity.json"),
                                     "sha256": continuation["parent_data_identity_sha256"]},
            "source_configuration": {"path": continuation["parent_config"],
                                     "sha256": continuation["parent_config_sha256"]},
            "continuation": {"start_step": best["step"], "updates": UPDATES,
                             "new_prediction_tokens": UPDATES * source["data"]["batch_size"]
                                                      * (source["data"]["sequence_length"] - 1),
                             "backbone_optimizer": "preserve pretrained moments and step",
                             "new_coefficient_optimizer": "zero moments and fresh Adam clock",
                             "data_start_sequence": best["step"] * source["data"]["batch_size"]},
        })
    selection = {
        "schema_version": 1, "opened": datetime.now(UTC).date().isoformat(),
        "question": f"Does enabling {conditioning} coefficients after Taylor pretraining improve matched continuation?",
        "settings": {"metric": "validation_perplexity", "parent_run_id": parent.name,
                     "parent_step": best["step"], "updates": UPDATES,
                     "evaluation_every": 100, "learning_rate": LEARNING_RATE,
                     "primary_comparison": "terminal held-out NLL/PPL at the same continuation update",
                     "validation_best": "reported separately from terminal"},
        "runs": [{"study": STUDY.name, "run_id": run_id, "role": role, "sources": {}}
                 for role, run_id in run_ids.items()],
    }
    topic.mkdir(parents=True, exist_ok=True)
    (topic / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(topic)
    regenerate_index(STUDY)
    print(json.dumps(run_ids, sort_keys=True), flush=True)
    return run_ids


def submission_command(run_id: str, commit: str, *, partition: str, time_limit: str) -> list[str]:
    run = STUDY / "runs" / run_id
    log = run / "training.log"
    return ["sbatch", "--parsable", f"--partition={partition}", "--gres=gpu:1",
            "--cpus-per-task=16", "--mem=128G", f"--time={time_limit}",
            f"--job-name=taylor-continue-{run_id}", f"--output={log}", f"--error={log}",
            "scaffolds/transformer/launchers/kennel.sbatch",
            "experiments.transformer_200m_4b.acceptance",
            str((run / "config.yaml").relative_to(ROOT)), commit, "1", "B300,H100"]


def live_job_state(job_id: str) -> str:
    """An expired Slurm job ID is absent; other scheduler failures remain errors."""
    result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                            text=True, capture_output=True)
    if result.returncode:
        if "invalid job id" in result.stderr.lower():
            return ""
        raise RuntimeError(f"Cannot check existing Slurm job {job_id}: {result.stderr.strip()}")
    return result.stdout.strip()


def submit(run_id: str, *, partition: str = "standard,long", time_limit: str = DEFAULT_TIME_LIMIT,
           conditioning: str = "static") -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = changes(ROOT, commit, STUDY.name, "transformer")
    if dirty:
        raise ValueError(f"Commit the frozen inputs before submitting: {dirty}")
    selection = yaml.safe_load((topic_path(conditioning) / "selection.yaml").read_text())
    if run_id not in {entry["run_id"] for entry in selection["runs"]}:
        raise ValueError("Run is not a member of the explicitly selected continuation pair")
    run = STUDY / "runs" / run_id
    record = load_run(run)
    validate_record(run, checkpoint_hashes=False)
    if record["status"] == "complete":
        raise ValueError("The run is already complete")
    previous = record.get("submission", {}).get("slurm_job_id")
    if previous:
        live = live_job_state(str(previous))
        if live:
            raise ValueError(f"Run already has live job {previous}: {live}")
    command = submission_command(run_id, commit, partition=partition, time_limit=time_limit)
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {job}")
    update_record(run, {
        "status": "submitted", "source_commit": commit,
        "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
        "submission": {"slurm_job_id": job, "partition": partition, "command": command,
                       "submitted_utc": datetime.now(UTC).isoformat()},
        "log_path": str(run / "training.log"),
    })
    regenerate_index(STUDY)
    print(f"{run_id} job_id={job}", flush=True)
    return job


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--parent", type=Path, required=True)
    prep.add_argument("--conditioning", choices=("static", "query"), default="static")
    launch = sub.add_parser("submit")
    launch.add_argument("run_id")
    launch.add_argument("--partition", default="standard,long")
    launch.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)
    launch.add_argument("--conditioning", choices=("static", "query"), default="static")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.parent, conditioning=args.conditioning)
    else:
        submit(args.run_id, partition=args.partition, time_limit=args.time_limit,
               conditioning=args.conditioning)


if __name__ == "__main__":
    main()
