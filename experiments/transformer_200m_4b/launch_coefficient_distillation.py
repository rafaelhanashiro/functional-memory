"""Allocate and launch teacher-activation coefficient distillation on Kennel."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import create_run, load_run, regenerate_index, update_record, validate_record
from scaffolds.records.selections import refresh_selection
from experiments.transformer_200m_4b.coefficient_distillation import read_batch, sha256

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/transformer_200m_4b"
TOPIC = STUDY / "results/exploratory/coefficient_oracle"


def source_descriptor(run):
    run = Path(run).resolve()
    record = load_run(run)
    if record["status"] != "complete" or record["home_cluster"] != "kennel":
        raise ValueError("Distillation sources must be completed Kennel runs")
    validate_record(run, checkpoint_hashes=False)
    selected = record["selected_checkpoint"]
    checkpoint = Path(selected["path"])
    if not checkpoint.is_absolute(): checkpoint = run.parents[3] / checkpoint
    if not checkpoint.is_dir(): raise ValueError("Source checkpoint unavailable")
    return {"run_id": run.name, "run_directory": str(run), "config_sha256": sha256(run / "config.yaml"),
            "checkpoint": str(checkpoint), "checkpoint_sha256": selected["sha256"],
            "checkpoint_step": int(checkpoint.name)}


def prepare(teacher):
    teacher = Path(teacher).resolve()
    original = yaml.safe_load((teacher / "config.yaml").read_text())
    if original["oracle"]["parameterization"] != "free_per_example_query_coefficients":
        raise ValueError("Expected the per-sequence free coefficient teacher")
    parent = Path(original["oracle"]["parent_config"]).parent
    source = yaml.safe_load((parent / "config.yaml").read_text())
    if source["model"]["attention_degree"] != 4 or original["oracle"]["sequence_count"] != 256:
        raise ValueError("This experiment deliberately selects the m4 B256 teacher")
    teacher_source, parent_source = source_descriptor(teacher), source_descriptor(parent)
    if parent_source["checkpoint_sha256"] != original["oracle"]["parent_checkpoint_sha256"]:
        raise ValueError("Teacher's frozen backbone does not match selected Taylor parent")
    outcome = json.loads((teacher / "completion.json").read_text())
    teacher_source["best_nll"] = outcome["best_nll"]
    recipe = deepcopy(source)
    model = recipe["model"]
    model.update(attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_readout="dense", attention_coefficient_feature_count=9,
        attention_coefficient_readout_scale=1/3, attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_projection_initial_std=model["hidden_size"] ** -.5,
        attention_coefficient_tanh_bias=False, attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_initialization="taylor",
        remat=False)
    # All QK/backbone fields remain those of the teacher's canonical-head-scale parent.
    expected_qk = {"qk_norm": True, "qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
                   "qk_norm_initial_scale": None, "qk_norm_head_scale": True}
    if any(model.get(key) != value for key, value in expected_qk.items()):
        raise ValueError("Expected the canonical per-head-scale parent")
    splits = {}
    for name, manifest, start in (
        ("fit", source["data"]["eval_manifest"], 0),
        ("validation", source["data"]["train_manifest"], 5_000_000),
        ("test", source["data"]["train_manifest"], 5_000_256),
    ):
        splits[name] = {"manifest": manifest, "manifest_sha256": sha256(manifest),
            "start_sequence": start, "sequence_count": 256,
            "role": "teacher fitting" if name == "fit" else "held out; never optimizer inputs"}
    checked = {}
    for split in splits.values():
        _, identity = read_batch(split, source["data"]["sequence_length"], checked)
        split["batch_sha256"] = identity
    if splits["fit"]["batch_sha256"] != original["oracle"]["batch_sha256"]:
        raise ValueError("Teacher batch differs from the proposed fit inputs")
    recipe.update(schema_version=1, experiment_kind="teacher_activation_coefficient_distillation",
                  condition="coefficient_oracle/dense_teacher_distillation_m4_head_scale")
    recipe["optim"].update(learning_rate=.001, min_learning_rate=.0001, weight_decay=0.,
        coefficient_penalty=0., beta1=.9, beta2=.95, grad_clip_norm=1., warmup_steps=0, total_steps=20000)
    recipe["train"].update(source_commit="auto", micro_batch_size=8, log_every=25, eval_every=1000,
        eval_batches=1, ckpt_every=1000, wandb_run_name="dense-teacher-distillation-m4-b256-head-scale",
        wandb_group="transformer_200m_4b", wandb_mode="online", wandb_dir=None)
    recipe["distillation"] = {"teacher": teacher_source, "parent": parent_source, "splits": splits,
        "trainable_parameters": ["query_polynomial_kernel", "query_polynomial_bias", "query_polynomial_mixing"],
        "frozen_parameters": "all Taylor backbone parameters", "input": "actual free-teacher attention entry after block RMSNorm",
        "target": "raw factorial coefficients u_r=r!*c_r", "objective": "mean squared error per layer/head",
        "gradient_scaling": "sum independent per-layer/head mean losses; clip each network before Adam",
        "regression_batch_size": 4096, "lm_microbatch_size": 8, "updates": 20000,
        "learning_rate": .001, "min_learning_rate": .0001, "schedule": "cosine over fixed budget",
        "beta1": .9, "beta2": .95, "epsilon": 1e-8, "clip_norm": 1.,
        "log_every": 25, "eval_every": 1000, "checkpoint_every": 1000,
        "checkpoint_selection": "lowest fresh validation deployed language-model NLL, including Taylor initialization",
        "capacity_selection": "lowest full fitting coefficient MSE; evaluated on fitting sequences only",
        "test_policy": "evaluate selected generator and Taylor baseline once after training; no adaptation",
        "sampling": "uniform query tokens with replacement; same positions across layers and heads; seed folded with update",
        "dense_architecture_reference_run": "r04a5b2a3e8dd",
        "retention": "validation-best generator only; source backbone remains at its parent"}
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    role = "dense_teacher_distillation_m4_head_scale"
    if any(entry["role"] == role for entry in selection["runs"]):
        raise ValueError("This distillation experiment is already allocated")
    run = create_run(STUDY, recipe, label=recipe["train"]["wandb_run_name"])
    update_record(run, {"home_cluster": "kennel", "seed": recipe["train"]["seed"],
        "teacher_run_id": teacher.name, "parent_run_id": parent.name,
        "artifact_root": str(run), "dataset_identity": splits})
    selection["runs"].append({"study": STUDY.name, "run_id": run.name, "role": role,
        "parent_run_id": parent.name, "teacher_run_id": teacher.name, "sources": {}})
    (TOPIC / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(TOPIC)
    regenerate_index(STUDY)
    print(run.name, flush=True)


def submit(run_id, gpu_type="auto"):
    run = STUDY / "runs" / run_id
    record = load_run(run)
    validate_record(run, checkpoint_hashes=False)
    if record["status"] == "complete": raise ValueError("Already complete")
    previous = record.get("submission", {}).get("slurm_job_id")
    if previous:
        checked = subprocess.run(["squeue", "-h", "-j", str(previous), "-o", "%T"], text=True, capture_output=True)
        if checked.returncode and "invalid job id" not in checked.stderr.lower():
            raise RuntimeError(checked.stderr)
        if checked.stdout.strip(): raise ValueError("Run already has a live Slurm job")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if changes(ROOT, commit, STUDY.name, "transformer"):
        raise ValueError("Commit the frozen execution set before submitting")
    log = run / "training.log"
    gpu_request = "gpu:1" if gpu_type == "auto" else f"gpu:{gpu_type}:1"
    command = ["sbatch", "--parsable", "--partition=standard", f"--gres={gpu_request}",
        "--exclude=collie,samoyed,doodle,terrier,shiba",
        "--cpus-per-task=12", "--mem=160G", "--time=24:00:00", f"--job-name=distill-{run_id}",
        f"--output={log}", f"--error={log}", str(STUDY / "coefficient_distillation.sbatch"), str(ROOT), commit, run_id]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit(): raise RuntimeError(f"Unexpected Slurm response: {job}")
    update_record(run, {"status": "submitted", "source_commit": commit,
        "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
        "submission": {"slurm_job_id": job, "partition": "standard", "command": command,
            "submitted_utc": datetime.now(timezone.utc).isoformat()}, "log_path": str(log)})
    regenerate_index(STUDY)
    print(f"{run_id} job_id={job}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--teacher", required=True, type=Path)
    launch = sub.add_parser("submit")
    launch.add_argument("run_id")
    launch.add_argument("--gpu-type", choices=("auto", "b300", "h100"), default="auto")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.teacher)
    else: submit(args.run_id, args.gpu_type)
