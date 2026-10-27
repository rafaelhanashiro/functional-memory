"""Allocate and launch query-key pair-coefficient distillation on Kennel."""
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
from experiments.transformer_200m_4b.launch_coefficient_distillation import source_descriptor

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/transformer_200m_4b"
TOPIC = STUDY / "results/exploratory/coefficient_oracle"
ROLE = "query_key_teacher_distillation_m4_head_scale"
LABEL = "query-key-teacher-distillation-m4-b256-head-scale"


def prepare(teacher):
    teacher = Path(teacher).resolve()
    original = yaml.safe_load((teacher / "config.yaml").read_text())
    oracle = original["oracle"]
    if oracle["parameterization"] != "free_per_example_query_key_coefficients":
        raise ValueError("Expected the per-sequence free query-key teacher")
    parent = Path(oracle["parent_config"]).parent
    source = yaml.safe_load((parent / "config.yaml").read_text())
    if source["model"]["attention_degree"] != 4 or oracle["sequence_count"] != 256:
        raise ValueError("This experiment deliberately selects the m4 B256 teacher")
    teacher_source, parent_source = source_descriptor(teacher), source_descriptor(parent)
    if parent_source["checkpoint_sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Teacher's frozen backbone does not match its Taylor parent")
    teacher_source["best_nll"] = json.loads((teacher / "completion.json").read_text())["best_nll"]
    recipe = deepcopy(source)
    model = recipe["model"]
    expected_qk = {"qk_norm": True, "qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
                   "qk_norm_initial_scale": None, "qk_norm_head_scale": True}
    if any(model.get(key) != value for key, value in expected_qk.items()):
        raise ValueError("Expected the canonical per-head-scale parent")
    backend = oracle["fit_backend"]
    model.update(attention_backend=backend, attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query_key", attention_coefficient_activation="tanh",
        attention_coefficient_readout="diagonal", attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_projection_initial_std=model["hidden_size"] ** -.5,
        attention_coefficient_tanh_bias=False, attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_initialization="taylor",
        remat=False)
    splits = {}
    for name, manifest, start in (("fit", source["data"]["eval_manifest"], 0),
                                  ("heldout", source["data"]["train_manifest"], 5_000_000)):
        splits[name] = {"manifest": manifest, "manifest_sha256": sha256(manifest),
            "start_sequence": start, "sequence_count": 256,
            "role": "teacher fitting" if name == "fit" else "held out; scored once after training, selects nothing"}
    checked = {}
    for split in splits.values():
        _, identity = read_batch(split, source["data"]["sequence_length"], checked)
        split["batch_sha256"] = identity
    if splits["fit"]["batch_sha256"] != oracle["batch_sha256"]:
        raise ValueError("Teacher batch differs from the proposed fit inputs")
    recipe.update(schema_version=1, experiment_kind="teacher_activation_pair_coefficient_distillation",
                  condition=f"coefficient_oracle/{ROLE}")
    recipe["optim"].update(learning_rate=.001, min_learning_rate=.0001, weight_decay=0.,
        coefficient_penalty=0., beta1=.9, beta2=.95, grad_clip_norm=1., warmup_steps=0, total_steps=20000)
    recipe["train"].update(source_commit="auto", micro_batch_size=8, log_every=25, eval_every=1000,
        eval_batches=1, ckpt_every=1000, wandb_run_name=LABEL,
        wandb_group="transformer_200m_4b", wandb_mode="online", wandb_dir=None)
    recipe["distillation"] = {"teacher": teacher_source, "parent": parent_source, "splits": splits,
        "trainable_parameters": ["query_polynomial_kernel", "query_polynomial_bias", "query_polynomial_amplitude",
                                 "key_polynomial_kernel", "key_polynomial_amplitude"],
        "frozen_parameters": "all Taylor backbone parameters",
        "input": "actual free query-key teacher attention entry after block RMSNorm, at the query and key tokens",
        "target": "factorial pair coefficients r! c_r(t) g_r(i) of the free teacher",
        "objective": "mean squared error over causal pairs and orders, per layer/head",
        "gradient_scaling": "sum independent per-layer/head mean losses; clip each network before Adam",
        "regression_pairs_per_update": 4096, "evaluation_pairs": 262144, "lm_microbatch_size": 8,
        "updates": 20000, "learning_rate": .001, "min_learning_rate": .0001, "schedule": "cosine over fixed budget",
        "beta1": .9, "beta2": .95, "epsilon": 1e-8, "clip_norm": 1.,
        "log_every": 25, "eval_every": 1000, "checkpoint_every": 1000, "backend": backend,
        "checkpoint_selection": "lowest pair-regression error on 262,144 fixed evaluation pairs of the fitting batch, "
                                "including the Taylor initialization",
        "heldout_policy": "evaluate the selected generator and the Taylor baseline once after training; no adaptation",
        "sampling": ("uniform causal (query, key) pairs of uniform sequences, with replacement; same pairs across "
                     "layers and heads; seed folded with update"),
        "retention": "selected generator only; source backbone remains at its parent"}
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    if any(entry["role"] == ROLE for entry in selection["runs"]):
        raise ValueError("This distillation experiment is already allocated")
    run = create_run(STUDY, recipe, label=LABEL)
    update_record(run, {"home_cluster": "kennel", "seed": recipe["train"]["seed"],
        "teacher_run_id": teacher.name, "parent_run_id": parent.name,
        "artifact_root": str(run), "dataset_identity": splits})
    selection["runs"].append({"study": STUDY.name, "run_id": run.name, "role": ROLE,
        "parent_run_id": parent.name, "teacher_run_id": teacher.name, "sources": {}})
    (TOPIC / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(TOPIC)
    regenerate_index(STUDY)
    print(run.name, flush=True)


def submit(run_id, gpu_type="auto", partition="standard"):
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
    command = ["sbatch", "--parsable", f"--partition={partition}", f"--gres={gpu_request}",
        "--exclude=collie,samoyed,doodle,terrier,shiba",
        "--cpus-per-task=12", "--mem=160G", "--time=24:00:00", f"--job-name=distill-{run_id}",
        f"--output={log}", f"--error={log}", str(STUDY / "query_key_distillation.sbatch"), str(ROOT), commit, run_id]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit(): raise RuntimeError(f"Unexpected Slurm response: {job}")
    update_record(run, {"status": "submitted", "source_commit": commit,
        "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
        "submission": {"slurm_job_id": job, "partition": partition, "command": command,
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
    launch.add_argument("--partition", default="standard")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.teacher)
    else: submit(args.run_id, args.gpu_type, args.partition)
