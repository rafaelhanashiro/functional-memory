"""Allocate and submit matched continuations of one pretrained checkpoint.

A continuation resumes a completed parent at its validation-best checkpoint and trains
all parameters on fresh data. A fixed Taylor or unnormalized exp parent can introduce
learned coefficients or gates, with a matched fixed-Taylor control. A learned static or
query parent can instead preserve its complete model and Adam state unchanged.
"""
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
from scaffolds.transformer.configs.base import fresh_recipe_uses_blocked_strips

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/transformer_pp"
LEARNING_RATE = 0.0001
DEFAULT_TIME_LIMIT = "06:00:00"
CONDITIONINGS = ("static", "query", "token_scale", "forget_gate")
# Default topics of the single-pair comparisons; several allocations may share one topic.
TOPICS = {"static": "taylor_static_continuation", "token_scale": "token_query_scale_continuation",
          "forget_gate": "forget_gate_continuation"}
# Additional updates and the introduced parameters' own constant learning rate; the
# backbone keeps LEARNING_RATE. The coefficient pairs allocated before 2026-10-28 used 500
# updates at the backbone's rate; their records hold those settings.
UPDATES = 2000
NEW_PARAMETER_LEARNING_RATE = 0.001
BRANCH_ROLES = {"static": "learned_static", "query": "learned_query", "token_scale": "token_query_scale",
                "forget_gate": "query_forget_gate"}
QUESTIONS = {
    "static": "Does enabling static coefficients after fixed-kernel pretraining improve matched continuation?",
    "query": "Does enabling query coefficients after fixed-kernel pretraining improve matched continuation?",
    "token_scale": "Does a per-token query scale gate after Taylor pretraining improve matched continuation?",
    "forget_gate": "Does the matched continuation detect the forget gate, which beats fixed Taylor in pretraining?",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topic_path(conditioning: str, topic: str | None = None) -> Path:
    if conditioning not in CONDITIONINGS:
        raise ValueError("Continuation conditioning must be static, query, token_scale or forget_gate")
    name = topic or TOPICS.get(conditioning)
    if not name:
        raise ValueError(f"The {conditioning} continuation needs an explicit --topic")
    return STUDY / "results/exploratory" / name


def parent_kind(model: dict) -> str:
    """"taylor" for a fixed Taylor polynomial parent, "exp" for an unnormalized exp parent."""
    return "exp" if model.get("attention_degree") is None else "taylor"


def branch_role(conditioning: str, *, kind: str, kernel_weight_decay: bool,
                no_rope: bool = False, token_scale_kernel_weight_decay: bool = False) -> str:
    """The branch's selection role; exp parents' runs carry an ``exp_`` prefix."""
    role = BRANCH_ROLES[conditioning]
    if no_rope:
        role += "_no_rope"
    if kernel_weight_decay or token_scale_kernel_weight_decay:
        role += "_decay"
    return f"{kind}_{role}" if kind == "exp" else role


def control_role(kind: str) -> str:
    return "exp_fixed_taylor" if kind == "exp" else "fixed_taylor"


def adopt_blocked_strips(model: dict) -> dict:
    """A copied recipe that can train on the blocked backend uses its causal strips, as
    ``scaffolds.records create`` writes them (prepare allocates with create_run directly)."""
    if fresh_recipe_uses_blocked_strips(model):
        model["attention_blocked_implementation"] = "strips"
    return model


def recipes(source: dict, continuation: dict, *, conditioning: str = "static", updates: int = UPDATES,
            new_rate: float | None = NEW_PARAMETER_LEARNING_RATE, kernel_weight_decay: bool = False,
            degree: int | None = None, no_rope: bool = False,
            token_scale_kernel_weight_decay: bool = False,
            preserve_all_parameters: bool = False) -> dict[str, dict]:
    """Match additional training while enabling the requested mechanism.

    An exp parent needs ``degree``: both branches switch to the degree-``degree`` polynomial
    kernel on Mosaic, the control as fixed Taylor and the branch with its learned mechanism.
    """
    if preserve_all_parameters:
        model = source["model"]
        if (model.get("attention_coefficients") != "learned" or conditioning not in ("static", "query")
                or model.get("attention_coefficient_conditioning") != conditioning
                or any((kernel_weight_decay, no_rope, token_scale_kernel_weight_decay, degree is not None))):
            raise ValueError("Preserved continuation keeps the learned parent's entire model recipe")
        role = conditioning + ("_decay" if model.get("attention_coefficient_kernel_weight_decay") else "") + "_self"
        recipe = deepcopy(source)
        recipe["optim"].update(learning_rate=LEARNING_RATE, min_learning_rate=LEARNING_RATE,
                               warmup_steps=0, total_steps=continuation["parent_step"] + updates)
        recipe["train"].update(source_commit="auto", micro_batch_size=8, log_every=10, eval_every=100,
                               eval_early_every=None, eval_early_until=0, ckpt_every=100,
                               wandb_run_name=f"{role}-m{model['attention_degree']}-continuation",
                               wandb_group=STUDY.name, wandb_dir=None, wandb_mode="online")
        recipe["continuation"] = {**deepcopy(continuation), "preserve_all_parameters": True,
                                  "new_parameter_learning_rate": None}
        adopt_blocked_strips(recipe["model"])
        return {role: recipe}
    kind = parent_kind(source["model"])
    if kind == "exp":
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Continuing an exp parent needs the polynomial degree")
    elif degree not in (None, source["model"]["attention_degree"]):
        raise ValueError("A Taylor parent keeps its degree")
    degree = source["model"]["attention_degree"] if degree is None else degree
    if kernel_weight_decay and conditioning not in ("query", "forget_gate"):
        raise ValueError("Kernel weight decay applies to query coefficients")
    if no_rope and conditioning != "forget_gate":
        raise ValueError("Removing RoPE is supported for the forget-gate continuation")
    if token_scale_kernel_weight_decay and conditioning != "token_scale":
        raise ValueError("Token-scale kernel weight decay requires token_scale conditioning")
    result = {}
    branch = branch_role(conditioning, kind=kind, kernel_weight_decay=kernel_weight_decay,
                         no_rope=no_rope, token_scale_kernel_weight_decay=token_scale_kernel_weight_decay)
    for role in (control_role(kind), branch):
        recipe = deepcopy(source)
        recipe["optim"].update(learning_rate=LEARNING_RATE, min_learning_rate=LEARNING_RATE,
                               warmup_steps=0, total_steps=continuation["parent_step"] + updates)
        recipe["train"].update(source_commit="auto", micro_batch_size=8, log_every=10, eval_every=100,
                               eval_early_every=None, eval_early_until=0, ckpt_every=100,
                               wandb_run_name=f"{kind}-m{degree}-continuation-{role.removeprefix('exp_')}",
                               wandb_group=STUDY.name, wandb_dir=None, wandb_mode="online")
        recipe["continuation"] = deepcopy(continuation)
        # The frozen recipes state every gate explicitly, including the control's closed ones.
        recipe["model"]["qk_norm_token_scale"] = "none"
        recipe["model"]["qk_norm_token_scale_kernel_weight_decay"] = False
        recipe["model"].setdefault("attention_forget_gate", False)
        if kind == "exp":
            recipe["model"].update(attention_degree=degree, attention_backend="mosaic")
        if role == control_role(kind):
            adopt_blocked_strips(recipe["model"])
            result[role] = recipe
            continue
        recipe["continuation"]["new_parameter_learning_rate"] = new_rate
        if conditioning == "token_scale":
            recipe["model"].update(qk_norm_token_scale="cosine",
                                   qk_norm_token_scale_kernel_weight_decay=token_scale_kernel_weight_decay)
        else:
            recipe["model"].update(attention_coefficients="learned",
                                   attention_coefficient_sharing="per_head",
                                   attention_coefficient_scale="factorial")
            if conditioning == "forget_gate":
                # The pretraining winner: zero-amplitude tanh query coefficients with Gated
                # DeltaNet's forget gate, on the blocked reference backend (Mosaic has no gate).
                recipe["model"].update(attention_forget_gate=True, attention_backend="blocked")
                if no_rope:
                    recipe["model"]["positional_encoding"] = "none"
            if conditioning in ("query", "forget_gate"):
                recipe["model"].update(
                    attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh",
                    attention_coefficient_amplitude_initial=0.0,
                    attention_coefficient_projection_initial_std=1 / math.sqrt(source["model"]["hidden_size"]),
                    attention_coefficient_tanh_bias=False,
                    attention_coefficient_amplitude_scale="factorial",
                    attention_coefficient_kernel_weight_decay=bool(kernel_weight_decay),
                )
        adopt_blocked_strips(recipe["model"])
        result[role] = recipe
    return result


def existing_control(control: str, parent: Path, continuation: dict, updates: int) -> dict:
    """A completed fixed-Taylor continuation of the same parent and budget, reused as the control."""
    run = STUDY / "runs" / control
    record = load_run(run)
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    if (record["status"] != "complete" or record.get("condition") not in ("fixed_taylor", "exp_fixed_taylor")
            or record.get("parent_run_id") != parent.name
            or recipe["continuation"]["parent_checkpoint_sha256"] != continuation["parent_checkpoint_sha256"]
            or recipe["optim"]["total_steps"] != continuation["parent_step"] + updates
            or recipe["optim"]["learning_rate"] != LEARNING_RATE
            or recipe["model"].get("attention_coefficients") != "fixed"):
        raise ValueError("The control must be a complete fixed-Taylor continuation of this parent with the same budget")
    return record


def prepare(parent: Path, *, conditioning: str = "static", control: str | None = None, topic: str | None = None,
            updates: int = UPDATES, new_rate: float | None = NEW_PARAMETER_LEARNING_RATE,
            kernel_weight_decay: bool = False, degree: int | None = None,
            control_only: bool = False, no_rope: bool = False,
            token_scale_kernel_weight_decay: bool = False,
            preserve_all_parameters: bool = False) -> dict[str, str]:
    """Allocate a branch and, unless a control is reused or declined, its fixed-Taylor control.

    ``control_only`` allocates just the fixed-Taylor continuation of the parent, for example
    the control of exp-parent branches allocated earlier. Allocations may share a topic: an
    existing selection gains the new runs.
    """
    if control_only and control is not None:
        raise ValueError("A control-only allocation does not reuse a control")
    if preserve_all_parameters:
        if control not in (None, "none") or control_only:
            raise ValueError("Preserved continuation allocates only the unchanged learned model")
        control, new_rate = "none", None
    parent = parent.resolve()
    topic_directory = topic_path(conditioning, topic)
    record = load_run(parent)
    validate_record(parent, checkpoint_hashes=True)
    if record["status"] != "complete" or record["home_cluster"] != "kennel":
        raise ValueError("The source must be a completed Kennel run")
    source = yaml.safe_load((parent / "config.yaml").read_text())
    model = source["model"]
    expected = {"attention_coefficients": "fixed",
                "attention_coefficient_conditioning": "static", "attention_normalization": False,
                "attention_output_scaling": "head_rms_norm", "qk_norm": True, "qk_norm_type": "rms_norm",
                "qk_norm_head_scale": True, "initialization": "transformer_truncated_reference"}
    kind = parent_kind(model)
    if preserve_all_parameters:
        expected.update(attention_coefficients="learned", attention_coefficient_conditioning=conditioning)
        kind = conditioning
    if kind == "taylor":
        expected["attention_backend"] = "mosaic"
    if (model.get("attention_fixed_coefficients") is not None or model.get("attention_forget_gate", False)
            or model.get("qk_norm_token_scale", "none") != "none"
            or (kind == "taylor" and (not isinstance(model["attention_degree"], int) or model["attention_degree"] < 1))
            or any(model.get(key) != value for key, value in expected.items())):
        raise ValueError("Expected a 4B fixed Taylor or exp parent with canonical QK/head RMSNorm")
    if source["optim"]["total_steps"] != 7570 or source["data"]["batch_size"] != 256:
        raise ValueError("Expected the 4B recipe and its 256-sequence scientific batch")
    descriptor = record["selected_checkpoint"]
    checkpoint = Path(descriptor["path"])
    if not checkpoint.is_absolute():
        checkpoint = parent.parents[3] / checkpoint
    best = json.loads((parent / "best_validation.json").read_text())
    if not checkpoint.is_dir() or descriptor["step"] != best["step"] or best["step"] != 7500:
        raise ValueError("Expected the validation-best checkpoint at update 7500")
    continuation = {
        "parent_config": str(parent / "config.yaml"),
        "parent_config_sha256": sha256(parent / "config.yaml"),
        "parent_data_identity_sha256": sha256(parent / "data_identity.json"),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": descriptor["sha256"],
        "parent_step": int(best["step"]),
    }
    run_ids = {}
    if control not in (None, "none"):
        existing_control(control, parent, continuation, updates)
        run_ids[control_role(kind)] = control
    allocated = recipes(source, continuation, conditioning=conditioning, updates=updates, new_rate=new_rate,
                        kernel_weight_decay=kernel_weight_decay, degree=degree, no_rope=no_rope,
                        token_scale_kernel_weight_decay=token_scale_kernel_weight_decay,
                        preserve_all_parameters=preserve_all_parameters)
    for role, recipe in allocated.items():
        if role == control_role(kind) and control is not None:
            continue
        if control_only and role != control_role(kind):
            continue
        run = create_run(STUDY, recipe, label=recipe["train"]["wandb_run_name"])
        run_ids[role] = run.name
        update_record(run, {
            "home_cluster": "kennel", "seed": source["train"]["seed"],
            "experiment_kind": "matched_checkpoint_continuation", "condition": role,
            "parent_run_id": parent.name, "parent_study": parent.parent.parent.name,
            "parent_kernel": kind,
            "source_checkpoint": {**descriptor, "path": str(checkpoint)},
            "source_data_identity": {"path": str(parent / "data_identity.json"),
                                     "sha256": continuation["parent_data_identity_sha256"]},
            "source_configuration": {"path": continuation["parent_config"],
                                     "sha256": continuation["parent_config_sha256"]},
            "continuation": {"start_step": best["step"], "updates": updates,
                             "new_prediction_tokens": updates * source["data"]["batch_size"]
                                                      * (source["data"]["sequence_length"] - 1),
                             "backbone_optimizer": "preserve pretrained moments and step",
                             "new_parameter_optimizer": "zero moments and fresh Adam clock"
                                                        + (f" at constant learning rate {new_rate}"
                                                           if role != control_role(kind) and new_rate is not None else ""),
                             "data_start_sequence": best["step"] * source["data"]["batch_size"]},
        })
        if preserve_all_parameters:
            record = load_run(run)
            update_record(run, {"continuation": {
                **record["continuation"],
                "backbone_optimizer": "preserve all pretrained parameters, moments and clocks, including coefficients",
                "new_parameter_optimizer": "not applicable; no parameters introduced",
                "learning_rate": LEARNING_RATE,
            }})
    selection_path = topic_directory / "selection.yaml"
    if selection_path.exists():
        selection = yaml.safe_load(selection_path.read_text())
        known = {entry["run_id"] for entry in selection["runs"]}
        selection["runs"].extend({"study": STUDY.name, "run_id": run_id, "role": role, "sources": {}}
                                 for role, run_id in run_ids.items() if run_id not in known)
    else:
        selection = {
            "schema_version": 1, "opened": datetime.now(UTC).date().isoformat(),
            "question": QUESTIONS[conditioning],
            "settings": {"metric": "validation_perplexity", "updates": updates,
                         "evaluation_every": 100, "learning_rate": LEARNING_RATE,
                         "new_parameter_learning_rate": new_rate,
                         "primary_comparison": "terminal held-out NLL/PPL at the same continuation update",
                         "validation_best": "reported separately from terminal"},
            "runs": [{"study": STUDY.name, "run_id": run_id, "role": role, "sources": {}}
                     for role, run_id in run_ids.items()],
        }
    topic_directory.mkdir(parents=True, exist_ok=True)
    if preserve_all_parameters:
        selection["settings"]["preserved_parameter_learning_rate"] = LEARNING_RATE
    selection_path.write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(topic_directory)
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
            "experiments.transformer_pp.acceptance",
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
           conditioning: str = "static", topic: str | None = None) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = changes(ROOT, commit, STUDY.name, "transformer")
    if dirty:
        raise ValueError(f"Commit the frozen inputs before submitting: {dirty}")
    selection = yaml.safe_load((topic_path(conditioning, topic) / "selection.yaml").read_text())
    if run_id not in {entry["run_id"] for entry in selection["runs"]}:
        raise ValueError("Run is not a member of the explicitly selected continuation topic")
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
    prep.add_argument("--conditioning", choices=CONDITIONINGS, default="static")
    prep.add_argument("--control", help="reuse this completed fixed-Taylor continuation as the control, "
                                        "or 'none' to allocate the branch alone")
    prep.add_argument("--topic", help="results/exploratory topic directory (default: the conditioning's)")
    prep.add_argument("--updates", type=int, default=UPDATES)
    prep.add_argument("--new-parameter-learning-rate", type=float, default=NEW_PARAMETER_LEARNING_RATE)
    prep.add_argument("--kernel-weight-decay", action="store_true",
                      help="apply optim.weight_decay to the query coefficient projection W")
    prep.add_argument("--token-scale-kernel-weight-decay", action="store_true",
                      help="apply optim.weight_decay to the query-temperature direction w only")
    prep.add_argument("--no-rope", action="store_true",
                      help="remove RoPE in the forget-gate branch; keep the fixed-Taylor control unchanged")
    prep.add_argument("--degree", type=int, help="polynomial degree when continuing an exp parent")
    prep.add_argument("--control-only", action="store_true",
                      help="allocate only the parent's fixed-Taylor continuation")
    prep.add_argument("--preserve-all-parameters", action="store_true",
                      help="continue a learned parent unchanged, retaining all Adam history at the backbone LR")
    launch = sub.add_parser("submit")
    launch.add_argument("run_id")
    launch.add_argument("--partition", default="standard,long")
    launch.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)
    launch.add_argument("--conditioning", choices=CONDITIONINGS, default="static")
    launch.add_argument("--topic")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.parent, conditioning=args.conditioning, control=args.control, topic=args.topic,
                updates=args.updates, new_rate=args.new_parameter_learning_rate,
                kernel_weight_decay=args.kernel_weight_decay, degree=args.degree,
                control_only=args.control_only, no_rope=args.no_rope,
                token_scale_kernel_weight_decay=args.token_scale_kernel_weight_decay,
                preserve_all_parameters=args.preserve_all_parameters)
    else:
        submit(args.run_id, partition=args.partition, time_limit=args.time_limit,
               conditioning=args.conditioning, topic=args.topic)


if __name__ == "__main__":
    main()
