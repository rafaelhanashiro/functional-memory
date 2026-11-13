"""Allocate and submit fixed-input polynomial coefficient fitting diagnostics."""
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
STUDY = ROOT / "experiments/transformer_pp"
TOPIC = STUDY / "results/exploratory/coefficient_oracle"
PARAMETERIZATIONS = ("query_coefficient_generator", "shared_batch_query_coefficients", "static_coefficients",
                    "free_per_example_query_coefficients", "free_per_example_query_key_coefficients")
# Submission policy per home cluster (CLUSTERS.md). ORCD requests must name one GPU type;
# its jobs take 12 hours with a 6-hour minimum and requeue before the limit.
CLUSTERS = {
    "kennel": {"partition": "standard,long", "exclude": "collie,samoyed,doodle,terrier,shiba",
               "gpu_types": {"B300", "H100"}, "cpus": 8, "memory": "96G", "time": "02:00:00",
               "time_min": None, "script": "coefficient_fit.sbatch"},
    "orcd": {"partition": "mit_preemptable,mit_normal_gpu",
             "exclude": "node3009,node3209,node1708,node1709,node4508,node2801,node3409",
             "gpu_types": {"H100"}, "cpus": 16, "memory": "128G", "time": "12:00:00",
             "time_min": "06:00:00", "script": "coefficient_fit_orcd.sbatch"},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fitting_scope(parameterization, parent_batch_size, sequence_count=None):
    """Resolve deliberately selected inputs without silently changing older modes."""
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(f"Unknown coefficient parameterization: {parameterization}")
    if parameterization == "query_coefficient_generator":
        count = 1 if sequence_count is None else sequence_count
        if count in {1, parent_batch_size}:
            return count, "query_generator_single_sequence" if count == 1 else "query_generator_full_batch"
        # A larger fixed batch: whole microbatches of eight, named by its size.
        if type(count) is not int or count <= parent_batch_size or count % 8:
            raise ValueError("Generator fitting requires --sequence-count 1, the parent's full 256-sequence "
                             "batch, or a larger multiple of 8")
        return count, f"query_generator_{count}_sequences"
    if parameterization in {"static_coefficients", "free_per_example_query_coefficients",
                            "free_per_example_query_key_coefficients"}:
        if sequence_count not in {1, 256} or sequence_count not in {1, parent_batch_size}:
            raise ValueError("Coefficient-table fitting requires --sequence-count 1 or the parent's full 256-sequence batch")
        suffix = "single_sequence" if sequence_count == 1 else "full_batch"
        prefix = {"static_coefficients": "static_coefficients",
                  "free_per_example_query_coefficients": "free_coefficients",
                  "free_per_example_query_key_coefficients": "free_query_key_coefficients"}[parameterization]
        return sequence_count, f"{prefix}_{suffix}"
    if sequence_count is not None:
        raise ValueError("--sequence-count is only supported for generator, static, or independent coefficient fits")
    if parameterization == "shared_batch_query_coefficients":
        return parent_batch_size, "shared_coefficients_full_batch"
    return 1, "query_generator_single_sequence"


def fitting_role(role, degree):
    """Keep existing degree-eight selections stable while distinguishing other degrees."""
    if type(degree) is not int or degree < 1:
        raise ValueError("Coefficient fitting requires a positive integer degree")
    return role if degree == 8 else f"{role}_m{degree}"


def parent_degree(model, parameterization="query_coefficient_generator"):
    """Check the requested fitted family against the unchanged source checkpoint."""
    from dataclasses import fields
    from scaffolds.transformer.configs import ModelConfig

    degree = model.get("attention_degree")
    fitting_role("query_generator_single_sequence", degree)
    generator = (model["attention_coefficients"] == "learned"
                 and model["attention_coefficient_activation"] == "tanh"
                 and model["attention_coefficient_scale"] == "factorial"
                 and model["attention_coefficient_amplitude_scale"] == "factorial")
    query = generator and model["attention_coefficient_conditioning"] == "query"
    # A trained query-key generator runs on the one-pass fused kernel; Mosaic would
    # evaluate it once per order.
    query_key = (generator and model["attention_coefficient_conditioning"] == "query_key"
                 and parameterization == "query_coefficient_generator")
    static = (model["attention_coefficient_conditioning"] == "static"
              and model["attention_coefficients"] == "learned"
              and model["attention_coefficient_activation"] == "identity"
              and model["attention_coefficient_scale"] == "factorial")
    taylor = (model["attention_coefficient_conditioning"] == "static"
              and model["attention_coefficients"] == "fixed"
              and model.get("attention_fixed_coefficients") is None
              and model.get("attention_fixed_square_factors") is None
              and model.get("attention_coefficient_initialization", "taylor") == "taylor")
    if parameterization == "query_coefficient_generator" and not (query or query_key):
        raise ValueError("Expected a 4B factorial-scaled query-key or query-only parent")
    if (not (query or query_key or static
             or (taylor and parameterization in {"free_per_example_query_coefficients",
                                                 "free_per_example_query_key_coefficients"}))
            or model["attention_normalization"]
            or model.get("attention_coefficient_parameterization", "direct") != "direct"
            or model["initialization"] != "transformer_truncated_reference"
            or not model["qk_norm"] or model["attention_backend"] != ("fused" if query_key else "mosaic")):
        raise ValueError("Expected a compatible 4B unnormalized query, query-key, static, or fixed Taylor parent")
    # The shared model validator owns supported Mosaic widths and degrees.
    checked = dict(model)
    if "gdn_polynomial_degree" not in {field.name for field in fields(ModelConfig)}:
        checked.pop("gdn_polynomial_degree", None)
    ModelConfig(**checked)
    return degree


def prepare(parent: Path, *, parameterization="query_coefficient_generator", sequence_count=None,
            generator_readout="diagonal", home_cluster="kennel"):
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(f"Unknown coefficient parameterization: {parameterization}")
    if generator_readout not in {"diagonal", "dense"}:
        raise ValueError("Generator readout must be diagonal or dense")
    if generator_readout != "diagonal" and parameterization != "query_coefficient_generator":
        raise ValueError("Dense readout requires query_coefficient_generator")
    static = parameterization == "static_coefficients"
    shared = parameterization in {"shared_batch_query_coefficients", "static_coefficients"}
    free_query_key = parameterization == "free_per_example_query_key_coefficients"
    free = parameterization == "free_per_example_query_coefficients" or free_query_key
    table = shared or free
    parent = parent.resolve()
    record = load_run(parent)
    validate_record(parent, checkpoint_hashes=False)
    if home_cluster not in CLUSTERS:
        raise ValueError(f"Home cluster must be one of {sorted(CLUSTERS)}")
    # The fit reads the parent's retained checkpoint, which stays on its home cluster.
    if record["status"] != "complete" or record["home_cluster"] != home_cluster:
        raise ValueError(f"The parent must be a completed {home_cluster} run")
    source = yaml.safe_load((parent / "config.yaml").read_text())
    count, role = fitting_scope(parameterization, int(source["data"]["batch_size"]), sequence_count)
    model = source["model"]
    degree = parent_degree(model, parameterization)
    if model.get("attention_coefficient_readout", "diagonal") != "diagonal":
        raise ValueError("The fitting parent must have the original diagonal query readout")
    query_key = model["attention_coefficient_conditioning"] == "query_key" and not table
    if query_key and generator_readout != "diagonal":
        raise ValueError("A query-key generator keeps its diagonal readout")
    if generator_readout == "dense":
        role = role.replace("query_generator_", "dense_query_generator_", 1)
    if query_key:
        role = role.replace("query_generator_", "query_key_generator_", 1)
    role = fitting_role(role, degree)
    if model.get("qk_norm_head_scale", False):
        role += "_head_scale"
    best = json.loads((parent / "best_validation.json").read_text())
    descriptor = record["selected_checkpoint"]
    checkpoint = Path(descriptor["path"])
    if not checkpoint.is_absolute():
        checkpoint = parent.parents[3] / checkpoint
    if not checkpoint.is_dir() or descriptor["step"] != best["step"]:
        raise ValueError("Parent retained checkpoint and validation best disagree")
    selection_path = TOPIC / "selection.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    if any(entry["role"] == role for entry in selection["runs"]):
        raise ValueError(f"A {role} fit is already allocated in this topic")
    recorded = yaml.safe_load((STUDY / "sweep.yaml").read_text())["data_manifests"]["eval"]["sha256"]
    sequence_identity = {"manifest_sha256": sha256(source["data"]["eval_manifest"])}
    if sequence_identity["manifest_sha256"] != recorded:
        raise ValueError("The fitting sequences' manifest differs from the study's evaluation manifest")
    microbatch = min(count, 8)
    learning_rate = 0.01 if table else 0.001
    import numpy as np
    from scaffolds.transformer.data.packed import PackedBatchIterator
    batch = next(iter(PackedBatchIterator(
        source["data"]["eval_manifest"], batch_size=count,
        sequence_length=source["data"]["sequence_length"], start_sequence=0,
    )))
    sequence_identity["batch_sha256"] = hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest()
    recipe = deepcopy(source)
    generator_prefix = ("query-generator-dense" if generator_readout == "dense" else
                        "query-key-generator" if query_key else "query-generator")
    suffix = "single-sequence" if count == 1 else "full-batch" if count == int(source["data"]["batch_size"]) else f"{count}-sequences"
    label = f"shared-coefficients-m{degree}-full-batch" if shared else f"{generator_prefix}-m{degree}-{suffix}"
    if static:
        label = f"static-coefficients-m{degree}-" + ("single-sequence" if count == 1 else "full-batch")
    if free:
        label = (f"free-query-key-coefficients-m{degree}-" if free_query_key else f"free-coefficients-m{degree}-") + (
            "single-sequence" if count == 1 else "full-batch")
    if model.get("qk_norm_head_scale", False):
        label += "-head-scale"
    recipe.update(schema_version=1, experiment_kind="fixed_input_coefficient_fit",
                  condition=f"coefficient_oracle/{role}")
    if not table and (sequence_count is not None or generator_readout == "dense"):
        recipe["model"]["attention_coefficient_readout"] = generator_readout
    recipe["data"]["batch_size"] = count
    recipe["optim"].update(learning_rate=learning_rate, min_learning_rate=learning_rate / 10,
                           weight_decay=0.0, warmup_steps=0, total_steps=10000)
    recipe["train"].update(source_commit="auto", micro_batch_size=microbatch,
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
        "sequence_count": count, "microbatch_size": microbatch, "data_start": 0,
        "data_split": "eval", "data_role": "fitting inputs; no generalization evaluation",
        "parameterization": parameterization,
        "trainable_parameters": ["query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel"],
        "coefficient_storage": "native trained generator parameters",
        "optimizer_coordinates": "native stored generator parameters; factorial scaling remains in the model",
        "frozen_parameters": "all backbone parameters outside the query coefficient generator",
        "initialization": "restore trained query coefficient generator without resetting parameters",
        "max_updates": 10000, "initial_updates": 2000, "plateau_window": 200,
        "plateau_tolerance": 0.001, "confirmation_updates": 500,
        "learning_rate": learning_rate, "learning_rate_decay": 0.1,
        "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8, "clip_norm": 1.0,
        "gradient_scaling": "mean next-token NLL on the one fixed sequence",
        "checkpoint_every": 100,
        "checkpoint_selection": "lowest fitting NLL; no independent validation",
    }
    if not table:
        recipe["oracle"]["generator_readout"] = generator_readout
        recipe["oracle"]["optimizer_initialization"] = "fresh zero moments for this diagnostic; no parent optimizer state"
        if count > 1:
            recipe["oracle"]["gradient_scaling"] = "mean next-token NLL across the full batch; accumulate all microbatch gradients before one clip and Adam update"
        if generator_readout == "dense":
            recipe["oracle"].update(
                trainable_parameters=["query_polynomial_bias", "query_polynomial_mixing", "query_polynomial_kernel"],
                initialization="restore parent W and b; initialize V=diag(parent amplitude), preserving the parent function",
            )
        if query_key:
            recipe["oracle"].update(
                trainable_parameters=["query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel",
                                      "key_polynomial_kernel", "key_polynomial_amplitude"],
                frozen_parameters=("all parameters outside the query and key coefficient generators, including "
                                   "Q/K/V projections, normalization gains, and per-head QK score scales"),
                initialization="restore the trained query and key coefficient generators without resetting parameters",
            )
    if shared:
        recipe["oracle"].update(
            trainable_parameters=["shared_effective_monomial_coefficients"],
            coefficient_storage="one [layer, query_position, head, order] table shared across all sequences and causal keys",
            optimizer_coordinates="factorial(order)*effective_monomial_coefficient",
            frozen_parameters="all pretrained parameters; original query generator bypassed by the shared table",
            initialization="mean of parent effective coefficients across the entire fitting batch at each layer/query/head/order",
            gradient_scaling="mean next-token NLL across the full batch; accumulate all microbatch gradients before one clip and Adam update",
        )
    if static:
        recipe["oracle"].update(
            trainable_parameters=["static_effective_monomial_coefficients"],
            coefficient_storage="one [layer, head, order] table shared across all query positions, sequences, and causal keys",
            initialization="mean of parent effective coefficients across all fitting sequences and query positions at each layer/head/order",
        )
        if model["attention_coefficient_conditioning"] == "static":
            recipe["oracle"].update(
                static_initialization="checkpoint",
                initialization="restore the parent's exact effective static coefficients without averaging",
                frozen_parameters="all pretrained parameters; attention coefficients replaced by the fitted static table",
            )
    if free:
        recipe["oracle"].update(
            trainable_parameters=["per_example_effective_monomial_coefficients"],
            coefficient_storage="independent [sequence, layer, query_position, head, order] coefficients; shared across causal keys",
            optimizer_coordinates="factorial(order)*effective_monomial_coefficient",
            frozen_parameters="all pretrained parameters; attention coefficients replaced by the per-example tables",
            initialization="capture actual coefficients on each sequence's original forward pass",
            gradient_scaling="per-sequence mean NLL, clipped independently per sequence",
        )
    if free_query_key:
        if model["attention_coefficient_conditioning"] != "static" or model["attention_coefficients"] != "fixed":
            raise ValueError("Free query-key coefficients start from a fixed Taylor parent")
        recipe["oracle"].update(
            trainable_parameters=["per_example_effective_monomial_coefficients", "per_example_key_factors"],
            coefficient_storage=("independent [sequence, layer, query_position, head, order] coefficients c and "
                                 "[sequence, layer, key_position, head, order] key factors g; pair coefficient c_r(t) g_r(i)"),
            optimizer_coordinates="factorial(order)*effective_monomial_coefficient for c; g directly",
            frozen_parameters="all pretrained parameters; attention coefficients replaced by the per-example tables",
            initialization="capture actual coefficients on each sequence's original forward pass; key factors one",
            gradient_scaling="per-sequence mean NLL; each sequence's c and g clipped jointly",
            fit_backend="fused",
        )
    run = create_run(STUDY, recipe, label=label)
    update_record(run, {
        "home_cluster": home_cluster, "seed": source["train"]["seed"],
        "parent_run_id": parent.name, "parent_study": parent.parent.parent.name,
        "coefficient_sharing": ("shared across query positions, sequences, and causal keys; separate per layer/head/order" if static else
                                "shared across sequences and causal keys; separate per layer/query position/head/order"
                                if shared else "query factor per sequence/layer/query position/head/order times a key factor "
                                "per sequence/layer/key position/head/order" if free_query_key else
                                "shared across causal keys; separate per sequence/layer/query position/head/order"
                                if free else "shared query- and key-conditioned generators per layer and head"
                                if query_key else "shared query-conditioned generator per layer and head"),
        "coefficient_parameterization": parameterization,
        **({"generator_readout": generator_readout} if not table else {}),
        "checkpoint_retention": "best fitting parameters; backbone referenced by source checkpoint",
        "source_checkpoint": {**descriptor, "path": str(checkpoint)},
    })
    previous_parent = selection["settings"].pop("parent_run_id", None)
    for entry in selection["runs"]:
        if previous_parent:
            entry.setdefault("parent_run_id", previous_parent)
    selection["question"] = "How much can polynomial coefficients fit fixed inputs with a frozen 4B backbone?"
    selection["settings"]["comparison"] = "Different source backbones; compare fitting capacity, not a matched-backbone parameterization ablation"
    selection["runs"].append({"study": STUDY.name, "run_id": run.name, "role": role,
                              "parent_run_id": parent.name, "sources": {}})
    selection_path.write_text(yaml.safe_dump(selection, sort_keys=False))
    refresh_selection(TOPIC)
    regenerate_index(STUDY)
    print(run.name, flush=True)


def live_job_state(job_id):
    """Expired Slurm IDs are absent; scheduler errors must not trigger a duplicate job."""
    result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                            text=True, capture_output=True)
    if result.returncode:
        if "invalid job id" in result.stderr.lower():
            return ""
        raise RuntimeError(f"Cannot check existing Slurm job {job_id}: {result.stderr.strip()}")
    return result.stdout.strip()


def submit(run_id, *, partition=None, time_limit=None, gpu_type=None,
           qualify_only=False, cluster="kennel"):
    if cluster not in CLUSTERS:
        raise ValueError(f"Cluster must be one of {sorted(CLUSTERS)}")
    policy = CLUSTERS[cluster]
    partition, time_limit = partition or policy["partition"], time_limit or policy["time"]
    if cluster == "orcd":
        # An untyped GPU request matches no ORCD Hopper node.
        gpu_type = gpu_type or "H100"
    if gpu_type not in {None, *policy["gpu_types"]}:
        raise ValueError(f"GPU type must be one of {sorted(policy['gpu_types'])} on {cluster}")
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
        live = live_job_state(previous)
        if live:
            raise ValueError(f"Run already has live job {previous}: {live}")
    if record.get("home_cluster") != cluster:
        raise ValueError(f"Run {run_id} belongs to {record.get('home_cluster')}, not {cluster}")
    log = run / "training.log"
    gpu_request = f"gpu:{gpu_type.lower()}:1" if gpu_type else "gpu:1"
    command = ["sbatch", "--parsable", "--nice=0", f"--partition={partition}",
               f"--gres={gpu_request}", f"--exclude={policy['exclude']}",
               f"--cpus-per-task={policy['cpus']}", f"--mem={policy['memory']}", f"--time={time_limit}",
               *([f"--time-min={policy['time_min']}"] if policy["time_min"] else []),
               f"--job-name=coefficient-fit-{run_id}", f"--output={log}", f"--error={log}",
               str(STUDY / policy["script"]), str(ROOT), commit, run_id]
    if cluster == "orcd":
        command.append(gpu_type)
    if qualify_only:
        command.append("--qualify-only")
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {job}")
    update_record(run, {"status": "submitted", "source_commit": commit,
        **({"error": None} if record.get("error") else {}),
        "frozen_execution_sha256": frozen_hash(ROOT, commit, STUDY.name, "transformer"),
        "submission": {"slurm_job_id": job, "partition": partition, "command": command,
                       "cluster": cluster, "gpu_type": gpu_type, "gpu_count": 1,
                       "phase": "qualification" if qualify_only else "training",
                       "submitted_utc": datetime.now(UTC).isoformat()}, "log_path": str(log)})
    regenerate_index(STUDY)
    print(f"{run_id} job_id={job}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--parent", type=Path, required=True)
    prep.add_argument("--parameterization", choices=PARAMETERIZATIONS, default=PARAMETERIZATIONS[0])
    # fitting_scope admits 1, the parent's batch, or a larger generator batch (a multiple of 8).
    prep.add_argument("--sequence-count", type=int)
    prep.add_argument("--generator-readout", choices=("diagonal", "dense"), default="diagonal")
    prep.add_argument("--home-cluster", choices=sorted(CLUSTERS), default="kennel")
    launch = sub.add_parser("submit")
    launch.add_argument("run_id")
    launch.add_argument("--cluster", choices=sorted(CLUSTERS), default="kennel")
    launch.add_argument("--partition", help="default: the cluster's partitions")
    launch.add_argument("--time-limit", help="default: the cluster's job limit")
    launch.add_argument("--gpu-type", choices=("B300", "H100"))
    launch.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent, parameterization=args.parameterization, sequence_count=args.sequence_count,
                generator_readout=args.generator_readout, home_cluster=args.home_cluster)
    else:
        submit(args.run_id, partition=args.partition, time_limit=args.time_limit, gpu_type=args.gpu_type,
               qualify_only=args.qualify_only, cluster=args.cluster)


if __name__ == "__main__":
    main()
