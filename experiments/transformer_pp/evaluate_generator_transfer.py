"""Evaluate frozen batch-fitted query generators on the next packed batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml

from scaffolds.records import capture_artifacts, update_record, validate_record
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import (
    convert_generator_readout, parent_config, restore_parent, split_generator_parameters,
)

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(spec):
    root = Path(spec["checkpoint_path"])
    for item in spec["checkpoint_files"]:
        path = root / item["path"]
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint changed: {path}")


def restore_generator(spec, template):
    root = Path(spec["checkpoint_path"])
    if spec["storage"] == "completed_best":
        identity = json.loads((root / "coefficient_identity.json").read_text())
        if identity["step"] != spec["best_step"]:
            raise ValueError("Selected best generator step changed")
        with ocp.StandardCheckpointer() as checkpointer:
            return checkpointer.restore(root / "best", target={
                "generator_parameters": [template],
            })["generator_parameters"][0]
    if spec["storage"] != "resumable_best":
        raise ValueError("Unknown generator checkpoint layout")
    controller = json.loads((root / "controller.json").read_text())
    if controller["best_step"] != spec["best_step"]:
        raise ValueError("Resumable checkpoint's best step changed")
    item = {"best": [template]}
    with ocp.PyTreeCheckpointer() as checkpointer:
        return checkpointer.restore(root / "default", args=ocp.args.PyTreeRestore(
            item=item, partial_restore=True,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item),
        ))["best"][0]


def make_evaluator(graph):
    @jax.jit
    def evaluate(generator, backbone, tokens):
        logits = nnx.merge(graph, backbone, generator)(tokens[:, :-1], deterministic=True)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)
    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "completion.json").exists():
        raise FileExistsError("Evaluation already completed")
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    evaluation = recipe["evaluation"]
    validate_record(run, checkpoint_hashes=False)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use one GPU")
    runtime = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.uname().nodename, "gpu_type": jax.devices()[0].device_kind, "gpu_count": 1,
        "python_executable": sys.executable, "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "versions": {name: version(name) for name in ("jax", "flax", "optax", "orbax-checkpoint")},
        "attention_backend": recipe["model"]["attention_backend"],
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_HOME", "XLA_FLAGS", "TRANSFORMER_CUDA_ENVIRONMENT", "XLA_PYTHON_CLIENT_PREALLOCATE")},
        "optimizer_updates": 0,
    }
    update_record(run, {"status": "running", "runtime": runtime, "attempts": [runtime]})
    parent = evaluation["parent"]
    if sha256(parent["config_path"]) != parent["config_sha256"]:
        raise ValueError("Parent configuration changed")
    print("Verifying source checkpoint checksums", flush=True)
    verify_checkpoint(parent)
    for spec in evaluation["generators"]:
        if sha256(spec["config_path"]) != spec["config_sha256"]:
            raise ValueError("Fitting recipe changed")
        verify_checkpoint(spec)
    config = parent_config(Path(parent["config_path"]))
    if sha256(config.data.eval_manifest) != evaluation["manifest_sha256"]:
        raise ValueError("Evaluation manifest changed")
    batches = {}
    for name, spec in evaluation["batches"].items():
        batch = next(iter(PackedBatchIterator(config.data.eval_manifest,
            batch_size=spec["sequence_count"], sequence_length=config.data.sequence_length,
            start_sequence=spec["start_sequence"])))
        if hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest() != spec["batch_sha256"]:
            raise ValueError(f"Batch checksum changed: {name}")
        batches[name] = batch
    fit, heldout = evaluation["batches"]["fitting"], evaluation["batches"]["heldout"]
    if fit["start_sequence"] + fit["sequence_count"] != heldout["start_sequence"]:
        raise ValueError("Held-out batch must immediately follow the fitting batch")
    print("Restoring shared pretrained backbone", flush=True)
    graph, source = restore_parent(config, Path(parent["checkpoint_path"]))
    source_hash = state_sha256(source)
    original_generator, backbone = split_generator_parameters(source)
    backbone_hash = state_sha256(backbone)
    if backbone_hash != evaluation["backbone_sha256"]:
        raise ValueError("Backbone differs from original fitting experiments")
    original_hash = state_sha256(original_generator)
    microbatch = evaluation["microbatch_size"]
    measurements = []
    checks = []
    models = [{"role": "pretrained", "run_id": parent["run_id"], "readout": "diagonal",
               "best_step": parent["step"], "expected_fitting_nll": evaluation["parent_fitting_nll"]},
              *evaluation["generators"]]
    for spec in models:
        converted_graph, converted_source = convert_generator_readout(config, graph, source, spec["readout"])
        template, frozen = split_generator_parameters(converted_source)
        if state_sha256(frozen) != backbone_hash:
            raise ValueError("Conversion changed backbone")
        generator = template if spec["role"] == "pretrained" else restore_generator(spec, template)
        generator_hash = state_sha256(generator)
        evaluate = make_evaluator(converted_graph)
        print(f"Evaluating {spec['role']} (checkpoint update {spec['best_step']})", flush=True)
        first = jax.device_put(batches["fitting"][:microbatch])
        replay_a = np.asarray(evaluate(generator, frozen, first))
        replay_b = np.asarray(evaluate(generator, frozen, first))
        if not np.array_equal(replay_a, replay_b):
            raise ValueError("Repeated frozen evaluation is not deterministic")
        for name, batch in batches.items():
            losses = []
            for start in range(0, len(batch), microbatch):
                values = np.asarray(evaluate(generator, frozen, jax.device_put(batch[start:start+microbatch])))
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite evaluation: {spec['role']} {name}")
                losses.extend(values.astype(np.float64).tolist())
            nll = float(np.mean(losses))
            if name == "fitting" and abs(nll - spec["expected_fitting_nll"]) > evaluation["replay_nll_atol"]:
                raise ValueError(f"Fitting replay differs: {nll} versus {spec['expected_fitting_nll']}")
            row = {"role": spec["role"], "source_run_id": spec["run_id"], "checkpoint_step": spec["best_step"],
                   "batch": name, **evaluation["batches"][name], "prediction_tokens": len(batch)*(batch.shape[1]-1),
                   "nll": nll, "perplexity": math.exp(nll), "per_sequence_nll": losses}
            measurements.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "per_sequence_nll"}), flush=True)
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if state_sha256(generator) != generator_hash or state_sha256(frozen) != backbone_hash:
            raise ValueError("Evaluation changed parameters")
        checks.append({"role": spec["role"], "generator_sha256": generator_hash,
                       "parameters_unchanged": True, "deterministic_replay": True,
                       "fitting_replay_absolute_error": abs(measurements[-2]["nll"] - spec["expected_fitting_nll"])})
    if state_sha256(source) != source_hash or state_sha256(original_generator) != original_hash:
        raise ValueError("Parent parameters changed")
    outcome = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer_updates": 0, "backbone_sha256": backbone_hash, "checks": checks,
               "measurements": [{k: v for k, v in row.items() if k != "per_sequence_nll"} for row in measurements],
               "interpretation": "Frozen shared generators evaluated on the next 256 packed sequences; no adaptation on the new batch."}
    (run / "completion.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    (run / "metadata.json").write_text(json.dumps({"status": "complete", "runtime": runtime}, indent=2, sort_keys=True) + "\n")
    update_record(run, {**capture_artifacts(run), "status": "complete", "evaluation_outcome": outcome})
    print("Frozen-generator transfer evaluation complete", flush=True)


if __name__ == "__main__":
    main()
