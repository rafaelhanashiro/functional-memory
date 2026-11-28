"""Qualify the instrumented rerun against the matching original degree-eight checkpoints."""
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scaffolds.records import load_run
from scaffolds.transformer.acceptance import attention_spec, main as shared_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.loader import config_to_dict
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.coefficient_fit import restore_parent
from scaffolds.transformer.training.query_dynamics import make_probe, measure_batch
from experiments.transformer_pp.evaluate_per_position import verify_checkpoint

RUNS = Path(__file__).resolve().parent / "runs"
PARENTS = {
    "query": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2",
              "67bcf58402898297744f32fb34e5aaa853d86a26136c5a44324bc61a7b3c7c28"),
    "static": ("r9c69dd7cfc8d", "f17be1f96b471b36d6c0369b4e4e7dc1ac93b0960376c920b8b40972760c9b8f",
               "3542744274506232dce3cbde2afdce9c2f7ffe70e876638fde4d5b712ab67ed1"),
}


def check_recipe(config):
    run_id, recipe_hash, _ = PARENTS[config.model.attention_coefficient_conditioning]
    path = RUNS / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != recipe_hash:
        raise ValueError("Original polynomial recipe hash differs")
    reference = load_config(path)
    reference.train = replace(reference.train, query_gradient_dynamics=True, gradient_event_capture=True,
                              workdir=config.train.workdir, checkpoint_dir=config.train.checkpoint_dir,
                              wandb_run_name=config.train.wandb_run_name, source_commit=config.train.source_commit)
    if config_to_dict(config) != config_to_dict(reference):
        raise ValueError("Dynamics rerun differs from the parent outside logging and run identity")


def check_parent(config):
    check_recipe(config)
    mode = config.model.attention_coefficient_conditioning
    run_id, _, checkpoint_hash = PARENTS[mode]
    checkpoint = load_run(RUNS / run_id)["selected_checkpoint"]
    if checkpoint["sha256"] != checkpoint_hash or Path(checkpoint["path"]).name != "7500":
        raise ValueError("Unexpected parent checkpoint")
    verify_checkpoint(dict(checkpoint_path=checkpoint["path"], checkpoint_files=checkpoint["files"]))
    graph, params = restore_parent(config, Path(checkpoint["path"]))
    # Four sequences in the repetitive part of the reference batch; same 2,048-token context.
    start = (6376 - 1)*config.data.batch_size + 210
    tokens = next(PackedBatchIterator(config.data.train_manifest, batch_size=4,
                                      sequence_length=config.data.sequence_length, start_sequence=start))
    measurements = measure_batch(make_probe(graph, config.model), params, tokens)
    facts = {key: float(measurements[key]) for key in measurements if key == "nll" or key.endswith("_relative_error")}
    for role in (("bias", "amplitude") if mode == "query" else ("bias",)):
        for suffix in ("_relative_error", "_geometry_relative_error"):
            error = measurements[role + suffix]
            if not np.isfinite(error) or error > .025:
                raise ValueError(f"Parent {role}{suffix} exceeds the reconstruction tolerance: {error}")
        signed, mass = measurements[role + "_signed"], measurements[role + "_abs"]
        if not (np.isfinite(mass).all() and mass.max() > 0 and (np.abs(signed) <= mass*1.0001 + 1.e-8).all()):
            raise ValueError(f"Invalid per-query absolute mass for {role}")
    return dict(parent=run_id, step=7500, sequence_start=start,
                batch_sha256=measurements["batch_sha256"], **facts)


def check_pilot(config, outcome, validation):
    # A fresh GPU process releases the replay buffers before the resume test.
    result = subprocess.run([sys.executable, "-m", "experiments.transformer_pp.query_dynamics_acceptance",
                             "--checkpoint-probe", str(config.train.workdir / "config.yaml")],
                            env={**os.environ, "JAX_PLATFORMS": "cuda"}, text=True,
                            stdout=subprocess.PIPE, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="mosaic", check_pilot=check_pilot,
                   cpu_tests=(*shared.cpu_tests, "scaffolds/transformer/tests/test_query_gradient_dynamics.py",
                              "experiments/transformer_pp/tests/test_query_dynamics_acceptance.py"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--checkpoint-probe":
        print(json.dumps(check_parent(load_config(sys.argv[2]))))
    else:
        shared_acceptance(partial(spec, entry_module=__name__))
