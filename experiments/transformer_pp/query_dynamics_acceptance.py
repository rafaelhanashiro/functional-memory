"""Qualify the instrumented rerun against the original degree-eight query model."""
from dataclasses import replace
from functools import partial
import hashlib
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

PARENT = Path(__file__).resolve().parent / "runs/r25fffc16c2f3"
RECIPE_HASH = "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"
CHECKPOINT_HASH = "67bcf58402898297744f32fb34e5aaa853d86a26136c5a44324bc61a7b3c7c28"


def original():
    path = PARENT / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RECIPE_HASH:
        raise ValueError("Original query recipe hash differs")
    return load_config(path)


def check_recipe(config):
    reference = original()
    reference.train = replace(reference.train, query_gradient_dynamics=True, gradient_event_capture=True,
                              workdir=config.train.workdir, checkpoint_dir=config.train.checkpoint_dir,
                              wandb_run_name=config.train.wandb_run_name)
    if config_to_dict(config) != config_to_dict(reference):
        raise ValueError("Dynamics rerun differs from the parent outside logging and run identity")


def check_parent(config):
    check_recipe(config)
    checkpoint = load_run(PARENT)["selected_checkpoint"]
    if checkpoint["sha256"] != CHECKPOINT_HASH or Path(checkpoint["path"]).name != "7500":
        raise ValueError("Unexpected parent checkpoint")
    verify_checkpoint(dict(checkpoint_path=checkpoint["path"], checkpoint_files=checkpoint["files"]))
    graph, params = restore_parent(config, Path(checkpoint["path"]))
    # Four sequences in the repetitive part of the reference batch; same 2,048-token context.
    start = (6376 - 1)*config.data.batch_size + 210
    tokens = next(PackedBatchIterator(config.data.train_manifest, batch_size=4,
                                      sequence_length=config.data.sequence_length, start_sequence=start))
    measurements = measure_batch(make_probe(graph, config.model), params, tokens)
    facts = {key: float(measurements[key]) for key in ("nll", "bias_relative_error", "amplitude_relative_error")}
    for role in ("bias", "amplitude"):
        signed, mass = measurements[role + "_signed"], measurements[role + "_abs"]
        if not (np.isfinite(mass).all() and mass.max() > 0 and (np.abs(signed) <= mass*1.0001 + 1.e-8).all()):
            raise ValueError(f"Invalid per-query absolute mass for {role}")
    return dict(parent=PARENT.name, step=7500, sequence_start=start,
                batch_sha256=measurements["batch_sha256"], **facts)


def spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="mosaic", check_initialization=check_parent,
                   cpu_tests=(*shared.cpu_tests, "scaffolds/transformer/tests/test_query_gradient_dynamics.py"))


if __name__ == "__main__":
    shared_acceptance(partial(spec, entry_module=__name__))
