"""Qualify the Q/K width 4 exponential on the unscaled cosine score.

The run is the exponential's width-four recipe (r2739c80cc032, key_width_acceptance) with its score alone changed, as
in the unscaled-cosine pair (affine_query_cosine_acceptance.py): the rotated queries and keys after QK RMSNorm's
channel gains are unit-normalized (``qk_norm_type: rms_norm_cosine``), without the learned head multiplier, so every
score is a cosine in [-1, 1] and the weight e^s lies within e^-1 and e. Head RMSNorm takes the max-abs prescale, as
every new run does (the same function as the partner's plain arithmetic). Seed 42, data order, schedule and every
other setting are the partner's, so its initial parameters are the partner's without the head multipliers.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

import numpy as np

from experiments.transformer_pp import key_width_acceptance
from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

PARTNER = Path(__file__).resolve().parent / "runs/r2739c80cc032/config.yaml"
PARTNER_SHA256 = "73ecd43b5132131762344ef4aba26c9b84d688f5d96dd8354f9b28646aefa4e1"
# The score's settings, all the run changes from its partner.
CHANGES = {"qk_norm_type": "rms_norm_cosine", "qk_norm_head_scale": False, "attention_head_rms_prescale": "max_abs"}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
# The partner's per-head query multipliers, which the unscaled cosine drops.
HEAD_SCALE = "qk_norm_head_log_scale"
# The fused exponential's GPU qualification at this width.
GPU_TYPES = key_width_acceptance.gpu_types("exponential", 4)
# The final query and key directions are rounded to BF16 before the dot product.
COSINE_BOUND = 1.01


def partner_config():
    if hashlib.sha256(PARTNER.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError("Pinned exponential partner r2739c80cc032 changed")
    return load_config(PARTNER)


def recipe():
    partner = partner_config()
    return replace(partner, model=replace(partner.model, **CHANGES))


def check_recipe(config):
    validate_config(config)
    actual, expected = config_to_dict(config), config_to_dict(recipe())
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from the unscaled-cosine exponential recipe: {changed}")


def check_initialization(config):
    """Every partner array bit-identical, without the head multipliers alone."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    dropped = sorted(path for path in original if HEAD_SCALE in path.split("/"))
    if not dropped or set(actual) != set(original) - set(dropped):
        raise RuntimeError("The cosine score changed the partner's parameters beyond its head multipliers")
    for path, value in actual.items():
        if value.shape != original[path].shape or not np.array_equal(value, original[path]):
            raise RuntimeError(f"Initial parameter {path} differs from the partner's")
    return {"partner_run_id": "r2739c80cc032", "dropped_parameters": dropped,
            "all_other_partner_arrays_identical": True}


def check_pilot(config, outcome, validation):
    layers = validation["attention_diagnostics"]["layers"]
    low = np.asarray([row["score_min"] for row in layers])
    high = np.asarray([row["score_max"] for row in layers])
    if (len(layers) != config.model.num_layers
            or not np.all((-COSINE_BOUND <= low) & (low <= high) & (high <= COSINE_BOUND))):
        raise RuntimeError("Pilot raw scores exceed the unscaled cosine range or are missing")
    return {"score_min": float(low.min()), "score_max": float(high.max())}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   check_pilot=check_pilot, required_backend="fused",
                   gpu_tests=(*shared.gpu_tests, *(test for test in key_width_acceptance.GPU_TESTS["exponential"]
                                                   if test not in shared.gpu_tests)),
                   cpu_tests=(*shared.cpu_tests, "scaffolds/transformer/tests/test_qk_rms_cosine.py",
                              "experiments/transformer_pp/tests/test_exp_cosine_acceptance.py"))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="B300")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"The fused exponential is qualified at width 4 on {' or '.join(GPU_TYPES)}, "
                         f"not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
