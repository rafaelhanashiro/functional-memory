"""Qualify e^s h_t(z), the degree-twelve query SoS core of z = softplus(s) under exp of the raw score, at Q/K width 64,
at one and two times the schedule's learning rate.

Each run is the degree-twelve z h_t(z) recipe (rf637de9b65b1, best 14.093) with the factor z replaced by exp of the
raw score (``model.attention_score_prefactor: exp``, ``attention_coefficient_input_power: 0``, total degree twelve)
on the Mosaic ``query_softplus_times_exp`` kernels, qualified to degree twelve at width 64
(``mosaic.api.PREFACTOR_DEGREE_LIMITS``), and its peak and floor learning rates multiplied by 1 or 2. Its degree-eight
core is the best RoPE polynomial at this width (raeaa47865867, 14.067; 14.001 at twice the rate). Head RMSNorm takes
the max-abs prescale, as every new run does. The prefactor replaces a structural factor and adds no parameter, so each
run's initial parameters are the partner's, bit for bit.
"""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

from experiments.transformer_pp import key_width_acceptance
from scaffolds.transformer.acceptance import attention_spec, comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
PARTNER = "rf637de9b65b1"
PARTNER_SHA256 = "22a94ee8cc46fa053cfe2d7789567552888882038d0eb666c705a2762a762903"
CHANGES = {"attention_degree": 12, "attention_coefficient_input_power": 0, "attention_score_prefactor": "exp",
           "attention_head_rms_prescale": "max_abs"}
SCALES = (1.0, 2.0)
RATES = ("learning_rate", "min_learning_rate")
# The Mosaic transformed query-polynomial kernels' qualification.
GPU_TYPES = ("H100", "B200")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_config():
    path = RUNS / PARTNER / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError(f"Pinned degree-twelve partner {PARTNER} changed")
    return load_config(path)


def recipe(scale):
    if scale not in SCALES:
        raise ValueError(f"The learning-rate scales are {SCALES}, not {scale}")
    partner = partner_config()
    return replace(partner, model=replace(partner.model, **CHANGES),
                   optim=replace(partner.optim, **{name: getattr(partner.optim, name) * scale for name in RATES}))


def scale_of(config) -> float:
    ratio = config.optim.learning_rate / partner_config().optim.learning_rate
    for scale in SCALES:
        if math.isclose(ratio, scale, rel_tol=1e-9):
            return scale
    raise ValueError(f"optim.learning_rate is {ratio:g} times the partner's; the scales are {SCALES}")


def check_recipe(config):
    validate_config(config)
    actual, expected = comparison_recipe(config), comparison_recipe(recipe(scale_of(config)))
    for entry in (actual, expected):
        for name in EXECUTION_KEYS:
            entry["train"].pop(name, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from {PARTNER} under exp of the raw score: {changed}")


def check_initialization(config):
    """Every partner array, bit for bit."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    if set(actual) != set(original) or any(
            actual[path].shape != value.shape or not np.array_equal(actual[path], value)
            for path, value in original.items()):
        raise RuntimeError(f"Initial parameters differ from the partner {PARTNER}'s")
    return {"partner_run_id": PARTNER, "all_partner_arrays_identical": True, "score_prefactor": "exp"}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   required_backend="mosaic",
                   cpu_tests=(*shared.cpu_tests,
                              "experiments/transformer_pp/tests/test_wide_exp_prefactor_sos_acceptance.py"))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="B300")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"The Mosaic transformed kernels are qualified on {' or '.join(GPU_TYPES)}, "
                         f"not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
