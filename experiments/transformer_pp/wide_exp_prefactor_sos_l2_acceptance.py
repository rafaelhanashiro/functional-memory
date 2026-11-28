"""Qualify e^s h_t(z) at Q/K width 64 with the raw-input L2 coefficient normalization, at one, two and four times
the schedule's learning rate.

Each run is the degree-eight e^s h_t(z) recipe (raeaa47865867, best 14.067; z = softplus(s), affine tanh query SoS
core) with its coefficient generator reading the raw attention-block input at unit L2 norm
(``attention_coefficient_input_source: pre_rmsnorm``, ``attention_coefficient_input_normalization: l2``) in place of
the gain-applied pre-attention RMSNorm activation, and its peak and floor rates multiplied by 1, 2 or 4. Q, K and V
keep the RMSNorm input. Head RMSNorm takes the max-abs prescale, as every new run does, so the runs at two and four
times the rate differ from ORCD's plain-input ones (r39d0469cc122, 14.001; r365f0c30f632, 14.179) in the coefficient
input alone. The normalization adds no parameter, so each run starts from the partner's parameters, bit for bit.
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
PARTNER = "raeaa47865867"
PARTNER_SHA256 = "6225bc0496c3da72aa51b4be41540cd0f88219f1622364dad16b972058102a9c"
CHANGES = {"attention_coefficient_input_source": "pre_rmsnorm", "attention_coefficient_input_normalization": "l2",
           "attention_head_rms_prescale": "max_abs"}
SCALES = (1.0, 2.0, 4.0)
RATES = ("learning_rate", "min_learning_rate")
# The plain-input run at each rate: the partner, and ORCD's runs at two and four times it.
PLAIN_INPUT_RUNS = {1.0: PARTNER, 2.0: "r39d0469cc122", 4.0: "r365f0c30f632"}
# The Mosaic transformed query-polynomial kernels' qualification.
GPU_TYPES = ("H100", "B200")
# The L2 generator input (CPU and GPU cases, the Mosaic e^s h_t(z) one among them) and the SoS core it feeds.
FEATURE_TESTS = ("scaffolds/transformer/tests/test_coefficient_input_normalization.py",
                 "scaffolds/transformer/tests/test_coefficient_raw_input.py",
                 "scaffolds/transformer/tests/test_softplus_sos_raw_input.py")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_config():
    path = RUNS / PARTNER / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError(f"Pinned e^s h_t(z) partner {PARTNER} changed")
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
        raise ValueError(f"Settings differ from {PARTNER} with the raw-input L2 coefficient input: {changed}")


def check_initialization(config):
    """Every partner array, bit for bit."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    if set(actual) != set(original) or any(
            actual[path].shape != value.shape or not np.array_equal(actual[path], value)
            for path, value in original.items()):
        raise RuntimeError(f"Initial parameters differ from the partner {PARTNER}'s")
    return {"partner_run_id": PARTNER, "all_partner_arrays_identical": True, "coefficient_input": "raw_l2"}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   required_backend="mosaic",
                   cpu_tests=(*shared.cpu_tests, *FEATURE_TESTS, "scaffolds/transformer/tests/test_softplus_sos.py",
                              "experiments/transformer_pp/tests/test_wide_exp_prefactor_sos_l2_acceptance.py"),
                   gpu_tests=(*shared.gpu_tests, *FEATURE_TESTS,
                              "scaffolds/transformer/tests/test_mosaic_softplus_sos.py"))


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
