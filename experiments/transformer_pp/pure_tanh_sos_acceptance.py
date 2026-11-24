"""Qualify z h_t(z) at Q/K width 4 with a degree-sixteen SoS core whose square factors are pure tanh fractions of
the Taylor factors, on the row-power shifted kernel.

The run is r2e7f5617fdf2's recipe (z h_t(z), z = softplus(s), h_t = A_t^2 + B_t^2 of core degree sixteen, total
seventeen) with its factor generator alone changed: each factor is F_j(x_t) = f_j tanh(w_j^T x_hat_t + beta_j), a
fraction of its fixed Taylor factor f_j, in place of f_j (1 + b_j + a_j tanh(W_j^T x_t)) / 2. x_hat_t is the raw
attention-block input at unit L2 norm (``attention_coefficient_input_source: pre_rmsnorm``,
``attention_coefficient_input_normalization: l2``); W starts at zero and every beta_j at atanh(1/2), so every factor
starts at half its Taylor factor and each query's z h_t(z) at a quarter of z T_16(z). W takes the weight decay and
beta none, as in the pure tanh direct runs (pure_tanh_prefactor_acceptance.py). Nothing else changes.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
import math
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

PARTNER = Path(__file__).parent / "runs/r2e7f5617fdf2/config.yaml"
PARTNER_SHA256 = "53e71c960ce46cbe21b420a810610ff8a80d4ec9484d0d8f4ca218b6a65473bf"
# The factor generator's settings, all the run changes from its partner.
CHANGES = {
    "attention_coefficient_activation": "pure_tanh",
    "attention_coefficient_tanh_bias": True,
    "attention_coefficient_tanh_bias_initial": math.atanh(0.5),
    "attention_coefficient_projection_initial_std": 0.0,
    "attention_coefficient_amplitude_initial": 1.0,
    "attention_coefficient_input_source": "pre_rmsnorm",
    "attention_coefficient_input_normalization": "l2",
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
# The fused shifted kernel's GPU qualification (validation README, "Row-power shifted query polynomials").
GPU_TYPES = ("B200", "B300")


def partner_config():
    if hashlib.sha256(PARTNER.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError("Pinned z h_t(z) partner r2e7f5617fdf2 changed")
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
        raise ValueError(f"Settings differ from the pure tanh square-factor recipe: {changed}")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="fused",
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_pure_tanh_sos_acceptance.py"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="B300")
    args, _ = parser.parse_known_args(argv)
    if args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"The row-power shifted kernel is qualified on {', '.join(GPU_TYPES)}, not {args.gpu_type}")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
