"""Qualify the pure tanh square-factor SoS core at Q/K width 4 under the softplus and exp prefactors, with and without
the raw-input L2 coefficient normalization, at four times the schedule's learning rate.

Each run is r70e98e1944d5's recipe (z h_t(z), z = softplus(s), with a degree-sixteen pure tanh SoS core, 19.385 at
the recipe's rate) with its peak and floor rates multiplied by 4, the rate where the affine core does best
(rdc43b5c9d2d0, 18.203), and at most two changes:
- the exp prefactor: e^s h_t(z) in place of z h_t(z) (``attention_score_prefactor: exp``, input power 0, total degree
  sixteen), the change from the affine z h_t(z) to codex's affine e^s h_t(z) (r8609e3605f85);
- the plain coefficient input: the gain-applied pre-attention RMSNorm activation without normalization (the affine
  runs' input) in place of the raw attention-block input at unit L2 norm.
Neither change alters a parameter, so each run starts from the partner's parameters, bit for bit.
"""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

from experiments.transformer_pp import key_width_acceptance
from scaffolds.transformer.acceptance import attention_spec, comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
PARTNER = "r70e98e1944d5"
PARTNER_SHA256 = "13fd0096ea3b42f63cc83eeb14dff939574197d28e5a6d2f76d942bef0dc7f26"
SCALE = 4.0
RATES = ("learning_rate", "min_learning_rate")
EXP_PREFACTOR = {"attention_score_prefactor": "exp", "attention_coefficient_input_power": 0, "attention_degree": 16}
PLAIN_INPUT = {"attention_coefficient_input_source": "post_rmsnorm", "attention_coefficient_input_normalization": "none"}
VARIANTS = {"softplus_l2": {}, "exp_l2": EXP_PREFACTOR, "softplus_plain": PLAIN_INPUT,
            "exp_plain": {**EXP_PREFACTOR, **PLAIN_INPUT}}
# The fused shifted kernel's GPU qualification (validation README, "Row-power shifted query polynomials").
GPU_TYPES = ("B200", "B300")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_config():
    path = RUNS / PARTNER / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError(f"Pinned pure tanh SoS partner {PARTNER} changed")
    return load_config(path)


def recipe(variant):
    if variant not in VARIANTS:
        raise ValueError(f"The variants are {sorted(VARIANTS)}, not {variant!r}")
    partner = partner_config()
    return replace(partner, model=replace(partner.model, **VARIANTS[variant]),
                   optim=replace(partner.optim, **{name: getattr(partner.optim, name) * SCALE for name in RATES}))


def variant_of(config) -> str:
    model = config.model
    prefactor = "exp" if model.attention_score_prefactor == "exp" else "softplus"
    plain = model.attention_coefficient_input_normalization == "none"
    return f"{prefactor}_{'plain' if plain else 'l2'}"


def check_recipe(config):
    validate_config(config)
    actual, expected = comparison_recipe(config), comparison_recipe(recipe(variant_of(config)))
    for entry in (actual, expected):
        for name in EXECUTION_KEYS:
            entry["train"].pop(name, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from {PARTNER} at four times its rate ({variant_of(config)}): {changed}")


def check_initialization(config):
    """Every partner array, bit for bit."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    if set(actual) != set(original) or any(
            actual[path].shape != value.shape or not np.array_equal(actual[path], value)
            for path, value in original.items()):
        raise RuntimeError(f"Initial parameters differ from the partner {PARTNER}'s")
    return {"partner_run_id": PARTNER, "all_partner_arrays_identical": True, "variant": variant_of(config)}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   required_backend="fused",
                   cpu_tests=(*shared.cpu_tests,
                              "experiments/transformer_pp/tests/test_pure_tanh_sos_prefactor_acceptance.py"))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="B300")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"The row-power shifted kernel is qualified on {' or '.join(GPU_TYPES)}, not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
