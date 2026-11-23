"""Qualify the Q/K width 4 weights P_t(s), softplus(s) P_t(s) and e^s P_t(s) at degrees 8 and 16, with pure tanh
coefficients c_r(x_t) = tanh(w_r^T x_hat_t + beta_r) / r!, on the row-power shifted kernel.

Each run is the degree-eight tanh query run at Q/K width 4 (rd5e3ec5a4f02) with: pure tanh coefficients of every
order 0..m from the raw attention-block input at unit L2 norm (``attention_coefficient_input_source: pre_rmsnorm``,
``attention_coefficient_input_normalization: l2``), W at zero and every beta_r at atanh(1/2), so each coefficient
starts at 0.5/r!; the degree; the prefactor of the signed score; and head RMSNorm's max-abs prescale with the
row-power shift on the fused kernel (``attention_polynomial_shift: row_power``, attention/shifted_polynomial.py), which
evaluates the prefactor's weights in units of each key's bound. The optimizer (the base learning rate at every order,
weight decay 0.1 on W and none on beta), seed, data order and schedule are the partner's.
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

PARTNER = Path(__file__).parent / "runs/rd5e3ec5a4f02/config.yaml"
PARTNER_SHA256 = "f0126846c2326b3e3c4cc74ef4c9151f442ef39452e950d41f3d82dcacb7a34d"
DEGREES = (8, 16)
PREFACTORS = ("none", "softplus", "exp")
# The settings every run changes from the partner, beside its degree and prefactor.
CHANGES = {
    "attention_coefficient_activation": "pure_tanh",
    "attention_coefficient_tanh_bias": True,
    "attention_coefficient_tanh_bias_initial": math.atanh(0.5),
    "attention_coefficient_projection_initial_std": 0.0,
    "attention_coefficient_amplitude_initial": 1.0,
    "attention_coefficient_amplitude_scale": "factorial",
    "attention_coefficient_scale": "none",
    "attention_coefficient_input_source": "pre_rmsnorm",
    "attention_coefficient_input_normalization": "l2",
    "attention_head_rms_prescale": "max_abs",
    "attention_polynomial_shift": "row_power",
    "attention_backend": "fused",
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
# The fused shifted kernel's GPU qualification (validation README, "Row-power shifted query polynomials").
GPU_TYPES = ("B200", "B300")


def partner_config():
    if hashlib.sha256(PARTNER.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError("Pinned tanh query partner rd5e3ec5a4f02 changed")
    return load_config(PARTNER)


def recipe(degree, prefactor):
    if degree not in DEGREES or prefactor not in PREFACTORS:
        raise ValueError(f"The grid takes degrees {DEGREES} and prefactors {PREFACTORS}, not {degree}, {prefactor!r}")
    partner = partner_config()
    model = replace(partner.model, **CHANGES, attention_degree=degree, attention_score_prefactor=prefactor)
    return replace(partner, model=model)


def recipe_key(config):
    return config.model.attention_degree, config.model.attention_score_prefactor


def check_recipe(config):
    validate_config(config)
    actual, expected = config_to_dict(config), config_to_dict(recipe(*recipe_key(config)))
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from the pure tanh {recipe_key(config)} recipe: {changed}")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="fused",
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_pure_tanh_prefactor_acceptance.py"))


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
