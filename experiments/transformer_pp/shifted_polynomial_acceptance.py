"""Qualify the degree-sixteen tanh query polynomial p_t(s) and fixed Taylor T_16(s) at Q/K width 4 on the row-power
shifted kernel.

Each run is its degree-eight partner at Q/K width 4 (query rd5e3ec5a4f02, fixed Taylor r1eaa4a4c3c77) at degree
sixteen, evaluated in units of each row's largest power (``model.attention_polynomial_shift: row_power``,
attention/shifted_polynomial.py) on its fused kernel, with head RMSNorm's max-abs prescale, through which the shift's
row factor divides out: the Mosaic kernels stop at degree twelve, and at width 4 the scores grow until s^16 overflows
FP32. Nothing else changes.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).parent / "runs"
# Each degree-eight partner at Q/K width 4: its run and recipe SHA256.
PARTNERS = {
    "query": ("rd5e3ec5a4f02", "f0126846c2326b3e3c4cc74ef4c9151f442ef39452e950d41f3d82dcacb7a34d"),
    "taylor": ("r1eaa4a4c3c77", "b28b17f7a43fd5a3d107280916a06968c9d4e792437507112e9bc468f5acd7f3"),
}
DEGREE = 16
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
# The fused kernel's GPU qualification (validation README, "Row-power shifted query polynomials").
GPU_TYPES = ("B200", "B300")


def partner_key(config):
    """Fixed coefficients take the Taylor partner, tanh query coefficients the query partner."""
    return "taylor" if config.model.attention_coefficients == "fixed" else "query"


def partner_config(key):
    run, digest = PARTNERS[key]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned degree-eight partner {run} changed")
    return load_config(path)


def recipe(key):
    partner = partner_config(key)
    model = replace(partner.model, attention_degree=DEGREE, attention_backend="fused",
                    attention_polynomial_shift="row_power", attention_head_rms_prescale="max_abs")
    return replace(partner, model=model)


def check_recipe(config):
    validate_config(config)
    actual, expected = config_to_dict(config), config_to_dict(recipe(partner_key(config)))
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from the degree-sixteen {partner_key(config)} recipe: {changed}")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="fused",
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_shifted_polynomial_acceptance.py"))


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
