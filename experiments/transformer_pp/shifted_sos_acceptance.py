"""Qualify z h_t(z) at Q/K width 4 with a degree-sixteen query SoS core, on the row-power shifted kernel.

The run is rabfbab32a594's recipe (z h_t(z), z = softplus(s), with a degree-eight core, total nine, at Q/K width 4)
with a degree-sixteen core (total seventeen), evaluated in units of each row's largest power
(``model.attention_polynomial_shift: row_power``, attention/shifted_polynomial.py) on its fused kernel: the Mosaic
kernels stop at degree twelve, and at width 4 the scores grow until z^17 overflows FP32. Nothing else changes.
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

PARENT = Path(__file__).parent / "runs/rabfbab32a594/config.yaml"
PARENT_SHA256 = "77f89f17c19a9f29d154807c3824586064d29491a6c9f814e7d973dbcadfd543"
CORE_DEGREE = 16
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
# The fused kernel's GPU qualification (validation README, "Row-power shifted query polynomials").
GPU_TYPES = ("B200", "B300")


def parent_config():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("Pinned z h_t(z) parent rabfbab32a594 changed")
    return load_config(PARENT)


def recipe():
    parent = parent_config()
    model = replace(parent.model, attention_degree=CORE_DEGREE + parent.model.attention_coefficient_input_power,
                    attention_backend="fused", attention_polynomial_shift="row_power")
    return replace(parent, model=model)


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
        raise ValueError(f"Settings differ from the degree-sixteen core recipe: {changed}")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="fused",
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_shifted_sos_acceptance.py"))


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
