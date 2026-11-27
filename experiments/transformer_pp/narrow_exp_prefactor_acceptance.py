"""Qualify the exponential under a softplus or sigmoid prefactor of the raw score at Q/K width 4.

Each run is the shifted exponential's recipe at width 4 (r2739c80cc032, best 19.190) with W = f(s) e^s, f = softplus or
sigmoid (``model.attention_score_prefactor``), on the blocked backend's causal strips, the one implementation that
applies the row-maximum shift to a score function (the fused kernel shifts the plain exponential alone; at width 4 the
scores pass e^s's FP32 range). Head RMSNorm takes the max-abs prescale, as every new run does. At width 64 the same
prefactors gained 0.88% and 0.50% over the exponential on its fused kernel (rbd24b948a240, 14.132; rd7c9f610e9dc,
14.186; exp 14.257). The prefactor adds no parameter, so each run's initial parameters are the partner's, bit for bit.
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
PARTNER = "r2739c80cc032"
PARTNER_SHA256 = "73ecd43b5132131762344ef4aba26c9b84d688f5d96dd8354f9b28646aefa4e1"
PREFACTORS = ("softplus", "sigmoid")
CHANGES = {"attention_backend": "blocked", "attention_blocked_implementation": "strips",
           "attention_head_rms_prescale": "max_abs"}
# The causal strips are plain JAX: any of the cluster's training GPUs.
GPU_TYPES = ("B200", "B300", "H100")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_config():
    path = RUNS / PARTNER / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError(f"Pinned exponential partner {PARTNER} changed")
    return load_config(path)


def recipe(prefactor):
    if prefactor not in PREFACTORS:
        raise ValueError(f"The exponential prefactor runs take {PREFACTORS}, not {prefactor!r}")
    partner = partner_config()
    return replace(partner, model=replace(partner.model, attention_score_prefactor=prefactor, **CHANGES))


def check_recipe(config):
    validate_config(config)
    actual, expected = comparison_recipe(config), comparison_recipe(recipe(config.model.attention_score_prefactor))
    for entry in (actual, expected):
        for name in EXECUTION_KEYS:
            entry["train"].pop(name, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from {PARTNER} under its prefactor: {changed}")


def check_initialization(config):
    """Every partner array, bit for bit."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    if set(actual) != set(original) or any(
            actual[path].shape != value.shape or not np.array_equal(actual[path], value)
            for path, value in original.items()):
        raise RuntimeError(f"Initial parameters differ from the partner {PARTNER}'s")
    return {"partner_run_id": PARTNER, "all_partner_arrays_identical": True,
            "score_prefactor": config.model.attention_score_prefactor}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   required_backend="blocked",
                   cpu_tests=(*shared.cpu_tests,
                              "experiments/transformer_pp/tests/test_narrow_exp_prefactor_acceptance.py"))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="B300")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"These runs train on {' or '.join(GPU_TYPES)}, not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
