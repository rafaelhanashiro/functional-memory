"""Qualify negative-score branches for the degree-eight query polynomial with direct coefficients at Q/K width 64.

Each run is the query polynomial's recipe (r25fffc16c2f3: tanh query coefficients (b_r + a_r tanh(W_r^T x_t))/r!
with W decayed) with the split-score kernel of the SoS runs (split_score_acceptance.py): h_t(s) for s >= 0 and, for
s < 0, exp(s) h_t(s), exp(s) h_t(0) or h_t(0)/D_t(-s) with a positive quartic D_t, on the blocked backend's causal
strips, with head RMSNorm's max-abs prescale. Seed 42, data order, schedule and every other setting are the
control's, so the initial parameters are its own, beside D_t's generator.
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

PARTNER = Path(__file__).resolve().parent / "runs/r25fffc16c2f3/config.yaml"
PARTNER_SHA256 = "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"
# The SoS runs' first three branches: exp(s) h_t(s), exp(s) h_t(0) and h_t(0)/D_t(-s) with a positive quartic.
BRANCHES = ("exp_times_sos", "exp_times_origin", "positive_quartic")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
TESTS = ("scaffolds/transformer/tests/test_split_score.py",)
# The blocked backend's causal strips, as the SoS runs.
GPU_TYPES = ("B200", "B300")


def partner_config():
    if hashlib.sha256(PARTNER.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError("Pinned query polynomial control r25fffc16c2f3 changed")
    return load_config(PARTNER)


def recipe(branch):
    if branch not in BRANCHES:
        raise ValueError(f"The query polynomial's negative-score branches are {BRANCHES}, not {branch!r}")
    partner = partner_config()
    return replace(partner, model=replace(partner.model, attention_negative_score_branch=branch,
                                          attention_backend="blocked", attention_blocked_implementation="strips",
                                          attention_head_rms_prescale="max_abs"))


def check_recipe(config):
    validate_config(config)
    actual, expected = config_to_dict(config), config_to_dict(recipe(config.model.attention_negative_score_branch))
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from the split-score query polynomial recipe: {changed}")


def check_initialization(config):
    """Every control array bit-identical; a quartic denominator adds its own generator alone."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    added = sorted(set(actual) - set(original))
    if set(original) - set(actual) or any("negative_quartic" not in path for path in added):
        raise RuntimeError(f"The split-score branch changed the control's parameters: {added}")
    for path, value in original.items():
        if actual[path].shape != value.shape or not np.array_equal(actual[path], value):
            raise RuntimeError(f"Initial parameter {path} differs from the control's")
    return {"control_run_id": "r25fffc16c2f3", "added_parameters": added, "all_control_arrays_identical": True}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   cpu_tests=(*shared.cpu_tests, *TESTS,
                              "experiments/transformer_pp/tests/test_split_score_query_acceptance.py"),
                   gpu_tests=(*shared.gpu_tests, *TESTS))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="B300")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"The split-score runs train on {' or '.join(GPU_TYPES)}, not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
