"""Qualify twice the Chinchilla budget for the Q/K width 4 learning-rate sweep's best recipes.

Each run is a partner's recipe at 2x Chinchilla (``derive_recipe.py --budget 2``: 15,140 updates, warmup 910,
validation every 200 updates through 3,000 and every 1,000 after), at the partner's learning rate or half of it
(peak and floor together), every other setting the partner's. The partners are the sweep's best rates at 1x
(learning_rate_acceptance.py): the exponential at twice the schedule's rate (rcc51ed9dbca8, best 18.667) and z h_t(z)
with a degree-sixteen SoS core at four times it (rdc43b5c9d2d0, best 18.203), on their fused kernels. A longer
schedule usually prefers a lower rate, so each also trains at half its partner's: the exponential at the schedule's
own rate, z h_t(z) at twice it. A budget and a rate change no initialization, so each run starts from its partner's
parameters, and each qualifies through the learning-rate adapter on the recipe at its partner's budget and rate.
"""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

from experiments.transformer_pp import key_width_acceptance, learning_rate_acceptance
from scaffolds.transformer.acceptance import comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Each partner's run and its recipe's SHA-256.
PARTNERS = {
    "exponential": ("rcc51ed9dbca8", "cad3d286d82a39d709ac5e5a73a518a46a45e777c58d6c3c197283a4fb85005e"),
    "shifted_sos": ("rdc43b5c9d2d0", "9ebd4be8e725651b6c2488e827d9588150b12f06dffe8d5f0ac8f617e5502a9a"),
}
# Learning rates as multiples of the partner's.
SCALES = (1.0, 0.5)
RATES = ("learning_rate", "min_learning_rate")
# derive_recipe's 2x schedule at the reference shape, batch unchanged; the tests check that derive_recipe.py agrees.
SCHEDULE = {"optim": {"total_steps": 15140, "warmup_steps": 910},
            "train": {"eval_early_every": 200, "eval_early_until": 3000, "eval_every": 1000}}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_key(config) -> str:
    return "exponential" if config.model.attention_degree is None else "shifted_sos"


def partner_config(key):
    run, digest = PARTNERS[key]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned partner {run} changed")
    return load_config(path)


def recipe(key, scale):
    if scale not in SCALES:
        raise ValueError(f"The learning-rate scales are {SCALES}, not {scale}")
    partner = partner_config(key)
    optim = {**SCHEDULE["optim"], **{name: getattr(partner.optim, name) * scale for name in RATES}}
    return replace(partner, optim=replace(partner.optim, **optim), train=replace(partner.train, **SCHEDULE["train"]))


def scale_of(config) -> float:
    ratio = config.optim.learning_rate / partner_config(partner_key(config)).optim.learning_rate
    for scale in SCALES:
        if math.isclose(ratio, scale, rel_tol=1e-9):
            return scale
    raise ValueError(f"optim.learning_rate is {ratio:g} times its partner's; the scales are {SCALES}")


def at_partner(config):
    """The recipe at its partner's budget and rate: the partner's own, which the learning-rate adapter qualifies."""
    partner = partner_config(partner_key(config))
    optim = {name: getattr(partner.optim, name) for name in (*RATES, *SCHEDULE["optim"])}
    return replace(config, optim=replace(config.optim, **optim),
                   train=replace(config.train, **{name: getattr(partner.train, name) for name in SCHEDULE["train"]}))


def check_recipe(config):
    validate_config(config)
    key = partner_key(config)
    actual, expected = comparison_recipe(config), comparison_recipe(recipe(key, scale_of(config)))
    for entry in (actual, expected):
        for name in EXECUTION_KEYS:
            entry["train"].pop(name, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from {PARTNERS[key][0]} at twice the Chinchilla budget: {changed}")
    learning_rate_acceptance.check_recipe(at_partner(config))


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    partner = learning_rate_acceptance.experiment_spec(at_partner(config), entry_module=entry_module)

    def check_initialization(current):
        # The partner's pins and comparisons, and its arrays bit for bit: initialization reads neither the schedule
        # nor the rate.
        facts = partner.check_initialization(at_partner(current))
        actual = key_width_acceptance.flat_initial_parameters(current)
        original = key_width_acceptance.flat_initial_parameters(partner_config(partner_key(current)))
        if set(actual) != set(original) or any(
                actual[path].shape != value.shape or not np.array_equal(actual[path], value)
                for path, value in original.items()):
            raise RuntimeError(f"Initial parameters differ from the partner {PARTNERS[partner_key(current)][0]}'s")
        return {**facts, "partner_run_id": PARTNERS[partner_key(current)][0], "all_partner_arrays_identical": True}

    return replace(partner, check_config=check_recipe, check_initialization=check_initialization,
                   cpu_tests=(*partner.cpu_tests,
                              "experiments/transformer_pp/tests/test_narrow_key_budget_acceptance.py"))


def gpu_types(config) -> tuple:
    """The partner's kernels train only on the GPUs they are qualified on."""
    return learning_rate_acceptance.gpu_types(at_partner(config))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config")
    parser.add_argument("--gpu-type", default="B300")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    allowed = gpu_types(load_config(args.config))
    if args.gpu_type.upper() not in allowed:
        raise ValueError(f"This run qualifies and trains on {' or '.join(allowed)} only, not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
