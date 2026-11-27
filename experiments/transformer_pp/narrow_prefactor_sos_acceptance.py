"""Qualify softplus(s) h_t(s), the degree-eight query SoS core under the softplus prefactor of the raw score, at Q/K
width 4, at one, two and four times the schedule's learning rate.

Each run is the raw-score SoS core's recipe at width 4 (reea3d6436271, h_t(s), best 19.762) with
``model.attention_score_prefactor: softplus`` on the Mosaic ``query_times_softplus`` kernels, and its peak and floor
learning rates multiplied by 1, 2 or 4. At width 64 the softplus prefactor helped as much on the raw score as on
z = softplus(s) (14.097 against z h_t(z)'s 14.095); at width 4 only z h_t(z) has run (rabfbab32a594, 19.463; 18.569
at four times). The prefactor adds no parameter and a rate changes no initialization, so each run's initial parameters
are the partner's, bit for bit. Head RMSNorm's max-abs prescale and gradient-event capture are the partner's.
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
PARTNER = "reea3d6436271"
PARTNER_SHA256 = "05b52e20f935438b9eeed180d247d5355828a87de4797f277d274176e3e5b67a"
SCALES = (1.0, 2.0, 4.0)
RATES = ("learning_rate", "min_learning_rate")
# The Mosaic transformed query-polynomial kernels' qualification, Q/K width 4 included.
GPU_TYPES = ("H100", "B200")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_config():
    path = RUNS / PARTNER / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError(f"Pinned raw SoS partner {PARTNER} changed")
    return load_config(path)


def recipe(scale):
    if scale not in SCALES:
        raise ValueError(f"The learning-rate scales are {SCALES}, not {scale}")
    partner = partner_config()
    return replace(partner, model=replace(partner.model, attention_score_prefactor="softplus"),
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
        raise ValueError(f"Settings differ from {PARTNER} under the softplus prefactor: {changed}")


def check_initialization(config):
    """Every partner array, bit for bit."""
    check_recipe(config)
    actual = key_width_acceptance.flat_initial_parameters(config)
    original = key_width_acceptance.flat_initial_parameters(partner_config())
    if set(actual) != set(original) or any(
            actual[path].shape != value.shape or not np.array_equal(actual[path], value)
            for path, value in original.items()):
        raise RuntimeError(f"Initial parameters differ from the partner {PARTNER}'s")
    return {"partner_run_id": PARTNER, "all_partner_arrays_identical": True, "score_prefactor": "softplus"}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_initialization=check_initialization,
                   required_backend="mosaic",
                   cpu_tests=(*shared.cpu_tests,
                              "experiments/transformer_pp/tests/test_narrow_prefactor_sos_acceptance.py"))


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
