"""Qualify the degree-eight tanh query polynomial under a prefactor of the raw score at Q/K width 64.

Each run is the plain query polynomial's recipe (r25fffc16c2f3, best 14.323) with W = f(s) p_t(s), f = exp or
softplus of the raw score (``model.attention_score_prefactor``), evaluated over the whole score range on the Mosaic
``query_times_<f>`` kernels, as the degree-eight SoS core under the same prefactors (r6525664e6696, r41662adb22fa;
softplus_sos_acceptance.py). They are the references the split-score query runs (split_score_query_acceptance.py),
which apply e^s to negative scores alone, lack. Each also takes head RMSNorm's max-abs prescale, the same function as
the partner's plain arithmetic, and gradient-event capture, as every new run does. The prefactor adds no parameter,
so each run's initial parameters are the partner's, bit for bit.
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
PARTNER = "r25fffc16c2f3"
PARTNER_SHA256 = "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"
PREFACTORS = ("exp", "softplus")
# Every new run's head RMSNorm evaluation and gradient-event capture.
CHANGES = {"attention_head_rms_prescale": "max_abs"}
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}
# The Mosaic transformed query-polynomial kernels' qualification.
GPU_TYPES = ("H100", "B200")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_config():
    path = RUNS / PARTNER / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError(f"Pinned query partner {PARTNER} changed")
    return load_config(path)


def recipe(prefactor):
    if prefactor not in PREFACTORS:
        raise ValueError(f"The query prefactor runs take {PREFACTORS}, not {prefactor!r}")
    partner = partner_config()
    return replace(partner, model=replace(partner.model, attention_score_prefactor=prefactor, **CHANGES),
                   train=replace(partner.train, **CAPTURE_POLICY))


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
                   required_backend="mosaic",
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_query_prefactor_acceptance.py"))


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
