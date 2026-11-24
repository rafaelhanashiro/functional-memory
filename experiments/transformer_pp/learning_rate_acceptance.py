"""Qualify learning-rate scalings of completed recipes: at Q/K width 4 the exponential and z h_t(z) with a
degree-sixteen SoS core, and at Q/K width 64 the degree-eight SoS core under a prefactor of the raw score.

Each run is a completed run's recipe with ``optim.learning_rate`` and ``optim.min_learning_rate`` multiplied by one of
its partner's scales (SCALES): the unnormalized exponential at width 4 (r2739c80cc032, key_width_acceptance) at half,
double and four times; z h_t(z), z = softplus(s), with a degree-sixteen query SoS core at width 4 (r2e7f5617fdf2,
shifted_sos_acceptance) also at eight times; and f(s) h_t(s), f = exp, softplus or sigmoid, with a degree-eight query
SoS core at width 64 (r6525664e6696, r41662adb22fa, r8f56a5b7e525; softplus_sos_acceptance) at double and four times.
Warmup, horizon, clipping, seed, data order and every model setting are the partner's, so a run starts from its
partner's parameters and differs from it by the step size alone; a partner that predates head RMSNorm's max-abs
prescale takes it, as every new run does (the same function as its plain arithmetic). The partner's adapter qualifies
the run, on the recipe at the partner's rate: tests, backend, initialization checks and GPU types do not read the
learning rate.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
import math
from pathlib import Path
import sys

from experiments.transformer_pp import key_width_acceptance, shifted_sos_acceptance, softplus_sos_acceptance
from scaffolds.transformer.acceptance import comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Each partner: its run, recipe SHA256 and adapter.
PARTNERS = {
    "exponential": ("r2739c80cc032", "73ecd43b5132131762344ef4aba26c9b84d688f5d96dd8354f9b28646aefa4e1",
                    key_width_acceptance),
    "shifted_sos": ("r2e7f5617fdf2", "53e71c960ce46cbe21b420a810610ff8a80d4ec9484d0d8f4ca218b6a65473bf",
                    shifted_sos_acceptance),
    # At Q/K width 64, the degree-eight SoS core times a prefactor of the raw score.
    "exp_prefactor_sos": ("r6525664e6696", "6a67ae627625f3ba560cd7169835848db9a330270c4f18d80156c7964d98e856",
                          softplus_sos_acceptance),
    "softplus_prefactor_sos": ("r41662adb22fa", "cdbb147dcb8b3b591a75590da398cfa3805880e770866471b45f6dadaa01599c",
                               softplus_sos_acceptance),
    "sigmoid_prefactor_sos": ("r8f56a5b7e525", "d0d05d4e9228f50190231025306db861cf0f9c5977a860aa299080d0396eb579",
                              softplus_sos_acceptance),
}
# Each partner's scales: z h_t(z) still improved from double to four times, and the exponential did not.
SCALES = {"exponential": (0.5, 2.0, 4.0), "shifted_sos": (0.5, 2.0, 4.0, 8.0), "exp_prefactor_sos": (2.0, 4.0),
          "softplus_prefactor_sos": (2.0, 4.0), "sigmoid_prefactor_sos": (2.0, 4.0)}
# The Mosaic transformed query-polynomial kernels of the prefactor partners admit these GPUs alone.
MOSAIC_TRANSFORMED_GPUS = ("H100", "B200")
# The schedule's peak and floor scale together; warmup and horizon stay.
RATES = ("learning_rate", "min_learning_rate")
# New runs' evaluation of head RMSNorm. The comparison leaves it to the run (acceptance.comparison_recipe), and the
# partner's adapter to its kernel: the exponential takes either evaluation, the row-power shift only this one.
PRESCALE = "max_abs"
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_key(config) -> str:
    """The exponential has no degree; a prefactor of the raw score names its partner; z h_t(z) has neither."""
    model = config.model
    if model.attention_degree is None:
        return "exponential"
    if model.attention_score_prefactor == "none":
        return "shifted_sos"
    key = f"{model.attention_score_prefactor}_prefactor_sos"
    if key not in PARTNERS:
        raise ValueError(f"No learning-rate partner takes the {model.attention_score_prefactor} prefactor")
    return key


def partner_config(key):
    run, digest, _ = PARTNERS[key]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned learning-rate partner {run} changed")
    return load_config(path)


def recipe(key, scale):
    if scale not in SCALES[key]:
        raise ValueError(f"The {key} learning-rate scales are {SCALES[key]}, not {scale}")
    partner = partner_config(key)
    return replace(partner, model=replace(partner.model, attention_head_rms_prescale=PRESCALE), optim=replace(
        partner.optim, **{name: getattr(partner.optim, name) * scale for name in RATES}))


def scale_of(config) -> float:
    """The recipe's scale: its peak rate over its partner's."""
    key = partner_key(config)
    ratio = config.optim.learning_rate / partner_config(key).optim.learning_rate
    for scale in SCALES[key]:
        if math.isclose(ratio, scale, rel_tol=1e-9):
            return scale
    raise ValueError(f"optim.learning_rate is {ratio:g} times its partner's; the scales are {SCALES[key]}")


def at_partner_rate(config):
    """The recipe with its partner's schedule, as the partner's adapter takes it."""
    partner = partner_config(partner_key(config))
    return replace(config, optim=replace(config.optim, **{name: getattr(partner.optim, name) for name in RATES}))


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
        raise ValueError(f"Settings differ from {PARTNERS[key][0]} at the scaled learning rate: {changed}")
    PARTNERS[key][2].check_recipe(at_partner_rate(config))


def gpu_types(config) -> tuple:
    """The partner's kernels train only on the GPUs they are qualified on."""
    key = partner_key(config)
    if key == "exponential":
        return key_width_acceptance.gpu_types(key_width_acceptance.partner_key(config), config.model.qk_head_dim)
    return shifted_sos_acceptance.GPU_TYPES if key == "shifted_sos" else MOSAIC_TRANSFORMED_GPUS


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    partner = PARTNERS[partner_key(config)][2].experiment_spec(at_partner_rate(config), entry_module=entry_module)

    def check_initialization(current):
        # The partner's pins and comparisons: initial parameters do not read the optimizer.
        return partner.check_initialization(at_partner_rate(current))

    return replace(partner, check_config=check_recipe, check_initialization=check_initialization,
                   cpu_tests=(*partner.cpu_tests, "experiments/transformer_pp/tests/test_learning_rate_acceptance.py"))


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
        raise ValueError(f"This learning-rate run qualifies and trains on {' or '.join(allowed)} only, "
                         f"not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
