"""Qualify runs that scale a completed run's learning-rate schedule.

Each run is a partner's recipe with ``optim.learning_rate`` and ``optim.min_learning_rate`` multiplied by one of the
partner's scales. Warmup, horizon, clipping, seed, data order and every model setting are the partner's, and per-group
multipliers such as ``optim.query_coefficient_learning_rate_scale`` stay relative, so every parameter's rate scales;
a run starts from its partner's parameters and differs from it by the step size alone.

- Q/K width 4, at 0.5 and 2: the unnormalized exponential (r2739c80cc032, key_width_acceptance) and z h_t(z),
  z = softplus(s), with a degree-sixteen query SoS core (r2e7f5617fdf2, shifted_sos_acceptance).
- Q/K width 64, at 2 and 4, on ORCD's H100s: the tanh query polynomial at degree 8 with W decay (r25fffc16c2f3), the
  exponential (r6a76e7fbde45), both under the shared runtime checks, and z h_t(z) with a degree-eight core
  (r841f35a24241, softplus_sos_acceptance).

New runs take head RMSNorm's max-abs prescale, and the exponential its row-maximum shift where its partner predates
it: the same functions as the partners' arithmetic. The partner's adapter qualifies a run on its recipe at the
partner's rate and manifest paths: tests, backend, initialization checks and GPU types do not read the learning rate,
and a run on another cluster reads byte copies of its partner's dataset manifests.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
import math
from pathlib import Path
import sys

from experiments.transformer_pp import key_width_acceptance, shifted_sos_acceptance, softplus_sos_acceptance
from scaffolds.transformer.acceptance import attention_spec, comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Each partner: its run, recipe SHA256 and adapter (None: the shared runtime checks alone).
PARTNERS = {
    "exponential": ("r2739c80cc032", "73ecd43b5132131762344ef4aba26c9b84d688f5d96dd8354f9b28646aefa4e1",
                    key_width_acceptance),
    "shifted_sos": ("r2e7f5617fdf2", "53e71c960ce46cbe21b420a810610ff8a80d4ec9484d0d8f4ca218b6a65473bf",
                    shifted_sos_acceptance),
    "query_dk64": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2", None),
    "exponential_dk64": ("r6a76e7fbde45", "599dc40917e12a8308ae41fcb39623e1871e43fbd6d35402692d25f02450eee6", None),
    "softplus_sos_dk64": ("r841f35a24241", "1643696258e0267ecba603c8e11cbfe9228cfcc43d9a9af91341d9b9a094a62b",
                          softplus_sos_acceptance),
}
# Each partner's learning-rate scales.
GRID = {"exponential": (0.5, 2.0), "shifted_sos": (0.5, 2.0),
        "query_dk64": (2.0, 4.0), "exponential_dk64": (2.0, 4.0), "softplus_sos_dk64": (2.0, 4.0)}
SCALES = tuple(sorted({scale for scales in GRID.values() for scale in scales}))
# The schedule's peak and floor scale together; warmup and horizon stay.
RATES = ("learning_rate", "min_learning_rate")
# New runs' evaluation of head RMSNorm. The comparison leaves it to the run (acceptance.comparison_recipe), and the
# partner's adapter to its kernel: the exponential takes either evaluation, the row-power shift only this one.
PRESCALE = "max_abs"
# The Q/K width 64 partners read Kennel's manifests; another cluster's runs read byte copies of them.
MANIFEST_SHA256 = {
    "train_manifest": "9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4",
    "eval_manifest": "0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f",
}
# The Q/K width 64 runs' GPU: each kernel's qualification on ORCD (validation README), and H100 for the Mosaic
# softplus-SoS kernels, which softplus_sos_acceptance qualifies on H100 and B200 only.
ORCD_GPU_TYPES = ("H100",)
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def partner_key(config) -> str:
    """At Q/K width 4, the exponential has no degree and z h_t(z) its total degree; at width 64, the exponential, the
    query SoS core and the direct tanh query polynomial."""
    model = config.model
    if model.qk_head_dim == 4:
        return "exponential" if model.attention_degree is None else "shifted_sos"
    if model.attention_degree is None:
        return "exponential_dk64"
    return "softplus_sos_dk64" if model.attention_coefficient_parameterization == "sum_of_squares" else "query_dk64"


def partner_config(key):
    run, digest, _ = PARTNERS[key]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned learning-rate partner {run} changed")
    return load_config(path)


def fresh_settings(key) -> dict:
    """What scaffolds.records create gives a fresh copy of the partner: the max-abs prescale, and the exponential's
    row-maximum shift where the partner predates it."""
    settings = {"attention_head_rms_prescale": PRESCALE}
    if key == "exponential_dk64":
        settings["attention_exp_shift"] = "row_max"
    return settings


def recipe(key, scale):
    if scale not in GRID[key]:
        raise ValueError(f"{PARTNERS[key][0]}'s learning-rate scales are {GRID[key]}, not {scale}")
    partner = partner_config(key)
    return replace(partner, model=replace(partner.model, **fresh_settings(key)), optim=replace(
        partner.optim, **{name: getattr(partner.optim, name) * scale for name in RATES}))


def scale_of(config) -> float:
    """The recipe's scale: its peak rate over its partner's."""
    key = partner_key(config)
    ratio = config.optim.learning_rate / partner_config(key).optim.learning_rate
    for scale in GRID[key]:
        if math.isclose(ratio, scale, rel_tol=1e-9):
            return scale
    raise ValueError(f"optim.learning_rate is {ratio:g} times its partner's; its scales are {GRID[key]}")


def with_partner_manifests(config):
    """The recipe with its partner's manifest paths, once its own are byte copies of the partner's manifests."""
    partner = partner_config(partner_key(config))
    for name, digest in MANIFEST_SHA256.items():
        path, partner_path = getattr(config.data, name), getattr(partner.data, name)
        if str(path) != str(partner_path) and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"data.{name} is not a copy of the partner's dataset manifest")
    return replace(config, data=replace(config.data, **{name: getattr(partner.data, name) for name in MANIFEST_SHA256}))


def at_partner_rate(config):
    """The recipe with its partner's schedule and manifest paths, as the partner's adapter takes it."""
    partner = partner_config(partner_key(config))
    local = with_partner_manifests(config)
    return replace(local, optim=replace(local.optim, **{name: getattr(partner.optim, name) for name in RATES}))


def check_recipe(config):
    validate_config(config)
    key = partner_key(config)
    actual = comparison_recipe(with_partner_manifests(config))
    expected = comparison_recipe(recipe(key, scale_of(config)))
    for entry in (actual, expected):
        for name in EXECUTION_KEYS:
            entry["train"].pop(name, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from {PARTNERS[key][0]} at the scaled learning rate: {changed}")
    if PARTNERS[key][2] is not None:
        PARTNERS[key][2].check_recipe(at_partner_rate(config))


def gpu_types(config) -> tuple:
    """The partner's kernels train only on the GPUs they are qualified on."""
    key = partner_key(config)
    if key == "exponential":
        return key_width_acceptance.gpu_types(key_width_acceptance.partner_key(config), config.model.qk_head_dim)
    if key == "shifted_sos":
        return shifted_sos_acceptance.GPU_TYPES
    return ORCD_GPU_TYPES


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    adapter, local = PARTNERS[partner_key(config)][2], at_partner_rate(config)
    partner = (attention_spec(local, entry_module=entry_module) if adapter is None
               else adapter.experiment_spec(local, entry_module=entry_module))

    def check_initialization(current):
        # The partner's pins and comparisons: initial parameters do not read the optimizer.
        return partner.check_initialization(at_partner_rate(current))

    return replace(partner, check_config=check_recipe,
                   check_initialization=None if partner.check_initialization is None else check_initialization,
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
