"""Qualify degree-eight direct (not SoS) polynomials of z = softplus(s), alone or times z.

Static runs are the static degree-eight run r9c69dd7cfc8d (learned per-head c_0..c_8, Taylor start, factorial
storage) with the softplus transform, W = p(z), or with it and the softplus prefactor, W = z p(z) =
softplus(s) p(softplus(s)); the constant c_0 stays learnable (no structural zero), so p starts at T_8(z). The query
run is ra3b54e2489f8, W = p_t(z) with direct tanh query coefficients starting at T_8(z) - 1 (b_0 = 0), with the
softplus prefactor, W = z p_t(z), starting at z (T_8(z) - 1). Every run adds gradient-event capture, which the query
partner already has. The prefactor is the general score prefactor f = softplus with phi = softplus (the Mosaic
transformed kernels, every query's coefficients equal for static ones); it adds no parameter, so every initial array
is the partner's: the pinned count and SHA256, computed on CPU with ``acceptance.initialization_facts``.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, comparison_recipe, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}
# Each partner, by coefficient conditioning: recipe SHA256, the model changes its runs may make (each a field and the
# values it may take; the first is the partner's), and its seed-42 initial parameters (count, SHA256).
PARTNERS = {
    "static": {
        "run_id": "r9c69dd7cfc8d",
        "config_sha256": "f17be1f96b471b36d6c0369b4e4e7dc1ac93b0960376c920b8b40972760c9b8f",
        "model": {"attention_score_transform": ("softplus",), "attention_score_prefactor": ("none", "softplus")},
        "initialization": (198_455_476, "ccf7631fc9802097a27a69ccedf70790c81f960fd77f7ceb238cde41f5b2caf7"),
    },
    "query": {
        "run_id": "ra3b54e2489f8",
        "config_sha256": "adcdb8b7e740ccba18fb7f2bb58fa61fbfdbd14b8012af7d340084b9230387e8",
        "model": {"attention_score_prefactor": ("softplus",)},
        "initialization": (200_150_806, "fc52bfcba4c07f72a8e53a4af86a6964b085d3f127b92e2f7c787fd756b521b0"),
    },
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
GPU_TYPES = ("H100", "B200")
CPU_TESTS = ("experiments/transformer_pp/tests/test_softplus_direct_acceptance.py",
             "scaffolds/transformer/tests/test_softplus_zero_constant.py")
GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_score_transform.py",
             "scaffolds/transformer/tests/test_mosaic_softplus_zero_constant.py")
PREFACTOR_CPU_TESTS = ("scaffolds/transformer/tests/test_score_prefactor.py",)
PREFACTOR_GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_score_prefactor.py",)


def partner_key(config) -> str:
    key = config.model.attention_coefficient_conditioning
    if key not in PARTNERS:
        raise ValueError(f"No direct softplus partner for coefficient conditioning {key!r}")
    return key


def partner_config(key):
    if key not in PARTNERS:
        raise ValueError(f"No direct softplus partner {key!r}")
    partner = PARTNERS[key]
    path = RUNS / partner["run_id"] / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != partner["config_sha256"]:
        raise ValueError(f"Pinned direct softplus partner recipe {partner['run_id']} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    key = partner_key(config)
    partner = PARTNERS[key]
    actual, expected = comparison_recipe(config), comparison_recipe(partner_config(key))
    for name, values in partner["model"].items():
        if actual["model"].get(name) not in values:
            raise ValueError(f"Direct softplus {key} runs take model.{name} in {values}, not "
                             f"{actual['model'].get(name)!r}")
        expected["model"][name] = actual["model"][name]
    if config.model.attention_score_transform != "softplus":
        raise ValueError("Direct softplus runs read z = softplus(s)")
    expected["train"].update(CAPTURE_POLICY)
    for recipe in (actual, expected):
        for name in EXECUTION_KEYS:
            recipe["train"].pop(name, None)
    if actual != expected:
        differences = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                             for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                             if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from direct softplus partner {partner['run_id']}: {differences}")


def initial_kernel(config) -> str:
    start = "T_8(z)" if config.model.attention_coefficient_conditioning == "static" else "(T_8(z) - 1)"
    return ("z*" if config.model.attention_score_prefactor == "softplus" else "") + start


def check_initialization(config):
    check_recipe(config)
    key = partner_key(config)
    partner = PARTNERS[key]
    count, sha256 = partner["initialization"]
    facts = initialization_facts(replace(config, model=replace(config.model, attention_backend="blocked")))
    if facts != {"parameter_count": count, "parameter_sha256": sha256}:
        raise RuntimeError(f"Initial parameters differ from direct softplus partner {partner['run_id']}: {facts}")
    return {**matches_softmax_backbone_initialization(config), "partner_run_id": partner["run_id"],
            "partner_parameter_count": count, "partner_parameter_sha256": sha256,
            "all_initial_parameters_match_partner": True, "score_transform": "softplus",
            "score_prefactor": config.model.attention_score_prefactor, "initial_kernel": initial_kernel(config)}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    prefactor = config.model.attention_score_prefactor != "none"
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS, *(PREFACTOR_CPU_TESTS if prefactor else ())),
                   gpu_tests=(*shared.gpu_tests, *GPU_TESTS, *(PREFACTOR_GPU_TESTS if prefactor else ())),
                   required_backend="mosaic", check_config=check_recipe, check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"Direct softplus Mosaic runs qualify and train on {' or '.join(GPU_TYPES)} only, "
                         f"not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
