"""Qualify softplus composed with query SoS, optionally multiplied by its input.

The SoS core has degree eight or twelve. The actual polynomial has degree m
for h(z), m+1 for z*h(z), z=softplus(s). Each pair starts from the completed
query-SoS run's exact parameters at its degree: h=T_m. The prefactor changes no learned parameter and is
represented by a structural zero prepended to the expanded coefficients. Mosaic
evaluates the resulting polynomial and differentiates through the full composition.
"""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Exact recipe and initial-parameter fingerprints retained by each completed
# raw-score SoS control. The prefactor must not add any learned parameter.
CONTROLS = {
    8: {
        "run_id": "reeefbcc19abf",
        "config_sha256": "7d3fca30ff8fb22520f5bdb6df9bffc3a962b2dc5ab539436a17b4d2b0e3d326",
        "parameter_sha256": "bacacd7eeee7c67dd4d5deb55ebfeef925ff4631a188d50d2e45524b6618c74d",
        "parameter_count": 200_150_806,
    },
    12: {
        "run_id": "r3d52b7ab43d1",
        "config_sha256": "9b76b2bf414442b851519a08f5bcf193598d96381f055bb1de31dbd238f6c0fa",
        "parameter_sha256": "9ece005626c3c8ecc0a53f69547fc9f393b1fee1b0dd83b71a0d20f38471ce16",
        "parameter_count": 200_905_126,
    },
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}
CPU_TESTS = ("scaffolds/transformer/tests/test_softplus_sos.py",
             "experiments/transformer_pp/tests/test_softplus_sos_acceptance.py")
GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_score_transform.py",
             "scaffolds/transformer/tests/test_mosaic_softplus_sos.py")


def control_config(core_degree=8):
    control = CONTROLS[core_degree]
    path = RUNS / control["run_id"] / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != control["config_sha256"]:
        raise ValueError(f"Pinned SoS control recipe {control['run_id']} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    power = config.model.attention_coefficient_input_power
    if (config.model.attention_degree, power) not in ((8, 0), (9, 1), (12, 0), (13, 1)):
        raise ValueError("Runs require a degree-eight or degree-twelve SoS core: "
                         "(total degree, input power) = (8,0), (9,1), (12,0) or (13,1)")
    core_degree = config.model.attention_degree - power
    control = CONTROLS[core_degree]
    actual, expected = config_to_dict(config), config_to_dict(control_config(core_degree))
    expected["model"].update(attention_score_transform="softplus",
                             attention_degree=core_degree + power,
                             attention_coefficient_input_power=power)
    expected["train"].update(CAPTURE_POLICY)
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Unrelated settings differ from SoS control {control['run_id']}: {sorted(differences)}")


def check_initialization(config):
    check_recipe(config)
    core_degree = config.model.attention_degree - config.model.attention_coefficient_input_power
    control = CONTROLS[core_degree]
    # Parameter initialization is backend independent; use the host reference.
    facts = initialization_facts(replace(config, model=replace(config.model, attention_backend="blocked")))
    expected = {key: control[key] for key in ("parameter_count", "parameter_sha256")}
    if facts != expected:
        raise RuntimeError(f"Initial parameters differ from the degree-{core_degree} SoS control: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": control["run_id"], "control_parameter_sha256": control["parameter_sha256"],
            "control_parameter_count": control["parameter_count"],
            "sos_core_degree": core_degree, "polynomial_degree": config.model.attention_degree,
            "input_power": config.model.attention_coefficient_input_power,
            "initial_kernel": ("z*" if config.model.attention_coefficient_input_power else "") + f"T_{core_degree}(z)",
            "score_transform": "softplus", "all_initial_parameters_match_control": True}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS),
                   gpu_tests=(*shared.gpu_tests, *GPU_TESTS), required_backend="mosaic",
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in ("H100", "B200"):
        raise ValueError("These Mosaic softplus-SoS runs qualify and train on H100 or B200 only")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
