"""Qualify coefficient-input L2 normalization against pinned m=8 and m=12 controls.

Only the input to the tanh coefficient generator changes. The direct and SoS
recipes retain their respective coefficient mappings, initialized parameters,
optimizer, data stream, and reporting policy.
"""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config


RUNS = Path(__file__).resolve().parent / "runs"
CONTROLS = {
    8: {
        "direct": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"),
        "sum_of_squares": ("reeefbcc19abf", "7d3fca30ff8fb22520f5bdb6df9bffc3a962b2dc5ab539436a17b4d2b0e3d326"),
    },
    12: {
        "direct": ("r0fceea0627c8", "785e3530949ee2733addde255b679ecf0d295d188fd19fd9ca29ebd9d5a7d2e0"),
        "sum_of_squares": ("r3d52b7ab43d1", "9b76b2bf414442b851519a08f5bcf193598d96381f055bb1de31dbd238f6c0fa"),
    },
}
# At each degree the controls have identical stored arrays (a=0, b=1, random W);
# SoS changes their interpretation, not their initialization. Pin each fingerprint.
INITIAL_PARAMETERS = {
    8: {
        "parameter_count": 200_150_806,
        "parameter_sha256": "bacacd7eeee7c67dd4d5deb55ebfeef925ff4631a188d50d2e45524b6618c74d",
    },
    12: {
        "parameter_count": 200_905_126,
        "parameter_sha256": "9ece005626c3c8ecc0a53f69547fc9f393b1fee1b0dd83b71a0d20f38471ce16",
    },
}
PATH_FIELDS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
               "wandb_group", "wandb_dir", "wandb_mode")
FEATURE_TESTS = "scaffolds/transformer/tests/test_coefficient_input_normalization.py"
OWN_TESTS = "experiments/transformer_pp/tests/test_coefficient_l2_acceptance.py"


def control_config(parameterization, degree=8):
    if degree not in CONTROLS or parameterization not in CONTROLS[degree]:
        raise ValueError(f"No coefficient-input L2 control for degree {degree!r}, {parameterization!r}")
    run_id, digest = CONTROLS[degree][parameterization]
    path = RUNS / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned coefficient-input L2 control recipe {run_id} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    expected = config_to_dict(control_config(config.model.attention_coefficient_parameterization,
                                            config.model.attention_degree))
    expected["model"]["attention_coefficient_input_normalization"] = "l2"
    actual = config_to_dict(config)
    for recipe in (actual, expected):
        for key in PATH_FIELDS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Settings differ from the pinned coefficient-input L2 recipe: {sorted(differences)}")


def check_initialization(config):
    check_recipe(config)
    facts = initialization_facts(replace(config, model=replace(config.model, attention_backend="blocked")))
    degree = config.model.attention_degree
    expected = INITIAL_PARAMETERS[degree]
    if facts != expected:
        raise RuntimeError(f"Initial parameters differ from the pinned m={degree} control: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": CONTROLS[degree][config.model.attention_coefficient_parameterization][0],
            "control_parameter_count": expected["parameter_count"],
            "control_parameter_sha256": expected["parameter_sha256"],
            "all_initial_parameters_match_control": True,
            "coefficient_input_normalization": "l2"}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, FEATURE_TESTS, OWN_TESTS),
                   gpu_tests=(*shared.gpu_tests, FEATURE_TESTS), required_backend="mosaic",
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None):
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
