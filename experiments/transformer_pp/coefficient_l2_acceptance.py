"""Qualify coefficient-input L2 normalization against pinned m=8 and m=12 controls.

The original runs normalize the gain-applied RMSNorm activation; the raw-input
extension bypasses RMSNorm only for the generator. The softplus-prefactor
extension pins the completed no-L2 prefactor parents at core degrees 8 and 12.
The direct and SoS recipes retain their coefficient mappings, initialized parameters,
optimizer, data stream, and reporting policy.
"""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, comparison_recipe, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import load_config
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
POST_RMSNORM_CONTROLS = {
    12: {
        "direct": ("r7e34dec9dbd5", "7ed059fbcbbb6e87795eb37594591b8d1a8985ea9040ccfe85a64e363981041a"),
        "sum_of_squares": ("r6b17c3463a60", "b75719f3622ac70fded0b5dcac1a283009a217b42a1a712432cf7820b7dea25c"),
    },
}
SOFTPLUS_PREFACTOR_CONTROLS = {
    9: {"sum_of_squares": ("r841f35a24241", "1643696258e0267ecba603c8e11cbfe9228cfcc43d9a9af91341d9b9a094a62b")},
    13: {"sum_of_squares": ("rf637de9b65b1", "22a94ee8cc46fa053cfe2d7789567552888882038d0eb666c705a2762a762903")},
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
RAW_FEATURE_TESTS = "scaffolds/transformer/tests/test_coefficient_raw_input.py"
SOFTPLUS_FEATURE_TESTS = "scaffolds/transformer/tests/test_softplus_sos_raw_input.py"
OWN_TESTS = "experiments/transformer_pp/tests/test_coefficient_l2_acceptance.py"


def control_descriptor(parameterization, degree=8, input_source="post_rmsnorm",
                       score_transform="identity", input_power=0):
    if input_source not in ("post_rmsnorm", "pre_rmsnorm"):
        raise ValueError(f"Unknown coefficient-input source: {input_source!r}")
    if (score_transform, input_power) == ("softplus", 1) and input_source == "pre_rmsnorm":
        controls = SOFTPLUS_PREFACTOR_CONTROLS
    elif (score_transform, input_power) == ("identity", 0):
        controls = POST_RMSNORM_CONTROLS if input_source == "pre_rmsnorm" else CONTROLS
    else:
        raise ValueError("No coefficient-input L2 control for this score transform, prefactor and input source")
    if degree not in controls or parameterization not in controls[degree]:
        raise ValueError(f"No coefficient-input L2 control for degree {degree!r}, {parameterization!r}, {input_source!r}")
    return controls[degree][parameterization]


def control_config(parameterization, degree=8, input_source="post_rmsnorm",
                   score_transform="identity", input_power=0):
    run_id, digest = control_descriptor(parameterization, degree, input_source, score_transform, input_power)
    path = RUNS / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned coefficient-input L2 control recipe {run_id} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    expected = comparison_recipe(control_config(config.model.attention_coefficient_parameterization,
                                            config.model.attention_degree,
                                            config.model.attention_coefficient_input_source,
                                            config.model.attention_score_transform,
                                            config.model.attention_coefficient_input_power))
    expected["model"]["attention_coefficient_input_normalization"] = "l2"
    expected["model"]["attention_coefficient_input_source"] = config.model.attention_coefficient_input_source
    actual = comparison_recipe(config)
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
    core_degree = degree - config.model.attention_coefficient_input_power
    expected = INITIAL_PARAMETERS[core_degree]
    if facts != expected:
        raise RuntimeError(f"Initial parameters differ from the pinned m={core_degree} control: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": control_descriptor(config.model.attention_coefficient_parameterization, degree,
                                                 config.model.attention_coefficient_input_source,
                                                 config.model.attention_score_transform,
                                                 config.model.attention_coefficient_input_power)[0],
            "control_parameter_count": expected["parameter_count"],
            "control_parameter_sha256": expected["parameter_sha256"],
            "all_initial_parameters_match_control": True,
            "coefficient_input_normalization": "l2",
            "coefficient_input_source": config.model.attention_coefficient_input_source}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    feature_tests = (FEATURE_TESTS, RAW_FEATURE_TESTS) if config.model.attention_coefficient_input_source == "pre_rmsnorm" else (FEATURE_TESTS,)
    if config.model.attention_score_transform == "softplus":
        return replace(shared,
                       cpu_tests=(*shared.cpu_tests, *feature_tests, SOFTPLUS_FEATURE_TESTS,
                                  "scaffolds/transformer/tests/test_softplus_sos.py", OWN_TESTS),
                       gpu_tests=(*shared.gpu_tests, *feature_tests, SOFTPLUS_FEATURE_TESTS,
                                  "scaffolds/transformer/tests/test_mosaic_softplus_sos.py"),
                       required_backend="mosaic", check_config=check_recipe,
                       check_initialization=check_initialization)
    return replace(shared, cpu_tests=(*shared.cpu_tests, *feature_tests, OWN_TESTS),
                   gpu_tests=(*shared.gpu_tests, *feature_tests), required_backend="mosaic",
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None):
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
