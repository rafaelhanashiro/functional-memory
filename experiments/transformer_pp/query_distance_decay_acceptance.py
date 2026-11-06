"""Qualify the query-distance ablation against its pinned FoX control recipe."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

import yaml

from scaffolds.transformer.acceptance import (
    attention_spec, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict
from scaffolds.transformer.configs.base import validate_config
from scaffolds.transformer.configs.loader import _update_dataclass

CONTROL = Path(__file__).resolve().parent / "runs/r4a80e28ce0a7/config.yaml"
CONTROL_SHA256 = "cbcbe1ebbf1c370e1b82711a636fddae03e58738822e61f65fe4463f748d8763"
INITIALIZATION_SHA256 = "41222e5822311b7c7828965e5f4e03b72e820cc14f0928d6af073c8649a7b7d7"
PARAMETER_COUNT = 199585066


def control_config() -> Config:
    contents = CONTROL.read_bytes()
    if hashlib.sha256(contents).hexdigest() != CONTROL_SHA256:
        raise ValueError("The pinned FoX control recipe changed")
    raw = yaml.safe_load(contents)
    # This source branch declared a key-generator decay switch. This query-only
    # control has no key generator, and the recorded switch must remain false.
    if raw["model"].pop("attention_coefficient_key_kernel_weight_decay") is not False:
        raise ValueError("Unexpected key-generator decay in the query-only control")
    config = Config()
    _update_dataclass(config, raw, context=str(CONTROL))
    validate_config(config)
    return config


def check_recipe(config: Config) -> None:
    validate_config(config)
    if config.model.attention_forget_gate_mode != "query_distance":
        raise ValueError("This experiment requires query-distance decay")
    actual, expected = config_to_dict(config), config_to_dict(control_config())
    actual["model"]["attention_forget_gate_mode"] = "cumulative"
    # Execution locations and monitoring labels are not scientific settings.
    for recipe in (actual, expected):
        for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode"):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from the FoX control: {sorted(differences)}")


def check_initialization(config: Config) -> dict:
    check_recipe(config)
    facts = initialization_facts(deepcopy(config))
    if (facts["parameter_sha256"] != INITIALIZATION_SHA256
            or facts["parameter_count"] != PARAMETER_COUNT):
        raise RuntimeError(f"Initialization differs from FoX control r4a80e28ce0a7: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": "r4a80e28ce0a7", "control_parameter_sha256": INITIALIZATION_SHA256,
            "control_parameter_count": PARAMETER_COUNT, "all_initial_parameters_match_control": True}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared,
                   cpu_tests=(*shared.cpu_tests,
                              "experiments/transformer_pp/tests/test_query_distance_decay_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
