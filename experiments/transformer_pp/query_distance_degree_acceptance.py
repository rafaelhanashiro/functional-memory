"""Qualify the query-distance gate at degree eight or twelve against its pinned degree-four recipe."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, comparison_recipe, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
CONTROL = RUNS / "r3e26412b2811/config.yaml"
CONTROL_SHA256 = "3fa5cddfc7d5633c6f5919b30bda0a1e9eda145e04e891ee6ad9668f03210a26"
# Cumulative FoX at degree eight. Both decay rules use the same gate parameters, so this recipe's
# initialization at the same seed is the variant's; its data paths are ORCD's, whose manifests hash
# identically to Kennel's, and initialization reads no data. No FoX run exists at degree twelve:
# that degree's reference is this recipe at degree twelve, built at the same seed.
REFERENCE = RUNS / "r9353f554b917/config.yaml"
REFERENCE_SHA256 = "9ff6d8e01a865f759e4a86e192533e46bed2ccd3d20c9a52927d47c2b71a2b65"
REFERENCE_DEGREE = 8
# Parameter count of each degree this experiment runs, the variant's and its reference's.
PARAMETER_COUNT = {8: 200339386, 12: 201093706}
DEGREE_NAMES = {8: "eight", 12: "twelve"}


def _pinned(path: Path, digest: str, name: str) -> Config:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned {name} recipe changed")
    return load_config(path)


def control_config() -> Config:
    return _pinned(CONTROL, CONTROL_SHA256, "degree-four query-distance")


def reference_config(degree: int = REFERENCE_DEGREE) -> Config:
    """Cumulative FoX r9353f554b917's recipe at ``degree``."""
    config = _pinned(REFERENCE, REFERENCE_SHA256, "degree-eight FoX")
    if degree != REFERENCE_DEGREE:
        config.model.attention_degree = degree
        validate_config(config)
    return config


def check_recipe(config: Config) -> None:
    validate_config(config)
    if config.model.attention_degree not in DEGREE_NAMES or config.model.attention_backend != "mosaic":
        raise ValueError("This experiment runs the query-distance gate at degree eight or twelve on the Mosaic "
                         "decay kernels")
    actual, expected = comparison_recipe(config), comparison_recipe(control_config())
    actual["model"]["attention_degree"] = expected["model"]["attention_degree"]
    # The attention implementation is an execution choice: both backends compute the same weights.
    actual["model"]["attention_backend"] = expected["model"]["attention_backend"]
    for recipe in (actual, expected):
        for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode"):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from the degree-four query-distance recipe: {sorted(differences)}")


def check_initialization(config: Config) -> dict:
    """Every initial parameter equals cumulative FoX's at the run's degree, built at the same seed."""
    check_recipe(config)
    degree = config.model.attention_degree
    name = DEGREE_NAMES[degree]
    reference = reference_config(degree)
    if (reference.model.attention_degree, reference.model.attention_forget_gate_mode, reference.train.seed) != (
            degree, "cumulative", config.train.seed):
        raise ValueError(f"The initialization reference must be cumulative FoX at degree {name} and the same seed")
    actual, expected = initialization_facts(deepcopy(config)), initialization_facts(reference)
    if actual != expected or actual["parameter_count"] != PARAMETER_COUNT[degree]:
        raise RuntimeError(f"Initialization differs from FoX r9353f554b917 at degree {name}: {actual} != {expected}")
    facts = {**matches_softmax_backbone_initialization(config),
             "control_run_id": "r3e26412b2811", "initialization_reference_run_id": "r9353f554b917",
             "initialization_reference_parameter_sha256": expected["parameter_sha256"],
             "parameter_count": PARAMETER_COUNT[degree], "all_initial_parameters_match_reference": True}
    if degree != REFERENCE_DEGREE:  # the reference run's recipe at this run's degree
        facts["initialization_reference_degree"] = degree
    return facts


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_query_distance_degree_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
