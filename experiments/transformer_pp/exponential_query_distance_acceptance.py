"""Qualify query-distance gates on softmax and exp against their pinned no-RoPE FoX controls."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, initialization_facts, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# The cumulative FoX control of each kernel: run ID, config SHA256, and its initial parameters' SHA256 and
# count at seed 42, computed once on CPU with initialization_facts on the control recipe. Both decay rules
# read the same gate parameters, so the query-distance variant starts from exactly these parameters.
CONTROLS = {
    "softmax": ("r5adc29a91e3b", "d2c4a2626e9828dbf124a14ab2d0900d394d2e7ac6b75ada83cfcaf34114c08b",
                "d8c5081278741619be5291ed20c77792ca7f623cb7444fdeef0149dba92f8a9f", 198628726),
    "exp": ("r183f08346ddc", "6eb0100dd7d4a8332b6c1ff5ec0c5526c99006cd3013d1d35499c12507e0689e",
            "7632f5fd650f635070b0d3044daa7a39844e1737c91a02b72002fec809e6f0ee", 198642166),
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")


def kernel(config: Config) -> str:
    """Softmax normalizes the decayed exponential; exp leaves it unnormalized."""
    return "softmax" if config.model.attention_normalization else "exp"


def control_config(name: str) -> Config:
    if name not in CONTROLS:
        raise ValueError(f"No FoX control for kernel {name!r}")
    run, digest = CONTROLS[name][:2]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned FoX control recipe {run} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the decay mode (and the causal strips) differ from the kernel's cumulative FoX control."""
    validate_config(config)
    model = config.model
    if model.attention_degree is not None or model.attention_forget_gate_mode != "query_distance":
        raise ValueError("This experiment requires query-distance decay of softmax or the unnormalized exponential")
    if model.attention_blocked_implementation != "strips":
        # The GPU tests qualify this decay through the strips at the production context, not the scan.
        raise ValueError("Query-distance softmax and exp train on the blocked backend's causal strips")
    run = CONTROLS[kernel(config)][0]
    actual, expected = config_to_dict(config), config_to_dict(control_config(kernel(config)))
    actual["model"]["attention_forget_gate_mode"] = "cumulative"
    # The causal strips compute the scan's arithmetic, on which the controls trained.
    actual["model"]["attention_blocked_implementation"] = expected["model"]["attention_blocked_implementation"]
    # Execution locations and monitoring labels are not scientific settings.
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from FoX control {run}: {sorted(differences)}")


def check_initialization(config: Config) -> dict:
    """Every initial parameter equals the FoX control's at the same seed."""
    check_recipe(config)
    run, _, digest, count = CONTROLS[kernel(config)]
    facts = initialization_facts(deepcopy(config))
    if facts["parameter_sha256"] != digest or facts["parameter_count"] != count:
        raise RuntimeError(f"Initialization differs from FoX control {run}: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": run, "control_parameter_sha256": digest, "control_parameter_count": count,
            "all_initial_parameters_match_control": True}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_exponential_query_distance_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
