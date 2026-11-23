"""Qualify query SoS negative-score branches against the matched Transformer++ recipe."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.attention.split_score import BRANCHES
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config

PARTNER = Path(__file__).parent / "runs/r6525664e6696/config.yaml"
PARTNER_HASH = "6a67ae627625f3ba560cd7169835848db9a330270c4f18d80156c7964d98e856"
TEST_PATH = "scaffolds/transformer/tests/test_split_score.py"


def check_recipe(config):
    validate_config(config)
    branch = config.model.attention_negative_score_branch
    if branch not in BRANCHES[1:]:
        raise ValueError(f"Expected one of the five split-score branches, got {branch!r}")
    if hashlib.sha256(PARTNER.read_bytes()).hexdigest() != PARTNER_HASH:
        raise ValueError("The pinned query SoS partner recipe changed")
    partner = load_config(PARTNER)
    model = replace(partner.model, attention_negative_score_branch=branch, attention_score_prefactor="none",
                    attention_backend="blocked", attention_blocked_implementation="strips",
                    attention_head_rms_prescale="max_abs")
    actual, expected = config_to_dict(config), config_to_dict(replace(partner, model=model))
    for entry in (actual, expected):
        for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group",
                    "wandb_dir", "wandb_mode"):
            entry["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in actual for key in actual[section]
                       if actual[section][key] != expected[section].get(key)]
        raise ValueError(f"Settings differ from the split-score recipe: {differences}")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, cpu_tests=(*shared.cpu_tests, TEST_PATH),
                   gpu_tests=(*shared.gpu_tests, TEST_PATH))


if __name__ == "__main__":
    run_acceptance(partial(experiment_spec, entry_module=__spec__.name))
