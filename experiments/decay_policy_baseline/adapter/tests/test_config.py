"""The recipe is the retained softmax reference with only its source and outputs changed."""
from pathlib import Path

import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity
from experiments.decay_policy_baseline.adapter.execution.acceptance import SPEC, WEIGHT_DECAY_POLICY

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "experiments/decay_policy_baseline"
CONTROLS = ROOT / "experiments/key_dimension_taylor/configs"
RUN = "200m-softmax-seed0"


def test_recipe_changes_only_source_and_output_locations():
    candidate = config_to_dict(load_config(RECORD / "configs" / f"{RUN}.yaml"))
    control = config_to_dict(load_config(CONTROLS / f"{RUN}.yaml"))
    assert candidate["train"].pop("source_commit") == "auto"
    assert control["train"].pop("source_commit") == "1a8836a75836671cdf83a3ab7763425a6a88cac7"
    assert candidate["train"].pop("workdir") == f"experiments/decay_policy_baseline/runs/{RUN}"
    assert candidate["train"].pop("checkpoint_dir") == f"experiments/decay_policy_baseline/checkpoints/{RUN}"
    for key in ("workdir", "checkpoint_dir"):
        control["train"].pop(key)
    assert candidate == control


def test_recipe_is_the_softmax_reference_under_the_canonical_decay_policy():
    config = load_config(RECORD / "configs" / f"{RUN}.yaml")
    SPEC.check_config(config)
    assert config.optim.weight_decay_policy == WEIGHT_DECAY_POLICY
    assert config.optim.weight_decay == 0.1
    assert config.model.tie_embeddings is False
    assert model_parameter_count(config.model) == 199_906_304
    assert attention_identity(config)["attention_kernel"] == "softmax"


@pytest.mark.parametrize("recipe,message", [
    ("key_dimension_taylor/configs/200m-m002-seed0.yaml", "exact-softmax reference"),
    ("gla_key_dimension/configs/200m-dk064-gla-seed0.yaml", "exact-softmax reference"),
])
def test_acceptance_rejects_other_mechanisms(recipe, message):
    with pytest.raises(ValueError, match=message):
        SPEC.check_config(load_config(ROOT / "experiments" / recipe))
