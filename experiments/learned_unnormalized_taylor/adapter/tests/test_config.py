"""Each recipe is its fixed unnormalized Taylor control with every coefficient learned per head."""
from pathlib import Path

import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity
from experiments.learned_unnormalized_taylor.adapter.execution.acceptance import DEGREES, SPEC

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "experiments/learned_unnormalized_taylor"
CONTROLS = ROOT / "experiments/unnormalized_taylor_attention/configs"


@pytest.mark.parametrize("degree", DEGREES)
def test_recipes_change_only_the_learned_coefficients_and_output_locations(degree):
    run = f"200m-m{degree:03d}-seed0"
    candidate = config_to_dict(load_config(RECORD / "configs" / f"{run}.yaml"))
    control = config_to_dict(load_config(CONTROLS / f"{run}.yaml"))
    assert candidate["model"].pop("attention_coefficients") == "learned"
    assert candidate["model"].pop("attention_coefficient_sharing") == "per_head"
    assert control["model"].pop("attention_coefficients") == "fixed"
    control["model"].pop("attention_coefficient_sharing")
    assert candidate["train"].pop("workdir") == f"experiments/learned_unnormalized_taylor/runs/{run}"
    assert candidate["train"].pop("checkpoint_dir") == f"experiments/learned_unnormalized_taylor/checkpoints/{run}"
    for key in ("workdir", "checkpoint_dir"):
        control["train"].pop(key)
    assert candidate == control
    assert candidate["train"]["source_commit"] == "auto"


@pytest.mark.parametrize("degree", DEGREES)
def test_recipes_pass_acceptance_and_add_every_coefficient_per_layer_and_head(degree):
    config = load_config(RECORD / "configs" / f"200m-m{degree:03d}-seed0.yaml")
    SPEC.check_config(config)
    model = config.model
    assert model_parameter_count(model) == 199_906_304 + model.num_layers * model.num_heads * (degree + 1)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial_unnormalized"
    assert identity["attention_coefficient_count"] == 160 * (degree + 1)
    assert identity["attention_signed"] is True


def test_acceptance_rejects_other_recipes():
    control = load_config(CONTROLS / "200m-m002-seed0.yaml")
    with pytest.raises(ValueError, match="learned per-head coefficients"):
        SPEC.check_config(control)
