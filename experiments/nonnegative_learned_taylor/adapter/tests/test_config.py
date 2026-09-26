"""Each recipe is the direct-coefficient normalized recipe with the kernel written as a sum of squares."""
import math
from pathlib import Path

import pytest

from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity
from experiments.nonnegative_learned_taylor.adapter.execution.acceptance import DEGREES, SPEC

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "experiments/nonnegative_learned_taylor"
CONTROLS = ROOT / "experiments/qk_norm_learned_taylor/configs"


def run_name(degree):
    return f"200m-qk-m{degree:03d}-nonnegative-seed0"


@pytest.mark.parametrize("degree", DEGREES)
def test_recipes_change_only_the_parameterization_and_output_locations(degree):
    candidate = config_to_dict(load_config(RECORD / "configs" / f"{run_name(degree)}.yaml"))
    control = config_to_dict(load_config(CONTROLS / f"200m-qk-m{degree:03d}-normalized-seed0.yaml"))
    assert candidate["model"].pop("attention_coefficient_parameterization") == "sum_of_squares"
    assert control["model"].pop("attention_coefficient_parameterization") == "direct"
    assert candidate["train"].pop("workdir") == f"experiments/nonnegative_learned_taylor/runs/{run_name(degree)}"
    assert candidate["train"].pop("checkpoint_dir") == (
        f"experiments/nonnegative_learned_taylor/checkpoints/{run_name(degree)}")
    for key in ("workdir", "checkpoint_dir"):
        control["train"].pop(key)
    assert candidate == control
    assert candidate["train"]["source_commit"] == "auto"
    assert candidate["optim"]["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"


@pytest.mark.parametrize("degree", DEGREES)
def test_recipes_pass_acceptance_and_keep_the_parameter_count(degree):
    config = load_config(RECORD / "configs" / f"{run_name(degree)}.yaml")
    SPEC.check_config(config)
    model = config.model
    heads = model.num_layers * model.num_heads
    assert model_parameter_count(model) == 199_906_304 + heads * (degree + 1) + heads
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial_sum_of_squares"
    assert identity["attention_output_normalization"] == "positive_row_sum"
    assert "attention_signed" not in identity
    assert identity["qk_norm_initial_scale"] == math.sqrt(64)
    assert len(taylor_square_factors(degree)) == degree + 1


def test_acceptance_rejects_other_recipes():
    for name, message in (("200m-qk-m002-normalized-seed0.yaml", "sum-of-squares"),
                          ("200m-qk-m004-unnormalized-seed0.yaml", "sum-of-squares")):
        with pytest.raises(ValueError, match=message):
            SPEC.check_config(load_config(CONTROLS / name))
