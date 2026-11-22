"""Recipe and numerical qualification of the bounded-gate polynomial comparisons."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.transformer_pp import even_gate_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = Path(adapter.__file__).parent / "runs"
# Allocated runs: (score, sigmoid bias initialization, gate learning-rate rule).
ALLOCATED = {
    "rf1185ae8a89c": ("rms_norm_cosine", "zero", "constant"),
    "r4ec7b748d430": ("rms_norm", "zero", "constant"),
    "r2bb06d5ee682": ("rms_norm", "logit_taylor", "constant"),
    "r4546c6fbc1a8": ("rms_norm_cosine", "zero", "two_over_order"),
}


@pytest.mark.parametrize(("score", "bias"), adapter.VARIANTS)
def test_accepts_the_three_recipes_and_rejects_an_unmatched_variant(score, bias):
    config = adapter.recipe(score, bias)
    adapter.check_recipe(config)
    with pytest.raises(ValueError):
        adapter.check_recipe(replace(config, model=replace(config.model, attention_coefficient_scale="factorial")))
    with pytest.raises(ValueError):
        adapter.recipe("rms_norm_cosine", "logit_taylor")


def test_pilot_detects_a_nonzero_constant_and_an_out_of_bounds_cosine():
    config = adapter.recipe("rms_norm_cosine", "zero")
    coefficients = np.zeros((15, 14, 9))
    coefficients[..., adapter.ORDER] = .5
    row = {"score_min": -1.001, "score_max": 1.001}
    validation = {"attention_diagnostics": {
        "coefficients": {"effective_summary": {"min": coefficients, "max": coefficients}}, "layers": [row]}}
    assert adapter.check_pilot(config, {}, validation)["max_effective_gate"] == .5
    coefficients[..., 0] = .25
    with pytest.raises(RuntimeError, match="constant"):
        adapter.check_pilot(config, {}, validation)
    coefficients[..., 0] = 0
    row["score_max"] = 1.05
    with pytest.raises(RuntimeError, match="cosine"):
        adapter.check_pilot(config, {}, validation)


@pytest.mark.parametrize("run_id", ALLOCATED)
def test_each_allocated_recipe_is_its_variant(run_id):
    config = load_config(RUNS / run_id / "config.yaml")
    adapter.check_recipe(config)
    assert (config.model.qk_norm_type, config.model.attention_coefficient_sigmoid_bias_initial,
            config.optim.coefficient_order_learning_rate) == ALLOCATED[run_id]


def test_the_order_rate_variant_changes_only_the_gate_learning_rates():
    constant, scaled = (config_to_dict(adapter.recipe("rms_norm_cosine", "zero", rate)) for rate in adapter.ORDER_RATES)
    assert {(section, name) for section in constant for name in constant[section]
            if constant[section][name] != scaled[section][name]} == {("optim", "coefficient_order_learning_rate")}
    with pytest.raises(ValueError, match="learning-rate rule"):
        adapter.recipe("rms_norm_cosine", "zero", "inverse")
