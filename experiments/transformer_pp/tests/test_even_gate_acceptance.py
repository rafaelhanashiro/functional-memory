"""Recipe and numerical qualification of the three bounded-gate polynomial comparisons."""
from dataclasses import replace

import numpy as np
import pytest

from experiments.transformer_pp import even_gate_acceptance as adapter


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
