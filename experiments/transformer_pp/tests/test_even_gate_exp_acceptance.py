"""The three final recipes, pilot evidence, and device qualification paths."""
from dataclasses import replace

import numpy as np
import pytest

from experiments.transformer_pp import even_gate_exp_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as acceptance_paths
from scaffolds.transformer.configs import config_to_dict


@pytest.mark.parametrize(("score", "bias"), adapter.PARENTS.keys())
def test_hybrid_recipe_keeps_its_matched_control_and_selects_kernel(score, bias):
    config = adapter.recipe(score, bias)
    adapter.check_recipe(config)
    model = config.model
    assert model.attention_coefficient_exp_gate and tuple(model.attention_coefficient_orders) == tuple(range(2, 17, 2))
    assert model.attention_degree == 16 and model.qk_head_dim == 4 and model.qk_norm_head_scale == (score == "rms_norm")
    assert model.attention_coefficient_input_source == "pre_rmsnorm" and model.attention_coefficient_input_normalization == "l2"
    assert model.attention_coefficient_kernel_weight_decay and model.attention_coefficient_scale == "none"
    assert config.train.seed == 42 and config.optim.total_steps == 7570
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    paths = acceptance_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=False)
    assert "scaffolds/transformer/tests/test_even_gate_exp_head.py" in paths
    assert "scaffolds/transformer/tests/test_even_gate_exp_attention.py" in paths
    with pytest.raises(ValueError):
        adapter.check_recipe(replace(config, model=replace(model, attention_coefficient_exp_gate=False)))


def test_pilot_reports_all_nine_gates_and_refuses_invalid_evidence():
    config = adapter.recipe("rms_norm_cosine", "zero")
    lo = np.zeros((15, 14, 17))
    lo[..., adapter.ORDERS] = .5
    high = lo.copy()
    exp = {"min": np.full((15, 14), .45), "max": np.full((15, 14), .55)}
    layer = {"score_min": -.999, "score_max": 1.001}
    validation = {"attention_diagnostics": {"coefficients": {
        "effective_summary": {"min": lo, "max": high}, "exp_gate_summary": exp}, "layers": [layer]}}
    facts = adapter.check_pilot(config, {}, validation)
    assert facts["polynomial_orders"] == list(range(2, 17, 2)) and facts["max_exp_gate"] == .55
    high[..., 1] = .1
    with pytest.raises(RuntimeError, match="odd"):
        adapter.check_pilot(config, {}, validation)
    high[..., 1] = 0
    exp["min"][0, 1] = np.nan
    with pytest.raises(RuntimeError, match="exponential"):
        adapter.check_pilot(config, {}, validation)
    exp["min"][0, 1] = .45
    layer["score_max"] = 1.1
    with pytest.raises(RuntimeError, match="Cosine"):
        adapter.check_pilot(config, {}, validation)
