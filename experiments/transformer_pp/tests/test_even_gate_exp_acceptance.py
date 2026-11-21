"""The final recipes at degrees eight and sixteen, pilot evidence, and device qualification paths."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.transformer_pp import even_gate_exp_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as acceptance_paths
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = Path(adapter.__file__).parent / "runs"
# Allocated runs: (score, sigmoid bias initialization, degree).
ALLOCATED = {
    "rd77e8919453a": ("rms_norm_cosine", "zero", 16),
    "r9047be7f01c3": ("rms_norm", "zero", 16),
    "r2572a21e0362": ("rms_norm", "logit_taylor", 16),
    "r0404d6d81446": ("rms_norm_cosine", "zero", 8),
    "rc766a892ce73": ("rms_norm", "zero", 8),
    "r6490cdf696da": ("rms_norm", "logit_taylor", 8),
}


@pytest.mark.parametrize("degree", (8, 16))
@pytest.mark.parametrize(("score", "bias"), adapter.PARENTS.keys())
def test_hybrid_recipe_keeps_its_matched_control_and_selects_kernel(score, bias, degree):
    config = adapter.recipe(score, bias, degree)
    adapter.check_recipe(config)
    model = config.model
    orders = tuple(range(2, degree + 1, 2))
    assert model.attention_coefficient_exp_gate and tuple(model.attention_coefficient_orders) == orders
    assert model.attention_degree == degree and model.qk_head_dim == 4 and model.qk_norm_head_scale == (score == "rms_norm")
    assert model.attention_coefficient_input_source == "pre_rmsnorm" and model.attention_coefficient_input_normalization == "l2"
    assert model.attention_coefficient_kernel_weight_decay and model.attention_coefficient_scale == "none"
    assert config.train.seed == 42 and config.optim.total_steps == 7570
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    paths = acceptance_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=False)
    assert "scaffolds/transformer/tests/test_even_gate_exp_head.py" in paths
    assert "scaffolds/transformer/tests/test_even_gate_exp_attention.py" in paths
    with pytest.raises(ValueError):
        adapter.check_recipe(replace(config, model=replace(model, attention_coefficient_exp_gate=False)))
    with pytest.raises(ValueError, match="degree"):
        adapter.recipe(score, bias, 12)


@pytest.mark.parametrize("run_id", ALLOCATED)
def test_each_allocated_recipe_is_its_matched_hybrid(run_id):
    config = load_config(RUNS / run_id / "config.yaml")
    adapter.check_recipe(config)
    model = config.model
    assert (model.qk_norm_type, model.attention_coefficient_sigmoid_bias_initial, model.attention_degree) == (
        ALLOCATED[run_id])


@pytest.mark.parametrize("degree", (8, 16))
def test_pilot_reports_every_gate_and_refuses_invalid_evidence(degree):
    config = adapter.recipe("rms_norm_cosine", "zero", degree)
    orders = list(range(2, degree + 1, 2))
    lo = np.zeros((15, 14, degree + 1))
    lo[..., orders] = .5
    high = lo.copy()
    exp = {"min": np.full((15, 14), .45), "max": np.full((15, 14), .55)}
    layer = {"score_min": -.999, "score_max": 1.001}
    validation = {"attention_diagnostics": {"coefficients": {
        "effective_summary": {"min": lo, "max": high}, "exp_gate_summary": exp}, "layers": [layer]}}
    facts = adapter.check_pilot(config, {}, validation)
    assert facts["polynomial_orders"] == orders and facts["max_exp_gate"] == .55
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
