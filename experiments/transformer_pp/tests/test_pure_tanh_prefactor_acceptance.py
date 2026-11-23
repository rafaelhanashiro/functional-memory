"""The six pure tanh runs at Q/K width 4: the degree-eight tanh query partner's recipe with pure tanh coefficients from
the L2-normalized raw block input (W at zero, beta at atanh(1/2)), the degree, the prefactor, and the row-power shift
on the fused kernel with head RMSNorm's max-abs prescale."""
import hashlib
import json
import math

import pytest

from experiments.transformer_pp import pure_tanh_prefactor_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

# Each allocated run: its label, degree and prefactor.
RUNS = {"rfd18b16513a0": ("query-pure-tanh-m8-dk4", 8, "none"),
        "rfd16ec25202a": ("query-pure-tanh-m16-dk4", 16, "none"),
        "rfa8c0b2f748f": ("softplus-times-query-pure-tanh-m8-dk4", 8, "softplus"),
        "r1a0e7b80c588": ("softplus-times-query-pure-tanh-m16-dk4", 16, "softplus"),
        "r6621b4589a3b": ("exp-times-query-pure-tanh-m8-dk4", 8, "exp"),
        "rf51ad9acc005": ("exp-times-query-pure-tanh-m16-dk4", 16, "exp")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("degree", adapter.DEGREES)
@pytest.mark.parametrize("prefactor", adapter.PREFACTORS)
def test_each_recipe_is_the_partner_with_the_pure_tanh_settings(degree, prefactor):
    config, partner = adapter.recipe(degree, prefactor), adapter.partner_config()
    expected = {f"model.{name}" for name, value in adapter.CHANGES.items() if getattr(partner.model, name) != value}
    expected |= {"model.attention_degree"} if degree != partner.model.attention_degree else set()
    expected |= {"model.attention_score_prefactor"} if prefactor != "none" else set()
    assert changed(config, partner) == expected
    model = config.model
    assert (model.qk_head_dim, model.attention_score_transform, model.attention_coefficient_activation) == (
        4, "identity", "pure_tanh")
    assert model.attention_coefficient_tanh_bias_initial == pytest.approx(math.atanh(0.5))
    assert (model.attention_coefficient_projection_initial_std, model.attention_coefficient_orders) == (0.0, None)
    assert model.attention_coefficient_kernel_weight_decay and config.optim.weight_decay == pytest.approx(0.1)
    assert config.optim.query_coefficient_learning_rate_scale == 1.0 and config.train.seed == partner.train.seed


@pytest.mark.parametrize("run_id", RUNS)
def test_the_allocated_runs_are_the_recipes(run_id):
    label, degree, prefactor = RUNS[run_id]
    path = adapter.PARTNER.parents[1] / run_id / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert adapter.recipe_key(config) == (degree, prefactor)
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_another_change_or_cell_is_refused():
    config = adapter.recipe(8, "exp")
    config.model.attention_coefficient_tanh_bias_initial = 0.0
    with pytest.raises(ValueError):
        adapter.check_recipe(config)
    with pytest.raises(ValueError):
        adapter.recipe(12, "exp")
