"""z h_t(z) at Q/K width 4 with a degree-sixteen core of pure tanh square factors: its partner's recipe with the
factor generator's settings alone, and a second run with the partner's learning-rate schedule doubled as well."""
import hashlib
import json
import math

import pytest

from experiments.transformer_pp import pure_tanh_sos_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

RUN = "r70e98e1944d5"
LABEL = "query-pure-tanh-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4"
# Each allocated run: its label and its learning-rate scale.
RUNS = {RUN: (LABEL, 1.0), "r772cbbc3a0d9": (f"{LABEL}-lr-double", 2.0)}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


def test_the_recipe_changes_the_factor_generator_alone():
    config, partner = adapter.recipe(), adapter.partner_config()
    assert changed(config, partner) == {f"model.{name}" for name in adapter.CHANGES}
    model = config.model
    assert (model.attention_coefficient_activation, model.attention_coefficient_parameterization) == (
        "pure_tanh", "sum_of_squares")
    assert model.attention_coefficient_tanh_bias and model.attention_coefficient_tanh_bias_initial == math.atanh(0.5)
    assert model.attention_coefficient_projection_initial_std == 0.0
    assert (model.attention_coefficient_input_source, model.attention_coefficient_input_normalization) == (
        "pre_rmsnorm", "l2")
    # The partner's weight, kernel and evaluation: z h_t(z), z = softplus(s), core sixteen, shifted, at d_K 4.
    assert (model.attention_degree, model.attention_coefficient_input_power, model.attention_score_transform) == (
        17, 1, "softplus")
    assert (model.attention_polynomial_shift, model.attention_head_rms_prescale, model.attention_backend) == (
        "row_power", "max_abs", "fused")
    assert model.qk_head_dim == 4 and model.attention_coefficient_kernel_weight_decay
    assert model.attention_coefficient_amplitude_scale == model.attention_coefficient_scale == "none"
    assert config.train.seed == partner.train.seed == 42 and config.optim == partner.optim


def test_the_doubled_schedule_changes_the_rates_alone():
    config, base = adapter.recipe(2.0), adapter.recipe()
    assert changed(config, base) == {f"optim.{name}" for name in adapter.RATES}
    assert (config.optim.learning_rate, config.optim.min_learning_rate) == (2e-3, 2e-4)
    assert (base.optim.learning_rate, base.optim.min_learning_rate) == (1e-3, 1e-4)
    with pytest.raises(ValueError):
        adapter.recipe(0.5)


@pytest.mark.parametrize("run", sorted(RUNS))
def test_the_allocated_runs_are_the_recipes(run):
    label, scale = RUNS[run]
    path = adapter.PARTNER.parents[1] / run / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert adapter.scale_of(config) == scale and changed(config, adapter.recipe(scale)) == set()
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    with pytest.raises(ValueError):  # the partner itself
        adapter.check_recipe(adapter.partner_config())
    for name, value in (("attention_degree", 13), ("qk_head_dim", 8),
                        ("attention_coefficient_tanh_bias_initial", 0.25),
                        ("attention_coefficient_kernel_weight_decay", False)):
        config = adapter.recipe()
        setattr(config.model, name, value)
        with pytest.raises(ValueError):
            adapter.check_recipe(config)
    config = adapter.recipe()
    config.optim.learning_rate = 2e-3
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_the_run_qualifies_on_the_fused_shifted_kernel():
    spec = adapter.experiment_spec(adapter.recipe(), entry_module=adapter.__name__)
    assert spec.required_backend == "fused" and spec.check_config is adapter.check_recipe
    assert spec.cpu_tests[-1] == "experiments/transformer_pp/tests/test_pure_tanh_sos_acceptance.py"
    assert adapter.GPU_TYPES == ("B200", "B300")
    with pytest.raises(ValueError):
        adapter.main(["config.yaml", "--gpu-type", "H100"])
