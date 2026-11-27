"""Twice the Chinchilla budget for the Q/K width 4 learning-rate sweep's best recipes: each partner's recipe with
derive_recipe's 2x schedule and the partner's rate or half of it, every other setting the partner's."""
from dataclasses import replace
import hashlib
import json

import pytest
import yaml

from experiments.transformer_pp import narrow_key_budget_acceptance as adapter
from experiments.transformer_pp.derive_recipe import derive
from scaffolds.transformer.configs import config_to_dict, load_config

# run: (partner key, scale, label)
RUNS = {
    "r41562d2eac42": ("exponential", 1.0, "exp-head-scale-row-max-dk4-lr-double-2x"),
    "r0303c5fcd9b6": ("exponential", 0.5, "exp-head-scale-row-max-dk4-2x"),
    "rc81683587538": ("shifted_sos", 1.0,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-quadruple-2x"),
    "r0ee4e14367df": ("shifted_sos", 0.5,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-double-2x"),
}
GPUS = {"exponential": ("H100", "B300"), "shifted_sos": ("B200", "B300")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("key", adapter.PARTNERS)
def test_the_schedule_is_derive_recipes_at_twice_the_budget(key):
    partner = yaml.safe_load((adapter.RUNS / adapter.PARTNERS[key][0] / "config.yaml").read_text())
    derived, _ = derive(partner, budget=2)
    for section, values in adapter.SCHEDULE.items():
        assert {name: derived[section][name] for name in values} == values


@pytest.mark.parametrize("key", adapter.PARTNERS)
@pytest.mark.parametrize("scale", adapter.SCALES)
def test_each_recipe_changes_the_schedule_and_rate_alone(key, scale):
    config, partner = adapter.recipe(key, scale), adapter.partner_config(key)
    rates = {"optim.learning_rate", "optim.min_learning_rate"} if scale != 1.0 else set()
    assert changed(config, partner) == {f"{section}.{name}" for section, values in adapter.SCHEDULE.items()
                                        for name in values} | rates
    assert config.optim.learning_rate == pytest.approx(scale * partner.optim.learning_rate)
    assert config.optim.min_learning_rate == pytest.approx(0.1 * config.optim.learning_rate)
    assert config.model.qk_head_dim == 4 and config.model.attention_backend == "fused"
    assert config.train.seed == partner.train.seed == 42
    assert adapter.scale_of(config) == scale and adapter.gpu_types(config) == GPUS[key]
    adapter.check_recipe(config)
    assert changed(adapter.at_partner(config), partner) == set()


def test_the_partners_are_the_sweeps_best_rates():
    assert adapter.partner_config("exponential").optim.learning_rate == pytest.approx(2e-3)
    assert adapter.partner_config("shifted_sos").optim.learning_rate == pytest.approx(4e-3)
    assert adapter.partner_config("shifted_sos").model.attention_polynomial_shift == "row_power"


@pytest.mark.parametrize("run", RUNS)
def test_the_allocated_runs_are_the_recipes(run):
    key, scale, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    assert changed(config, adapter.recipe(key, scale)) == set()
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    config = adapter.recipe("shifted_sos", 1.0)
    for other in (adapter.partner_config("shifted_sos"),
                  replace(config, optim=replace(config.optim, learning_rate=3 * config.optim.learning_rate)),
                  replace(config, optim=replace(config.optim, warmup_steps=455)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, qk_norm_eps=2 * config.model.qk_norm_eps))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


@pytest.mark.parametrize("key", adapter.PARTNERS)
def test_the_spec_is_the_partners_with_this_adapters_checks(key):
    config = adapter.recipe(key, 0.5)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.required_backend == "fused"
    assert "experiments/transformer_pp/tests/test_narrow_key_budget_acceptance.py" in spec.cpu_tests
    assert "experiments/transformer_pp/tests/test_learning_rate_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule(tmp_path):
    from scaffolds.transformer.configs import save_config
    path = tmp_path / "config.yaml"
    save_config(adapter.recipe("shifted_sos", 1.0), path)
    adapter.require_gpu_type([str(path), "--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type([str(path), "--gpu-type", "H100"])
    adapter.require_gpu_type([str(path), "--gpu-type", "H100", "--runtime-env"])
