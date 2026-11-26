"""The pure tanh square-factor SoS core at Q/K width 4 under the softplus and exp prefactors, with and without raw-input
L2: r70e98e1944d5's recipe at four times its rate with the prefactor and coefficient input alone changed."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import pure_tanh_sos_prefactor_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_tests
from scaffolds.transformer.configs import config_to_dict, load_config

LABEL = "query-pure-tanh-sos-core-m16"
RUNS = {"rab2bd658a26f": ("softplus_l2", f"{LABEL}-softplus-times-input-total-m17-rope-w-decay-dk4-lr-quadruple"),
        "r7d5720ce44c8": ("exp_l2", f"{LABEL}-exp-prefactor-softplus-input-rope-w-decay-dk4-lr-quadruple"),
        "re92a9fc818e5": ("softplus_plain",
                          f"{LABEL}-softplus-times-input-total-m17-rope-w-decay-dk4-plain-input-lr-quadruple"),
        "rdd3a1dfc453f": ("exp_plain",
                          f"{LABEL}-exp-prefactor-softplus-input-rope-w-decay-dk4-plain-input-lr-quadruple")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("variant", adapter.VARIANTS)
def test_each_recipe_changes_the_prefactor_input_and_rate_alone(variant):
    config, partner = adapter.recipe(variant), adapter.partner_config()
    assert changed(config, partner) == ({f"model.{name}" for name in adapter.VARIANTS[variant]}
                                        | {"optim.learning_rate", "optim.min_learning_rate"})
    model = config.model
    assert model.attention_coefficient_activation == "pure_tanh" and model.qk_head_dim == 4
    assert model.attention_degree - model.attention_coefficient_input_power == 16
    assert (model.attention_score_prefactor == "exp") == variant.startswith("exp")
    assert (model.attention_coefficient_input_normalization == "none") == variant.endswith("plain")
    assert (model.attention_backend, model.attention_polynomial_shift) == ("fused", "row_power")
    assert config.optim.learning_rate == pytest.approx(4e-3) and config.optim.min_learning_rate == pytest.approx(4e-4)
    assert adapter.variant_of(config) == variant
    adapter.check_recipe(config)


@pytest.mark.parametrize("run", RUNS)
def test_the_allocated_runs_are_the_recipes(run):
    variant, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    assert changed(config, adapter.recipe(variant)) == set()
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    config = adapter.recipe("exp_plain")
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=2e-3)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, qk_norm_eps=2 * config.model.qk_norm_eps))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)
    with pytest.raises(ValueError):
        adapter.recipe("sigmoid_l2")


def test_the_spec_trains_on_the_row_power_shifted_kernel():
    config = adapter.recipe("exp_l2")
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "fused" and spec.check_config is adapter.check_recipe
    assert selected_tests(spec, config_to_dict(config), resolved_backend="fused", cpu=False)
    assert "experiments/transformer_pp/tests/test_pure_tanh_sos_prefactor_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule():
    adapter.require_gpu_type(["--gpu-type", "B300"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type(["--gpu-type", "H100"])
    adapter.require_gpu_type(["--gpu-type", "H100", "--runtime-env"])
