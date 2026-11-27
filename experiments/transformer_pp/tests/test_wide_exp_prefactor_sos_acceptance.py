"""e^s h_t(z) with a degree-twelve SoS core at Q/K width 64: the degree-twelve z h_t(z) recipe with exp of the raw
score in place of the factor z, the max-abs prescale and the learning rate scaled alone."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import wide_exp_prefactor_sos_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_tests
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = {"r99fe75016666": (1.0, "query-sos-m12-exp-prefactor-softplus-input-w-decay"),
        "rafbba8d1794f": (2.0, "query-sos-m12-exp-prefactor-softplus-input-w-decay-lr-double")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("scale", adapter.SCALES)
def test_each_recipe_changes_the_prefactor_and_rate_alone(scale):
    config, partner = adapter.recipe(scale), adapter.partner_config()
    rates = {"optim.learning_rate", "optim.min_learning_rate"} if scale != 1.0 else set()
    assert changed(config, partner) <= {f"model.{name}" for name in adapter.CHANGES} | rates
    model = config.model
    assert (model.attention_degree, model.attention_coefficient_input_power, model.qk_head_dim) == (12, 0, 64)
    assert (model.attention_score_transform, model.attention_score_prefactor) == ("softplus", "exp")
    assert model.attention_coefficient_parameterization == "sum_of_squares" and model.attention_backend == "mosaic"
    assert config.optim.learning_rate == pytest.approx(scale * 1e-3) and adapter.scale_of(config) == scale
    adapter.check_recipe(config)


@pytest.mark.parametrize("run", RUNS)
def test_the_allocated_runs_are_the_recipes(run):
    scale, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    assert changed(config, adapter.recipe(scale)) == set()
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    config = adapter.recipe(1.0)
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=4e-3)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, qk_norm_eps=2 * config.model.qk_norm_eps))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


def test_the_spec_runs_the_degree_twelve_prefactor_kernels():
    config = adapter.recipe(2.0)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic" and spec.check_config is adapter.check_recipe
    gpu = selected_tests(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert "scaffolds/transformer/tests/test_mosaic_score_prefactor.py" in gpu
    assert "experiments/transformer_pp/tests/test_wide_exp_prefactor_sos_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule():
    adapter.require_gpu_type(["--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type(["--gpu-type", "B300"])
    adapter.require_gpu_type(["--gpu-type", "B300", "--runtime-env"])
