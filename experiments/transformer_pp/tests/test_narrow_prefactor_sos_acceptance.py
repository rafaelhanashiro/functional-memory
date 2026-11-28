"""softplus(s) h_t(s) at Q/K width 4: the raw-score SoS core's recipe with the softplus prefactor and the learning rate
scaled alone."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import narrow_prefactor_sos_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_tests
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = {"r382ea631f6f4": (1.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-dk4"),
        "r0df2e1783a49": (2.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-dk4-lr-double"),
        "rbb13b7964401": (4.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-dk4-lr-quadruple")}


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
    assert changed(config, partner) == {"model.attention_score_prefactor"} | rates
    model = config.model
    assert (model.attention_score_prefactor, model.attention_score_transform) == ("softplus", "identity")
    assert (model.attention_degree, model.attention_coefficient_input_power, model.qk_head_dim) == (8, 0, 4)
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
    config = adapter.recipe(2.0)
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=3e-3)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, attention_score_prefactor="exp"))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


def test_the_spec_runs_the_mosaic_prefactor_kernels():
    config = adapter.recipe(4.0)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic" and spec.check_config is adapter.check_recipe
    gpu = selected_tests(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert "scaffolds/transformer/tests/test_mosaic_score_prefactor.py" in gpu
    assert "experiments/transformer_pp/tests/test_narrow_prefactor_sos_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule():
    adapter.require_gpu_type(["--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type(["--gpu-type", "B300"])
    adapter.require_gpu_type(["--gpu-type", "B300", "--runtime-env"])
