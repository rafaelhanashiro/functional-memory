"""The degree-eight tanh query polynomial under exp or softplus of the raw score at Q/K width 64: the plain query
run's recipe with the prefactor, max-abs head RMSNorm and gradient-event capture alone changed."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import query_prefactor_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config, save_config

RUNS = {"rdb8deaab4c91": ("exp", "query-m8-exp-prefactor-raw-input-w-decay"),
        "r7aa6c853dafd": ("softplus", "query-m8-softplus-prefactor-raw-input-w-decay")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("prefactor", adapter.PREFACTORS)
def test_each_recipe_changes_the_prefactor_and_new_run_settings_alone(prefactor):
    config, partner = adapter.recipe(prefactor), adapter.partner_config()
    assert changed(config, partner) <= ({"model.attention_score_prefactor", "model.attention_head_rms_prescale"}
                                        | {f"train.{name}" for name in adapter.CAPTURE_POLICY})
    assert config.model.attention_score_prefactor == prefactor and config.model.attention_score_transform == "identity"
    assert config.model.attention_coefficient_parameterization == "direct" and config.model.attention_degree == 8
    assert config.model.qk_head_dim == 64 and config.model.attention_backend == "mosaic"
    assert config.optim.learning_rate == partner.optim.learning_rate and config.train.seed == partner.train.seed
    adapter.check_recipe(config)


@pytest.mark.parametrize("run", RUNS)
def test_the_allocated_runs_are_the_recipes(run):
    prefactor, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    assert changed(config, adapter.recipe(prefactor)) == set()
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    config = adapter.recipe("exp")
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=2 * config.optim.learning_rate)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, attention_score_prefactor="sigmoid"))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


def test_the_spec_runs_the_mosaic_prefactor_kernels():
    from scaffolds.transformer.acceptance import test_paths

    config = adapter.recipe("exp")
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic" and spec.check_config is adapter.check_recipe
    gpu = test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert {"scaffolds/transformer/tests/test_mosaic_score_prefactor.py",
            "scaffolds/transformer/tests/test_mosaic_score_transform.py"} <= set(gpu)
    assert "experiments/transformer_pp/tests/test_query_prefactor_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule():
    adapter.require_gpu_type(["--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type(["--gpu-type", "B300"])
    adapter.require_gpu_type(["--gpu-type", "B300", "--runtime-env"])
