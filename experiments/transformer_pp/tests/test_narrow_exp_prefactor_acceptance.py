"""The exponential under a softplus or sigmoid prefactor at Q/K width 4: the shifted exponential's recipe with the
prefactor, the blocked causal strips and the max-abs prescale alone changed."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import narrow_exp_prefactor_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_tests
from scaffolds.transformer.configs import config_to_dict, load_config

RUNS = {"r8aead78ffa55": ("softplus", "exp-head-scale-row-max-softplus-prefactor-dk4"),
        "r7530874e8610": ("sigmoid", "exp-head-scale-row-max-sigmoid-prefactor-dk4")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("prefactor", adapter.PREFACTORS)
def test_each_recipe_changes_the_prefactor_and_implementation_alone(prefactor):
    config, partner = adapter.recipe(prefactor), adapter.partner_config()
    assert changed(config, partner) <= {"model.attention_score_prefactor",
                                        *(f"model.{name}" for name in adapter.CHANGES)}
    model = config.model
    assert model.attention_degree is None and model.attention_score_prefactor == prefactor
    assert (model.attention_backend, model.attention_blocked_implementation) == ("blocked", "strips")
    assert model.attention_exp_shift == "row_max" and model.qk_head_dim == 4
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
    config = adapter.recipe("softplus")
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=2 * config.optim.learning_rate)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, attention_score_prefactor="swish"))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


def test_the_spec_trains_on_the_causal_strips():
    config = adapter.recipe("sigmoid")
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "blocked" and spec.check_config is adapter.check_recipe
    assert selected_tests(spec, config_to_dict(config), resolved_backend="blocked", cpu=False)
    assert "experiments/transformer_pp/tests/test_narrow_exp_prefactor_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule():
    adapter.require_gpu_type(["--gpu-type", "B300"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type(["--gpu-type", "A100"])
    adapter.require_gpu_type(["--gpu-type", "A100", "--runtime-env"])
