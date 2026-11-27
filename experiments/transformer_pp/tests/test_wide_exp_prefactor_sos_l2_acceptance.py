"""e^s h_t(z) at Q/K width 64 with the raw-input L2 coefficient normalization: the degree-eight recipe with the
generator's input and the learning rate alone changed, equal to the plain-input run at each rate but for that input."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import wide_exp_prefactor_sos_l2_acceptance as adapter
from scaffolds.transformer.acceptance import comparison_recipe, test_paths as selected_tests
from scaffolds.transformer.configs import config_to_dict, load_config

LABEL = "query-sos-m8-exp-prefactor-softplus-input-raw-l2-w-decay"
RUNS = {"rd5daec72b8e7": (1.0, LABEL), "r2c608f03d08c": (2.0, f"{LABEL}-lr-double"),
        "rdd827b2c2b7d": (4.0, f"{LABEL}-lr-quadruple")}
L2 = {"model.attention_coefficient_input_source", "model.attention_coefficient_input_normalization"}
DATA = {"data.train_manifest", "data.eval_manifest"}


def changed(first, second):
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("scale", adapter.SCALES)
def test_each_recipe_changes_the_coefficient_input_and_rate_alone(scale):
    config, partner = adapter.recipe(scale), adapter.partner_config()
    rates = {"optim.learning_rate", "optim.min_learning_rate"} if scale != 1.0 else set()
    assert changed(config_to_dict(config), config_to_dict(partner)) == (
        {f"model.{name}" for name in adapter.CHANGES} | rates)
    model = config.model
    assert (model.attention_coefficient_input_source, model.attention_coefficient_input_normalization) == (
        "pre_rmsnorm", "l2")
    assert (model.attention_degree, model.attention_coefficient_input_power, model.qk_head_dim) == (8, 0, 64)
    assert (model.attention_score_transform, model.attention_score_prefactor) == ("softplus", "exp")
    assert model.attention_coefficient_activation == "tanh" and model.attention_backend == "mosaic"
    assert config.optim.learning_rate == pytest.approx(scale * 1e-3) and adapter.scale_of(config) == scale
    adapter.check_recipe(config)


@pytest.mark.parametrize("scale", adapter.SCALES)
def test_each_recipe_is_the_plain_input_run_but_for_the_coefficient_input(scale):
    plain = load_config(adapter.RUNS / adapter.PLAIN_INPUT_RUNS[scale] / "config.yaml")
    difference = changed(comparison_recipe(adapter.recipe(scale)), comparison_recipe(plain))
    assert L2 <= difference <= L2 | DATA


@pytest.mark.parametrize("run", RUNS)
def test_the_allocated_runs_are_the_recipes(run):
    scale, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    assert changed(config_to_dict(config), config_to_dict(adapter.recipe(scale))) == set()
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    config = adapter.recipe(2.0)
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=3e-3)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, qk_norm_eps=2 * config.model.qk_norm_eps))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


def test_the_spec_qualifies_the_l2_input_and_the_prefactor_kernels():
    config = adapter.recipe(4.0)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic" and spec.check_config is adapter.check_recipe
    gpu = selected_tests(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    for path in (*adapter.FEATURE_TESTS, "scaffolds/transformer/tests/test_mosaic_score_prefactor.py",
                 "scaffolds/transformer/tests/test_mosaic_softplus_sos.py"):
        assert path in gpu
    assert set(adapter.FEATURE_TESTS) <= set(spec.cpu_tests)
    assert "experiments/transformer_pp/tests/test_wide_exp_prefactor_sos_l2_acceptance.py" in spec.cpu_tests


def test_the_gpu_type_rule():
    adapter.require_gpu_type(["--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type(["--gpu-type", "B300"])
    adapter.require_gpu_type(["--gpu-type", "B300", "--runtime-env"])
