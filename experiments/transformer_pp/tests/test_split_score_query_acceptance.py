"""The degree-eight query polynomial with direct coefficients and the SoS runs' first three negative-score branches:
its control's recipe with the branch, the blocked causal strips and the max-abs prescale alone."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import split_score_query_acceptance as adapter
from scaffolds.transformer.configs import load_config, save_config

# run: (branch, label)
RUNS = {
    "r0d95b6ee7e66": ("exp_times_sos", "query-m8-dk64-w-decay-negative-exp-times-query"),
    "r19c58fc28b21": ("exp_times_origin", "query-m8-dk64-w-decay-negative-exp-times-origin"),
    "rdd7def8722ca": ("positive_quartic", "query-m8-dk64-w-decay-negative-positive-quartic"),
}


@pytest.mark.parametrize("branch", adapter.BRANCHES)
def test_a_recipe_is_the_query_control_with_the_branch(branch):
    config, partner = adapter.recipe(branch), adapter.partner_config()
    assert config.model.attention_negative_score_branch == branch
    assert config.model.attention_coefficient_parameterization == partner.model.attention_coefficient_parameterization
    assert partner.model.attention_coefficient_parameterization == "direct"
    assert (config.model.attention_backend, config.model.attention_blocked_implementation) == ("blocked", "strips")
    assert config.model.attention_head_rms_prescale == "max_abs" and config.model.qk_head_dim == 64
    assert config.optim == partner.optim and config.data == partner.data
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))


@pytest.mark.parametrize("run", RUNS)
def test_an_allocated_run_is_its_recipe(run):
    branch, label = RUNS[run]
    path = adapter.PARTNER.parents[1] / run / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert config.model.attention_negative_score_branch == branch
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    for branch in ("none", "sos_quartic", "signed_quartic"):
        with pytest.raises(ValueError):
            adapter.recipe(branch)
    config = adapter.recipe("exp_times_sos")
    for other in (adapter.partner_config(), replace(config, optim=replace(config.optim, learning_rate=0.002)),
                  replace(config, model=replace(config.model, attention_coefficient_parameterization="sum_of_squares",
                                                attention_coefficient_scale="none",
                                                attention_coefficient_amplitude_scale="none"))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


@pytest.mark.parametrize("branch", adapter.BRANCHES)
def test_initial_parameters_are_the_controls(branch):
    facts = adapter.check_initialization(adapter.recipe(branch))
    assert facts["all_control_arrays_identical"]
    assert bool(facts["added_parameters"]) == (branch == "positive_quartic")


def test_the_spec_and_gpu_rule(tmp_path):
    config = adapter.recipe("positive_quartic")
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe and spec.check_initialization is adapter.check_initialization
    assert "scaffolds/transformer/tests/test_split_score.py" in spec.cpu_tests and spec.gpu_tests
    path = tmp_path / "config.yaml"
    save_config(config, path)
    adapter.require_gpu_type([str(path), "--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type([str(path), "--gpu-type", "H100"])
