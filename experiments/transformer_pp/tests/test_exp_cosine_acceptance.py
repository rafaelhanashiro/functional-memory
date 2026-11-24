"""The Q/K width 4 exponential on the unscaled cosine score: its partner's recipe with the score alone changed."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import exp_cosine_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config, save_config

RUN, LABEL = "r15f803273258", "exp-row-max-dk4-unit-cosine"


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


def test_the_recipe_changes_the_score_alone():
    config, partner = adapter.recipe(), adapter.partner_config()
    assert changed(config, partner) == {f"model.{name}" for name in adapter.CHANGES}
    assert (config.model.qk_norm_type, config.model.qk_norm_head_scale) == ("rms_norm_cosine", False)
    assert config.model.attention_degree is None and not config.model.attention_normalization
    assert config.model.attention_exp_shift == "row_max" and config.model.qk_head_dim == 4
    assert config.train.seed == partner.train.seed == 42 and config.optim.learning_rate == partner.optim.learning_rate
    adapter.check_recipe(config)


def test_the_allocated_run_is_the_recipe_at_its_launch_commit():
    path = adapter.PARTNER.parents[1] / RUN / "config.yaml"
    config = load_config(path)
    assert changed(config, adapter.recipe()) == set()
    # Acceptance replaces "auto" with the launch commit before it checks the recipe.
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == RUN and record["label"] == LABEL == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_other_recipes_are_refused():
    config = adapter.recipe()
    for other in (adapter.partner_config(),
                  replace(config, optim=replace(config.optim, learning_rate=2 * config.optim.learning_rate)),
                  replace(config, train=replace(config.train, seed=43)),
                  replace(config, model=replace(config.model, qk_norm_eps=2 * config.model.qk_norm_eps))):
        with pytest.raises(ValueError):
            adapter.check_recipe(other)


def test_initial_parameters_are_the_partners_without_its_head_multipliers():
    facts = adapter.check_initialization(adapter.recipe())
    assert facts["dropped_parameters"] == [adapter.HEAD_SCALE] and facts["all_other_partner_arrays_identical"]


def test_the_spec_qualifies_the_fused_exponential():
    config = adapter.recipe()
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "fused" and spec.check_config is adapter.check_recipe
    assert "scaffolds/transformer/tests/test_unnormalized_fused_attention.py" in spec.gpu_tests
    assert {"scaffolds/transformer/tests/test_qk_rms_cosine.py",
            "experiments/transformer_pp/tests/test_exp_cosine_acceptance.py"} <= set(spec.cpu_tests)
    assert adapter.GPU_TYPES == ("H100", "B300")


def test_the_pilot_holds_scores_to_the_cosine_range():
    config = adapter.recipe()
    layers = [{"score_min": -1.0, "score_max": 1.005}] * config.model.num_layers
    assert adapter.check_pilot(config, None, {"attention_diagnostics": {"layers": layers}})["score_max"] == 1.005
    for bad in ([{"score_min": -2.5, "score_max": 2.5}] * config.model.num_layers, layers[:-1]):
        with pytest.raises(RuntimeError):
            adapter.check_pilot(config, None, {"attention_diagnostics": {"layers": bad}})


def test_the_gpu_type_rule(tmp_path):
    path = tmp_path / "config.yaml"
    save_config(adapter.recipe(), path)
    adapter.require_gpu_type([str(path), "--gpu-type", "B300"])
    adapter.require_gpu_type([str(path), "--runtime-env", "--gpu-type", "B200"])
    with pytest.raises(ValueError):
        adapter.require_gpu_type([str(path), "--gpu-type", "B200"])
