"""The affine comparison changes initialization only and preserves source optimizer policy."""

import pytest

from experiments.transformer_pp import spline_affine_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_test_paths
from scaffolds.transformer.configs import Config, config_to_dict


def variant(conditioning="query"):
    config = adapter.source_config(conditioning)
    for field, value in adapter.AFFINE_INITIALIZATION.items():
        setattr(config.model, field, value)
    return config


@pytest.mark.parametrize("conditioning", ("static", "query"))
def test_affine_variant_preserves_the_pinned_source(conditioning):
    config = variant(conditioning)
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.train.seed == 42
    assert config.model.attention_spline_knots == (-32.0, -6.0, 0.0, 6.0, 32.0)
    assert config.model.attention_blocked_implementation == "strips"
    assert config.model.positional_encoding == "rope" and not config.model.attention_forget_gate


@pytest.mark.parametrize(("section", "field", "value"), (
    ("model", "attention_spline_initial_slope", .5),
    ("model", "attention_spline_initial_intercept", 0.),
    ("model", "attention_degree", 3),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "qk_norm_head_scale", False),
    ("model", "positional_encoding", "none"),
    ("optim", "learning_rate", .002),
    ("optim", "total_steps", 100),
    ("train", "seed", 7),
))
def test_unintended_recipe_changes_are_rejected(section, field, value):
    config = variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_execution_paths_can_change():
    config = variant()
    config.train.workdir = "/different/run"
    config.train.checkpoint_dir = "/different/run/checkpoints"
    config.train.wandb_run_name = "spline-query-affine"
    adapter.check_recipe(config)


def test_acceptance_keeps_full_spline_and_strips_qualification():
    config = variant()
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    selected = selected_test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    assert adapter.ADAPTER_TESTS in selected
    assert any("test_spline_attention.py" in path for path in selected)
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization
    assert spec.pilot_steps == 20


@pytest.mark.parametrize("query", (False, True))
def test_actual_initialization_and_optimizer_masks_match_source(query, monkeypatch):
    from scaffolds.transformer.tests.test_spline_attention import tiny_model

    source = Config(model=tiny_model(query))
    source.data.sequence_length = 33
    source.data.synthetic_vocab_size = 64
    config = Config(model=tiny_model(query, **adapter.AFFINE_INITIALIZATION))
    config.data.sequence_length = 33
    config.data.synthetic_vocab_size = 64
    monkeypatch.setattr(adapter, "check_recipe", lambda config: None)
    monkeypatch.setattr(adapter, "source_config", lambda conditioning: source)
    facts = adapter.check_initialization(config)["affine_spline_start"]
    assert facts["initial_coefficients"] == list(adapter.INITIAL_COEFFICIENTS)
    assert facts["coefficient_shape"] == [2, 2, 8]
    assert facts["all_other_parameters_match_source_initialization"]
    assert facts["optimizer_decay_mask_matches_source"]
    expected = ({"query_spline_bias": False, "query_spline_amplitude": False, "query_spline_kernel": True}
                if query else {"spline_coefficients": False})
    assert facts["spline_parameter_decay_mask"] == expected
