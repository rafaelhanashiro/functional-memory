"""The binomial comparison changes initialization only and preserves source optimizer policy."""

import pytest

from experiments.transformer_pp import spline_binomial_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as selected_test_paths
from scaffolds.transformer.configs import Config, config_to_dict


def variant(conditioning="query"):
    config = adapter.source_config(conditioning)
    for field, value in adapter.BINOMIAL_INITIALIZATION.items():
        setattr(config.model, field, value)
    return config


@pytest.mark.parametrize("conditioning", ("static", "query"))
def test_binomial_variant_preserves_the_pinned_source(conditioning):
    config = variant(conditioning)
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.train.seed == 42
    assert config.model.attention_spline_knots == (-32.0, -6.0, 0.0, 6.0, 32.0)
    assert config.model.attention_blocked_implementation == "strips"
    assert config.model.positional_encoding == "rope" and not config.model.attention_forget_gate


@pytest.mark.parametrize(("section", "field", "value"), (
    ("model", "attention_spline_initial_power_coefficients", [1.0, 0.5, 0.125]),
    ("model", "attention_spline_initial_power_coefficients", [1.0, 1.0]),
    ("model", "attention_spline_initial_intercept", 1.0),
    ("model", "attention_spline_initial_power_coefficients", [1 / 16 + 1e-3, 1 / 16, 3 / 128, 1 / 256, 1 / 4096]),
    ("model", "attention_spline_initial_power_coefficients", [1, 1, 3 / 8, 1 / 16, 1 / 256]),
    ("model", "rms_norm_eps", 1e-8),
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
    config.train.wandb_run_name = "spline-query-binomial"
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
    assert spec.check_pilot is adapter.check_pilot


@pytest.mark.parametrize("query", (False, True))
def test_actual_initialization_and_optimizer_masks_match_source(query, monkeypatch):
    from scaffolds.transformer.tests.test_spline_attention import tiny_model

    source = Config(model=tiny_model(query))
    source.data.sequence_length = 33
    source.data.synthetic_vocab_size = 64
    config = Config(model=tiny_model(query, **adapter.BINOMIAL_INITIALIZATION))
    config.data.sequence_length = 33
    config.data.synthetic_vocab_size = 64
    monkeypatch.setattr(adapter, "check_recipe", lambda config: None)
    monkeypatch.setattr(adapter, "source_config", lambda conditioning: source)
    facts = adapter.check_initialization(config)["binomial_spline_start"]
    import numpy as np
    assert facts["initial_coefficients"] == np.asarray(adapter.INITIAL_COEFFICIENTS, np.float32).tolist()
    assert facts["largest_coefficient_fp32_spacing"] == float(np.spacing(np.float32(6561 / 16)))
    assert facts["coefficient_shape"] == [2, 2, 8]
    assert facts["all_other_parameters_match_source_initialization"]
    assert facts["optimizer_decay_mask_matches_source"]
    expected = ({"query_spline_bias": False, "query_spline_amplitude": False, "query_spline_kernel": True}
                if query else {"spline_coefficients": False})
    assert facts["spline_parameter_decay_mask"] == expected


def pilot_validation(config, *, moved=True, amplitude_moved=True):
    import numpy as np

    shape = (config.model.num_layers, config.model.num_heads, 8)
    final = np.broadcast_to(np.asarray(adapter.INITIAL_COEFFICIENTS, np.float32), shape).copy()
    if moved:
        final[0, 0, 3] += np.float32(1e-4)
    coefficients = {"basis": "bspline", "finite": True}
    if config.model.attention_coefficient_conditioning == "query":
        amplitude = np.zeros(shape, np.float32)
        if amplitude_moved:
            amplitude[0, 0, 3] = np.float32(1e-5)
        coefficients.update(projection_bias=final.tolist(), modulation_amplitude=amplitude.tolist())
    else:
        coefficients["raw"] = final.tolist()
    return {"step": 20, "attention_diagnostics": {"coefficients": coefficients}}


@pytest.mark.parametrize("conditioning", ("static", "query"))
def test_pilot_checks_actual_updates_without_requiring_inactive_outer_basis(conditioning):
    config = variant(conditioning)
    facts = adapter.check_pilot(config, {"final_step": 20}, pilot_validation(config))
    assert facts["active_central_coefficients_moved"]
    assert facts["moved_layer_head_count_per_coefficient"] == [0, 0, 0, 1, 0, 0, 0, 0]
    assert not facts["outer_coefficient_movement_required"]
    if conditioning == "query":
        assert facts["query_amplitudes_moved_from_zero"]


@pytest.mark.parametrize("conditioning", ("static", "query"))
def test_pilot_rejects_frozen_coefficient_offsets(conditioning):
    config = variant(conditioning)
    with pytest.raises(RuntimeError, match="No active central"):
        adapter.check_pilot(config, {"final_step": 20}, pilot_validation(config, moved=False))


def test_pilot_rejects_query_generator_without_amplitude_learning():
    config = variant()
    with pytest.raises(RuntimeError, match="No active query spline amplitude"):
        adapter.check_pilot(config, {"final_step": 20}, pilot_validation(config, amplitude_moved=False))


def test_pilot_rejects_nonfinite_coefficients():
    config = variant()
    validation = pilot_validation(config)
    validation["attention_diagnostics"]["coefficients"]["projection_bias"][0][0][3] = float("nan")
    with pytest.raises(RuntimeError, match="invalid spline coefficient"):
        adapter.check_pilot(config, {"final_step": 20}, validation)
