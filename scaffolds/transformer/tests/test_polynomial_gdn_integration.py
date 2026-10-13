"""Quadratic GDN integration: nested initialization, trainability, and provenance."""

import csv
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.records import create_run, regenerate_index
from scaffolds.transformer.attention.gated_deltanet import GatedDeltaNet
from scaffolds.transformer.configs import Config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_gated_deltanet import gdn_config, gdn_model
from scaffolds.transformer.training import loop, telemetry, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.state import create_state


def quadratic_model(**updates):
    return gdn_model(gdn_erase_mode="independent", gdn_polynomial_degree=2, **updates)


@pytest.mark.parametrize("updates", [
    {"gdn_polynomial_degree": 0}, {"gdn_polynomial_degree": 3}, {"gdn_polynomial_degree": True},
    {"gdn_polynomial_degree": 2, "gdn_erase_mode": "tied"},
    {"gdn_polynomial_degree": 2, "attention_mechanism": "dot_product"},
])
def test_degree_configuration_rejects_ambiguous_architectures(updates):
    with pytest.raises(ValueError, match="gdn_polynomial_degree"):
        gdn_model(**updates)


def test_legacy_recipes_remain_degree_one_and_quadratic_is_not_resume_compatible():
    config = Config()
    config.model = gdn_model(gdn_erase_mode="independent")
    linear = config_to_dict(config)
    legacy = {**linear, "model": {k: v for k, v in linear["model"].items() if k != "gdn_polynomial_degree"}}
    assert config_dicts_equal(linear, legacy)
    quadratic = {**linear, "model": {**linear["model"], "gdn_polynomial_degree": 2}}
    assert not config_dicts_equal(quadratic, legacy)


@pytest.mark.parametrize("initialization", ["flax_default", "gated_deltanet_reference"])
def test_quadratic_initialization_nests_linear_model_without_perturbing_any_existing_draw(initialization):
    config = gdn_model(gdn_erase_mode="independent", initialization=initialization)
    linear = build_model(config, jax.random.PRNGKey(17))
    quadratic_config = replace(config, gdn_polynomial_degree=2)
    quadratic = build_model(quadratic_config, jax.random.PRNGKey(17))
    flatten = lambda model: {path: np.asarray(value[...])
                             for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    old, new = flatten(linear), flatten(quadratic)
    assert len(new) - len(old) == 2 * config.num_layers
    for path, value in old.items():
        np.testing.assert_array_equal(new[path], value, err_msg=str(path))
    for name in quadratic.layer_names:
        layer = getattr(quadratic, name).attention
        np.testing.assert_array_equal(layer.b0_proj.kernel[...], 0)
        np.testing.assert_array_equal(layer.b2_proj.kernel[...], 0)
        np.testing.assert_array_equal(layer.lambda_proj.kernel[...], layer.b_proj.kernel[...])
    assert sum(value.size for value in new.values()) == model_parameter_count(quadratic_config)
    assert model_parameter_count(quadratic_config) - model_parameter_count(config) == (
        2 * config.num_layers * config.hidden_size * config.num_heads)
    tokens = jnp.asarray([[1, 2, 3, 4, 5], [6, 4, 2, 0, 7]])
    np.testing.assert_allclose(quadratic(tokens), linear(tokens), rtol=3e-5, atol=7e-6)


@pytest.mark.parametrize("initialization", ["flax_default", "gated_deltanet_reference"])
def test_zero_coefficient_projections_have_independent_donated_buffers(initialization):
    model = build_model(quadratic_model(initialization=initialization), jax.random.PRNGKey(17))
    layer = getattr(model, model.layer_names[0]).attention
    names = ("b0_proj", "b_proj", "b2_proj", "lambda_proj")
    weights = tuple(getattr(layer, name).kernel[...] for name in names)
    before = tuple(np.asarray(weight).copy() for weight in weights)
    # Zero arrays as well as equal write/erase arrays must not alias donated buffers.
    def update(parameters):
        return tuple(parameter + index + 1 for index, parameter in enumerate(parameters))
    after = jax.jit(update, donate_argnums=(0,))(weights)
    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        np.testing.assert_allclose(new, old + index + 1, rtol=2e-6, atol=2e-7)


def test_zero_initialized_coefficients_receive_task_gradients_and_degree_telemetry():
    module = GatedDeltaNet(32, 4, 8, 8, erase_mode="independent", polynomial_degree=2, rngs=nnx.Rngs(19))
    x = jax.random.normal(jax.random.PRNGKey(20), (2, 9, 32))
    cotangent = jax.random.normal(jax.random.PRNGKey(21), x.shape)
    gradients = nnx.grad(lambda layer: jnp.sum(layer(x) * cotangent))(module)
    for name in ("b0_proj", "b_proj", "b2_proj", "lambda_proj"):
        gradient = np.asarray(getattr(gradients, name).kernel[...])
        assert np.all(np.isfinite(gradient)) and np.linalg.norm(gradient) > 0, name
    with telemetry.collect() as collector:
        _, diagnostics = module(x, return_diagnostics=True)
    assert diagnostics["invalid_coefficient_count"] == 0
    for degree in (0, 2):
        for statistic in ("mean", "min", "max", "rms"):
            assert diagnostics[f"beta_degree_{degree}_{statistic}"] == 0
            assert collector.values[f"beta_degree_{degree}_{statistic}"] == 0
    assert diagnostics["beta_degree_1_rms"] > 0
    assert telemetry.parameter_group(("layers_0", "attention", "b0_proj", "kernel")) == "attention_write_degree_0"
    assert telemetry.parameter_group(("layers_0", "attention", "b2_proj", "kernel")) == "attention_write_degree_2"
    module.b2_proj.kernel[...] = module.b2_proj.kernel[...].at[0, 0].set(jnp.nan)
    _, invalid = module(x, return_diagnostics=True)
    assert invalid["invalid_coefficient_count"] > 0
    assert invalid["invalid_output_count"] > 0


def test_identity_tags_and_run_index_expose_geometry_degree_and_gate_initialization(tmp_path):
    config = Config()
    config.model = quadratic_model(initialization="gated_deltanet_reference")
    identity = attention_identity(config)
    assert identity["attention_degree"] == identity["gdn_polynomial_degree"] == 2
    assert identity["gdn_query_transport_space"] == "original_key_space"
    assert identity["gdn_write_activation"] == ["tanh", "sigmoid", "tanh"]
    assert identity["gdn_state_elements_per_layer"] == 4 * 8 * (1 + 8 + 8 ** 2)
    assert "one decay shared by all degrees" in identity["attention_formula"]
    assert identity["gdn_write_initialization"] == "zero_constant_and_quadratic_projections"
    tags = tracking.run_tags(config, {})
    assert "m2" in tags and "quadratic_gdn" in tags and "erase_independent" in tags
    study = tmp_path / "study"
    create_run(study, config_to_dict(config))
    with regenerate_index(study).open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["gdn_polynomial_degree"] == row["attention_degree"] == "2"
    assert row["gdn_write_initialization"] == identity["gdn_write_initialization"]


def test_quadratic_training_learns_zero_gates_and_resumes_exactly(tmp_path):
    config = gdn_config(tmp_path / "full", initialization="gated_deltanet_reference",
                        gdn_erase_mode="independent", gdn_polynomial_degree=2)
    loop.run_training(config)
    resumed = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed, stop_after_step=1, audit_checkpoint=True)
    finished = loop.run_training(resumed, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == finished["checkpoint_audit"]["start_state_sha256"]
    states = []
    for current in (config, resumed):
        manager = make_checkpoint_manager(current)
        try:
            states.append(restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))[0])
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    attention = states[0].params["layers_0"]["attention"]
    for name in ("b0_proj", "b2_proj"):
        assert np.linalg.norm(np.asarray(attention[name]["kernel"][...])) > 0
    metrics = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validation = [row for row in metrics if row.get("event") == "validation"]
    assert validation and all(row["attention_diagnostics"]["valid"] for row in validation)
    layer = validation[-1]["attention_diagnostics"]["layers"][0]
    assert layer["invalid_coefficient_count"] == 0
    assert layer["beta_degree_0_rms"] > 0 and layer["beta_degree_2_rms"] > 0
