"""Polynomial GDN integration: nested initialization, trainability, and provenance."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.gated_deltanet import GatedDeltaNet
from scaffolds.transformer.configs import Config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_gated_deltanet import gdn_config, gdn_model
from scaffolds.transformer.training import loop, telemetry, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.state import create_state


def polynomial_model(degree, **updates):
    return gdn_model(gdn_erase_mode="independent", gdn_polynomial_degree=degree, **updates)


def zero_degrees(degree):
    return (0, *range(2, degree + 1))


@pytest.mark.parametrize("updates", [
    {"gdn_polynomial_degree": 0}, {"gdn_polynomial_degree": 4}, {"gdn_polynomial_degree": True},
    {"gdn_polynomial_degree": 2, "gdn_erase_mode": "tied"},
    {"gdn_polynomial_degree": 3, "gdn_erase_mode": "tied"},
    {"gdn_polynomial_degree": 2, "attention_mechanism": "dot_product"},
    {"gdn_polynomial_degree": 3, "attention_mechanism": "dot_product"},
])
def test_degree_configuration_rejects_ambiguous_architectures(updates):
    with pytest.raises(ValueError, match="gdn_polynomial_degree"):
        gdn_model(**updates)


def test_legacy_recipes_remain_degree_one_and_degrees_are_not_resume_compatible():
    config = Config()
    config.model = gdn_model(gdn_erase_mode="independent")
    linear = config_to_dict(config)
    legacy = {**linear, "model": {k: v for k, v in linear["model"].items() if k != "gdn_polynomial_degree"}}
    assert config_dicts_equal(linear, legacy)
    polynomial = [{**linear, "model": {**linear["model"], "gdn_polynomial_degree": degree}}
                  for degree in (2, 3)]
    for recipe in polynomial:
        assert not config_dicts_equal(recipe, legacy)
    assert not config_dicts_equal(*polynomial)


@pytest.mark.parametrize("initialization", ["flax_default", "gated_deltanet_reference"])
@pytest.mark.parametrize("degree", (2, 3))
def test_polynomial_initialization_nests_linear_without_perturbing_existing_draws(initialization, degree):
    config = gdn_model(gdn_erase_mode="independent", initialization=initialization)
    linear = build_model(config, jax.random.PRNGKey(17))
    polynomial_config = replace(config, gdn_polynomial_degree=degree)
    polynomial = build_model(polynomial_config, jax.random.PRNGKey(17))
    flatten = lambda model: {path: np.asarray(value[...])
                             for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    old, new = flatten(linear), flatten(polynomial)
    assert len(new) - len(old) == degree * config.num_layers
    for path, value in old.items():
        np.testing.assert_array_equal(new[path], value, err_msg=str(path))
    for name in polynomial.layer_names:
        layer = getattr(polynomial, name).attention
        for r in zero_degrees(degree):
            np.testing.assert_array_equal(getattr(layer, f"b{r}_proj").kernel[...], 0)
        np.testing.assert_array_equal(layer.lambda_proj.kernel[...], layer.b_proj.kernel[...])
    assert sum(value.size for value in new.values()) == model_parameter_count(polynomial_config)
    assert model_parameter_count(polynomial_config) - model_parameter_count(config) == (
        degree * config.num_layers * config.hidden_size * config.num_heads)
    tokens = jnp.asarray([[1, 2, 3, 4, 5], [6, 4, 2, 0, 7]])
    np.testing.assert_allclose(polynomial(tokens), linear(tokens), rtol=3e-5, atol=7e-6)
    if degree == 3:
        quadratic = build_model(replace(config, gdn_polynomial_degree=2), jax.random.PRNGKey(17))
        for path, value in flatten(quadratic).items():
            np.testing.assert_array_equal(new[path], value, err_msg=str(path))
        np.testing.assert_array_equal(polynomial(tokens), quadratic(tokens))


@pytest.mark.parametrize("initialization", ["flax_default", "gated_deltanet_reference"])
@pytest.mark.parametrize("degree", (2, 3))
def test_zero_coefficient_projections_have_independent_donated_buffers(initialization, degree):
    model = build_model(polynomial_model(degree, initialization=initialization), jax.random.PRNGKey(17))
    layer = getattr(model, model.layer_names[0]).attention
    names = (*[f"b{r}_proj" for r in zero_degrees(degree)], "b_proj", "lambda_proj")
    weights = tuple(getattr(layer, name).kernel[...] for name in names)
    before = tuple(np.asarray(weight).copy() for weight in weights)
    # Zero arrays as well as equal write/erase arrays must not alias donated buffers.
    def update(parameters):
        return tuple(parameter + index + 1 for index, parameter in enumerate(parameters))
    after = jax.jit(update, donate_argnums=(0,))(weights)
    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        np.testing.assert_allclose(new, old + index + 1, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("degree", (2, 3))
def test_zero_initialized_coefficients_receive_task_gradients_and_degree_telemetry(degree):
    module = GatedDeltaNet(32, 4, 8, 8, erase_mode="independent", polynomial_degree=degree, rngs=nnx.Rngs(19))
    x = jax.random.normal(jax.random.PRNGKey(20), (2, 9, 32))
    cotangent = jax.random.normal(jax.random.PRNGKey(21), x.shape)
    gradients = nnx.grad(lambda layer: jnp.sum(layer(x) * cotangent))(module)
    for name in (*[f"b{r}_proj" for r in zero_degrees(degree)], "b_proj", "lambda_proj"):
        gradient = np.asarray(getattr(gradients, name).kernel[...])
        assert np.all(np.isfinite(gradient)) and np.linalg.norm(gradient) > 0, name
    with telemetry.collect() as collector:
        _, diagnostics = module(x, return_diagnostics=True)
    assert diagnostics["invalid_coefficient_count"] == 0
    for r in zero_degrees(degree):
        for statistic in ("mean", "min", "max", "rms"):
            assert diagnostics[f"beta_degree_{r}_{statistic}"] == 0
            assert collector.values[f"beta_degree_{r}_{statistic}"] == 0
        assert telemetry.parameter_group(("layers_0", "attention", f"b{r}_proj", "kernel")) == f"attention_write_degree_{r}"
    assert diagnostics["beta_degree_1_rms"] > 0
    projection = getattr(module, f"b{degree}_proj")
    projection.kernel[...] = projection.kernel[...].at[0, 0].set(jnp.nan)
    _, invalid = module(x, return_diagnostics=True)
    assert invalid["invalid_coefficient_count"] > 0
    assert invalid["invalid_output_count"] > 0


@pytest.mark.parametrize("degree", (2, 3))
def test_identity_tags_expose_geometry_degree_and_gate_initialization(degree):
    config = Config()
    config.model = polynomial_model(degree, initialization="gated_deltanet_reference")
    identity = attention_identity(config)
    assert identity["attention_degree"] == identity["gdn_polynomial_degree"] == degree
    assert identity["gdn_query_transport_space"] == "original_key_space"
    assert identity["gdn_write_activation"] == ["sigmoid" if r == 1 else "tanh" for r in range(degree + 1)]
    assert identity["gdn_state_elements_per_layer"] == 4 * 8 * sum(8 ** r for r in range(degree + 1))
    assert "one decay shared by all degrees" in identity["attention_formula"]
    assert f"sum_{{r=0}}^{degree}" in identity["attention_formula"]
    assert f"beta_{degree}=tanh(W_{degree} x)" in identity["attention_formula"]
    assert identity["gdn_write_initialization"] == {
        2: "zero_constant_and_quadratic_projections", 3: "zero_constant_quadratic_and_cubic_projections"}[degree]
    tags = tracking.run_tags(config, {})
    assert f"m{degree}" in tags and {2: "quadratic_gdn", 3: "cubic_gdn"}[degree] in tags and "erase_independent" in tags



@pytest.mark.parametrize("degree", (2, 3))
def test_polynomial_training_learns_zero_gates_and_resumes_exactly(tmp_path, degree):
    config = gdn_config(tmp_path / "full", initialization="gated_deltanet_reference",
                        gdn_erase_mode="independent", gdn_polynomial_degree=degree)
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
    for r in zero_degrees(degree):
        assert np.linalg.norm(np.asarray(attention[f"b{r}_proj"]["kernel"][...])) > 0
    metrics = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validation = [row for row in metrics if row.get("event") == "validation"]
    assert validation and all(row["attention_diagnostics"]["valid"] for row in validation)
    layer = validation[-1]["attention_diagnostics"]["layers"][0]
    assert layer["invalid_coefficient_count"] == 0
    for r in zero_degrees(degree):
        assert layer[f"beta_degree_{r}_rms"] > 0


@pytest.mark.parametrize("degree", (2, 3))
def test_polynomial_backend_selection_checks_width_and_acceptance_covers_degree(degree):
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.attention.backends.selection import resolve_attention_backend

    model = polynomial_model(degree, attention_backend="auto", max_seq_len=2048)
    assert resolve_attention_backend(model, dtype="bfloat16", platform="gpu") == "fused"
    wide = replace(model, qk_head_dim=128)
    assert resolve_attention_backend(wide, dtype="bfloat16", platform="gpu") == "blocked"
    with pytest.raises(ValueError, match="dimensions at most 64"):
        resolve_attention_backend(replace(wide, attention_backend="fused"), dtype="bfloat16", platform="gpu")
    config = config_to_dict(Config(model=model))
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    cpu = acceptance.test_paths(spec, config, resolved_backend="fused", cpu=True)
    gpu = acceptance.test_paths(spec, config, resolved_backend="fused", cpu=False)
    assert acceptance.POLYNOMIAL_GDN_TESTS in cpu
    assert acceptance.POLYNOMIAL_GDN_INTEGRATION_TESTS in cpu
    assert acceptance.POLYNOMIAL_GDN_KERNEL_TESTS in gpu
    assert (acceptance.CUBIC_GDN_TESTS in cpu) == (degree == 3)
    assert (acceptance.CUBIC_GDN_KERNEL_TESTS in gpu) == (degree == 3)


def test_polynomial_runtime_identity_records_options_without_changing_recipe(monkeypatch):
    from scaffolds.transformer.attention.backends import polynomial_gdn

    monkeypatch.setattr(polynomial_gdn, "_default_num_warps", lambda: 4)
    config = Config(model=polynomial_model(3, attention_backend="fused"))
    recipe = config_to_dict(config)
    monkeypatch.setenv("POLY_GDN_VALUE_PRECISION", "bf16_read")
    monkeypatch.setenv("POLY_GDN_NUM_WARPS", "8")
    monkeypatch.setenv("POLY_GDN_MIN_CTAS_PER_SM", "1")
    identity = attention_identity(config)
    assert identity["gdn_value_precision"] == "bf16_read"
    assert identity["gdn_num_warps"] == 8
    assert identity["gdn_min_ctas_per_sm"] == 1
    assert identity["gdn_transport_dot_precision"] == "tf32x3"
    monkeypatch.setenv("POLY_GDN_MIN_CTAS_PER_SM", "0")
    assert attention_identity(config)["gdn_min_ctas_per_sm"] == 0
    monkeypatch.delenv("POLY_GDN_MIN_CTAS_PER_SM")
    assert attention_identity(config)["gdn_min_ctas_per_sm"] == 0
    assert config_to_dict(config) == recipe
