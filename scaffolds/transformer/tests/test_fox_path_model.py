"""FoX and PaTH compose through the shared model and optimizer."""
from dataclasses import replace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample,
)
from scaffolds.transformer.training.state import create_state


def _config():
    config = Config(model=ModelConfig(
        vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=16,
        num_heads=2, num_kv_heads=2, qk_head_dim=8, value_head_dim=8,
        intermediate_size=24, initialization="transformer_truncated_reference",
        attention_backend="blocked", positional_encoding="path", path_w_rank=4,
        attention_degree=4, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_sharing="per_head", attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_projection_initial_std=0.25,
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
        attention_coefficient_kernel_weight_decay=True, attention_forget_gate=True,
        attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
        qk_norm_sharing="per_layer", qk_norm_head_scale=True, qk_norm_initial_scale=None,
    ))
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 9, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 3, 0
    return config


def test_combined_model_owns_both_gates_and_preserves_backbone_initialization():
    config = _config()
    model = build_model(config.model, 3)
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == model_parameter_count(config.model)
    facts = acceptance.matches_softmax_backbone_initialization(config)
    assert {"forget_gate_kernel", "path_w_down", "path_w_up", "query_polynomial_kernel"} <= set(facts["mechanism_parameters"])
    plain = build_model(replace(config.model, attention_forget_gate=False), 3)
    actual = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    for path, value in nnx.to_flat_state(nnx.state(plain, nnx.Param)):
        np.testing.assert_array_equal(actual[path][...], value[...])
    tokens = jnp.arange(16).reshape(2, 8) % 16
    grads = nnx.grad(lambda m: jnp.square(m(tokens).astype(jnp.float32)).sum())(model)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads["forget_gate_kernel"][...])) > 0
    assert float(jnp.linalg.norm(grads["path_w_up"][...])) > 0


def test_real_optimizer_decays_projections_and_exempts_coefficients_and_norms():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(0))
    # Nonzero values expose accidental decay of zero-initialized a and biases.
    state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    actual = dict(nnx.to_flat_state(updated.params))
    specifications = {spec.name: spec.weight_decay for spec in attention_parameter_specs(config.model)}
    for name in ("query_polynomial_kernel", "forget_gate_kernel", "path_w_down", "path_w_up", "path_w_conv", "path_beta_kernel"):
        assert specifications[name]
    for name in ("query_polynomial_bias", "query_polynomial_amplitude", "forget_gate_log_rate", "forget_gate_dt_bias", "path_beta_bias"):
        assert not specifications[name]
    for path, value in nnx.to_flat_state(state.params):
        decays = specifications.get(path[-1], path[-1] not in ("embedding", "scale", "bias", "qk_norm_log_scale"))
        expected = value[...] * (1 - config.optim.learning_rate * config.optim.weight_decay * decays)
        np.testing.assert_allclose(actual[path][...], expected, rtol=1e-6, atol=1e-7)


def test_diagnostics_and_acceptance_cover_both_mechanisms():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    layer = report["layers"][0]
    assert 0 < layer["path_beta_mean"] < 2
    gate = report["coefficients"]["forget_gate"]
    assert gate["finite"]
    assert np.max(gate["log_alpha_sample"]["max"]) <= 0
    identity = attention_identity(config)
    assert identity["attention_forget_gate"] and identity["positional_encoding"] == "path"
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (True, False):
        paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=cpu)
        assert acceptance.FOX_PATH_ATTENTION_TESTS in paths
        assert acceptance.FOX_PATH_MODEL_TESTS in paths
        assert acceptance.FOX_PATH_ATTENTION_FUSED_TESTS in paths


@pytest.mark.parametrize("backend", ("blocked", "fused", "auto"))
def test_supported_backends_accept_the_combination(backend):
    replace(_config().model, attention_backend=backend, qk_head_dim=64, value_head_dim=64)


def test_forget_gate_on_path_softmax_is_path_fox():
    # The gate is applied, never silently dropped: PaTH-FoX softmax is qualified in test_path_attention.py.
    model = ModelConfig(positional_encoding="path", attention_forget_gate=True)
    assert model.path_kernel() == "softmax"
    assert {"forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"} <= {
        spec.name for spec in attention_parameter_specs(model)}


def test_path_rejects_query_distance_decay():
    # The PaTH reference and fused kernels apply only the cumulative decay exp(L_t - L_i).
    with pytest.raises(ValueError, match="query-distance decay mode requires RoPE"):
        replace(_config().model, attention_forget_gate_mode="query_distance")
