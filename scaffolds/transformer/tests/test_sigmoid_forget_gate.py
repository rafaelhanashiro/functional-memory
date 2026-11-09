"""The Forgetting Transformer's original sigmoid forget gate on softmax, exp and query coefficients.

alpha_t = sigmoid(x_t w_f + b_f) per layer and head (Lin et al., 2025), with w_f ~ N(0, 0.02^2)
and b_f = 0 as in the paper's code; w_f decays and b_f does not. The cumulative decay joins the
attention exactly as the Gated DeltaNet gate's does, which ``test_forget_gate_attention`` and
``test_coefficient_forms`` qualify on the blocked kernels; these tests cover what the form changes.
"""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.forget_gate import (
    SIGMOID_KERNEL_STD, log_decay, parameter_names, parameter_specs, sigmoid_log_decay,
)
from scaffolds.transformer.attention.parameters import (
    attention_parameter_count, attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
)
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample,
)
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state

KERNELS = ("softmax", "exp", "query")
GATE = ("forget_gate_kernel", "forget_gate_bias")


def _config(kernel="softmax", **changes):
    """Tiny no-RoPE recipes: plain-score softmax (FoX LLaMA); exp and query on the canonical scaffold."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=4, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
                 attention_forget_gate=True, attention_forget_gate_form="sigmoid", positional_encoding="none",
                 initialization="transformer_reference")
    if kernel == "softmax":
        model.update(attention_normalization=True, attention_output_scaling="none", qk_norm=False)
    else:
        model.update(attention_normalization=False, attention_output_scaling="head_rms_norm", qk_norm=True,
                     qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True)
    if kernel == "query":
        model.update(attention_degree=4, attention_coefficients="learned", attention_coefficient_sharing="per_head",
                     attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                     attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=8 ** -.5,
                     attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                     attention_coefficient_kernel_weight_decay=True)
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


def test_parameters_and_initialization_follow_the_paper():
    assert parameter_names("sigmoid") == GATE
    specs = {spec.name: spec for spec in parameter_specs(hidden_size=896, form="sigmoid")}
    assert set(specs) == set(GATE)
    assert specs["forget_gate_kernel"].weight_decay and not specs["forget_gate_bias"].weight_decay
    values = initial_attention_parameter_values(tuple(specs.values()), num_layers=15, num_heads=14,
                                                key=mechanism_key(jax.random.PRNGKey(42)))
    kernel, bias = np.asarray(values["forget_gate_kernel"]), np.asarray(values["forget_gate_bias"])
    assert kernel.shape == (15, 14, 896) and bias.shape == (15, 14, 1)
    assert abs(kernel.mean()) < 1e-3 and abs(kernel.std() - SIGMOID_KERNEL_STD) < 1e-3
    assert np.all(bias == 0)


def test_log_decay_is_log_sigmoid_and_the_input_form_at_unit_rate():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(2, 5, 8)), jnp.float32)
    kernel = jnp.asarray(rng.normal(size=(3, 8)), jnp.float32)
    bias = jnp.asarray([[-2.0], [0.0], [3.0]], jnp.float32)
    z = np.einsum("btd,hd->bth", np.asarray(x, np.float64), np.asarray(kernel, np.float64)) + np.asarray(bias)[:, 0]
    expected = -np.logaddexp(0.0, -z)  # log sigmoid(z) in float64
    actual = np.asarray(sigmoid_log_decay(x, kernel, bias))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.dtype == np.float32 and np.all(actual <= 0)
    # The Gated DeltaNet form with exp(A_log) = 1 and the projection negated is the same gate.
    np.testing.assert_allclose(log_decay(x, -kernel, jnp.zeros((3, 1)), -bias), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="sigmoid forget gate"):
        sigmoid_log_decay(x, kernel, bias[:, 0])


@pytest.mark.parametrize("kernel", KERNELS)
def test_configurations(kernel):
    config = _config(kernel)
    assert config.model.attention_forget_gate and config.model.attention_forget_gate_form == "sigmoid"
    with pytest.raises(ValueError, match="requires"):
        _config(kernel, attention_forget_gate=False)
    with pytest.raises(ValueError, match="sigmoid forget gate form"):
        _config(kernel, positional_encoding="path", qk_head_dim=16, value_head_dim=16)
    if kernel == "query":
        with pytest.raises(ValueError, match="sigmoid forget gate form"):
            _config(kernel, attention_forget_gate_mode="query_distance")
    assert attention_parameter_count(config.model) - attention_parameter_count(
        _config(kernel, attention_forget_gate=False, attention_forget_gate_form="input").model) == (8 + 1) * 2
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    state = create_state(config, jax.random.PRNGKey(0))
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(state.params, excluded_names=excluded))}
    assert mask[("forget_gate_kernel",)] is True and mask[("forget_gate_bias",)] is False
    assert ("forget_gate_log_rate",) not in mask and ("forget_gate_dt_bias",) not in mask


@pytest.mark.parametrize("kernel", KERNELS)
def test_identity_and_diagnostics(kernel):
    config = _config(kernel)
    identity = attention_identity(config)
    assert identity["attention_forget_gate_form"] == "sigmoid"
    assert identity["forget_gate_parameter_count"] == (8 + 1) * 2
    assert "log sigmoid(x_t w_f + b_f)" in identity["attention_formula"]
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    gate = report["coefficients"]["forget_gate"]
    assert gate["finite"] and np.all(np.asarray(gate["bias"]) == 0)
    # w_f is small at the start, so every gate starts near sigmoid(0) = 1/2.
    np.testing.assert_allclose(np.asarray(gate["log_alpha_sample"]["mean"]), np.log(0.5), atol=0.1)


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_gate_that_never_decays_changes_nothing(kernel):
    tokens = jnp.arange(16).reshape(2, 8) % 16
    gated = build_model(_config(kernel).model, 19)
    plain = build_model(_config(kernel, attention_forget_gate=False, attention_forget_gate_form="input").model, 19)
    assert np.abs(np.asarray(gated(tokens) - plain(tokens))).max() > 1e-4
    gated.forget_gate_bias[...] = jnp.full_like(gated.forget_gate_bias[...], 200.0)
    np.testing.assert_allclose(gated(tokens), plain(tokens), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kernel", KERNELS)
def test_training_updates_the_gate(kernel):
    config = _config(kernel)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(18).reshape(2, 9) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    second, _, _ = step(first, jax.random.PRNGKey(2))
    assert np.isfinite(loss)
    for name in GATE:
        assert np.linalg.norm(np.asarray(second.params[name][...] - initial.params[name][...])) > 0, name


def test_acceptance_selects_these_checks():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for kernel in KERNELS:
        recipe = config_to_dict(_config(kernel))
        assert acceptance.SIGMOID_GATE_TESTS in acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
        recipe = config_to_dict(_config(kernel, attention_forget_gate_form="input"))
        assert acceptance.SIGMOID_GATE_TESTS not in acceptance.test_paths(
            spec, recipe, resolved_backend="blocked", cpu=True)
