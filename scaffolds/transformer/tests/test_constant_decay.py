"""The constant (ALiBi-style) form of the forget gate: one learned rate per head, no token dependence."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.forget_gate import alibi_log_slopes, constant_log_decay, parameter_names, parameter_specs
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_identity, forget_gate_report
from scaffolds.transformer.training.state import create_state


def _config(form="constant", **changes):
    """Degree-four tanh query coefficients with a forget gate on the blocked backend."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=4,
                 attention_backend="blocked", attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.0,
                 attention_coefficient_projection_initial_std=8 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_forget_gate=True, attention_forget_gate_form=form,
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def test_constant_form_has_one_parameter_at_alibi_slopes():
    assert parameter_names("constant") == ("forget_gate_log_rate",)
    assert len(parameter_names("input")) == 3
    with pytest.raises(ValueError):
        parameter_names("other")
    specs = {spec.name: spec for spec in parameter_specs(hidden_size=8, form="constant")}
    assert set(specs) == {"forget_gate_log_rate"} and specs["forget_gate_log_rate"].initializer == "alibi_log_slope"
    config = _config()
    assert {spec.name for spec in attention_parameter_specs(config.model) if spec.name.startswith("forget")} == {
        "forget_gate_log_rate"}
    state = create_state(config, jax.random.PRNGKey(0))
    log_rate = np.asarray(state.params["forget_gate_log_rate"][...])
    assert log_rate.shape == (2, 2, 1)
    expected = np.asarray(alibi_log_slopes(2), np.float32)
    np.testing.assert_allclose(log_rate[:, :, 0], np.broadcast_to(expected, (2, 2)), rtol=1e-6)
    np.testing.assert_allclose(np.exp(expected), [2 ** -4, 2 ** -8], rtol=1e-6)
    count = sum(leaf.size for leaf in jax.tree.leaves(state.params))
    assert count == model_parameter_count(config.model)
    assert model_parameter_count(_config("input").model) - count == 2 * 2 * (8 + 1)
    identity = attention_identity(config)
    assert identity["attention_forget_gate_form"] == "constant" and identity["forget_gate_parameter_count"] == 4
    assert "ALiBi" in identity["attention_formula"]


def test_acceptance_softmax_comparison_drops_the_form_with_the_gate():
    """The acceptance's derived softmax controls must validate without the gate."""
    from scaffolds.transformer import acceptance

    config = _config(positional_encoding="rope")
    facts = acceptance.matches_softmax_backbone_initialization(config)  # raises on any mismatch
    assert facts["mechanism_parameters"]["forget_gate_log_rate"]["initializer"] == "alibi_log_slope"


def test_constant_form_requires_the_gate():
    with pytest.raises(ValueError, match="requires the forget gate"):
        _config(attention_forget_gate=False)
    with pytest.raises(ValueError, match="Unknown forget gate form"):
        _config("alibi")


@pytest.mark.parametrize("changes", (
    {"attention_forget_gate_mode": "query_distance"},
    {"positional_encoding": "path"},
    {"positional_encoding": "path", "attention_backend": "fused", "qk_head_dim": 16, "value_head_dim": 16},
))
def test_constant_form_is_qualified_only_as_cumulative_decay_with_rope_or_none(changes):
    """PaTH (reference and fused kernels) and query-distance decay stay unqualified for the constant form."""
    from scaffolds.transformer.attention.multihead import attention_from_config

    with pytest.raises(ValueError, match="constant forget gate form"):
        _config(**changes)
    model = _config(positional_encoding="none").model
    for name, value in changes.items():
        setattr(model, name, value)  # bypass config validation: the module rejects it too
    with pytest.raises(ValueError, match="constant forget gate form"):
        attention_from_config(model, rngs=nnx.Rngs(0))


def test_recipes_without_a_form_are_the_input_gate_and_select_the_constant_checks_on_cpu():
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.configs import config_to_dict
    from scaffolds.transformer.configs.loader import config_dicts_equal

    gate = config_to_dict(_config("input"))
    legacy = {**gate, "model": {k: v for k, v in gate["model"].items() if k != "attention_forget_gate_form"}}
    assert config_dicts_equal(gate, legacy) and not config_dicts_equal(config_to_dict(_config()), legacy)
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for form in ("constant", "input"):
        for cpu in (True, False):
            paths = acceptance.test_paths(spec, config_to_dict(_config(form)), resolved_backend="blocked", cpu=cpu)
            assert (acceptance.CONSTANT_DECAY_TESTS in paths) == (cpu and form == "constant")


def test_constant_log_decay_is_the_rate_for_every_token():
    x = jnp.zeros((2, 5, 8), jnp.float32)
    log_rate = jnp.asarray([[math.log(0.5)], [math.log(0.01)]], jnp.float32)
    decay = np.asarray(constant_log_decay(x, log_rate))
    assert decay.shape == (2, 5, 2)
    np.testing.assert_allclose(decay, np.broadcast_to([-0.5, -0.01], (2, 5, 2)), rtol=1e-6)
    with pytest.raises(ValueError):
        constant_log_decay(x, log_rate[:, 0])


def test_model_applies_the_same_decay_to_every_token_and_learns_the_rate():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(1))
    batch = jnp.asarray(np.random.default_rng(0).integers(0, 16, (2, 9)), jnp.int32)
    _, diagnostics = nnx.merge(state.graphdef, state.params)(batch[:, :-1], deterministic=True,
                                                               return_diagnostics=True)
    rate = np.exp(np.asarray(state.params["forget_gate_log_rate"][...])[..., 0])
    for layer, raw in enumerate(diagnostics):
        for statistic in ("mean", "min", "max"):
            np.testing.assert_allclose(np.asarray(raw[f"forget_gate_log_alpha_{statistic}"]), -rate[layer], rtol=1e-6)
    report = forget_gate_report(state.params, diagnostics, form="constant")
    assert report["finite"] and "softplus_dt_bias" not in report
    np.testing.assert_allclose(np.asarray(report["mean_half_life_tokens"]), math.log(2) / rate, rtol=1e-5)
    after, *_ = loop.train_step(state, batch, jax.random.PRNGKey(0))
    assert np.any(np.asarray(after.params["forget_gate_log_rate"][...]) != np.asarray(state.params["forget_gate_log_rate"][...]))
