"""The constant (ALiBi-style) form of the forget gate: one rate per head, no token dependence.

The rate is learned from ALiBi's slopes, or held fixed at them with no parameter. On nonlinear
query-coefficient polynomials the decay multiplies every weight by exp(-m_h (t - i)); on softmax and
the unnormalized exponential it adds -m_h (t - i) to every score, which for softmax is ALiBi.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import nnx

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention, strip_dot_product_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import (
    alibi_log_rate, alibi_log_slopes, constant_log_decay, parameter_names, parameter_specs,
)
from scaffolds.transformer.attention.multihead import MultiHeadAttention, attention_from_config
from scaffolds.transformer.attention.parameters import (
    attention_parameter_count, attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
)
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample, forget_gate_report,
)
from scaffolds.transformer.training.state import create_state

KERNELS = ("softmax", "exp")
RATES = ("learned", "fixed")
# Five 8-query scan blocks, or five causal strips, of 37 tokens.
BLOCKED = {"scan": partial(blocked_dot_product_attention, query_block_size=8),
           "strips": partial(strip_dot_product_attention, strip_size=8)}


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


def _exponential_config(kernel="softmax", rate="fixed", **changes):
    """Tiny ALiBi recipes as allocated: no positional encoding, canonical QK RMSNorm, causal strips,
    and per-head output RMSNorm for exp; three heads, so the slopes are not powers of two."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=12, num_heads=3, num_kv_heads=3,
                 qk_head_dim=4, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
                 attention_blocked_implementation="strips", attention_normalization=kernel == "softmax",
                 attention_forget_gate=True, attention_forget_gate_form="constant",
                 attention_forget_gate_constant_rate=rate, positional_encoding="none",
                 attention_output_scaling="none" if kernel == "softmax" else "head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _slopes(heads):
    """ALiBi's slopes 2^(-8 h / H), h = 1..H, in float64, independent of the scaffold."""
    return np.asarray([2.0 ** (-8 * head / heads) for head in range(1, heads + 1)])


def _alibi_reference(q, k, v, *, slopes, normalized):
    """Dense ALiBi: softmax(s + B) V, or exp(s + B) V unnormalized, with B_ti = -m_h (t - i)."""
    time = q.shape[1]
    distance = jnp.arange(time)[:, None] - jnp.arange(time)[None, :]
    causal = distance >= 0
    logits = (jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
              - slopes[None, :, None, None] * distance)
    if normalized:
        weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1)
    else:
        weights = jnp.where(causal, jnp.exp(jnp.where(causal, logits, 0.0)), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def _numpy_alibi_vjp(q, k, v, cotangent, *, normalized):
    """Host FP64 ALiBi and its analytic VJP (dQ, dK, dV), one head at a time, with no JAX compilation."""
    q, k, v, cotangent = (np.asarray(x, np.float64) for x in (q, k, v, cotangent))
    batch, time, heads, width = q.shape
    output, dq, dk, dv = np.zeros_like(v), np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    distance = np.arange(time)[:, None] - np.arange(time)[None, :]
    causal = distance >= 0
    for b in range(batch):
        for h, slope in enumerate(_slopes(heads)):
            logits = q[b, :, h] @ k[b, :, h].T / math.sqrt(width) - slope * distance
            if normalized:
                logits = np.where(causal, logits, -np.inf)
                weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
                weights /= weights.sum(axis=-1, keepdims=True)
            else:
                weights = np.where(causal, np.exp(np.where(causal, logits, 0.0)), 0.0)
            output[b, :, h] = weights @ v[b, :, h]
            dweights = cotangent[b, :, h] @ v[b, :, h].T
            dlogits = weights * (dweights - np.sum(dweights * weights, axis=-1, keepdims=True) if normalized
                                 else dweights)
            dq[b, :, h] = dlogits @ k[b, :, h] / math.sqrt(width)
            dk[b, :, h] = dlogits.T @ q[b, :, h] / math.sqrt(width)
            dv[b, :, h] = weights.T @ cotangent[b, :, h]
    return output, (dq, dk, dv)


def _fixed_alibi_log_decay(batch, time, heads):
    """The attention module's cumulative log decay L [B,T,H] for the fixed rate: L_t - L_i = -m_h (t - i)."""
    return jnp.cumsum(constant_log_decay(jnp.zeros((batch, time, 1)), alibi_log_rate(heads)), axis=1)


def _relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _flat(state):
    return {path: np.asarray(value.get_value()) for path, value in nnx.to_flat_state(state)}


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
    with pytest.raises(ValueError, match="constant forget gate form"):
        _config(**changes)
    model = _config(positional_encoding="none").model
    for name, value in changes.items():
        setattr(model, name, value)  # bypass config validation: the module rejects it too
    with pytest.raises(ValueError, match="constant forget gate form"):
        attention_from_config(model, rngs=nnx.Rngs(0))


def test_recipes_without_a_form_are_the_input_gate_and_select_the_constant_checks_on_cpu():
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


@pytest.mark.parametrize("heads", (1, 3, 8, 14))
def test_one_slope_function_gives_the_fixed_rate_and_the_learned_start(heads):
    """ALiBi's slopes 2^(-8 h / H); eight heads give 1/2 ... 1/256 (Press et al.). The fixed rate is the
    FP32 array the learned rate's initializer stores in every layer."""
    np.testing.assert_allclose(np.exp(alibi_log_slopes(heads)), _slopes(heads), rtol=1e-14)
    if heads == 8:
        np.testing.assert_allclose(_slopes(heads), [2.0 ** -h for h in range(1, 9)], rtol=0)
    rate = alibi_log_rate(heads)
    assert rate.dtype == jnp.float32 and rate.shape == (heads, 1)
    np.testing.assert_array_equal(np.asarray(rate)[:, 0], np.asarray(alibi_log_slopes(heads), np.float32))
    # Rounding a log slope (|log m_h| <= 8 log 2) to FP32 moves m_h by at most 2.4e-7 relative.
    np.testing.assert_allclose(np.exp(np.asarray(rate, np.float64)[:, 0]), _slopes(heads), rtol=3e-7)
    values = initial_attention_parameter_values(parameter_specs(hidden_size=8, form="constant"), num_layers=3,
                                                num_heads=heads, key=mechanism_key(jax.random.PRNGKey(0)))
    np.testing.assert_array_equal(np.asarray(values["forget_gate_log_rate"]),
                                  np.broadcast_to(np.asarray(rate), (3, heads, 1)))


def test_the_fixed_rate_declares_no_parameter():
    assert parameter_names("constant", constant_rate="fixed") == ()
    assert parameter_specs(hidden_size=8, form="constant", constant_rate="fixed") == ()
    assert parameter_specs(hidden_size=8, form="constant", constant_rate="learned") == parameter_specs(
        hidden_size=8, form="constant")
    for form in ("input", "sigmoid"):
        with pytest.raises(ValueError, match="requires the constant forget gate form"):
            parameter_names(form, constant_rate="fixed")
    with pytest.raises(ValueError, match="Unknown constant forget gate rate"):
        parameter_names("constant", constant_rate="alibi")
    configs = {kernel: {rate: _exponential_config(kernel, rate) for rate in RATES} for kernel in KERNELS}
    configs["query"] = {rate: _config(attention_forget_gate_constant_rate=rate) for rate in RATES}
    for kernel, by_rate in configs.items():
        fixed, learned = by_rate["fixed"].model, by_rate["learned"].model
        layers_heads = fixed.num_layers * fixed.num_heads
        assert not any(spec.name.startswith("forget_gate") for spec in attention_parameter_specs(fixed)), kernel
        ungated = (_exponential_config(kernel, attention_forget_gate=False, attention_forget_gate_form="input",
                                       attention_forget_gate_constant_rate="learned").model if kernel != "query"
                   else _config(attention_forget_gate=False, form="input").model)
        assert attention_parameter_count(fixed) == attention_parameter_count(ungated), kernel
        assert model_parameter_count(learned) - model_parameter_count(fixed) == layers_heads, kernel
        state = create_state(by_rate["fixed"], jax.random.PRNGKey(0))
        names = {path[0] for path in _flat(state.params)}
        assert not any(name.startswith("forget_gate") for name in names), kernel
        assert sum(leaf.size for leaf in jax.tree.leaves(state.params)) == model_parameter_count(fixed)
        identity = attention_identity(by_rate["fixed"])
        assert identity["forget_gate_parameter_count"] == 0 and identity["attention_forget_gate_constant_rate"] == "fixed"
        assert "attention_forget_gate_constant_rate" not in attention_identity(by_rate["learned"])
        assert attention_identity(by_rate["learned"])["forget_gate_parameter_count"] == layers_heads


@pytest.mark.parametrize("implementation", tuple(BLOCKED))
@pytest.mark.parametrize("kernel", KERNELS)
def test_blocked_alibi_and_its_derivatives_match_float64(kernel, implementation):
    """The module's decay at ALiBi's fixed slopes, through either blocked implementation, is ALiBi:
    the output and dQ, dK, dV against FP64 softmax(s - m_h (t - i)) (or exp), and the derivative of a
    learned log rate through the cumulative sum against FP64 autodiff."""
    b, t, h, d = 2, 37, 3, 8
    rng = np.random.default_rng(29)
    q, k, v, cotangent = (rng.standard_normal((b, t, h, d)) for _ in range(4))
    normalized = kernel == "softmax"
    attend = BLOCKED[implementation]

    def production(q, k, v, log_rate):
        log_decay = jnp.cumsum(constant_log_decay(q[:, :, 0], log_rate), axis=1)
        return attend(q, k, v, attention_normalization=normalized, query_log_decay=log_decay)

    def exact(q, k, v, log_rate):
        return _alibi_reference(q, k, v, slopes=jnp.exp(log_rate[:, 0]), normalized=normalized)

    log_rate = np.asarray(alibi_log_rate(h), np.float64)
    with jax.enable_x64():
        inputs = tuple(jnp.asarray(x, jnp.float64) for x in (q, k, v, log_rate))
        alibi = np.asarray(_alibi_reference(*inputs[:3], slopes=jnp.asarray(_slopes(h)), normalized=normalized))
        _, pullback = jax.vjp(exact, *inputs)
        gradients64 = [np.asarray(g) for g in pullback(jnp.asarray(cotangent))]
    operands = tuple(jnp.asarray(x, jnp.float32) for x in (q, k, v, log_rate))

    def value_and_gradients(*operands):
        output, pull = jax.vjp(production, *operands)
        return output, pull(jnp.asarray(cotangent, jnp.float32))

    with jax.default_matmul_precision("highest"):
        output, gradients = jax.jit(value_and_gradients)(*operands)
    np.testing.assert_allclose(np.asarray(output), alibi, rtol=2e-5, atol=2e-5)
    for name, value, expected in zip(("q", "k", "v", "log_rate"), gradients, gradients64, strict=True):
        assert np.linalg.norm(expected) > 0, name
        assert _relative_l2(value, expected) < 2e-5, name


@pytest.mark.parametrize("implementation", tuple(BLOCKED))
@pytest.mark.parametrize("kernel", KERNELS)
def test_the_attention_module_computes_alibi(kernel, implementation, monkeypatch):
    """Projections, no positional encoding, the fixed rate, and the output projection against FP64
    ALiBi built from the module's weights; the blocked backend runs the recipe's implementation
    (counted) in 8-query blocks or strips."""
    calls = []

    def counted(original, name, **options):
        def attend(*args, **kwargs):
            calls.append(name)
            return original(*args, **options, **kwargs)
        return attend

    monkeypatch.setattr(reference, "strip_dot_product_attention",
                        counted(reference.strip_dot_product_attention, "strips", strip_size=8))
    monkeypatch.setattr(reference, "blocked_dot_product_attention",
                        counted(reference.blocked_dot_product_attention, "scan", query_block_size=8))
    batch, time, hidden, heads, width = 2, 37, 12, 3, 4
    module = MultiHeadAttention(hidden, heads, heads, width, width, backend="blocked",
                                attention_normalization=kernel == "softmax", positional_encoding="none",
                                attention_forget_gate=True, attention_forget_gate_form="constant",
                                attention_forget_gate_constant_rate="fixed",
                                attention_blocked_implementation=implementation, rngs=nnx.Rngs(0))
    x = np.random.default_rng(5).standard_normal((batch, time, hidden))
    with jax.default_matmul_precision("highest"):
        actual = np.asarray(module(jnp.asarray(x, jnp.float32)))
    assert calls == [implementation]
    weights = {name: np.asarray(getattr(module, f"{name}_proj").kernel[...], np.float64) for name in "qkvo"}
    with jax.enable_x64():
        q, k, v = (jnp.asarray((x @ weights[name]).reshape(batch, time, heads, width)) for name in "qkv")
        attention = _alibi_reference(q, k, v, slopes=jnp.asarray(_slopes(heads)), normalized=kernel == "softmax")
        expected = np.asarray(attention).reshape(batch, time, heads * width) @ weights["o"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("kernel", (*KERNELS, "query"))
def test_learned_and_fixed_rates_agree_at_initialization(kernel):
    """The same backbone, decays, logits, and gradients of every shared parameter; only the learned
    rate's array is extra. They agree to rounding: the fixed rate's exp may be evaluated at compile time."""
    configs = {rate: (_config(attention_forget_gate_constant_rate=rate) if kernel == "query"
                      else _exponential_config(kernel, rate)) for rate in RATES}
    states = {rate: create_state(config, jax.random.PRNGKey(3)) for rate, config in configs.items()}
    flat = {rate: _flat(state.params) for rate, state in states.items()}
    assert set(flat["learned"]) - set(flat["fixed"]) == {("forget_gate_log_rate",)}
    assert set(flat["fixed"]) <= set(flat["learned"])
    for path, value in flat["fixed"].items():
        np.testing.assert_array_equal(value, flat["learned"][path], err_msg=str(path))
    model = configs["fixed"].model
    slopes = np.broadcast_to(_slopes(model.num_heads), (model.num_layers, model.num_heads))
    batch = jnp.asarray(np.random.default_rng(2).integers(0, 16, (2, 9)), jnp.int32)
    logits, decays, gradients = {}, {}, {}
    for rate, state in states.items():
        output, diagnostics = nnx.merge(state.graphdef, state.params)(batch[:, :-1], deterministic=True,
                                                                        return_diagnostics=True)
        logits[rate] = np.asarray(output)
        decays[rate] = np.stack([np.asarray(raw[f"forget_gate_log_alpha_{statistic}"])
                                 for raw in diagnostics for statistic in ("mean", "min", "max")])
        gradients[rate] = _flat(loop.loss_and_grads(state, batch, jax.random.PRNGKey(0))[1])
    np.testing.assert_allclose(decays["fixed"], np.repeat(-slopes, 3, axis=0), rtol=1e-6)
    np.testing.assert_allclose(decays["fixed"], decays["learned"], rtol=1e-6)
    np.testing.assert_allclose(logits["fixed"], logits["learned"], rtol=1e-5, atol=1e-6)
    assert set(gradients["learned"]) - set(gradients["fixed"]) == {("forget_gate_log_rate",)}
    for path, value in gradients["fixed"].items():
        expected = gradients["learned"][path]
        assert np.linalg.norm(value - expected) <= 1e-5 * np.linalg.norm(expected) + 1e-9, path


def test_recipes_without_a_rate_learn_it(tmp_path):
    """Historical recipes predate the field: absent, the constant form keeps its learned rate."""
    current = config_to_dict(_config())
    assert current["model"]["attention_forget_gate_constant_rate"] == "learned"
    legacy = {**current, "model": {key: value for key, value in current["model"].items()
                                   if key != "attention_forget_gate_constant_rate"}}
    path = tmp_path / "historical.yaml"
    path.write_text(yaml.safe_dump(legacy, sort_keys=False))
    historical = load_config(path)
    assert historical.model.attention_forget_gate_constant_rate == "learned"
    assert "forget_gate_log_rate" in {spec.name for spec in attention_parameter_specs(historical.model)}
    assert model_parameter_count(historical.model) == model_parameter_count(_config().model)
    assert config_dicts_equal(legacy, current)
    assert not config_dicts_equal(legacy, config_to_dict(_config(attention_forget_gate_constant_rate="fixed")))
    # Every other form, and a recipe without any gate, keeps the default as well.
    for config in (_exponential_config(attention_forget_gate_form="input", attention_forget_gate_constant_rate="learned"),
                   _config("sigmoid"), _config(attention_forget_gate=False, form="input")):
        recipe = config_to_dict(config)
        assert recipe["model"]["attention_forget_gate_constant_rate"] == "learned"
        del recipe["model"]["attention_forget_gate_constant_rate"]
        assert config_dicts_equal(recipe, config_to_dict(config))


@pytest.mark.parametrize("kernel", KERNELS)
def test_alibi_is_accepted_on_softmax_and_exp(kernel):
    """Learned or fixed slopes, RoPE or no positional encoding, the scan or the strips, on the blocked backend."""
    for rate in RATES:
        for encoding in ("none", "rope"):
            for implementation in ("scan", "strips"):
                config = _exponential_config(kernel, rate, positional_encoding=encoding,
                                             attention_blocked_implementation=implementation)
                module = attention_from_config(config.model, rngs=nnx.Rngs(0))
                assert module.attention_forget_gate_form == "constant"
                assert module.attention_forget_gate_constant_rate == rate
    assert _config(attention_forget_gate_constant_rate="fixed").model.attention_forget_gate_constant_rate == "fixed"


PATH = {"positional_encoding": "path", "qk_head_dim": 16, "value_head_dim": 16, "attention_blocked_implementation": "scan"}
INVALID = (
    ({"attention_forget_gate_form": "input"}, "fixed forget gate rate requires the constant forget gate form"),
    ({"attention_forget_gate_form": "sigmoid"}, "fixed forget gate rate requires the constant forget gate form"),
    ({"attention_forget_gate": False, "attention_forget_gate_form": "input"},
     "fixed forget gate rate requires the constant forget gate form"),
    ({"attention_forget_gate_constant_rate": "alibi"}, "Unknown constant forget gate rate"),
    ({"attention_forget_gate_mode": "query_distance"}, "constant forget gate form"),
    (PATH, "constant forget gate form"),
    ({**PATH, "attention_backend": "fused"}, "constant forget gate form"),
    ({"attention_degree": 2}, "constant forget gate form"),
    ({"attention_backend": "fused", "positional_encoding": "rope"}, None),
    ({"attention_backend": "mosaic", "positional_encoding": "rope"}, None),
    ({"attention_backend": "cudnn", "positional_encoding": "rope"}, None),
)


@pytest.mark.parametrize("change,message", INVALID, ids=[",".join(f"{k}={v}" for k, v in c.items()) for c, _ in INVALID])
@pytest.mark.parametrize("kernel", KERNELS)
def test_unqualified_alibi_recipes_are_rejected(kernel, change, message):
    """Query-distance decay, PaTH, Taylor kernels, other backends, and a fixed rate on another form."""
    with pytest.raises(ValueError, match=message):
        _exponential_config(kernel, "fixed", **change)
    model = _exponential_config(kernel, "fixed").model
    for name, value in change.items():
        setattr(model, name, value)  # bypass config validation: the module rejects it too
    with pytest.raises(ValueError):
        attention_from_config(model, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("kernel", KERNELS)
def test_acceptance_selects_the_alibi_checks_on_cpu_and_gpu(kernel, rate):
    """attention_spec runs this module on CPU and its production-context cases on the GPU, with the
    forget-gate and causal-strips checks; the polynomial constant form keeps its CPU-only selection."""
    config = _exponential_config(kernel, rate)
    spec = acceptance.attention_spec(config, entry_module="tests")
    recipe = config_to_dict(config)
    cpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=True)
    gpu = acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=False)
    for paths in (cpu, gpu):
        assert acceptance.CONSTANT_DECAY_TESTS in paths and acceptance.FORGET_GATE_ATTENTION_TESTS in paths
    assert acceptance.BLOCKED_STRIPS_TESTS in cpu
    gpu_cases = [name for name, test in globals().items() if name.startswith("test_")
                 and any(mark.name == "gpu" for mark in getattr(test, "pytestmark", ()))]
    assert sorted(gpu_cases) == ["test_gpu_strips_alibi_matches_float64_at_the_production_context"]


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("kernel", KERNELS)
def test_acceptance_initialization_is_the_softmax_backbone(kernel, rate):
    facts = acceptance.matches_softmax_backbone_initialization(_exponential_config(kernel, rate))
    gate = facts["mechanism_parameters"].get("forget_gate_log_rate")
    assert (gate is None) == (rate == "fixed")
    if gate is not None:
        assert gate["initializer"] == "alibi_log_slope" and gate["weight_decay"] is False


@pytest.mark.parametrize("kernel", KERNELS)
def test_fixed_rate_identity_and_diagnostics(kernel):
    config = _exponential_config(kernel)
    model = config.model
    identity = attention_identity(config)
    assert identity["attention_forget_gate_form"] == "constant"
    assert identity["attention_kernel"] == ("softmax" if kernel == "softmax" else "exp_unnormalized")
    assert identity["attention_formula"].startswith(
        ("causal_softmax" if kernel == "softmax" else "causal_exp")
        + "(QK^T / sqrt(d_k) + D) V, D_ti = -m_h (t - i), m_h = 2^(-8 h / H)")
    assert identity["attention_implementation"] == "blocked_causal_strips"
    learned = attention_identity(_exponential_config(kernel, "learned"))
    assert "A_log starting at ALiBi's log slopes" in learned["attention_formula"]
    assert "forget_gate" in tracking.run_tags(config, {})
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    gate = report["coefficients"]["forget_gate"]
    slopes = np.broadcast_to(_slopes(model.num_heads), (model.num_layers, model.num_heads))
    assert gate["finite"] and "rate_exp_A_log" not in gate
    np.testing.assert_allclose(np.asarray(gate["fixed_rate"]), slopes, rtol=1e-6)
    for statistic in ("mean", "min", "max"):
        np.testing.assert_allclose(np.asarray(gate["log_alpha_sample"][statistic]), -slopes, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(gate["mean_half_life_tokens"]), math.log(2) / slopes, rtol=1e-5)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("kernel", KERNELS)
def test_training_updates_only_a_learned_rate(kernel, rate):
    config = _exponential_config(kernel, rate)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(18).reshape(2, 9) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    second, _, _ = step(first, jax.random.PRNGKey(2))
    assert np.isfinite(loss)
    names = {path[0] for path in _flat(second.params)}
    assert {name for name in names if name.startswith("forget_gate")} == (
        {"forget_gate_log_rate"} if rate == "learned" else set())
    if rate == "learned":
        before, after = (np.asarray(state.params["forget_gate_log_rate"][...]) for state in (initial, second))
        assert np.linalg.norm(after - before) > 0


@pytest.mark.parametrize("time", (1, 5, 19))
@pytest.mark.parametrize("kernel", KERNELS)
def test_numpy_alibi_oracle_matches_float64_autodiff(kernel, time):
    """The host oracle of the GPU case: output and analytic dQ, dK, dV against FP64 JAX autodiff."""
    rng = np.random.default_rng(time)
    q, k, v, cotangent = (rng.standard_normal((2, time, 3, 4)) for _ in range(4))
    output, gradients = _numpy_alibi_vjp(q, k, v, cotangent, normalized=kernel == "softmax")
    with jax.enable_x64():
        reference_function = partial(_alibi_reference, slopes=jnp.asarray(_slopes(3)), normalized=kernel == "softmax")
        expected, pullback = jax.vjp(reference_function, *(jnp.asarray(x) for x in (q, k, v)))
        expected_gradients = pullback(jnp.asarray(cotangent))
    for observed, wanted in zip((output, *gradients), (expected, *expected_gradients), strict=True):
        np.testing.assert_allclose(observed, np.asarray(wanted), rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", KERNELS)
def test_gpu_strips_alibi_matches_float64_at_the_production_context(kernel):
    """BF16 Q/K/V at the production context, head count, and head width through the module's fixed
    decay and the causal strips: the output and dQ, dK, dV within 1.2x their BF16 rounding floor (the
    host FP64 oracle rounded to BF16) against host FP64 ALiBi of the same BF16 inputs."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    batch, time, heads, width = 1, 2048, 14, 64
    rng = np.random.default_rng(23)
    q, k, v, cotangent = (jnp.asarray(rng.standard_normal((batch, time, heads, width)), jnp.bfloat16)
                          for _ in range(4))
    attend = partial(attention_core, backend="blocked", attention_normalization=kernel == "softmax",
                     query_log_decay=_fixed_alibi_log_decay(batch, time, heads), blocked_implementation="strips")

    def value_and_gradients(q, k, v, cotangent):
        output, pullback = jax.vjp(attend, q, k, v)
        return (output, *pullback(cotangent))

    observed = jax.jit(value_and_gradients)(q, k, v, cotangent)
    output, gradients = _numpy_alibi_vjp(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
                                         cotangent.astype(jnp.float32), normalized=kernel == "softmax")
    for name, got, want in zip(("output", "dq", "dk", "dv"), observed, (output, *gradients), strict=True):
        assert got.dtype == jnp.bfloat16, name
        got = np.asarray(got.astype(jnp.float32), np.float64)
        assert np.isfinite(got).all(), name
        floor = _relative_l2(want.astype(jnp.bfloat16).astype(np.float64), want)
        error = _relative_l2(got, want)
        assert error < 1.2 * floor, f"{name}: relative L2 error {error:.3e}, BF16 floor {floor:.3e}"
