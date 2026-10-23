"""Head-shared channel gains, arithmetic, telemetry, and compatibility for QK RMSNorm."""

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.parameters import attention_parameter_specs, parameters_for_layer
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import rms_score_operands
from scaffolds.transformer.configs import load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking, telemetry
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.state import create_state

EPS = 1e-6


def rms_config(path):
    config = tiny_config(path)
    config.model = replace(config.model, qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer")
    validate_config(config)
    return config


def _reference_operands(q, k, q_gain, k_gain):
    def normalize(x, gain):
        return x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + EPS) * gain
    return normalize(q, q_gain), normalize(k, k_gain)


@pytest.mark.parametrize("degree,normalized", [(None, True), (None, False), (8, False), (12, False)])
def test_operator_and_all_gradients_match_independent_fp64(degree, normalized):
    rng = np.random.default_rng(13)
    q, k = (jnp.asarray(rng.normal(size=(2, 7, 3, 8)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, 7, 3, 5)), jnp.float32)
    q_gain, k_gain = (jnp.asarray(rng.uniform(0.4, 1.2, size=8), jnp.float32) for _ in range(2))
    cotangent = jnp.asarray(rng.normal(size=v.shape), jnp.float32)

    def actual(q, k, v, q_gain, k_gain):
        q, k = rms_score_operands(q, k, q_gain, k_gain, eps=EPS)
        return blocked_dot_product_attention(q, k, v, attention_degree=degree,
                                             attention_normalization=normalized, query_block_size=4)

    def reference(q, k, v, q_gain, k_gain):
        q, k = _reference_operands(q, k, q_gain, k_gain)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
        weights = (jnp.exp(scores) if degree is None else
                   sum(scores ** r / math.factorial(r) for r in range(degree + 1)))
        weights = jnp.where(jnp.tril(jnp.ones(scores.shape[-2:], bool)), weights, 0)
        if normalized:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return jnp.einsum("bhts,bshd->bthd", weights, v)

    operands = (q, k, v, q_gain, k_gain)
    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(actual, *operands)
        gradients = pullback(cotangent)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected, pullback = jax.vjp(reference, *(x.astype(jnp.float64) for x in operands))
        expected_gradients = pullback(cotangent.astype(jnp.float64))
    for got, want in zip((output, *gradients), (expected, *expected_gradients), strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=4e-6)


def test_bf16_uses_fp32_statistics_and_keeps_head_statistics_independent():
    q = jnp.asarray([[[[10000, -20000, 30000, 40000], [0, 0, 0, 0]],
                      [[1, 1, 1, 1], [1, 2, 3, 4]]]], jnp.bfloat16)
    gains = jnp.asarray([0.3, 0.8, 1.1, 1.9], jnp.float32)
    got_q, got_k = rms_score_operands(q, q, gains, jnp.ones(4), eps=EPS)
    values = np.asarray(q, dtype=np.float64)
    expected = values / np.sqrt(np.mean(values ** 2, axis=-1, keepdims=True) + EPS)
    np.testing.assert_array_equal(got_q, jnp.asarray(expected * np.asarray(gains), jnp.bfloat16))
    np.testing.assert_array_equal(got_k, jnp.asarray(expected, jnp.bfloat16))
    assert got_q.dtype == got_k.dtype == jnp.bfloat16
    np.testing.assert_array_equal(got_q[0, 0, 1], jnp.zeros(4))


def test_layer_owns_two_shared_gain_vectors_without_decay(tmp_path):
    config = rms_config(tmp_path)
    config.model = replace(config.model, num_layers=2)
    model = build_model(config.model, 0)
    params = nnx.state(model, nnx.Param)
    specs = attention_parameter_specs(config.model)
    assert {spec.name for spec in specs} == {"qk_norm_q_gain", "qk_norm_k_gain"}
    for spec in specs:
        assert spec.sharing == "per_layer" and not spec.weight_decay
        gain = params[spec.name][...]
        assert gain.shape == (2, config.model.qk_head_dim) and gain.dtype == jnp.float32
        np.testing.assert_array_equal(gain, 1)
    assert "qk_norm_log_scale" not in params
    without = model_parameter_count(replace(config.model, qk_norm=False))
    assert model_parameter_count(config.model) - without == 2 * 2 * config.model.qk_head_dim
    assert sum(x.size for x in jax.tree.leaves(params)) == model_parameter_count(config.model)
    model.qk_norm_q_gain[0] = jnp.asarray([0.5, 1.5])
    np.testing.assert_array_equal(parameters_for_layer(model, specs, 1)["qk_norm_q_gain"], 1)
    np.testing.assert_array_equal(model.qk_norm_k_gain[...], 1)
    params = nnx.state(model, nnx.Param)
    tx = make_optimizer(config)
    updates, _ = tx.update(jax.tree.map(jnp.zeros_like, params), tx.init(params), params)
    updated = optax.apply_updates(params, updates)
    for name in ("qk_norm_q_gain", "qk_norm_k_gain"):
        np.testing.assert_array_equal(updated[name][...], params[name][...])
        assert telemetry.parameter_group([jax.tree_util.DictKey(name)]) == "qk_norm_gains"


def test_channel_gains_apply_before_rope_and_require_both_parameters(monkeypatch):
    attention = multihead.MultiHeadAttention(
        8, 2, 2, 4, 4, qk_norm=True, qk_norm_type="rms_norm", rngs=nnx.Rngs(9))
    x = jax.random.normal(jax.random.key(4), (1, 7, 8))
    parameters = {"qk_norm_q_gain": jnp.asarray([0.3, 0.8, 1.1, 1.9]),
                  "qk_norm_k_gain": jnp.asarray([1.8, 1.1, 0.8, 0.3])}
    captured = {}
    def capture(q, k, v, **_kwargs):
        captured.update(q=q, k=k)
        return v
    monkeypatch.setattr(multihead, "attention_core", capture)
    attention(x, attention_parameters=parameters)
    q, k = attention.q_proj(x).reshape(1, 7, 2, 4), attention.k_proj(x).reshape(1, 7, 2, 4)
    normalized = _reference_operands(q, k, parameters["qk_norm_q_gain"], parameters["qk_norm_k_gain"])
    for name, operand in zip(("q", "k"), normalized, strict=True):
        np.testing.assert_allclose(captured[name], apply_rope(operand, jnp.arange(7)), rtol=1e-6, atol=1e-6)
    wrong_order = _reference_operands(apply_rope(q, jnp.arange(7)), apply_rope(k, jnp.arange(7)),
                                      parameters["qk_norm_q_gain"], parameters["qk_norm_k_gain"])[0]
    assert np.max(np.abs(np.asarray(captured["q"] - wrong_order))) > 0.1
    with pytest.raises(ValueError, match="qk_norm_q_gain"):
        attention(x, attention_parameters={"qk_norm_k_gain": parameters["qk_norm_k_gain"]})
    with pytest.raises(ValueError, match="qk_norm_log_scale"):
        attention(x, attention_parameters={**parameters, "qk_norm_log_scale": jnp.ones(1)})


def test_config_roundtrip_and_historical_cosine_compatibility(tmp_path):
    config = rms_config(tmp_path)
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert config_to_dict(load_config(path)) == config_to_dict(config)
    historical = replace(config, model=replace(config.model, qk_norm_type="cosine_log_scale", qk_norm_sharing="per_head"))
    legacy = config_to_dict(historical)
    legacy["model"].pop("qk_norm_type")
    legacy["train"].pop("eval_early_every")
    legacy["train"].pop("eval_early_until")
    assert config_dicts_equal(legacy, config_to_dict(historical))
    assert not config_dicts_equal(legacy, config_to_dict(config))
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy))
    restored = load_config(legacy_path)
    assert restored.model.qk_norm_type == "cosine_log_scale"
    old, explicit = (nnx.state(build_model(c.model, 17), nnx.Param) for c in (restored, historical))
    for left, right in zip(jax.tree.leaves(old), jax.tree.leaves(explicit), strict=True):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="sharing"):
        replace(config.model, qk_norm_sharing="per_head")
    with pytest.raises(ValueError, match="initial_scale"):
        replace(config.model, qk_norm_initial_scale=8)


def test_diagnostics_and_wandb_report_gains_without_scalar_temperature(tmp_path):
    config = rms_config(tmp_path)
    state = create_state(config, jax.random.key(0))
    state.params["qk_norm_q_gain"][...] = jnp.asarray([[0.5, 1.5]])
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"] and report["qk_norm_type"] == "rms_norm"
    assert report["qk_norm_parameter_count"] == 4
    assert "qk_norm_initial_scale" not in report and "qk_norm_scale" not in report
    gains = np.asarray(report["qk_norm_gains"]["effective"])
    np.testing.assert_array_equal(gains[..., 0], [[0.5, 1.5]])
    np.testing.assert_array_equal(gains[..., 1], [[1, 1]])
    metrics = tracking.attention_metrics(report)
    assert metrics["attention/softmax/qk_norm_gains/g_Q/max"] == 1.5
    state.params["qk_norm_q_gain"][...] = jnp.full((1, 2), jnp.inf)
    report = attention_diagnostics(state, config, sample, provenance)
    assert not report["valid"] and "Nonfinite QK RMSNorm gains" in report["issues"]
    json.dumps(report, allow_nan=False)


def test_training_and_resume_preserve_gains_and_optimizer_state(tmp_path):
    config = rms_config(tmp_path / "full")
    config.train.dtype = "bfloat16"
    loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    outcome = loop.run_training(resumed_config, audit_checkpoint=True)
    assert outcome["restored_step"] == 1
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == config.optim.total_steps
            states.append(state)
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    for name in ("qk_norm_q_gain", "qk_norm_k_gain"):
        assert np.any(np.asarray(states[0].params[name][...]) != 1.0)
