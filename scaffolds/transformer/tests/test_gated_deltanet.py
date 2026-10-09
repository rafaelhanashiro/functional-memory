"""CPU contracts for the Gated DeltaNet token mixer, its reference recurrence, and its recipe."""

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import attention_test_paths
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.gated_deltanet import (
    GatedDeltaNet, ShortConvolution, gated_delta_rule_reference,
)
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training import telemetry
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state


def gdn_model(**updates):
    values = dict(vocab_size=128, max_seq_len=16, num_layers=2, hidden_size=32, num_heads=4, num_kv_heads=4,
                  qk_head_dim=8, value_head_dim=8, intermediate_size=64, tie_embeddings=False,
                  attention_backend="blocked", attention_mechanism="gated_deltanet")
    return ModelConfig(**{**values, **updates})


def gdn_config(path, **updates):
    config = tiny_config(path)
    config.model = replace(config.model, attention_mechanism="gated_deltanet", attention_backend="blocked",
                           qk_norm=False, **updates)
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-3, 1e-4
    validate_config(config)
    return config


def _naive_layer(module, x):
    """Independent NumPy FP64 forward of FLA's GatedDeltaNet layer for one parameter set."""
    p = {name: np.asarray(value[...], np.float64) for name, value in nnx.to_flat_state(nnx.state(module, nnx.Param))
         for name in ["/".join(str(part) for part in name)]}
    x = np.asarray(x, np.float64)
    batch, length, _ = x.shape
    heads, dk, dv = module.num_heads, module.key_head_dim, module.value_head_dim
    silu = lambda z: z / (1 + np.exp(-z))

    def conv(z, w):
        padded = np.concatenate([np.zeros((batch, w.shape[0] - 1, z.shape[-1])), z], axis=1)
        return silu(sum(padded[:, i:i + length] * w[i] for i in range(w.shape[0])))

    q = conv(x @ p["q_proj/kernel"], p["q_conv/weight"]).reshape(batch, length, heads, dk)
    k = conv(x @ p["k_proj/kernel"], p["k_conv/weight"]).reshape(batch, length, heads, dk)
    v = conv(x @ p["v_proj/kernel"], p["v_conv/weight"]).reshape(batch, length, heads, dv)
    q = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)
    beta = 1 / (1 + np.exp(-(x @ p["b_proj/kernel"])))
    erase = 1 / (1 + np.exp(-(x @ p["lambda_proj/kernel"]))) if module.erase_mode == "independent" else beta
    g = -np.exp(p["A_log"]) * np.log1p(np.exp(x @ p["a_proj/kernel"] + p["dt_bias"]))
    out = np.zeros((batch, length, heads, dv))
    state = np.zeros((batch, heads, dk, dv))
    for t in range(length):
        state = state * np.exp(g[:, t])[..., None, None]
        prediction = np.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + np.einsum("bhk,bhv->bhkv", k[:, t],
                                  beta[:, t][..., None] * v[:, t] - erase[:, t][..., None] * prediction)
        out[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t] / math.sqrt(dk), state)
    gate = (x @ p["g_proj/kernel"]).reshape(batch, length, heads, dv)
    normed = out / np.sqrt((out ** 2).mean(-1, keepdims=True) + module.norm_eps)
    y = normed * p["o_norm_scale"] * silu(gate)
    return y.reshape(batch, length, heads * dv) @ p["o_proj/kernel"]


@pytest.mark.parametrize("erase_mode", ["tied", "independent"])
def test_layer_matches_an_independent_fp64_implementation(erase_mode):
    module = GatedDeltaNet(32, 4, 8, 8, conv_size=4, erase_mode=erase_mode, rngs=nnx.Rngs(3))
    if erase_mode == "independent":
        module.lambda_proj.kernel[...] = 0.3 * module.b_proj.kernel[...] + 0.1
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 13, 32)), jnp.float32)
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(module(x), _naive_layer(module, x), rtol=2e-5, atol=2e-6)


def test_reference_recurrence_reduces_to_known_cases():
    rng = np.random.default_rng(1)
    q, k = (rng.normal(size=(1, 6, 2, 4)) for _ in range(2))
    q, k = (z / np.linalg.norm(z, axis=-1, keepdims=True) for z in (q, k))
    v = rng.normal(size=(1, 6, 2, 3))
    # beta = 0 never writes; the state stays zero.
    zero = gated_delta_rule_reference(q, k, v, np.zeros((1, 6, 2)), np.zeros((1, 6, 2)), 1.0)
    assert np.all(np.asarray(zero) == 0)
    # beta = 1 with unit keys stores v exactly along k: reading with q = k returns v_t (no decay).
    stored = gated_delta_rule_reference(k, k, v, np.zeros((1, 6, 2)), np.ones((1, 6, 2)), 1.0)
    np.testing.assert_allclose(stored, v, rtol=1e-5, atol=1e-5)


def test_short_convolution_is_causal_and_starts_at_pytorch_default_bound():
    conv = ShortConvolution(5, 4, rngs=nnx.Rngs(0))
    assert np.abs(np.asarray(conv.weight[...])).max() <= 0.5
    x = jnp.asarray(np.random.default_rng(2).normal(size=(1, 9, 5)), jnp.float32)
    perturbed = x.at[:, 6].add(10.0)
    np.testing.assert_array_equal(conv(x)[:, :6], conv(perturbed)[:, :6])
    assert not np.allclose(conv(x)[:, 6:], conv(perturbed)[:, 6:])


def test_decay_parameters_follow_mamba2_and_skip_weight_decay():
    module = GatedDeltaNet(32, 64, 8, 8, rngs=nnx.Rngs(4))
    decay_rate = np.exp(np.asarray(module.A_log[...]))
    assert np.all((decay_rate > 0) & (decay_rate < 16))
    dt = np.log1p(np.exp(np.asarray(module.dt_bias[...], np.float64)))
    assert np.all((dt >= 1e-3 - 1e-9) & (dt <= 0.1 + 1e-9))
    for parameter in (module.A_log, module.dt_bias, module.o_norm_scale):
        assert parameter.get_metadata("weight_decay", True) is False


@pytest.mark.parametrize("erase_mode", ["tied", "independent"])
def test_parameter_count_matches_the_formula_across_key_widths(erase_mode):
    for key_dim in (2, 8, 16):
        config = gdn_model(qk_head_dim=key_dim, gdn_erase_mode=erase_mode)
        params = nnx.state(build_model(config, jax.random.PRNGKey(0)), nnx.Param)
        assert sum(leaf.size for leaf in jax.tree.leaves(params)) == model_parameter_count(config)


@pytest.mark.parametrize("updates,error", [
    (dict(attention_degree=2), "Taylor"),
    (dict(attention_normalization=False), "recurrence"),
    (dict(attention_backend="manual"), "backend"),
    (dict(qk_norm=True), "QK-norm"),
    (dict(gdn_conv_size=0), "gdn_conv_size"),
    (dict(gdn_erase_mode="unknown"), "gdn_erase_mode"),
    (dict(initialization="gla_reference"), "initialization"),
])
def test_invalid_configurations_are_rejected(updates, error):
    with pytest.raises(ValueError, match=error):
        gdn_model(**updates)


@pytest.mark.parametrize("mechanism", ["dot_product", "gla"])
def test_independent_erase_is_rejected_for_other_mechanisms(mechanism):
    with pytest.raises(ValueError, match="gdn_erase_mode"):
        gdn_model(attention_mechanism=mechanism, gdn_erase_mode="independent")


def test_independent_erase_recovers_addition_and_can_erase_without_writing():
    rng = np.random.default_rng(11)
    q, k = (jnp.asarray(rng.normal(size=(1, 7, 2, 4)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(1, 7, 2, 3)), jnp.float32)
    zeros, ones = jnp.zeros((1, 7, 2)), jnp.ones((1, 7, 2))
    actual = gated_delta_rule_reference(q, k, v, zeros, ones, 1.0, erase=zeros)
    scores = jnp.einsum("bthk,bshk->bhts", q, k)
    expected = jnp.einsum("bhts,bshv->bthv", jnp.tril(scores), v)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    keys = jnp.ones((1, 2, 1, 1))
    values = jnp.array([2., 9.]).reshape(1, 2, 1, 1)
    beta = jnp.array([1., 0.]).reshape(1, 2, 1)
    erase = jnp.array([0., 1.]).reshape(1, 2, 1)
    actual = gated_delta_rule_reference(keys, keys, values, jnp.zeros_like(beta), beta, 1., erase=erase)
    np.testing.assert_array_equal(actual.reshape(-1), [2., 0.])


def test_independent_erase_recovers_tied_outputs_and_gradient_chain_rule():
    rng = np.random.default_rng(12)
    q, k = (jnp.asarray(rng.normal(size=(1, 7, 2, 4)), jnp.float32) for _ in range(2))
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jnp.asarray(rng.normal(size=(1, 7, 2, 3)), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=v.shape), jnp.float32)
    g = -jnp.ones((1, 7, 2)) * 0.1
    beta = jnp.asarray(rng.uniform(0.1, 0.9, size=g.shape), jnp.float32)
    tied = lambda write: jnp.sum(gated_delta_rule_reference(q, k, v, g, write, 0.5) * cotangent)
    free = lambda write, erase: jnp.sum(
        gated_delta_rule_reference(q, k, v, g, write, 0.5, erase=erase) * cotangent)
    tied_value, tied_grad = jax.value_and_grad(tied)(beta)
    free_value, (write_grad, erase_grad) = jax.value_and_grad(free, argnums=(0, 1))(beta, beta)
    np.testing.assert_allclose(free_value, tied_value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(write_grad + erase_grad, tied_grad, rtol=2e-6, atol=2e-6)
    assert np.linalg.norm(write_grad) > 0 and np.linalg.norm(erase_grad) > 0
    assert not np.allclose(write_grad, erase_grad)


@pytest.mark.parametrize("initialization", ["flax_default", "gated_deltanet_reference"])
def test_independent_erase_initialization_preserves_every_existing_parameter(initialization):
    config = gdn_model(initialization=initialization)
    tied = build_model(config, jax.random.PRNGKey(17))
    independent = build_model(replace(config, gdn_erase_mode="independent"), jax.random.PRNGKey(17))
    flatten = lambda model: {path: np.asarray(value[...])
                             for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    before, after = flatten(tied), flatten(independent)
    assert len(after) - len(before) == config.num_layers
    for path, value in before.items():
        np.testing.assert_array_equal(after[path], value, err_msg=str(path))
    for name in independent.layer_names:
        attention = getattr(independent, name).attention
        assert attention.lambda_proj.kernel is not attention.b_proj.kernel
        np.testing.assert_array_equal(attention.lambda_proj.kernel[...], attention.b_proj.kernel[...])
    tokens = jnp.asarray([[1, 2, 3, 4]])
    np.testing.assert_allclose(independent(tokens), tied(tokens), rtol=3e-5, atol=2e-6)


def test_independent_projection_gradients_and_gate_telemetry():
    module = GatedDeltaNet(32, 4, 8, 8, erase_mode="independent", rngs=nnx.Rngs(19))
    x = jax.random.normal(jax.random.PRNGKey(20), (2, 9, 32))
    cotangent = jax.random.normal(jax.random.PRNGKey(21), x.shape)
    grads = nnx.grad(lambda layer: jnp.sum(layer(x) * cotangent))(module)
    write, erase = np.asarray(grads.b_proj.kernel[...]), np.asarray(grads.lambda_proj.kernel[...])
    assert np.linalg.norm(write) > 0 and np.linalg.norm(erase) > 0
    assert not np.allclose(write, erase)
    with telemetry.collect() as collector:
        _, diagnostics = module(x, return_diagnostics=True)
    assert diagnostics["lambda_minus_beta_rms"] == 0
    assert diagnostics["invalid_erase_count"] == 0
    for name in ("beta", "lambda"):
        assert 0 <= collector.values[f"{name}_min"] <= collector.values[f"{name}_max"] <= 1
    assert collector.values["lambda_minus_beta_rms"] == 0
    assert collector.values["attention_output_rms"] > 0
    module.lambda_proj.kernel[...] = module.lambda_proj.kernel[...] + 0.1
    _, diagnostics = module(x, return_diagnostics=True)
    assert diagnostics["lambda_minus_beta_rms"] > 0


def test_odd_key_width_is_allowed_without_rope_and_auto_backend_is_blocked_on_cpu():
    config = gdn_model(qk_head_dim=3, attention_backend="auto")
    assert config.qk_norm is False
    assert resolve_attention_backend(config, sequence_length=16, platform="cpu") == "blocked"


def test_paper_initialization_draws_by_role():
    layers = 3
    config = gdn_model(num_layers=layers, hidden_size=64, intermediate_size=96, initialization="gated_deltanet_reference")
    model = build_model(config, jax.random.PRNGKey(0))
    flat = {"/".join(str(p) for p in path): np.asarray(v[...]) for path, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    for key, value in flat.items():
        if key.endswith("/kernel") and "lm_head" not in key:
            bound = 1 / math.sqrt(value.shape[0]) / (math.sqrt(2 * layers) if key.endswith("feed_forward/down_proj/kernel") else 1)
            assert np.abs(value).max() <= bound * (1 + 1e-6), key
            assert abs(value.std() * math.sqrt(3) / bound - 1) < 0.2, key
    for key in ("token_embedding/embedding", "lm_head/kernel"):
        assert abs(flat[key].std() / 0.02 - 1) < 0.05, key
    reference = build_model(gdn_model(num_layers=layers, hidden_size=64, intermediate_size=96), jax.random.PRNGKey(0))
    # Construction-time draws (short convolutions, A_log, dt_bias) are kept.
    np.testing.assert_array_equal(np.asarray(model.layers_0.attention.A_log[...]),
                                  np.asarray(reference.layers_0.attention.A_log[...]))
    np.testing.assert_array_equal(np.asarray(model.layers_1.attention.k_conv.weight[...]),
                                  np.asarray(reference.layers_1.attention.k_conv.weight[...]))


def test_identity_tags_acceptance_and_legacy_recipes():
    config = Config()
    config.model = gdn_model(initialization="gated_deltanet_reference")
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 128, 9, 2
    config.train.micro_batch_size = 2
    validate_config(config)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == identity["attention_mechanism"] == "gated_deltanet"
    tags = tracking.run_tags(config, {})
    assert "gated_deltanet" in tags and "init_gated_deltanet_reference" in tags
    values = config_to_dict(config)
    assert "scaffolds/transformer/tests/test_gated_deltanet.py" in attention_test_paths(values, "fused", True)
    assert "scaffolds/transformer/tests/test_gated_deltanet_kernel.py" in attention_test_paths(values, "fused", False)
    legacy = config_to_dict(Config())
    assert config_dicts_equal(legacy, {**legacy, "model": {k: v for k, v in legacy["model"].items()
                                                           if k not in ("gdn_conv_size", "gdn_erase_mode")}})


@pytest.mark.parametrize("erase_mode", ["tied", "independent"])
def test_training_reports_diagnostics_and_resumes_exactly(tmp_path, erase_mode):
    config = gdn_config(tmp_path / "full", initialization="gated_deltanet_reference", gdn_erase_mode=erase_mode)
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
    mask = attention_weight_decay_mask(states[0].params, excluded_names=set())
    attention_mask = mask["layers_0"]["attention"]
    for name in ("A_log", "dt_bias", "o_norm_scale"):
        assert jax.tree.leaves(attention_mask[name]) == [False], name
    assert jax.tree.leaves(attention_mask["q_proj"]["kernel"]) == [True]
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validations = [r for r in records if r.get("event") == "validation"]
    assert validations and all(r["attention_diagnostics"]["valid"] for r in validations)
    layer = validations[-1]["attention_diagnostics"]["layers"][0]
    assert layer["invalid_gate_count"] == 0 and layer["log_decay_max"] <= 0
    if erase_mode == "independent":
        assert layer["invalid_erase_count"] == 0 and layer["lambda_minus_beta_rms"] > 0
