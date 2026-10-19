"""Signed GDN writes: independent mathematics, unchanged draws, and exact resumption."""

import csv
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import enable_x64

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


def signed_state_oracle(args, cotangent, scale):
    """NumPy FP64 full transition matrices and a separately derived manual adjoint.

    Production uses chunked WY kernels and custom JAX derivatives. This oracle
    materializes P=I-lambda*k*k^T and retains token states; its reverse pass
    differentiates that matrix update directly, without production imports or AD.
    """
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    cotangent = np.asarray(cotangent, np.float64)
    batch, length, heads, key_dim = q.shape
    states = np.zeros((length + 1, batch, heads, key_dim, v.shape[-1]))
    output = np.empty_like(v)
    identity = np.eye(key_dim)
    for t in range(length):
        kt = k[:, t]
        p = identity - erase[:, t, :, None, None] * kt[..., :, None] * kt[..., None, :]
        states[t + 1] = np.exp(g[:, t])[..., None, None] * (p @ states[t])
        states[t + 1] += beta[:, t, :, None, None] * kt[..., :, None] * v[:, t, :, None, :]
        output[:, t] = scale * np.einsum("bhk,bhkv->bhv", q[:, t], states[t + 1])
    dq, dk, dv, dg, dbeta, derase = (np.zeros_like(a) for a in (q, k, v, g, beta, erase))
    adjoint = np.zeros_like(states[0])
    for t in range(length - 1, -1, -1):
        qt, kt, vt, previous, ct = q[:, t], k[:, t], v[:, t], states[t], cotangent[:, t]
        alpha, et, bt = np.exp(g[:, t]), erase[:, t], beta[:, t]
        dq[:, t] = scale * np.einsum("bhkv,bhv->bhk", states[t + 1], ct)
        adjoint += scale * qt[..., :, None] * ct[..., None, :]
        prediction = np.einsum("bhk,bhkv->bhv", kt, previous)
        key_adjoint = np.einsum("bhk,bhkv->bhv", kt, adjoint)
        value_adjoint = np.einsum("bhkv,bhv->bhk", adjoint, vt)
        dbeta[:, t] = np.sum(key_adjoint * vt, axis=-1)
        dv[:, t] = bt[..., None] * key_adjoint
        derase[:, t] = -alpha * np.sum(key_adjoint * prediction, axis=-1)
        transported = previous - et[..., None, None] * kt[..., :, None] * prediction[..., None, :]
        dg[:, t] = alpha * np.sum(adjoint * transported, axis=(-2, -1))
        dk[:, t] = bt[..., None] * value_adjoint - (alpha * et)[..., None] * (
            np.einsum("bhkv,bhv->bhk", adjoint, prediction)
            + np.einsum("bhkv,bhv->bhk", previous, key_adjoint))
        adjoint = alpha[..., None, None] * (
            adjoint - et[..., None, None] * kt[..., :, None] * key_adjoint[..., None, :])
    return output, (dq, dk, dv, dg, dbeta, derase)


def signed_operands(dtype=jnp.float32, *, length=192, key_dim=64, value_dim=64, heads=2,
                    strong_decay=False, erase_value=None, seed=104):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(1, length, heads, key_dim)) for _ in range(2))
    q, k = (a / np.linalg.norm(a, axis=-1, keepdims=True) for a in (q, k))
    v = rng.normal(size=(1, length, heads, value_dim))
    g = -rng.uniform(1., 3., size=(1, length, heads)) if strong_decay else -rng.uniform(.01, .2, size=(1, length, heads))
    logits = rng.normal(size=g.shape)
    # Zero and sign crossings straddle chunk boundaries, not just the first token.
    for start in (0, 63, 127):
        stop = min(start + 3, length)
        if start < length:
            logits[:, start:stop] = np.asarray([0., -1e-6, 1e-6])[:stop - start, None]
    beta = np.tanh(logits)
    erase = rng.uniform(.1, .9, size=g.shape) if erase_value is None else np.full_like(g, erase_value)
    cast = lambda a: jnp.asarray(a, dtype)
    return (cast(q), cast(k), cast(v), jnp.asarray(g, jnp.float32), cast(beta), cast(erase)), cast(rng.normal(size=v.shape))


@pytest.mark.parametrize("erase_value", [None, 0.])
def test_manual_adjoint_matches_fp64_matrix_state_autodiff(erase_value):
    args, cotangent = signed_operands(length=9, key_dim=3, value_dim=2, heads=1, erase_value=erase_value)

    def matrix_recurrence(q, k, v, g, beta, erase):
        def step(state, inputs):
            qt, kt, vt, gt, bt, et = inputs
            p = jnp.eye(q.shape[-1]) - et[..., None, None] * kt[..., :, None] * kt[..., None, :]
            state = jnp.exp(gt)[..., None, None] * (p @ state)
            state += bt[..., None, None] * kt[..., :, None] * vt[..., None, :]
            return state, .5 * jnp.einsum("bhk,bhkv->bhv", qt, state)
        _, output = jax.lax.scan(step, jnp.zeros((1, 1, 3, 2), q.dtype),
                                 tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, v, g, beta, erase)))
        return jnp.moveaxis(output, 0, 1)

    expected = signed_state_oracle(args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        args = tuple(a.astype(jnp.float64) for a in args)
        actual = matrix_recurrence(*args)
        gradients = jax.grad(lambda *a: jnp.sum(matrix_recurrence(*a) * cotangent), argnums=range(6))(*args)
    for actual, reference in zip((actual, *gradients), (expected[0], *expected[1]), strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-12)


def test_legacy_recipes_default_to_sigmoid_and_cannot_resume_as_tanh():
    config = Config()
    config.model = gdn_model(gdn_erase_mode="independent")
    sigmoid = config_to_dict(config)
    legacy = {**sigmoid, "model": {k: v for k, v in sigmoid["model"].items() if k != "gdn_write_activation"}}
    tanh = {**sigmoid, "model": {**sigmoid["model"], "gdn_write_activation": "tanh"}}
    assert config.model.gdn_write_activation == "sigmoid"
    assert config_dicts_equal(sigmoid, legacy)
    assert not config_dicts_equal(tanh, legacy)
    assert not config_dicts_equal(tanh, sigmoid)


@pytest.mark.parametrize("updates", [
    {"gdn_write_activation": "relu", "gdn_erase_mode": "independent"},
    {"gdn_write_activation": "tanh"},
    {"gdn_write_activation": "tanh", "attention_mechanism": "dot_product"},
    {"gdn_write_activation": "tanh", "attention_mechanism": "gla"},
])
def test_configuration_rejects_ambiguous_signed_writes(updates):
    with pytest.raises(ValueError, match="gdn_write_activation"):
        gdn_model(**updates)


@pytest.mark.parametrize("initialization", ["flax_default", "gated_deltanet_reference"])
def test_activation_change_preserves_all_weights_and_parameter_count(initialization):
    config = gdn_model(initialization=initialization, gdn_erase_mode="independent")
    sigmoid = build_model(config, jax.random.PRNGKey(17))
    signed_config = replace(config, gdn_write_activation="tanh")
    signed = build_model(signed_config, jax.random.PRNGKey(17))
    before, after = nnx.to_flat_state(nnx.state(sigmoid, nnx.Param)), nnx.to_flat_state(nnx.state(signed, nnx.Param))
    for (path, left), (other_path, right) in zip(before, after, strict=True):
        assert path == other_path
        np.testing.assert_array_equal(left[...], right[...], err_msg=str(path))
    assert model_parameter_count(config) == model_parameter_count(signed_config)
    for name in signed.layer_names:
        attention = getattr(signed, name).attention
        assert attention.write_activation == "tanh"
        assert np.linalg.norm(np.asarray(attention.b_proj.kernel[...])) > 0
        np.testing.assert_array_equal(attention.b_proj.kernel[...], attention.lambda_proj.kernel[...])


def _signed_layer_oracle(module, x):
    """Independent FP64 layer computation; both gate nonlinearities are explicit."""
    p = {"/".join(str(part) for part in path): np.asarray(value[...], np.float64)
         for path, value in nnx.to_flat_state(nnx.state(module, nnx.Param))}
    x = np.asarray(x, np.float64)
    batch, length, _ = x.shape
    heads, dk, dv = module.num_heads, module.key_head_dim, module.value_head_dim
    silu = lambda z: z / (1 + np.exp(-z))

    def conv(z, w):
        padded = np.concatenate((np.zeros((batch, len(w) - 1, z.shape[-1])), z), axis=1)
        return silu(sum(padded[:, i:i + length] * w[i] for i in range(len(w))))

    q = conv(x @ p["q_proj/kernel"], p["q_conv/weight"]).reshape(batch, length, heads, dk)
    k = conv(x @ p["k_proj/kernel"], p["k_conv/weight"]).reshape(batch, length, heads, dk)
    v = conv(x @ p["v_proj/kernel"], p["v_conv/weight"]).reshape(batch, length, heads, dv)
    q, k = (a / np.sqrt(np.sum(a * a, axis=-1, keepdims=True) + 1e-6) for a in (q, k))
    beta = np.tanh(x @ p["b_proj/kernel"])
    erase = 1 / (1 + np.exp(-(x @ p["lambda_proj/kernel"])))
    g = -np.exp(p["A_log"]) * np.logaddexp(0., x @ p["a_proj/kernel"] + p["dt_bias"])
    read, _ = signed_state_oracle((q, k, v, g, beta, erase), np.zeros_like(v), dk ** -.5)
    read /= np.sqrt(np.mean(read * read, axis=-1, keepdims=True) + module.norm_eps)
    gate = (x @ p["g_proj/kernel"]).reshape(batch, length, heads, dv)
    output = read * p["o_norm_scale"] * silu(gate)
    return output.reshape(batch, length, heads * dv) @ p["o_proj/kernel"], beta, erase


def test_tanh_layer_matches_independent_math_and_has_signed_finite_telemetry():
    module = GatedDeltaNet(32, 4, 8, 8, erase_mode="independent", write_activation="tanh", rngs=nnx.Rngs(19))
    x = jax.random.normal(jax.random.PRNGKey(20), (2, 9, 32))
    expected, beta, erase = _signed_layer_oracle(module, x)
    assert beta.min() < 0 < beta.max()
    with jax.default_matmul_precision("highest"), telemetry.collect() as collector:
        actual, diagnostics = module(x, return_diagnostics=True)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    assert diagnostics["beta_min"] < 0 < diagnostics["beta_max"]
    assert diagnostics["invalid_write_count"] == diagnostics["invalid_erase_count"] == 0
    assert all(np.all(np.isfinite(value)) for value in diagnostics.values())
    for gate_name, expected_gate in (("beta", beta), ("lambda", erase)):
        for statistic, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
            np.testing.assert_allclose(collector.values[f"{gate_name}_{statistic}"], function(expected_gate), rtol=2e-6, atol=2e-7)
    assert collector.values["lambda_minus_beta_rms"] > 0
    cotangent = jax.random.normal(jax.random.PRNGKey(21), x.shape)
    gradients, input_gradient = nnx.grad(lambda layer, inputs: jnp.sum(layer(inputs) * cotangent), argnums=(0, 1))(module, x)
    for value in (gradients.b_proj.kernel[...], gradients.lambda_proj.kernel[...], input_gradient):
        value = np.asarray(value)
        assert np.all(np.isfinite(value)) and np.linalg.norm(value) > 0
    assert not np.allclose(gradients.b_proj.kernel[...], gradients.lambda_proj.kernel[...])


def test_identity_and_run_index_distinguish_signed_writes(tmp_path):
    config = Config()
    config.model = gdn_model(initialization="gated_deltanet_reference", gdn_erase_mode="independent",
                             gdn_write_activation="tanh")
    identity = attention_identity(config)
    assert identity["gdn_linear_write_activation"] == "tanh"
    assert identity["gdn_write_initialization"] == "tanh_write_projection"
    assert "tanh_write" in tracking.run_tags(config, {})
    study = tmp_path / "study"
    create_run(study, config_to_dict(config))
    with regenerate_index(study).open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["gdn_write_activation"] == "tanh"


def test_tanh_training_checkpoint_resume_is_exact_and_reports_valid_signed_gates(tmp_path):
    config = gdn_config(tmp_path / "full", initialization="gated_deltanet_reference",
                        gdn_erase_mode="independent", gdn_write_activation="tanh")
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
    rows = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validation = [row for row in rows if row.get("event") == "validation"]
    assert validation and all(row["attention_diagnostics"]["valid"] for row in validation)
    layer = validation[-1]["attention_diagnostics"]["layers"][0]
    assert layer["invalid_write_count"] == 0
    assert layer["beta_min"] < 0 < layer["beta_max"]
