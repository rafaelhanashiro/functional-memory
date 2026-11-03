"""Softmax and unnormalized exponential attention with the Gated DeltaNet forget gate (FoX).

The decay joins every score, ``softmax(s + D)`` or ``exp(s + D)`` with ``D_ti = L_t - L_i``, on
the blocked reference backend; PaTH-FoX is qualified with the PaTH tests.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.parameters import attention_parameter_count
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics, attention_identity, fixed_diagnostic_sample,
)
from scaffolds.transformer.training.state import create_state

KERNELS = ("softmax", "exp")
GATE = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")


def _config(kernel="softmax", **changes):
    """Tiny FoX recipes without RoPE: canonical QK RMSNorm, and per-head output RMSNorm for exp."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=4, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
                 attention_normalization=kernel == "softmax", attention_forget_gate=True,
                 positional_encoding="none",
                 attention_output_scaling="none" if kernel == "softmax" else "head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


def _naive(normalized):
    def attend(q, k, v, log_decay):
        time = q.shape[1]
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
        gap = jnp.swapaxes(log_decay, 1, 2)[..., :, None] - jnp.swapaxes(log_decay, 1, 2)[..., None, :]
        causal = jnp.tril(jnp.ones((time, time), bool))
        if normalized:
            weights = jax.nn.softmax(jnp.where(causal, scores + gap, -jnp.inf), axis=-1)
        else:
            weights = jnp.where(causal, jnp.exp(jnp.where(causal, scores + gap, 0.0)), 0.0)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return attend


def _float64_gradients(reference, arrays, cotangent):
    with jax.enable_x64():
        inputs = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        value = np.asarray(reference(*inputs))
        gradients = jax.grad(lambda *x: jnp.sum(reference(*x) * jnp.asarray(cotangent)),
                             argnums=tuple(range(len(arrays))))(*inputs)
    return value, [np.asarray(g) for g in gradients]


@pytest.mark.parametrize("kernel", KERNELS)
def test_blocked_decay_and_all_derivatives_match_float64(kernel):
    rng = np.random.default_rng(11)
    b, t, h, d = 2, 37, 3, 4
    arrays = (rng.normal(size=(b, t, h, d)), rng.normal(size=(b, t, h, d)), rng.normal(size=(b, t, h, d)),
              np.cumsum(-np.abs(rng.normal(size=(b, t, h))) * .3, axis=1))
    cotangent = rng.normal(size=(b, t, h, d))
    normalized = kernel == "softmax"
    wanted, gradients64 = _float64_gradients(_naive(normalized), arrays, cotangent)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    actual = lambda q, k, v, L: blocked_dot_product_attention(
        q, k, v, attention_normalization=normalized, query_log_decay=L, query_block_size=16)
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(jax.jit(actual)(*operands), wanted, rtol=2e-5, atol=2e-5)
        gradients = jax.grad(lambda *x: jnp.sum(actual(*x) * jnp.asarray(cotangent, jnp.float32)),
                             argnums=tuple(range(4)))(*operands)
    for name, value, expected in zip(("q", "k", "v", "L"), gradients, gradients64):
        scale = np.linalg.norm(expected)
        assert scale > 0, name
        assert np.linalg.norm(np.asarray(value) - expected) / scale < 2e-5, name
    # Without decay the same call is the ordinary kernel.
    with jax.default_matmul_precision("highest"):
        plain = blocked_dot_product_attention(*operands[:3], attention_normalization=normalized, query_block_size=16)
        zero = actual(*operands[:3], jnp.zeros_like(operands[3]))
    np.testing.assert_allclose(zero, plain, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kernel", KERNELS)
def test_configurations(kernel):
    for encoding in ("none", "rope"):
        assert _config(kernel, positional_encoding=encoding).model.attention_forget_gate
    path = _config(kernel, positional_encoding="path", qk_head_dim=16, value_head_dim=16)
    assert path.model.path_kernel() == kernel
    assert replace(path.model, attention_backend="fused").path_fused_supported()
    for backend in ("fused", "cudnn", "mosaic", "xla"):
        with pytest.raises(ValueError):
            _config(kernel, attention_backend=backend)
    with pytest.raises(ValueError, match="forget gate"):
        _config(kernel, attention_degree=2)  # a fixed Taylor kernel takes no forget gate


@pytest.mark.parametrize("kernel", KERNELS)
def test_identity_parameters_and_diagnostics(kernel):
    config = _config(kernel)
    identity = attention_identity(config)
    assert identity["attention_forget_gate"] is True
    assert identity["attention_kernel"] == ("softmax" if kernel == "softmax" else "exp_unnormalized")
    assert identity["attention_formula"].startswith(
        ("causal_softmax" if kernel == "softmax" else "causal_exp") + "(QK^T / sqrt(d_k) + D) V, D_ti = ")
    assert identity["attention_formula"].endswith("with a learned gain") is (kernel == "exp")
    assert identity["positional_encoding"] == "none"
    assert identity["forget_gate_parameter_count"] == (8 + 2) * 2
    assert attention_parameter_count(config.model) - attention_parameter_count(
        replace(config.model, attention_forget_gate=False)) == (8 + 2) * 2
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert np.all(np.asarray(report["layers"][0]["forget_gate_log_alpha_max"]) <= 0)
    gate = report["coefficients"]["forget_gate"]
    assert gate["finite"] and np.asarray(gate["rate_exp_A_log"]).shape == (1, 2)


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_gate_that_never_decays_changes_nothing(kernel):
    tokens = jnp.arange(16).reshape(2, 8) % 16
    gated = build_model(_config(kernel).model, 19)
    plain = build_model(_config(kernel, attention_forget_gate=False).model, 19)
    assert np.abs(np.asarray(gated(tokens) - plain(tokens))).max() > 1e-4
    gated.forget_gate_dt_bias[...] = jnp.full_like(gated.forget_gate_dt_bias[...], -200.0)
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
        for cpu in (True, False):
            assert acceptance.FORGET_GATE_ATTENTION_TESTS in acceptance.test_paths(
                spec, recipe, resolved_backend="blocked", cpu=cpu)
        recipe = config_to_dict(_config(kernel, attention_forget_gate=False))
        assert acceptance.FORGET_GATE_ATTENTION_TESTS not in acceptance.test_paths(
            spec, recipe, resolved_backend="blocked", cpu=True)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", KERNELS)
def test_gpu_production_shape_matches_fp32_highest(kernel):
    """BF16 Q/K/V at the production width and context, under Gated DeltaNet's initial decay range."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    rng = np.random.default_rng(17)
    b, t, h, d = 1, 2048, 4, 64
    q, k = (rng.normal(size=(b, t, h, d)) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) * 4 for x in (q, k))
    v, cotangent = rng.normal(size=(b, t, h, d)), rng.normal(size=(b, t, h, d))
    log_alpha = -rng.uniform(0, 16, size=h) * np.exp(rng.uniform(np.log(1e-3), np.log(0.1), size=(b, t, h)))
    log_decay = jnp.asarray(np.cumsum(log_alpha, axis=1), jnp.float32)
    normalized = kernel == "softmax"
    attend = lambda q, k, v, L: blocked_dot_product_attention(q, k, v, attention_normalization=normalized,
                                                              query_log_decay=L)

    def value_and_gradients(*operands):
        output, pull = jax.vjp(attend, *operands)
        return (output, *pull(jnp.asarray(cotangent, output.dtype)))

    bf16 = [jnp.asarray(x, jnp.bfloat16) for x in (q, k, v)]
    actual = jax.jit(value_and_gradients)(*bf16, log_decay)
    with jax.default_matmul_precision("highest"):
        expected = jax.jit(value_and_gradients)(*(x.astype(jnp.float32) for x in bf16), log_decay)
    for name, got, want in zip(("output", "dq", "dk", "dv", "dL"), actual, expected):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        assert np.isfinite(got).all(), name
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2, name
