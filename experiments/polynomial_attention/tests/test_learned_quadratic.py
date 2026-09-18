"""Scientific checks for learned shapes, sharing, and optimizer/checkpoint state."""
import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from experiments.polynomial_attention.configs import Config, ModelConfig, load_config
from experiments.polynomial_attention.models import build_model, model_parameter_count
from experiments.polynomial_attention.models.components.attention import blocked_dot_product_attention
from experiments.polynomial_attention.training.checkpointing import (
    make_checkpoint_manager, restore_latest_checkpoint, save_checkpoint,
)
from experiments.polynomial_attention.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
from experiments.polynomial_attention.training.loop import train_step
from experiments.polynomial_attention.training.state import create_state


VARIANTS = [(mode, sharing) for mode in ("positive", "unrestricted") for sharing in ("global", "per_head")]


def _config(path, mode="positive", sharing="global", remat=False):
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=16, max_seq_len=6, num_layers=2, hidden_size=8,
                           num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
                           intermediate_size=12, attention_degree=2, quadratic_mode=mode,
                           quadratic_sharing=sharing, remat=remat)
    cfg.data.synthetic_vocab_size, cfg.data.sequence_length, cfg.data.batch_size = 16, 7, 2
    cfg.train.micro_batch_size = 2
    cfg.train.workdir = path
    cfg.optim.total_steps, cfg.optim.warmup_steps = 4, 0
    return cfg


def _dense_reference(q, k, v, coefficients, mode):
    """Independent full-matrix FP64 computation; no production helpers."""
    score = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    coeff = jnp.broadcast_to(coefficients, (q.shape[2], 2))
    first, second = coeff[:, 0][None, :, None, None], coeff[:, 1][None, :, None, None]
    kernel = (0.5 + 0.5 * (first * score + second) ** 2 if mode == "positive" else
              1.0 + first * score + second * score ** 2)
    mask = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    kernel = jnp.where(mask, kernel, 0.0)
    weights = kernel / jnp.sum(kernel, axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


def _operands(time=7):
    keys = jax.random.split(jax.random.key(21), 3)
    return (0.5 * jax.random.normal(keys[0], (2, time, 2, 4)),
            0.5 * jax.random.normal(keys[1], (2, time, 2, 4)),
            jax.random.normal(keys[2], (2, time, 2, 3)))


@pytest.mark.parametrize("mode,sharing", VARIANTS)
def test_outputs_and_all_gradients_match_independent_fp64(mode, sharing):
    operands = _operands()
    coeff = jnp.asarray([0.8, -0.3] if mode == "positive" else [0.7, 0.3], dtype=jnp.float32)
    if sharing == "per_head":
        coeff = jnp.stack((coeff, coeff * 1.2))
    cotangent = jax.random.normal(jax.random.key(22), operands[2].shape)
    def implementation(q, k, v, c):
        return blocked_dot_product_attention(q, k, v, attention_degree=2, coefficients=c,
                                             quadratic_mode=mode, query_block_size=3)
    actual = implementation(*operands, coeff)
    actual_grads = jax.grad(lambda *args: jnp.sum(implementation(*args) * cotangent), argnums=(0, 1, 2, 3))(*operands, coeff)
    with jax.enable_x64():
        args64 = tuple(x.astype(jnp.float64) for x in (*operands, coeff))
        expected = _dense_reference(*args64, mode)
        expected_grads = jax.grad(
            lambda *args: jnp.sum(_dense_reference(*args, mode) * cotangent.astype(jnp.float64)),
            argnums=(0, 1, 2, 3),
        )(*args64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=4e-5, atol=4e-6)
    changed = tuple(x.at[:, 5:].multiply(-3) for x in operands)
    np.testing.assert_array_equal(implementation(*changed, coeff)[:, :5], actual[:, :5])


@pytest.mark.parametrize("mode", ("positive", "unrestricted"))
def test_taylor_initialization_output_and_qkv_gradient_parity(mode):
    args = _operands()
    coeff = jnp.array([1., 1. if mode == "positive" else .5])
    fixed = lambda *args: blocked_dot_product_attention(*args, attention_degree=2, query_block_size=3)
    learned = lambda *args: blocked_dot_product_attention(*args, attention_degree=2, query_block_size=3,
                                                         coefficients=coeff, quadratic_mode=mode)
    np.testing.assert_allclose(learned(*args), fixed(*args), atol=5e-7, rtol=2e-6)
    expected = jax.grad(lambda *args: jnp.square(fixed(*args)).sum(), argnums=(0, 1, 2))(*args)
    actual = jax.grad(lambda *args: jnp.square(learned(*args)).sum(), argnums=(0, 1, 2))(*args)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("mode,sharing", VARIANTS)
def test_backbone_is_bitwise_identical_and_production_added_counts(tmp_path, mode, sharing):
    cfg = _config(tmp_path, mode, sharing).model
    fixed = dict(nnx.to_flat_state(nnx.state(build_model(replace(cfg, quadratic_mode="fixed"), 0), nnx.Param)))
    learned = dict(nnx.to_flat_state(nnx.state(build_model(cfg, 0), nnx.Param)))
    assert set(learned) - set(fixed) == {("quadratic_coefficients",)}
    for key in fixed:
        np.testing.assert_array_equal(learned[key][...], fixed[key][...])
    base = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml").model
    production = replace(base, attention_degree=2, quadratic_mode=mode, quadratic_sharing=sharing)
    params = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    assert sum(x.size for x in jax.tree.leaves(params)) == 199_906_304 + (2 if sharing == "global" else 320)
    assert model_parameter_count(production) == sum(x.size for x in jax.tree.leaves(params))
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(params))


@pytest.mark.parametrize("mode", ("positive", "unrestricted"))
def test_global_gradients_sum_over_all_heads_and_layers_with_remat(tmp_path, mode):
    cfg = _config(tmp_path, mode).model
    tokens = jnp.arange(12).reshape(2, 6) % 16
    results = []
    for sharing, remat in (("global", False), ("per_head", False), ("global", True)):
        graph, params = nnx.split(build_model(replace(cfg, quadratic_sharing=sharing, remat=remat), 0), nnx.Param)
        value, grads = jax.value_and_grad(lambda p: jnp.square(nnx.merge(graph, p)(tokens)).sum())(params)
        results.append((value, grads["quadratic_coefficients"][...]))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-6)
    np.testing.assert_allclose(results[0][1], jnp.sum(results[1][1], axis=(0, 1)), atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(results[0][1], results[2][1], atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("mode,sharing", VARIANTS)
def test_only_coefficient_array_is_excluded_from_weight_decay(tmp_path, mode, sharing):
    cfg = _config(tmp_path, mode, sharing)
    state = create_state(cfg, jax.random.PRNGKey(0))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    before, after = dict(nnx.to_flat_state(state.params)), dict(nnx.to_flat_state(updated.params))
    for path, variable in before.items():
        if path == ("quadratic_coefficients",):
            np.testing.assert_array_equal(variable[...], after[path][...])
        else:
            np.testing.assert_allclose(after[path][...], variable[...] * (1-cfg.optim.learning_rate*cfg.optim.weight_decay),
                                       atol=1e-7, rtol=2e-7)


@pytest.mark.parametrize("mode,sharing", VARIANTS)
def test_checkpoint_resume_preserves_coefficient_and_optimizer_updates(tmp_path, mode, sharing):
    cfg = _config(tmp_path, mode, sharing)
    cfg.train.dtype = "bfloat16"
    rng = jax.random.PRNGKey(0)
    state = create_state(cfg, rng)
    batch = jnp.arange(14).reshape(2, 7) % 16
    state, _ = train_step(state, batch, rng)
    manager = make_checkpoint_manager(cfg)
    try:
        save_checkpoint(manager, 1, state, metrics={"validation_nll": 3.0})
        manager.wait_until_finished()
        restored, step = restore_latest_checkpoint(manager, create_state(cfg, rng))
        assert step == 1
    finally:
        manager.close()
    expected, _ = train_step(state, batch, rng)
    actual, _ = train_step(restored, batch, rng)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)
    initial = [1., 1. if mode == "positive" else .5]
    assert not np.array_equal(np.asarray(actual.params["quadratic_coefficients"][...]).reshape(-1, 2)[0], initial)


def _signed(scores, values=None):
    q = jnp.ones((1, len(scores), 1, 1))
    k = jnp.asarray(scores, dtype=jnp.float32).reshape(q.shape)
    v = jnp.asarray(values if values is not None else np.arange(len(scores)), dtype=jnp.float32).reshape(q.shape)
    return blocked_dot_product_attention(q, k, v, attention_degree=2, coefficients=jnp.array([1., 0.]),
                                         quadratic_mode="unrestricted", query_block_size=2, return_diagnostics=True)


def test_signed_denominators_and_normalized_negative_weights():
    output, diagnostic = _signed([-3., 0.], [2., 7.])
    np.testing.assert_array_equal(output.reshape(-1), [2., -3.])
    assert int(diagnostic["invalid_normalizer_count"]) == 0
    assert float(diagnostic["negative_weight_fraction"]) == pytest.approx(1/3)
    assert float(diagnostic["negative_kernel_fraction"]) == pytest.approx(2/3)
    assert float(diagnostic["min_abs_normalizer"]) == 1
    assert float(diagnostic["cancellation_ratio_max"]) == 3


def test_zero_intermediate_sum_and_zero_kernel_allow_valid_final_row():
    output, diagnostic = _signed([0., -2., 0.], [2., 7., 3.])
    assert np.isnan(output[0, 1, 0, 0])  # This earlier query's final sum is zero.
    assert output[0, 2, 0, 0] == -2  # Its first two keys cancel, but the final sum is valid.
    assert int(diagnostic["invalid_normalizer_count"]) == 1
    output, diagnostic = _signed([-1., 0.], [2., 7.])
    assert output[0, 1, 0, 0] == 7
    assert int(diagnostic["invalid_kernel_count"]) == 0


def test_strong_cancellation_is_unclipped_and_reported():
    output, diagnostic = _signed([0., -2.+2.**-12], [1., 0.])
    assert float(output[0, 1, 0, 0]) == 4096.
    assert float(diagnostic["cancellation_ratio_max"]) == 8191.
    assert float(diagnostic["min_abs_normalizer"]) == 2.**-12


@pytest.mark.parametrize("mode,sharing", VARIANTS)
def test_diagnostic_report_records_coefficients_and_json_failures(tmp_path, mode, sharing):
    cfg = _config(tmp_path, mode, sharing)
    state = create_state(cfg, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(cfg)
    report = attention_diagnostics(state, cfg, sample, provenance)
    assert report["valid"]
    assert report["coefficients"]["finite"]
    np.testing.assert_array_equal(np.asarray(report["coefficients"]["effective"]).reshape(-1, 3)[0], [1., 1., .5])
    for layer in report["layers"]:
        assert layer["negative_weight_fraction"] == 0
        assert layer["cancellation_ratio_max"] == 1
        assert layer["min_abs_normalizer"] > 0
    state.params["quadratic_coefficients"][...] = jnp.full_like(state.params["quadratic_coefficients"][...], jnp.nan)
    failed = attention_diagnostics(state, cfg, sample, provenance)
    assert not failed["valid"] and not failed["coefficients"]["finite"]
    json.dumps(failed, allow_nan=False)


@pytest.mark.parametrize("settings", ({"attention_degree": None, "quadratic_mode": "positive"},
                                      {"attention_degree": 4, "quadratic_mode": "unrestricted"},
                                      {"quadratic_mode": "bad"}, {"quadratic_sharing": "bad"}))
def test_invalid_quadratic_configs_are_rejected(settings):
    with pytest.raises(ValueError):
        ModelConfig(**settings)
