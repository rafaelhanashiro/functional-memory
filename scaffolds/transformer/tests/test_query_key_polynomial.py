"""Query-key polynomial coefficients: per-query coefficients times per-key factors."""

from dataclasses import replace
from functools import partial
import json
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.input_dependent import (
    project_key_factors, query_key_polynomial_attention, query_polynomial_attention,
)
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_query_tanh_polynomial import _assert_close, _interpret
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import attention_metrics


DEGREES = (1, 2, 3, 4)
KEY_PARAMETERS = ("key_polynomial_kernel", "key_polynomial_amplitude")


def _model(degree=3, **updates):
    settings = dict(vocab_size=16, max_seq_len=13, num_layers=2, hidden_size=8, num_heads=2,
                    num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
                    attention_degree=degree, attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query_key",
                    attention_coefficient_activation="tanh", attention_backend="blocked",
                    attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=0.03125,
                    initialization="transformer_reference", qk_norm=True, attention_output_scaling="head_rms_norm")
    return ModelConfig(**{**settings, **updates})


def _config(degree=3):
    config = Config(model=_model(degree))
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 14, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 3, 0
    return config


def _inputs(degree, *, time=17, dtype=jnp.float32, qk_dim=7, value_dim=5, batch=2, heads=3):
    random = np.random.default_rng(431 + degree)
    q, k = (jnp.asarray(random.normal(size=(batch, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(random.normal(size=(batch, time, heads, value_dim)), dtype)
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
    coefficients = jnp.asarray(taylor + random.normal(size=(batch, time, heads, degree + 1)) * 0.5, jnp.float32)
    factors = jnp.asarray(1 + random.normal(size=(batch, time, heads, degree + 1)) * 0.4, jnp.float32)
    cotangent = jnp.asarray(random.normal(size=v.shape), dtype)
    return (q, k, v, coefficients, factors), cotangent


def _dense(q, k, v, coefficients, factors):
    """Independent expanded-power oracle with the pair coefficient c_r(t) f_r(i)."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    weights = sum(jnp.swapaxes(coefficients[..., order], 1, 2)[..., None]
                  * jnp.swapaxes(factors[..., order], 1, 2)[:, :, None, :] * scores ** order
                  for order in range(coefficients.shape[-1]))
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, weights, 0.0), v)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _oracle(operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _vjp(_dense, tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _compare(actual, expected, tolerance=1e-4):
    _assert_close(actual[0], expected[0], tolerance, "output")
    for index, (observed, reference) in enumerate(zip(actual[1], expected[1], strict=True)):
        _assert_close(observed, reference, tolerance, f"gradient {index}")
        if index >= 3:  # every head and order of the coefficient and factor gradients must train
            for head in range(reference.shape[2]):
                for order in range(reference.shape[-1]):
                    _assert_close(observed[..., head, order], reference[..., head, order], tolerance,
                                  f"gradient {index}, head {head}, order {order}")


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_forward_and_all_gradients_match_independent_fp64(monkeypatch, degree, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(degree)
    with jax.default_matmul_precision("highest"):
        actual = _vjp(partial(query_key_polynomial_attention, backend=backend), operands, cotangent)
    _compare(actual, _oracle(operands, cotangent))
    assert all(np.linalg.norm(gradient) > 0 for gradient in actual[1])


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_unit_key_factors_recover_query_only_attention(monkeypatch, degree, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    (q, k, v, coefficients, _), _ = _inputs(degree)
    with jax.default_matmul_precision("highest"):
        actual = query_key_polynomial_attention(q, k, v, coefficients, jnp.ones_like(coefficients), backend=backend)
        expected = query_polynomial_attention(q, k, v, coefficients, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_outputs_and_all_gradients_are_causal(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(3, time=7, batch=1, heads=2)
    prefix = 3
    cotangent = cotangent.at[:, prefix:].set(0)
    operation = partial(query_key_polynomial_attention, backend=backend)
    actual = _vjp(operation, operands, cotangent)
    changed = tuple(x.at[:, prefix:].add(2) for x in operands)
    altered = _vjp(operation, changed, cotangent)
    np.testing.assert_allclose(actual[0][:, :prefix], altered[0][:, :prefix], rtol=1e-5, atol=2e-5)
    for left, right in zip(actual[1], altered[1], strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=2e-5)
        np.testing.assert_array_equal(left[:, prefix:], 0)


def test_nonfinite_pass_poisons_exactly_the_affected_rows(monkeypatch):
    _interpret(monkeypatch)
    q = jnp.full((1, 3, 1, 1), 10., jnp.float32)
    ones = jnp.ones((1, 3, 1, 3), jnp.float32)
    for backend in ("blocked", "fused"):
        # An overflowing query coefficient touches only its own row.
        output = query_key_polynomial_attention(q, q, jnp.ones_like(q), ones.at[:, 1, :, -1].set(1e38), ones,
                                                backend=backend)
        assert bool(jnp.isfinite(output[:, [0, 2]]).all()) and bool(jnp.isnan(output[:, 1]).all()), backend
        # An overflowing key factor touches every row that can see that key.
        output = query_key_polynomial_attention(q, q, jnp.ones_like(q), ones, ones.at[:, 1, :, -1].set(1e38),
                                                backend=backend)
        assert bool(jnp.isfinite(output[:, 0]).all()) and bool(jnp.isnan(output[:, 1:]).all()), backend


@pytest.mark.parametrize("degree", DEGREES)
def test_key_factors_start_at_one_and_learn_through_the_amplitude(degree):
    random = np.random.default_rng(7 + degree)
    x = jnp.asarray(random.normal(size=(2, 7, 6)), jnp.float32)
    kernel = jnp.asarray(random.normal(size=(3, 6, degree + 1)) * 0.3, jnp.float32)
    amplitude = jnp.zeros((3, degree + 1), jnp.float32)
    cotangent = jnp.asarray(random.normal(size=(2, 7, 3, degree + 1)), jnp.float32)
    values, pullback = jax.vjp(project_key_factors, x, kernel, amplitude)
    dx, dw, da = pullback(cotangent)
    np.testing.assert_array_equal(values, 1.0)
    np.testing.assert_array_equal(dx, 0)
    np.testing.assert_array_equal(dw, 0)
    assert np.all(np.abs(np.asarray(da)) > 0)
    shifted = project_key_factors(x, kernel, jnp.full_like(amplitude, 0.5))
    np.testing.assert_allclose(shifted, 1 + 0.5 * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, kernel)), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        project_key_factors(x, kernel, amplitude[:, :1])


@pytest.mark.parametrize("degree", DEGREES)
def test_model_starts_static_counts_its_parameters_and_passes_the_initialization_check(degree):
    config = _model(degree)
    query_only = replace(config, attention_coefficient_conditioning="query")
    static_config = replace(query_only, attention_coefficient_conditioning="static",
                            attention_coefficient_activation="identity",
                            attention_coefficient_amplitude_initial=1.0, attention_coefficient_projection_initial_std=0.0)
    model, query_model, static = build_model(config, 13), build_model(query_only, 13), build_model(static_config, 13)
    assert model.key_polynomial_kernel.shape == (2, 2, 8, degree + 1)
    assert model.key_polynomial_amplitude.shape == (2, 2, degree + 1)
    np.testing.assert_array_equal(model.key_polynomial_amplitude[...], 0)
    key_kernel = np.asarray(model.key_polynomial_kernel[...])
    assert 0.5 * 0.03125 < key_kernel.std() < 1.5 * 0.03125 and len(np.unique(key_kernel)) == key_kernel.size
    # The key draw is its own stream: it neither equals nor disturbs the query draw.
    assert not np.array_equal(key_kernel, model.query_polynomial_kernel[...])
    np.testing.assert_array_equal(model.query_polynomial_kernel[...], query_model.query_polynomial_kernel[...])
    for parameter in KEY_PARAMETERS:
        assert not hasattr(query_model, parameter)
    assert model_parameter_count(config) - model_parameter_count(query_only) == 2 * 2 * 9 * (degree + 1)
    tokens = jnp.arange(26).reshape(2, 13) % 16
    np.testing.assert_allclose(model(tokens), static(tokens), rtol=2e-5, atol=2e-6)
    facts = acceptance.matches_softmax_backbone_initialization(
        Config(model=replace(config, qk_head_dim=8, value_head_dim=64, attention_backend="mosaic")))
    assert facts["mechanism_parameters"]["key_polynomial_amplitude"] == {
        "shape": [2, 2, degree + 1], "initial": [0.0] * (degree + 1), "weight_decay": False}
    assert facts["mechanism_parameters"]["key_polynomial_kernel"]["initial_std"] == 0.03125
    # A nonzero key amplitude changes the function; a nonzero query amplitude alone does not touch the key side.
    model.key_polynomial_amplitude[...] = jnp.linspace(-0.5, 0.5, degree + 1)
    assert not np.allclose(model(tokens), static(tokens), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("updates", [
    {"attention_coefficient_activation": "identity", "attention_degree": 1},
    {"attention_normalization": True}, {"attention_coefficients": "fixed"},
    {"attention_coefficient_sharing": "global"}, {"attention_coefficient_conditioning": "key"},
    {"attention_coefficient_amplitude_initial": 0.0, "attention_coefficient_projection_initial_std": 0.0},
])
def test_rejects_unsupported_query_key_configurations(updates):
    with pytest.raises(ValueError):
        _model(**updates)


def test_identity_names_the_pair_kernel_and_counts_key_parameters():
    config, query_only = _config(2), _config(2)
    query_only.model = replace(query_only.model, attention_coefficient_conditioning="query")
    identity, plain = attention_identity(config), attention_identity(query_only)
    assert identity["attention_kernel"] == "query_key_polynomial_unnormalized"
    assert identity["attention_coefficient_conditioning"] == "query_key"
    assert "c_r(x_t) f_r(x_i) s_ti^r v_i" in identity["attention_formula"]
    assert "f(x_i) = 1 + a_k*tanh(x_i W_k)" in identity["attention_formula"]
    assert identity["attention_coefficient_count"] - plain["attention_coefficient_count"] == 2 * 2 * 9 * 3
    assert plain["attention_kernel"] == "query_polynomial_unnormalized"
    paths = acceptance.test_paths(acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=()),
                                  config_to_dict(config), resolved_backend="mosaic", cpu=True)
    assert acceptance.QUERY_KEY_POLYNOMIAL_TESTS in paths and acceptance.QUERY_TANH_POLYNOMIAL_TESTS in paths
    assert len(paths) == len(set(paths))


def test_diagnostics_report_actual_key_factors_and_keep_fp32_without_weight_decay():
    config = _config(2)
    state = create_state(config, jax.random.PRNGKey(0))
    for name in KEY_PARAMETERS:
        assert state.params[name].dtype == jnp.float32
        state.params[name][...] += 0.125
    after = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    for name in KEY_PARAMETERS:
        np.testing.assert_array_equal(state.params[name][...], after.params[name][...])
    state.params["key_polynomial_kernel"][...] = jnp.asarray(
        np.random.default_rng(3).normal(size=state.params["key_polynomial_kernel"].shape) * 0.3, jnp.float32)
    state.params["key_polynomial_amplitude"][...] = 0.5
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["attention_coefficient_count"] == 2 * 2 * (8 + 2 + 9) * 3
    key = report["coefficients"]["key_factors"]
    assert np.asarray(key["effective_summary"]["mean"]).shape == (2, 2, 3)
    assert np.min(key["effective_summary"]["std"]) > 0
    np.testing.assert_allclose(key["amplitude"], np.full((2, 2, 3), 0.5), rtol=1e-6)
    model = nnx.merge(state.graphdef, state.params)
    x = model.layers_0.attention_norm(model.token_embedding(jnp.asarray(sample[:, :-1])))
    expected = project_key_factors(x, model.key_polynomial_kernel[0], model.key_polynomial_amplitude[0])
    np.testing.assert_allclose(key["effective_summary"]["mean"][0], jnp.mean(expected, axis=(0, 1)), rtol=1e-5, atol=1e-6)
    assert all(np.isfinite(value) for value in attention_metrics(report).values())


def test_tiny_training_and_checkpoint_resume_preserve_key_parameters(tmp_path):
    config = _config()
    config.train.workdir = tmp_path / "full"
    config.train.wandb_mode = "disabled"
    config.train.eval_every, config.train.eval_batches, config.train.ckpt_every = 1, 1, 1
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-3, 1e-4
    full = loop.run_training(config, audit_checkpoint=True)
    restarted = replace(config, train=replace(config.train, workdir=tmp_path / "restarted"))
    before = loop.run_training(restarted, stop_after_step=1, audit_checkpoint=True)
    after = loop.run_training(restarted, audit_checkpoint=True)
    assert before["checkpoint_audit"]["checkpoint_state_sha256"] == after["checkpoint_audit"]["start_state_sha256"]
    assert after["checkpoint_audit"]["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    manager = make_checkpoint_manager(restarted)
    try:
        initial = create_state(restarted, jax.random.PRNGKey(0))
        state, step = restore_latest_checkpoint(manager, initial)
        assert step == config.optim.total_steps
        for name in KEY_PARAMETERS:
            assert not np.array_equal(state.params[name][...], initial.params[name][...]), name
    finally:
        manager.close()
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert records and all(row["attention_diagnostics"]["valid"] for row in records)
    assert all("key_factors" in row["attention_diagnostics"]["coefficients"] for row in records)


def _requires_gpu(backend):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    if backend == "mosaic" and jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Mosaic kernels require an H100 or B300")


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ("fused", "mosaic"))
def test_gpu_per_degree_passes_match_the_blocked_reference_in_production_context(degree, backend):
    _requires_gpu(backend)
    operands, cotangent = _inputs(degree, time=2048, dtype=jnp.bfloat16, qk_dim=64, value_dim=64, batch=1, heads=4)
    actual = jax.jit(partial(_vjp, partial(query_key_polynomial_attention, backend=backend)))(operands, cotangent)
    with jax.default_matmul_precision("highest"):
        reference = jax.jit(partial(_vjp, query_key_polynomial_attention))(operands, cotangent)
    _compare(actual, reference, tolerance=0.025)
    assert actual[0].dtype == jnp.bfloat16
    assert all(gradient.dtype == jnp.float32 for gradient in actual[1][3:])


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("fused", "mosaic"))
def test_gpu_overflowing_pass_poisons_exactly_the_affected_rows(backend):
    _requires_gpu(backend)
    q = jnp.full((1, 256, 1, 8), 10., jnp.bfloat16)
    ones = jnp.ones((1, 256, 1, 3), jnp.float32)
    v = jnp.ones((1, 256, 1, 64), jnp.bfloat16)
    attend = jax.jit(partial(query_key_polynomial_attention, backend=backend))
    output = attend(q, q, v, ones.at[:, 1, :, -1].set(1e38), ones)
    assert bool(jnp.isfinite(output[:, 0]).all()) and bool(jnp.isnan(output[:, 1]).all())
    assert bool(jnp.isfinite(output[:, 2:]).all())
    output = attend(q, q, v, ones, ones.at[:, 1, :, -1].set(1e38))
    assert bool(jnp.isfinite(output[:, 0]).all()) and bool(jnp.isnan(output[:, 1:]).all())
