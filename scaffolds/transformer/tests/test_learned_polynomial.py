"""Mathematical, parameter, and reporting contracts for learned unnormalized polynomial kernels."""

from dataclasses import replace
from functools import partial
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import matches_softmax_backbone_initialization
from scaffolds.transformer.attention.backends import fused_learned_polynomial as fused
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import tracking
from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state

DEGREES = (1, 2, 3, 4)
SHARINGS = ("global", "per_head")


def _model_config(degree=2, sharing="per_head", **updates):
    values = dict(vocab_size=16, max_seq_len=6, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                  qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=degree,
                  attention_normalization=False, attention_coefficients="learned",
                  attention_coefficient_sharing=sharing, attention_backend="blocked")
    return ModelConfig(**{**values, **updates})


def _config(path, degree=2, sharing="per_head"):
    config = Config()
    config.model = _model_config(degree, sharing)
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 7, 2
    config.train.micro_batch_size = 2
    config.train.workdir = path
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


def _dense_reference(q, k, v, coefficients):
    """Independent explicit power sum over the full causal matrix; callers run it in FP64."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    coefficients = jnp.broadcast_to(coefficients, (q.shape[2], coefficients.shape[-1]))
    kernel = sum(coefficients[:, order][None, :, None, None] * scores ** order
                 for order in range(coefficients.shape[-1]))
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, kernel, 0.0), v)


def _operands(time=13, heads=3, seed=11):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(0.9 * rng.normal(size=(2, time, heads, 8)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, time, heads, 5)), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=v.shape), jnp.float32)
    return (q, k, v), cotangent


def _signed_coefficients(degree, heads, sharing, seed=5):
    # Mixed signs and magnitudes so every head's kernel changes sign on the sampled scores.
    rng = np.random.default_rng(seed + degree)
    values = rng.normal(size=(heads, degree + 1)) / np.array([math.factorial(r) for r in range(degree + 1)])
    return jnp.asarray(values[0] if sharing == "global" else values, jnp.float32)


def _assert_relative(observed, expected, tolerance, label):
    """Relative L2 and max errors against the whole array, as in the fused-kernel gates.

    Signed kernels cancel: an individual output can be far smaller than the terms
    summed into it, so elementwise relative tolerances would test cancellation, not
    the implementation.
    """
    observed, expected = (np.asarray(x, dtype=np.float64) for x in (observed, expected))
    assert observed.shape == expected.shape, label
    assert np.isfinite(observed).all() and np.isfinite(expected).all(), label
    error = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert error <= tolerance, f"{label}: relative L2 error {error:.3g} > {tolerance}"
    assert np.max(np.abs(observed - expected)) <= 3 * tolerance * np.max(np.abs(expected)), label


def _fp64_vjp(function, operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(function, *(x.astype(jnp.float64) for x in operands))
        return output, pullback(cotangent.astype(jnp.float64))


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("sharing", SHARINGS)
def test_reference_outputs_and_all_gradients_match_independent_fp64(degree, sharing):
    (q, k, v), cotangent = _operands()
    coefficients = _signed_coefficients(degree, q.shape[2], sharing)

    def blocked(q, k, v, c):
        return blocked_dot_product_attention(q, k, v, attention_degree=degree, attention_normalization=False,
                                             polynomial_coefficients=c, query_block_size=4)

    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(blocked, q, k, v, coefficients)
        grads = pullback(cotangent)
    expected, expected_grads = _fp64_vjp(_dense_reference, (q, k, v, coefficients), cotangent)
    _assert_relative(output, expected, 3e-5, "output")
    for name, observed, reference in zip("qkvc", grads, expected_grads, strict=True):
        _assert_relative(observed, reference, 1e-4, f"d{name}")
    _, diagnostics = blocked_dot_product_attention(q, k, v, attention_degree=degree, attention_normalization=False,
                                                   polynomial_coefficients=coefficients, return_diagnostics=True)
    assert float(diagnostics["kernel_min"]) < 0, "expected negative weights"


@pytest.mark.parametrize("degree", DEGREES)
def test_taylor_initialized_coefficients_match_fixed_unnormalized_taylor(degree):
    (q, k, v), cotangent = _operands(seed=12)
    taylor = jnp.asarray([1 / math.factorial(r) for r in range(degree + 1)], jnp.float32)
    fixed = partial(blocked_dot_product_attention, attention_degree=degree, attention_normalization=False)
    learned = partial(fixed, polynomial_coefficients=jnp.broadcast_to(taylor, (q.shape[2], degree + 1)))
    np.testing.assert_allclose(learned(q, k, v), fixed(q, k, v), rtol=2e-6, atol=5e-7)
    expected = jax.vjp(fixed, q, k, v)[1](cotangent)
    actual = jax.vjp(learned, q, k, v)[1](cotangent)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=3e-6)


def _interpreted(monkeypatch):
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))

    # Pallas' interpreter lacks mixed slice/array indexing; this load uses array indices only.
    def interpreted_load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)

    monkeypatch.setattr(fused, "_load", interpreted_load)
    return lambda q, k, v, c: fused._attention(q, k, v, c, True)


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("time", (5, 67))
def test_interpreted_fused_kernel_outputs_and_all_gradients_match_fp64(monkeypatch, degree, time):
    (q, k, v), cotangent = _operands(time=time, seed=20 + degree)
    coefficients = _signed_coefficients(degree, q.shape[2], "per_head")
    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(_interpreted(monkeypatch), q, k, v, coefficients)
        grads = pullback(cotangent)
    expected, expected_grads = _fp64_vjp(_dense_reference, (q, k, v, coefficients), cotangent)
    assert grads[3].dtype == jnp.float32 and grads[3].shape == coefficients.shape
    _assert_relative(output, expected, 3e-5, "output")
    for name, observed, reference in zip("qkvc", grads, expected_grads, strict=True):
        _assert_relative(observed, reference, 1e-4, f"d{name}")


def test_interpreted_fused_kernel_poisons_rows_with_nonfinite_kernels(monkeypatch):
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    k = jnp.asarray([0.0, 0.0, np.inf], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0, 11.0], jnp.float32).reshape(q.shape)
    coefficients = jnp.asarray([[2.0, -1.0]], jnp.float32)
    output = _interpreted(monkeypatch)(q, k, v, coefficients)
    np.testing.assert_allclose(np.asarray(output[:, :2]).reshape(-1), [6.0, 20.0])
    assert bool(jnp.all(jnp.isnan(output[:, 2:])))


@pytest.mark.parametrize("updates,error", [
    ({"attention_normalization": True, "attention_backend": "fused"}, "no fused kernel"),
    ({"attention_degree": None}, "attention_degree >= 1"),
    ({"attention_degree": 0}, "attention_degree >= 1"),
    ({"quadratic_mode": "positive"}, "Unnormalized attention requires"),
    ({"attention_coefficients": "free"}, "Unknown attention coefficients"),
    ({"attention_coefficient_sharing": "layer"}, "Unknown attention coefficient sharing"),
])
def test_invalid_learned_coefficient_configurations_are_rejected(updates, error):
    with pytest.raises(ValueError, match=error):
        _model_config(**updates)


def test_configs_without_learned_coefficient_fields_compare_equal():
    config = Config()
    stored = config_to_dict(config)
    for name in ("attention_coefficients", "attention_coefficient_sharing"):
        stored["model"].pop(name)
    assert config_dicts_equal(stored, config_to_dict(config))


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("sharing", SHARINGS)
def test_backbone_is_the_fixed_taylor_initialization_and_production_counts(degree, sharing):
    learned_config = _model_config(degree, sharing)
    fixed = dict(nnx.to_flat_state(nnx.state(build_model(replace(learned_config, attention_coefficients="fixed"), 0),
                                             nnx.Param)))
    learned = dict(nnx.to_flat_state(nnx.state(build_model(learned_config, 0), nnx.Param)))
    assert set(learned) - set(fixed) == {("polynomial_coefficients",)}
    for key in fixed:
        np.testing.assert_array_equal(learned[key][...], fixed[key][...])
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)], np.float32)
    coefficients = np.asarray(learned[("polynomial_coefficients",)][...])
    assert coefficients.shape == ((degree + 1,) if sharing == "global" else (2, 2, degree + 1))
    np.testing.assert_array_equal(coefficients, np.broadcast_to(taylor, coefficients.shape))
    base = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml").model
    production = replace(base, attention_degree=degree, attention_normalization=False,
                         attention_coefficients="learned", attention_coefficient_sharing=sharing)
    params = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    added = (degree + 1) * (1 if sharing == "global" else production.num_layers * production.num_heads)
    added += production.num_layers * production.num_heads  # the base recipe's QK-norm log scales
    assert sum(x.size for x in jax.tree.leaves(params)) == 199_906_304 + added == model_parameter_count(production)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(params))


def test_shared_coefficient_gradient_sums_every_layer_and_head():
    tokens = jnp.arange(12).reshape(2, 6) % 16
    results = []
    for sharing in SHARINGS:
        graph, params = nnx.split(build_model(_model_config(3, sharing), 0), nnx.Param)
        value, grads = jax.value_and_grad(lambda p: jnp.square(nnx.merge(graph, p)(tokens)).sum())(params)
        results.append((value, grads["polynomial_coefficients"][...]))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-6)
    np.testing.assert_allclose(results[0][1], jnp.sum(results[1][1], axis=(0, 1)), rtol=1e-5, atol=2e-6)
    assert bool(jnp.all(results[1][1] != 0))


@pytest.mark.parametrize("sharing", SHARINGS)
def test_coefficient_array_and_backbone_exclusions_have_no_weight_decay(tmp_path, sharing):
    config = _config(tmp_path, 2, sharing)
    state = create_state(config, jax.random.PRNGKey(0))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    before, after = dict(nnx.to_flat_state(state.params)), dict(nnx.to_flat_state(updated.params))
    decay = 1 - config.optim.learning_rate * config.optim.weight_decay
    for path, variable in before.items():
        if path == ("polynomial_coefficients",) or path[-1] in ("scale", "embedding", "bias", "qk_norm_log_scale"):
            np.testing.assert_array_equal(after[path][...], variable[...])
        else:
            np.testing.assert_allclose(after[path][...], variable[...] * decay, rtol=1e-6)


def test_diagnostics_accept_signed_kernels_and_report_every_coefficient(tmp_path):
    config = _config(tmp_path, 3, "per_head")
    state = create_state(config, jax.random.PRNGKey(0))
    # Negative constants make kernels and row sums negative: valid for this mechanism.
    state.params["polynomial_coefficients"][...] = state.params["polynomial_coefficients"][...].at[..., 0].set(-4.0)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["attention_kernel"] == "learned_polynomial_unnormalized"
    assert report["kernel_values"] == "unshifted_learned_polynomial"
    assert report["attention_signed"] is True
    assert (report["attention_coefficients"], report["attention_coefficient_sharing"]) == ("learned", "per_head")
    assert report["attention_coefficient_count"] == 2 * 2 * 4
    assert any(layer["kernel_min"] < 0 for layer in report["layers"])
    coefficients = report["coefficients"]
    assert coefficients["effective_coefficient_names"] == ["constant", "linear", "quadratic", "cubic"]
    assert np.asarray(coefficients["effective"]).shape == (2, 2, 4)
    assert coefficients["effective"][1][1][0] == -4.0
    metrics = tracking.attention_metrics(report)
    prefix = "attention/learned_polynomial_unnormalized/coefficients"
    assert metrics[f"{prefix}/constant/mean"] == -4.0
    assert metrics[f"{prefix}/cubic/max"] == pytest.approx(1 / 6)


def test_softmax_backbone_initialization_check_accepts_learned_mechanism_parameters(tmp_path):
    config = _config(tmp_path, 4, "per_head")
    facts = matches_softmax_backbone_initialization(config)
    assert facts["mechanism_parameters"]["polynomial_coefficients"] == {
        "shape": [2, 2, 5], "initial": [1.0, 1.0, 0.5, 1 / 6, 1 / 24], "weight_decay": False}
    assert config.model.attention_backend == "blocked"
