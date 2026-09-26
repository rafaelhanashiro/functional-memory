"""Row-normalized attention with learned polynomial coefficients, alone and under QK-norm.

The blocked reference is the only implementation. Tests compare it with an
independent FP64 dense evaluation of p(s) V / rowsum for degrees 1-4, check parity
with the fixed Taylor kernels at the Taylor initialization, the signed-row-sum
diagnostics, configuration and backend selection, parameter ownership, and training
with resume. GPU cases exercise the production BF16 context on cosine operands.
"""

from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import matches_softmax_backbone_initialization
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.attention.taylor import taylor_attention
from scaffolds.transformer.configs import Config, ModelConfig, load_config, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state

DEGREES = (1, 2, 3, 4)
SHARINGS = ("global", "per_head")
EPS = 1e-6
BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def _model_config(degree=2, sharing="per_head", qk_norm=False, **updates):
    values = dict(vocab_size=16, max_seq_len=6, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                  qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=degree,
                  attention_normalization=True, attention_coefficients="learned",
                  attention_coefficient_sharing=sharing, attention_backend="blocked", qk_norm=qk_norm)
    return ModelConfig(**{**values, **updates})


def _config(path, degree=2, sharing="per_head", qk_norm=False):
    config = Config()
    config.model = _model_config(degree, sharing, qk_norm)
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 7, 2
    config.train.micro_batch_size = 2
    config.train.workdir = path
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _operands(time=13, heads=3, qk_dim=8, value_dim=5, seed=11, scale=0.45, dtype=jnp.float32):
    # Scores of order one keep every row sum well away from zero for near-Taylor coefficients.
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(scale * rng.normal(size=(2, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, time, heads, value_dim)), dtype)
    cotangent = jnp.asarray(rng.normal(size=v.shape), dtype)
    return (q, k, v), cotangent


def _near_taylor_coefficients(degree, heads, sharing, seed=5):
    rng = np.random.default_rng(seed + degree)
    taylor = np.array([1 / math.factorial(order) for order in range(degree + 1)])
    values = taylor * (1 + 0.15 * rng.normal(size=(heads, degree + 1)))
    return jnp.asarray(values[0] if sharing == "global" else values, jnp.float32)


def _dense_reference(q, k, v, coefficients, log_scale=None):
    """Independent dense p(s) V / rowsum over the causal matrix; callers run it in FP64."""
    if log_scale is not None:
        unit = lambda x: x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + EPS)
        alpha = jnp.exp(jnp.broadcast_to(log_scale, (q.shape[2], 1)))[None, :, :, None]
        scores = alpha * jnp.einsum("bthd,bshd->bhts", unit(q), unit(k))
    else:
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    coefficients = jnp.broadcast_to(coefficients, (q.shape[2], coefficients.shape[-1]))
    kernel = sum(coefficients[:, order][None, :, None, None] * scores ** order
                 for order in range(coefficients.shape[-1]))
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    kernel = jnp.where(causal, kernel, 0.0)
    return jnp.einsum("bhts,bshd->bthd", kernel / jnp.sum(kernel, axis=-1, keepdims=True), v)


def _assert_relative(observed, expected, tolerance, label):
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
    coefficients = _near_taylor_coefficients(degree, q.shape[2], sharing)

    def blocked(q, k, v, c):
        return attention_core(q, k, v, backend="blocked", attention_degree=degree, attention_normalization=True,
                              polynomial_coefficients=c)

    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(blocked, q, k, v, coefficients)
        grads = pullback(cotangent)
    expected, expected_grads = _fp64_vjp(_dense_reference, (q, k, v, coefficients), cotangent)
    _assert_relative(output, expected, 3e-5, f"m={degree} output")
    for name, observed, reference in zip("qkvc", grads, expected_grads, strict=True):
        _assert_relative(observed, reference, 1e-4, f"m={degree} d{name}")
    assert grads[3].shape == coefficients.shape


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("sharing", SHARINGS)
def test_composed_with_qk_norm_matches_independent_fp64(degree, sharing):
    (q, k, v), cotangent = _operands(scale=1.7)
    coefficients = _near_taylor_coefficients(degree, q.shape[2], sharing)
    log_scale = jnp.full((q.shape[2], 1) if sharing == "per_head" else (1,), math.log(1.5), jnp.float32)

    def composed(q, k, v, c, log_scale):
        scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
        return blocked_dot_product_attention(scaled_q, unit_k, v, attention_degree=degree, attention_normalization=True,
                                             polynomial_coefficients=c, query_block_size=4)

    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(composed, q, k, v, coefficients, log_scale)
        grads = pullback(cotangent)
    reference = lambda q, k, v, c, g: _dense_reference(q, k, v, c, g)
    expected, expected_grads = _fp64_vjp(reference, (q, k, v, coefficients, log_scale), cotangent)
    _assert_relative(output, expected, 3e-5, f"m={degree} output")
    for name, observed, target in zip(("q", "k", "v", "c", "gamma"), grads, expected_grads, strict=True):
        _assert_relative(observed, target, 1e-4, f"m={degree} d{name}")


@pytest.mark.parametrize("degree", DEGREES)
def test_taylor_initialized_coefficients_match_fixed_normalized_taylor(degree):
    (q, k, v), cotangent = _operands(seed=12)
    taylor = jnp.asarray([1 / math.factorial(order) for order in range(degree + 1)], jnp.float32)
    fixed = partial(taylor_attention, attention_degree=degree, backend="blocked")
    learned = partial(attention_core, backend="blocked", attention_degree=degree, attention_normalization=True,
                      polynomial_coefficients=jnp.broadcast_to(taylor, (q.shape[2], degree + 1)))
    np.testing.assert_allclose(learned(q, k, v), fixed(q, k, v), rtol=2e-6, atol=5e-7)
    expected = jax.vjp(fixed, q, k, v)[1](cotangent)
    actual = jax.vjp(learned, q, k, v)[1](cotangent)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=3e-6)


def test_signed_row_sums_are_valid_and_zero_sums_poison_only_their_rows():
    # p(s) = 2 - s with one query whose causal kernel values cancel exactly.
    coefficients = jnp.asarray([[2.0, -1.0]], jnp.float32)
    q = jnp.ones((1, 4, 1, 1), jnp.float32)
    k = jnp.asarray([1.0, 3.0, 5.0, -2.0], jnp.float32).reshape(1, 4, 1, 1)  # kernels 1, -1, -3, 4
    v = jnp.asarray([1.0, 2.0, 3.0, 4.0], jnp.float32).reshape(1, 4, 1, 1)
    output, diagnostics = blocked_dot_product_attention(q, k, v, attention_degree=1, attention_normalization=True,
                                                        polynomial_coefficients=coefficients, return_diagnostics=True)
    values = np.asarray(output).reshape(-1)
    # Row sums: 1, 0, -3, 1. Row 1 is exactly zero and poisoned; row 2 is negative and valid.
    np.testing.assert_allclose(values[0], 1.0)
    assert np.isnan(values[1])
    np.testing.assert_allclose(values[2], (1 * 1 - 1 * 2 - 3 * 3) / -3.0, rtol=1e-6)
    np.testing.assert_allclose(values[3], (1 - 2 - 9 + 16) / 1.0, rtol=1e-6)
    assert int(diagnostics["invalid_normalizer_count"]) == 1
    assert int(diagnostics["negative_weight_count"]) > 0
    assert float(diagnostics["normalizer_min"]) == -3.0


def test_configuration_validation_and_backend_selection():
    config = _model_config(2, "per_head")  # normalized learned coefficients are valid on the blocked backend
    assert config.attention_normalization and config.attention_coefficients == "learned"
    with pytest.raises(ValueError, match="no fused kernel"):
        _model_config(2, "per_head", attention_backend="fused")
    auto = _model_config(2, "per_head", attention_backend="auto")
    assert resolve_attention_backend(auto, dtype="bfloat16", platform="gpu") == "blocked"
    assert resolve_attention_backend(auto, dtype="float32", platform="cpu") == "blocked"
    unnormalized = replace(auto, attention_normalization=False)
    assert resolve_attention_backend(unnormalized, dtype="bfloat16", platform="gpu") == "fused"
    # An explicit fused request cannot be built as a ModelConfig; the resolver rejects it independently.
    explicit = SimpleNamespace(**{**vars(auto), "attention_backend": "fused"})
    with pytest.raises(ValueError, match="no fused kernel"):
        resolve_attention_backend(explicit, dtype="bfloat16", platform="gpu")
    with pytest.raises(ValueError, match="quadratic"):
        blocked_dot_product_attention(*_operands()[0], attention_degree=2, attention_normalization=True,
                                      quadratic_mode="positive", coefficients=jnp.ones((2,)),
                                      polynomial_coefficients=jnp.ones((3,)))


@pytest.mark.parametrize("qk_norm", (False, True))
def test_identity_diagnostics_and_mirror_for_the_normalized_learned_kernel(tmp_path, qk_norm):
    config = _config(tmp_path, 3, "per_head", qk_norm)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial"
    assert identity["attention_signed"] is True
    assert identity["attention_output_normalization"] == "signed_row_sum"
    assert identity["attention_coefficient_count"] == 2 * 2 * 4
    assert identity.get("qk_norm", False) is qk_norm
    state = create_state(config, jax.random.PRNGKey(0))
    # Negative constants make kernels and row sums negative: valid for this signed mechanism.
    state.params["polynomial_coefficients"][...] = state.params["polynomial_coefficients"][...].at[..., 0].set(-4.0)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == "unshifted_learned_polynomial"
    assert any(layer["kernel_min"] < 0 for layer in report["layers"])
    assert all("min_abs_normalizer" in layer and "negative_weight_fraction" in layer for layer in report["layers"])
    assert report["coefficients"]["effective"][1][1][0] == -4.0
    assert ("qk_norm_scale" in report) is qk_norm
    metrics = tracking.attention_metrics(report)
    assert metrics["attention/learned_polynomial/coefficients/constant/mean"] == -4.0
    if qk_norm:
        assert metrics["attention/learned_polynomial/qk_norm_scale/alpha/max"] == pytest.approx(math.sqrt(2))
    assert tracking.run_tags(config, {}) == (["learned_polynomial", "dk2", "m3"] + (["qk_norm"] if qk_norm else []))


@pytest.mark.parametrize("degree", DEGREES)
def test_production_parameter_counts_and_softmax_backbone(degree):
    base = load_config(BASE_CONFIG).model
    production = replace(base, attention_degree=degree, attention_normalization=True, attention_backend="blocked",
                         attention_coefficients="learned", attention_coefficient_sharing="per_head", qk_norm=True)
    params = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    heads = production.num_layers * production.num_heads
    expected = 199_906_304 + heads * (degree + 1) + heads
    assert sum(x.size for x in jax.tree.leaves(params)) == expected == model_parameter_count(production)
    config = Config()
    config.model = production
    facts = matches_softmax_backbone_initialization(config)
    assert set(facts["mechanism_parameters"]) == {"polynomial_coefficients", "qk_norm_log_scale"}
    assert facts["mechanism_parameters"]["polynomial_coefficients"]["initial"] == [
        1 / math.factorial(order) for order in range(degree + 1)]


def test_model_forward_backward_is_finite_and_causal_with_qk_norm():
    config = _model_config(4, "per_head", qk_norm=True, vocab_size=128, max_seq_len=16, hidden_size=32,
                           num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8, intermediate_size=64)
    model = build_model(config, 0)
    tokens = jnp.arange(32).reshape(2, 16)
    logits, diagnostics = model(tokens, return_diagnostics=True)
    assert bool(jnp.all(jnp.isfinite(logits)))
    for layer in diagnostics:
        assert abs(float(layer["score_min"])) <= math.sqrt(8) * (1 + 1e-5)
        assert abs(float(layer["score_max"])) <= math.sqrt(8) * (1 + 1e-5)
        assert int(layer["invalid_normalizer_count"]) == 0
    graph, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda candidate: jnp.square(nnx.merge(graph, candidate)(tokens)).mean())(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads["polynomial_coefficients"][...])) > 0
    assert float(jnp.linalg.norm(grads["qk_norm_log_scale"][...])) > 0
    altered = tokens.at[:, 9:].set(127)
    np.testing.assert_array_equal(model(altered)[:, :9], logits[:, :9])


@pytest.mark.parametrize("degree,qk_norm", [(2, True), (1, False)])
def test_training_resume_and_identity(tmp_path, degree, qk_norm):
    config = tiny_config(tmp_path / "full")
    config.model.attention_backend = "blocked"
    config.model.attention_degree = degree
    config.model.attention_normalization = True
    config.model.attention_coefficients = "learned"
    config.model.attention_coefficient_sharing = "per_head"
    config.model.qk_norm = qk_norm
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert resumed["restored_step"] == 1
    assert full["attention_kernel"] == resumed["attention_kernel"] == "learned_polynomial"
    assert full["attention_backend"] == "blocked"
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
    coefficients = np.asarray(states[0].params["polynomial_coefficients"][...])
    taylor = np.array([1 / math.factorial(order) for order in range(degree + 1)], np.float32)
    assert np.all(coefficients.reshape(-1, degree + 1) != taylor)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validations = [record for record in records if record.get("event") == "validation"]
    assert validations and all(record["attention_diagnostics"]["valid"] for record in validations)
    assert all(record["attention_kernel"] == "learned_polynomial" for record in validations)
    json.dumps(records, allow_nan=False)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_production_context_bf16_with_cosine_operands(degree):
    """The blocked reference at the production context: finite, and matching FP64 on a causal prefix."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    (q, k, v), cotangent = _operands(time=2048, heads=2, qk_dim=64, value_dim=64, seed=92, scale=1.0,
                                     dtype=jnp.bfloat16)
    coefficients = jnp.broadcast_to(jnp.asarray([1 / math.factorial(order) for order in range(degree + 1)],
                                                jnp.float32), (2, degree + 1))
    log_scale = jnp.asarray([[math.log(8.0)], [math.log(3.0)]], jnp.float32)

    def function(q, k, v, c, log_scale):
        scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
        return attention_core(scaled_q, unit_k, v, backend="blocked", attention_degree=degree,
                              attention_normalization=True, polynomial_coefficients=c)

    @jax.jit
    def forward_and_vjp(operands, cotangent):
        output, pullback = jax.vjp(function, *operands)
        return output, pullback(cotangent)

    output, grads = forward_and_vjp((q, k, v, coefficients, log_scale), cotangent)
    assert output.dtype == jnp.bfloat16 and bool(jnp.all(jnp.isfinite(output)))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grads)
    assert all(float(jnp.linalg.norm(grad.astype(jnp.float32))) > 0 for grad in grads)
    # The FP64 comparison certifies the arithmetic, so it runs where the signed kernel is
    # well conditioned: degree 1 at scale 2 (1 + s over [-2, 2]). At scale 8 its row sums
    # of 1 + s cancel so heavily that the constant coefficient's gradient is a difference
    # of near-equal FP32 sums (30% off FP64 on B300), a property of the kernel, not the code.
    prefix_log_scale = jnp.asarray([[math.log(2.0 if degree == 1 else 8.0)]], jnp.float32)
    prefix = tuple(x[:, :129, :1] for x in (q, k, v))
    prefix_output, prefix_grads = forward_and_vjp((*prefix, coefficients[:1], prefix_log_scale), cotangent[:, :129, :1])
    expected, expected_grads = _fp64_vjp(lambda q, k, v, c, g: _dense_reference(q, k, v, c, g),
                                         (*prefix, coefficients[:1], prefix_log_scale), cotangent[:, :129, :1])
    _assert_relative(prefix_output, expected, 0.006, f"m={degree} prefix output")
    for name, observed, reference, tolerance in zip(("q", "k", "v", "c", "gamma"), prefix_grads, expected_grads,
                                                    (0.025, 0.025, 0.015, 0.02, 0.02), strict=True):
        _assert_relative(observed, reference, tolerance, f"m={degree} prefix d{name}")
