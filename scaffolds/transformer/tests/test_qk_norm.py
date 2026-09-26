"""Mathematical, parameter, reporting, and training contracts for QK-norm cosine scores.

QK-norm is a score option, not a kernel: the module hands every dot-product
mechanism ``q' = exp(gamma) sqrt(d_K) q/||q||`` and ``k' = k/||k||``. Tests compare
the composed operator against an independent FP64 reference of ``kernel(exp(gamma)
cos(q, k))`` and check that the fused kernels see it unchanged.
"""

import json
import math
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import matches_softmax_backbone_initialization
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import cosine_score_operands, initial_scale, unit_normalize
from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.state import create_state

EPS = 1e-6
# (attention_degree, attention_normalization, attention_coefficients): every dot-product family.
MECHANISMS = {
    "softmax": (None, True, "fixed"),
    "taylor2": (2, True, "fixed"),
    "unnormalized_exp": (None, False, "fixed"),
    "unnormalized_taylor3": (3, False, "fixed"),
    "learned_polynomial2": (2, False, "learned"),
}
SHARINGS = ("global", "per_head")


def _model_config(mechanism="softmax", sharing="per_head", **updates):
    degree, normalized, coefficients = MECHANISMS[mechanism]
    values = dict(vocab_size=128, max_seq_len=16, num_layers=2, hidden_size=32, num_heads=4, num_kv_heads=4,
                  qk_head_dim=8, value_head_dim=8, intermediate_size=64, tie_embeddings=False,
                  attention_backend="blocked", attention_degree=degree, attention_normalization=normalized,
                  attention_coefficients=coefficients, attention_coefficient_sharing="per_head",
                  qk_norm=True, qk_norm_sharing=sharing)
    return ModelConfig(**{**values, **updates})


def _operands(time=13, heads=3, qk_dim=8, value_dim=5, seed=11, dtype=jnp.float32):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(1.7 * rng.normal(size=(2, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, time, heads, value_dim)), dtype)
    cotangent = jnp.asarray(rng.normal(size=v.shape), dtype)
    return (q, k, v), cotangent


def _log_scale(heads, sharing, seed=3):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.2, 1.3, size=(heads, 1))
    # The model stores one shared value as [1] and per-head values as [heads, 1].
    return jnp.asarray(values[0] if sharing == "global" else values, jnp.float32)


def _dense_reference(q, k, v, log_scale, *, degree, normalized, coefficients=None):
    """Independent dense evaluation of kernel(exp(gamma) cos(q, k)) V; callers run it in FP64."""
    unit = lambda x: x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + EPS)
    cosine = jnp.einsum("bthd,bshd->bhts", unit(q), unit(k))
    scores = jnp.exp(jnp.broadcast_to(log_scale, (q.shape[2], 1)))[None, :, :, None] * cosine
    if coefficients is not None:
        coefficients = jnp.broadcast_to(coefficients, (q.shape[2], coefficients.shape[-1]))
        kernel = sum(coefficients[:, order][None, :, None, None] * scores ** order
                     for order in range(coefficients.shape[-1]))
    elif degree is None:
        kernel = jnp.exp(scores)
    else:
        kernel = sum(scores ** order / math.factorial(order) for order in range(degree + 1))
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    kernel = jnp.where(causal, kernel, 0.0)
    if normalized:
        kernel = kernel / jnp.sum(kernel, axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", kernel, v)


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


def test_unit_normalize_matches_fp64_and_keeps_zero_rows_finite():
    (q, _, _), _ = _operands()
    q = q.at[0, 3, 1].set(0.0)
    normalized = unit_normalize(q, EPS)
    with jax.enable_x64():
        q64 = q.astype(jnp.float64)
        expected = q64 / jnp.sqrt(jnp.sum(jnp.square(q64), axis=-1, keepdims=True) + EPS)
    assert normalized.dtype == q.dtype
    np.testing.assert_allclose(normalized, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(normalized[0, 3, 1], jnp.zeros(q.shape[-1]))
    norms = jnp.linalg.norm(normalized, axis=-1)
    assert float(jnp.max(jnp.abs(jnp.where(norms > 0, norms - 1.0, 0.0)))) < 1e-5
    assert unit_normalize(q.astype(jnp.bfloat16), EPS).dtype == jnp.bfloat16
    with pytest.raises(ValueError, match="eps"):
        unit_normalize(q, 0.0)


@pytest.mark.parametrize("sharing", SHARINGS)
def test_operands_give_bounded_scaled_cosine_scores_in_the_mechanisms_arithmetic(sharing):
    (q, k, _), _ = _operands()
    log_scale = _log_scale(q.shape[2], sharing)
    scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
    scores = jnp.einsum("bthd,bshd->bhts", scaled_q, unit_k) / math.sqrt(q.shape[-1])
    with jax.enable_x64():
        unit = lambda x: x / jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float64)), axis=-1, keepdims=True) + EPS)
        cosine = jnp.einsum("bthd,bshd->bhts", unit(q), unit(k))
        alpha = jnp.exp(jnp.broadcast_to(log_scale.astype(jnp.float64), (q.shape[2], 1)))
        expected = alpha[None, :, :, None] * cosine
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)
    bound = np.asarray(jnp.exp(log_scale)).reshape(-1)
    bound = np.broadcast_to(bound, (q.shape[2],))
    assert np.all(np.abs(np.asarray(scores)).max(axis=(0, 2, 3)) <= bound * (1 + 1e-5))
    assert scaled_q.dtype == unit_k.dtype == q.dtype
    bf16 = cosine_score_operands(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), log_scale, eps=EPS)
    assert all(operand.dtype == jnp.bfloat16 for operand in bf16)
    with pytest.raises(ValueError, match="shape"):
        cosine_score_operands(q, k, jnp.zeros((2, 1)), eps=EPS)


def test_normalization_commutes_with_rope():
    (q, _, _), _ = _operands()
    positions = jnp.arange(q.shape[1])
    before = apply_rope(unit_normalize(q, EPS), positions)
    after = unit_normalize(apply_rope(q, positions), EPS)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
@pytest.mark.parametrize("sharing", SHARINGS)
def test_composed_operator_and_all_gradients_match_independent_fp64(mechanism, sharing):
    degree, normalized, learned = MECHANISMS[mechanism]
    (q, k, v), cotangent = _operands()
    log_scale = _log_scale(q.shape[2], sharing)
    coefficients = None
    if learned == "learned":
        rng = np.random.default_rng(degree)
        coefficients = jnp.asarray(rng.normal(size=(q.shape[2], degree + 1)), jnp.float32)
    extra = {} if coefficients is None else {"polynomial_coefficients": coefficients}

    def composed(q, k, v, log_scale):
        scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
        return blocked_dot_product_attention(scaled_q, unit_k, v, attention_degree=degree,
                                             attention_normalization=normalized, query_block_size=4, **extra)

    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(composed, q, k, v, log_scale)
        grads = pullback(cotangent)
    reference = partial(_dense_reference, degree=degree, normalized=normalized, coefficients=coefficients)
    expected, expected_grads = _fp64_vjp(reference, (q, k, v, log_scale), cotangent)
    _assert_relative(output, expected, 3e-5, f"{mechanism} output")
    for name, observed, target in zip(("q", "k", "v", "gamma"), grads, expected_grads, strict=True):
        _assert_relative(observed, target, 1e-4, f"{mechanism} d{name}")
    assert grads[3].shape == log_scale.shape


@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
@pytest.mark.parametrize("sharing", SHARINGS)
def test_model_owns_the_log_scale_with_its_declared_initialization(mechanism, sharing):
    config = _model_config(mechanism, sharing)
    model = build_model(config, 0)
    params = nnx.state(model, nnx.Param)
    log_scale = np.asarray(params["qk_norm_log_scale"][...])
    assert log_scale.dtype == np.float32
    assert log_scale.shape == ((1,) if sharing == "global" else (config.num_layers, config.num_heads, 1))
    np.testing.assert_array_equal(log_scale, np.float32(math.log(math.sqrt(config.qk_head_dim))))
    assert initial_scale(config) == math.sqrt(config.qk_head_dim)
    assert initial_scale(replace(config, qk_norm_initial_scale=2.5)) == 2.5
    explicit = nnx.state(build_model(replace(config, qk_norm_initial_scale=2.5), 0), nnx.Param)
    np.testing.assert_array_equal(np.asarray(explicit["qk_norm_log_scale"][...]), np.float32(math.log(2.5)))
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    assert count == model_parameter_count(config)
    without = model_parameter_count(replace(config, qk_norm=False))
    assert count - without == (1 if sharing == "global" else config.num_layers * config.num_heads)
    names = [spec.name for spec in attention_parameter_specs(config)]
    assert names[-1] == "qk_norm_log_scale"
    if mechanism == "learned_polynomial2":
        assert names == ["polynomial_coefficients", "qk_norm_log_scale"]
    facts = matches_softmax_backbone_initialization(_config_for(config))
    assert "qk_norm_log_scale" in facts["mechanism_parameters"]
    assert facts["mechanism_parameters"]["qk_norm_log_scale"]["weight_decay"] is False


def _config_for(model_config, path=None):
    config = Config()
    config.model = model_config
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 128, 9, 2
    config.train.micro_batch_size = 2
    if path is not None:
        config.train.workdir = path
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_model_scores_are_bounded_and_invariant_to_projection_scale(mechanism):
    config = _model_config(mechanism)
    model = build_model(config, 0)
    tokens = jnp.arange(32).reshape(2, 16)
    logits, diagnostics = model(tokens, return_diagnostics=True)
    assert bool(jnp.all(jnp.isfinite(logits)))
    alpha = math.sqrt(config.qk_head_dim)
    for layer in diagnostics:
        assert -alpha * (1 + 1e-5) <= float(layer["score_min"]) <= float(layer["score_max"]) <= alpha * (1 + 1e-5)
        assert all(int(value) == 0 for key, value in layer.items() if key.endswith("invalid_score_count"))
    # Cosine scores ignore the magnitude of Q and K: rescaling the projections changes nothing.
    for layer_name in model.layer_names:
        attention = getattr(model, layer_name).attention
        attention.q_proj.kernel[...] = attention.q_proj.kernel[...] * 3.0
        attention.k_proj.kernel[...] = attention.k_proj.kernel[...] * 0.25
    np.testing.assert_allclose(model(tokens), logits, rtol=1e-4, atol=1e-5)
    # The learned scale does change the operator.
    model.qk_norm_log_scale[...] = model.qk_norm_log_scale[...] + 0.5
    assert float(jnp.max(jnp.abs(model(tokens) - logits))) > 1e-3
    graph, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda candidate: jnp.square(nnx.merge(graph, candidate)(tokens)).mean())(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads["qk_norm_log_scale"][...])) > 0


def test_manual_backend_matches_blocked_and_the_parameter_is_mandatory():
    kwargs = dict(hidden_size=32, num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8, qk_norm=True)
    blocked = MultiHeadAttention(**kwargs, backend="blocked", rngs=nnx.Rngs(8))
    manual = MultiHeadAttention(**kwargs, backend="manual", rngs=nnx.Rngs(8))
    x = jax.random.normal(jax.random.key(1), (2, 11, 32))
    parameters = {"qk_norm_log_scale": _log_scale(4, "per_head")}
    np.testing.assert_allclose(manual(x, attention_parameters=parameters),
                               blocked(x, attention_parameters=parameters), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="qk_norm_log_scale"):
        blocked(x)
    plain = MultiHeadAttention(**{**kwargs, "qk_norm": False}, backend="blocked", rngs=nnx.Rngs(8))
    with pytest.raises(ValueError, match="qk_norm_log_scale"):
        plain(x, attention_parameters=parameters)
    with pytest.raises(ValueError, match="degree 0"):
        MultiHeadAttention(**kwargs, attention_degree=0, rngs=nnx.Rngs(8))


def test_optimizer_excludes_the_log_scale_from_weight_decay():
    config = _config_for(_model_config())
    config.optim.warmup_steps = 0
    state = create_state(config, jax.random.PRNGKey(0))
    tx = make_optimizer(config)
    grads = jax.tree.map(jnp.zeros_like, state.params)
    updates, _ = tx.update(grads, tx.init(state.params), state.params)
    updated = optax.apply_updates(state.params, updates)
    np.testing.assert_array_equal(updated["qk_norm_log_scale"][...], state.params["qk_norm_log_scale"][...])
    kernel = state.params["layers_0"]["attention"]["q_proj"]["kernel"][...]
    factor = 1 - config.optim.learning_rate * config.optim.weight_decay
    np.testing.assert_allclose(updated["layers_0"]["attention"]["q_proj"]["kernel"][...], kernel * factor, rtol=1e-6)


@pytest.mark.parametrize("updates,error", [
    (dict(attention_mechanism="gla", attention_degree=None, attention_normalization=True), "GLA"),
    (dict(qk_norm_eps=0.0), "qk_norm_eps"),
    (dict(qk_norm_eps=float("nan")), "qk_norm_eps"),
    (dict(qk_norm_initial_scale=-1.0), "qk_norm_initial_scale"),
    (dict(qk_norm_initial_scale=float("inf")), "qk_norm_initial_scale"),
    (dict(qk_norm_sharing="head"), "sharing"),
    (dict(qk_norm="true"), "boolean"),
])
def test_invalid_qk_norm_configurations_are_rejected(updates, error):
    with pytest.raises(ValueError, match=error):
        _model_config(**updates)


def test_qk_norm_resolves_from_the_mechanism_when_unset():
    # Unset: on for dot-product kernels that read scores, off for GLA and degree 0.
    assert _model_config().qk_norm is True
    assert replace(_model_config(), qk_norm=None, attention_degree=0).qk_norm is False
    assert replace(_model_config(), qk_norm=None, attention_mechanism="gla", attention_degree=None,
                   attention_normalization=True).qk_norm is False
    # Degree 0 never reads Q or K, so an explicit request resolves off instead of failing.
    assert _model_config(attention_degree=0).qk_norm is False


def test_recipes_round_trip_and_legacy_recipes_compare_equal(tmp_path):
    for initial in (None, 8.0):
        config = _config_for(_model_config(qk_norm_initial_scale=initial), tmp_path)
        path = tmp_path / f"qk-{initial}.yaml"
        save_config(config, path)
        restored = load_config(path)
        assert restored.model.qk_norm is True
        assert restored.model.qk_norm_initial_scale == initial
        assert restored.model.qk_norm_eps == 1e-6
        assert config_to_dict(restored) == config_to_dict(config)
    # A run recorded before QK-norm existed trained without it: it equals the plain
    # configuration, not today's default, which is QK-norm on.
    plain = Config()
    plain.model = replace(plain.model, qk_norm=False)
    legacy = config_to_dict(plain)
    for key in ("qk_norm", "qk_norm_sharing", "qk_norm_initial_scale", "qk_norm_eps"):
        legacy["model"].pop(key)
    assert config_dicts_equal(legacy, config_to_dict(plain))
    assert not config_dicts_equal(legacy, config_to_dict(Config()))
    assert not config_dicts_equal(legacy, config_to_dict(_config_for(_model_config())))
    assert Config().model.qk_norm is True
    assert "qk_norm" not in attention_identity(plain)
    identity = attention_identity(_config_for(_model_config("unnormalized_taylor3")))
    assert identity["qk_norm"] is True
    assert identity["attention_kernel"] == "taylor_unnormalized"
    assert identity["qk_norm_initial_scale"] == math.sqrt(8)
    assert identity["qk_norm_parameter_count"] == 8


@pytest.mark.parametrize("sharing", SHARINGS)
def test_validation_diagnostics_report_scales_and_mirror_them(tmp_path, sharing):
    config = _config_for(_model_config("learned_polynomial2", sharing), tmp_path)
    state = create_state(config, jax.random.PRNGKey(0))
    state.params["qk_norm_log_scale"][...] = state.params["qk_norm_log_scale"][...] + 0.25
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["qk_norm"] is True
    scales = report["qk_norm_scale"]
    raw = np.asarray(scales["raw"])
    assert raw.shape == ((1,) if sharing == "global" else (config.model.num_layers, config.model.num_heads, 1))
    np.testing.assert_allclose(raw, math.log(math.sqrt(8)) + 0.25, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(scales["effective"]), np.exp(raw), rtol=1e-12)
    assert scales["raw_parameter_names"] == ["gamma"] and scales["effective_coefficient_names"] == ["alpha"]
    assert "coefficients" in report
    payload = tracking.attention_metrics(report)
    prefix = "attention/learned_polynomial_unnormalized"
    if sharing == "global":
        np.testing.assert_allclose(payload[f"{prefix}/qk_norm_scale/alpha"], float(np.exp(raw[0])))
    else:
        for statistic in ("mean", "min", "max"):
            np.testing.assert_allclose(payload[f"{prefix}/qk_norm_scale/alpha/{statistic}"], float(np.exp(raw[0, 0, 0])))
    assert f"{prefix}/coefficients/constant/mean" in payload or f"{prefix}/coefficients/constant" in payload
    state.params["qk_norm_log_scale"][...] = jnp.full_like(state.params["qk_norm_log_scale"][...], jnp.inf)
    poisoned = attention_diagnostics(state, config, sample, provenance)
    assert not poisoned["valid"] and "Nonfinite QK-norm scales" in poisoned["issues"]
    json.dumps(poisoned, allow_nan=False)


def _qk_norm_training(config, mechanism="softmax"):
    degree, normalized, coefficients = MECHANISMS[mechanism]
    config.model.attention_backend = "blocked"
    config.model.attention_degree = degree
    config.model.attention_normalization = normalized
    config.model.attention_coefficients = coefficients
    config.model.attention_coefficient_sharing = "per_head"
    config.model.qk_norm = True
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    return config


@pytest.mark.parametrize("mechanism", ("softmax", "learned_polynomial2"))
def test_training_resume_identity_and_recorded_scales(tmp_path, mechanism):
    config = _qk_norm_training(tiny_config(tmp_path / "full"), mechanism)
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    initial_audit, resumed_audit = interrupted["checkpoint_audit"], resumed["checkpoint_audit"]
    assert initial_audit["final_state_sha256"] == initial_audit["checkpoint_state_sha256"]
    assert initial_audit["final_state_sha256"] == resumed_audit["start_state_sha256"]
    assert resumed["restored_step"] == 1
    assert full["qk_norm"] is resumed["qk_norm"] is True
    assert full["attention_kernel"] == resumed["attention_kernel"]
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
    log_scale = np.asarray(states[0].params["qk_norm_log_scale"][...])
    assert log_scale.shape == (config.model.num_layers, config.model.num_heads, 1)
    assert np.all(log_scale != np.float32(math.log(math.sqrt(config.model.qk_head_dim))))
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validations = [record for record in records if record.get("event") == "validation"]
    assert validations and all(record["attention_diagnostics"]["valid"] for record in validations)
    assert all(record["qk_norm"] is True for record in validations)
    reported = np.asarray(validations[-1]["attention_diagnostics"]["qk_norm_scale"]["raw"])
    np.testing.assert_allclose(reported, log_scale, rtol=1e-7)
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["qk_norm"] is True and metadata["qk_norm_sharing"] == "per_head"
    json.dumps(records, allow_nan=False)


@pytest.mark.gpu
@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_gpu_fused_paths_see_cosine_operands_unchanged(mechanism):
    """The fused kernels and cuDNN consume q'/k' exactly as the blocked reference does."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    degree, normalized, learned = MECHANISMS[mechanism]
    (q, k, v), cotangent = _operands(time=2048, heads=2, qk_dim=64, value_dim=64, seed=92, dtype=jnp.bfloat16)
    log_scale = jnp.asarray([[math.log(8.0)], [math.log(3.0)]], jnp.float32)
    extra = {}
    if learned == "learned":
        rng = np.random.default_rng(degree)
        extra["polynomial_coefficients"] = jnp.asarray(rng.normal(size=(2, degree + 1)), jnp.float32)

    def composed(backend):
        def function(q, k, v, log_scale):
            scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
            return attention_core(scaled_q, unit_k, v, backend=backend, attention_degree=degree,
                                  attention_normalization=normalized, **extra)

        @jax.jit
        def forward_and_vjp(operands, cotangent):
            output, pullback = jax.vjp(function, *operands)
            return output, pullback(cotangent)

        return forward_and_vjp

    fused = composed("fused")((q, k, v, log_scale), cotangent)
    blocked = composed("blocked")((q, k, v, log_scale), cotangent)
    assert fused[0].dtype == jnp.bfloat16
    tolerances = (0.006, 0.03, 0.03, 0.02, 0.02)
    for name, observed, reference, tolerance in zip(
            ("output", "dq", "dk", "dv", "dgamma"), (fused[0], *fused[1]), (blocked[0], *blocked[1]), tolerances,
            strict=True):
        _assert_relative(observed, reference, tolerance, f"{mechanism} {name}")
    scores = jnp.einsum("bthd,bshd->bhts", *cosine_score_operands(q, k, log_scale, eps=EPS)) / 8.0
    assert float(jnp.max(jnp.abs(scores[:, 0].astype(jnp.float32)))) <= 8.0 * 1.02
    assert float(jnp.max(jnp.abs(scores[:, 1].astype(jnp.float32)))) <= 3.0 * 1.02
