"""PaTH position encoding: transported scores, kernels, gradients, model integration."""

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
from scaffolds.transformer.attention.path import (
    PATH_ARGUMENTS, chunk_transforms, path_attention, path_gates, path_scores_reference,
)
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state


def _inputs(*, batch=2, time=48, heads=3, dim=8, value_dim=5, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    q, k = rng.normal(size=(batch, time, heads, dim)), rng.normal(size=(batch, time, heads, dim))
    v = rng.normal(size=(batch, time, heads, value_dim))
    w = rng.normal(size=(batch, time, heads, dim))
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    beta = 2 / (1 + np.exp(-rng.normal(size=(batch, time, heads))))
    taylor = 1 / np.array([math.factorial(r) for r in range(degree + 1)])
    c = taylor + 0.3 * rng.normal(size=(batch, time, heads, degree + 1))
    cotangent = rng.normal(size=v.shape)
    return q, k, v, w, beta, c, cotangent


def _log_decay(shape, seed=0):
    """A forget gate's cumulative log decay L [B, T, H]: nonincreasing, about -0.2 per token."""
    return np.cumsum(-0.3 * np.logaddexp(0, np.random.default_rng(seed).normal(size=shape)), axis=1)


def _dense_reference(q, k, v, w, beta, c=None, log_decay=None, exponential=False):
    """Independent outputs from the definition (NumPy FP64)."""
    scores = path_scores_reference(q, k, w, beta) / math.sqrt(q.shape[-1])
    causal = np.tril(np.ones(scores.shape[-2:], bool))
    if log_decay is not None:
        scores = scores + np.moveaxis(log_decay, 1, 2)[..., :, None] - np.moveaxis(log_decay, 1, 2)[..., None, :]
    if exponential:
        weights = np.where(causal, np.exp(np.where(causal, scores, 0.0)), 0.0)
    elif c is None:
        masked = np.where(causal, scores, -np.inf)
        weights = np.exp(masked - masked.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
    else:
        weights = sum(np.moveaxis(c[..., r], 1, 2)[..., None] * scores ** r for r in range(c.shape[-1]))
        weights = np.where(causal, weights, 0.0)
    return np.einsum("bhts,bshv->bthv", weights, v)


def _jax_definition(q, k, v, w, beta, c=None, log_decay=None, exponential=False):
    """Differentiable definition (sequential transport per query), for gradient checks."""
    batch, time, heads, dim = q.shape
    scale = 1 / math.sqrt(dim)
    rows = []
    for t in range(time):
        transport = jnp.broadcast_to(jnp.eye(dim), (batch, heads, dim, dim))
        scores = []
        for s in range(t, -1, -1):
            score = jnp.einsum("bhd,bhde,bhe->bh", q[:, t], transport, k[:, s]) * scale
            scores.append(score if log_decay is None else score + log_decay[:, t] - log_decay[:, s])
            householder = jnp.eye(dim) - beta[:, s, :, None, None] * jnp.einsum("bhd,bhe->bhde", w[:, s], w[:, s])
            transport = transport @ householder
        scores = jnp.stack(scores[::-1], axis=-1)  # [B, H, t+1]
        if exponential:
            weights = jnp.exp(scores)
        elif c is None:
            weights = jax.nn.softmax(scores, axis=-1)
        else:
            weights = sum(c[:, t, :, r, None] * scores ** r for r in range(c.shape[-1]))
        rows.append(jnp.einsum("bhs,bshv->bhv", weights, v[:, :t + 1]))
    return jnp.moveaxis(jnp.stack(rows, axis=1), 2, 2)  # [B, T, H, V]


def _relative(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("chunk", (8, 16, 48))
@pytest.mark.parametrize("kernel", ("softmax", "query_polynomial"))
def test_chunked_outputs_match_the_definition(chunk, kernel):
    q, k, v, w, beta, c, _ = _inputs()
    coefficients = jnp.asarray(c, jnp.float32) if kernel == "query_polynomial" else None
    with jax.default_matmul_precision("highest"):
        actual = path_attention(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)), kernel=kernel,
                                coefficients=coefficients, chunk_size=chunk)
    assert _relative(actual, _dense_reference(q, k, v, w, beta, c if coefficients is not None else None)) < 1e-5


@pytest.mark.parametrize("chunk", (8, 16, 48))
@pytest.mark.parametrize("kernel,decayed", [("exp", False), ("softmax", True), ("exp", True)])
def test_exponential_kernels_and_the_forget_gate_match_the_definition(chunk, kernel, decayed):
    q, k, v, w, beta, _, _ = _inputs()
    log_decay = _log_decay(beta.shape) if decayed else None
    with jax.default_matmul_precision("highest"):
        actual = path_attention(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)), kernel=kernel,
                                log_decay=None if log_decay is None else jnp.asarray(log_decay, jnp.float32),
                                chunk_size=chunk)
    expected = _dense_reference(q, k, v, w, beta, log_decay=log_decay, exponential=kernel == "exp")
    assert _relative(actual, expected) < 1e-5


def test_decayed_softmax_matches_the_authors_reference():
    """FLA adds its gate to the unscaled scores, so its per-token gate is log alpha * sqrt(d_K)."""
    torch = pytest.importorskip("torch")
    from fla.ops.path_attn.naive import naive_path_attn

    q, k, v, w, beta, _, _ = _inputs(time=64, heads=2, dim=16, value_dim=16, seed=1)
    log_decay = _log_decay(beta.shape, seed=2)
    log_alpha = np.diff(log_decay, axis=1, prepend=0.0)
    as_torch = lambda x: torch.tensor(x, dtype=torch.float32)
    expected = naive_path_attn(as_torch(q), as_torch(k), as_torch(v), as_torch(w), as_torch(beta),
                               as_torch(log_alpha * 4.0), scale=16 ** -0.5, chunk_size=16).numpy()
    with jax.default_matmul_precision("highest"):
        actual = path_attention(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)), kernel="softmax",
                                log_decay=jnp.asarray(log_decay, jnp.float32), chunk_size=16)
    assert _relative(actual, expected) < 1e-5


def test_softmax_matches_the_authors_reference():
    torch = pytest.importorskip("torch")
    from fla.ops.path_attn.naive import naive_path_attn

    q, k, v, w, beta, _, _ = _inputs(time=64, heads=2, dim=16, value_dim=16, seed=1)
    as_torch = lambda x: torch.tensor(x, dtype=torch.float32)
    expected = naive_path_attn(as_torch(q), as_torch(k), as_torch(v), as_torch(w), as_torch(beta),
                               torch.zeros(beta.shape), scale=16 ** -0.5, chunk_size=16).numpy()
    with jax.default_matmul_precision("highest"):
        actual = path_attention(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)), kernel="softmax",
                                chunk_size=16)
    assert _relative(actual, expected) < 1e-5


def test_identity_transport_recovers_plain_dot_product_scores():
    q, k, v, w, beta, c, _ = _inputs()
    zero = np.zeros_like(beta)
    scores = path_scores_reference(q, k, w, zero)
    np.testing.assert_allclose(scores[np.isfinite(scores)],
                               np.einsum("bthd,bshd->bhts", q, k)[np.isfinite(scores)], rtol=1e-10, atol=1e-10)
    with jax.default_matmul_precision("highest"):
        local, q_start, k_end, hmat = chunk_transforms(*(jnp.asarray(np.moveaxis(x, 2, 1)[:, :, None], jnp.float32)
                                                         for x in (q, k, w)), jnp.asarray(np.moveaxis(zero, 2, 1)[:, :, None]))
    np.testing.assert_allclose(hmat, 0, atol=1e-7)


@pytest.mark.parametrize("kernel", ("softmax", "query_polynomial"))
def test_gradients_match_the_differentiable_definition(kernel):
    q, k, v, w, beta, c, cotangent = _inputs(batch=1, time=12, heads=2, dim=4, value_dim=3, degree=2, seed=3)
    operands = [jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)]
    if kernel == "query_polynomial":
        operands.append(jnp.asarray(c, jnp.float32))
    chunked = lambda *x: path_attention(*x[:5], kernel=kernel, coefficients=x[5] if len(x) > 5 else None, chunk_size=4)
    definition = lambda *x: _jax_definition(*x[:5], x[5] if len(x) > 5 else None)
    with jax.default_matmul_precision("highest"):
        out, pull = jax.vjp(chunked, *operands)
        ref, pull_ref = jax.vjp(definition, *operands)
        grads, grads_ref = pull(jnp.asarray(cotangent, jnp.float32)), pull_ref(jnp.asarray(cotangent, jnp.float32))
    assert _relative(out, ref) < 1e-5
    for index, (actual, expected) in enumerate(zip(grads, grads_ref, strict=True)):
        assert np.linalg.norm(expected) > 0, index
        assert _relative(actual, expected) < 1e-4, index


@pytest.mark.parametrize("kernel", ("softmax", "exp"))
def test_decayed_gradients_match_the_differentiable_definition(kernel):
    q, k, v, w, beta, _, cotangent = _inputs(batch=1, time=12, heads=2, dim=4, value_dim=3, seed=5)
    operands = [jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta, _log_decay(beta.shape, seed=6))]
    chunked = lambda *x: path_attention(*x[:5], kernel=kernel, log_decay=x[5], chunk_size=4)
    definition = lambda *x: _jax_definition(*x[:5], log_decay=x[5], exponential=kernel == "exp")
    with jax.default_matmul_precision("highest"):
        out, pull = jax.vjp(chunked, *operands)
        ref, pull_ref = jax.vjp(definition, *operands)
        grads, grads_ref = pull(jnp.asarray(cotangent, jnp.float32)), pull_ref(jnp.asarray(cotangent, jnp.float32))
    assert _relative(out, ref) < 1e-5
    for index, (actual, expected) in enumerate(zip(grads, grads_ref, strict=True)):
        assert np.linalg.norm(expected) > 0, index
        assert _relative(actual, expected) < 1e-4, index


@pytest.mark.parametrize("kernel", ("softmax", "exp", "query_polynomial"))
def test_every_kernel_rejects_a_misshapen_forget_gate(kernel):
    """The polynomial's gate is qualified in test_fox_path_attention.py; every kernel takes L [B, T, H]."""
    q, k, v, w, beta, c, _ = _inputs(batch=1, time=8, heads=1, dim=4, value_dim=3, degree=2)
    coefficients = jnp.asarray(c, jnp.float32) if kernel == "query_polynomial" else None
    with pytest.raises(ValueError, match="cumulative log decay"):
        path_attention(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)), kernel=kernel,
                       coefficients=coefficients, log_decay=jnp.zeros(beta.shape[:2]), chunk_size=4)


def test_outputs_and_gradients_are_causal():
    q, k, v, w, beta, c, cotangent = _inputs(batch=1, time=16, heads=2, dim=4, value_dim=3, degree=2, seed=4)
    prefix = 7
    cotangent[:, prefix:] = 0
    operands = [jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta, c)]
    attend = lambda *x: path_attention(*x[:5], kernel="query_polynomial", coefficients=x[5], chunk_size=4)
    out, pull = jax.vjp(attend, *operands)
    changed = [x.at[:, prefix:].add(1.5) for x in operands]
    changed[3] = changed[3] / jnp.linalg.norm(changed[3], axis=-1, keepdims=True)
    changed[4] = jnp.clip(changed[4], 0.01, 1.99)
    out2, pull2 = jax.vjp(attend, *changed)
    np.testing.assert_allclose(out[:, :prefix], out2[:, :prefix], rtol=1e-5, atol=1e-5)
    for a, b in zip(pull(jnp.asarray(cotangent, jnp.float32)), pull2(jnp.asarray(cotangent, jnp.float32)), strict=True):
        np.testing.assert_array_equal(np.asarray(a)[:, prefix:], 0)


def test_nonfinite_coefficient_poisons_only_its_row():
    q, k, v, w, beta, c, _ = _inputs(batch=1, time=8, heads=1, dim=4, value_dim=3, degree=2)
    c = c.copy()
    c[0, 3, 0, -1] = np.inf
    out = path_attention(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, w, beta)), kernel="query_polynomial",
                         coefficients=jnp.asarray(c, jnp.float32), chunk_size=4)
    assert bool(jnp.isnan(out[0, 3]).all())
    assert bool(jnp.isfinite(jnp.delete(out, 3, axis=1)).all())


def test_gates_are_unit_vectors_with_bounded_strengths_and_a_causal_convolution():
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(2, 9, 16)), jnp.float32)
    weights = [jnp.asarray(rng.normal(size=shape), jnp.float32)
               for shape in ((16, 4), (3, 4, 8), (3, 3, 8), (3, 16))] + [jnp.zeros((3, 1))]
    w, beta = path_gates(x, *weights, num_heads=3)
    np.testing.assert_allclose(jnp.linalg.norm(w, axis=-1), 1, rtol=1e-5)
    assert w.shape == (2, 9, 3, 8) and beta.shape == (2, 9, 3)
    assert bool(((beta > 0) & (beta < 2)).all())
    changed, _ = path_gates(x.at[:, 5].add(3.0), *weights, num_heads=3)
    np.testing.assert_array_equal(changed[:, :5], w[:, :5])
    assert not np.allclose(changed[:, 5:8], w[:, 5:8])


def _model(kernel="query_polynomial", **updates):
    settings = dict(vocab_size=16, max_seq_len=32, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                    qk_head_dim=8, value_head_dim=8, intermediate_size=12, attention_backend="blocked",
                    initialization="transformer_reference", qk_norm=True, positional_encoding="path")
    if kernel == "query_polynomial":
        settings.update(attention_degree=2, attention_normalization=False, attention_coefficients="learned",
                        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                        attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.0,
                        attention_coefficient_projection_initial_std=0.25, attention_output_scaling="head_rms_norm")
    elif kernel == "exp":
        settings.update(attention_normalization=False, attention_output_scaling="head_rms_norm")
    return ModelConfig(**{**settings, **updates})


def _config(kernel="query_polynomial", **updates):
    config = Config(model=_model(kernel, **updates))
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 33, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 3, 0
    return config


@pytest.mark.parametrize("kernel", ("softmax", "query_polynomial"))
def test_model_keeps_the_backbone_draws_and_counts_its_parameters(kernel):
    config = _model(kernel)
    model, rope = build_model(config, 3), build_model(replace(config, positional_encoding="rope"), 3)
    params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    rope_params = dict(nnx.to_flat_state(nnx.state(rope, nnx.Param)))
    assert set(params) - set(rope_params) == {(name,) for name in PATH_ARGUMENTS}
    for path, value in rope_params.items():
        np.testing.assert_array_equal(params[path][...], value[...])
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == model_parameter_count(config)
    assert model_parameter_count(config) - model_parameter_count(replace(config, positional_encoding="rope")) == (
        2 * 16 * 32 + 2 * 2 * ((32 + 3) * 8 + 16 + 1))
    assert model.path_w_down.shape == (2, 16, 32) and model.path_w_conv.shape == (2, 2, 3, 8)
    np.testing.assert_array_equal(model.path_beta_bias[...], 0)
    assert 0.01 < float(np.std(model.path_w_down[...])) < 0.03
    tokens = jnp.arange(64).reshape(2, 32) % 16
    assert bool(jnp.isfinite(model(tokens)).all())
    facts = acceptance.matches_softmax_backbone_initialization(Config(model=config))
    assert facts["mechanism_parameters"]["path_w_up"]["shape"] == [2, 2, 32, 8]
    assert facts["mechanism_parameters"]["path_beta_bias"]["weight_decay"] is False


@pytest.mark.parametrize("updates", [
    {"positional_encoding": "alibi"}, {"attention_backend": "mosaic"}, {"attention_backend": "fused"},
    {"attention_coefficient_activation": "identity", "attention_degree": 1,
     "attention_coefficient_amplitude_initial": 1.0, "attention_coefficient_projection_initial_std": 0.0},
    {"path_w_rank": 0}, {"path_conv_size": True},
])
def test_rejects_unsupported_path_configurations(updates):
    with pytest.raises(ValueError):
        _model(**updates)


def test_new_fields_keep_legacy_recipes_and_name_themselves_in_the_identity():
    config = _config()
    rope = _config(); rope.model = replace(rope.model, positional_encoding="rope")
    current, legacy = config_to_dict(rope), config_to_dict(rope)
    for key in ("positional_encoding", "path_w_rank", "path_conv_size"):
        del legacy["model"][key]
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(current, config_to_dict(config))
    identity = attention_identity(config)
    assert identity["positional_encoding"] == "path"
    assert identity["attention_implementation"] == "jax_path_chunked_ut_transform"
    assert "positional_encoding" not in attention_identity(rope)
    paths = acceptance.test_paths(acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=()),
                                  config_to_dict(config), resolved_backend="blocked", cpu=True)
    assert acceptance.PATH_ATTENTION_TESTS in paths


@pytest.mark.parametrize("kernel,forget_gate", [
    ("softmax", False), ("query_polynomial", False), ("exp", False), ("softmax", True), ("exp", True)])
def test_diagnostics_are_valid_and_report_the_gates(kernel, forget_gate):
    config = _config(kernel, attention_forget_gate=forget_gate)
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    layer = report["layers"][0]
    assert 0 < layer["path_beta_min"] <= layer["path_beta_mean"] <= layer["path_beta_max"] < 2
    if kernel == "query_polynomial":
        assert np.asarray(report["coefficients"]["effective_summary"]["mean"]).shape == (2, 2, 3)
    if forget_gate:
        gate = report["coefficients"]["forget_gate"]
        assert gate["finite"] and np.all(np.asarray(gate["log_alpha_sample"]["max"]) <= 0)
        identity = attention_identity(config)
        assert identity["attention_forget_gate"] and "(QK^T / sqrt(d_k) + D) V" in identity["attention_formula"]
        assert identity["attention_score_position_encoding"].startswith("PaTH")


@pytest.mark.parametrize("kernel", ("softmax", "exp"))
def test_a_forget_gate_that_never_decays_changes_nothing_and_trains(kernel):
    tokens = jnp.arange(64).reshape(2, 32) % 16
    plain = build_model(_model(kernel), 5)
    gated = build_model(_model(kernel, attention_forget_gate=True), 5)
    assert np.abs(np.asarray(gated(tokens) - plain(tokens))).max() > 1e-4
    gated.forget_gate_dt_bias[...] = jnp.full_like(gated.forget_gate_dt_bias[...], -200.0)
    np.testing.assert_allclose(gated(tokens), plain(tokens), rtol=1e-6, atol=1e-6)
    config = _config(kernel, attention_forget_gate=True)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(66).reshape(2, 33) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    updated, loss, _ = step(initial, jax.random.PRNGKey(1))
    assert np.isfinite(loss)
    for name in ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias", *PATH_ARGUMENTS):
        assert np.linalg.norm(np.asarray(updated.params[name][...] - initial.params[name][...])) > 0, name


def test_tiny_training_and_checkpoint_resume_preserve_the_gates(tmp_path):
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
        for name in PATH_ARGUMENTS:
            assert not np.array_equal(state.params[name][...], initial.params[name][...]), name
    finally:
        manager.close()
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert records and all(row["attention_diagnostics"]["valid"] for row in records)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ("softmax", "query_polynomial"))
def test_gpu_production_shape_matches_fp32_highest(kernel):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    q, k, v, w, beta, c, cotangent = _inputs(batch=1, time=2048, heads=4, dim=64, value_dim=64, degree=2, seed=7)
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) * 4 for x in (q, k))
    bf16 = [jnp.asarray(x, jnp.bfloat16) for x in (q, k, v)] + [jnp.asarray(x, jnp.float32) for x in (w, beta)]
    coefficients = jnp.asarray(c, jnp.float32) if kernel == "query_polynomial" else None
    attend = jax.jit(partial(path_attention, kernel=kernel, coefficients=coefficients))
    reference = jax.jit(partial(path_attention, kernel=kernel, coefficients=coefficients, precision="highest"))
    actual = attend(*bf16)
    expected = reference(*[x.astype(jnp.float32) for x in bf16])
    assert bool(jnp.isfinite(actual).all())
    assert _relative(actual, expected) < 0.02
