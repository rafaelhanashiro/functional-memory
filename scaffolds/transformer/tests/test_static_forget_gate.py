"""Static polynomial FoX: independent forward/VJP oracle and parameter ownership."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import log_decay
from scaffolds.transformer.attention.multihead import attention_from_config
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state


def _config(**changes):
    values = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2,
                  num_kv_heads=2, qk_head_dim=8, value_head_dim=8, intermediate_size=32,
                  attention_backend="blocked", attention_degree=4, attention_normalization=False,
                  attention_coefficients="learned", attention_coefficient_sharing="per_head",
                  attention_coefficient_conditioning="static", attention_coefficient_scale="factorial",
                  attention_forget_gate=True, positional_encoding="none",
                  attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                  qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                  initialization="transformer_truncated_reference")
    return Config(model=ModelConfig(**{**values, **changes}))


def _inputs(time=19, *, width=4, value_width=3, batch=2, heads=2, sharing="per_head"):
    rng = np.random.default_rng(47)
    q, k = (rng.normal(size=(batch, time, heads, width)) * .5 for _ in range(2))
    v = rng.normal(size=(batch, time, heads, value_width))
    c = rng.normal(size=(heads, 5) if sharing == "per_head" else (5,))
    x = rng.normal(size=(batch, time, 7))
    w = rng.normal(size=(heads, 7)) * .2
    A, bias = rng.uniform(-2, 1, size=(heads, 1)), rng.uniform(-5, -2, size=(heads, 1))
    return (q, k, v, c, x, w, A, bias), rng.normal(size=v.shape)


def _production(q, k, v, c, x, w, A, bias):
    factorials = jnp.asarray([math.factorial(r) for r in range(c.shape[-1])], jnp.float32)
    return attention_core(q, k, v, attention_degree=c.shape[-1] - 1, attention_normalization=False,
                          polynomial_coefficients=c / factorials,
                          query_log_decay=jnp.cumsum(log_decay(x, w, A, bias), axis=1))


def _dense(q, k, v, c, x, w, A, bias):
    """Expanded static polynomial and explicit causal attenuation, independent of production helpers."""
    score = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    ch = jnp.broadcast_to(c, (q.shape[2], c.shape[-1]))
    polynomial = sum(ch[None, :, None, r, None] * score ** r / math.factorial(r)
                     for r in range(c.shape[-1]))
    projected = jnp.einsum("btd,hd->bth", x, w) + bias[:, 0]
    alpha = jnp.exp(-jnp.exp(A[:, 0]) * jnp.logaddexp(0., projected))
    # Explicit products avoid reproducing the implementation's prefix-sum indexing.
    decay = jnp.stack([jnp.stack([jnp.prod(alpha[:, i + 1:t + 1, :], axis=1)
                                 if i <= t else jnp.zeros_like(alpha[:, 0, :])
                                 for i in range(q.shape[1])], axis=-1)
                       for t in range(q.shape[1])], axis=2)
    return jnp.einsum("bhti,bihd->bthd", polynomial * decay, v)


def _numpy_vjp(arrays, cotangent):
    """Host FP64 polynomial/gate algebra and analytic derivatives; no GPU reference compile."""
    q, k, v, c, x, w, A, bias = (np.asarray(a, np.float64) for a in arrays)
    qh, kh, vh, dh = (a.transpose(0, 2, 1, 3) for a in (q, k, v, np.asarray(cotangent, np.float64)))
    projected = np.einsum("btd,hd->bth", x, w) + bias[:, 0]
    softplus, rate = np.logaddexp(0., projected), np.exp(A[:, 0])
    log_alpha = -rate * softplus
    L = np.cumsum(log_alpha, axis=1).transpose(0, 2, 1)
    gap = L[..., :, None] - L[..., None, :]
    causal = np.tri(q.shape[1], dtype=bool)
    decay = np.where(causal, np.exp(np.where(causal, gap, 0.)), 0.)
    score = qh @ kh.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    factorials = np.asarray([math.factorial(r) for r in range(c.shape[-1])])
    ch = np.broadcast_to(c / factorials, (q.shape[2], c.shape[-1]))
    polynomial, slope = np.zeros_like(score), np.zeros_like(score)
    for r in range(c.shape[-1]):
        polynomial += ch[None, :, None, r, None] * score ** r
        if r:
            slope += r * ch[None, :, None, r, None] * score ** (r - 1)
    weights = polynomial * decay
    dweight = dh @ vh.swapaxes(-1, -2)
    dscore = dweight * decay * slope
    dc = np.stack([(dweight * decay * score ** r).sum(axis=(0, 2, 3)) / factorials[r]
                   for r in range(c.shape[-1])], axis=-1)
    if c.ndim == 1:
        dc = dc.sum(axis=0)
    dgap = dweight * weights
    dL = (dgap.sum(axis=-1) - dgap.sum(axis=-2)).transpose(0, 2, 1)
    dlog_alpha = np.cumsum(dL[:, ::-1], axis=1)[:, ::-1]
    dprojected = -dlog_alpha * rate / (1 + np.exp(-projected))
    gradients = ((dscore @ kh / math.sqrt(q.shape[-1])).transpose(0, 2, 1, 3),
                 (dscore.swapaxes(-1, -2) @ qh / math.sqrt(q.shape[-1])).transpose(0, 2, 1, 3),
                 (weights.swapaxes(-1, -2) @ dh).transpose(0, 2, 1, 3), dc,
                 np.einsum("bth,hd->btd", dprojected, w),
                 np.einsum("bth,btd->hd", dprojected, x),
                 (dlog_alpha * log_alpha).sum(axis=(0, 1))[:, None],
                 dprojected.sum(axis=(0, 1))[:, None])
    return (weights @ vh).transpose(0, 2, 1, 3), gradients


def _vjp(function, arrays, cotangent):
    output, pullback = jax.vjp(function, *arrays)
    return output, pullback(cotangent)


def _compare(actual, expected, tolerance):
    for label, observed, wanted in zip(("output", "q", "k", "v", "coefficients", "x", "gate W", "A", "bias"),
                                      (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert observed.shape == wanted.shape and np.isfinite(observed).all(), label
        scale = np.linalg.norm(wanted)
        if scale < 1e-12:
            assert np.linalg.norm(observed) < tolerance, label
        else:
            error = np.linalg.norm(observed - wanted) / scale
            assert error < tolerance, f"{label}: relative error {error} >= {tolerance}"


@pytest.mark.parametrize("sharing", ("global", "per_head"))
def test_numpy_oracle_matches_explicit_products_and_autodiff(sharing):
    arrays, cotangent = _inputs(7, sharing=sharing)
    # Reference computation stays on CPU even during a GPU acceptance stage.
    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(), jax.default_matmul_precision("highest"):
        reference = _vjp(_dense, tuple(jnp.asarray(a, jnp.float64) for a in arrays),
                         jnp.asarray(cotangent, jnp.float64))
    _compare(_numpy_vjp(arrays, cotangent), reference, 2e-12)


@pytest.mark.parametrize("time", (1, 37))
@pytest.mark.parametrize("sharing", ("global", "per_head"))
def test_forward_and_all_gradients_match_dense_reference(time, sharing):
    arrays, cotangent = _inputs(time, sharing=sharing)
    expected = _numpy_vjp(arrays, cotangent)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(_production, a, jnp.asarray(cotangent, jnp.float32)))(
            *(jnp.asarray(a, jnp.float32) for a in arrays))
    _compare(actual, expected, 6e-5)


def test_zero_decay_is_static_attention_and_future_values_are_masked():
    arrays, _ = _inputs(19)
    q, k, v, c = (jnp.asarray(a, jnp.float32) for a in arrays[:4])
    c = c / jnp.asarray([1, 1, 2, 6, 24])
    kwargs = dict(attention_degree=4, attention_normalization=False, polynomial_coefficients=c)
    with jax.default_matmul_precision("highest"):
        ordinary = attention_core(q, k, v, **kwargs)
        gated = attention_core(q, k, v, query_log_decay=jnp.zeros(q.shape[:3]), **kwargs)
        changed = attention_core(q, k, v.at[:, 5:].set(1000.), query_log_decay=jnp.zeros(q.shape[:3]), **kwargs)
    np.testing.assert_allclose(gated, ordinary, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(gated[:, :5], changed[:, :5])


def test_model_owns_static_coefficients_and_applies_factorial_once():
    config = _config()
    model = build_model(config.model, 42)
    params = nnx.state(model, nnx.Param)
    assert "query_polynomial_kernel" not in params and "query_polynomial_bias" not in params
    assert "query_polynomial_amplitude" not in params
    np.testing.assert_array_equal(params["polynomial_coefficients"][...], np.ones((2, 2, 5)))
    assert params["forget_gate_kernel"].shape == (2, 2, 16)
    tokens = jnp.arange(16).reshape(2, 8) % 32
    _, coefficients = model(tokens, return_coefficients=True)
    np.testing.assert_allclose(coefficients, np.broadcast_to(1 / np.asarray([1, 1, 2, 6, 24]), coefficients.shape),
                               rtol=1e-7, atol=0)
    # Changing the representation must not change the gate or the represented polynomial.
    plain = build_model(replace(config.model, attention_coefficient_scale="none"), 42)
    np.testing.assert_allclose(model(tokens), plain(tokens), rtol=2e-5, atol=2e-6)
    model.forget_gate_dt_bias[...] = jnp.full_like(model.forget_gate_dt_bias[...], -200.)
    ungated = build_model(replace(config.model, attention_forget_gate=False), 42)
    np.testing.assert_allclose(model(tokens), ungated(tokens), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("change", ({"attention_backend": "fused"}, {"attention_backend": "mosaic"},
                                    {"attention_normalization": True}, {"positional_encoding": "path"},
                                    {"attention_forget_gate_mode": "query_distance"},
                                    {"attention_forget_gate_form": "constant"},
                                    {"attention_coefficients": "fixed"}))
def test_unqualified_static_gate_configurations_remain_rejected(change):
    with pytest.raises(ValueError):
        _config(**change)


def test_core_rejects_unsupported_combinations():
    arrays, _ = _inputs(3)
    q, k, v, c = (jnp.asarray(a, jnp.float32) for a in arrays[:4])
    arguments = dict(attention_degree=4, attention_normalization=False, polynomial_coefficients=c,
                     query_log_decay=jnp.zeros(q.shape[:3]))
    for change in ({"backend": "fused"}, {"attention_normalization": True},
                   {"query_decay_mode": "query_distance"}, {"polynomial_coefficients": jnp.ones((3, 5))}):
        with pytest.raises(ValueError):
            attention_core(q, k, v, **{**arguments, **change})


def test_acceptance_and_identity_use_static_polynomial_fox():
    config = _config()
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial_unnormalized"
    assert "learned static c" in identity["attention_formula"]
    assert "prod_(j=i+1..t) alpha_j" in identity["attention_formula"]
    assert "causal_exp" not in identity["attention_formula"]
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (True, False):
        assert acceptance.STATIC_FORGET_GATE_TESTS in acceptance.test_paths(
            spec, config_to_dict(config), resolved_backend="blocked", cpu=cpu)
    for encoding in ("none", "rope"):
        attention_from_config(_config(positional_encoding=encoding).model, rngs=nnx.Rngs(0))
    # Even supported Mosaic widths must keep cumulative FoX on its qualified blocked path.
    supported = dict(qk_head_dim=64, value_head_dim=64, max_seq_len=2048, positional_encoding="rope")
    assert not _config(**supported).model.mosaic_supported()
    assert _config(**supported, attention_forget_gate=False).model.mosaic_supported()


def test_diagnostics_report_static_coefficients_and_gate_together():
    config = _config()
    config.data.sequence_length = 9
    config.data.synthetic_vocab_size = 32
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    coefficients = report["coefficients"]
    assert "effective" in coefficients and coefficients["finite"]
    assert coefficients["forget_gate"]["finite"]


@pytest.mark.gpu
def test_full_context_bfloat16_forward_and_all_gradients_match_host_oracle():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    arrays, cotangent = _inputs(2048, width=64, value_width=64, batch=1, heads=2)
    operands = tuple(jnp.asarray(a, jnp.bfloat16 if i in (0, 1, 2, 4) else jnp.float32)
                     for i, a in enumerate(arrays))
    cotangent = jnp.asarray(cotangent, jnp.bfloat16)
    expected = _numpy_vjp(operands, cotangent)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(_production, a, cotangent))(*operands)
    _compare(actual, expected, .018)
