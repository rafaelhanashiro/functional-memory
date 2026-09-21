"""GPU gates for the tiled exp attention forward and explicit custom VJP."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.unnormalized_attention.models.components.attention import blocked_dot_product_attention
from experiments.unnormalized_attention.models.components.fused_attention import fused_unnormalized_attention
from experiments.unnormalized_attention.tests.test_attention import _dense_reference


pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation; the compiled fused kernel is under test")


def _inputs(shape, *, dtype=jnp.float32, seed=91):
    batch, time, heads, qk_dim, value_dim = shape
    rng = np.random.default_rng(seed)
    qk_shape = (batch, time, heads, qk_dim)
    value_shape = (batch, time, heads, value_dim)
    q, k = (jnp.asarray(0.7 * rng.standard_normal(qk_shape), dtype=dtype) for _ in range(2))
    v = jnp.asarray(rng.standard_normal(value_shape), dtype=dtype)
    cotangent = jnp.asarray(rng.standard_normal(value_shape), dtype=dtype)
    return (q, k, v), cotangent


def _forward_and_vjp(attend, operands, cotangent):
    output, pullback = jax.vjp(attend, *operands)
    return output, pullback(cotangent)


def _compiled_forward_and_vjp(attend):
    return jax.jit(partial(_forward_and_vjp, attend))


def _fp64_forward_and_vjp(operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        operands64 = tuple(x.astype(jnp.float64) for x in operands)
        output, grads = _compiled_forward_and_vjp(_dense_reference)(operands64, cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), (output, grads))


def _assert_relative(observed, expected, *, rtol, name):
    """Per-head relative L2/max errors, so tiny or missing VJPs cannot pass."""
    actual = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    assert actual.shape == reference.shape, name
    assert np.isfinite(actual).all(), f"{name}: nonfinite actual values"
    assert np.isfinite(reference).all(), f"{name}: nonfinite reference values"
    for batch in range(reference.shape[0]):
        for head in range(reference.shape[2]):
            got = actual[batch, :, head, :]
            want = reference[batch, :, head, :]
            norm = np.linalg.norm(want)
            label = f"{name}, batch={batch}, head={head}"
            if norm == 0:
                np.testing.assert_array_equal(got, want, err_msg=label)
                continue
            error = np.linalg.norm(got - want) / norm
            assert error <= rtol, f"{label}: relative L2 error {error:.6g} exceeds {rtol}"
            max_error = np.max(np.abs(got - want))
            max_limit = 3 * rtol * np.max(np.abs(want))
            assert max_error <= max_limit, f"{label}: max error {max_error:.6g} exceeds {max_limit:.6g}"


def _assert_result(actual, expected, *, output_rtol, gradient_rtols, label):
    _assert_relative(actual[0], expected[0], rtol=output_rtol, name=f"{label} output")
    for index, (observed, reference, tolerance) in enumerate(zip(actual[1], expected[1], gradient_rtols, strict=True)):
        _assert_relative(observed, reference, rtol=tolerance, name=f"{label} d{'qkv'[index]}")


@pytest.mark.parametrize("shape", [(2, 17, 3, 7, 5), (1, 131, 2, 33, 65)])
def test_fp32_outputs_and_vjp_match_blocked_and_independent_fp64(shape):
    # Odd widths exercise padding; 131 crosses 32/64/128 query/key boundaries.
    operands, cotangent = _inputs(shape)
    fused = _compiled_forward_and_vjp(fused_unnormalized_attention)
    blocked = _compiled_forward_and_vjp(partial(blocked_dot_product_attention, query_block_size=32))
    with jax.default_matmul_precision("highest"):
        for scale in (1.0, 1e-5):
            scaled_cotangent = cotangent * scale
            actual = fused(operands, scaled_cotangent)
            assert actual[0].dtype == jnp.float32
            assert all(grad.dtype == jnp.float32 for grad in actual[1])
            reference = blocked(operands, scaled_cotangent)
            dense64 = _fp64_forward_and_vjp(operands, scaled_cotangent)
            for label, expected in (("blocked", reference), ("dense FP64", dense64)):
                _assert_result(actual, expected, output_rtol=3e-5, gradient_rtols=(1e-4,) * 3,
                               label=f"{label}, cotangent scale={scale}")


def test_bf16_production_context_outputs_and_vjp_match_blocked():
    operands, cotangent = _inputs((1, 2048, 4, 64, 64), dtype=jnp.bfloat16, seed=92)
    fused = _compiled_forward_and_vjp(fused_unnormalized_attention)
    blocked = _compiled_forward_and_vjp(blocked_dot_product_attention)
    actual = fused(operands, cotangent)
    reference = blocked(operands, cotangent)
    assert actual[0].dtype == jnp.bfloat16
    assert all(grad.dtype == jnp.bfloat16 for grad in actual[1])
    _assert_result(actual, reference, output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015),
                   label="BF16 production context")
    prefix = tuple(x[:, :129, :1] for x in operands)
    prefix_cotangent = cotangent[:, :129, :1]
    prefix_actual = fused(prefix, prefix_cotangent)
    dense64 = _fp64_forward_and_vjp(prefix, prefix_cotangent)
    _assert_result(prefix_actual, dense64, output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015),
                   label="BF16 prefix vs dense FP64")
    _assert_relative(actual[0][:, :129, :1], prefix_actual[0], rtol=0.006, name="causal prefix across shapes")


def test_causal_outputs_and_vjp_ignore_future_tokens():
    operands, cotangent = _inputs((1, 131, 2, 33, 5), dtype=jnp.bfloat16, seed=93)
    prefix = 65
    fused = _compiled_forward_and_vjp(fused_unnormalized_attention)
    restricted_cotangent = cotangent.at[:, prefix:].set(0)
    output, grads = fused(operands, restricted_cotangent)
    q, k, v = operands
    changed = (q.at[:, prefix:].multiply(-2), k.at[:, prefix:].multiply(3), v.at[:, prefix:].add(2))
    changed_output, _ = fused(changed, restricted_cotangent)
    np.testing.assert_array_equal(changed_output[:, :prefix], output[:, :prefix])
    for name, grad in zip("qkv", grads, strict=True):
        assert bool(jnp.all(jnp.isfinite(grad))), name
        np.testing.assert_array_equal(grad[:, prefix:], jnp.zeros_like(grad[:, prefix:]), err_msg=f"future d{name}")


def test_zero_scores_are_prefix_sum_with_reverse_prefix_sum_value_vjp():
    (q, k, v), cotangent = _inputs((2, 131, 3, 7, 5))
    q = jnp.zeros_like(q)
    output, grads = _compiled_forward_and_vjp(fused_unnormalized_attention)((q, k, v), cotangent)
    _assert_relative(output, jnp.cumsum(v, axis=1), rtol=2e-6, name="zero-score prefix sum")
    expected_dv = jnp.flip(jnp.cumsum(jnp.flip(cotangent, axis=1), axis=1), axis=1)
    _assert_relative(grads[2], expected_dv, rtol=2e-6, name="zero-score reverse prefix sum dV")
    np.testing.assert_array_equal(grads[1], jnp.zeros_like(k))
    assert np.linalg.norm(np.asarray(grads[0])) > 0


def test_row_score_offset_multiplies_output():
    q = jnp.ones((1, 131, 1, 1), dtype=jnp.float32)
    k = jnp.linspace(-2, 2, 131).reshape(q.shape)
    v = jnp.sin(jnp.arange(131, dtype=jnp.float32)).reshape(q.shape)
    attend = jax.jit(fused_unnormalized_attention)
    original, shifted = attend(q, k, v), attend(q, k + 3, v)
    _assert_relative(shifted, original * math.exp(3), rtol=3e-5, name="row-shift scale")


def test_masked_future_overflow_does_not_poison_valid_rows():
    q = jnp.asarray([1.0, 0.0], dtype=jnp.float32).reshape(1, 2, 1, 1)
    k = jnp.asarray([0.0, 1e30], dtype=jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0], dtype=jnp.float32).reshape(q.shape)
    output = jax.jit(fused_unnormalized_attention)(q, k, v)
    np.testing.assert_array_equal(output.reshape(-1), [3.0, 10.0])


@pytest.mark.parametrize("bad_score", [100.0, np.inf, -np.inf, np.nan])
def test_invalid_causal_scores_or_kernel_poison_only_affected_rows(bad_score):
    q = jnp.ones((1, 3, 1, 1), dtype=jnp.float32)
    k = jnp.asarray([0.0, 0.0, bad_score], dtype=jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0, 11.0], dtype=jnp.float32).reshape(q.shape)
    output = jax.jit(fused_unnormalized_attention)(q, k, v)
    reference = blocked_dot_product_attention(q, k, v)
    np.testing.assert_array_equal(jnp.isnan(output), jnp.isnan(reference))
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [3.0, 10.0])
    assert bool(jnp.all(jnp.isnan(output[:, 2:])))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_exponential_underflow_is_allowed(dtype):
    q = jnp.ones((1, 131, 1, 1), dtype=dtype)
    output = jax.jit(fused_unnormalized_attention)(q, q * -1000, q)
    np.testing.assert_array_equal(output, jnp.zeros_like(output))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_weight_sum_overflow_is_not_a_denominator(dtype):
    q = jnp.ones((1, 4, 1, 1), dtype=dtype)
    output = jax.jit(fused_unnormalized_attention)(q, q * 88, q * 1e-20)
    dense64 = _fp64_forward_and_vjp((q, q * 88, q * 1e-20), q)[0]
    _assert_relative(output, dense64, rtol=0.006 if dtype == jnp.bfloat16 else 3e-5,
                     name="finite output despite overflowing weight sum")


def test_output_overflow_is_visible():
    q = jnp.ones((1, 2, 1, 1), dtype=jnp.float32)
    output = jax.jit(fused_unnormalized_attention)(q, q * 80, q * 1e10)
    assert not bool(jnp.any(jnp.isfinite(output)))


def test_bf16_broad_scores_and_small_vjp_match_fp64():
    (q, k, v), cotangent = _inputs((1, 131, 2, 32, 17), dtype=jnp.bfloat16, seed=184)
    operands = (q * 3, k * 3, v * (2.0 ** 12))
    cotangent = cotangent * (2.0 ** -12)
    actual = _compiled_forward_and_vjp(fused_unnormalized_attention)(operands, cotangent)
    dense64 = _fp64_forward_and_vjp(operands, cotangent)
    _assert_result(actual, dense64, output_rtol=0.006, gradient_rtols=(0.006,) * 3,
                   label="BF16 broad scores vs FP64")


@pytest.mark.parametrize("normalized", [False, True])
def test_model_fused_dispatch_retains_real_output_when_collecting_diagnostics(monkeypatch, normalized):
    from flax import nnx
    from experiments.unnormalized_attention.models.components import attention as attention_module
    from experiments.unnormalized_attention.models.components import fused_attention as fused_module

    settings = dict(hidden_size=16, num_heads=2, num_kv_heads=2, qk_head_dim=4,
                    value_head_dim=6, attention_normalization=normalized)
    layer = attention_module.MultiHeadAttention(**settings, backend="fused", rngs=nnx.Rngs(19))
    calls = []
    captured_operands = []

    def fast(q, k, v, **kwargs):
        calls.append(("fast", kwargs))
        captured_operands.append((q, k, v))
        return jnp.full_like(v, 3)

    def diagnostics(q, k, v, **kwargs):
        calls.append(("diagnostics", kwargs))
        for actual, expected in zip((q, k, v), captured_operands[-1], strict=True):
            np.testing.assert_array_equal(actual, expected)
        return jnp.full_like(v, -11), {"weight_sum_min": jnp.asarray(7.0)}

    monkeypatch.setattr(fused_module, "fused_unnormalized_attention", fast)
    monkeypatch.setattr(attention_module, "padded_dot_product_attention", fast)
    monkeypatch.setattr(attention_module, "blocked_dot_product_attention", diagnostics)
    x = jnp.arange(3 * 16, dtype=jnp.float32).reshape(1, 3, 16) / 48
    expected = layer.o_proj(jnp.full((1, 3, 12), 3, dtype=jnp.float32))
    plain = layer(x)
    output, observed_diagnostics = layer(x, return_diagnostics=True)
    assert [kind for kind, _ in calls] == ["fast", "fast", "diagnostics"]
    expected_fast_kwargs = {"backend": "cudnn"} if normalized else {}
    assert calls[0][1] == calls[1][1] == expected_fast_kwargs
    assert calls[2][1] == {"attention_normalization": normalized, "return_diagnostics": True}
    np.testing.assert_array_equal(plain, expected)
    np.testing.assert_array_equal(output, expected)
    assert float(observed_diagnostics["weight_sum_min"]) == 7.0


def test_normalized_control_matches_cudnn():
    from experiments.unnormalized_attention.models.components.attention import padded_dot_product_attention

    operands, cotangent = _inputs((1, 256, 4, 64, 64), dtype=jnp.bfloat16, seed=13)
    blocked = _compiled_forward_and_vjp(partial(blocked_dot_product_attention, attention_normalization=True))
    cudnn = _compiled_forward_and_vjp(partial(padded_dot_product_attention, backend="cudnn"))
    _assert_result(blocked(operands, cotangent), cudnn(operands, cotangent),
                   output_rtol=0.006, gradient_rtols=(0.03, 0.03, 0.02), label="normalized cuDNN control")
