"""Fixed signed Taylor kernels: independent FP64 reference and CUDA gates.

The CPU tests interpret the real Pallas programs with interpreter-compatible
loads and FP32 dots. GPU tests compile the production Triton implementation.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.models.components import fused_attention as fused
from scaffolds.transformer.models.components.attention import blocked_dot_product_attention
from scaffolds.transformer.tests.test_fused_attention import (
    _assert_result, _compiled_forward_and_vjp, _fp64_forward_and_vjp, _inputs,
)


DEGREES = (1, 3)
SIGNED_CASES = ("zero_partial", "negative_partial", "negative_final", "cancellation")


def _interpreted(monkeypatch, degree):
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST,
    ))
    def load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)
    monkeypatch.setattr(fused, "_load", load)
    original = fused.pl.pallas_call
    monkeypatch.setattr(fused.pl, "pallas_call", partial(original, interpret=True))
    return partial(fused.fused_polynomial_attention, attention_degree=degree)


def _signed_inputs(degree, case):
    # Only the last query has signed weights, so preceding causal rows remain
    # valid even when the last query has zero intermediate or final sums.
    time = (130 if degree == 1 else 129) if case == "zero_final" else 195
    scores = np.zeros(time, np.float32)
    if case == "zero_partial":
        if degree == 1:
            scores[32:64] = -2  # +32 - 32 = 0 after the first key tile.
        else:
            scores[64:128] = -3  # +64 - 128 + 64 = 0 after three tiles.
    elif case == "negative_partial":
        scores[:64] = -2 if degree == 1 else -3
    elif case == "negative_final":
        scores = np.linspace(-4, -3, time, dtype=np.float32)
    elif case in ("zero_final", "cancellation"):
        if degree == 1:
            if time % 2:
                scores[-1] = -1  # one exactly zero kernel entry
            scores[:time // 2] = -2
        else:
            scores[:time // 3] = -3  # p_3(-3)=-2 exactly; p_3(0)=1.
        if case == "cancellation":
            scores[time // 2] += 0.125
    elif case == "zero_kernel" and degree == 1:
        scores[:32] = -1
    else:
        raise ValueError((degree, case))
    q = jnp.zeros((1, time, 1, 1), jnp.float32).at[:, -1].set(1)
    k = jnp.asarray(scores).reshape(q.shape)
    rng = np.random.default_rng(814)
    v, cotangent = (jnp.asarray(rng.standard_normal(q.shape), jnp.float32) for _ in range(2))
    return (q, k, v), cotangent


def _check(attend, operands, cotangent, degree, *, stress=False, bf16=False):
    actual = _compiled_forward_and_vjp(attend)(operands, cotangent)
    expected = _fp64_forward_and_vjp(operands, cotangent, degree)
    _assert_result(actual, expected,
                   output_rtol=0.006 if bf16 else 1e-3 if stress else 4e-5,
                   gradient_rtols=((0.015,) * 3 if bf16 else (3e-3,) * 3 if stress else (3e-4,) * 3),
                   label=f"odd degree={degree}, independent FP64")
    return actual


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("time", (5, 131))
def test_cpu_blocked_and_interpreted_outputs_qkv_gradients_and_padding(monkeypatch, degree, time):
    operands, cotangent = _inputs((2, time, 3, 7, 5), seed=812)
    for attend in (partial(blocked_dot_product_attention, attention_degree=degree, query_block_size=32),
                   _interpreted(monkeypatch, degree)):
        _check(attend, operands, cotangent, degree)


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("case", SIGNED_CASES)
def test_cpu_signed_sums_outputs_gradients_and_diagnostics(monkeypatch, degree, case):
    operands, cotangent = _signed_inputs(degree, case)
    _check(_interpreted(monkeypatch, degree), operands, cotangent, degree, stress=case == "cancellation")
    attend = partial(blocked_dot_product_attention, attention_degree=degree, query_block_size=32)
    _check(attend, operands, cotangent, degree, stress=case == "cancellation")
    _, diagnostics = blocked_dot_product_attention(*operands, attention_degree=degree, return_diagnostics=True)
    assert int(diagnostics["invalid_kernel_count"]) == 0
    assert int(diagnostics["invalid_normalizer_count"]) == 0
    assert int(diagnostics["invalid_output_count"]) == 0
    assert float(diagnostics["negative_kernel_fraction"]) > 0
    assert float(diagnostics["min_abs_normalizer"]) > 0
    if case == "negative_final":
        assert float(diagnostics["normalizer_min"]) < 0
        # Every final-row kernel is negative, and division by its negative
        # sum produces positive normalized weights.
        assert int(diagnostics["negative_weight_count"]) == 0
    elif case == "cancellation":
        assert float(diagnostics["cancellation_ratio_max"]) > 1000
        assert float(diagnostics["negative_weight_fraction"]) > 0


@pytest.mark.parametrize("degree", DEGREES)
def test_cpu_zero_final_sum_fails_explicitly(monkeypatch, degree):
    operands, _ = _signed_inputs(degree, "zero_final")
    output = jax.jit(_interpreted(monkeypatch, degree))(*operands)
    reference, diagnostics = blocked_dot_product_attention(*operands, attention_degree=degree, return_diagnostics=True)
    for actual in (output, reference):
        assert bool(jnp.all(jnp.isfinite(actual[:, :-1])))
        assert bool(jnp.all(jnp.isnan(actual[:, -1])))
    assert int(diagnostics["invalid_normalizer_count"]) == 1
    assert float(diagnostics["min_abs_normalizer"]) == 0


def test_cpu_zero_individual_kernel_is_valid(monkeypatch):
    operands, cotangent = _signed_inputs(1, "zero_kernel")
    _check(_interpreted(monkeypatch, 1), operands, cotangent, 1)
    _, diagnostics = blocked_dot_product_attention(*operands, attention_degree=1, return_diagnostics=True)
    assert int(diagnostics["invalid_kernel_count"]) == 0
    assert float(diagnostics["kernel_min"]) == 0


@pytest.fixture
def require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_gpu_independent_fp64_outputs_qkv_gradients_and_partial_tiles(require_gpu, degree, dtype):
    operands, cotangent = _inputs((2, 131, 3, 33, 17), dtype=dtype, seed=813)
    _check(partial(fused.fused_polynomial_attention, attention_degree=degree), operands, cotangent, degree,
           bf16=dtype == jnp.bfloat16)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("case", SIGNED_CASES)
def test_gpu_signed_sums_outputs_and_qkv_gradients(require_gpu, degree, case):
    operands, cotangent = _signed_inputs(degree, case)
    _check(partial(fused.fused_polynomial_attention, attention_degree=degree), operands, cotangent, degree,
           stress=case == "cancellation")


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_zero_final_sum_fails_explicitly(require_gpu, degree):
    operands, _ = _signed_inputs(degree, "zero_final")
    actual = jax.jit(partial(fused.fused_polynomial_attention, attention_degree=degree))(*operands)
    assert bool(jnp.all(jnp.isfinite(actual[:, :-1])))
    assert bool(jnp.all(jnp.isnan(actual[:, -1])))


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_causality_and_nonfinite_masking(require_gpu, degree):
    operands, cotangent = _inputs((1, 131, 2, 7, 5), dtype=jnp.bfloat16, seed=815)
    attend = partial(fused.fused_polynomial_attention, attention_degree=degree)
    prefix = 65
    actual = _compiled_forward_and_vjp(attend)(operands, cotangent.at[:, prefix:].set(0))
    changed = (operands[0].at[:, prefix:].multiply(-2), operands[1].at[:, prefix:].multiply(3),
               operands[2].at[:, prefix:].add(5))
    changed_output = jax.jit(attend)(*changed)
    np.testing.assert_array_equal(changed_output[:, :prefix], actual[0][:, :prefix])
    for grad in actual[1]:
        np.testing.assert_array_equal(grad[:, prefix:], jnp.zeros_like(grad[:, prefix:]))
    q = jnp.asarray([1., 0.], jnp.float32).reshape(1, 2, 1, 1)
    k = jnp.asarray([0., 1e30], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3., 7.], jnp.float32).reshape(q.shape)
    np.testing.assert_array_equal(jax.jit(attend)(q, k, v).reshape(-1), [3., 5.])
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    k = jnp.asarray([0., 0., jnp.inf], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3., 7., 11.], jnp.float32).reshape(q.shape)
    output = jax.jit(attend)(q, k, v)
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [3., 5.])
    assert bool(jnp.all(jnp.isnan(output[:, -1])))


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_full_production_shape_bf16_matches_blocked(require_gpu, degree):
    # Complete per-device microbatch/context/head dimensions of the 4-H100 run.
    operands, cotangent = _inputs((16, 2048, 16, 64, 64), dtype=jnp.bfloat16, seed=816)
    # Use well-conditioned random scores for a strict arithmetic comparison;
    # the separate stress cases test signed cancellation directly.
    operands = (operands[0] * jnp.bfloat16(0.5), operands[1] * jnp.bfloat16(0.5), operands[2])
    attend = partial(fused.fused_polynomial_attention, attention_degree=degree)
    actual = _compiled_forward_and_vjp(attend)(operands, cotangent)
    reference = _compiled_forward_and_vjp(partial(blocked_dot_product_attention, attention_degree=degree))(
        operands, cotangent,
    )
    _assert_result(actual, reference, output_rtol=0.006, gradient_rtols=(0.03, 0.03, 0.015),
                   label=f"full production shape degree={degree}")
    prefix = tuple(x[:1, :129, :1] for x in operands)
    _check(attend, prefix, cotangent[:1, :129, :1], degree, bf16=True)
