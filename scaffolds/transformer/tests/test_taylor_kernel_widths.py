"""Numerical gates for every approved key-width / fixed-Taylor-degree pair.

The production fused implementation is reused unchanged. Independent references
expand the polynomial as a direct power sum in FP64; full production-shape gates
use the bounded-memory blocked implementation, never a dense T-by-T reference.
"""
from functools import partial
import gc
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.backends.fused_taylor import fused_polynomial_attention


WIDTHS = (8, 16, 32, 64, 128)
DEGREES = (2, 4, 8, 16)
GRID = tuple((width, degree) for width in WIDTHS for degree in DEGREES)
GRID_IDS = tuple(f"dk{width:03d}-m{degree:03d}" for width, degree in GRID)
# Preflight must require every selected test to pass; no GPU skips are accepted.
CPU_CASE_COUNT = len(GRID)
GPU_CASE_COUNT = 5 * len(GRID) + len(WIDTHS)


def _inputs(width, *, batch=2, time=131, heads=2, value_width=17, dtype=jnp.float32, seed=612):
    rng = np.random.default_rng(seed)
    qk_shape = (batch, time, heads, width)
    value_shape = (batch, time, heads, value_width)
    q, k = (jnp.asarray(rng.normal(0, 0.7, qk_shape), dtype=dtype) for _ in range(2))
    v, cotangent = (jnp.asarray(rng.standard_normal(value_shape), dtype=dtype) for _ in range(2))
    return (q, k, v), cotangent


def _dense_reference(q, k, v, *, degree):
    # Scale by the real Q/K width, including width8's internal padding to16.
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    kernel = sum(scores ** order / float(math.factorial(order)) for order in range(degree + 1))
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=jnp.bool_))
    kernel = jnp.where(causal, kernel, 0.0)
    weights = kernel / jnp.sum(kernel, axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


def _forward_vjp(attend, operands, cotangent):
    output, backward = jax.vjp(attend, *operands)
    return output, backward(cotangent)


def _program(attend):
    return jax.jit(partial(_forward_vjp, attend))


def _fp64(operands, cotangent, degree):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _program(partial(_dense_reference, degree=degree))(
            tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64),
        )
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), result)


def _assert_relative(actual, expected, tolerance, label):
    # No absolute floor: tiny cotangents must not permit missing Q/K gradients.
    actual, expected = (np.asarray(x, dtype=np.float64) for x in (actual, expected))
    assert actual.shape == expected.shape, label
    assert np.isfinite(actual).all(), f"{label}: nonfinite observed values"
    assert np.isfinite(expected).all(), f"{label}: nonfinite reference values"
    for batch in range(expected.shape[0]):
        for head in range(expected.shape[2]):
            observed, reference = actual[batch, :, head], expected[batch, :, head]
            norm = np.linalg.norm(reference)
            context = f"{label}, batch={batch}, head={head}"
            if norm == 0:
                np.testing.assert_array_equal(observed, reference, err_msg=context)
                continue
            difference = observed - reference
            relative = np.linalg.norm(difference) / norm
            assert relative <= tolerance, f"{context}: relative L2 {relative} > {tolerance}"
            assert np.max(np.abs(difference)) <= 3 * tolerance * np.max(np.abs(reference)), context


def _assert_result(actual, expected, *, output_tolerance=3e-5, gradient_tolerances=(1e-4,) * 3, label=""):
    _assert_relative(actual[0], expected[0], output_tolerance, f"{label} output")
    for name, observed, reference, tolerance in zip("qkv", actual[1], expected[1], gradient_tolerances, strict=True):
        _assert_relative(observed, reference, tolerance, f"{label} d{name}")


@pytest.fixture
def gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation; preflight must reject this skip")


@pytest.mark.parametrize("width,degree", GRID, ids=GRID_IDS)
def test_cpu_blocked_all_pairs_match_independent_fp64(width, degree):
    operands, cotangent = _inputs(width, time=9, value_width=5)
    actual = _program(partial(blocked_dot_product_attention, attention_degree=degree, query_block_size=4))(
        operands, cotangent,
    )
    _assert_result(actual, _fp64(operands, cotangent, degree), label=f"CPU dk={width}, m={degree}")


@pytest.mark.gpu
@pytest.mark.parametrize("width,degree", GRID, ids=GRID_IDS)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
def test_gpu_all_pairs_fp64_partial_tiles_padding_and_causality(gpu, width, degree, dtype):
    # T131 crosses both32/64 tiles, width8 requires Q/K padding, and V17
    # requires independent value-width padding. All scales use actual dK.
    operands, cotangent = _inputs(width, dtype=dtype)
    attend = partial(fused_polynomial_attention, attention_degree=degree)
    fused = _program(attend)
    blocked = _program(partial(blocked_dot_product_attention, attention_degree=degree, query_block_size=32))
    bf16 = dtype == jnp.bfloat16
    with jax.default_matmul_precision("highest"):
        for scale in (1.0, 1e-5):
            scaled = (cotangent * scale).astype(dtype)
            actual = fused(operands, scaled)
            assert all(x.dtype == dtype for x in (actual[0], *actual[1]))
            _assert_result(actual, _fp64(operands, scaled, degree),
                           output_tolerance=0.006 if bf16 else 3e-5,
                           gradient_tolerances=(0.008 if bf16 else 1e-4,) * 3,
                           label=f"FP64 dk={width}, m={degree}, {dtype}, cotangent={scale}")
            _assert_result(actual, blocked(operands, scaled),
                           output_tolerance=0.006 if bf16 else 3e-5,
                           gradient_tolerances=(0.025, 0.025, 0.015) if bf16 else (1e-4,) * 3,
                           label=f"blocked dk={width}, m={degree}, {dtype}")
    if bf16:
        prefix = 65
        output, grads = fused(operands, cotangent.at[:, prefix:].set(0))
        changed = (operands[0].at[:, prefix:].multiply(-2), operands[1].at[:, prefix:].multiply(3),
                   operands[2].at[:, prefix:].add(5))
        changed_output = jax.jit(attend)(*changed)
        np.testing.assert_array_equal(changed_output[:, :prefix], output[:, :prefix])
        for name, gradient in zip("qkv", grads, strict=True):
            np.testing.assert_array_equal(gradient[:, prefix:], jnp.zeros_like(gradient[:, prefix:]), err_msg=f"future d{name}")


@pytest.mark.gpu
@pytest.mark.parametrize("width,degree", GRID, ids=GRID_IDS)
def test_gpu_all_pairs_negative_score_arithmetic_matches_fp64(gpu, width, degree):
    # Deliberately cover the negative-score cancellation region in Horner
    # evaluation, including m16 near -5, separately from benign random scores.
    scores = np.concatenate(([-5.2, -5.19, -5.04], np.linspace(-20, 20, 128))).astype(np.float32)
    time = len(scores)
    q = jnp.zeros((1, time, 1, width), jnp.float32).at[..., 0].set(math.sqrt(width))
    rng = np.random.default_rng(613)
    k = jnp.asarray(rng.normal(0, 0.2, q.shape), jnp.float32).at[0, :, 0, 0].set(scores)
    v, cotangent = (jnp.asarray(rng.standard_normal((1, time, 1, 17)), jnp.float32) for _ in range(2))
    operands = (q, k, v)
    actual = _program(partial(fused_polynomial_attention, attention_degree=degree))(operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent, degree), output_tolerance=1e-3,
                   gradient_tolerances=(3e-3,) * 3, label=f"negative scores dk={width}, m={degree}")


@pytest.mark.gpu
@pytest.mark.parametrize("width", WIDTHS, ids=lambda width: f"dk{width:03d}")
def test_gpu_m16_broad_bf16_scores_and_small_cotangent(gpu, width):
    operands, cotangent = _inputs(width, batch=1, heads=2, value_width=17, dtype=jnp.bfloat16, seed=614)
    operands = (operands[0] * 5, operands[1] * 5, operands[2] * (2.0 ** 24))
    cotangent = cotangent * (2.0 ** -20)
    host_q, host_k = (np.asarray(x, np.float64) for x in operands[:2])
    scores = np.einsum("bthd,bshd->bhts", host_q, host_k) / math.sqrt(width)
    values = sum(scores ** order / math.factorial(order) for order in range(17))
    causal_values = values[..., np.tril(np.ones((131, 131), dtype=bool))]
    assert np.all(np.isfinite(causal_values) & (causal_values > 0))
    assert causal_values.max() / causal_values.min() > 1e6
    assert causal_values.max() * 131 < np.finfo(np.float32).max
    actual = _program(partial(fused_polynomial_attention, attention_degree=16))(operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent, 16), output_tolerance=0.006,
                   gradient_tolerances=(0.008,) * 3, label=f"m16 broad BF16 scores dk={width}")


@pytest.mark.gpu
@pytest.mark.parametrize("width,degree", GRID, ids=GRID_IDS)
def test_gpu_all_pairs_large_finite_kernel_and_values(gpu, width, degree):
    # Kernels, final normalizers, normalized outputs and gradients are finite;
    # an implementation forming kernel*V before normalization would overflow.
    # A transverse key coordinate ensures Q gradients remain substantial even
    # when a large-score polynomial is nearly a homogeneous monomial.
    dtype = jnp.bfloat16
    scale = {2: 1e10, 4: 1e5, 8: 1e3, 16: 1e2}[degree]
    q = jnp.zeros((1, 3, 1, width), dtype).at[..., 0].set(math.sqrt(width))
    k = jnp.zeros_like(q).at[0, :, 0, :2].set(
        jnp.asarray([[0.5, -0.25], [0.75, 0.5], [1.0, -0.75]], dtype) * scale,
    )
    v = (jnp.asarray([[1.0, -2.0], [-3.0, 2.0], [2.0, 1.0]], dtype) * 1e25)[None, :, None, :]
    cotangent = jnp.asarray([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.7]], dtype)[None, :, None, :]
    operands = (q, k, v)
    expected = _fp64(operands, cotangent, degree)
    assert np.linalg.norm(expected[1][0][..., 1]) >= 0.1 * np.linalg.norm(expected[1][0])
    actual = _program(partial(fused_polynomial_attention, attention_degree=degree))(operands, cotangent)
    _assert_result(actual, expected, output_tolerance=0.006, gradient_tolerances=(0.008,) * 3,
                   label=f"large finite kernel/V dk={width}, m={degree}")


@pytest.fixture
def release_full_shape_executables():
    yield
    # Twenty distinct large compiled programs should not retain caches across
    # cells; each reference still runs at the real full per-device shape.
    jax.clear_caches()
    gc.collect()


@pytest.mark.gpu
@pytest.mark.parametrize("width,degree", GRID, ids=GRID_IDS)
def test_gpu_all_pairs_full_production_shape_bf16(gpu, release_full_shape_executables, width, degree):
    operands, cotangent = _inputs(width, batch=16, time=2048, heads=16, value_width=64,
                                  dtype=jnp.bfloat16, seed=615)
    attend = partial(fused_polynomial_attention, attention_degree=degree)
    actual = _program(attend)(operands, cotangent)
    # Blocked attention recomputes 128-query score blocks in backward instead
    # of allocating a full [16,16,2048,2048] score/gradient tensor.
    reference = _program(partial(blocked_dot_product_attention, attention_degree=degree))(operands, cotangent)
    _assert_result(actual, reference, output_tolerance=0.006,
                   gradient_tolerances=(0.025, 0.025, 0.015), label=f"full BF16 dk={width}, m={degree}")
    assert all(x.dtype == jnp.bfloat16 for x in (actual[0], *actual[1]))
    # Independently anchor a prefix of these exact BF16 operands to FP64.
    prefix = tuple(x[:1, :129, :1] for x in operands)
    prefix_cotangent = cotangent[:1, :129, :1]
    prefix_actual = _program(attend)(prefix, prefix_cotangent)
    _assert_result(prefix_actual, _fp64(prefix, prefix_cotangent, degree), output_tolerance=0.006,
                   gradient_tolerances=(0.008,) * 3, label=f"production prefix FP64 dk={width}, m={degree}")
    _assert_relative(actual[0][:1, :129, :1], prefix_actual[0], 0.006, "causal prefix across shapes")
