"""Full-context GPU parity with Q/K RMSNorm and the 14-head reference geometry.

The mechanism kernels consume prepared Q/K; this gate checks their composition with
learned Q/K gains, RoPE, and output RMSNorm, including every input/gain gradient.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.attention.qk_norm import rms_score_operands, scale_rms_queries
from scaffolds.transformer.attention.positional import apply_rope


CASES = [("softmax", 0, "cudnn"), ("exp", 0, "fused")] + [
    (family, degree, "fused" if family == "query_key" else "mosaic")
    for family in ("fixed", "static", "query", "query_key") for degree in (2, 4, 8, 12)
]


def _inputs(family, degree, *, head_scale=False):
    rng = np.random.default_rng(1605 + degree)
    shape = (1, 2048, 14, 64)
    operands = [jnp.asarray(rng.normal(size=shape), jnp.bfloat16) for _ in range(3)]
    # Distinct nonuniform gains expose broadcasting and normalization-order errors.
    operands += [jnp.asarray(0.9 + 0.2 * rng.random(64), jnp.float32) for _ in range(2)]
    if head_scale:
        # Nonunit, distinct scales expose missing head gradients or broadcasting.
        operands.append(jnp.log(jnp.linspace(0.7, 1.3, 14, dtype=jnp.float32))[:, None])
    if family != "softmax":
        operands.append(jnp.asarray(0.9 + 0.2 * rng.random((14, 64)), jnp.float32))
    if family in ("static", "query", "query_key"):
        coefficient_shape = (14, degree + 1) if family == "static" else (*shape[:3], degree + 1)
        # Stored whole coefficients are order-one; the operation divides once by r!.
        operands.append(jnp.asarray(1.0 + 0.3 * np.tanh(rng.normal(size=coefficient_shape)), jnp.float32))
    if family == "query_key":
        operands.append(jnp.asarray(1.0 + 0.2 * np.tanh(rng.normal(size=(*shape[:3], degree + 1))), jnp.float32))
    return tuple(operands), jnp.asarray(rng.normal(size=shape), jnp.bfloat16)


def _operation(family, degree, backend, *operands, head_scale=False):
    q, k, v, query_gain, key_gain, *remaining = operands
    q, k = rms_score_operands(q, k, query_gain, key_gain, eps=1e-6)
    if head_scale:
        q = scale_rms_queries(q, remaining.pop(0))
    positions = jnp.arange(q.shape[1])
    q, k = (apply_rope(x, positions, 10000.0).astype(jnp.bfloat16) for x in (q, k))
    gain = remaining.pop(0) if family != "softmax" else None
    kwargs = {}
    if family in ("static", "query", "query_key"):
        factorials = jnp.asarray([math.factorial(r) for r in range(degree + 1)], jnp.float32)
        kwargs["polynomial_coefficients" if family == "static" else "query_polynomial_coefficients"] = (
            remaining.pop(0) / factorials)
    if family == "query_key":
        kwargs["key_polynomial_factors"] = remaining.pop(0)
    output = attention_core(q, k, v, backend=backend, attention_degree=degree or None,
                            attention_normalization=family == "softmax",
                            attention_coefficient_activation="tanh", **kwargs)
    return scale_attention_output(output, "none" if gain is None else "head_rms_norm", output_norm_scale=gain)


def _vjp(operation, operands, cotangent):
    output, pullback = jax.vjp(operation, *operands)
    return output, pullback(cotangent)


def _smooth_head_scale_reference(family, *operands):
    """Independent dense exp/Taylor math, preserving FP32 or FP64 throughout.

    Production helpers explicitly cast their RMS statistics to FP32 and their
    operands to BF16. This oracle instead differentiates the smooth attention
    formula and does not share those implementation or rounding decisions.
    """
    q, k, v, query_gain, key_gain, gamma, output_gain = operands
    q = q / jnp.sqrt(jnp.mean(q * q, axis=-1, keepdims=True) + 1e-6) * query_gain
    k = k / jnp.sqrt(jnp.mean(k * k, axis=-1, keepdims=True) + 1e-6) * key_gain
    q = q * jnp.exp(gamma)[None, None]
    dimension = q.shape[-1]
    frequencies = jnp.arange(q.shape[1], dtype=q.dtype)[:, None] / (
        10000 ** (2 * jnp.arange(dimension // 2, dtype=q.dtype) / dimension))[None]
    cos, sin = jnp.cos(frequencies)[None, :, None], jnp.sin(frequencies)[None, :, None]

    def rotate(x):
        first, second = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)

    scores = jnp.einsum("bthd,bshd->bhts", rotate(q), rotate(k)) / math.sqrt(dimension)
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(k.shape[1])[None, :]
    if family == "exp":
        weights = jnp.exp(jnp.where(causal, scores, -jnp.inf))
    elif family == "fixed":
        weights = jnp.where(causal, sum(scores ** r / math.factorial(r) for r in range(5)), 0)
    else:
        raise ValueError(f"Unsupported head-scale control: {family}")
    output = jnp.einsum("bhts,bshd->bthd", weights, v)
    return output / jnp.sqrt(jnp.mean(output * output, axis=-1, keepdims=True) + 1e-6) * output_gain


def _assert_relative(actual, expected, label, tolerance=0.04):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert np.isfinite(actual).all(), label
    norm = np.linalg.norm(expected.ravel())
    assert norm > 1e-12, f"{label} oracle gradient is vacuous"
    error = np.linalg.norm((actual - expected).ravel()) / norm
    assert error <= tolerance, f"{label}: relative L2 error {error:.6g} > {tolerance}"


def _check_full_context(family, degree, backend, *, head_scale=False):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    if backend == "mosaic" and jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Mosaic requires H100 or B300")
    operands, cotangent = _inputs(family, degree, head_scale=head_scale)
    operation = partial(_operation, family, degree, backend, head_scale=head_scale)
    actual = jax.jit(partial(_vjp, operation))(operands, cotangent)
    with jax.default_matmul_precision("highest"):
        reference = partial(_operation, family, degree, "blocked", head_scale=head_scale)
        expected = jax.jit(partial(_vjp, reference))(operands, cotangent)
    _assert_relative(actual[0], expected[0], "output")
    names = ["q", "k", "v", "q_gain", "k_gain"]
    if head_scale:
        names.append("head_log_scale")
    if family != "softmax":
        names.append("output_gain")
    if family in ("static", "query", "query_key"):
        names.append("stored_coefficients")
    if family == "query_key":
        names.append("key_factors")
    for name, got, want in zip(names, actual[1], expected[1], strict=True):
        if name == "head_log_scale":
            # Random scalar contractions can nearly cancel even when the local
            # BF16 VJPs agree. Keep their aggregate check and independently probe
            # every head with the well-conditioned cotangent below.
            _assert_relative(got, want, f"d{name}")
        elif name in ("stored_coefficients", "key_factors"):
            # Orderwise checks prevent a correct constant gradient from hiding a
            # missing or incorrectly factorial-scaled high-order gradient.
            for order in range(degree + 1):
                _assert_relative(got[..., order], want[..., order], f"d{name}[r={order}]", tolerance=0.05)
        else:
            _assert_relative(got, want, f"d{name}")
    assert actual[0].dtype == jnp.bfloat16


@pytest.mark.gpu
@pytest.mark.parametrize(("family", "degree", "backend"), CASES,
                         ids=[f"{family}-{degree}-{backend}" for family, degree, backend in CASES])
def test_full_context_qk_rms_backend(family, degree, backend):
    _check_full_context(family, degree, backend)


@pytest.mark.gpu
@pytest.mark.parametrize(("family", "degree", "backend"), [("exp", 0, "fused"), ("fixed", 4, "mosaic")],
                         ids=["exp-0-fused", "fixed-4-mosaic"])
def test_full_context_qk_rms_head_scale_backend(family, degree, backend):
    """Fixed controls retain QK/output gains and differentiate each head's scalar."""
    _check_full_context(family, degree, backend, head_scale=True)
    operands, _ = _inputs(family, degree, head_scale=True)
    reference_operands = tuple(x.astype(jnp.float32) for x in operands)

    def reference(gamma):
        return _smooth_head_scale_reference(family, *reference_operands[:5], gamma, reference_operands[6])

    with jax.default_matmul_precision("highest"):
        tangent = jax.jit(lambda gamma: jax.jvp(reference, (gamma,), (jnp.ones_like(gamma),))[1])(
            reference_operands[5])
    cotangent = tangent.astype(jnp.bfloat16)
    # Heads are independent. For each head, cotangent=dY/dgamma makes the
    # scalar VJP its squared tangent norm (up to the cotangent's BF16 rounding),
    # so missing gradients cannot hide behind a cancellation-dominated target.
    expected = jnp.sum(tangent * cotangent.astype(jnp.float32), axis=(0, 1, 3))[:, None]

    def loss(gamma):
        output = _operation(family, degree, backend, *operands[:5], gamma, operands[6], head_scale=True)
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual = jax.jit(jax.grad(loss))(operands[5])
    assert np.all(np.isfinite(expected)) and np.all(np.asarray(expected) > 1e-8)
    for head in range(actual.shape[0]):
        _assert_relative(actual[head], expected[head], f"aligned dhead_log_scale[h={head}]")
