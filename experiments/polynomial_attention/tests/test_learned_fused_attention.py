"""Independent reference and CUDA gates for learned quadratic attention.

CPU tests interpret the same Pallas reduction programs with an FP32 reference
dot and equivalent interpreter-supported loads. GPU tests compile the actual
CUDA paths, including their mixed BF16/FP32 products and production loads.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.polynomial_attention.models.components import fused_quadratic_attention as fused


MODES = ("positive", "unrestricted")


def _inputs(shape, *, dtype=jnp.float32, mode="positive", seed=431):
    batch, time, heads, dq, dv = shape
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(rng.normal(0, 0.7, (batch, time, heads, dq)), dtype=dtype)
            for _ in range(2))
    v, cotangent = (jnp.asarray(rng.normal(size=(batch, time, heads, dv)), dtype=dtype)
                    for _ in range(2))
    # Distinct heads expose accidentally shared coefficient loads/reductions.
    first = np.linspace(0.6, 1.2, heads)
    second = np.linspace(0.4, 1.1, heads) if mode == "positive" else np.linspace(0.8, 1.1, heads)
    coefficients = jnp.asarray(np.stack((first, second), axis=1), dtype=jnp.float32)
    return (q, k, v, coefficients), cotangent


def _dense_reference(q, k, v, coefficients, *, mode):
    """Direct expanded polynomial, independent of fused evaluation/derivatives."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    first, second = coefficients[:, 0][None, :, None, None], coefficients[:, 1][None, :, None, None]
    if mode == "positive":
        constant, linear, quadratic = 0.5 * (1 + second ** 2), first * second, 0.5 * first ** 2
    else:
        constant, linear, quadratic = 1.0, first, second
    values = constant + linear * scores + quadratic * scores ** 2
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=jnp.bool_))
    values = jnp.where(causal, values, 0)
    weights = values / jnp.sum(values, axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


def _forward_vjp(attend, operands, cotangent):
    output, pullback = jax.vjp(attend, *operands)
    return output, pullback(cotangent)


def _evaluate(attend, operands, cotangent):
    return jax.jit(partial(_forward_vjp, attend))(operands, cotangent)


def _fp64(operands, cotangent, mode):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _evaluate(partial(_dense_reference, mode=mode),
                           tuple(x.astype(jnp.float64) for x in operands),
                           cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), result)


def _assert_array(actual, expected, tolerance, label):
    observed, reference = (np.asarray(x, dtype=np.float64) for x in (actual, expected))
    assert observed.shape == reference.shape, label
    assert np.isfinite(observed).all(), label
    assert np.isfinite(reference).all(), label
    # Check each head independently, including the small coefficient arrays.
    views = ((observed[h], reference[h]) for h in range(reference.shape[0])) if reference.ndim == 2 else (
        (observed[b, :, h], reference[b, :, h])
        for b in range(reference.shape[0]) for h in range(reference.shape[2])
    )
    for got, want in views:
        scale = np.linalg.norm(want)
        if scale == 0:
            np.testing.assert_array_equal(got, want, err_msg=label)
            continue
        error = np.linalg.norm(got - want) / scale
        assert error <= tolerance, f"{label}: relative norm error {error} > {tolerance}"
        assert np.max(np.abs(got - want)) <= 3 * tolerance * np.max(np.abs(want)), label


def _assert_result(actual, expected, *, output_tolerance=3e-5,
                   gradient_tolerance=1e-4, coefficient_tolerance=1e-4):
    _assert_array(actual[0], expected[0], output_tolerance, "output")
    for name, observed, reference in zip("qkvc", actual[1], expected[1], strict=True):
        _assert_array(observed, reference,
                      coefficient_tolerance if name == "c" else gradient_tolerance,
                      f"d{name}")


def _interpreted(monkeypatch, mode):
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST,
    ))
    # Pallas' HLO interpreter does not support mixed slice/array indexing.
    # This equivalent load uses only array indices; production keeps the
    # established Triton load expression shared with fixed Taylor attention.
    def interpreted_load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)
    monkeypatch.setattr(fused, "_load", interpreted_load)
    return partial(fused._attention, mode=mode, interpret=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("time", (5, 67))
def test_cpu_interpreted_outputs_all_gradients_and_padded_rows(monkeypatch, mode, time):
    operands, cotangent = _inputs((2, time, 3, 7, 5), mode=mode)
    actual = _evaluate(_interpreted(monkeypatch, mode), operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent, mode))
    assert actual[1][3].dtype == jnp.float32


def _signed_inputs(case, *, time=130):
    q = jnp.zeros((1, time, 1, 1), jnp.float32).at[:, -1].set(1)
    if case == "zero_partial":
        # First 64 final-query weights are +1 then -1, summing to zero.
        scores = np.zeros(time, np.float32)
        scores[32:64] = -2
    elif case == "negative_partial":
        scores = np.ones(time, np.float32)
        scores[:64] = -2
    elif case == "negative_final":
        # Vary the negative weights so score/coefficient VJPs do not vanish
        # through an unrelated uniform-kernel rescaling symmetry.
        scores = np.linspace(-3, -2, time, dtype=np.float32)
    elif case == "zero_kernel":
        scores = np.zeros(time, np.float32)
        scores[:32] = -1
    elif case == "zero_final":
        scores = np.zeros(time, np.float32)
        scores[time // 2:] = -2
    else:
        raise ValueError(case)
    k = jnp.asarray(scores).reshape(q.shape)
    rng = np.random.default_rng(432)
    v, cotangent = (jnp.asarray(rng.normal(size=q.shape), dtype=jnp.float32) for _ in range(2))
    return (q, k, v, jnp.asarray([[1.0, 0.0]], jnp.float32)), cotangent


@pytest.mark.parametrize("case", ("zero_partial", "negative_partial", "negative_final", "zero_kernel"))
def test_cpu_signed_partial_and_final_normalizers(monkeypatch, case):
    operands, cotangent = _signed_inputs(case)
    actual = _evaluate(_interpreted(monkeypatch, "unrestricted"), operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent, "unrestricted"),
                   gradient_tolerance=3e-4, coefficient_tolerance=3e-4)


def test_cpu_zero_final_normalizer_poisoning(monkeypatch):
    operands, _ = _signed_inputs("zero_final")
    output = jax.jit(_interpreted(monkeypatch, "unrestricted"))(*operands)
    assert bool(jnp.all(jnp.isfinite(output[:, :-1])))
    assert bool(jnp.all(jnp.isnan(output[:, -1])))


@pytest.mark.parametrize("mode", MODES)
def test_cpu_global_coefficient_broadcast_gradient_sums_heads(monkeypatch, mode):
    operands, cotangent = _inputs((2, 7, 3, 4, 5), mode=mode)
    shared = jnp.asarray([1.0, 1.0 if mode == "positive" else 0.5], jnp.float32)
    attend = _interpreted(monkeypatch, mode)
    per_head = jnp.broadcast_to(shared, (3, 2))
    expected = _evaluate(attend, (*operands[:3], per_head), cotangent)[1][3].sum(axis=0)
    observed = jax.grad(lambda pair: jnp.sum(attend(
        *operands[:3], jnp.broadcast_to(pair, (3, 2)),
    ) * cotangent))(shared)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("bad", (jnp.ones((2,), jnp.float32), jnp.ones((1, 3), jnp.float32),
                                 jnp.ones((1, 2), jnp.bfloat16)))
def test_reject_invalid_coefficient_shape_or_dtype(bad):
    q = jnp.ones((1, 3, 1, 4), jnp.float32)
    with pytest.raises(ValueError, match="coefficients"):
        fused.fused_quadratic_attention(q, q, q, bad, quadratic_mode="positive")


def test_reject_unknown_family():
    q = jnp.ones((1, 3, 1, 4), jnp.float32)
    with pytest.raises(ValueError, match="quadratic_mode"):
        fused.fused_quadratic_attention(q, q, q, jnp.ones((1, 2)), quadratic_mode="fixed")


@pytest.fixture
def require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_gpu_independent_fp64_outputs_and_all_gradients(require_gpu, mode, dtype):
    operands, cotangent = _inputs((2, 131, 3, 33, 17), mode=mode, dtype=dtype)
    actual = _evaluate(partial(fused.fused_quadratic_attention, quadratic_mode=mode), operands, cotangent)
    bf16 = dtype == jnp.bfloat16
    _assert_result(actual, _fp64(operands, cotangent, mode),
                   output_tolerance=0.006 if bf16 else 3e-5,
                   gradient_tolerance=0.008 if bf16 else 2e-4,
                   coefficient_tolerance=3e-4)
    assert all(value.dtype == dtype for value in (actual[0], *actual[1][:3]))
    assert actual[1][3].dtype == jnp.float32


@pytest.mark.gpu
@pytest.mark.parametrize("case", ("zero_partial", "negative_partial", "negative_final", "zero_kernel"))
def test_gpu_signed_tile_sums_outputs_and_all_gradients(require_gpu, case):
    operands, cotangent = _signed_inputs(case)
    actual = _evaluate(partial(fused.fused_quadratic_attention, quadratic_mode="unrestricted"), operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent, "unrestricted"),
                   gradient_tolerance=4e-4, coefficient_tolerance=4e-4)


@pytest.mark.gpu
def test_gpu_zero_final_sum_poisoning(require_gpu):
    operands, _ = _signed_inputs("zero_final")
    output = jax.jit(partial(fused.fused_quadratic_attention, quadratic_mode="unrestricted"))(*operands)
    assert bool(jnp.all(jnp.isfinite(output[:, :-1])))
    assert bool(jnp.all(jnp.isnan(output[:, -1])))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
def test_gpu_masked_overflow_and_nonfinite_actual_scores(require_gpu, mode):
    coefficients = jnp.asarray([[1.0, 1.0 if mode == "positive" else 0.5]], jnp.float32)
    attend = jax.jit(partial(fused.fused_quadratic_attention, quadratic_mode=mode))
    q = jnp.asarray([1.0, 0.0], jnp.float32).reshape(1, 2, 1, 1)
    k = jnp.asarray([0.0, 1e30], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0], jnp.float32).reshape(q.shape)
    # The enormous future score is masked before the polynomial is evaluated.
    np.testing.assert_array_equal(attend(q, k, v, coefficients).reshape(-1), [3, 5])
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    k = jnp.asarray([0.0, 0.0, jnp.inf], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3.0, 7.0, 11.0], jnp.float32).reshape(q.shape)
    output = attend(q, k, v, coefficients)
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [3, 5])
    assert bool(jnp.all(jnp.isnan(output[:, -1])))


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
def test_gpu_causality_and_padding_do_not_contribute_coefficient_gradients(require_gpu, mode):
    operands, cotangent = _inputs((1, 131, 2, 7, 5), mode=mode, dtype=jnp.bfloat16)
    attend = partial(fused.fused_quadratic_attention, quadratic_mode=mode)
    prefix = 65
    restricted = cotangent.at[:, prefix:].set(0)
    actual = _evaluate(attend, operands, restricted)
    prefix_result = _evaluate(attend, tuple(x[:, :prefix] for x in operands[:3]) + (operands[3],), cotangent[:, :prefix])
    _assert_array(actual[0][:, :prefix], prefix_result[0], 0.006, "prefix output")
    _assert_array(actual[1][3], prefix_result[1][3], 3e-4, "prefix coefficient gradient")
    for grad in actual[1][:3]:
        np.testing.assert_array_equal(grad[:, prefix:], jnp.zeros_like(grad[:, prefix:]))
    changed = (operands[0].at[:, prefix:].multiply(-2), operands[1].at[:, prefix:].multiply(3),
               operands[2].at[:, prefix:].add(5), operands[3])
    changed_output = jax.jit(attend)(*changed)
    np.testing.assert_array_equal(changed_output[:, :prefix], actual[0][:, :prefix])


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
def test_gpu_full_production_shape_bf16_matches_blocked(require_gpu, mode):
    from experiments.polynomial_attention.models.components.attention import blocked_dot_product_attention
    # Four-GPU production has sixteen sequences per device/microbatch.
    operands, cotangent = _inputs((16, 2048, 16, 64, 64), mode=mode, dtype=jnp.bfloat16, seed=433)
    actual = _evaluate(partial(fused.fused_quadratic_attention, quadratic_mode=mode), operands, cotangent)
    reference = _evaluate(lambda q, k, v, coefficients: blocked_dot_product_attention(
        q, k, v, attention_degree=2, coefficients=coefficients, quadratic_mode=mode,
    ), operands, cotangent)
    _assert_result(actual, reference, output_tolerance=0.006,
                   gradient_tolerance=0.03, coefficient_tolerance=0.003)
    # Anchor an independently compiled BF16 prefix directly to FP64 as well.
    prefix = tuple(x[:1, :129, :1] for x in operands[:3]) + (operands[3][:1],)
    prefix_cotangent = cotangent[:1, :129, :1]
    actual_prefix = _evaluate(partial(fused.fused_quadratic_attention, quadratic_mode=mode), prefix, prefix_cotangent)
    _assert_result(actual_prefix, _fp64(prefix, prefix_cotangent, mode),
                   output_tolerance=0.006, gradient_tolerance=0.008,
                   coefficient_tolerance=3e-4)
