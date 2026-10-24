"""H100/B300 backward tiles, including the composed head-RMSNorm gradient."""

from functools import partial
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
from scaffolds.transformer.attention.input_dependent import query_key_polynomial_attention
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.tests.test_query_key_polynomial import _dense, _vjp
from scaffolds.transformer.tests.test_query_tanh_polynomial import _assert_close


@pytest.fixture(autouse=True)
def _fresh_kernel_traces(request):
    # Tests deliberately switch geometry. A cached custom-VJP trace must not
    # silently retain a preceding test's monkeypatched launch geometry.
    gpu = request.node.get_closest_marker("gpu") is not None
    if gpu:
        jax.clear_caches()
    yield
    if gpu:
        jax.clear_caches()


def _device(kind="NVIDIA H100 80GB HBM3", capability="9.0"):
    return SimpleNamespace(device_kind=kind, compute_capability=capability)


@pytest.mark.parametrize("updates,device,expected", [
    ({}, _device(), (32, 32)),
    ({"degree": 12}, _device("NVIDIA H100 PCIe"), (32, 32)),
    ({"degree": 4}, _device(), (64, 64)),
    ({"degree": 16}, _device(), (64, 64)),
    ({"time": 1024}, _device(), (64, 64)),
    ({"dtype": jnp.float32}, _device(), (64, 64)),
    ({"dq": 32}, _device(), (64, 64)),
    ({"dv": 32}, _device(), (64, 64)),
    ({}, _device("NVIDIA H200", "9.0"), (64, 64)),
    ({}, _device("NVIDIA B300 SXM6 AC", "10.3"), (32, 32)),
    ({"degree": 12}, _device("NVIDIA B300 SXM6 AC", "10.3"), (32, 32)),
    ({"degree": 4}, _device("NVIDIA B300", "10.3"), (64, 64)),
    ({"time": 1024}, _device("NVIDIA B300", "10.3"), (64, 64)),
    ({"dtype": jnp.float32}, _device("NVIDIA B300", "10.3"), (64, 64)),
    ({"dq": 32}, _device("NVIDIA B300", "10.3"), (64, 64)),
    ({"dv": 32}, _device("NVIDIA B300", "10.3"), (64, 64)),
    ({}, _device("NVIDIA B300", "10.0"), (64, 64)),
    ({}, _device("NVIDIA B200", "10.0"), (64, 64)),
    ({}, _device("NVIDIA B200", "10.3"), (64, 64)),
    ({}, _device("cpu", None), (64, 64)),
])
def test_tuned_tile_selection_is_limited_to_qualified_shapes_and_devices(monkeypatch, updates, device, expected):
    monkeypatch.setattr(kernel.jax, "local_devices", lambda: [device])
    settings = dict(dq=64, dv=64, degree=8, time=2048, dtype=jnp.bfloat16, interpret=False)
    settings.update(updates)
    assert kernel._backward_tile_sizes(**settings) == expected
    # The forward geometry remains the existing generic geometry.
    assert kernel._tile_sizes(64, 64) == (64, 64)


@pytest.mark.parametrize("devices,expected", [
    ([], (64, 64)),
    ([_device("NVIDIA B300", "10.3"), _device("NVIDIA B300 SXM6 AC", "10.3")], (32, 32)),
    ([_device("NVIDIA B300", "10.3"), _device("NVIDIA H200", "9.0")], (64, 64)),
    ([_device("NVIDIA B300", "10.3"), _device()], (64, 64)),
])
def test_tuned_tile_selection_requires_a_nonempty_qualified_device_family(monkeypatch, devices, expected):
    monkeypatch.setattr(kernel.jax, "local_devices", lambda: devices)
    assert kernel._backward_tile_sizes(64, 64, degree=8, time=2048, dtype=jnp.bfloat16,
                                       interpret=False) == expected


@pytest.mark.parametrize("cpu", (True, False))
@pytest.mark.parametrize("backend", ("fused", "blocked", "mosaic"))
def test_fused_query_key_acceptance_requires_tiling_checks(cpu, backend):
    from scaffolds.transformer import acceptance
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=())
    config = {"model": {"attention_coefficient_conditioning": "query_key"}}
    paths = acceptance.test_paths(spec, config, resolved_backend=backend, cpu=cpu)
    assert (acceptance.QUERY_KEY_TILING_TESTS in paths) == (backend == "fused" and cpu)


@pytest.mark.parametrize("devices,degree,expected", [
    ([_device()], 8, True),
    ([_device("NVIDIA B300 SXM6 AC", "10.3")], 8, True),
    ([_device("NVIDIA B300 SXM6 AC", "10.3")], 12, True),
    ([_device("NVIDIA B300 SXM6 AC", "10.3")], 4, False),
    ([_device("NVIDIA H200", "9.0")], 8, False),
    ([_device("NVIDIA A100", "8.0")], 8, False),
    ([_device("NVIDIA B200", "10.0")], 8, False),
    ([_device("NVIDIA B300", "10.3"), _device("NVIDIA H200", "9.0")], 8, False),
    ([], 8, False),
])
def test_gpu_acceptance_follows_selected_geometry_without_skipping_fallbacks(monkeypatch, devices, degree, expected):
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.configs import Config, config_to_dict

    monkeypatch.setattr(kernel.jax, "local_devices", lambda: devices)
    config = Config()
    config.model.attention_coefficient_conditioning = "query_key"
    config.model.attention_degree = degree
    config.model.qk_head_dim = config.model.value_head_dim = 64
    config.data.sequence_length = 2049
    config.train.dtype = "bfloat16"
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=())
    runtime_paths = acceptance._runtime_test_paths(config, "fused")
    gpu_paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=False,
                                      runtime_test_paths=runtime_paths)
    assert (acceptance.QUERY_KEY_TILING_TESTS in gpu_paths) == expected
    assert acceptance.QUERY_KEY_POLYNOMIAL_TESTS in gpu_paths
    cpu_paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=True,
                                      runtime_test_paths=runtime_paths)
    assert acceptance.QUERY_KEY_TILING_TESTS in cpu_paths
    assert acceptance._runtime_test_paths(config, "blocked") == ()


def test_interpreter_does_not_inspect_hardware_and_preserves_tile_override(monkeypatch):
    def forbidden():
        raise AssertionError("Interpreter must not inspect hardware")
    monkeypatch.setattr(kernel.jax, "local_devices", forbidden)
    monkeypatch.setattr(kernel, "_tile_sizes", lambda dq, dv: (16, 32))
    assert kernel._backward_tile_sizes(64, 64, degree=12, time=2048, dtype=jnp.bfloat16,
                                     interpret=True) == (16, 32)


def _requires_qualified_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires an H100 or B300 allocation")
    devices = jax.local_devices()
    h100 = all(getattr(d, "compute_capability", None) == "9.0"
               and d.device_kind.startswith("NVIDIA H100") for d in devices)
    b300 = all(getattr(d, "compute_capability", None) == "10.3"
               and d.device_kind.startswith("NVIDIA B300") for d in devices)
    if not devices or not (h100 or b300):
        pytest.skip("This launch geometry requires an H100 or B300 device family")


def _inputs(degree, *, time, dtype):
    random = np.random.default_rng(8613 + degree)
    shape = (1, time, 2, 64)
    q, k, v, cotangent = (jnp.asarray(random.normal(size=shape), dtype) for _ in range(4))
    q, k = cosine_score_operands(q, k, jnp.full((2, 1), math.log(8), jnp.float32), eps=1e-6)
    factorials = np.asarray([math.factorial(r) for r in range(degree + 1)])
    beta = 1 + .2 * np.tanh(random.normal(size=(2, degree + 1)))
    alpha = .4 * np.tanh(random.normal(size=(2, degree + 1)))
    coefficients = (beta + alpha * np.tanh(random.normal(size=(1, time, 2, degree + 1)))) / factorials
    # One head exercises signed kernels rather than only near-exponential weights.
    coefficients[:, :, 1, 0] *= .1
    coefficients[:, :, 1, 1] *= -1
    factors = 1 + .3 * np.tanh(random.normal(size=coefficients.shape))
    gain = 1 + .1 * random.normal(size=(2, 64))
    return (q, k, v, jnp.asarray(coefficients, jnp.float32), jnp.asarray(factors, jnp.float32),
            jnp.asarray(gain, jnp.float32)), cotangent


def _composed(q, k, v, coefficients, factors, gain, *, backend="fused"):
    output = query_key_polynomial_attention(q, k, v, coefficients, factors, backend=backend)
    return scale_attention_output(output, "head_rms_norm", output_norm_scale=gain)


def _dense_composed(q, k, v, coefficients, factors, gain):
    output = _dense(q, k, v, coefficients, factors)
    return output / jnp.sqrt(jnp.mean(output ** 2, axis=-1, keepdims=True) + 1e-6) * gain[None, None]


def _compare_all(actual, expected, tolerance):
    _assert_close(actual[0], expected[0], tolerance, "output")
    for index, (observed, reference) in enumerate(zip(actual[1], expected[1], strict=True)):
        _assert_close(observed, reference, tolerance, f"gradient {index}")
        if index in (3, 4):
            for head in range(reference.shape[2]):
                for degree in range(reference.shape[-1]):
                    _assert_close(observed[..., head, degree], reference[..., head, degree], tolerance,
                                  f"gradient {index}, head {head}, degree {degree}")


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (8, 12))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_gpu_tuned_head_rmsnorm_and_all_gradients_match_small_fp64(monkeypatch, degree, dtype):
    _requires_qualified_gpu()
    operands, cotangent = _inputs(degree, time=67, dtype=dtype)
    # Exercise the optimized backward's 32-token boundaries in a compact FP64
    # comparison, independent of the production-only selector restriction.
    monkeypatch.setattr(kernel, "_backward_tile_sizes", lambda *a, **kw: (32, 32))
    actual = jax.jit(partial(_vjp, _composed))(operands, cotangent)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(_dense_composed, tuple(x.astype(jnp.float64) for x in operands),
                        cotangent.astype(jnp.float64))
        expected = jax.tree.map(lambda x: np.asarray(x, np.float64), expected)
    _compare_all(actual, expected, 0.025 if dtype == jnp.bfloat16 else 2e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (8, 12))
def test_gpu_tuned_production_context_head_rmsnorm_matches_blocked_and_original(monkeypatch, degree):
    _requires_qualified_gpu()
    assert kernel._backward_tile_sizes(64, 64, degree=degree, time=2048, dtype=jnp.bfloat16,
                                       interpret=False) == (32, 32)
    operands, cotangent = _inputs(degree, time=2048, dtype=jnp.bfloat16)
    actual = jax.jit(partial(_vjp, _composed))(operands, cotangent)
    actual = jax.tree.map(lambda x: np.array(jax.device_get(x), copy=True), actual)
    with jax.default_matmul_precision("highest"):
        reference = jax.jit(partial(_vjp, partial(_composed, backend="blocked")))(operands, cotangent)
    _compare_all(actual, reference, 0.025)
    # Force an independent fresh trace of the original backward geometry.
    monkeypatch.setattr(kernel, "_backward_tile_sizes", lambda *a, **kw: (64, 64))
    jax.clear_caches()
    original = jax.jit(partial(_vjp, _composed))(operands, cotangent)
    np.testing.assert_array_equal(actual[0], original[0])
    _compare_all(actual, original, 0.025)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (8, 12))
def test_gpu_tuned_production_prefix_and_gradients_are_causal(degree):
    _requires_qualified_gpu()
    operands, cotangent = _inputs(degree, time=2048, dtype=jnp.bfloat16)
    prefix = 67  # Cross both the original and optimized tile boundaries.
    cotangent = cotangent.at[:, prefix:].set(0)
    operation = jax.jit(partial(_vjp, _composed))
    original = operation(operands, cotangent)
    altered_inputs = tuple(x.at[:, prefix:].add(.125) for x in operands[:5]) + operands[5:]
    altered = operation(altered_inputs, cotangent)
    np.testing.assert_array_equal(original[0][:, :prefix], altered[0][:, :prefix])
    for index, (before, after) in enumerate(zip(original[1], altered[1], strict=True)):
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
        if index < 5:
            np.testing.assert_array_equal(before[:, prefix:], 0)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (8, 12))
def test_gpu_tuned_future_nonfinite_keys_and_factors_leave_prefix_finite(degree):
    _requires_qualified_gpu()
    q = jnp.ones((1, 2048, 1, 64), jnp.bfloat16)
    coefficients = jnp.ones((1, 2048, 1, degree + 1), jnp.float32)
    factors = jnp.ones_like(coefficients)
    attend = jax.jit(kernel.fused_query_key_polynomial_attention)
    boundary = 67
    output = attend(q, q.at[:, boundary, :, 0].set(jnp.nan), q, coefficients, factors)
    assert bool(jnp.isfinite(output[:, :boundary]).all())
    assert bool(jnp.isnan(output[:, boundary:]).all())
    output = attend(q, q, q, coefficients, factors.at[:, boundary, :, -1].set(jnp.inf))
    assert bool(jnp.isfinite(output[:, :boundary]).all())
    assert bool(jnp.isnan(output[:, boundary:]).all())
    output = attend(q, q, q, coefficients.at[:, boundary, :, -1].set(jnp.inf), factors)
    assert bool(jnp.isfinite(output[:, :boundary]).all())
    assert bool(jnp.isnan(output[:, boundary]).all())
    assert bool(jnp.isfinite(output[:, boundary + 1:]).all())
