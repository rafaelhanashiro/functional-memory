"""FLA's fused PaTH kernels (softmax, the exponential, the forget gate) and their polynomial fork,
against the JAX implementation.

CPU tests run the bridge's launches in Triton's interpreter (FP32 operands, Hopper tiles
unless stated), so the orchestration, buffer semantics, and every gradient are checked
without a GPU. GPU tests (``-m gpu``, H100 or B300) qualify the compiled kernels in BF16 at
the production width and context, at tile-boundary lengths, for causality, NaN rows, and
repeatability, and the model's dispatch.
"""
from dataclasses import replace
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.backends import triton_port
from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention, plan
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.path import path_attention
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.diagnostics import attention_identity

NAMES = ("output", "dq", "dk", "dv", "dw", "dbeta", "dc")
HOPPER = triton_port.Device(9, 0, triton_port.SHARED_MEMORY_TIERS["hopper"])


def _inputs(*, batch=1, time=256, heads=2, dim=32, degree=2, seed=0, norm=3.0, dtype=jnp.float32):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(batch, time, heads, dim)) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) * norm for x in (q, k))
    v = rng.normal(size=(batch, time, heads, dim))
    w = rng.normal(size=(batch, time, heads, dim))
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    beta = 2 / (1 + np.exp(-rng.normal(size=(batch, time, heads))))
    taylor = 1 / np.array([math.factorial(r) for r in range(max(degree, 1) + 1)])
    c = taylor + 0.3 * rng.normal(size=(batch, time, heads, taylor.size))
    cotangent = rng.normal(size=v.shape)
    cast = lambda x, d=dtype: jnp.asarray(x, d)
    operands = [cast(q), cast(k), cast(v), cast(w, jnp.float32), cast(beta, jnp.float32)]
    if degree:
        operands.append(cast(c, jnp.float32))
    return operands, cast(cotangent)


def _gated_inputs(*, decayed=True, **options):
    """Softmax/exp operands, with a cumulative log decay L [B, T, H] in Gated DeltaNet's initial range."""
    operands, cotangent = _inputs(degree=0, **options)
    if decayed:
        rng = np.random.default_rng(options.get("seed", 0) + 1000)
        batch, time, heads = operands[4].shape
        log_alpha = -rng.uniform(0, 16, size=heads) * np.exp(rng.uniform(np.log(1e-3), np.log(0.1),
                                                                         size=(batch, time, heads)))
        operands.append(jnp.asarray(np.cumsum(log_alpha, axis=1), jnp.float32))
    return operands, cotangent


def _names(kernel, operands):
    """Output and gradient names; the sixth operand is coefficients (polynomial) or the decay."""
    return NAMES[:len(operands) + 1] if kernel == "query_polynomial" else (*NAMES[:6], "dL")[:len(operands) + 1]


def _attend(implementation, kernel, **options):
    def attend(*x):
        extra = {} if len(x) == 5 else {"coefficients": x[5]} if kernel == "query_polynomial" else {"log_decay": x[5]}
        return implementation(*x[:5], kernel=kernel, **extra, **options)
    return attend


def _value_and_vjp(function, operands, cotangent):
    output, pull = jax.vjp(function, *operands)
    return (output, *pull(cotangent.astype(output.dtype)))


def _relative(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _reference(kernel, operands, cotangent):
    """FP32 JAX PaTH at HIGHEST precision."""
    with jax.default_matmul_precision("highest"):
        return _value_and_vjp(_attend(path_attention, kernel), [x.astype(jnp.float32) for x in operands],
                              cotangent.astype(jnp.float32))


def _interpreted(kernel, operands, cotangent, emulate=HOPPER):
    with triton_port.interpreter(emulate):
        return jax.block_until_ready(_value_and_vjp(_attend(fused_path_attention, kernel), operands, cotangent))


def _kernel(degree):
    return "softmax" if degree == 0 else "query_polynomial"


def _change_suffix(operands, prefix):
    """Different inputs from token ``prefix`` on, still unit w and beta in (0, 2); the prefix is untouched."""
    changed = [x.at[:, prefix:].add(1.5) for x in operands]
    suffix = changed[3][:, prefix:]
    changed[3] = changed[3].at[:, prefix:].set(suffix / jnp.linalg.norm(suffix, axis=-1, keepdims=True))
    changed[4] = changed[4].at[:, prefix:].set(jnp.clip(changed[4][:, prefix:], 0.01, 1.99))
    return changed


# ---------------------------------------------------------------------------------------
# CPU: Triton's interpreter
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("degree,time,dim", [
    (0, 128, 32), (0, 640, 16), (1, 640, 16), (2, 256, 32), (3, 640, 16), (4, 1152, 16),
])
def test_interpreted_outputs_and_every_gradient_match_the_jax_implementation(degree, time, dim):
    """One query block, and lengths that end inside a second or third 512-token split."""
    operands, cotangent = _inputs(time=time, dim=dim, degree=degree, seed=degree)
    expected = _reference(_kernel(degree), operands, cotangent)
    actual = _interpreted(_kernel(degree), operands, cotangent)
    for name, got, want in zip(NAMES, actual, expected):
        # FLA keeps FP16 copies of the Householder vectors in the transports.
        assert _relative(got, want) < 1e-3, name


@pytest.mark.parametrize("kernel,decayed,time,dim", [
    ("exp", False, 128, 32), ("exp", False, 640, 16), ("softmax", True, 128, 32), ("softmax", True, 640, 16),
    ("exp", True, 256, 32), ("exp", True, 1152, 16),
])
def test_interpreted_exponential_and_forget_gate_match_the_jax_implementation(kernel, decayed, time, dim):
    """The exponential and the gate, in one query block and in lengths ending inside later splits."""
    operands, cotangent = _gated_inputs(time=time, dim=dim, seed=time + dim, decayed=decayed)
    expected = _reference(kernel, operands, cotangent)
    actual = _interpreted(kernel, operands, cotangent)
    for name, got, want in zip(_names(kernel, operands), actual, expected, strict=True):
        assert _relative(got, want) < 1e-3, name


@pytest.mark.parametrize("shared_memory,tiles", [(166912, (32, 128)), (101376, (32, 64))])
def test_interpreted_smaller_shared_memory_tiles_follow_fla(shared_memory, tiles):
    """FLA's A100- and Ada-class branches: 32-token chunks with 128- or 64-token query blocks."""
    device = triton_port.Device(8, 0, shared_memory)
    assert (plan(device, 32).BS, plan(device, 32).BT) == tiles
    for degree in (0, 2):
        operands, cotangent = _inputs(time=640, dim=16, degree=degree, seed=5)
        expected = _reference(_kernel(degree), operands, cotangent)
        actual = _interpreted(_kernel(degree), operands, cotangent, emulate=device)
        for name, got, want in zip(NAMES, actual, expected):
            assert _relative(got, want) < 1e-3, (degree, name)


@pytest.mark.parametrize("kernel", ("query_polynomial", "exp"))
def test_interpreted_outputs_and_gradients_are_causal(kernel):
    operands, cotangent = (_inputs(time=640, dim=16, degree=2, seed=6) if kernel == "query_polynomial"
                           else _gated_inputs(time=640, dim=16, seed=6))
    prefix = 300
    cotangent = cotangent.at[:, prefix:].set(0)
    changed = _change_suffix(operands, prefix)
    first = _interpreted(kernel, operands, cotangent)
    second = _interpreted(kernel, changed, cotangent)
    np.testing.assert_array_equal(first[0][:, :prefix], second[0][:, :prefix])
    for name, gradient in zip(_names(kernel, operands)[1:], second[1:]):
        np.testing.assert_array_equal(np.asarray(gradient)[:, prefix:], 0, err_msg=name)


@pytest.mark.parametrize("field,degree", [("c", 2), ("q", 2), ("q", 0)])
def test_interpreted_nonfinite_values_poison_only_their_rows(field, degree):
    operands, _ = _inputs(time=256, dim=16, degree=degree, seed=7)
    row = 77
    if field == "c":
        operands[5] = operands[5].at[0, row, 1, -1].set(jnp.inf)
    else:
        operands[0] = operands[0].at[0, row, 1, 3].set(jnp.nan)
    with triton_port.interpreter():
        output = np.asarray(_attend(fused_path_attention, _kernel(degree))(*operands))
    reference = np.asarray(_attend(path_attention, _kernel(degree))(*operands))
    assert np.isnan(output[0, row, 1]).all()
    np.testing.assert_array_equal(np.isnan(output), np.isnan(reference))


def test_interpreter_leaves_the_triton_language_unpatched():
    """Later compilations in the process (other tests, the GPU path) see the real language."""
    import triton.language as tl

    objects = (tl, tl.core, tl.math, tl.tensor, tl.dtype)
    before = [dict(vars(obj)) for obj in objects]
    operands, cotangent = _inputs(time=128, dim=16, degree=0)
    _interpreted("softmax", operands, cotangent)
    assert [dict(vars(obj)) for obj in objects] == before


@pytest.mark.parametrize("change,message", [
    (lambda x: [y[:, :192] for y in x], "divisible by 128"),
    (lambda x: [y[..., :24] if y.ndim == 4 and i < 4 else y for i, y in enumerate(x)], "widths"),
    (lambda x: [x[0].astype(jnp.float16), *x[1:]], "one dtype"),
    (lambda x: x, "BF16 q, k, v on a GPU"),  # FP32 outside the interpreter
    (lambda x: [*x[:5], jnp.ones((*x[0].shape[:3], 6), jnp.float32)], "m in 1-4"),
])
def test_rejects_unsupported_operands(change, message):
    operands, _ = _inputs(time=256, dim=32, degree=2)
    operands = change(operands)
    with pytest.raises(ValueError, match=message):
        _attend(fused_path_attention, "query_polynomial")(*operands)


@pytest.mark.parametrize("kernel", ("softmax", "exp", "query_polynomial"))
def test_rejects_a_misshapen_or_non_fp32_decay(kernel):
    """The polynomial's gate is qualified in test_fox_path_attention_fused.py."""
    operands, _ = _inputs(time=256, dim=32, degree=2)
    coefficients = {"coefficients": operands[5]} if kernel == "query_polynomial" else {}
    decay = jnp.zeros(operands[4].shape, jnp.float32)
    for wrong in (decay[:, :128], decay.astype(jnp.bfloat16)):
        with triton_port.interpreter(), pytest.raises(ValueError, match="forget gate"):
            fused_path_attention(*operands[:5], kernel=kernel, log_decay=wrong, **coefficients)


def _model_config(kernel, backend, **updates):
    settings = dict(vocab_size=32, max_seq_len=128, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2,
                    qk_head_dim=16, value_head_dim=16, intermediate_size=48, attention_backend=backend,
                    initialization="transformer_reference", qk_norm=True, positional_encoding="path")
    if kernel.removesuffix("_forget_gate") == "query_polynomial":
        settings.update(attention_degree=2, attention_normalization=False, attention_coefficients="learned",
                        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                        attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.5,
                        attention_coefficient_projection_initial_std=0.25, attention_output_scaling="head_rms_norm")
    elif kernel in ("exp", "exp_forget_gate"):
        settings.update(attention_normalization=False, attention_output_scaling="head_rms_norm")
    if kernel.endswith("_forget_gate"):
        settings.update(attention_forget_gate=True)
    return ModelConfig(**{**settings, **updates})


def _config(model):
    config = Config(model=model)
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 32, 129, 2
    config.train.micro_batch_size, config.train.dtype = 2, "bfloat16"
    return config


def _model_loss_and_gradients(config, tokens, dtype=jnp.float32):
    model = build_model(config, 3)
    for _, module in nnx.iter_graph(model):  # the compute dtype, as create_state sets it
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype

    def loss(model):
        logits = model(tokens[:, :-1])
        return jnp.mean(-jnp.take_along_axis(jax.nn.log_softmax(logits.astype(jnp.float32)),
                                             tokens[:, 1:, None], axis=-1))

    value, gradients = jax.value_and_grad(loss)(model)
    return value, dict(nnx.to_flat_state(nnx.state(gradients, nnx.Param)))


MODEL_KERNELS = ("softmax", "query_polynomial", "exp", "softmax_forget_gate", "exp_forget_gate",
                 "query_polynomial_forget_gate")


@pytest.mark.parametrize("kernel", MODEL_KERNELS)
def test_interpreted_fused_model_matches_the_blocked_model(kernel):
    tokens = jnp.asarray(np.random.default_rng(8).integers(0, 32, (2, 129)))
    with jax.default_matmul_precision("highest"):
        expected, expected_gradients = _model_loss_and_gradients(_model_config(kernel, "blocked"), tokens)
    with triton_port.interpreter():
        actual, actual_gradients = _model_loss_and_gradients(_model_config(kernel, "fused"), tokens)
    assert abs(float(actual) / float(expected) - 1) < 1e-5
    for path, gradient in expected_gradients.items():
        assert _relative(actual_gradients[path][...], gradient[...]) < 1e-3, path


def test_configuration_backend_resolution_identity_and_acceptance():
    for kernel in MODEL_KERNELS:
        fused = _model_config(kernel, "fused")
        assert fused.path_fused_supported()
        assert resolve_attention_backend(replace(fused, attention_backend="auto"), dtype="bfloat16",
                                         platform="gpu", sequence_length=2048) == "fused"
        assert resolve_attention_backend(replace(fused, attention_backend="auto"), dtype="bfloat16",
                                         platform="cpu", sequence_length=2048) == "blocked"
        assert resolve_attention_backend(replace(fused, attention_backend="auto"), dtype="bfloat16",
                                         platform="gpu", sequence_length=2000) == "blocked"
        wide = replace(fused, attention_backend="auto", qk_head_dim=128, value_head_dim=128)
        for capability, expected_backend in ((9, "fused"), (10, "blocked")):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr("scaffolds.transformer.attention.backends.selection._runtime_platform", lambda: "gpu")
                patch.setattr("scaffolds.transformer.attention.backends.selection._cuda_compute_capability",
                              lambda capability=capability: capability)
                assert resolve_attention_backend(wide, dtype="bfloat16", sequence_length=2048) == expected_backend
        with pytest.raises(ValueError, match="sm90"):
            resolve_attention_backend(fused, dtype="bfloat16", platform="gpu", sequence_length=2000)
        identity = attention_identity(_config(fused))
        gated = "_forget_gate" if kernel.endswith("_forget_gate") else ""
        assert identity["attention_implementation"] == {
            "softmax": "fla_parallel_path_attention_triton", "exp": "fla_parallel_path_attention_triton_times_row_sum",
            "query_polynomial": "fla_parallel_path_query_polynomial_triton"}[kernel.removesuffix("_forget_gate")] + gated
        assert identity.get("attention_forget_gate", False) is bool(gated)
        assert identity["diagnostic_backend"] == "jax_path_chunked_ut_transform"
        blocked = attention_identity(_config(_model_config(kernel, "blocked")))
        assert blocked["attention_implementation"] == "jax_path_chunked_ut_transform" + gated
        recipe = config_to_dict(_config(fused))
        spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
        for cpu in (True, False):
            paths = acceptance.test_paths(spec, recipe, resolved_backend="fused", cpu=cpu)
            assert acceptance.PATH_ATTENTION_FUSED_TESTS in paths
            assert (acceptance.TRITON_PORT_TESTS in paths) == cpu
        assert acceptance.PATH_ATTENTION_FUSED_TESTS not in acceptance.test_paths(
            spec, recipe, resolved_backend="blocked", cpu=False)
        assert "--xla_gpu_enable_command_buffer=" in acceptance.attention_spec(
            _config(fused), entry_module="tests").runtime_env["XLA_FLAGS"]
    for updates in ({"value_head_dim": 32}, {"qk_head_dim": 8, "value_head_dim": 8}, {"attention_backend": "mosaic"}):
        with pytest.raises(ValueError):
            _model_config("softmax", "fused", **updates)


# ---------------------------------------------------------------------------------------
# GPU qualification
# ---------------------------------------------------------------------------------------

def _require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires an H100 or B300 allocation")


def _compiled(kernel, operands, cotangent):
    return jax.jit(lambda *x: _value_and_vjp(_attend(fused_path_attention, kernel), x, cotangent))(*operands)


# Relative L2 error against FP32 HIGHEST on BF16 q/k/v. FLA rounds intermediates to BF16
# (dw and dbeta are stored in BF16) and keeps FP16 copies of w; the JAX implementation on
# the same BF16 inputs is roughly 2x more accurate (recorded in the validation README).
TOLERANCE = {"output": 1e-2, "dq": 2e-2, "dk": 2e-2, "dv": 1e-2, "dw": 2.5e-2, "dbeta": 2.5e-2, "dc": 5e-3,
             "dL": 2e-2}


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (0, 1, 2, 3, 4))
def test_gpu_production_context_matches_the_jax_implementation(degree):
    _require_gpu()
    operands, cotangent = _inputs(batch=2, time=2048, heads=4, dim=64, degree=degree, seed=11, norm=4.0,
                                  dtype=jnp.bfloat16)
    expected = jax.jit(lambda *x: _reference(_kernel(degree), x, cotangent))(*operands)
    actual = _compiled(_kernel(degree), operands, cotangent)
    for name, got, want in zip(NAMES, actual, expected):
        assert bool(jnp.isfinite(got).all()), name
        assert _relative(got, want) < TOLERANCE[name], (name, _relative(got, want))


@pytest.mark.gpu
@pytest.mark.parametrize("kernel,decayed", [("softmax", True), ("exp", False), ("exp", True)])
def test_gpu_exponential_and_forget_gate_production_context_match_the_jax_implementation(kernel, decayed):
    _require_gpu()
    operands, cotangent = _gated_inputs(batch=2, time=2048, heads=4, dim=64, seed=21, norm=4.0, dtype=jnp.bfloat16,
                                        decayed=decayed)
    expected = jax.jit(lambda *x: _reference(kernel, x, cotangent))(*operands)
    actual = _compiled(kernel, operands, cotangent)
    for name, got, want in zip(_names(kernel, operands), actual, expected, strict=True):
        assert bool(jnp.isfinite(got).all()), name
        assert _relative(got, want) < TOLERANCE[name], (name, _relative(got, want))


@pytest.mark.gpu
@pytest.mark.parametrize("time", (128, 256, 384, 512, 640, 1024, 1152, 1536))
@pytest.mark.parametrize("kernel", ("softmax", "exp"))
def test_gpu_forget_gate_tile_boundary_lengths_match_the_jax_implementation(time, kernel):
    _require_gpu()
    operands, cotangent = _gated_inputs(batch=2, time=time, heads=3, dim=64, seed=time + 1, norm=4.0,
                                        dtype=jnp.bfloat16)
    expected = jax.jit(lambda *x: _reference(kernel, x, cotangent))(*operands)
    actual = _compiled(kernel, operands, cotangent)
    for name, got, want in zip(_names(kernel, operands), actual, expected, strict=True):
        assert _relative(got, want) < TOLERANCE[name], (name, _relative(got, want))


@pytest.mark.gpu
@pytest.mark.parametrize("time", (128, 256, 384, 512, 640, 1024, 1152, 1536))
@pytest.mark.parametrize("degree", (0, 3))
def test_gpu_tile_boundary_lengths_match_the_jax_implementation(time, degree):
    """One 128-token query block up to three 512-token splits, ending inside or at a split."""
    _require_gpu()
    operands, cotangent = _inputs(batch=2, time=time, heads=3, dim=64, degree=degree, seed=time, norm=4.0,
                                  dtype=jnp.bfloat16)
    expected = jax.jit(lambda *x: _reference(_kernel(degree), x, cotangent))(*operands)
    actual = _compiled(_kernel(degree), operands, cotangent)
    for name, got, want in zip(NAMES, actual, expected):
        assert _relative(got, want) < TOLERANCE[name], (name, _relative(got, want))


@pytest.mark.gpu
@pytest.mark.parametrize("dim", (16, 32, 128))
def test_gpu_other_head_widths_match_the_jax_implementation(dim):
    _require_gpu()
    for degree in (0, 2):
        operands, cotangent = _inputs(batch=1, time=1024, heads=2, dim=dim, degree=degree, seed=dim, norm=4.0,
                                      dtype=jnp.bfloat16)
        if dim == 128 and triton_port.device().blackwell:
            # FLA's dq kernel fails to launch at width 128 on Blackwell; the operation refuses it.
            with pytest.raises(ValueError, match="widths"):
                _compiled(_kernel(degree), operands, cotangent)
            continue
        expected = jax.jit(lambda *x: _reference(_kernel(degree), x, cotangent))(*operands)
        actual = _compiled(_kernel(degree), operands, cotangent)
        for name, got, want in zip(NAMES, actual, expected):
            assert _relative(got, want) < TOLERANCE[name], (dim, degree, name, _relative(got, want))


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ("softmax", "exp"))
def test_gpu_forget_gate_outputs_and_gradients_are_causal(kernel):
    _require_gpu()
    operands, cotangent = _gated_inputs(batch=1, time=2048, heads=2, dim=64, seed=22, dtype=jnp.bfloat16)
    prefix = 1100  # inside the third split
    cotangent = cotangent.at[:, prefix:].set(0)
    changed = _change_suffix(operands, prefix)
    first = _compiled(kernel, operands, cotangent)
    second = _compiled(kernel, changed, cotangent)
    np.testing.assert_array_equal(np.asarray(first[0][:, :prefix], np.float32),
                                  np.asarray(second[0][:, :prefix], np.float32))
    for name, gradient in zip(_names(kernel, operands)[1:], second[1:]):
        np.testing.assert_array_equal(np.asarray(gradient, np.float32)[:, prefix:], 0, err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (0, 2))
def test_gpu_outputs_and_gradients_are_causal(degree):
    _require_gpu()
    operands, cotangent = _inputs(batch=1, time=2048, heads=2, dim=64, degree=degree, seed=12, dtype=jnp.bfloat16)
    prefix = 1100  # inside the third split
    cotangent = cotangent.at[:, prefix:].set(0)
    changed = _change_suffix(operands, prefix)
    first = _compiled(_kernel(degree), operands, cotangent)
    second = _compiled(_kernel(degree), changed, cotangent)
    np.testing.assert_array_equal(np.asarray(first[0][:, :prefix], np.float32),
                                  np.asarray(second[0][:, :prefix], np.float32))
    for name, gradient in zip(NAMES[1:], second[1:]):
        np.testing.assert_array_equal(np.asarray(gradient, np.float32)[:, prefix:], 0, err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("field,degree", [("c", 4), ("q", 2), ("q", 0), ("k", 0)])
def test_gpu_nonfinite_values_poison_the_same_rows_as_the_jax_implementation(field, degree):
    _require_gpu()
    operands, _ = _inputs(batch=1, time=1024, heads=2, dim=64, degree=degree, seed=13, dtype=jnp.bfloat16)
    row = 700
    if field == "c":
        operands[5] = operands[5].at[0, row, 1, -1].set(jnp.inf)
    elif field == "q":
        operands[0] = operands[0].at[0, row, 1, 3].set(jnp.nan)
    else:
        operands[1] = operands[1].at[0, row, 1, 3].set(jnp.nan)
    output = np.asarray(jax.jit(_attend(fused_path_attention, _kernel(degree)))(*operands), np.float32)
    reference = np.asarray(jax.jit(_attend(path_attention, _kernel(degree)))(*operands), np.float32)
    assert np.isnan(output[0, row, 1]).all()
    np.testing.assert_array_equal(np.isnan(output), np.isnan(reference))
    if field == "k":  # every later query reads the key
        assert np.isnan(output[0, row:, 1]).all() and np.isfinite(output[0, :row]).all()
    else:
        assert np.isfinite(np.delete(output, row, axis=1)).all()


REPEATS = 10


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (0, 2))
def test_gpu_repeatability_at_the_production_batch(degree):
    """Outputs, dq, and dc repeat bitwise; dk, dv, dw, and dbeta sum FP32 atomics in any order."""
    _require_gpu()
    operands, cotangent = _inputs(batch=32, time=2048, heads=16, dim=64, degree=degree, seed=14, norm=4.0,
                                  dtype=jnp.bfloat16)
    run = jax.jit(lambda *x: _value_and_vjp(_attend(fused_path_attention, _kernel(degree)), x, cotangent))
    first = [np.asarray(x, np.float32) for x in run(*operands)]
    for attempt in range(REPEATS):
        again = [np.asarray(x, np.float32) for x in run(*operands)]
        for name, got, want in zip(NAMES, again, first):
            if name in ("output", "dq", "dc"):
                np.testing.assert_array_equal(got, want, err_msg=f"{name}, attempt {attempt}")
            else:  # dw and dbeta are rounded to BF16 after the atomic sums
                bound = 1e-4 if name in ("dk", "dv") else 5e-3
                assert _relative(got, want) < bound, (attempt, name, _relative(got, want))


@pytest.mark.gpu
def test_gpu_plan_follows_fla_branches_on_hopper_and_blackwell():
    _require_gpu()
    device = triton_port.device()
    assert device.arch in ("hopper", "blackwell") and device.shared_memory >= triton_port.SHARED_MEMORY_TIERS["hopper"]
    tiles = plan(device, 64)
    assert (tiles.BS, tiles.BT, tiles.inter_warps, tiles.cumprod_warps) == (64, 128, 4, 4)
    assert (tiles.cumprod_fwd_stages, tiles.dq_stages, tiles.cumprod_bwd_stages, tiles.intra_bwd_stages) == (3, 3, 2, 3)
    assert plan(device, 128).inter_warps == 8


# The gated polynomial is compared here only in FP32 (the interpreter case above): in BF16 its
# fork's path_beta_bias gradient exceeds twice the blocked model's error at this scale (its dbeta
# is stored in BF16), and its BF16 gradients are qualified per operator in
# test_fox_path_attention_fused.py.
@pytest.mark.gpu
@pytest.mark.parametrize("kernel", [kernel for kernel in MODEL_KERNELS if kernel != "query_polynomial_forget_gate"])
def test_gpu_fused_model_matches_the_blocked_model(kernel):
    _require_gpu()
    tokens = jnp.asarray(np.random.default_rng(15).integers(0, 32, (2, 1025)))
    settings = dict(max_seq_len=1024, qk_head_dim=64, value_head_dim=64, hidden_size=128)
    with jax.default_matmul_precision("highest"):
        expected, expected_gradients = _model_loss_and_gradients(_model_config(kernel, "blocked", **settings), tokens)
    blocked, blocked_gradients = _model_loss_and_gradients(_model_config(kernel, "blocked", **settings), tokens,
                                                           jnp.bfloat16)
    actual, actual_gradients = _model_loss_and_gradients(_model_config(kernel, "fused", **settings), tokens,
                                                         jnp.bfloat16)
    # BF16 compute against the FP32 model, next to the JAX implementation's own BF16 error.
    assert abs(float(actual) / float(expected) - 1) < max(2 * abs(float(blocked) / float(expected) - 1), 1e-3)
    for path, gradient in expected_gradients.items():
        bound = max(2 * _relative(blocked_gradients[path][...], gradient[...]), 2e-2)
        assert _relative(actual_gradients[path][...], gradient[...]) < bound, path
