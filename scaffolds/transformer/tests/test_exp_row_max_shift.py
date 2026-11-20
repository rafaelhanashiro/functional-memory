"""The unnormalized exponential shifted by its causal row maximum (``model.attention_exp_shift: row_max``).

y_t = sum_(i<=t) e^(z_ti - m_t) v_i, m_t = max_(i<=t) z_ti over the row's logits z (scores, decayed or transformed),
and head RMSNorm with the offset, y_t / sqrt(mean(y_t^2) + eps e^(-2 m_t)) g, which is the unshifted
o_t / sqrt(mean(o_t^2) + eps) g in real arithmetic (attention/output_scaling.py). The result does not depend on
m_t, so its total derivative through m_t vanishes and m_t is a constant to differentiation. These checks hold the
shifted scan and strips to that function in FP64 (also where e^s overflows FP32), to the unshifted path wherever
that path is finite, and to each other; the fused kernel's cases need a GPU (test_unnormalized_fused_attention.py).
"""
from functools import partial
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import (
    blocked_backend_attention, blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.attention.unnormalized import unnormalized_attention
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.configs.base import fresh_recipe_uses_exp_shift
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict

EPS = 1e-6
UNNORMALIZED = {"attention_normalization": False}
SHIFTED = {"attention_normalization": False, "exp_shift": "row_max"}


def _arrays(*, scale=1.0, batch=2, time=37, heads=3, width=8, value_width=5, seed=0):
    """FP64 q, k, v (scores scale with ``scale**2``), a head RMSNorm gain, and an output cotangent."""
    rng = np.random.default_rng(seed)
    q, k = (scale * rng.standard_normal((batch, time, heads, width)) for _ in range(2))
    v = rng.standard_normal((batch, time, heads, value_width))
    gain = rng.uniform(0.5, 2.0, (heads, value_width))
    cotangent = rng.standard_normal((batch, time, heads, value_width))
    return q, k, v, gain, cotangent


def _logits(q, k, *, log_decay=None, transform="identity"):
    """[b, h, t, s] scores, transformed, plus the cumulative decay L_t - L_i on causal pairs."""
    from scaffolds.transformer.attention.score_transform import apply

    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    logits = apply(jnp.where(causal, scores, 0.0), transform)
    if log_decay is not None:
        decay = jnp.swapaxes(log_decay, 1, 2)
        logits = logits + jnp.where(causal, decay[..., :, None] - decay[..., None, :], 0.0)
    return logits, causal


def _dense(q, k, v, gain, *, shift="none", log_decay=None, transform="identity", eps=EPS):
    """The represented function o_t / sqrt(mean(o_t^2) + eps) g, independent of the implementations: unshifted, or
    shifted with m_t held constant ("stop") or differentiated ("differentiable"). Equal in real arithmetic."""
    logits, causal = _logits(q, k, log_decay=log_decay, transform=transform)
    maximum = jnp.zeros(logits.shape[:-1] + (1,), logits.dtype)
    if shift != "none":
        maximum = jnp.max(jnp.where(causal, logits, -jnp.inf), axis=-1, keepdims=True)
        if shift == "stop":
            maximum = jax.lax.stop_gradient(maximum)
    weights = jnp.where(causal, jnp.exp(jnp.where(causal, logits - maximum, 0.0)), 0.0)
    output = jnp.einsum("bhts,bshd->bthd", weights, v)
    offset = jnp.swapaxes(maximum[..., 0], 1, 2)[..., None]
    denominator = jnp.mean(output * output, axis=-1, keepdims=True) + eps * jnp.exp(-2.0 * offset)
    return output * jax.lax.rsqrt(denominator) * gain


def _vjp(function, arrays, cotangent):
    output, pullback = jax.vjp(function, *arrays)
    return output, pullback(cotangent)


def _relative(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


def _assert_close(actual, expected, *, rtol, label):
    """Relative L2 per array and every element within 3 rtol of the array's largest magnitude."""
    for index, (got, want) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        assert got.shape == want.shape, f"{label}[{index}]"
        assert np.isfinite(got).all() and np.isfinite(want).all(), f"{label}[{index}]: nonfinite values"
        assert _relative(got, want) <= rtol, f"{label}[{index}]: relative L2 {_relative(got, want):.3g} > {rtol}"
        assert np.max(np.abs(got - want)) <= 3 * rtol * np.max(np.abs(want)), f"{label}[{index}]: element error"


def _implementation(implementation, *, time):
    """Shifted attention and head RMSNorm on the blocked backend's scan or strips (FP32 arithmetic)."""
    if implementation == "scan":
        attend = partial(blocked_dot_product_attention, query_block_size=8)
    else:
        attend = partial(strip_dot_product_attention, strip_size=16)

    def function(q, k, v, gain, shifted=True, **arguments):
        if shifted:
            output, shift = attend(q, k, v, **SHIFTED, **arguments)
            return scale_attention_output(output, "head_rms_norm", output_norm_scale=gain, output_shift=shift)
        output = attend(q, k, v, **UNNORMALIZED, **arguments)
        return scale_attention_output(output, "head_rms_norm", output_norm_scale=gain)

    return function


@pytest.mark.parametrize("scale", (1.0, 2.0, 2.5))
def test_the_shifted_function_is_the_unshifted_one_and_its_gradient_ignores_the_shift(scale):
    """In FP64 (scores up to about 60 at the largest scale): output and VJP in q, k, v and the gain agree among
    the unshifted function, the shifted one with m_t held constant, and the shifted one differentiated through
    m_t. So the gradient through the shift vanishes, and holding it constant is exact."""
    q, k, v, gain, cotangent = _arrays(scale=scale)
    with jax.enable_x64():
        arrays = tuple(jnp.asarray(x, jnp.float64) for x in (q, k, v, gain))
        cotangent = jnp.asarray(cotangent, jnp.float64)
        unshifted = _vjp(_dense, arrays, cotangent)
        stopped = _vjp(partial(_dense, shift="stop"), arrays, cotangent)
        differentiated = _vjp(partial(_dense, shift="differentiable"), arrays, cotangent)
    for label, result in (("held shift", stopped), ("differentiated shift", differentiated)):
        _assert_close(result, unshifted, rtol=1e-11, label=label)


@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("scale,largest", ((1.0, 5.0), (3.0, 45.0), (6.0, 190.0), (9.0, 440.0)))
def test_scan_and_strips_match_the_fp64_function_where_the_exponential_overflows(implementation, scale, largest):
    """FP32 shifted attention and head RMSNorm against the FP64 function of the same FP32 inputs (``_dense``),
    output and VJP, at row maxima from a few units to far beyond FP32's e^88.7, where the unshifted FP32 output
    overflows."""
    q, k, v, gain, cotangent = _arrays(scale=scale, seed=1)
    arrays = tuple(jnp.asarray(x, jnp.float32) for x in (q, k, v, gain))
    cotangent = jnp.asarray(cotangent, jnp.float32)
    with jax.enable_x64():
        arrays64 = tuple(jnp.asarray(np.asarray(x, np.float64)) for x in arrays)
        expected = _vjp(partial(_dense, shift="stop"), arrays64, jnp.asarray(np.asarray(cotangent, np.float64)))
        largest_score = float(jnp.max(_logits(*arrays64[:2])[0]))
    assert largest_score >= largest  # The case reaches its intended scores.
    function = _implementation(implementation, time=q.shape[1])
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: _vjp(function, a[:4], a[4]))(*arrays, cotangent)
        unshifted = jax.jit(partial(function, shifted=False))(*arrays)
    _assert_close(actual[0], expected[0], rtol=5e-6, label="output")
    _assert_close(actual[1], expected[1], rtol=2e-5, label="VJP")
    assert np.isfinite(np.asarray(unshifted)).all() == (largest_score < 88)


def _cases():
    """(name, attention arguments, FP64 reference arguments): the plain exponential, decay and score functions."""
    rng = np.random.default_rng(7)
    cumulative = np.cumsum(-rng.uniform(0.005, 0.05, (2, 37, 3)), axis=1)
    return {
        "exp": ({}, {}),
        "cumulative_decay": ({"query_log_decay": cumulative}, {"log_decay": cumulative}),
        "softplus_transform": ({"score_transform": "softplus"}, {"transform": "softplus"}),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_the_shift_is_each_rows_largest_causal_logit(name):
    """The returned m_t is the largest causal logit of row t (decay and transform included), and the shifted output
    times e^(m_t) is the unshifted output, in FP64 with the scan's own code."""
    arguments, dense = _cases()[name]
    q, k, v, _, _ = _arrays(scale=1.5, seed=2)
    with jax.enable_x64():
        arguments = {key: jnp.asarray(value, jnp.float64) if isinstance(value, np.ndarray) else value
                     for key, value in arguments.items()}
        q, k, v = (jnp.asarray(x, jnp.float64) for x in (q, k, v))
        scan = partial(reference._blocked_scan, compute=jnp.float64, query_block_size=8)
        output, shift = scan(q, k, v, **SHIFTED, **arguments)
        unshifted = np.asarray(scan(q, k, v, **UNNORMALIZED, **arguments))
        logits, causal = _logits(q, k, log_decay=dense.get("log_decay"), transform=dense.get("transform", "identity"))
        expected = np.asarray(jnp.swapaxes(jnp.max(jnp.where(causal, logits, -jnp.inf), axis=-1), 1, 2))
        assert output.dtype == shift.dtype == jnp.float64 and shift.shape == q.shape[:3]
        output, shift = np.asarray(output), np.asarray(shift)
    np.testing.assert_allclose(shift, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(output * np.exp(shift)[..., None], unshifted, rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(unshifted)))


STRIP_CASES = {
    "exp": ({}, ()),
    "cumulative_decay": ({}, ("query_log_decay",)),
    "query_distance_decay": ({"query_decay_mode": "query_distance"}, ("query_log_alpha",)),
    "endpoint_decay": ({"query_decay_mode": "query_distance"}, ("query_gate_input", "key_gate_input", "gate_log_rate")),
    "softplus_transform": ({"score_transform": "softplus"}, ()),
    "sigmoid_prefactor": ({"score_prefactor": "sigmoid"}, ()),
    "softplus_transform_with_swish_prefactor": ({"score_transform": "softplus", "score_prefactor": "swish"}, ()),
}


def _strip_arrays(name, *, time=37, batch=2, heads=2, width=8, value_width=5):
    rng = np.random.default_rng(11 + list(STRIP_CASES).index(name))
    arrays = {"q": 1.5 * rng.standard_normal((batch, time, heads, width)),
              "k": 1.5 * rng.standard_normal((batch, time, heads, width)),
              "v": rng.standard_normal((batch, time, heads, value_width))}
    for key in STRIP_CASES[name][1]:
        if key == "query_log_decay":
            arrays[key] = np.cumsum(-rng.uniform(0.005, 0.05, (batch, time, heads)), axis=1)
        elif key == "query_log_alpha":
            arrays["query_log_decay"] = -rng.uniform(0.005, 0.1, (batch, time, heads))
        elif key in ("query_gate_input", "key_gate_input"):
            arrays[key] = rng.standard_normal((batch, time, heads))
        else:
            arrays[key] = np.log(rng.uniform(0.005, 0.03, (heads, 1)))
    return arrays


@pytest.mark.parametrize("time,strip,block", ((37, 8, 4), (5, 8, 4)))
@pytest.mark.parametrize("name", list(STRIP_CASES))
def test_strips_shift_as_the_scan_in_exact_arithmetic(name, time, strip, block):
    """Output, shifts and the gradient of every differentiable input (the cotangent on the output, the shifts held
    constant), strips against the scan's own code in FP64, with any decay or score function."""
    static, _ = STRIP_CASES[name]
    arrays = _strip_arrays(name, time=time)
    names = list(arrays)
    cotangent = np.random.default_rng(time).standard_normal(arrays["v"].shape)

    def run(attend):
        def function(*values):
            given = dict(zip(names, values))
            return attend(given.pop("q"), given.pop("k"), given.pop("v"), **SHIFTED, **static, **given)

        (output, shift), pullback = jax.vjp(function, *(jnp.asarray(arrays[key]) for key in names))
        return output, shift, pullback((jnp.asarray(cotangent), jnp.zeros_like(shift)))

    with jax.enable_x64():
        expected = run(partial(reference._blocked_scan, compute=jnp.float64, query_block_size=block))
        actual = run(partial(strip_dot_product_attention, strip_size=strip))
    for label, observed, wanted in zip(["output", "shift", *names], [*actual[:2], *actual[2]],
                                       [*expected[:2], *expected[2]], strict=True):
        observed, wanted = np.asarray(observed), np.asarray(wanted)
        assert observed.dtype == np.float64 and np.isfinite(wanted).all(), label
        np.testing.assert_allclose(observed, wanted, rtol=1e-11, atol=1e-11 * max(np.max(np.abs(wanted)), 1.0),
                                   err_msg=label)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize("implementation", ("scan", "strips"))
def test_shifted_equals_unshifted_where_the_unshifted_path_is_finite(implementation, dtype):
    """Q/K/V in the model's dtype at scores up to about 18, where the unshifted path is finite and accurate: the
    two agree in output and VJP to FP32 rounding, or in BF16 to the BF16 rounding of the attention output both
    store (and of their Q/K/V gradients)."""
    q, k, v, gain, cotangent = _arrays(scale=1.8, seed=3)
    arrays = tuple(jnp.asarray(x, dtype) for x in (q, k, v)) + (jnp.asarray(gain, jnp.float32),)
    function = _implementation(implementation, time=q.shape[1])
    cotangent = jnp.asarray(cotangent, dtype)
    with jax.default_matmul_precision("highest"):
        shifted = jax.jit(lambda *a: _vjp(function, a[:4], a[4]))(*arrays, cotangent)
        unshifted = jax.jit(lambda *a: _vjp(partial(function, shifted=False), a[:4], a[4]))(*arrays, cotangent)
    assert shifted[0].dtype == dtype and all(grad.dtype == array.dtype for grad, array in zip(shifted[1], arrays))
    tolerance = 1e-5 if dtype == jnp.float32 else 1.5e-2
    _assert_close(shifted, unshifted, rtol=tolerance, label=f"{implementation} {jnp.dtype(dtype).name}")


def test_head_rms_norm_with_a_shift_is_head_rms_norm_of_the_unshifted_output():
    """scale_attention_output(e^(-m) o, offset m) is scale_attention_output(o): output and VJP in o and the gain, FP32
    (shifts in [-8, 8]), and BF16 rows; with the offset far beyond FP32's range it matches FP64; a zero row stays zero
    with a finite gradient; and only head RMSNorm takes an offset of the right shape."""
    rng = np.random.default_rng(5)
    output = rng.standard_normal((2, 9, 3, 6))
    shift = rng.uniform(-8, 8, (2, 9, 3))
    gain = jnp.asarray(rng.uniform(0.5, 2.0, (3, 6)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal(output.shape), jnp.float32)
    scaled = partial(scale_attention_output, scaling="head_rms_norm")
    for dtype, rtol in ((jnp.float32, 3e-6), (jnp.bfloat16, 1e-2)):
        shifted_output = jnp.asarray(output * np.exp(-shift)[..., None], dtype)
        # The unshifted output those BF16 values represent, exact in FP32.
        represented = jnp.asarray(np.asarray(shifted_output, np.float64) * np.exp(shift)[..., None], jnp.float32)
        actual = _vjp(lambda y, g: scaled(y, output_norm_scale=g, output_shift=jnp.asarray(shift, jnp.float32)),
                      (shifted_output, gain), cotangent.astype(dtype))
        expected = _vjp(lambda y, g: scaled(y, output_norm_scale=g), (represented, gain), cotangent)
        _assert_close(actual[0], expected[0], rtol=rtol, label="output")
        # d/dy' = e^m d/do: compare the shifted gradient with the unshifted one carried to y'.
        carried = np.asarray(expected[1][0], np.float64) * np.exp(shift)[..., None]
        _assert_close(actual[1][0], carried, rtol=rtol, label="dy")
        _assert_close(actual[1][1], expected[1][1], rtol=rtol, label="dgain")
    # Offsets of 300 to 2000: the unshifted output is far beyond FP32, the FP64 function is not.
    shift = rng.uniform(300, 2000, (2, 9, 3))
    with jax.enable_x64():
        unit = jnp.asarray(output, jnp.float64)
        function = lambda y, g: y * jax.lax.rsqrt(jnp.mean(y * y, -1, keepdims=True)  # noqa: E731
                                                   + EPS * jnp.exp(-2 * jnp.asarray(shift)[..., None])) * g
        expected = _vjp(function, (unit, jnp.asarray(gain, jnp.float64)), jnp.asarray(cotangent, jnp.float64))
    actual = _vjp(lambda y, g: scaled(y, output_norm_scale=g, output_shift=jnp.asarray(shift, jnp.float32)),
                  (jnp.asarray(output, jnp.float32), gain), cotangent)
    _assert_close(actual, expected, rtol=3e-6, label="beyond FP32")
    zero = jnp.zeros((1, 1, 1, 6), jnp.float32)
    for offset in (0.0, 60.0, 1e4):
        result, (gradient,) = _vjp(lambda y: scaled(y, output_norm_scale=jnp.ones((1, 6)),
                                                    output_shift=jnp.full((1, 1, 1), offset)), (zero,),
                                   jnp.ones_like(zero))
        np.testing.assert_array_equal(result, zero)
        assert np.isfinite(np.asarray(gradient)).all()
    for bad in (np.inf, np.nan):
        row = jnp.ones((1, 1, 1, 6), jnp.float32).at[..., 2].set(bad)
        result = scaled(row, output_norm_scale=jnp.ones((1, 6)), output_shift=jnp.zeros((1, 1, 1)))
        assert np.isnan(np.asarray(result)).all()
    y = jnp.ones((2, 9, 3, 6), jnp.float32)
    with pytest.raises(ValueError, match="Only head RMSNorm"):
        scale_attention_output(y, "inverse_position", output_shift=jnp.zeros((2, 9, 3)))
    with pytest.raises(ValueError, match="Only head RMSNorm"):
        scale_attention_output(y, "none", output_shift=jnp.zeros((2, 9, 3)))
    with pytest.raises(ValueError, match="output_shift must have shape"):
        scaled(y, output_norm_scale=jnp.ones((3, 6)), output_shift=jnp.zeros((2, 9)))


def _model(*, exp_shift, implementation="scan", dtype=jnp.float32, backend="blocked", **settings):
    """A two-layer model with the canonical QK RMSNorm and head RMSNorm, its parameters in FP32 and its layers
    computing in ``dtype``, as training builds it (training/state.py); the GPU module runs it on the fused kernel."""
    from scaffolds.transformer.models import build_model

    config = ModelConfig(vocab_size=64, max_seq_len=32, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2,
                         qk_head_dim=8, value_head_dim=16, intermediate_size=64, attention_backend=backend,
                         attention_normalization=False, attention_output_scaling="head_rms_norm",
                         attention_exp_shift=exp_shift, attention_blocked_implementation=implementation,
                         initialization="transformer_reference", qk_norm=True, qk_norm_type="rms_norm",
                         qk_norm_sharing="per_layer", qk_norm_head_scale=True, **settings)
    model = build_model(config, nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    return nnx.split(model, nnx.Param)


def _loss_and_gradients(graphdef, params, tokens):
    def loss(params):
        logits = nnx.merge(graphdef, params)(tokens[:, :-1]).astype(jnp.float32)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tokens[:, 1:, None], axis=-1))

    return jax.jit(jax.value_and_grad(loss))(params)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize("implementation", ("scan", "strips"))
def test_model_shifted_equals_unshifted(implementation, dtype):
    """A two-layer model (QK RMSNorm with head scales, RoPE, head RMSNorm): loss and every parameter gradient. In
    FP32 the shifted model's are the unshifted one's to rounding; with BF16 layers both carry BF16 rounding, and the
    shifted model's lie as close to the FP32 unshifted model's as the BF16 unshifted model's do (within twice
    their distance, or 1%)."""
    tokens = jnp.asarray(np.random.default_rng(5).integers(0, 64, (2, 33)), jnp.int32)
    with jax.default_matmul_precision("highest"):
        unshifted = _loss_and_gradients(*_model(exp_shift="none", implementation=implementation, dtype=dtype), tokens)
        shifted = _loss_and_gradients(*_model(exp_shift="row_max", implementation=implementation, dtype=dtype),
                                      tokens)
        reference32 = (unshifted if dtype == jnp.float32 else
                       _loss_and_gradients(*_model(exp_shift="none", implementation=implementation), tokens))
    np.testing.assert_allclose(shifted[0], reference32[0], rtol=1e-6 if dtype == jnp.float32 else 1e-4)
    for (path, observed), wanted, other in zip(jax.tree_util.tree_leaves_with_path(shifted[1]),
                                               jax.tree.leaves(reference32[1]), jax.tree.leaves(unshifted[1]),
                                               strict=True):
        assert observed.dtype == jnp.float32 and np.isfinite(np.asarray(observed)).all(), str(path)
        tolerance = 1e-5 if dtype == jnp.float32 else max(2 * _relative(other, wanted), 1e-2)
        assert _relative(observed, wanted) <= tolerance, (str(path), _relative(observed, wanted), tolerance)


def test_layer_far_beyond_fp32s_exponential_trains_on_finite_values():
    """Plain scores scaled to row maxima in the hundreds: the unshifted layer's output is nonfinite, the shifted
    layer's output and every gradient finite, and its output the FP64 function's of its own projections."""
    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.standard_normal((2, 19, 32)), jnp.float32)
    gain = jnp.ones((4, 8), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((2, 19, 32)), jnp.float32)
    common = dict(hidden_size=32, num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                  attention_normalization=False, backend="blocked", attention_output_scaling="head_rms_norm")
    layers = {}
    for shift in ("none", "row_max"):
        layer = MultiHeadAttention(**common, attention_exp_shift=shift, rngs=nnx.Rngs(8))
        layer.q_proj.kernel[...] = layer.q_proj.kernel[...] * 60.0
        layers[shift] = nnx.split(layer, nnx.Param)

    def run(graphdef, params):
        def function(params, gain):
            return nnx.merge(graphdef, params)(x, attention_parameters={"output_norm_scale": gain})

        return jax.jit(lambda params, gain: _vjp(function, (params, gain), cotangent))(params, gain)

    with jax.default_matmul_precision("highest"):
        unshifted = run(*layers["none"])
        output, gradients = run(*layers["row_max"])
    assert not np.isfinite(np.asarray(unshifted[0])).all()
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves((output, gradients)))
    from scaffolds.transformer.attention.positional import apply_rope

    layer = nnx.merge(*layers["row_max"])
    q, k, v = (projection(x).reshape(2, 19, 4, 8) for projection in (layer.q_proj, layer.k_proj, layer.v_proj))
    q, k = (apply_rope(array, jnp.arange(19), layer.rope_theta) for array in (q, k))
    with jax.enable_x64():
        operands = tuple(jnp.asarray(np.asarray(array, np.float64)) for array in (q, k, v, gain))
        heads = np.asarray(_dense(*operands, shift="stop"))
        assert float(jnp.max(_logits(*operands[:2])[0])) > 150
    expected = heads.reshape(2, 19, 32) @ np.asarray(layer.o_proj.kernel[...], np.float64)
    _assert_close(output, expected, rtol=2e-5, label="layer output")


@pytest.mark.parametrize("settings,error", [
    ({"attention_output_scaling": "none"}, "head RMSNorm"),
    ({"attention_output_scaling": "inverse_position"}, "head RMSNorm"),
    ({"attention_normalization": True}, "unnormalized exponential"),
    ({"attention_degree": 2}, "unnormalized exponential"),
    ({"attention_degree": 4, "attention_coefficients": "learned"}, "unnormalized exponential"),
    ({"positional_encoding": "path"}, "PaTH"),
    ({"attention_backend": "mosaic", "attention_degree": None}, "(?i)backend"),
    ({"attention_backend": "fused", "attention_score_transform": "softplus"}, "fused kernel"),
    ({"attention_backend": "fused", "attention_score_prefactor": "sigmoid"}, "fused kernel"),
    ({"attention_score_mlp_width": 4, "attention_score_mlp_amplitude_initial": (1.0,) * 4,
      "attention_score_mlp_slope_initial": (1.0,) * 4, "attention_score_mlp_offset_initial": (0.0,) * 4},
     "unnormalized exponential"),
    ({"attention_exp_shift": "flash"}, "Unknown model.attention_exp_shift"),
])
def test_recipes_where_the_shift_would_change_the_function_are_refused(settings, error):
    base = dict(attention_backend="blocked", attention_normalization=False, attention_output_scaling="head_rms_norm",
                attention_exp_shift="row_max")
    with pytest.raises(ValueError, match=error):
        ModelConfig(**{**base, **settings})


@pytest.mark.parametrize("settings", [
    {}, {"attention_backend": "fused"}, {"attention_backend": "auto"}, {"positional_encoding": "none"},
    {"attention_blocked_implementation": "strips"}, {"attention_forget_gate": True},
    {"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance"},
    {"attention_forget_gate": True, "attention_forget_gate_form": "constant"},
    {"attention_score_transform": "softplus"}, {"attention_score_prefactor": "sigmoid"},
    {"attention_backend": "fused", "qk_head_dim": 4},
])
def test_the_shift_applies_to_the_unnormalized_exponential_with_head_rms_norm(settings):
    """Blocked with any forget gate or score function, fused (and auto, which resolves to one of the two) plain."""
    model = ModelConfig(**{**dict(attention_backend="blocked", attention_normalization=False,
                                  attention_output_scaling="head_rms_norm"), **settings})
    assert model.attention_exp_shift == "none" and fresh_recipe_uses_exp_shift(model)
    model.attention_exp_shift = "row_max"
    model.__post_init__()


@pytest.mark.parametrize("settings", [
    {"attention_normalization": True}, {"attention_output_scaling": "none"}, {"attention_degree": 4},
    {"positional_encoding": "path"}, {"attention_backend": "fused", "attention_score_prefactor": "softplus"},
])
def test_fresh_recipes_keep_the_unshifted_exponential_where_the_shift_does_not_apply(settings):
    model = ModelConfig(**{**dict(attention_backend="blocked", attention_normalization=False,
                                  attention_output_scaling="head_rms_norm"), **settings})
    assert not fresh_recipe_uses_exp_shift(model)


def test_recipes_without_the_field_are_unshifted():
    assert ModelConfig().attention_exp_shift == "none"
    config = Config(model=ModelConfig(attention_backend="blocked", attention_normalization=False,
                                      attention_output_scaling="head_rms_norm"))
    validate_config(config)
    current = config_to_dict(config)
    legacy = json.loads(json.dumps(current))
    del legacy["model"]["attention_exp_shift"]
    assert config_dicts_equal(current, legacy)
    config.model.attention_exp_shift = "row_max"
    assert not config_dicts_equal(config_to_dict(config), legacy)


def test_kernels_and_dispatch_refuse_the_shift_elsewhere():
    q, k, v = (jnp.ones((1, 4, 2, 4), jnp.float32) for _ in range(3))
    for arguments in ({}, {"attention_degree": 2, "attention_normalization": False}):
        with pytest.raises(ValueError, match="unnormalized exponential alone"):
            blocked_dot_product_attention(q, k, v, exp_shift="row_max", **arguments)
        with pytest.raises(ValueError, match="unnormalized exponential alone"):
            attention_core(q, k, v, backend="blocked", exp_shift="row_max", **arguments)
    with pytest.raises(ValueError, match="Unknown exponential shift"):
        blocked_dot_product_attention(q, k, v, attention_normalization=False, exp_shift="max")
    with pytest.raises(ValueError, match="exponential, on the blocked backend or fused kernel"):
        unnormalized_attention(q, k, v, attention_degree=2, exp_shift="row_max")
    with pytest.raises(ValueError, match="exponential, on the blocked backend or fused kernel"):
        unnormalized_attention(q, k, v, backend="mosaic", exp_shift="row_max")
    from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

    with pytest.raises(ValueError, match="plain exponential alone"):
        fused_unnormalized_attention(q, k, v, score_transform="softplus", exp_shift="row_max")
    with pytest.raises(ValueError, match="plain exponential alone"):
        fused_unnormalized_attention(q, k, v, attention_degree=2, exp_shift="row_max")
    with pytest.raises(ValueError, match="Unknown exponential shift"):
        fused_unnormalized_attention(q, k, v, exp_shift="max")
    common = dict(hidden_size=8, num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=4,
                  attention_normalization=False, attention_exp_shift="row_max", rngs=nnx.Rngs(0))
    for settings in ({"backend": "blocked"}, {"backend": "fused", "attention_output_scaling": "head_rms_norm",
                                              "attention_score_transform": "softplus"}):
        with pytest.raises(ValueError, match="row-maximum shift applies"):
            MultiHeadAttention(**common, **settings)


def test_unshifted_calls_return_the_output_alone():
    """The default leaves every call's return as it was (receipts check the arithmetic over every recorded run)."""
    q, k, v = (jnp.asarray(np.random.default_rng(1).standard_normal((1, 6, 2, 4)), jnp.float32) for _ in range(3))
    for attend in (blocked_dot_product_attention, strip_dot_product_attention,
                   partial(blocked_backend_attention, implementation="strips")):
        assert isinstance(attend(q, k, v, attention_normalization=False), jax.Array)
        output, shift = attend(q, k, v, **SHIFTED)
        assert output.shape == v.shape and shift.shape == (1, 6, 2) and shift.dtype == jnp.float32


def test_diagnostics_describe_the_shifted_weights_and_the_shifts():
    """The scan's kernel and weight-sum diagnostics are those of e^(s - m_t), in [0, 1] with a row sum of at least
    one, and exp_shift_min/max bound the shifts; the strips take the scan's diagnostics."""
    q, k, v, _, _ = _arrays(scale=6.0, seed=4)
    q, k, v = (jnp.asarray(x, jnp.float32) for x in (q, k, v))
    (output, shift), diagnostics = blocked_dot_product_attention(q, k, v, **SHIFTED, return_diagnostics=True)
    # The largest weight is e^0 = 1 up to the rounding of the scores (FP32 at scores near 200, contracted products).
    assert 0 <= float(diagnostics["kernel_min"]) and abs(float(diagnostics["kernel_max"]) - 1.0) < 1e-5
    assert float(diagnostics["weight_sum_min"]) >= 1.0 - 1e-5
    assert float(diagnostics["exp_shift_min"]) == float(jnp.min(shift))
    assert float(diagnostics["exp_shift_max"]) == float(jnp.max(shift)) == float(diagnostics["score_max"]) > 89
    assert int(diagnostics["invalid_kernel_count"]) == int(diagnostics["invalid_output_count"]) == 0
    (strip_output, strip_shift), strip_diagnostics = blocked_backend_attention(
        q, k, v, implementation="strips", **SHIFTED, return_diagnostics=True)
    assert set(strip_diagnostics) == set(diagnostics)
    np.testing.assert_allclose(strip_shift, shift, rtol=1e-6)


def _training_config(path, implementation="scan"):
    from scaffolds.transformer.tests.test_training import tiny_config

    config = tiny_config(path)
    config.model.attention_backend = "blocked"
    config.model.attention_normalization = False
    config.model.attention_output_scaling = "head_rms_norm"
    config.model.attention_exp_shift = "row_max"
    config.model.attention_blocked_implementation = implementation
    config.model.qk_norm, config.model.qk_norm_type, config.model.qk_norm_sharing = True, "rms_norm", "per_layer"
    config.model.qk_norm_head_scale = True
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    return config


def test_identity_and_telemetry_record_the_shift(tmp_path):
    from scaffolds.transformer.acceptance import attention_test_paths
    from scaffolds.transformer.training import telemetry
    from scaffolds.transformer.training.diagnostics import attention_identity

    config = _training_config(tmp_path)
    identity = attention_identity(config)
    assert identity["attention_exp_shift"] == "row_max" and identity["attention_implementation"] == "blocked"
    assert "shifted by their causal maximum" in identity["attention_formula"]
    config.model.attention_backend = "fused"
    assert attention_identity(config)["attention_implementation"] == "pallas_triton_row_max_shifted"
    config.model.attention_exp_shift = "none"
    assert attention_identity(config)["attention_implementation"] == "pallas_triton"
    assert "attention_exp_shift" not in attention_identity(config)
    shifted = config_to_dict(_training_config(tmp_path))
    assert "scaffolds/transformer/tests/test_exp_row_max_shift.py" in attention_test_paths(shifted, "blocked", True)
    assert "scaffolds/transformer/tests/test_exp_row_max_shift.py" not in attention_test_paths(
        config_to_dict(config), "fused", True)
    # Telemetry: the RMS statistics of the shifted output, the gain of the unshifted one, and the largest shift.
    rng = np.random.default_rng(3)
    y = jnp.asarray(rng.standard_normal((2, 5, 3, 4)), jnp.float32)
    shift = jnp.asarray(rng.uniform(-3, 3, (2, 5, 3)), jnp.float32)
    with telemetry.collect() as shifted_collector:
        telemetry.record_attention_output(y, head_rms_norm_eps=EPS, output_shift=shift)
    with telemetry.collect() as unshifted_collector:
        telemetry.record_attention_output(y * jnp.exp(shift)[..., None], head_rms_norm_eps=EPS)
    values = shifted_collector.values
    assert set(values) == {"attention_output_rms", "attention_output_head_rms_min", "head_rms_norm_gain_max",
                           "attention_exp_shift_max"}
    assert shifted_collector.reductions["attention_exp_shift_max"] == "max"
    np.testing.assert_allclose(values["head_rms_norm_gain_max"], unshifted_collector.values["head_rms_norm_gain_max"],
                               rtol=1e-5)
    np.testing.assert_allclose(values["attention_output_rms"], jnp.mean(jnp.square(y)), rtol=1e-6)
    assert float(values["attention_exp_shift_max"]) == float(jnp.max(shift))
    with telemetry.collect() as overflow:
        telemetry.record_attention_output(y, head_rms_norm_eps=EPS, output_shift=jnp.full((2, 5, 3), 500.0))
    assert float(overflow.values["head_rms_norm_gain_max"]) == 0.0


@pytest.mark.parametrize("implementation", ("scan", "strips"))
def test_training_resumes_exactly_and_reports_the_shifted_kernel(tmp_path, implementation):
    from dataclasses import replace

    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
    from scaffolds.transformer.training.state import create_state

    config = _training_config(tmp_path / "full", implementation)
    loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
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
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["attention_exp_shift"] == "row_max"
    measurements = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    reports = [row["attention_diagnostics"] for row in measurements if "attention_diagnostics" in row]
    assert reports and all(report["valid"] and report["kernel_values"] == "row_max_shifted_exp" for report in reports)
    assert all("exp_shift_max" in layer for report in reports for layer in report["layers"])


@pytest.mark.gpu
@pytest.mark.parametrize("decay", (False, True), ids=("plain", "cumulative_decay"))
@pytest.mark.parametrize("scale", (1.0, 8.0))
def test_gpu_strips_and_scan_at_production_precision_match_fp64(scale, decay):
    """On the GPU, BF16 Q/K/V at 2,048 tokens (Q/K width 16, value width 64), at ordinary scores and at row maxima
    beyond FP32's e^88.7: the causal strips train what the scan computes (output, shifts and VJP), and both match
    the FP64 function on a 129-token, one-head prefix, with and without a cumulative forget-gate decay."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation; the strips' production precision is under test")
    rng = np.random.default_rng(17)
    q, k = (jnp.asarray(scale * 0.7 * rng.standard_normal((1, 2048, 2, 16)), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((1, 2048, 2, 64)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.standard_normal(v.shape), jnp.bfloat16)
    log_decay = jnp.asarray(np.cumsum(-rng.uniform(0.001, 0.01, (1, 2048, 2)), axis=1), jnp.float32)
    arguments = {"query_log_decay": log_decay} if decay else {}

    def run(attend, q, k, v, cotangent, **extra):
        def function(q, k, v):
            return attend(q, k, v, **SHIFTED, **extra)

        (output, shift), pullback = jax.vjp(function, q, k, v)
        return output, shift, pullback((cotangent, jnp.zeros_like(shift)))

    scan = jax.jit(partial(run, blocked_dot_product_attention))(q, k, v, cotangent, **arguments)
    strips = jax.jit(partial(run, strip_dot_product_attention))(q, k, v, cotangent, **arguments)
    assert (float(jnp.max(scan[1])) > 89) == (scale > 1)
    # The same FP32 logits up to the order of their sums.
    np.testing.assert_allclose(np.asarray(strips[1]), np.asarray(scan[1]), rtol=1e-5, atol=1e-5)
    _assert_close((strips[0], strips[2]), (scan[0], scan[2]), rtol=2e-2, label=f"strips vs scan x{scale}")
    prefix = (q[:, :129, :1], k[:, :129, :1], v[:, :129, :1], cotangent[:, :129, :1])
    prefix_decay = {"query_log_decay": log_decay[:, :129, :1]} if decay else {}
    with jax.enable_x64():
        operands = tuple(jnp.asarray(np.asarray(array, np.float64)) for array in prefix)

        def dense(q, k, v):
            logits, causal = _logits(q, k, log_decay=jnp.asarray(np.asarray(log_decay[:, :129, :1], np.float64))
                                     if decay else None)
            maximum = jax.lax.stop_gradient(jnp.max(jnp.where(causal, logits, -jnp.inf), axis=-1, keepdims=True))
            weights = jnp.where(causal, jnp.exp(jnp.where(causal, logits - maximum, 0.0)), 0.0)
            return jnp.einsum("bhts,bshd->bthd", weights, v), jnp.swapaxes(maximum[..., 0], 1, 2)

        (output, shift), pullback = jax.vjp(dense, *operands[:3])
        expected = (np.asarray(output), np.asarray(shift),
                    [np.asarray(g) for g in pullback((operands[3], jnp.zeros_like(shift)))])
    for label, result in (("scan", run(blocked_dot_product_attention, *prefix, **prefix_decay)),
                          ("strips", run(strip_dot_product_attention, *prefix, **prefix_decay))):
        np.testing.assert_allclose(np.asarray(result[1], np.float64), expected[1], rtol=1e-5,
                                   atol=1e-5 * max(1.0, float(np.max(np.abs(expected[1])))), err_msg=label)
        _assert_close((result[0], result[2]), (expected[0], expected[2]), rtol=2e-2, label=f"{label} vs FP64")
