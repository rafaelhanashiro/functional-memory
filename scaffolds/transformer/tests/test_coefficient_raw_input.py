"""Raw-residual unit-L2 coefficients: source routing and both attention branches' VJPs."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer.attention import input_dependent, multihead
from scaffolds.transformer.attention.parameters import attention_parameter_specs, parameters_for_layer
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_coefficient_input_normalization import (
    EPS, _assert_generator_gradients, _config, _factorials, _oracle, _production, _unit_numpy,
)

DEGREE = 12
PARAMETERIZATIONS = ("direct", "sum_of_squares")


def _raw_config(parameterization="direct", **changes):
    return _config(parameterization, degree=DEGREE,
                   attention_coefficient_input_source="pre_rmsnorm", **changes)


def _parameters(model):
    return parameters_for_layer(model, model.attention_parameter_specs, 0)


def _nonzero_generators(model):
    model.query_polynomial_amplitude[...] = jnp.linspace(-.7, .9, 26).reshape(1, 2, 13)
    model.query_polynomial_bias[...] = jnp.linspace(.6, 1.3, 26).reshape(1, 2, 13)


def _rms(x, gain, eps):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * gain


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_raw_source_preserves_initialization_and_block_routes_only_coefficient_input(monkeypatch, parameterization):
    configs = (_config(parameterization, degree=DEGREE), _raw_config(parameterization))
    assert attention_parameter_specs(configs[0].model) == attention_parameter_specs(configs[1].model)
    models = [build_model(config.model, 19) for config in configs]
    parameters = [dict(nnx.to_flat_state(nnx.state(model, nnx.Param))) for model in models]
    assert parameters[0].keys() == parameters[1].keys()
    for path in parameters[0]:
        np.testing.assert_array_equal(parameters[0][path][...], parameters[1][path][...])
    tokens = jnp.arange(16).reshape(2, 8) % 32
    # a=0 makes both source choices exactly the reference initialization.
    np.testing.assert_array_equal(models[0](tokens), models[1](tokens))
    for model in models:
        _nonzero_generators(model)
        model.layers_0.attention_norm.scale[...] = jnp.linspace(.15, 2.1, 16)
    raw = jnp.asarray(np.random.default_rng(109).normal(size=(2, 8, 16)) * 3., jnp.float32)
    projection_inputs, generated, operands = [], [], []
    project, core = input_dependent.project_coefficients, multihead.attention_core

    def capture_projection(x, *args, **kwargs):
        projection_inputs.append(np.asarray(x))
        result = project(x, *args, **kwargs)
        generated.append(np.asarray(result))
        return result

    def capture_core(q, k, v, **kwargs):
        operands.append(tuple(np.asarray(value) for value in (q, k, v)))
        return core(q, k, v, **kwargs)

    monkeypatch.setattr(input_dependent, "project_coefficients", capture_projection)
    monkeypatch.setattr(multihead, "attention_core", capture_core)
    outputs = [model.layers_0(raw, attention_parameters=_parameters(model)) for model in models]
    assert len(projection_inputs) == len(operands) == 2
    np.testing.assert_allclose(projection_inputs[0], _unit_numpy(models[0].layers_0.attention_norm(raw)),
                               rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(projection_inputs[1], _unit_numpy(raw), rtol=3e-6, atol=1e-7)
    for before, after in zip(operands[0], operands[1], strict=True):
        np.testing.assert_array_equal(before, after)
    assert not np.allclose(generated[0], generated[1], rtol=1e-5, atol=1e-7)
    assert not np.allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)
    # An anisotropic gain change, including a zero channel, must affect Q/K/V only.
    models[1].layers_0.attention_norm.scale[...] = jnp.linspace(2.6, 0., 16)
    models[1].layers_0(raw, attention_parameters=_parameters(models[1]))
    np.testing.assert_array_equal(projection_inputs[1], projection_inputs[2])
    np.testing.assert_array_equal(generated[1], generated[2])
    for before, after in zip(operands[1], operands[2], strict=True):
        assert not np.allclose(before, after, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_coefficient_only_vjp_ignores_rms_gain_and_matches_float64(parameterization):
    config = _raw_config(parameterization)
    model = build_model(config.model, 23)
    attention, fixed = model.layers_0.attention, _parameters(model)
    rng = np.random.default_rng(113)
    arrays = (rng.normal(size=(2, 8, 16)) * 3., np.linspace(.2, 2.1, 16),
              rng.normal(size=(2, 16, 13)) * .5, rng.uniform(.6, 1.2, size=(2, 13)),
              rng.uniform(-.8, .9, size=(2, 13)))
    cotangent = rng.normal(size=(2, 8, 2, 13)) * _factorials(DEGREE)

    def coefficients(x, gain, w, b, a):
        params = fixed | dict(query_polynomial_kernel=w, query_polynomial_bias=b,
                              query_polynomial_amplitude=a)
        _, result = attention(_rms(x, gain, config.model.rms_norm_eps), coefficient_source_input=x,
                              attention_parameters=params, return_coefficients=True)
        return result

    actual, backward = jax.vjp(jax.jit(coefficients), *(jnp.asarray(x, jnp.float32) for x in arrays))
    gradients = backward(jnp.asarray(cotangent, jnp.float32))
    with jax.enable_x64():
        wide = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        expected, pullback = jax.vjp(lambda x, w, b, a: _oracle(x, w, b, a, parameterization=parameterization),
                                     wide[0], *wide[2:])
        expected_gradients = jax.tree.map(np.asarray, pullback(jnp.asarray(cotangent)))
        expected = np.asarray(expected)
    np.testing.assert_allclose(np.asarray(actual) * _factorials(DEGREE), expected * _factorials(DEGREE),
                               rtol=5e-6, atol=2e-6)
    np.testing.assert_array_equal(gradients[1], 0.)
    for name, got, want in zip(("x", "W", "b", "a"), (gradients[0], *gradients[2:]), expected_gradients,
                               strict=True):
        assert np.linalg.norm(want) > 0, name
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5, err_msg=name)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_shared_residual_and_gain_vjp_match_independent_full_attention_float64(parameterization):
    # Remove QK/output normalization and RoPE only to make an independent complete attention
    # oracle small. The production Block test above exercises the canonical normalized path.
    config = _raw_config(parameterization, qk_norm=False, qk_norm_head_scale=False,
                         attention_output_scaling="none", positional_encoding="none")
    model = build_model(config.model, 29)
    attention, fixed = model.layers_0.attention, _parameters(model)
    rng = np.random.default_rng(127)
    arrays = (rng.normal(size=(2, 8, 16)) * 3., np.linspace(.2, 1.7, 16),
              rng.normal(size=(2, 16, 13)) * .5, rng.uniform(.6, 1.2, size=(2, 13)),
              rng.uniform(-.8, .9, size=(2, 13)))
    cotangent = rng.normal(size=(2, 8, 16))
    weights = tuple(np.asarray(proj.kernel) for proj in
                    (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj))

    def production(x, gain, w, b, a):
        return attention(_rms(x, gain, config.model.rms_norm_eps), coefficient_source_input=x,
                         attention_parameters=fixed | dict(query_polynomial_kernel=w,
                         query_polynomial_bias=b, query_polynomial_amplitude=a))

    def reference(x, gain, w, b, a):
        normalized = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + config.model.rms_norm_eps)
        h = normalized * gain
        q, k, v = ((h @ jnp.asarray(weight)).reshape(2, 8, 2, 8) for weight in weights[:3])
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(8)
        c = _oracle(x, w, b, a, parameterization=parameterization).transpose(0, 2, 1, 3)
        p = sum(c[..., r, None] * scores ** r for r in range(13))
        p = jnp.where(jnp.tril(jnp.ones((8, 8), dtype=bool)), p, 0.)
        y = jnp.einsum("bhts,bshd->bthd", p, v)
        return y.reshape(2, 8, 16) @ jnp.asarray(weights[3])

    actual, pullback = jax.vjp(jax.jit(production), *(jnp.asarray(x, jnp.float32) for x in arrays))
    gradients = pullback(jnp.asarray(cotangent, jnp.float32))
    with jax.enable_x64():
        expected, pullback = jax.vjp(reference, *(jnp.asarray(x, jnp.float64) for x in arrays))
        expected_gradients = jax.tree.map(np.asarray, pullback(jnp.asarray(cotangent)))
        expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    for name, got, want in zip(("shared residual", "RMS gain", "W", "b", "a"), gradients, expected_gradients,
                               strict=True):
        assert np.linalg.norm(want) > 0, name
        np.testing.assert_allclose(got, want, rtol=5e-4, atol=3e-5, err_msg=name)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
@pytest.mark.parametrize("normalization", ("none", "l2"))
def test_omitted_source_preserves_legacy_config_and_outputs(tmp_path, parameterization, normalization):
    current = config_to_dict(_config(parameterization, normalization, degree=DEGREE))
    legacy = {**current, "model": dict(current["model"])}
    del legacy["model"]["attention_coefficient_input_source"]
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(config_to_dict(_raw_config(parameterization)), legacy)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    loaded = load_config(path)
    assert loaded.model.attention_coefficient_input_source == "post_rmsnorm"
    explicit = _config(parameterization, normalization, degree=DEGREE,
                       attention_coefficient_input_source="post_rmsnorm")
    models = [build_model(config.model, 37) for config in (loaded, explicit)]
    for model in models:
        _nonzero_generators(model)
        model.layers_0.attention_norm.scale[...] = jnp.linspace(.2, 2.1, 16)
    np.testing.assert_array_equal(models[0](jnp.arange(8)[None]), models[1](jnp.arange(8)[None]))
    raw = config_to_dict(_raw_config(parameterization))
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).model.attention_coefficient_input_source == "pre_rmsnorm"
    mosaic = replace(_raw_config(parameterization).model, attention_backend="mosaic", max_seq_len=2048,
                     qk_head_dim=64, value_head_dim=64)
    assert mosaic.mosaic_supported()


@pytest.mark.parametrize("changes", (
    {"attention_coefficient_input_source": "raw"},
    {"attention_coefficient_input_source": True},
    {"attention_coefficient_input_normalization": "none"},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_tanh_bias": True},
    {"attention_score_transform": "softplus"},
    {"attention_forget_gate": True},
    {"attention_coefficient_readout": "dense"},
    {"attention_coefficient_gate": "sigmoid"},
))
def test_invalid_source_and_unsupported_raw_combinations_are_rejected(changes):
    options = dict(attention_coefficient_input_source="pre_rmsnorm") | changes
    with pytest.raises(ValueError):
        _config(degree=DEGREE, **options)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_mha_requires_unambiguous_source_and_overrides_can_bypass_generator(parameterization):
    model = build_model(_raw_config(parameterization).model, 41)
    attention, params = model.layers_0.attention, _parameters(model)
    x = jnp.ones((1, 8, 16))
    with pytest.raises(ValueError, match="coefficient_source_input"):
        attention(x, attention_parameters=params)
    with pytest.raises(ValueError, match="shape"):
        attention(x, attention_parameters=params, coefficient_source_input=x[..., :-1])
    post = build_model(_config(parameterization, degree=DEGREE).model, 41)
    with pytest.raises(ValueError, match="pre_rmsnorm"):
        post.layers_0.attention(x, attention_parameters=_parameters(post), coefficient_source_input=x)
    # Effective coefficients supplied by an analysis need no unused raw source,
    # whether their normal generator produces coefficients directly or through SoS.
    supplied = jnp.broadcast_to(jnp.asarray(1 / _factorials(DEGREE), jnp.float32), (1, 8, 2, 13))
    without = attention(x, attention_parameters=params, query_coefficients=supplied)
    with_source = attention(x, attention_parameters=params, query_coefficients=supplied,
                            coefficient_source_input=x * 3.)
    np.testing.assert_array_equal(without, with_source)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_diagnostic_source_formula_and_norms_describe_raw_activation(parameterization):
    from scaffolds.transformer.training.diagnostics import attention_identity, coefficient_input_report

    config = _raw_config(parameterization)
    model = build_model(config.model, 43)
    model.layers_0.attention_norm.scale[...] = jnp.linspace(.2, 2.1, 16)
    raw = jnp.asarray(np.random.default_rng(139).normal(size=(2, 8, 16)) * 7., jnp.float32)
    _, layer = model.layers_0(raw, attention_parameters=_parameters(model), return_diagnostics=True)
    report = coefficient_input_report([layer], config.model)
    identity = attention_identity(config)
    assert identity["attention_coefficient_input_source"] == report["source"] == "pre_rmsnorm"
    assert identity["attention_coefficient_input"] == "unit_l2_of_pre_attention_residual_activation"
    assert "no RMSNorm channel gain" in identity["attention_formula"]
    assert "Q/K/V retain RMSNorm(h_t)" in identity["attention_formula"]
    assert "raw attention input" in report["input_norm_definitions"]["before"]
    assert "no RMSNorm channel gain" in report["formula"]
    assert report["finite"] and report["normalization"] == "l2" and report["epsilon"] == EPS
    expected_labels = ([f"r{i}" for i in range(13)] if parameterization == "direct" else
                       [f"U{i}" for i in range(7)] + [f"V{i}" for i in range(6)])
    assert report["coordinate_labels"] == expected_labels
    for stat, operation in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        np.testing.assert_allclose(report["input_norms"]["before"][stat][0],
                                   operation(np.linalg.norm(np.asarray(raw, np.float64), axis=-1)), rtol=3e-6)
        np.testing.assert_allclose(report["input_norms"]["after"][stat][0],
                                   operation(np.linalg.norm(_unit_numpy(raw), axis=-1)), rtol=3e-6, atol=1e-7)
    expected_projection = np.einsum("btd,hdr->bthr", _unit_numpy(raw),
                                    np.asarray(_parameters(model)["query_polynomial_kernel"], np.float64))
    np.testing.assert_allclose(report["projection_summary"]["mean"][0], expected_projection.mean(axis=(0, 1)),
                               rtol=1e-5, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.parametrize("parameterization,precision", (
    ("direct", "bf16"), ("sum_of_squares", "bf16"),
    ("direct", "bf16x3"), ("sum_of_squares", "bf16x3"),
), ids=("direct-bf16", "sum_of_squares-bf16", "direct-bf16x3", "sum_of_squares-bf16x3"))
def test_mosaic_raw_coefficient_and_rms_qkv_branches_match_independent_fp64(parameterization, precision):
    """One raw residual feeds both branches: verify their summed dx and the QKV-only dGain."""
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires H100/B200/B300")
    from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
    from scaffolds.transformer.tests.test_fused_attention import _assert_relative
    from scaffolds.transformer.tests.test_mosaic_query import _numpy_fp64

    rng = np.random.default_rng(149)
    raw = rng.normal(size=(1, 512, 16)).astype(np.float32) * 4.
    gain = np.linspace(.25, 1.6, 16, dtype=np.float32)
    matrices = tuple((rng.normal(size=(16, 128)) * scale / 4.).astype(np.float32) for scale in (.6, .6, .85))
    generator = (rng.normal(size=(2, 16, 13)).astype(np.float32) * .5,
                 rng.uniform(.7, 1.2, size=(2, 13)).astype(np.float32),
                 rng.uniform(-.6, .8, size=(2, 13)).astype(np.float32))
    arrays = (raw, gain, *matrices, *generator)
    cotangent = jnp.asarray(rng.normal(size=(1, 512, 2, 64)), jnp.bfloat16)
    rms_eps = 1e-6

    def qkv_branch(x, scale, *weights):
        h = _rms(x, scale, rms_eps)
        return tuple(jnp.matmul(h, weight, precision=jax.lax.Precision.HIGHEST)
                     .reshape(1, 512, 2, 64).astype(jnp.bfloat16) for weight in weights)

    def composed(x, scale, wq, wk, wv, w, b, a):
        q, k, v = qkv_branch(x, scale, wq, wk, wv)
        coefficients = _production(x, w, b, a, parameterization=parameterization)
        return query_polynomial_attention(q, k, v, coefficients, precision=precision)

    def compute(*args):
        output, backward = jax.vjp(composed, *args)
        return output, backward(cotangent)

    operands = tuple(jnp.asarray(value) for value in arrays)
    actual, gradients = jax.jit(compute)(*operands)
    # The kernel oracle is evaluated at the actual BF16 boundary operands. Their
    # independent FP64 pre-cast values and the entire pre-cast VJP are checked below.
    qkv = qkv_branch(*operands[:5])
    with jax.enable_x64():
        wide = tuple(jnp.asarray(value, jnp.float64) for value in arrays)

        def reference_qkv(x, scale, *weights):
            h = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + rms_eps) * scale
            return tuple((h @ weight).reshape(1, 512, 2, 64) for weight in weights)

        reference_operands, qkv_backward = jax.vjp(reference_qkv, *wide[:5])
        for name, rounded, unrounded in zip("qkv", qkv, reference_operands, strict=True):
            _assert_relative(rounded, unrounded, rtol=.003, name="RMS projected " + name)
        coefficients, coefficient_backward = jax.vjp(
            lambda x, w, b, a: _oracle(x, w, b, a, parameterization=parameterization), wide[0], *wide[5:])
        expected, attention_gradients = _numpy_fp64((*qkv, np.asarray(coefficients)), cotangent)
        # Cast VJPs pass through the representable BF16 kernel gradients, just as in training.
        expected_qkv = qkv_backward(tuple(jnp.asarray(value, jnp.bfloat16).astype(jnp.float64)
                                          for value in attention_gradients[:3]))
        expected_generator = coefficient_backward(jnp.asarray(attention_gradients[3]))
        expected_gradients = jax.tree.map(np.asarray, (expected_qkv[0] + expected_generator[0],
                                                     *expected_qkv[1:], *expected_generator[1:]))
    _assert_relative(actual, expected, rtol=.006, name="raw/RMS composition output")
    assert all(value.dtype == jnp.float32 for value in gradients)
    # Existing generator tolerances remain unchanged; each W/b/a coordinate is checked separately.
    _assert_generator_gradients((gradients[0], *gradients[5:]),
                                (expected_gradients[0], *expected_gradients[5:]), rtol=.025)
    _assert_relative(gradients[1][None, None, None, :], expected_gradients[1][None, None, None, :],
                     rtol=.025, name="raw/RMS composition dGain")
    for name, got, want in zip("qkv", gradients[2:5], expected_gradients[2:5], strict=True):
        _assert_relative(got.reshape(1, 16, 2, 64), want.reshape(1, 16, 2, 64),
                         rtol=.025 if name != "v" else .015, name="raw/RMS composition dW" + name)
