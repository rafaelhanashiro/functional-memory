"""Raw unit-L2 generators composed with softplus times query SoS.

Independent direct-square references check the shared residual's two gradient
paths. The generator bypasses RMSNorm; Q/K/V keep its learned channel gains.
"""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import input_dependent, multihead
from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients, taylor_square_factors
from scaffolds.transformer.attention.parameters import attention_parameter_specs, parameters_for_layer
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_coefficient_input_normalization import EPS, _assert_generator_gradients, _unit_numpy
from scaffolds.transformer.tests.test_coefficient_raw_input import _rms
from scaffolds.transformer.tests.test_softplus_sos import _config as _sos_config

CORES = (8, 12)


def _config(core_degree, *, normalization="l2", source="pre_rmsnorm", **changes):
    return _sos_config(1, core_degree, num_layers=1,
                       attention_coefficient_input_normalization=normalization,
                       attention_coefficient_input_source=source, **changes)


def _parameters(model):
    return parameters_for_layer(model, model.attention_parameter_specs, 0)


def _set_generator(model, core_degree):
    count = 2 * (core_degree + 1)
    model.query_polynomial_amplitude[...] = jnp.linspace(-.7, .9, count).reshape(1, 2, core_degree + 1)
    model.query_polynomial_bias[...] = jnp.linspace(.6, 1.3, count).reshape(1, 2, core_degree + 1)


@pytest.mark.parametrize("core_degree", CORES)
def test_raw_l2_preserves_parameters_and_changes_only_generator_inputs(core_degree, monkeypatch):
    configs = (_config(core_degree, normalization="none", source="post_rmsnorm"), _config(core_degree))
    assert attention_parameter_specs(configs[0].model) == attention_parameter_specs(configs[1].model)
    models = [build_model(c.model, 19) for c in configs]
    params = [dict(nnx.to_flat_state(nnx.state(model, nnx.Param))) for model in models]
    assert params[0].keys() == params[1].keys()
    for key in params[0]:
        np.testing.assert_array_equal(params[0][key][...], params[1][key][...])
    tokens = jnp.arange(16).reshape(2, 8) % 32
    np.testing.assert_array_equal(models[0](tokens), models[1](tokens))  # a=0 at initialization
    for model in models:
        _set_generator(model, core_degree)
        model.layers_0.attention_norm.scale[...] = jnp.linspace(.15, 2.1, 16)
    raw = jnp.asarray(np.random.default_rng(311).normal(size=(2, 8, 16)) * 3., jnp.float32)
    inputs, generated, operands = [], [], []
    project, attend = input_dependent.project_coefficients, multihead.attention_core

    def capture_projection(x, *args, **kwargs):
        inputs.append(np.asarray(x))
        result = project(x, *args, **kwargs)
        generated.append(np.asarray(result))
        return result

    def capture_attention(q, k, v, **kwargs):
        operands.append(tuple(np.asarray(value) for value in (q, k, v)))
        assert kwargs["score_transform"] == "softplus"
        assert kwargs["query_polynomial_coefficients"].shape[-1] == core_degree + 2
        np.testing.assert_array_equal(kwargs["query_polynomial_coefficients"][..., 0], 0.)
        return attend(q, k, v, **kwargs)

    monkeypatch.setattr(input_dependent, "project_coefficients", capture_projection)
    monkeypatch.setattr(multihead, "attention_core", capture_attention)
    outputs = [model.layers_0(raw, attention_parameters=_parameters(model)) for model in models]
    np.testing.assert_array_equal(inputs[0], models[0].layers_0.attention_norm(raw))
    np.testing.assert_allclose(inputs[1], _unit_numpy(raw), rtol=3e-6, atol=1e-7)
    assert generated[1].shape == (2, 8, 2, core_degree + 1)
    for first, second in zip(operands[0], operands[1], strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.allclose(generated[0], generated[1], rtol=1e-5, atol=1e-7)
    assert not np.allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)
    models[1].layers_0.attention_norm.scale[...] = jnp.linspace(2.6, 0., 16)
    models[1].layers_0(raw, attention_parameters=_parameters(models[1]))
    np.testing.assert_array_equal(inputs[1], inputs[2])
    np.testing.assert_array_equal(generated[1], generated[2])
    for first, second in zip(operands[1], operands[2], strict=True):
        assert not np.allclose(first, second, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("activation", ("tanh", "pure_tanh"))
def test_raw_l2_takes_the_exp_prefactor_of_an_input_power_zero_core(activation, monkeypatch):
    """e^s h_t(z): the softplus SoS core of input power zero under exp of the raw score takes raw-input L2 like
    z h_t(z) does; its parameters are the plain-input run's and only the generator's input changes. Without a
    prefactor the input-power-zero core stays refused."""
    core_degree = 8
    pure = dict(attention_coefficient_activation="pure_tanh", attention_coefficient_tanh_bias=True,
                attention_coefficient_tanh_bias_initial=math.atanh(.5), attention_coefficient_projection_initial_std=0.,
                attention_coefficient_amplitude_initial=1.) if activation == "pure_tanh" else {}
    configs = [_sos_config(0, core_degree, num_layers=1, attention_score_prefactor="exp", **pure,
                           attention_coefficient_input_normalization=normalization,
                           attention_coefficient_input_source=source)
               for normalization, source in (("none", "post_rmsnorm"), ("l2", "pre_rmsnorm"))]
    assert attention_parameter_specs(configs[0].model) == attention_parameter_specs(configs[1].model)
    models = [build_model(c.model, 19) for c in configs]
    params = [dict(nnx.to_flat_state(nnx.state(model, nnx.Param))) for model in models]
    assert params[0].keys() == params[1].keys()
    for key in params[0]:
        np.testing.assert_array_equal(params[0][key][...], params[1][key][...])
    raw = jnp.asarray(np.random.default_rng(313).normal(size=(2, 8, 16)) * 3., jnp.float32)
    inputs, prefactors = [], []
    project, attend = input_dependent.project_coefficients, multihead.attention_core

    def capture_projection(x, *args, **kwargs):
        inputs.append(np.asarray(x))
        return project(x, *args, **kwargs)

    def capture_attention(q, k, v, **kwargs):
        prefactors.append((kwargs["score_transform"], kwargs["score_prefactor"]))
        assert kwargs["query_polynomial_coefficients"].shape[-1] == core_degree + 1
        return attend(q, k, v, **kwargs)

    monkeypatch.setattr(input_dependent, "project_coefficients", capture_projection)
    monkeypatch.setattr(multihead, "attention_core", capture_attention)
    for model in models:
        model.layers_0(raw, attention_parameters=_parameters(model))
    np.testing.assert_array_equal(inputs[0], models[0].layers_0.attention_norm(raw))
    np.testing.assert_allclose(inputs[1], _unit_numpy(raw), rtol=3e-6, atol=1e-7)
    assert prefactors == [("softplus", "exp")] * 2
    with pytest.raises(ValueError, match="softplus SoS with an input or score prefactor"):
        _sos_config(0, core_degree, num_layers=1, **pure, attention_coefficient_input_normalization="l2",
                    attention_coefficient_input_source="pre_rmsnorm")


def _direct_squares(q, k, v, x, w, a, b, core_degree):
    """Independent FP64 attention: no production normalizer or polynomial expansion."""
    unit = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), EPS)
    generated = b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", unit, w))
    fixed = jnp.asarray(np.asarray(taylor_square_factors(core_degree), np.float32), x.dtype)
    factors = (fixed * (1. + generated) / 2.).transpose(0, 2, 1, 3)
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    z = jnp.logaddexp(0., scores)
    split = core_degree // 2 + 1
    u = sum(factors[..., r, None] * z ** r for r in range(split))
    vpoly = sum(factors[..., split + r, None] * z ** r for r in range(split - 1))
    weights = jnp.where(jnp.tril(jnp.ones((q.shape[1], q.shape[1]), bool)), z * (u * u + vpoly * vpoly), 0.)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


@pytest.mark.parametrize("core_degree", CORES)
def test_full_attention_and_shared_residual_vjp_match_direct_squares(core_degree):
    config = _config(core_degree, qk_norm=False, qk_norm_head_scale=False,
                     attention_output_scaling="none", positional_encoding="none")
    model = build_model(config.model, 23)
    attention, fixed = model.layers_0.attention, _parameters(model)
    rng = np.random.default_rng(313)
    count = core_degree + 1
    arrays = (rng.normal(size=(2, 8, 16)) * 3., np.linspace(.2, 1.7, 16),
              rng.normal(size=(2, 16, count)) * .5, rng.uniform(-.8, .9, size=(2, count)),
              rng.uniform(.6, 1.2, size=(2, count)))
    cotangent = rng.normal(size=(2, 8, 16))
    matrices = tuple(np.asarray(proj.kernel) for proj in
                     (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj))

    def production(x, gain, w, a, b):
        return attention(_rms(x, gain, config.model.rms_norm_eps), coefficient_source_input=x,
                         attention_parameters=fixed | dict(query_polynomial_kernel=w,
                         query_polynomial_amplitude=a, query_polynomial_bias=b))

    def reference_separate(qkv_x, coefficient_x, gain, w, a, b):
        h = qkv_x / jnp.sqrt(jnp.mean(qkv_x ** 2, axis=-1, keepdims=True) + config.model.rms_norm_eps) * gain
        q, k, v = ((h @ jnp.asarray(matrix)).reshape(2, 8, 2, 8) for matrix in matrices[:3])
        result = _direct_squares(q, k, v, coefficient_x, w, a, b, core_degree)
        return result.reshape(2, 8, 16) @ jnp.asarray(matrices[3])

    actual, backward = jax.vjp(jax.jit(production), *(jnp.asarray(value, jnp.float32) for value in arrays))
    gradients = backward(jnp.asarray(cotangent, jnp.float32))
    with jax.enable_x64():
        wide = tuple(jnp.asarray(value, jnp.float64) for value in arrays)
        expected, backward = jax.vjp(reference_separate, wide[0], wide[0], *wide[1:])
        split_gradients = backward(jnp.asarray(cotangent))
        expected_gradients = jax.tree.map(np.asarray, (split_gradients[0] + split_gradients[1], *split_gradients[2:]))
        expected = np.asarray(expected)
    assert np.linalg.norm(split_gradients[0]) > 0 and np.linalg.norm(split_gradients[1]) > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    for name, actual, expected in zip(("shared residual", "QKV-only RMS gain", "W", "a", "b"),
                                      gradients, expected_gradients, strict=True):
        assert np.linalg.norm(expected) > 0, name
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=3e-5, err_msg=name)


@pytest.mark.parametrize("core_degree", CORES)
def test_raw_l2_diagnostics_name_core_coordinates_and_native_routing(core_degree):
    from scaffolds.transformer.training.diagnostics import attention_identity, coefficient_input_report

    config = _config(core_degree)
    model = build_model(config.model, 29)
    _set_generator(model, core_degree)
    model.layers_0.attention_norm.scale[...] = jnp.linspace(.2, 2.1, 16)
    raw = jnp.asarray(np.random.default_rng(317).normal(size=(2, 8, 16)) * 7., jnp.float32)
    _, layer = model.layers_0(raw, attention_parameters=_parameters(model), return_diagnostics=True)
    report = coefficient_input_report([layer], config.model)
    identity = attention_identity(config)
    assert (report["core_degree"], report["total_degree"], report["input_power"]) == (core_degree, core_degree + 1, 1)
    assert report["coordinate_labels"] == [f"U{r}" for r in range(core_degree // 2 + 1)] + [f"V{r}" for r in range(core_degree // 2)]
    assert np.asarray(report["saturation_fraction"]).shape == (1, 2, core_degree + 1)
    assert report["source"] == "pre_rmsnorm" and report["finite"]
    assert identity["attention_coefficient_core_degree"] == core_degree
    assert "no RMSNorm channel gain" in identity["attention_formula"]
    np.testing.assert_allclose(report["input_norms"]["after"]["mean"], 1., rtol=3e-6)
    expected = np.einsum("btd,hdr->bthr", _unit_numpy(raw), np.asarray(_parameters(model)["query_polynomial_kernel"]))
    np.testing.assert_allclose(report["projection_summary"]["mean"][0], expected.mean(axis=(0, 1)), rtol=1e-5, atol=1e-7)
    mosaic = replace(config.model, attention_backend="mosaic", max_seq_len=2048, qk_head_dim=64, value_head_dim=64)
    assert mosaic.mosaic_score_transform_supported()


# z h_t(z) at both cores, and e^s h_t(z) (input power zero under exp of the raw score) at the degree the
# prefactor kernels are qualified to.
MOSAIC_FORMS = (*(pytest.param(core, 1, "none", id=f"z-core{core}") for core in CORES),
                pytest.param(8, 0, "exp", id="exp-core8"))


@pytest.mark.gpu
@pytest.mark.parametrize("core_degree,input_power,prefactor", MOSAIC_FORMS)
@pytest.mark.parametrize("precision", ("bf16", "bf16x3"))
def test_mosaic_raw_l2_softplus_sos_shared_branches_match_direct_fp64(core_degree, input_power, prefactor,
                                                                      precision):
    """Independent factors plus analytic unit-L2 VJP, including summed shared-input dx."""
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0"):
        pytest.skip("Requires the admitted H100/B200 transformed Mosaic backend")
    from scaffolds.transformer.attention.backends.mosaic.api import transformed_query_polynomial_attention
    from scaffolds.transformer.tests.test_fused_attention import _assert_relative
    from scaffolds.transformer.tests.test_mosaic_softplus_sos import _numpy_fp64

    rng = np.random.default_rng(331)
    raw = rng.normal(size=(1, 512, 16)).astype(np.float32) * 4.
    gain = np.linspace(.25, 1.6, 16, dtype=np.float32)
    matrices = tuple((rng.normal(size=(16, 128)) * scale / 4.).astype(np.float32) for scale in (.6, .6, .85))
    count = core_degree + 1
    generator = (rng.normal(size=(2, 16, count)).astype(np.float32) * .5,
                 rng.uniform(-.6, .8, size=(2, count)).astype(np.float32),
                 rng.uniform(.7, 1.2, size=(2, count)).astype(np.float32))
    arrays = (raw, gain, *matrices, *generator)
    cotangent = jnp.asarray(rng.normal(size=(1, 512, 2, 64)), jnp.bfloat16)
    rms_eps = 1e-6

    def qkv_branch(x, scale, *weights):
        h = _rms(x, scale, rms_eps)
        return tuple(jnp.matmul(h, weight, precision=jax.lax.Precision.HIGHEST)
                     .reshape(1, 512, 2, 64).astype(jnp.bfloat16) for weight in weights)

    def composed(x, scale, wq, wk, wv, w, a, b):
        q, k, v = qkv_branch(x, scale, wq, wk, wv)
        unit = input_dependent.normalize_coefficient_input(x, "l2")
        factors = input_dependent.project_coefficients(unit, w, b, amplitude=a, activation="tanh")
        coefficients = query_square_coefficients(factors, core_degree + input_power, input_power=input_power)
        return transformed_query_polynomial_attention(q, k, v, coefficients, transform="softplus",
                                                      prefactor=prefactor, precision=precision)

    def compute(*args):
        output, backward = jax.vjp(composed, *args)
        return output, backward(cotangent)

    operands = tuple(jnp.asarray(value) for value in arrays)
    actual, gradients = jax.jit(compute)(*operands)
    qkv = qkv_branch(*operands[:5])
    unit = _unit_numpy(raw)
    expected, attention_gradients = _numpy_fp64(input_power, (*qkv, unit, *operands[5:]), cotangent,
                                                prefactor=prefactor)
    with jax.enable_x64():
        wide = tuple(jnp.asarray(value, jnp.float64) for value in arrays)

        def reference_qkv(x, scale, *weights):
            h = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + rms_eps) * scale
            return tuple((h @ weight).reshape(1, 512, 2, 64) for weight in weights)

        reference_operands, qkv_backward = jax.vjp(reference_qkv, *wide[:5])
        for name, rounded, unrounded in zip("qkv", qkv, reference_operands, strict=True):
            _assert_relative(rounded, unrounded, rtol=.003, name="RMS projected " + name)
        expected_qkv = jax.tree.map(np.asarray, qkv_backward(tuple(
            jnp.asarray(value, jnp.bfloat16).astype(jnp.float64) for value in attention_gradients[:3])))
    norm = np.linalg.norm(raw.astype(np.float64), axis=-1, keepdims=True)
    upstream = attention_gradients[3]
    dx_generator = (upstream - unit * np.sum(unit * upstream, axis=-1, keepdims=True)) / norm
    assert np.linalg.norm(dx_generator) > 0 and np.linalg.norm(expected_qkv[0]) > 0
    expected_gradients = (expected_qkv[0] + dx_generator, *expected_qkv[1:], *attention_gradients[4:])
    _assert_relative(actual, expected, rtol=.006, name="raw-L2 softplus SoS output")
    assert all(value.dtype == jnp.float32 for value in gradients)
    # The assertion orders generator parameters as W,b,a; operands use W,a,b.
    _assert_generator_gradients((gradients[0], gradients[5], gradients[7], gradients[6]),
                                (expected_gradients[0], expected_gradients[5], expected_gradients[7], expected_gradients[6]), rtol=.025)
    _assert_relative(gradients[1][None, None, None, :], expected_gradients[1][None, None, None, :],
                     rtol=.025, name="QKV-only RMS gain")
    for name, got, want in zip("qkv", gradients[2:5], expected_gradients[2:5], strict=True):
        _assert_relative(got.reshape(1, 16, 2, 64), want.reshape(1, 16, 2, 64),
                         rtol=.025 if name != "v" else .015, name="raw-L2 softplus SoS dW" + name)
