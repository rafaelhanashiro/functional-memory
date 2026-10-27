"""Distillation observes real teacher inputs and deploys an unchanged backbone."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_distillation_model import (
    build_dense_student, make_student_losses, make_teacher_functions,
    pack_generator, predict_coefficients, replace_generator,
)
from scaffolds.transformer.training.coefficient_fit import split_generator_parameters


def parent():
    config = Config(model=ModelConfig(
        vocab_size=24, max_seq_len=8, num_layers=2, hidden_size=8,
        num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=4,
        intermediate_size=12, attention_degree=4, attention_backend="blocked",
        attention_normalization=False, attention_coefficients="fixed",
        attention_output_scaling="head_rms_norm", qk_norm=True,
        qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        initialization="transformer_reference"))
    config.train.dtype = "float32"
    config.train.seed = 17
    model = build_model(config.model, nnx.Rngs(params=17, dropout=17))
    graph, params = nnx.split(model, nnx.Param)
    return config, graph, params


def test_taylor_embedding_preserves_backbone_and_native_predictions():
    config, parent_graph, params = parent()
    student_config, graph, student, audit = build_dense_student(
        config, params, feature_count=9, readout_scale=1 / 3)
    assert audit["feature_count"] == 9
    assert audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]
    assert state_sha256(params) == state_sha256(split_generator_parameters(student)[1])
    assert config.model.attention_coefficients == "fixed"
    assert student_config.model.qk_norm_head_scale
    tokens = jnp.arange(27, dtype=jnp.int32).reshape(3, 9) % 24
    old_nll = make_student_losses(parent_graph)(params, tokens)
    new_nll = make_student_losses(graph)(student, tokens)
    np.testing.assert_allclose(new_nll, old_nll, rtol=1e-6, atol=1e-6)
    coefficients = predict_coefficients(pack_generator(student), jnp.ones((2, 11, 8)),
                                        readout_scale=1 / 3)
    np.testing.assert_allclose(coefficients, np.broadcast_to(
        np.asarray([1 / math.factorial(r) for r in range(5)]), coefficients.shape), atol=1e-8)


@pytest.mark.parametrize("feature_count,tanh_bias", [(5, False), (9, True)])
def test_packed_predictor_matches_native_capture_and_replacement_preserves_backbone(feature_count, tanh_bias):
    config, _, parent_params = parent()
    _, graph, params, _ = build_dense_student(
        config, parent_params, feature_count=feature_count, readout_scale=1 / 3,
        tanh_bias=tanh_bias)
    packed = pack_generator(params)
    rng = np.random.default_rng(41)
    packed = {key: jnp.asarray(rng.normal(size=value.shape) * .3, jnp.float32)
              for key, value in packed.items()}
    packed["bias"] += 1
    updated = replace_generator(params, packed)
    assert state_sha256(split_generator_parameters(updated)[1]) == state_sha256(parent_params)
    tokens = jnp.arange(27, dtype=jnp.int32).reshape(3, 9) % 24

    @jax.jit
    def native_capture(state, tokens):
        return nnx.merge(graph, state)(tokens[:, :-1], deterministic=True, return_coefficients=True)

    _, actual = native_capture(updated, tokens)
    capture, native_losses = make_teacher_functions(graph)
    captured = capture(updated, tokens, actual.swapaxes(0, 1))
    x = captured["inputs"].transpose(1, 0, 2, 3).reshape(2, -1, 8)
    predicted = jax.jit(lambda p, h: predict_coefficients(p, h, readout_scale=1 / 3))(packed, x)
    expected = actual.reshape(2, -1, 2, 5)
    np.testing.assert_allclose(predicted, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(captured["nll"], native_losses(updated, tokens, actual.swapaxes(0, 1)),
                               rtol=2e-6, atol=2e-6)


def test_capture_uses_teacher_activations_and_matches_native_override():
    config, graph, params = parent()
    tokens = jnp.arange(18, dtype=jnp.int32).reshape(2, 9) % 24
    taylor = jnp.broadcast_to(jnp.asarray([1 / math.factorial(r) for r in range(5)], jnp.float32),
                              (2, 2, 8, 2, 5))
    coefficients = taylor.at[:, 0, :, :, 1].multiply(-2.)
    capture, native_losses = make_teacher_functions(graph)
    original = capture(params, tokens, taylor)
    result = capture(params, tokens, coefficients)
    assert result["inputs"].shape == (2, 2, 8, 8)
    # No earlier coefficients affect layer zero; the changed first layer must
    # actually feed the recorded second-layer input.
    np.testing.assert_array_equal(result["inputs"][:, 0], original["inputs"][:, 0])
    assert np.max(np.abs(np.asarray(result["inputs"][:, 1] - original["inputs"][:, 1]))) > 1e-4
    np.testing.assert_allclose(result["nll"], native_losses(params, tokens, coefficients),
                               rtol=1e-6, atol=1e-6)
    assert float(result["joint_logit_max_absolute_error"]) == 0
    assert float(result["joint_logit_relative_l2_error"]) == 0


def test_student_rejects_non_taylor_parent():
    config, _, params = parent()
    config.model = replace(config.model, attention_normalization=True)
    with pytest.raises(ValueError, match="Taylor"):
        build_dense_student(config, params)
