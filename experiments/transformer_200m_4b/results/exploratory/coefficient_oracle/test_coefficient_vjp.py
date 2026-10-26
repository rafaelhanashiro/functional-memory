"""Small CPU checks of the independent diagnostic formulas and native hook."""
from dataclasses import replace
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from scaffolds.transformer.attention import multihead
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_training import tiny_config
from experiments.transformer_200m_4b.results.exploratory.coefficient_oracle.inspect_coefficient_vjp import (
    basis_outputs, make_capture, rms_jacobian, tangent_geometry,
)


def test_independent_coefficient_vjp_and_tangent_geometry():
    rng = np.random.default_rng(7801)
    q, k, v = (rng.normal(size=(17, 8)) for _ in range(3))
    positions = np.asarray([0, 7, 16])
    matrices, _, _ = basis_outputs(q, k, v, positions, 4)
    gain = rng.uniform(.5, 1.5, size=8)
    for matrix in matrices:
        coordinates = rng.normal(size=5)
        cotangent = rng.normal(size=8)
        eps = 1e-6
        def output(u):
            y = matrix @ u
            return gain * y / np.sqrt(np.mean(y ** 2) + eps)
        jacobian = rms_jacobian(matrix @ coordinates, gain, eps) @ matrix
        step = 1e-5
        numerical = np.stack([(output(coordinates + step * unit) - output(coordinates - step * unit)) / (2 * step)
                              for unit in np.eye(5)], axis=1)
        np.testing.assert_allclose(jacobian, numerical, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(cotangent @ jacobian, cotangent @ numerical, rtol=1e-6, atol=1e-8)
        geometry = tangent_geometry(matrix, coordinates, matrix @ coordinates, gain, eps)
        y = matrix @ coordinates
        radial = gain * y * eps / (np.mean(y ** 2) + eps) ** 1.5
        np.testing.assert_allclose(geometry["log_radius_response_norm"], np.linalg.norm(radial), rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(geometry["direction_tangent_singular_values"],
                                   np.linalg.norm(coordinates) * np.asarray(geometry["coefficient_tangent_singular_values"]))
    # Query zero uses exactly one key, so basis-output directions have rank one.
    assert np.linalg.matrix_rank(matrices[0]) == 1


def test_reference_retains_absolute_causal_positions():
    q, k, v = np.ones((5, 3)), np.ones((5, 3)), np.arange(15).reshape(5, 3)
    matrices, _, mask = basis_outputs(q, k, v, np.asarray([1, 4]), 4)
    np.testing.assert_array_equal(mask.sum(axis=1), [2, 5])
    np.testing.assert_allclose(matrices[:, :, 0], [v[:2].sum(axis=0), v.sum(axis=0)])


def test_native_zero_probes_preserve_forward_and_capture_task_cotangent(tmp_path):
    config = tiny_config(tmp_path)
    config.model = replace(config.model, num_layers=2, attention_degree=4, attention_backend="blocked",
        attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_projection_initial_std=.2, attention_coefficient_amplitude_initial=.25,
        attention_coefficient_sharing="per_head", attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        initialization="transformer_truncated_reference")
    model = build_model(config.model, 7)
    # Exercise exactly the tracing hooks while substituting the CPU reference
    # for the GPU primitive; this does not pretend to qualify the GPU kernel.
    for name in model.layer_names:
        getattr(model, name).attention.backend = "mosaic"
    graph, params = nnx.split(model, nnx.Param)
    tokens = jnp.arange(5)[None] % 16
    probes = tuple(jnp.zeros((1, 4, config.model.num_heads, config.model.value_head_dim), jnp.float32)
                   for _ in model.layer_names)
    original_core = multihead.attention_core
    def reference_core(q, k, v, **kwargs):
        return original_core(q, k, v, **{**kwargs, "backend": "blocked"})
    with patch.object(multihead, "attention_core", reference_core):
        capture, compare = make_capture(graph, config.model)
        (loss, (captured, gains, logits)), cotangents = capture(probes, params, tokens)
        parity = jax.device_get(compare(params, tokens, logits))
        assert float(parity["logit_max_absolute_error"]) == 0.
        assert len(captured) == len(gains) == len(cotangents) == 2
        assert all(float(jnp.linalg.norm(value)) > 0 for value in cotangents)
        direction = np.zeros(probes[0].shape, np.float32)
        location = np.unravel_index(np.argmax(np.abs(np.asarray(cotangents[0]))), direction.shape)
        direction[location] = 1.
        original_scaling = multihead.scale_attention_output
        def perturbed_loss(delta):
            index = 0
            def scaling(y, mode, **kwargs):
                nonlocal index
                output = original_scaling(y, mode, **kwargs)
                if index == 0:
                    output = output + delta * jnp.asarray(direction)
                index += 1
                return output
            with patch.object(multihead, "scale_attention_output", scaling):
                output = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
            return optax.softmax_cross_entropy_with_integer_labels(output, tokens[:, 1:]).mean()
        objective = jax.jit(perturbed_loss)
        step = 2e-3
        numerical = (objective(step) - objective(-step)) / (2 * step)
        np.testing.assert_allclose(numerical, np.asarray(cotangents[0])[location], rtol=.04, atol=2e-5)
        np.testing.assert_allclose(objective(0.), loss, atol=1e-6)
