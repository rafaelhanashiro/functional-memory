from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.training.loop import accumulated_train_step
from scaffolds.transformer.training.query_dynamics import QueryDynamics, make_probe, measure_batch
from scaffolds.transformer.training.state import create_state


@pytest.mark.parametrize("static", [False, True])
def test_query_cancellation_and_actual_updates(tmp_path, static):
    config = Config()
    config.train.workdir = tmp_path
    config.model = ModelConfig(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2,
                               num_kv_heads=2, qk_head_dim=4, value_head_dim=8, intermediate_size=32,
                               attention_degree=8, attention_normalization=False, attention_coefficients="learned",
                               attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                               attention_coefficient_activation="tanh", attention_coefficient_scale="factorial",
                               attention_coefficient_amplitude_initial=0, attention_coefficient_projection_initial_std=.25,
                               attention_coefficient_amplitude_scale="factorial", attention_backend="blocked",
                               initialization="transformer_reference", attention_output_scaling="head_rms_norm")
    if static:
        config.model = replace(config.model, attention_coefficient_conditioning="static",
                               attention_coefficient_amplitude_scale="none", attention_coefficient_activation="identity",
                               attention_coefficient_amplitude_initial=1., attention_coefficient_projection_initial_std=0.)
    config.train.dtype = "float32"
    config.data = replace(config.data, synthetic_vocab_size=32, sequence_length=9)
    rng = jax.random.key(42)
    state = create_state(config, rng)
    batches = jax.random.randint(rng, (2, 4, 9), 0, 32)
    rngs = jax.random.split(rng, 2)
    updated, loss, coords = accumulated_train_step(state, batches, rngs, query_gradient_dynamics=True)
    baseline, plain_loss = accumulated_train_step(state, batches, rngs)
    np.testing.assert_allclose(loss, plain_loss, atol=1.e-6)
    for x, y in zip(jax.tree.leaves(updated.params), jax.tree.leaves(baseline.params), strict=True):
        np.testing.assert_allclose(x, y, atol=2.e-6)
    assert coords.shape == (2 if static else 6, 2, 2, 9)
    assert static or np.all(np.asarray(coords[4]) == 0)  # a=0 on the first update.
    offset = "polynomial_coefficients" if static else "query_polynomial_bias"
    np.testing.assert_allclose(coords[1], updated.params[offset][...] - state.params[offset][...], atol=1.e-7)
    probe = make_probe(state.graphdef, config.model)
    measured = measure_batch(probe, state.params, batches)
    np.testing.assert_allclose(measured["rows_bias_key_bias_gradient"].sum(-1), measured["rows_bias_bias"],
                               atol=2.e-6, rtol=1.e-4)
    np.testing.assert_allclose(measured["rows_bias_moment_dot_h"], measured["rows_bias_bias"], atol=2.e-6)
    z, g, scale = [measured["rows_bias_"+name] for name in ("z", "g", "gain")]
    variance = np.mean(z*z, axis=-1, keepdims=True) + config.model.rms_norm_eps
    h = (g*scale - z * np.mean(z*g*scale, axis=-1, keepdims=True)/variance) / np.sqrt(variance)
    np.testing.assert_allclose(h, measured["rows_bias_h"], atol=2.e-5, rtol=.0002)
    np.testing.assert_allclose(z, measured["rows_bias_reconstructed_z"], atol=2.e-5, rtol=.0002)
    assert np.all(measured["bias_top8_abs"] <= measured["bias_abs"] + 1.e-6)
    for role, i in ([("bias", 0)] if static else [("bias", 0), ("amplitude", 2)]):
        np.testing.assert_allclose(measured[role+"_signed"], coords[i], rtol=.002, atol=2.e-6)
        assert np.all(np.abs(measured[role+"_signed"]) <= measured[role+"_abs"] + 2.e-6)
        assert np.any(measured[role+"_abs"] > np.abs(measured[role+"_signed"]) + 1.e-6)
    # A restart drops measurements beyond the restored boundary, including complete chunks.
    recorder = QueryDynamics(config, state.graphdef, 0, None)
    recorder.observe(2, coords, state.params, updated.params, batches, gradient_norm=3., clip_scale=1./3,
                     checkpoint=True, evaluation=True, event=False)
    with np.load(tmp_path / "query_dynamics_updates_000002.npz") as archive:
        np.testing.assert_array_equal(archive["bias_grad"][0], coords[0])
        np.testing.assert_array_equal(archive["global_gradient_norm"], [3.])
        np.testing.assert_allclose(archive["clip_scale"], [1./3])
    QueryDynamics(config, state.graphdef, 1, None)
    assert not list(tmp_path.glob("query_dynamics_*.npz"))
