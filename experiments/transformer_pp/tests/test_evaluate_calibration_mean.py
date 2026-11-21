"""Compare all-layer calibration means with explicit single-sequence forwards."""
from functools import partial

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.transformer_pp.evaluate_calibration_mean import evaluate, sequence_nll
from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.models import build_model


@pytest.fixture(params=["direct", "sum_of_squares", "softplus_prefactor"])
def model(request):
    prefactor = request.param == "softplus_prefactor"
    form = "sum_of_squares" if prefactor else request.param
    config = ModelConfig(vocab_size=32, max_seq_len=6, num_layers=3, hidden_size=8, num_heads=2,
                         num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
                         attention_degree=9 if prefactor else 8, attention_normalization=False, attention_coefficients="learned",
                         attention_score_transform="softplus" if prefactor else "identity",
                         attention_coefficient_input_power=int(prefactor),
                         attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                         attention_coefficient_activation="tanh", attention_coefficient_parameterization=form,
                         attention_coefficient_scale="factorial" if form == "direct" else "none",
                         attention_coefficient_amplitude_initial=0., attention_coefficient_projection_initial_std=.3,
                         attention_backend="blocked", attention_output_scaling="head_rms_norm",
                         initialization="transformer_reference", qk_norm=True)
    model = build_model(config, 7)
    model.query_polynomial_amplitude[...] = jnp.full_like(model.query_polynomial_amplitude[...], .8)
    return model


def test_calibration_mean_matches_individual_sequence_forwards(model):
    rows = np.random.default_rng(20).integers(0, 32, (8, 7), dtype=np.int32)
    calibration, recipients = rows[:4], rows[4:]
    graph, state = nnx.split(model, nnx.Param)
    order, (native, intervention) = evaluate(graph, state, calibration, recipients, microbatch=2, sequence_seed=42)
    assert native["condition"] == "native" and intervention["condition"] == "calibration_mean"
    assert native["layer_scope"] == intervention["layer_scope"] == "all_layers"
    np.testing.assert_array_equal(order, np.random.default_rng(42).permutation(len(recipients)))

    # Each calibration forward contributes every token to each layer/head's mean.
    domain = model.config.attention_score_transform
    forward = partial(model, coefficient_score_transform=domain)
    tables = [forward(row[None, :-1], return_coefficients=True)[1] for row in calibration]
    mean = np.concatenate(tables, axis=1).mean(axis=(1, 2), keepdims=True)
    assert np.max(np.std(np.concatenate(tables, axis=1), axis=(1, 2))) > 1e-3
    if model.config.attention_coefficient_input_power:
        assert mean.shape[-1] == 10
        np.testing.assert_array_equal(mean[..., 0], 0.)
        with pytest.raises(ValueError, match="score transform"):
            model(recipients[:, :-1], return_coefficients=True)
    expected_native, expected_mean = [], []
    for row in recipients[order]:
        batch = row[None, :]
        inputs = batch[:, :-1]
        fixed = jnp.broadcast_to(mean, tables[0].shape)
        expected_native.append(float(sequence_nll(model(inputs), batch)[0]))
        expected_mean.append(float(sequence_nll(forward(inputs, query_coefficients=fixed), batch)[0]))
    np.testing.assert_allclose(native["sequence_nll"], expected_native, atol=3e-6)
    np.testing.assert_allclose(native["identity_sequence_nll"], expected_native, atol=3e-6)
    np.testing.assert_allclose(intervention["sequence_nll"], expected_mean, atol=3e-6)
    delta = (np.array(expected_mean) - expected_native).reshape(-1, 2).mean(axis=1)
    assert np.max(np.abs(delta)) > 1e-6
    np.testing.assert_allclose(intervention["delta_nll"], delta.mean(), atol=3e-6)
    np.testing.assert_allclose(intervention["paired_se"], delta.std(ddof=1) / np.sqrt(len(delta)), atol=3e-6)
