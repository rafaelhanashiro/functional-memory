"""Check the layerwise measurement against explicit single-recipient forwards."""
from functools import partial

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.transformer_pp.evaluate_layerwise_polynomials import evaluate, sequence_nll
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


def test_layerwise_losses_against_individual_donor_and_mean_forwards(model):
    config = model.config
    rows = np.random.default_rng(20).integers(0, 32, (8, 7), dtype=np.int32)
    calibration, recipients = rows[:4], rows[4:]
    graph, state = nnx.split(model, nnx.Param)
    order, metrics = evaluate(graph, state, calibration, recipients, microbatch=2, pairing_seed=42)
    np.testing.assert_array_equal(np.sort(order), np.arange(4))
    assert len(metrics) == 1 + 2 * config.num_layers
    native = np.asarray(sequence_nll(model(recipients[order, :-1]), recipients[order]))
    np.testing.assert_allclose(metrics[0]["sequence_nll"], native, atol=2e-6)
    identity = np.asarray(metrics[0]["identity_sequence_nll_by_layer"])
    np.testing.assert_allclose(identity, np.broadcast_to(native, (config.num_layers, len(order))), atol=2e-6)
    assert metrics[0]["joint_replay_max_abs_sequence_nll_error"] <= 1e-5
    assert metrics[0]["external_replay_atol"] == 2e-4
    forward = partial(model, coefficient_score_transform=config.attention_score_transform)
    _, calibration_coeff = forward(calibration[:, :-1], return_coefficients=True)
    _, donor_coeff = forward(recipients[:, :-1], return_coefficients=True)
    if config.attention_coefficient_input_power:
        assert donor_coeff.shape[-1] == 10
        np.testing.assert_array_equal(donor_coeff[..., 0], 0.)
        with pytest.raises(ValueError, match="score transform"):
            model(recipients[:, :-1], return_coefficients=True)
    mean = calibration_coeff.mean(axis=(1, 2))
    for record in metrics[1:]:
        layer = record["layer_index"]
        expected = []
        for offset, row in enumerate(order):
            own_row = recipients[row:row + 1]
            donor = order[offset ^ 1]
            assert row != donor
            replacement = donor_coeff[layer, donor:donor + 1]
            if record["condition"] == "calibration_mean":
                replacement = jnp.broadcast_to(mean[layer], replacement.shape)
            logits, actual = forward(own_row[:, :-1], query_coefficients=replacement,
                                   coefficient_override_layer=layer, return_coefficients=True)
            np.testing.assert_allclose(actual[:layer, 0], donor_coeff[:layer, row], atol=2e-6)
            np.testing.assert_array_equal(actual[layer], replacement)
            if layer < config.num_layers - 1:
                assert not np.allclose(actual[layer + 1, 0], donor_coeff[layer + 1, row])
            expected.append(float(sequence_nll(logits, own_row)[0]))
        np.testing.assert_allclose(record["sequence_nll"], expected, atol=3e-6)
        delta = (np.asarray(expected) - native).reshape(-1, 2).mean(axis=1)
        np.testing.assert_allclose(record["delta_nll"], delta.mean(), atol=3e-6)
        np.testing.assert_allclose(record["paired_se"], delta.std(ddof=1) / np.sqrt(2), atol=3e-6)
        matched = (np.asarray(record["sequence_nll"]) - identity[layer]).reshape(-1, 2).mean(axis=1)
        np.testing.assert_allclose(record["delta_nll_vs_identity_replay"], matched.mean(), atol=1e-12)
        np.testing.assert_allclose(record["paired_se_vs_identity_replay"], matched.std(ddof=1) / np.sqrt(2), atol=1e-12)


def test_all_layers_use_one_native_donor_or_the_calibration_mean(model):
    rows = np.random.default_rng(20).integers(0, 32, (8, 7), dtype=np.int32)
    calibration, recipients = rows[:4], rows[4:]
    graph, state = nnx.split(model, nnx.Param)
    order, records = evaluate(graph, state, calibration, recipients, microbatch=2, pairing_seed=42,
                              layer_scope="all_layers")
    assert [r["condition"] for r in records] == ["native", "swapped", "calibration_mean"]
    assert all(r["layer_scope"] == "all_layers" for r in records)
    np.testing.assert_array_equal(order, np.random.default_rng(42).permutation(len(recipients)))
    identity = np.array(records[0]["identity_sequence_nll"])
    np.testing.assert_allclose(identity, records[0]["sequence_nll"], atol=2e-6)
    # Independent one-sequence forwards keep the donor's entire native layer trajectory.
    # Average expanded calibration tables across individual sequences and token positions.
    forward = partial(model, coefficient_score_transform=model.config.attention_score_transform)
    tables = [forward(row[None, :-1], return_coefficients=True)[1] for row in calibration]
    mean = jnp.concatenate(tables, axis=1).mean(axis=(1, 2), keepdims=True)
    for condition in records[1:]:
        expected = []
        for i, row in enumerate(recipients[order]):
            donor = recipients[order[i ^ 1]]
            _, coefficients = forward(donor[None, :-1], return_coefficients=True)
            if condition["condition"] == "calibration_mean":
                coefficients = jnp.broadcast_to(mean, coefficients.shape)
            logits = forward(row[None, :-1], query_coefficients=coefficients)
            expected.append(float(sequence_nll(logits, row[None, :])[0]))
        np.testing.assert_allclose(condition["sequence_nll"], expected, atol=3e-6)
        np.testing.assert_allclose(condition["delta_nll_vs_identity_replay"],
                                   (np.array(expected) - identity).mean(), atol=3e-6)
