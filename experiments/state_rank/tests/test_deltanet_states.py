"""All-layer cache/measurement checks, plus optional native CUDA comparisons."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.state_rank.deltanet_states import (
    METRICS, compute_gdn_spectra_layers, iter_gdn_states_layers,
)
from experiments.state_rank.spectrum import CACHE_RTOL, compute_gdn_spectra


class CachedRecurrence(torch.nn.Module):
    """FP32 test recurrence whose output depends on every convolution cache.

    This is a cache-transport fixture, not an approximation to DeltaNet's rule.
    The native CUDA test below independently exercises the actual architecture.
    """

    def __init__(self, *, zero=False, mismatch=False, invalid_dtype=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
        for layer in self.layers:
            layer.attn = SimpleNamespace(mode="fused_recurrent", num_v_heads=2)
        self.calls = []
        self.zero = zero
        self.mismatch = mismatch
        self.invalid_dtype = invalid_dtype

    def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
        assert use_cache and return_dict and torch.is_inference_mode_enabled()
        fresh = past_key_values is None
        self.calls.append((input_ids.shape[1], fresh))
        if fresh:
            past_key_values = [
                {"recurrent_state": torch.zeros(1, 2, 3, 4), "conv_state": torch.zeros(2)}
                for _ in self.layers
            ]
        assert len(past_key_values) == len(self.layers)
        for token in input_ids[0]:
            hidden = token.float()
            for layer_index, cached in enumerate(past_key_values):
                convolution = cached["conv_state"]
                convolution.mul_(0.5).add_(hidden)
                for head in range(2):
                    key = torch.sin(torch.arange(4) + token + layer_index + head)
                    value = torch.cos(torch.arange(3) + convolution[head] + head)
                    state = cached["recurrent_state"][0, head]
                    if not self.zero:
                        state.mul_(0.75).add_(torch.outer(value, key))
                hidden = hidden + convolution.mean() * 0.1
        if self.mismatch and fresh and input_ids.shape[1] > 1:
            past_key_values[1]["recurrent_state"].add_(1.0)
        if self.invalid_dtype:
            past_key_values[0]["recurrent_state"] = past_key_values[0]["recurrent_state"].double()
        return SimpleNamespace(past_key_values=past_key_values)


@pytest.fixture
def mock_model():
    return CachedRecurrence().eval()


def test_iterator_runs_once_per_token_preserves_complete_cache_and_layer_order(mock_model):
    tokens = torch.tensor([[1, 3, 7, 4]])
    states = list(iter_gdn_states_layers(mock_model, tokens, [2, 0, 1]))
    assert mock_model.calls == [(1, True), (1, False), (1, False), (1, False)]
    for prefix, state in enumerate(states, 1):
        assert state.shape == (3, 2, 3, 4)
        assert state.dtype == np.float64
        with torch.inference_mode():
            cache = mock_model(tokens[:, :prefix]).past_key_values
        expected = np.stack([cache[layer]["recurrent_state"][0].double().numpy() for layer in [2, 0, 1]])
        np.testing.assert_array_equal(state, expected)
    # Every earlier snapshot must survive in-place updates of the model's cache.
    assert not np.array_equal(states[0], states[-1])
    assert all(not np.shares_memory(states[0], state) for state in states[1:])


def test_metrics_selected_spectra_and_baseline_layer_match_existing_helper(mock_model):
    tokens = torch.tensor([[1, 3, 7, 4, 8]])
    layers, prefixes = [2, 0, 1], [1, 3, 5]
    arrays, validation = compute_gdn_spectra_layers(
        mock_model, tokens, layers, prefixes, full_spectrum_layers=[1], svd_workers=2,
    )
    assert sum(not fresh for _, fresh in mock_model.calls) == tokens.shape[1] - 1
    assert len(mock_model.calls) == tokens.shape[1] + len(prefixes)
    for name in METRICS:
        assert arrays[name].shape == (3, 5, 2), name
    assert arrays["normalized_eigenvalues"].shape == (3, 3, 2, 3)
    assert arrays["prefix_check_relative_error"].shape == (3, 3, 2)
    np.testing.assert_array_equal(arrays["layers"], layers)
    np.testing.assert_array_equal(arrays["head_indices"], [0, 1])
    np.testing.assert_array_equal(arrays["prefixes"], np.arange(1, 6))
    np.testing.assert_array_equal(arrays["spectrum_prefixes"], prefixes)
    for index, layer in enumerate(layers):
        old = compute_gdn_spectra(mock_model, tokens, layer, [0, 1], check_prefixes=prefixes)
        for name in METRICS:
            np.testing.assert_array_equal(arrays[name][index], old[name])
        np.testing.assert_array_equal(
            arrays["normalized_eigenvalues"][index], old["normalized_eigenvalues"][np.array(prefixes) - 1],
        )
        if layer == 1:
            np.testing.assert_array_equal(validation["full_spectra"][layer], old["normalized_eigenvalues"])
    assert set(validation["full_spectra"]) == {1}
    assert validation["maximum_prefix_relative_error"] == 0.0
    assert validation["maximum_prefix_relative_error_by_layer"] == [0.0, 0.0, 0.0]
    assert validation["checked_prefixes"] == prefixes
    assert validation["svd_workers"] == 2
    assert not {"states", "singular_values", "keys", "values", "scaled_eigenvalues"}.intersection(arrays)
    validation.pop("full_spectra")
    json.dumps(validation, allow_nan=False)


def test_serial_and_threaded_svd_are_identical(mock_model):
    tokens = torch.tensor([[1, 4, 3]])
    serial, _ = compute_gdn_spectra_layers(mock_model, tokens, [0, 1, 2], [1, 3], svd_workers=1)
    parallel, _ = compute_gdn_spectra_layers(mock_model, tokens, [0, 1, 2], [1, 3], svd_workers=3)
    for name in serial:
        np.testing.assert_array_equal(serial[name], parallel[name])


def test_zero_states_preserve_zero_ranks_and_negative_infinite_log_trace():
    model = CachedRecurrence(zero=True).eval()
    arrays, validation = compute_gdn_spectra_layers(model, torch.tensor([[1, 2]]), [0, 2], [1, 2])
    for name in ("threshold_rank", "energy_rank90", "energy_rank99", "participation_ratio", "normalized_eigenvalues"):
        np.testing.assert_array_equal(arrays[name], 0)
    assert np.isneginf(arrays["log_trace"]).all()
    assert validation["maximum_prefix_relative_error"] == 0


def test_independent_prefix_checks_detect_a_nonbaseline_layer_mismatch():
    model = CachedRecurrence(mismatch=True).eval()
    with pytest.raises(FloatingPointError, match="Prefix 3 cache mismatch at layer 1"):
        compute_gdn_spectra_layers(model, torch.tensor([[1, 2, 3]]), [0, 1, 2], [1, 3])


@pytest.mark.parametrize("layers", [[], [3], [-1], [0, 0], [True], [0.0]])
def test_invalid_layers_are_rejected(mock_model, layers):
    with pytest.raises(ValueError, match="[Ll]ayer"):
        list(iter_gdn_states_layers(mock_model, torch.tensor([[1, 2]]), layers))
    assert mock_model.calls == []


@pytest.mark.parametrize("prefixes", [[], [0], [4], [2, 1], [1, 1], [True], [1.0]])
def test_invalid_spectrum_prefixes_are_rejected(mock_model, prefixes):
    with pytest.raises(ValueError, match="[Pp]refix"):
        compute_gdn_spectra_layers(mock_model, torch.tensor([[1, 2, 3]]), [0, 1], prefixes)
    assert mock_model.calls == []


def test_model_mode_and_native_state_dtype_are_enforced(mock_model):
    tokens = torch.tensor([[1, 2]])
    mock_model.train()
    with pytest.raises(ValueError, match="eval model"):
        list(iter_gdn_states_layers(mock_model, tokens, [0]))
    mock_model.eval().layers[2].attn.mode = "chunk"
    with pytest.raises(ValueError, match="fused_recurrent"):
        list(iter_gdn_states_layers(mock_model, tokens, [0]))
    model = CachedRecurrence(invalid_dtype=True).eval()
    with pytest.raises(RuntimeError, match="FP32 recurrent state"):
        list(iter_gdn_states_layers(model, tokens, [0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native FLA kernels require CUDA")
def test_native_all_layer_states_match_existing_spectra_and_prefix_checks():
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel

    with torch.random.fork_rng():
        torch.manual_seed(123)
        config = GatedDeltaNetConfig(
            vocab_size=127, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_heads=2, head_dim=16, expand_v=1, attn_mode="fused_recurrent", conv_size=4,
        )
        model = GatedDeltaNetModel(config).float().cuda().eval()
        tokens = torch.randint(0, 127, (1, 70))
    prefixes = [1, 2, 8, 32, 70]
    arrays, validation = compute_gdn_spectra_layers(
        model, tokens, [1, 0], prefixes, full_spectrum_layers=[0, 1], svd_workers=2,
    )
    assert validation["maximum_prefix_relative_error"] < CACHE_RTOL
    for index, layer in enumerate([1, 0]):
        reference = compute_gdn_spectra(model, tokens, layer, [0, 1], check_prefixes=prefixes)
        for name in METRICS:
            np.testing.assert_allclose(arrays[name][index], reference[name], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            validation["full_spectra"][layer], reference["normalized_eigenvalues"], rtol=1e-10, atol=1e-12,
        )
        assert np.all(arrays["threshold_rank"][index] <= np.minimum(np.arange(1, 71), 16)[:, None])
