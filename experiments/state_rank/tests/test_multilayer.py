"""Matched multi-layer cache extraction; no downloaded models or data."""

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from experiments.state_rank.spectrum import extract_kv, extract_kv_layers


@pytest.fixture
def tiny_llama():
    config = LlamaConfig(
        vocab_size=127, hidden_size=32, intermediate_size=48,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        return LlamaModel(config).eval()


def test_all_layers_use_one_forward_and_preserve_requested_order(tiny_llama):
    token_ids = torch.tensor([[1, 11, 22, 33, 44]])
    calls = []

    def capture(_model, _args, kwargs):
        calls.append((kwargs["input_ids"].clone(), torch.is_inference_mode_enabled()))

    handle = tiny_llama.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        extracted = extract_kv_layers(tiny_llama, token_ids, iter([2, 0, 1]))
    finally:
        handle.remove()
    assert list(extracted) == [2, 0, 1]
    assert len(calls) == 1 and calls[0][1]
    torch.testing.assert_close(calls[0][0], token_ids)

    # Compare directly with the model cache as well as the legacy single-layer API.
    with torch.inference_mode():
        native_cache = tiny_llama(input_ids=token_ids, use_cache=True).past_key_values
    for layer, tensors in extracted.items():
        legacy_tensors = extract_kv(tiny_llama, token_ids, layer)
        for tensor, native, legacy in zip(tensors, native_cache[layer], legacy_tensors):
            assert tensor.shape == (5, 2, 8)
            assert tensor.dtype == np.float64
            np.testing.assert_array_equal(tensor, native[0].transpose(0, 1).double().numpy())
            np.testing.assert_array_equal(tensor, legacy)
        head_keys, head_values = extract_kv(tiny_llama, token_ids, layer, kv_head=1)
        np.testing.assert_array_equal(head_keys, tensors[0][:, 1])
        np.testing.assert_array_equal(head_values, tensors[1][:, 1])


def test_every_layer_cache_matches_independent_causal_prefixes(tiny_llama):
    token_ids = torch.tensor([[1, 8, 13, 21, 34, 55, 89, 12, 6]])
    layers = range(tiny_llama.config.num_hidden_layers)
    full = extract_kv_layers(tiny_llama, token_ids, layers)
    for length in (1, 3, 6, 9):
        prefix = extract_kv_layers(tiny_llama, token_ids[:, :length], layers)
        for layer in layers:
            for full_tensor, prefix_tensor in zip(full[layer], prefix[layer]):
                np.testing.assert_allclose(full_tensor[:length], prefix_tensor, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("layers", [[], [-1], [3], [0.0], [True], [0, 0], None, 1])
def test_invalid_layer_requests_fail_before_inference(tiny_llama, layers):
    def unexpected_forward(*_args, **_kwargs):
        pytest.fail("Invalid layer request reached model inference")

    handle = tiny_llama.register_forward_pre_hook(unexpected_forward)
    try:
        with pytest.raises(ValueError, match="[Ll]ayer"):
            extract_kv_layers(tiny_llama, torch.tensor([[1, 2]]), layers)
    finally:
        handle.remove()


@pytest.mark.parametrize("shape", [(0,), (2,), (1, 0), (2, 3), (1, 1, 3)])
def test_invalid_input_shapes_are_rejected(tiny_llama, shape):
    with pytest.raises(ValueError, match="one nonempty sequence"):
        extract_kv_layers(tiny_llama, torch.zeros(shape, dtype=torch.long), [0, 1])


def test_training_mode_is_rejected(tiny_llama):
    tiny_llama.train()
    with pytest.raises(ValueError, match=r"model\.eval\(\)"):
        extract_kv_layers(tiny_llama, torch.tensor([[1, 2]]), [0, 1])
