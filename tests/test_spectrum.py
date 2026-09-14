"""Numerical and extraction checks; no pretrained weights or data required."""

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from scripts.spectrum import (
    compute_llama_spectra,
    compute_spectra,
    direct_prefix_matrix,
    extract_kv,
    iter_article_bodies,
    iter_prefix_matrices,
    select_inputs,
)


@pytest.mark.parametrize("scale", [None, 0.2])
def test_incremental_state_and_spectrum_match_direct_calculation(scale):
    rng = np.random.default_rng(37)
    keys = rng.normal(size=(9, 4)) * 0.4
    values = rng.normal(size=(9, 6))
    factor = 1 / np.sqrt(keys.shape[1]) if scale is None else scale
    states = list(iter_prefix_matrices(keys, values, scale=scale))
    result = compute_spectra(keys, values, check_prefixes=(1, 4, 9), scale=scale)

    assert len(states) == len(keys)
    assert result["scaled_eigenvalues"].shape == (9, 6)
    for t, (matrix, log_scale) in enumerate(states, start=1):
        prefix_keys, prefix_values = keys[:t], values[:t]
        expected_scale = np.max(np.sum(prefix_keys**2, axis=1)) * factor
        gram = np.exp(factor * (prefix_keys @ prefix_keys.T) - expected_scale)
        expected = prefix_values.T @ gram @ prefix_values
        np.testing.assert_allclose(log_scale, expected_scale, rtol=1e-14)
        np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            direct_prefix_matrix(prefix_keys, prefix_values, log_scale, scale=scale),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        eigenvalues = np.linalg.eigvalsh(expected)[::-1].clip(min=0)
        np.testing.assert_allclose(
            result["scaled_eigenvalues"][t - 1], eigenvalues, rtol=1e-10, atol=1e-12
        )
        normalized = eigenvalues / eigenvalues.sum()
        np.testing.assert_allclose(
            result["normalized_eigenvalues"][t - 1], normalized, atol=1e-12
        )
        np.testing.assert_allclose(
            result["participation_ratio"][t - 1], 1 / np.sum(normalized**2)
        )
        np.testing.assert_allclose(
            result["log_trace"][t - 1], log_scale + np.log(np.trace(expected))
        )
        for energy, field in [(0.90, "energy_rank90"), (0.99, "energy_rank99")]:
            assert result[field][t - 1] == np.searchsorted(np.cumsum(normalized), energy) + 1
    np.testing.assert_allclose(result["log_scale"], [entry[1] for entry in states])
    np.testing.assert_allclose(result["key_norms"], np.linalg.norm(keys, axis=1))


def test_single_token_and_prefix_rank_ceiling():
    rng = np.random.default_rng(81)
    keys = rng.normal(size=(10, 4)) * 0.5
    values = rng.normal(size=(10, 7))
    result = compute_spectra(keys, values)
    eigenvalues = result["scaled_eigenvalues"]
    np.testing.assert_allclose(eigenvalues[0, 0], np.dot(values[0], values[0]))
    np.testing.assert_allclose(eigenvalues[0, 1:], 0, atol=1e-12)
    for t, spectrum in enumerate(eigenvalues, start=1):
        assert np.count_nonzero(spectrum > 1e-10 * spectrum[0]) <= min(t, 7)
        assert np.all(spectrum >= 0)
        assert np.all(np.diff(spectrum) <= 0)
    assert result["participation_ratio"][0] == pytest.approx(1)
    assert result["energy_rank90"][0] == result["energy_rank99"][0] == 1


@pytest.mark.parametrize("scale", [None, 0.2])
def test_llama_multihead_spectra_keep_heads_independent(scale):
    rng = np.random.default_rng(51)
    keys = rng.normal(size=(7, 3, 4)) * 0.3
    values = rng.normal(size=(7, 3, 5))
    keys[:, 0] = keys[0, 0]  # Identical keys give a rank-one state for head 0.
    values[:, 2] = 0  # A zero state must not borrow another head's energy.
    result = compute_llama_spectra(keys, values, check_prefixes=(1, 4, 7), scale=scale)
    factor = 1 / np.sqrt(keys.shape[-1]) if scale is None else scale

    for name, array in result.items():
        expected_shape = (7, 3, 5) if name in {
            "scaled_eigenvalues", "normalized_eigenvalues",
        } else (7, 3)
        assert array.shape == expected_shape, name
    np.testing.assert_allclose(result["key_norms"], np.linalg.norm(keys, axis=-1))
    for head in range(3):
        for t in range(1, 8):
            prefix_keys, prefix_values = keys[:t, head], values[:t, head]
            shift = factor * np.square(prefix_keys).sum(axis=-1).max()
            gram = np.exp(factor * (prefix_keys @ prefix_keys.T) - shift)
            expected = prefix_values.T @ gram @ prefix_values
            eigenvalues = np.linalg.eigvalsh(expected)[::-1].clip(min=0)
            np.testing.assert_allclose(result["log_scale"][t - 1, head], shift)
            np.testing.assert_allclose(
                result["scaled_eigenvalues"][t - 1, head], eigenvalues, atol=1e-12,
            )
        errors = result["direct_check_relative_error"][:, head]
        assert np.isnan(errors[[1, 2, 4, 5]]).all()
        assert np.all(errors[[0, 3, 6]] < 1e-10)
    np.testing.assert_allclose(result["normalized_eigenvalues"][:, :2].sum(-1), 1)
    np.testing.assert_array_equal(result["normalized_eigenvalues"][:, 2], 0)
    np.testing.assert_array_equal(result["threshold_rank"][:, 0], 1)
    np.testing.assert_array_equal(result["threshold_rank"][:, 2], 0)
    np.testing.assert_allclose(result["participation_ratio"][:, 0], 1)
    np.testing.assert_array_equal(result["participation_ratio"][:, 2], 0)
    assert np.all(result["threshold_rank"] <= np.minimum(np.arange(1, 8), 5)[:, None])
    assert np.isneginf(result["log_trace"][:, 2]).all()


def test_opposite_values_cancel_the_state():
    keys = np.array([[2.0, -1.0], [2.0, -1.0]])
    values = np.array([[1.0, -2.0, 0.5], [-1.0, 2.0, -0.5]])
    states = list(iter_prefix_matrices(keys, values))
    np.testing.assert_array_equal(states[-1][0], np.zeros((3, 3)))
    result = compute_spectra(keys, values, check_prefixes=(1, 2))
    np.testing.assert_array_equal(result["scaled_eigenvalues"][-1], np.zeros(3))
    np.testing.assert_array_equal(result["normalized_eigenvalues"][-1], np.zeros(3))
    assert result["participation_ratio"][-1] == 0
    assert result["energy_rank90"][-1] == result["energy_rank99"][-1] == 0
    assert np.isneginf(result["log_trace"][-1])


def test_large_key_norms_stay_finite_when_the_log_scale_changes():
    keys = np.array([[40.0, -30.0], [40.0, -29.0], [45.0, -30.0]])
    values = np.array([[1.0, 2.0], [-0.5, 1.0], [2.0, -3.0]])
    result = compute_spectra(keys, values, check_prefixes=(1, 2, 3))
    assert np.all(result["log_scale"] > np.log(np.finfo(np.float64).max))
    assert result["log_scale"][-1] > result["log_scale"][0]
    for field in ["scaled_eigenvalues", "normalized_eigenvalues", "log_trace"]:
        assert np.isfinite(result[field]).all()
    for t, (matrix, log_scale) in enumerate(iter_prefix_matrices(keys, values), start=1):
        logits = (keys[:t] @ keys[:t].T) / np.sqrt(2)
        expected = values[:t].T @ np.exp(logits - log_scale) @ values[:t]
        np.testing.assert_allclose(matrix, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result["normalized_eigenvalues"].sum(axis=1), 1)


def test_article_boundaries_and_nested_headings():
    rows = [
        " = First article = \n",
        "\n",
        " First paragraph. \n",
        " = = Background = = \n",
        " Second paragraph. \n",
        " = = = Details = = = \n",
        " Third paragraph. \n",
        " = Second article = \n",
        " Final article body. \n",
    ]
    assert list(iter_article_bodies(rows)) == [
        {
            "title": "First article",
            "text": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        },
        {"title": "Second article", "text": "Final article body."},
    ]


def test_input_selection_preserves_order_and_uses_exact_bos_plus_text_length():
    class IntegerTokenizer:
        bos_token_id = 1

        def encode(self, text, *, add_special_tokens, truncation, max_length):
            assert add_special_tokens is False
            assert truncation is True
            return [int(word) for word in text.split()][:max_length]

        def decode(self, ids, *, clean_up_tokenization_spaces):
            assert clean_up_tokenization_spaces is False
            return " ".join(map(str, ids))

    rows = [
        "= Short =", "9",
        "= First =", "11 12 13 14 15",
        "= Second =", "21 22 23 24",
        "= Third =", "31 32", "= = Subsection = =", "33 34",
        "= Fourth =", "41 42 43 44 45",
        "= Fifth =", "51 52 53 54",
    ]
    selected = select_inputs(IntegerTokenizer(), rows, tokens=5)
    assert [item["title"] for item in selected] == ["First", "Second", "Third", "Fourth"]
    assert [item["article_index"] for item in selected] == [1, 2, 3, 4]
    assert [item["token_ids"] for item in selected] == [
        [1, 11, 12, 13, 14], [1, 21, 22, 23, 24],
        [1, 31, 32, 33, 34], [1, 41, 42, 43, 44],
    ]
    assert selected[0]["text"] == "11 12 13 14"
    with pytest.raises(ValueError, match="only 0 articles"):
        select_inputs(IntegerTokenizer(), rows, tokens=8)


@pytest.fixture
def tiny_llama():
    torch.manual_seed(19)
    config = LlamaConfig(
        vocab_size=127,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaModel(config).eval()


def test_full_pass_cache_prefixes_match_separate_causal_passes(tiny_llama):
    token_ids = torch.tensor([[1, 8, 13, 21, 34, 55, 89, 12, 6]])
    keys, values = extract_kv(tiny_llama, token_ids, layer=1, kv_head=1)
    assert keys.shape == values.shape == (9, 8)
    assert keys.dtype == values.dtype == np.float64
    for t in [1, 3, 6, 9]:
        prefix_keys, prefix_values = extract_kv(tiny_llama, token_ids[:, :t], layer=1, kv_head=1)
        np.testing.assert_allclose(keys[:t], prefix_keys, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(values[:t], prefix_values, rtol=2e-6, atol=2e-7)


def test_llama_all_kv_heads_are_extracted_in_one_forward_pass(tiny_llama):
    token_ids = torch.tensor([[1, 11, 22, 33, 44]])
    forward_inputs = []

    def record_forward(_model, _args, kwargs):
        forward_inputs.append(kwargs["input_ids"].clone())

    handle = tiny_llama.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        keys, values = extract_kv(tiny_llama, token_ids, layer=1)
    finally:
        handle.remove()
    assert len(forward_inputs) == 1
    torch.testing.assert_close(forward_inputs[0], token_ids)
    assert keys.shape == values.shape == (5, 2, 8)
    assert keys.dtype == values.dtype == np.float64
    for head in range(tiny_llama.config.num_key_value_heads):
        head_keys, head_values = extract_kv(tiny_llama, token_ids, layer=1, kv_head=head)
        np.testing.assert_array_equal(keys[:, head], head_keys)
        np.testing.assert_array_equal(values[:, head], head_values)


def test_extracted_keys_include_rope_and_values_match_the_selected_kv_head(tiny_llama):
    token_ids = torch.tensor([[1, 11, 22, 33, 44]])
    captured = {}
    attention = tiny_llama.layers[1].self_attn

    def capture(name):
        def hook(_module, _args, output):
            captured[name] = output.detach()

        return hook

    handles = [
        attention.k_proj.register_forward_hook(capture("keys")),
        attention.v_proj.register_forward_hook(capture("values")),
    ]
    try:
        keys, values = extract_kv(tiny_llama, token_ids, layer=1, kv_head=1)
    finally:
        for handle in handles:
            handle.remove()
    raw_keys = captured["keys"].reshape(1, 5, 2, 8)[:, :, 1]
    expected_values = captured["values"].reshape(1, 5, 2, 8)[0, :, 1]
    with torch.no_grad():
        cos, sin = tiny_llama.rotary_emb(raw_keys, torch.arange(5).unsqueeze(0))
    rotated = torch.cat((-raw_keys[..., 4:], raw_keys[..., :4]), dim=-1)
    expected_keys = raw_keys * cos + rotated * sin
    np.testing.assert_allclose(keys, expected_keys[0].numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(values, expected_values.numpy(), rtol=1e-6, atol=1e-8)
    assert not np.allclose(keys[1:], raw_keys[0, 1:].numpy())


def test_threshold_rank_strict_boundary_scale_and_zero():
    from scripts.spectrum import threshold_rank, spectral_metrics, RANK_TOL
    singular = np.array([[1., 2e-4, 1e-4, 5e-5, 0.], [0., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(threshold_rank(singular, RANK_TOL), [2, 0])
    for scale in (1e-100, 1., 1e100):
        scaled = singular * scale
        np.testing.assert_array_equal(threshold_rank(scaled, RANK_TOL), [2, 0])
        metrics = spectral_metrics(scaled**2, np.zeros(2))
        np.testing.assert_array_equal(metrics['threshold_rank'], [2, 0])
        assert metrics['participation_ratio'][1] == 0
        assert metrics['energy_rank90'][1] == metrics['energy_rank99'][1] == 0
        assert np.isneginf(metrics['log_trace'][1])
    with pytest.raises(ValueError):
        threshold_rank(np.array([np.nan]), RANK_TOL)


def test_gdn_input_selection_keeps_reference_articles():
    class Tokenizer:
        bos_token_id = 1
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))[:kwargs['max_length']]
        def decode(self, ids, **kwargs):
            return str(ids)
    rows = ['= First =', 'a b c d', '= Second =', 'e f g h', '= Third =', 'i j k l']
    selected = select_inputs(Tokenizer(), rows, 4, count=2, article_indices=[0, 2])
    assert [item['title'] for item in selected] == ['First', 'Third']
    assert all(len(item['token_ids']) == 4 for item in selected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native FLA kernels require CUDA')
def test_native_gated_delta_rule_matches_explicit_recurrence():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    torch.manual_seed(123)
    q = torch.randn(1, 9, 2, 16, device='cuda')
    k = torch.randn_like(q)
    v = torch.randn(1, 9, 2, 12, device='cuda')
    g = -torch.rand(1, 9, 2, device='cuda')
    beta = torch.sigmoid(torch.randn(1, 9, 2, device='cuda'))
    state = torch.zeros(2, 16, 12, dtype=torch.float64, device='cuda')
    with torch.inference_mode():
        for t in range(1, 10):
            key = k[0, t - 1].double()
            key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            state = state * g[0, t - 1].double().exp()[:, None, None]
            residual = v[0, t - 1].double() - torch.einsum('hkv,hk->hv', state, key)
            state += beta[0, t - 1].double()[:, None, None] * key[:, :, None] * residual[:, None, :]
            _, native = fused_recurrent_gated_delta_rule(
                q=q[:, :t], k=k[:, :t], v=v[:, :t], g=g[:, :t], beta=beta[:, :t],
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            assert native.dtype == torch.float32
            torch.testing.assert_close(native[0].double(), state, rtol=2e-5, atol=2e-6)
            sigma = torch.linalg.svdvals(native[0].double()).cpu().numpy()
            from scripts.spectrum import threshold_rank, RANK_TOL
            assert np.all(threshold_rank(sigma, RANK_TOL) <= min(t, 12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native FLA kernels require CUDA')
def test_tiny_gdn_cached_states_match_independent_prefixes():
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel
    from scripts.spectrum import compute_gdn_spectra
    torch.manual_seed(42)
    config = GatedDeltaNetConfig(
        vocab_size=127, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_heads=2, head_dim=16, expand_v=1, attn_mode='fused_recurrent', conv_size=4,
    )
    model = GatedDeltaNetModel(config).float().cuda().eval()
    ids = torch.randint(0, 127, (1, 70))
    result = compute_gdn_spectra(model, ids, layer=1, heads=[0, 1], check_prefixes=[1, 2, 8, 70])
    assert result['scaled_eigenvalues'].shape == (70, 2, 16)
    assert result['singular_values'].dtype == np.float64
    np.testing.assert_array_equal(result['threshold_rank'][0], [1, 1])
    assert np.all(result['threshold_rank'] <= np.minimum(np.arange(1, 71), 16)[:, None])
    assert np.isfinite(result['scaled_eigenvalues']).all()
    np.testing.assert_allclose(result['normalized_eigenvalues'].sum(-1), 1)
