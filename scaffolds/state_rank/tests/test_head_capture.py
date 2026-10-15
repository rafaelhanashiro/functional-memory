import json

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from scaffolds.state_rank.head_capture import (
    _token_digest,
    _write_deterministic_npz,
    capture_llama_heads,
    causal_softmax_output,
    load_token_corpus,
    select_document_windows,
    sha256_file,
)


@pytest.fixture
def tiny_llama():
    config = LlamaConfig(vocab_size=80, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(16)
        return LlamaModel(config).eval().requires_grad_(False)


def test_capture_matches_native_outputs_gqa_rope_and_causal_prefixes(tiny_llama):
    ids = torch.tensor([[1, 7, 11, 13, 17, 19], [1, 9, 23, 29, 31, 37]])
    specs = [(2, 3), (0, 0), (2, 2), (0, 1)]
    calls = []
    handle = tiny_llama.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        captured = capture_llama_heads(tiny_llama, ids, specs)
    finally:
        handle.remove()
    assert len(calls) == 1
    assert list(captured) == specs
    for (layer, head), result in captured.items():
        assert result.kv_head == head // 2
        assert result.q.shape == result.k.shape == result.v.shape == result.teacher.shape == (2, 6, 8)
        assert result.q.dtype == result.teacher.dtype == torch.float32
        assert result.q.device.type == "cpu"
        assert not result.q.is_inference()  # Frozen data must still be usable in autograd fitting.
        assert result.native_max_abs_error < 1e-6
        for length in (1, 3, 5):
            prefix = capture_llama_heads(tiny_llama, ids[:, :length], [(layer, head)])[layer, head]
            for name in ("q", "k", "v", "teacher"):
                torch.testing.assert_close(getattr(prefix, name), getattr(result, name)[:, :length],
                                           atol=2e-7, rtol=1e-5)
    # Consecutive query heads share the GQA key/value head but not their queries.
    torch.testing.assert_close(captured[2, 3].k, captured[2, 2].k, atol=0, rtol=0)
    torch.testing.assert_close(captured[2, 3].v, captured[2, 2].v, atol=0, rtol=0)
    assert not torch.allclose(captured[2, 3].q, captured[2, 2].q)


def test_capture_uses_actual_rotary_positions(tiny_llama):
    ids = torch.tensor([[1, 11, 22, 33, 44]])
    raw = {}
    attention = tiny_llama.layers[1].self_attn
    handles = [
        attention.q_proj.register_forward_hook(lambda _m, _a, out: raw.update(q=out.detach())),
        attention.k_proj.register_forward_hook(lambda _m, _a, out: raw.update(k=out.detach())),
    ]
    try:
        result = capture_llama_heads(tiny_llama, ids, [(1, 3)])[1, 3]
    finally:
        for handle in handles:
            handle.remove()
    for name, count, index in (("q", 4, 3), ("k", 2, 1)):
        unrotated = raw[name].reshape(1, 5, count, 8)[:, :, index]
        with torch.no_grad():
            cos, sin = tiny_llama.rotary_emb(unrotated, torch.arange(5)[None])
        rotated_half = torch.cat((-unrotated[..., 4:], unrotated[..., :4]), dim=-1)
        expected = unrotated * cos + rotated_half * sin
        torch.testing.assert_close(getattr(result, name), expected, atol=0, rtol=0)
        assert not torch.allclose(getattr(result, name)[:, 1:], unrotated[:, 1:])


@pytest.mark.parametrize("specs", [[], [(0, 0), (0, 0)], [(3, 0)], [(0, 4)]])
def test_invalid_head_requests_fail_before_model_forward(tiny_llama, specs):
    calls = []
    handle = tiny_llama.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError):
            capture_llama_heads(tiny_llama, torch.tensor([[1, 2]]), specs)
    finally:
        handle.remove()
    assert not calls


def test_capture_restores_hooks_after_model_failure(tiny_llama):
    attention = tiny_llama.layers[0].self_attn
    with pytest.raises((IndexError, RuntimeError)):
        capture_llama_heads(tiny_llama, torch.tensor([[999]]), [(0, 0)])
    for module in (attention, attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_softmax_teacher_includes_self_and_excludes_future():
    q = k = torch.ones(1, 3, 2)
    v = torch.tensor([[[1.0], [3.0], [8.0]]])
    result = causal_softmax_output(q, k, v, scale=0.5)
    torch.testing.assert_close(result, torch.tensor([[[1.0], [2.0], [4.0]]]))


class CharacterTokenizer:
    bos_token_id = 0

    def encode(self, text, *, add_special_tokens, truncation, max_length):
        assert not add_special_tokens and truncation
        return [ord(character) for character in text[:max_length]]


def article_rows(split, count=5):
    rows = []
    for index in range(count):
        rows.extend([f" = {split} document {index} = ", f"{split}{index}" * 20])
    return rows


def test_document_splits_are_disjoint_and_selection_is_reproducible():
    rows = {split: article_rows(split) for split in ("train", "validation", "test")}
    # Same title with different bodies, and same body with different titles, both
    # count as cross-split overlap and must be excluded from every split.
    rows["train"] += [" = Shared title = ", "train duplicate " * 10,
                      " = A title = ", "duplicate body " * 10]
    rows["test"] += [" = Shared title = ", "test duplicate " * 10,
                     " = B title = ", "duplicate body " * 10]
    counts = dict.fromkeys(rows, 5)
    first, metadata = select_document_windows(CharacterTokenizer(), rows, counts,
                                              tokens=12, seed=18,
                                              long_test_windows=2, long_tokens=30)
    second, second_metadata = select_document_windows(CharacterTokenizer(), dict(reversed(list(rows.items()))),
                                                      counts, tokens=12, seed=18,
                                                      long_test_windows=2, long_tokens=30)
    assert metadata == second_metadata
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert (first[name][:, 0] == CharacterTokenizer.bos_token_id).all()
        assert all("title" not in entry["title"].casefold() for entry in metadata[name])
    identities = [{entry["body_sha256"] for entry in metadata[name]}
                  for name in ("train", "validation", "test")]
    assert not identities[0] & identities[1]
    assert not identities[0] & identities[2]
    assert not identities[1] & identities[2]
    assert {entry["body_sha256"] for entry in metadata["test_long"]} <= identities[2]


def test_insufficient_distinct_documents_fails_instead_of_reusing_windows():
    rows = {split: article_rows(split, count=1) for split in ("train", "validation", "test")}
    with pytest.raises(ValueError, match="requested 2 articles"):
        select_document_windows(CharacterTokenizer(), rows, dict.fromkeys(rows, 2), tokens=5, seed=0)


def test_token_archive_regenerates_byte_identically_and_rejects_corruption(tmp_path):
    arrays = {"train": np.asarray([[1, 2, 3], [1, 4, 5]], dtype=np.int64)}
    path = tmp_path / "tokens.npz"
    _write_deterministic_npz(path, arrays)
    expected = sha256_file(path)
    _write_deterministic_npz(path, arrays)
    assert sha256_file(path) == expected
    manifest = {"tokens_sha256": expected, "shapes": {"train": [2, 3]},
                "selected": {"train": [{"token_ids_sha256": _token_digest(ids)} for ids in arrays["train"]]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    torch.testing.assert_close(load_token_corpus(tmp_path, expected_sha256=expected)["train"],
                               torch.from_numpy(arrays["train"]))
    with pytest.raises(ValueError, match="checksum"):
        load_token_corpus(tmp_path, expected_sha256="0" * 64)
    corrupted = path.read_bytes() + b"modified"
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum"):
        load_token_corpus(tmp_path)
