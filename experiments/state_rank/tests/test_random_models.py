from copy import deepcopy
import json

import pytest
import torch
from transformers import AutoModel, LlamaConfig

from experiments.state_rank.random_models import (
    build_random_llama,
    load_saved_inputs,
    parameter_fingerprint,
)


@pytest.fixture
def config():
    return LlamaConfig(
        vocab_size=127, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
        bos_token_id=1, pad_token_id=0,
    )


def test_same_seed_matches_all_parameters_and_huggingface_initializer(config):
    first, second = build_random_llama(config, 17), build_random_llama(config, 17)
    assert not first.training
    assert first.config._attn_implementation == "eager"
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in first.parameters())
    reference_config = deepcopy(config)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(17)
        reference = AutoModel.from_config(reference_config, dtype=torch.float32, attn_implementation="eager")
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, second.get_parameter(name))
        assert torch.equal(parameter, reference.get_parameter(name))
    assert parameter_fingerprint(first) == parameter_fingerprint(second)
    assert parameter_fingerprint(first) != parameter_fingerprint(build_random_llama(config, 18))


def test_layers_heads_and_embeddings_have_independent_reference_draws(config):
    model = build_random_llama(config, 9)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        first = getattr(model.layers[0].self_attn, name).weight
        second = getattr(model.layers[1].self_attn, name).weight
        assert first.data_ptr() != second.data_ptr()
        assert not torch.equal(first, second)
    for name in ("q_proj", "k_proj", "v_proj"):
        weight = getattr(model.layers[0].self_attn, name).weight
        # Distinct contiguous output-channel slices are distinct head draws.
        head_dim = config.hidden_size // config.num_attention_heads
        assert not torch.equal(weight[:head_dim], weight[head_dim:2 * head_dim])
    assert not torch.equal(model.layers[0].mlp.gate_proj.weight, model.layers[1].mlp.gate_proj.weight)
    assert torch.count_nonzero(model.embed_tokens.weight[1:]) > 0
    assert torch.count_nonzero(model.embed_tokens.weight[config.pad_token_id]) == 0
    for name, parameter in model.named_parameters():
        if "norm" in name:
            assert torch.equal(parameter, torch.ones_like(parameter))


def test_initialization_restores_rng_dtype_device_and_configuration(config):
    original_config = deepcopy(config.to_dict())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        before_rng = torch.random.get_rng_state().clone()
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            # An ambient meta device catches accidental device inheritance without CUDA.
            with torch.device("meta"):
                model = build_random_llama(config, 3)
                assert torch.empty(0).device.type == "meta"
                assert torch.get_default_dtype() == torch.float64
            assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
            assert torch.equal(torch.random.get_rng_state(), before_rng)
        finally:
            torch.set_default_dtype(old_dtype)
    assert config.to_dict() == original_config


def test_initialization_restores_rng_and_dtype_on_failure(config, monkeypatch):
    def fail(*args, **kwargs):
        torch.rand(5)
        raise RuntimeError("construction failed")

    before_rng = torch.random.get_rng_state().clone()
    before_dtype = torch.get_default_dtype()
    monkeypatch.setattr(AutoModel, "from_config", fail)
    with pytest.raises(RuntimeError, match="construction failed"):
        build_random_llama(config, 0)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.get_default_dtype() == before_dtype


@pytest.mark.parametrize("seed", [-1, True, 1.5, "0", 2**64])
def test_invalid_seeds_rejected(config, seed):
    with pytest.raises(ValueError, match="seed"):
        build_random_llama(config, seed)


def test_fingerprint_tracks_content_names_shape_dtype_and_registration_order():
    def module(parameters):
        result = torch.nn.Module()
        for name, value in parameters:
            result.register_parameter(name, torch.nn.Parameter(value))
        return result

    original = module([("a", torch.arange(8, dtype=torch.float32).reshape(2, 4)), ("b", torch.ones(2))])
    variants = [
        module([("a", torch.arange(8, dtype=torch.float32).reshape(4, 2)), ("b", torch.ones(2))]),
        module([("a", torch.arange(8, dtype=torch.float64).reshape(2, 4)), ("b", torch.ones(2))]),
        module([("other", torch.arange(8, dtype=torch.float32).reshape(2, 4)), ("b", torch.ones(2))]),
        module([("b", torch.ones(2)), ("a", torch.arange(8, dtype=torch.float32).reshape(2, 4))]),
    ]
    before = parameter_fingerprint(original)
    assert len(before) == 64
    assert all(parameter_fingerprint(other) != before for other in variants)
    with torch.no_grad():
        original.a[0, 0] = 99
    assert parameter_fingerprint(original) != before


def test_fingerprint_is_independent_of_chunk_size(config, monkeypatch):
    model = build_random_llama(config, 0)
    expected = parameter_fingerprint(model)
    monkeypatch.setattr("experiments.state_rank.random_models._HASH_CHUNK_BYTES", 13)
    assert parameter_fingerprint(model) == expected


@pytest.fixture
def saved_records():
    return [dict(title=f"article {i}", article_index=i, body_sha256=f"hash-{i}",
                 text="stored text stays untouched", token_ids=[1, 8, 13, 21]) for i in range(4)]


def test_saved_inputs_preserve_every_field_and_exact_token_ids(tmp_path, saved_records):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(saved_records))
    before = path.read_bytes()
    loaded = load_saved_inputs(path, expected_tokens=4, vocab_size=127, bos_token_id=1)
    assert loaded == saved_records
    assert path.read_bytes() == before
    path.write_text(json.dumps(saved_records[:1]))
    assert load_saved_inputs(path, 4, 127, 1, expected_count=1) == saved_records[:1]
    assert load_saved_inputs(path, 4, 127, 1, expected_count=None) == saved_records[:1]


@pytest.mark.parametrize("bad_ids,error", [
    ([1, 8, 13], "exactly"), ([1, 8, 13, 127], "within"),
    ([1, 8, -1, 21], "within"), ([1, 8, 13.0, 21], "integers"),
    ([1, 8, True, 21], "integers"), ([1, [8], 13, 21], "flat"),
    ([2, 8, 13, 21], "BOS"), (None, "exactly"),
])
def test_saved_inputs_reject_invalid_ids(tmp_path, saved_records, bad_ids, error):
    saved_records[0]["token_ids"] = bad_ids
    path = tmp_path / "bad-inputs.json"
    path.write_text(json.dumps(saved_records))
    with pytest.raises(ValueError, match=error):
        load_saved_inputs(path, 4, 127, 1)


@pytest.mark.parametrize("records,error", [([], "nonempty"), ({}, "nonempty"), ([{}], "Expected 4"), ([1, 2, 3, 4], "mapping")])
def test_saved_inputs_require_record_list_and_default_count(tmp_path, records, error):
    path = tmp_path / "bad-inputs.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match=error):
        load_saved_inputs(path, 4, 127, 1)
