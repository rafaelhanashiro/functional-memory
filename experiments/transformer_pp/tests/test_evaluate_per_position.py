"""Per-position evaluation: recipe loading with declared foreign keys, and the NLL curve."""
import json
from copy import copy

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

from experiments.transformer_pp.evaluate_per_position import extend_forward_length, load_model_config, per_position_nll
from scaffolds.transformer.configs import Config, config_to_dict
from scaffolds.transformer.training.state import create_state


def _recipe(tmp_path, **model):
    config = Config()
    config.model.vocab_size, config.model.max_seq_len, config.model.num_layers = 32, 8, 1
    config.model.hidden_size, config.model.num_heads, config.model.num_kv_heads = 16, 2, 2
    config.model.qk_head_dim = config.model.value_head_dim = 8
    config.model.intermediate_size, config.model.attention_backend = 24, "blocked"
    config.data.source, config.data.sequence_length, config.data.synthetic_vocab_size = "synthetic", 9, 32
    raw = config_to_dict(config)
    raw["model"].update(model)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_load_model_config_drops_only_declared_foreign_keys(tmp_path):
    path = _recipe(tmp_path, attention_coefficient_key_kernel_weight_decay=False)
    spec = {"config_path": str(path)}
    with pytest.raises(ValueError, match="Unknown config key"):
        load_model_config(spec)
    spec["ignored_config_keys"] = {"model.attention_coefficient_key_kernel_weight_decay": False}
    config = load_model_config(spec)
    assert config.model.max_seq_len == 8
    longer = load_model_config(spec, max_seq_len=16)
    assert longer.model.max_seq_len == 16 and longer.data.sequence_length == 17
    with pytest.raises(ValueError, match="expected True"):
        load_model_config({**spec, "ignored_config_keys": {"model.attention_coefficient_key_kernel_weight_decay": True}})
    with pytest.raises(ValueError, match="cannot be ignored"):
        load_model_config({**spec, "ignored_config_keys": {"model.max_seq_len": 8}})


def test_per_position_nll_matches_the_direct_mean(tmp_path):
    config = load_model_config({"config_path": str(_recipe(tmp_path))})
    state = create_state(config, jax.random.PRNGKey(0))
    tokens = np.random.default_rng(1).integers(0, 32, size=(6, 9), dtype=np.uint16)
    np.save(tmp_path / "shard-00000.npy", tokens)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dtype": "uint16", "format": "ofm-packed-sequences", "metadata": {},
                                    "num_sequences": 6, "num_tokens": 54, "sequence_length": 9, "version": 1,
                                    "shards": [{"num_sequences": 6, "path": "shard-00000.npy", "sequence_start": 0}]}))
    curve = per_position_nll(state.graphdef, state.params, manifest, sequence_length=9, sequences=6, microbatch=3)
    model = nnx.merge(state.graphdef, state.params)
    batch = jnp.asarray(tokens, jnp.int32)
    logits = model(batch[:, :-1], deterministic=True)
    direct = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch[:, 1:]).mean(axis=0)
    assert curve.shape == (8,)
    np.testing.assert_allclose(curve, np.asarray(direct), rtol=1e-5, atol=1e-6)


def test_forward_extension_keeps_parameters_and_qualifies_causal_prefix(tmp_path):
    config = load_model_config({"config_path": str(_recipe(tmp_path))})
    state = create_state(config, jax.random.PRNGKey(0))
    graph, params = extend_forward_length(state.graphdef, state.params, max_seq_len=16)
    assert nnx.merge(graph, params).config.max_seq_len == 16
    assert nnx.merge(state.graphdef, state.params).config.max_seq_len == 8
    before, after = jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(params)
    assert len(before) == len(after)
    for original, extended in zip(before, after, strict=True):
        np.testing.assert_array_equal(original, extended)
    tokens = np.random.default_rng(12).integers(0, 32, size=(2, 17), dtype=np.uint16)
    np.save(tmp_path / "shard-00000.npy", tokens)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dtype": "uint16", "format": "ofm-packed-sequences", "metadata": {},
                                    "num_sequences": 2, "num_tokens": 34, "sequence_length": 17, "version": 1,
                                    "shards": [{"num_sequences": 2, "path": "shard-00000.npy", "sequence_start": 0}]}))
    details = {}
    curve = per_position_nll(graph, params, manifest, sequence_length=17, sequences=2, microbatch=1,
                             qualification={"prefix_tokens": 8, "mean_nll_atol": 1e-5,
                                            "max_token_nll_atol": 1e-5}, details=details)
    assert curve.shape == (16,)
    assert details["causal_prefix_qualification"]["passed"]
    assert details["subsequent_batch_count"] == 1
    for length in (0, -1, True, 4):
        with pytest.raises(ValueError):
            extend_forward_length(state.graphdef, state.params, max_seq_len=length)


def test_forward_extension_does_not_relax_mosaic_training_limit(tmp_path):
    config = load_model_config({"config_path": str(_recipe(tmp_path))})
    config.model.hidden_size, config.model.qk_head_dim, config.model.value_head_dim = 128, 64, 64
    config.model.max_seq_len, config.data.sequence_length = 2048, 2049
    config.model.attention_backend, config.model.attention_degree = "mosaic", 4
    config.model.attention_normalization = False
    config.model.attention_coefficient_conditioning = "query"
    config.model.attention_coefficient_activation = "tanh"
    config.model.attention_coefficients = "learned"
    config.model.attention_coefficient_sharing = "per_head"
    config.train.dtype = "bfloat16"
    from scaffolds.transformer.models.registry import build_model
    model = build_model(config.model, nnx.Rngs(1))
    graph, params = nnx.split(model, nnx.Param)
    longer_graph, longer_params = extend_forward_length(graph, params, max_seq_len=16384)
    assert nnx.merge(longer_graph, longer_params).config.max_seq_len == 16384
    training_config = copy(config.model)
    training_config.max_seq_len = 16384
    with pytest.raises(ValueError, match="mosaic"):
        build_model(training_config, nnx.Rngs(2))
