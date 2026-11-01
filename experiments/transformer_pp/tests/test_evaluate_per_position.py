"""Per-position evaluation: recipe loading with declared foreign keys, and the NLL curve."""
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

from experiments.transformer_pp.evaluate_per_position import load_model_config, per_position_nll
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
