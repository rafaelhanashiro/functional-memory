from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from experiments.key_dimension.configs import KEY_DIMENSIONS, ModelConfig, load_config
from experiments.key_dimension.models import build_model, model_parameter_count


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"
EXPECTED_COUNTS = (179_590_144, 180_245_504, 181_556_224, 184_177_664, 189_420_544, 199_906_304, 220_877_824)


def _small_config(**overrides):
    settings = dict(vocab_size=128, max_seq_len=16, num_layers=2, hidden_size=32,
                    num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                    intermediate_size=64, tie_embeddings=False)
    return ModelConfig(**(settings | overrides))


@pytest.mark.parametrize("qk_dim,expected", list(zip(KEY_DIMENSIONS, EXPECTED_COUNTS, strict=True)))
def test_production_parameter_counts_from_actual_flax_shapes(qk_dim, expected):
    cfg = replace(load_config(BASE_CONFIG).model, qk_head_dim=qk_dim)
    # Trace real NNX initialization; no allocation of seven 200M parameter sets.
    params = jax.eval_shape(lambda: nnx.state(build_model(cfg, 7), nnx.Param))
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    assert count == expected
    assert model_parameter_count(cfg) == count
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))


@pytest.mark.parametrize("qk_dim", KEY_DIMENSIONS)
def test_transformer_forward_backward_and_causality(qk_dim):
    cfg = _small_config(qk_head_dim=qk_dim, attention_backend="xla")
    model = build_model(cfg, nnx.Rngs(0))
    tokens = jnp.arange(16).reshape(2, 8)
    logits = model(tokens)
    assert logits.shape == (2, 8, 128)
    graph, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda candidate: jnp.square(nnx.merge(graph, candidate)(tokens)).mean())(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert sum(leaf.size for leaf in jax.tree.leaves(params)) == model_parameter_count(cfg)
    altered = tokens.at[:, 5:].set(127)
    np.testing.assert_allclose(model(altered)[:, :5], logits[:, :5], atol=1e-6, rtol=1e-6)


def test_remat_and_state_roundtrip_preserve_logits_and_parameter_storage():
    cfg = _small_config()
    plain, remat = build_model(cfg, 4), build_model(replace(cfg, remat=True), 4)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    expected = plain(tokens)
    np.testing.assert_array_equal(remat(tokens), expected)
    graph, params = nnx.split(plain, nnx.Param)
    np.testing.assert_array_equal(nnx.merge(graph, params)(tokens), expected)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))


def test_max_sequence_length_is_enforced():
    model = build_model(_small_config(), 0)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(jnp.ones((1, 17), dtype=jnp.int32))


def test_shared_weights_have_identical_initialization_across_key_dimensions():
    small, large = (build_model(_small_config(qk_head_dim=dim), 0) for dim in (2, 64))
    small_params = jax.tree_util.tree_flatten_with_path(nnx.state(small, nnx.Param))[0]
    large_params = jax.tree_util.tree_flatten_with_path(nnx.state(large, nnx.Param))[0]
    assert [path for path, _ in small_params] == [path for path, _ in large_params]
    compared = 0
    for path, small_value in small_params:
        large_value = dict(large_params)[path]
        if "q_proj" in jax.tree_util.keystr(path) or "k_proj" in jax.tree_util.keystr(path):
            assert small_value.shape != large_value.shape
        else:
            np.testing.assert_array_equal(small_value[...], large_value[...])
            compared += 1
    assert compared > 10
