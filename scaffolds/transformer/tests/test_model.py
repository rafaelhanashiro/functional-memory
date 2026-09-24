from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.studies.key_dimension_taylor.width_sweep_configs import ModelConfig as ReferenceModelConfig
from scaffolds.transformer.models import build_model as build_reference_model
from scaffolds.transformer.configs import ATTENTION_DEGREES, ModelConfig, load_config
from scaffolds.transformer.models import build_model, model_parameter_count


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def _small_config(**overrides):
    settings = dict(vocab_size=128, max_seq_len=16, num_layers=2, hidden_size=32,
                    num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                    intermediate_size=64, tie_embeddings=False)
    return ModelConfig(**(settings | overrides))


@pytest.mark.parametrize("degree", ATTENTION_DEGREES)
def test_production_parameter_count_and_fp32_storage(degree):
    cfg = replace(load_config(BASE_CONFIG).model, attention_degree=degree)
    params = jax.eval_shape(lambda: nnx.state(build_model(cfg, 0), nnx.Param))
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    assert count == 199_906_304
    assert model_parameter_count(cfg) == count
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))


@pytest.mark.parametrize("degree", ATTENTION_DEGREES)
def test_transformer_forward_backward_causality_and_diagnostics(degree):
    cfg = _small_config(attention_degree=degree)
    model = build_model(cfg, nnx.Rngs(0))
    tokens = jnp.arange(16).reshape(2, 8)
    logits, diagnostics = model(tokens, return_diagnostics=True)
    assert logits.shape == (2, 8, 128)
    assert len(diagnostics) == cfg.num_layers
    np.testing.assert_array_equal(logits, model(tokens))
    for layer in diagnostics:
        assert float(layer["normalizer_min"]) > 0
        assert all(int(value) == 0 for key, value in layer.items() if key.endswith("_count"))
    graph, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda candidate: jnp.square(nnx.merge(graph, candidate)(tokens)).mean())(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert sum(leaf.size for leaf in jax.tree.leaves(params)) == model_parameter_count(cfg)
    altered = tokens.at[:, 5:].set(127)
    np.testing.assert_array_equal(model(altered)[:, :5], logits[:, :5])
    if degree == 0:
        for path, value in jax.tree_util.tree_flatten_with_path(grads)[0]:
            if any(name in jax.tree_util.keystr(path) for name in ("q_proj", "k_proj")):
                np.testing.assert_array_equal(value, jnp.zeros_like(value))


@pytest.mark.parametrize("degree", (None, 0, 16))
def test_remat_and_state_roundtrip_preserve_logits_and_diagnostics(degree):
    cfg = _small_config(attention_degree=degree)
    plain, remat = build_model(cfg, 4), build_model(replace(cfg, remat=True), 4)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    expected = plain(tokens, return_diagnostics=True)
    actual = remat(tokens, return_diagnostics=True)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(x, y)
    graph, params = nnx.split(plain, nnx.Param)
    np.testing.assert_array_equal(nnx.merge(graph, params)(tokens), expected[0])
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    graph, params = nnx.split(remat, nnx.Param)
    grads = jax.grad(lambda params: jnp.square(nnx.merge(graph, params)(tokens)).sum())(params)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))


def test_max_sequence_length_is_enforced():
    model = build_model(_small_config(), 0)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(jnp.ones((1, 17), dtype=jnp.int32))


def test_compatibility_backbone_initialization_is_identical_across_fixed_degrees():
    cfg = _small_config()
    reference_cfg = ReferenceModelConfig(**{
        name: value for name, value in vars(cfg).items()
        if name not in ("attention_degree", "quadratic_mode", "quadratic_sharing")
    } | {"attention_backend": "xla"})
    reference = build_reference_model(reference_cfg, 0)
    expected, expected_tree = jax.tree.flatten(nnx.state(reference, nnx.Param))
    for degree in ATTENTION_DEGREES:
        model = build_model(replace(cfg, attention_degree=degree), 0)
        actual, actual_tree = jax.tree.flatten(nnx.state(model, nnx.Param))
        assert actual_tree == expected_tree
        for observed, baseline in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(observed, baseline)
