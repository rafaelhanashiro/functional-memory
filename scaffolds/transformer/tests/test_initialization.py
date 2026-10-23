"""Reference weight initializations re-draw weights by role and keep everything else."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import attention_test_paths, matches_softmax_backbone_initialization
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.models.initialization import REFERENCE, _weight_initializer
from scaffolds.transformer.training import tracking
from scaffolds.transformer.training.checkpointing import state_sha256

LAYERS = 3


def _model_config(initialization="flax_default", mechanism="dot_product", **updates):
    values = dict(vocab_size=256, max_seq_len=16, num_layers=LAYERS, hidden_size=128, num_heads=4, num_kv_heads=4,
                  qk_head_dim=32, value_head_dim=32, intermediate_size=192, tie_embeddings=False,
                  attention_backend="blocked", attention_mechanism=mechanism, initialization=initialization)
    if mechanism == "gla":
        values.update(qk_norm=False)
    return ModelConfig(**{**values, **updates})


def _weights(config, seed=0):
    model = build_model(config, jax.random.PRNGKey(seed))
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    return {"/".join(str(part) for part in path): value for path, value in flat}, model


def _std(weights, suffix):
    return float(np.std(np.concatenate([np.asarray(v[...]).ravel() for k, v in weights.items() if k.endswith(suffix)])))


def test_flax_default_is_unchanged_and_reference_schemes_are_deterministic():
    default = _weights(_model_config())[1]
    explicit = _weights(ModelConfig(**{**_model_config().__dict__}))[1]
    assert state_sha256(nnx.state(default, nnx.Param)) == state_sha256(nnx.state(explicit, nnx.Param))
    for scheme, mechanism in (("transformer_reference", "dot_product"),
                              ("transformer_truncated_reference", "dot_product"), ("gla_reference", "gla")):
        first = nnx.state(_weights(_model_config(scheme, mechanism))[1], nnx.Param)
        again = nnx.state(_weights(_model_config(scheme, mechanism))[1], nnx.Param)
        other = nnx.state(_weights(_model_config(scheme, mechanism), seed=1)[1], nnx.Param)
        plain = nnx.state(_weights(_model_config(mechanism=mechanism))[1], nnx.Param)
        assert state_sha256(first) == state_sha256(again)
        assert len({state_sha256(first), state_sha256(other), state_sha256(plain)}) == 3


def test_transformer_reference_draws_normal_weights_and_scales_residual_projections():
    weights, model = _weights(_model_config("transformer_reference"))
    scaled = 0.02 / math.sqrt(2 * LAYERS)
    for suffix, expected in (("token_embedding/embedding", 0.02), ("lm_head/kernel", 0.02),
                             ("attention/q_proj/kernel", 0.02), ("attention/k_proj/kernel", 0.02),
                             ("attention/v_proj/kernel", 0.02), ("feed_forward/gate_proj/kernel", 0.02),
                             ("feed_forward/up_proj/kernel", 0.02), ("attention/o_proj/kernel", scaled),
                             ("feed_forward/down_proj/kernel", scaled)):
        assert abs(_std(weights, suffix) / expected - 1) < 0.05, suffix
    for key, value in weights.items():
        if key.endswith("norm/scale"):
            assert np.all(np.asarray(value[...]) == 1), key
    # Mechanism-owned parameters keep their declared initial values; metadata survives the re-draw.
    np.testing.assert_allclose(np.asarray(weights["qk_norm_log_scale"][...]), math.log(math.sqrt(32)), rtol=1e-6)
    assert model.token_embedding.embedding.get_metadata("weight_decay", True) is False


def test_gla_reference_uses_xavier_for_queries_keys_and_gates():
    weights, model = _weights(_model_config("gla_reference", "gla"))
    gain = 2 ** -2.5
    for name in ("q_proj", "k_proj", "gate_low_rank", "gate_proj"):
        for key, value in weights.items():
            if key.endswith(f"attention/{name}/kernel"):
                fan_in, fan_out = value.shape
                bound = gain * math.sqrt(6 / (fan_in + fan_out))
                array = np.asarray(value[...])
                assert np.abs(array).max() <= bound * (1 + 1e-6), key
                assert abs(array.std() * math.sqrt(3) / bound - 1) < 0.1, key
    for suffix in ("token_embedding/embedding", "lm_head/kernel", "attention/v_proj/kernel",
                   "attention/output_gate_proj/kernel", "attention/o_proj/kernel", "feed_forward/down_proj/kernel"):
        assert abs(_std(weights, suffix) / 0.02 - 1) < 0.05, suffix
    for key, value in weights.items():
        if key.endswith("gate_proj/bias"):
            assert not np.any(np.asarray(value[...])), key


def test_truncated_reference_preserves_the_conditional_gaussian_variance_and_scales_support():
    weights, model = _weights(_model_config("transformer_truncated_reference"))
    # Variance of Z | |Z| < a, where Z is standard normal: 1 - 2a phi(a)/(2 Phi(a)-1).
    variance = 1 - 4 * math.exp(-2) / math.sqrt(2 * math.pi) / math.erf(math.sqrt(2))
    for key, value in weights.items():
        if not key.endswith(("/kernel", "/embedding")):
            continue
        residual = key.endswith(("attention/o_proj/kernel", "feed_forward/down_proj/kernel"))
        std = 0.02 / math.sqrt(2 * LAYERS) if residual else 0.02
        array = np.asarray(value[...])
        assert np.all(np.abs(array) < 2 * std), key  # no clipped mass on the bounds
        assert abs(array.std() / (std * math.sqrt(variance)) - 1) < 0.05, key
        assert value.dtype == jnp.float32
    assert model.token_embedding.embedding.get_metadata("weight_decay", True) is False
    for key, value in weights.items():
        if key.endswith("norm/scale"):
            np.testing.assert_array_equal(value[...], 1)
            assert value.get_metadata("weight_decay", True) is False
    np.testing.assert_allclose(weights["qk_norm_log_scale"][...], math.log(math.sqrt(32)), rtol=1e-6)


def test_new_initialization_leaves_existing_reference_draws_and_defaults_unchanged():
    key = jax.random.PRNGKey(81)
    for path, scale in ((('token_embedding',), 0.02),
                        (('layers_0', 'attention', 'o_proj'), 0.02 / math.sqrt(2 * LAYERS))):
        actual = _weight_initializer("transformer_reference", path, LAYERS)(key, (128, 256), jnp.float32)
        expected = jax.nn.initializers.normal(scale)(key, (128, 256), jnp.float32)
        np.testing.assert_array_equal(actual, expected)
    assert REFERENCE == {"dot_product": "transformer_reference", "gla": "gla_reference",
                         "gated_deltanet": "gated_deltanet_reference"}


@pytest.mark.parametrize("scheme,mechanism", [("transformer_reference", "gla"), ("gla_reference", "dot_product"),
                                              ("transformer_truncated_reference", "gla"),
                                              ("xavier", "dot_product")])
def test_mismatched_or_unknown_schemes_are_rejected(scheme, mechanism):
    with pytest.raises(ValueError, match="initialization"):
        _model_config(scheme, mechanism)


def test_recipes_without_the_field_tags_and_acceptance():
    config = Config()
    config.model = _model_config("transformer_reference")
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 128, 9, 2
    config.train.micro_batch_size = 2
    validate_config(config)
    values = config_to_dict(config)
    assert values["model"]["initialization"] == "transformer_reference"
    legacy = config_to_dict(Config())
    stripped = {**legacy, "model": {k: v for k, v in legacy["model"].items() if k != "initialization"}}
    assert config_dicts_equal(legacy, stripped)
    assert "init_transformer_reference" in tracking.run_tags(config, {})
    # The softmax counterpart is built with the same scheme, so backbones still match.
    matches_softmax_backbone_initialization(config)
    assert "scaffolds/transformer/tests/test_initialization.py" in attention_test_paths(values, "fused", True)
    assert "scaffolds/transformer/tests/test_initialization.py" not in attention_test_paths(
        config_to_dict(Config()), "fused", True)
