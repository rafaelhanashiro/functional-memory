"""FP32 backbone normalization is selectable without changing legacy arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.models.components.normalization import RMSNorm


def test_fp32_rms_norm_matches_forward_and_gradient_reference_for_bfloat16_inputs():
    x = (jax.random.normal(jax.random.PRNGKey(0), (2, 3, 64)) * 7.3).astype(jnp.bfloat16)
    gains = jnp.linspace(0.5, 1.5, 64)
    norm = RMSNorm(64, fp32=True)
    norm.scale[...] = gains

    def reference(value):
        value = value.astype(jnp.float32)
        return value * jax.lax.rsqrt(jnp.mean(value ** 2, axis=-1, keepdims=True) + 1e-6) * gains

    actual = norm(x)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(actual, reference(x))
    np.testing.assert_array_equal(jax.grad(lambda value: jnp.sum(norm(value) ** 2))(x),
                                  jax.grad(lambda value: jnp.sum(reference(value) ** 2))(x))
    assert norm.scale.get_metadata("weight_decay", True) is False


def test_legacy_rms_norm_arithmetic_is_unchanged():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16)).astype(jnp.bfloat16)
    norm = RMSNorm(16)
    expected = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
    expected = expected * norm.scale[...]
    np.testing.assert_array_equal(norm(x), expected)
    assert not np.array_equal(norm(x), RMSNorm(16, fp32=True)(x))


def test_fp32_norm_recipe_wires_every_backbone_norm_and_preserves_legacy_identity():
    config = Config()
    legacy = config_to_dict(config)
    old = {**legacy, "model": {key: value for key, value in legacy["model"].items()
                               if key != "rms_norm_fp32"}}
    assert config_dicts_equal(old, legacy)
    config.model.rms_norm_fp32 = True
    assert not config_dicts_equal(old, config_to_dict(config))
    model = build_model(config.model, jax.random.PRNGKey(2))
    norms = [module for _, module in nnx.iter_graph(model) if isinstance(module, RMSNorm)]
    assert len(norms) == 2 * config.model.num_layers + 1
    assert all(norm.fp32 for norm in norms)


@pytest.mark.parametrize("value", [1, "float32", None])
def test_rms_norm_precision_option_requires_boolean(value):
    with pytest.raises(ValueError, match="rms_norm_fp32 must be a boolean"):
        ModelConfig(rms_norm_fp32=value)
