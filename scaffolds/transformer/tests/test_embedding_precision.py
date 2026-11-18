"""Forward and backward precision of the model's token lookup with repeated IDs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.training.state import create_state


@pytest.mark.parametrize("platform", ("cpu", pytest.param("gpu", marks=pytest.mark.gpu)))
@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_repeated_tokens_accumulate_in_fp32(platform, dtype):
    if platform == "gpu" and jax.devices()[0].platform != "gpu":
        pytest.skip("GPU required")
    with jax.default_device(jax.devices(platform)[0]):
        cfg = Config()
        cfg.model = ModelConfig(vocab_size=16, hidden_size=8, max_seq_len=4, num_layers=1,
                                num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=4,
                                intermediate_size=16, attention_backend="blocked")
        cfg.data.synthetic_vocab_size = cfg.model.vocab_size
        cfg.train.dtype = dtype
        state = create_state(cfg, jax.random.PRNGKey(0))
        tokens = jnp.array([[3, 9], [7, 3]], dtype=jnp.int32)
        # Both contributions to row 3 fit in BF16, but their sum rounds to 1 in BF16.
        contributions = jnp.array([[1.0, 2.0], [-0.5, 2.0**-10]], dtype=dtype)
        cotangent = jnp.broadcast_to(contributions[..., None], (*tokens.shape, cfg.model.hidden_size))

        def lookup(params):
            return nnx.merge(state.graphdef, params).token_embedding(tokens)

        @jax.jit
        def forward_backward(params, cotangent):
            embeddings, pullback = jax.vjp(lookup, params)
            (gradients,) = pullback(cotangent)
            return embeddings, gradients["token_embedding"]["embedding"][...]

        rows, gradient = forward_backward(state.params, cotangent)
        table = state.params["token_embedding"]["embedding"][...]
        assert table.dtype == gradient.dtype == jnp.float32
        assert rows.dtype == jnp.dtype(dtype)
        np.testing.assert_array_equal(rows, table.astype(dtype)[tokens])
        expected = np.zeros(table.shape, dtype=np.float32)
        np.add.at(expected, np.asarray(tokens), np.asarray(cotangent, dtype=np.float32))
        np.testing.assert_array_equal(gradient, expected)
