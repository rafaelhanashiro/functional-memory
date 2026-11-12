"""positional_encoding 'none' skips RoPE and composes with the forget gate."""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import multihead
from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.training.randomness import model_forward_rngs
from scaffolds.transformer.training.state import create_state


def _config(**changes):
    """The forget-gate test recipe: query-only m=4 tanh coefficients on the blocked backend."""
    model = dict(vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=4,
                 attention_backend="blocked", attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.0,
                 attention_coefficient_projection_initial_std=8 ** -.5, attention_coefficient_amplitude_scale="factorial",
                 attention_coefficient_scale="factorial", attention_output_scaling="head_rms_norm", qk_norm=True,
                 qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    return config


def test_none_validates_with_and_without_the_forget_gate():
    _config(positional_encoding="none")
    _config(positional_encoding="none", attention_forget_gate=True)
    with pytest.raises(ValueError, match="positional encoding"):
        _config(positional_encoding="sinusoidal")
    _config(positional_encoding="path", attention_forget_gate=True)
    with pytest.raises(ValueError, match="forget gate"):
        _config(positional_encoding="none", attention_forget_gate=True, attention_backend="fused")


def test_none_never_applies_rope(monkeypatch):
    calls = []
    original = multihead.apply_rope
    monkeypatch.setattr(multihead, "apply_rope", lambda *a, **k: calls.append(1) or original(*a, **k))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 16, size=(2, 8)), jnp.int32)
    logits = {}
    for encoding in ("rope", "none"):
        state = create_state(_config(positional_encoding=encoding, attention_forget_gate=True), jax.random.PRNGKey(0))
        model = nnx.merge(state.graphdef, state.params)
        before = len(calls)
        logits[encoding] = np.asarray(model(tokens, deterministic=True, rngs=model_forward_rngs(jax.random.PRNGKey(1))))
        assert (len(calls) > before) == (encoding == "rope"), encoding
    assert np.isfinite(logits["none"]).all()
    assert not np.allclose(logits["none"], logits["rope"])
