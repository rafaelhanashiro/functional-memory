"""Attention-score diagnostic: binned statistics against a direct computation, and call capture."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from experiments.transformer_pp.diagnose_attention_scores import AttentionRecorder, head_statistics, kernel_kind
from experiments.transformer_pp.evaluate_per_position import load_model_config
from scaffolds.transformer.configs import Config, config_to_dict
from scaffolds.transformer.training.state import create_state


def _direct(q, k, coefficients, kind, degree, bin_width, bins, training_length):
    time, width = q.shape
    scores = q @ k.T / math.sqrt(width)
    rows = []
    stats = {name: np.zeros(bins) for name in ("count", "sum", "square", "late")}
    minimum, maximum = np.full(bins, np.inf), np.full(bins, -np.inf)
    far = 0.0
    for t in range(time):
        s = scores[t, :t + 1]
        if kind == "exp":
            w = np.exp(s - s.max())
        elif kind == "taylor":
            w = sum(s ** r / math.factorial(r) for r in range(degree + 1))
        else:
            w = sum(coefficients[t, r] * s ** r for r in range(degree + 1))
        share = np.abs(w) / np.abs(w).sum()
        for i in range(t + 1):
            b = min((t - i) // bin_width, bins - 1)
            stats["count"][b] += 1; stats["sum"][b] += s[i]; stats["square"][b] += s[i] ** 2
            minimum[b], maximum[b] = min(minimum[b], s[i]), max(maximum[b], s[i])
            if t >= training_length:
                stats["late"][b] += share[i]
                far += share[i] if t - i >= training_length else 0.0
    return stats["count"], stats["sum"], stats["square"], minimum, maximum, stats["late"], far


@pytest.mark.parametrize("kind,degree", [("exp", 0), ("taylor", 4), ("query", 4)])
def test_head_statistics_match_a_direct_computation(kind, degree):
    rng = np.random.default_rng(3)
    q, k = rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
    coefficients = rng.normal(size=(12, degree + 1)) if kind == "query" else np.zeros((12, 1))
    actual = head_statistics(jnp.asarray(q, jnp.float32), jnp.asarray(k, jnp.float32), jnp.asarray(coefficients, jnp.float32),
                             kind=kind, degree=degree, bin_width=4, bins=3, training_length=8)
    expected = _direct(q, k, coefficients, kind, degree, 4, 3, 8)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=2e-5, atol=1e-5)


def test_recorder_captures_every_layer_and_names_the_kernel(tmp_path):
    config = Config()
    config.model.vocab_size, config.model.max_seq_len, config.model.num_layers = 32, 8, 2
    config.model.hidden_size, config.model.num_heads, config.model.num_kv_heads = 16, 2, 2
    config.model.qk_head_dim = config.model.value_head_dim = 8
    config.model.intermediate_size, config.model.attention_backend = 24, "blocked"
    config.model.attention_normalization, config.model.attention_degree = False, 2
    config.data.source, config.data.sequence_length, config.data.synthetic_vocab_size = "synthetic", 9, 32
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(config)))
    config = load_model_config({"config_path": str(path)})
    state = create_state(config, jax.random.PRNGKey(0))
    from flax import nnx
    model = nnx.merge(state.graphdef, state.params)
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 32, size=(1, 8)), jnp.int32)
    with AttentionRecorder() as recorder:
        model(tokens, deterministic=True)
    assert len(recorder.calls) == 2
    q, k, arguments = recorder.calls[0]
    assert q.shape == k.shape == (1, 8, 2, 8)
    assert kernel_kind(arguments) == "taylor" and arguments["attention_degree"] == 2
