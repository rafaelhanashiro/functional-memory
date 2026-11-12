"""Compiled accumulation preserves the Python accumulator's update semantics."""
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.benchmarks.benchmark_accumulation import python_accumulation
from scaffolds.transformer.tests.test_training import tiny_config, assert_state_close
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.state import create_state


@pytest.mark.parametrize("microbatches", [1, 4])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_compiled_updates_match_python_with_rngs_and_optimizer_state(tmp_path, microbatches, dropout):
    config = tiny_config(tmp_path)
    config.model.attention_backend = "manual"
    config.model.dropout = dropout
    initial = create_state(config, jax.random.PRNGKey(0))
    reference = compiled = initial
    batches = jnp.asarray(np.arange(20).reshape(microbatches, 4 // microbatches, 5) % 16)
    for step in range(3):
        keys = jax.random.split(jax.random.PRNGKey(step), microbatches)
        reference, loss, numerics = python_accumulation(reference, batches, keys)
        compiled, actual_loss, actual_numerics = loop.accumulated_train_step(
            compiled, batches, keys, check_all_gradients=True)
        assert_state_close(reference, compiled, atol=2e-7)
        np.testing.assert_allclose(actual_loss, loss, rtol=2e-6)
        np.testing.assert_allclose(actual_numerics, numerics, rtol=2e-6)
        assert int(compiled.step) == step + 1


@pytest.mark.parametrize("count,rng_count", [(0, 0), (2, 1)])
def test_accumulation_rejects_empty_or_mismatched_inputs(tmp_path, count, rng_count):
    state = create_state(tiny_config(tmp_path), jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="nonempty batches and one RNG"):
        loop.accumulated_train_step(state, jnp.zeros((count, 2, 5), jnp.int32),
                                    jax.random.split(jax.random.PRNGKey(1), rng_count))


@pytest.mark.parametrize("include_denominator", [False, True])
def test_compiled_accumulation_matches_python_with_coefficient_penalty(tmp_path, include_denominator):
    from scaffolds.transformer.tests.test_learned_polynomial_rational import _config

    config = _config(tmp_path, qk_norm=True, penalty=0.25, initialization="taylor")
    initial = create_state(config, jax.random.PRNGKey(0))
    batches = jnp.asarray(np.arange(14).reshape(2, 1, 7) % 16)
    keys = jax.random.split(jax.random.PRNGKey(2), 2)
    kwargs = {"coefficient_penalty": 0.25, "coefficient_penalty_denominator": include_denominator}
    expected, loss, numerics = python_accumulation(initial, batches, keys, **kwargs)
    actual, actual_loss, actual_numerics = loop.accumulated_train_step(
        initial, batches, keys, check_all_gradients=True, **kwargs)
    assert_state_close(expected, actual, atol=2e-7)
    np.testing.assert_allclose(actual_loss, loss, rtol=2e-6)
    np.testing.assert_allclose(actual_numerics, numerics, rtol=2e-6)


@pytest.mark.parametrize("implementation", ["python", "compiled"])
def test_benchmark_uses_full_updates_and_preserves_source_recipe(tmp_path, monkeypatch, implementation):
    from scaffolds.transformer.benchmarks import benchmark_accumulation as benchmark
    from scaffolds.transformer.configs import save_config

    config = tiny_config(tmp_path / "unused")
    source = tmp_path / "recipe.yaml"
    save_config(config, source)
    original = source.read_bytes()
    monkeypatch.setattr(benchmark.jax, "local_devices", lambda: [SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": 123})])
    monkeypatch.setattr(benchmark, "_execution_provenance", lambda *_: {"source_commit": "a" * 40})
    output = tmp_path / "report.json"
    report = benchmark.benchmark(SimpleNamespace(
        config=source, output=output, micro_batch_size=1, implementation=implementation,
        warmup=1, iterations=2))
    assert report["status"] == "complete"
    assert report["accumulation_steps"] == 4
    assert report["configuration"]["data"]["batch_size"] == 4
    assert report["configuration"]["train"]["micro_batch_size"] == 1
    assert report["input_recipe"]["configuration"]["train"]["micro_batch_size"] == 2
    assert [sample["step"] for sample in report["samples"]] == [1, 2, 3]
    assert report["steady_compute"]["updates"] == 2
    assert report["tokens_per_update"] == 16
    assert json.loads(output.read_text()) == report
    assert source.read_bytes() == original
    assert not config.train.workdir.exists()
