from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import telemetry
from scaffolds.transformer.training.distributed import make_data_parallel_functions, put_replicated, put_train_batches
from scaffolds.transformer.training.state import create_state


def _devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("Requires two host devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    return devices[:2]


def _run(cfg, devices, *, collect):
    state = create_state(cfg, jax.random.PRNGKey(0))
    functions = make_data_parallel_functions(state, devices, check_all_gradients=True, collect_telemetry=collect)
    state = put_replicated(state, functions.replicated_sharding)
    batches = put_train_batches(np.arange(40, dtype=np.int64).reshape(2, 4, 5) % 16, functions.train_batch_sharding)
    outputs = functions.train_step(state, batches, jax.random.PRNGKey(3))
    return state, outputs


@pytest.mark.parametrize("scaling", ["none", "head_rms_norm"])
def test_telemetry_leaves_the_update_unchanged(tmp_path, scaling):
    devices = _devices()
    cfg = tiny_config(tmp_path)
    cfg.model = replace(cfg.model, attention_output_scaling=scaling, attention_normalization=scaling == "none")
    _, plain = _run(cfg, devices, collect=False)
    _, observed = _run(cfg, devices, collect=True)
    assert len(plain) == 3 and len(observed) == 4
    np.testing.assert_array_equal(np.asarray(plain[1]), np.asarray(observed[1]))
    np.testing.assert_array_equal(np.asarray(plain[2]), np.asarray(observed[2]))
    for before, after in zip(jax.tree.leaves(plain[0].params), jax.tree.leaves(observed[0].params), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    statistics = {key: float(value) for key, value in observed[3].items()}
    assert all(np.isfinite(value) for value in statistics.values())
    expected = {"layer_00/residual_rms", "layer_00/residual_max_abs", "layer_00/attention_output_rms",
                "layer_00/attention_output_head_rms_min", "final/residual_rms", "final/residual_max_abs"}
    assert expected <= statistics.keys()
    assert ("layer_00/head_rms_norm_gain_max" in statistics) == (scaling == "head_rms_norm")
    assert 0 < statistics["layer_00/attention_output_head_rms_min"] <= statistics["layer_00/attention_output_rms"]
    # The global gradient norm equals the norm of the per-group gradient norms.
    group_norms = [value for key, value in statistics.items() if key.endswith("/gradient_norm")]
    np.testing.assert_allclose(np.sqrt(np.sum(np.square(group_norms))), float(np.asarray(observed[2])[1]), rtol=1e-5)


def test_activation_statistics_match_direct_computation(tmp_path):
    cfg = tiny_config(tmp_path)
    state = create_state(cfg, jax.random.PRNGKey(0))
    model = nnx.merge(state.graphdef, state.params)
    tokens = jnp.asarray(np.arange(8).reshape(2, 4) % 16)
    with telemetry.collect() as collector:
        logits = model(tokens, deterministic=True)
    values = {key: float(value) for key, value in collector.values.items()}
    embedded = model.token_embedding(tokens).astype(jnp.float32)
    np.testing.assert_allclose(values["layer_00/residual_rms"], float(jnp.mean(jnp.square(embedded))), rtol=1e-6)
    np.testing.assert_allclose(values["layer_00/residual_max_abs"], float(jnp.max(jnp.abs(embedded))), rtol=1e-6)
    assert collector.reductions["layer_00/residual_rms"] == "rms"
    assert collector.reductions["layer_00/attention_output_head_rms_min"] == "min"
    # Outside a collector, recording is a no-op and the forward is unchanged.
    np.testing.assert_array_equal(np.asarray(model(tokens, deterministic=True)), np.asarray(logits))


def test_reductions_combine_microbatches_and_devices():
    stacked = {"a": jnp.asarray([1.0, 3.0]), "b": jnp.asarray([2.0, -1.0]), "c": jnp.asarray([4.0, 16.0])}
    kinds = {"a": "mean", "b": "min", "c": "rms"}
    reduced = telemetry.reduce_microbatches(stacked, kinds)
    assert float(reduced["a"]) == 2.0 and float(reduced["b"]) == -1.0 and float(reduced["c"]) == 10.0


def test_parameter_groups_cover_the_model_roles():
    keys = lambda *names: tuple(jax.tree_util.DictKey(name) for name in names)
    assert telemetry.parameter_group(keys("token_embedding", "embedding")) == "embedding"
    assert telemetry.parameter_group(keys("layers_3", "attention", "o_proj", "kernel")) == "attention_output_projection"
    assert telemetry.parameter_group(keys("layers_3", "attention", "q_proj", "kernel")) == "attention_qkv"
    assert telemetry.parameter_group(keys("layers_3", "feed_forward", "up_proj", "kernel")) == "mlp"
    assert telemetry.parameter_group(keys("layers_3", "ffn_norm", "scale")) == "norm_scales"
    assert telemetry.parameter_group(keys("qk_norm_log_scale")) == "qk_norm_scale"
    assert telemetry.parameter_group(keys("output_norm_scale")) == "attention_output_norm"
    assert telemetry.parameter_group(keys("polynomial_coefficients")) == "attention_coefficients"
    assert telemetry.parameter_group(keys("lm_head", "kernel")) == "output_head"
