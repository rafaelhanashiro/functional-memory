from dataclasses import replace
from functools import partial
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, telemetry
from scaffolds.transformer.training.distributed import (
    make_data_parallel_functions, put_replicated, put_train_batches, resolve_batch_layout,
)
from scaffolds.transformer.training.state import create_state

# Two microbatches of four five-token sequences.
TOKENS = np.arange(40, dtype=np.int64).reshape(2, 4, 5) % 16


def _devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("Requires two host devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    return devices[:2]


def _run(cfg, devices, *, collect):
    state = create_state(cfg, jax.random.PRNGKey(0))
    functions = make_data_parallel_functions(state, devices, check_all_gradients=True, collect_telemetry=collect)
    state = put_replicated(state, functions.replicated_sharding)
    batches = put_train_batches(TOKENS, functions.train_batch_sharding)
    outputs = functions.train_step(state, batches, jax.random.PRNGKey(3))
    return functions, outputs


def _single_device_update(cfg, *, collect, accumulate):
    state = create_state(cfg, jax.random.PRNGKey(0))
    rngs = jax.random.split(jax.random.PRNGKey(3), TOKENS.shape[0])
    recording = {"collect_telemetry": True} if collect else {}
    if accumulate:
        return loop.accumulated_train_step(state, jnp.asarray(TOKENS), rngs, check_all_gradients=True, **recording)
    return loop.train_step(state, jnp.asarray(TOKENS[0]), rngs[0], check_all_gradients=True, **recording)


def _single_device_training(monkeypatch):
    """Train on one device even when the test process exposes several."""
    monkeypatch.setattr(loop, "resolve_batch_layout", partial(resolve_batch_layout, available_device_count=1))
    monkeypatch.setattr(loop, "select_data_parallel_devices", lambda layout: tuple(jax.local_devices()[:1]))


@pytest.mark.parametrize("scaling", ["none", "head_rms_norm"])
def test_distributed_telemetry_leaves_the_update_unchanged(tmp_path, scaling):
    devices = _devices()
    cfg = tiny_config(tmp_path)
    cfg.model = replace(cfg.model, attention_output_scaling=scaling, attention_normalization=scaling == "none")
    _, plain = _run(cfg, devices, collect=False)
    functions, observed = _run(cfg, devices, collect=True)
    assert len(plain) == 3 and len(observed) == 4
    np.testing.assert_array_equal(np.asarray(plain[1]), np.asarray(observed[1]))
    np.testing.assert_array_equal(np.asarray(plain[2]), np.asarray(observed[2]))
    for before, after in zip(jax.tree.leaves(plain[0].params), jax.tree.leaves(observed[0].params), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    statistics = dict(zip(functions.telemetry_keys, np.asarray(observed[3]).tolist(), strict=True))
    assert functions.telemetry_kinds["layer_00/attention_output_head_rms_min"] == "min"
    assert functions.telemetry_kinds["update/mlp/gradient_norm"] == "norm"
    assert all(np.isfinite(value) for value in statistics.values())
    expected = {"layer_00/residual_rms", "layer_00/residual_max_abs", "layer_00/attention_output_rms",
                "layer_00/attention_output_head_rms_min", "final/residual_rms", "final/residual_max_abs"}
    assert expected <= statistics.keys()
    assert ("layer_00/head_rms_norm_gain_max" in statistics) == (scaling == "head_rms_norm")
    assert 0 < statistics["layer_00/attention_output_head_rms_min"] <= statistics["layer_00/attention_output_rms"]
    # The global gradient norm equals the norm of the per-group gradient norms.
    group_norms = [value for key, value in statistics.items() if key.endswith("/gradient_norm")]
    np.testing.assert_allclose(np.sqrt(np.sum(np.square(group_norms))), float(np.asarray(observed[2])[1]), rtol=1e-5)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("scaling", ["none", "head_rms_norm"])
def test_single_device_telemetry_leaves_the_update_unchanged(tmp_path, scaling, accumulate):
    cfg = tiny_config(tmp_path)
    cfg.model = replace(cfg.model, attention_output_scaling=scaling, attention_normalization=scaling == "none")
    plain = _single_device_update(cfg, collect=False, accumulate=accumulate)
    observed = _single_device_update(cfg, collect=True, accumulate=accumulate)
    assert len(plain) == 3 and len(observed) == 4
    np.testing.assert_array_equal(np.asarray(plain[1]), np.asarray(observed[1]))
    np.testing.assert_array_equal(np.asarray(plain[2]), np.asarray(observed[2]))
    for before, after in zip(jax.tree.leaves(plain[0].params), jax.tree.leaves(observed[0].params), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    statistics = {key: float(value) for key, value in observed[3].items()}
    assert all(np.isfinite(value) for value in statistics.values())
    expected = {"layer_00/residual_rms", "layer_00/residual_max_abs", "layer_00/attention_output_rms",
                "layer_00/attention_output_head_rms_min", "final/residual_rms", "final/residual_max_abs",
                "update/mlp/gradient_norm", "update/mlp/parameter_norm", "update/mlp/update_norm"}
    assert expected <= statistics.keys()
    assert ("layer_00/head_rms_norm_gain_max" in statistics) == (scaling == "head_rms_norm")
    assert 0 < statistics["layer_00/attention_output_head_rms_min"] <= statistics["layer_00/attention_output_rms"]
    group_norms = [value for key, value in statistics.items() if key.endswith("/gradient_norm")]
    np.testing.assert_allclose(np.sqrt(np.sum(np.square(group_norms))), float(np.asarray(observed[2])[1]), rtol=1e-5)
    # The host fetches the statistics as one vector in sorted key order.
    np.testing.assert_array_equal(np.asarray(telemetry.pack(observed[3])),
                                  np.asarray([observed[3][key] for key in sorted(observed[3])]))


def test_single_device_statistics_match_distributed_training(tmp_path):
    # Equal-sized microbatches on one device or split across two summarize the same batch alike.
    devices = _devices()
    cfg = tiny_config(tmp_path)
    cfg.model = replace(cfg.model, attention_output_scaling="head_rms_norm", attention_normalization=False)
    functions, distributed = _run(cfg, devices, collect=True)
    expected = dict(zip(functions.telemetry_keys, np.asarray(distributed[3]).tolist(), strict=True))
    single = _single_device_update(cfg, collect=True, accumulate=True)
    observed = {key: float(value) for key, value in single[3].items()}
    assert observed.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(observed[key], value, rtol=1e-4, atol=1e-7, err_msg=key)


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
    assert telemetry.parameter_group(keys("layers_0", "attention", "lambda_proj", "kernel")) == "attention_erase_gate"
    assert telemetry.parameter_group(keys("layers_0", "attention", "b_proj", "kernel")) == "attention_write_gate"
    assert telemetry.parameter_group(keys("layers_0", "attention", "a_proj", "kernel")) == "attention_decay_gate"
    assert telemetry.parameter_group(keys("layers_0", "attention", "A_log")) == "attention_decay_gate"
    assert telemetry.parameter_group(keys("layers_0", "attention", "o_norm_scale")) == "attention_output_norm"
    assert telemetry.parameter_group(keys("polynomial_coefficients")) == "attention_coefficients"
    assert telemetry.parameter_group(keys("score_mlp_offset_kernel")) == "attention_score_mlp"
    assert telemetry.parameter_group(keys("lm_head", "kernel")) == "output_head"


def test_window_summarizes_updates_and_records_gradient_spikes():
    window = telemetry.TelemetryWindow()
    keys = ["layer_00/attention_output_head_rms_min", "update/mlp/gradient_norm"]
    for step in range(1, 31):
        assert window.add(step, keys, [0.5, 1.0], 1.0) is None
    spike = window.add(31, keys, [0.01, 9.0], 9.0)
    assert spike["step"] == 31 and spike["running_median"] == 1.0
    assert spike["statistics"]["layer_00/attention_output_head_rms_min"] == 0.01
    summary = window.summary(keys)
    assert summary["train_telemetry_update_count"] == 31 and summary["train_gradient_spike_count"] == 1
    assert summary["train_telemetry"]["layer_00/attention_output_head_rms_min"]["min"] == 0.01
    assert summary["train_telemetry"]["update/mlp/gradient_norm"]["max"] == 9.0
    window.reset()
    assert window.summary(keys) == {}
    # The running median spans validation windows.
    assert window.add(32, keys, [0.5, 1.0], 6.0) is not None


def test_log_summary_takes_extremes_over_layers():
    values = {"layer_00/attention_output_head_rms_min": 0.3, "layer_01/attention_output_head_rms_min": 0.1,
              "layer_00/residual_max_abs": 2.0, "final/residual_max_abs": 5.0}
    assert telemetry.log_summary(values) == {"attention_output_head_rms_min": 0.1, "residual_max_abs": 5.0}


def test_distributed_training_records_telemetry(tmp_path, capsys):
    _devices()
    cfg = tiny_config(tmp_path)
    loop.run_training(cfg)
    output = capsys.readouterr().out
    assert "attention_output_head_rms_min=" in output and "residual_max_abs=" in output
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["train_telemetry_update_count"] for record in records] == [2, 2]
    summary = records[0]["train_telemetry"]
    assert {"layer_00/attention_output_head_rms_min", "final/residual_rms", "update/mlp/gradient_norm"} <= summary.keys()
    assert summary["update/mlp/gradient_norm"]["min"] <= summary["update/mlp/gradient_norm"]["max"]
    assert records[0]["train_gradient_spike_count"] == 0 and records[0]["train_gradient_spikes"] == []


@pytest.mark.parametrize("micro_batch_size", [2, 4])
def test_single_device_training_records_telemetry(tmp_path, monkeypatch, capsys, micro_batch_size):
    # Microbatch 2 accumulates two microbatches per update; 4 is one batch per update.
    _single_device_training(monkeypatch)
    cfg = tiny_config(tmp_path)
    cfg.train.micro_batch_size = micro_batch_size
    loop.run_training(cfg)
    output = capsys.readouterr().out
    assert "data_parallel_devices=1 " in output
    assert "attention_output_head_rms_min=" in output and "residual_max_abs=" in output
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["train_telemetry_update_count"] for record in records] == [2, 2]
    summary = records[0]["train_telemetry"]
    assert {"layer_00/attention_output_head_rms_min", "final/residual_rms", "update/mlp/gradient_norm"} <= summary.keys()
    assert summary["update/mlp/gradient_norm"]["min"] <= summary["update/mlp/gradient_norm"]["max"]
    assert records[0]["train_gradient_spike_count"] == 0 and records[0]["train_gradient_spikes"] == []


def test_single_device_numerical_failure_keeps_telemetry(tmp_path, monkeypatch):
    _single_device_training(monkeypatch)
    cfg = tiny_config(tmp_path)
    cfg.train.log_every, cfg.train.eval_every, cfg.train.ckpt_every = 3, 4, 4
    actual = loop.accumulated_train_step

    def failing_second_update(*args, **kwargs):
        updated, loss, *rest = actual(*args, **kwargs)
        return (updated, jnp.where(updated.step == 2, jnp.nan, loss), *rest)

    monkeypatch.setattr(loop, "accumulated_train_step", failing_second_update)
    with pytest.raises(FloatingPointError, match="at step 2"):
        loop.run_training(cfg)
    failure = json.loads((tmp_path / "numerical_failure_step_00000002.json").read_text())
    recorded = failure["training_telemetry"]
    assert recorded["failed_update"]["layer_00/attention_output_head_rms_min"] > 0
    assert recorded["preceding_update"]["update/mlp/gradient_norm"] > 0


def test_remat_training_records_no_telemetry(tmp_path, monkeypatch):
    # Recorded values cannot leave a rematerialized block, so remat training skips telemetry.
    _single_device_training(monkeypatch)
    cfg = tiny_config(tmp_path)
    cfg.model = replace(cfg.model, remat=True)
    loop.run_training(cfg)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records and not any("train_telemetry" in record for record in records)
    assert all(record["train_gradient_norm_count"] == 2 for record in records)


def test_distributed_numerical_failure_keeps_telemetry(tmp_path, monkeypatch):
    _devices()
    cfg = tiny_config(tmp_path)
    cfg.train.log_every, cfg.train.eval_every, cfg.train.ckpt_every = 3, 4, 4
    actual = loop.make_data_parallel_functions

    def failing_second_update(*args, **kwargs):
        functions = actual(*args, **kwargs)

        def train_step(state, batches, rng):
            updated, loss, *rest = functions.train_step(state, batches, rng)
            return (updated, jnp.where(updated.step == 2, jnp.nan, loss), *rest)

        return replace(functions, train_step=train_step)

    monkeypatch.setattr(loop, "make_data_parallel_functions", failing_second_update)
    with pytest.raises(FloatingPointError, match="at step 2"):
        loop.run_training(cfg)
    failure = json.loads((tmp_path / "numerical_failure_step_00000002.json").read_text())
    recorded = failure["training_telemetry"]
    assert recorded["failed_update"]["layer_00/attention_output_head_rms_min"] > 0
    assert recorded["preceding_update"]["update/mlp/gradient_norm"] > 0
