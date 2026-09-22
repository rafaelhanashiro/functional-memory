"""Host placement must preserve every physical replica and resumed updates.

Run with JAX_PLATFORMS=cpu and
XLA_FLAGS=--xla_force_host_platform_device_count=4 for the multi-device checks.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import distributed, loop
from scaffolds.transformer.training.state import create_state


@pytest.fixture
def devices():
    if jax.local_device_count() < 4:
        pytest.skip("Requires four JAX devices")
    return tuple(jax.local_devices()[:4])


@pytest.mark.parametrize("value", ["", "true", "yes", "2", "-1"])
def test_host_staging_rejects_ambiguous_flags(monkeypatch, value):
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        distributed._host_staging_enabled()


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_host_staging_is_explicitly_opt_in(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(distributed.HOST_STAGING_ENV, raising=False)
    else:
        monkeypatch.setenv(distributed.HOST_STAGING_ENV, value)
    assert distributed._host_staging_enabled() is expected


def test_host_state_transfer_preserves_parameters_moments_and_counters(tmp_path, monkeypatch, devices):
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, "1")
    state = create_state(tiny_config(tmp_path), jax.random.PRNGKey(3))
    # Nonzero moments/counts catch accidentally resetting an optimizer on resume.
    state = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    sharding = NamedSharding(Mesh(np.asarray(devices), ("data",)), P())
    original_put = jax.device_put
    sources = []

    def checked_put(value, target, *args, **kwargs):
        if target is sharding:
            assert isinstance(value, np.ndarray), "A GPU source would require peer copying"
            sources.append((value.dtype, value.shape))
        return original_put(value, target, *args, **kwargs)

    monkeypatch.setattr(distributed.jax, "device_put", checked_put)
    actual = distributed.put_replicated(state, sharding)
    assert int(actual.step) == 1
    assert len(sources) == len(jax.tree.leaves(state))
    assert any(shape == () and np.issubdtype(dtype, np.integer) for dtype, shape in sources)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(actual.params))
    for expected, placed in zip(jax.tree.leaves(state), jax.tree.leaves(actual), strict=True):
        host = np.asarray(expected)
        assert len(placed.addressable_shards) == 4
        for shard in placed.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), host, strict=True)


def test_corrupt_nonprimary_replica_is_rejected(monkeypatch, devices):
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, "1")
    sharding = NamedSharding(Mesh(np.asarray(devices), ("data",)), P())
    original_put = jax.device_put

    def corrupt_put(value, target, *args, **kwargs):
        placed = original_put(value, target, *args, **kwargs)
        if target is sharding:
            arrays = [shard.data for shard in placed.addressable_shards]
            changed = np.asarray(value).copy()
            # A byte-exact check must catch signed-zero corruption as well.
            changed[0] = np.float32(-0.0)
            arrays[2] = original_put(changed, devices[2])
            return jax.make_array_from_single_device_arrays(value.shape, target, arrays)
        return placed

    state = {"parameters": original_put(np.asarray([0.0, 2.0], np.float32), devices[0])}
    monkeypatch.setattr(distributed.jax, "device_put", corrupt_put)
    with pytest.raises(RuntimeError, match="replica mismatch.*parameters.*cpu:2"):
        distributed.put_replicated(state, sharding)


@pytest.mark.parametrize("host_staging", ["0", "1"])
def test_host_batch_shards_and_rng_are_exact(tmp_path, monkeypatch, devices, host_staging):
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, host_staging)
    state = create_state(tiny_config(tmp_path), jax.random.PRNGKey(3))
    functions = distributed.make_data_parallel_functions(state, devices)
    train = np.arange(80, dtype=np.int64).reshape(2, 8, 5)
    evaluation = np.arange(40, dtype=np.int64).reshape(8, 5)
    for source, placed in (
        (train, distributed.put_train_batches(train, functions.train_batch_sharding)),
        (evaluation, distributed.put_eval_batch(evaluation, functions.eval_batch_sharding)),
    ):
        assert placed.dtype == jnp.int32
        for shard in placed.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), source[shard.index].astype(np.int32))
    rng = jax.random.fold_in(jax.random.PRNGKey(17), 19)
    placed_rng = distributed._put_from_host(rng, functions.replicated_sharding)
    for shard in placed_rng.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(rng))


def test_host_staged_checkpoint_resume_matches_default_training(tmp_path, monkeypatch, devices):
    assert len(devices) == jax.local_device_count(), "Run this check with exactly four devices"
    config = tiny_config(tmp_path / "default")
    config.model.attention_degree = 2
    config.data.batch_size = 8
    config.train.micro_batch_size = 4
    config.optim.total_steps = 3
    config.train.eval_every = config.train.ckpt_every = 1
    monkeypatch.delenv(distributed.HOST_STAGING_ENV, raising=False)
    full = loop.run_training(config, audit_checkpoint=True)

    staged = replace(config, train=replace(config.train, workdir=tmp_path / "host_staged"))
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, "1")
    original_put = jax.device_put
    staged_rngs = []

    def no_peer_sources(value, target=None, *args, **kwargs):
        if isinstance(target, NamedSharding) and len(target.device_set) == 4:
            assert isinstance(value, np.ndarray), "Explicit distributed placement must use host data"
            if value.shape == (2,) and value.dtype == np.uint32:
                staged_rngs.append(value.copy())
        return original_put(value, target, *args, **kwargs)

    monkeypatch.setattr(distributed.jax, "device_put", no_peer_sources)
    before = loop.run_training(staged, stop_after_step=1, audit_checkpoint=True)
    after = loop.run_training(staged, audit_checkpoint=True)
    paused, resumed = before["checkpoint_audit"], after["checkpoint_audit"]
    assert paused["final_state_sha256"] == paused["checkpoint_state_sha256"] == resumed["start_state_sha256"]
    assert paused["next_update"] == resumed["first_update"]
    assert paused["first_update"] == full["checkpoint_audit"]["first_update"]
    assert resumed["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    assert after["validation_nll"] == full["validation_nll"]
    assert after["data_start_sequence"] == config.data.batch_size
    # Both update RNGs and held-out RNGs pass through explicit host placement.
    assert len(staged_rngs) >= 6


def test_runtime_provenance_records_execution_changes_separately(tmp_path, monkeypatch):
    import json
    config = tiny_config(tmp_path)
    config.train.workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, "0")
    loop._write_status(config, status="running")
    monkeypatch.setenv(distributed.HOST_STAGING_ENV, "1")
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    loop._write_status(config, status="running")
    loop._write_status(config, status="complete")
    record = json.loads((config.train.workdir / "metadata.json").read_text())
    assert len(record["runtime_history"]) == 2
    assert record["runtime_history"][0]["runtime"]["state_transfer"] == "direct"
    assert record["execution_runtime"]["state_transfer"] == "host_staged_verified"
    assert record["execution_runtime"]["batch_transfer"] == "host_to_shard"
    assert record["execution_runtime"]["environment"]["NCCL_P2P_DISABLE"] == "1"
    assert not hasattr(config.train, "host_staging")
