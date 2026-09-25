"""GLA uses the shared model, training, diagnostics, and checkpoint machinery."""

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.models.architectures.transformer import TransformerLM
from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import diagnostics, loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.state import create_state


def gla_config(path, **overrides):
    config = tiny_config(path)
    config.model = replace(
        config.model, attention_mechanism="gla", attention_backend="blocked", **overrides,
    )
    config.optim.learning_rate = 1e-4
    config.optim.min_learning_rate = 1e-5
    return config


@pytest.mark.parametrize("overrides", [
    {"attention_degree": 0},
    {"attention_degree": 2},
    {"attention_degree": 2, "quadratic_mode": "positive"},
    {"attention_degree": 2, "quadratic_mode": "unrestricted"},
    {"attention_backend": "manual"},
    {"attention_backend": "xla"},
    {"attention_backend": "cudnn"},
    {"attention_normalization": False},
])
def test_gla_rejects_incompatible_dot_product_controls(overrides):
    with pytest.raises(ValueError):
        ModelConfig(attention_mechanism="gla", **overrides)


def test_gla_fused_config_does_not_inherit_softmax_dtype_constraint(tmp_path):
    config = Config(model=ModelConfig(attention_mechanism="gla", attention_backend="fused"))
    assert config.train.dtype == "float32"
    validate_config(config)
    path = tmp_path / "gla.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.model == config.model
    assert restored.model.gla_gate_rank == 16
    assert restored.model.gla_gate_tau == 16.0
    assert restored.model.gla_chunk_size == 64
    assert restored.model.gla_inner_chunk_size == 16
    assert restored.model.gla_use_rope is True


@pytest.mark.parametrize("key_dim,tied", [(2, False), (8, False), (8, True)])
def test_gla_parameter_count_matches_shared_model_with_independent_key_width(tmp_path, key_dim, tied):
    config = gla_config(tmp_path, qk_head_dim=key_dim, tie_embeddings=tied)
    model = build_model(config.model, 0)
    assert isinstance(model, TransformerLM)
    params = nnx.state(model, nnx.Param)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    assert sum(leaf.size for leaf in jax.tree.leaves(params)) == model_parameter_count(config.model)
    attention = model.layers_0.attention
    assert attention.q_proj.kernel.shape == (config.model.hidden_size, config.model.num_heads * key_dim)
    assert attention.k_proj.kernel.shape == attention.q_proj.kernel.shape
    assert attention.v_proj.kernel.shape == (
        config.model.hidden_size, config.model.num_heads * config.model.value_head_dim,
    )
    # V and the recurrent read stay fixed while the key dimension is varied.
    assert attention.o_proj.kernel.shape == (
        config.model.num_heads * config.model.value_head_dim, config.model.hidden_size,
    )


@pytest.mark.parametrize("remat,dtype", [(False, "float32"), (True, "bfloat16")])
def test_gla_all_gate_and_norm_parameters_receive_task_gradients(tmp_path, remat, dtype):
    config = gla_config(tmp_path, remat=remat, qk_head_dim=4)
    config.train.dtype = dtype
    state = create_state(config, jax.random.PRNGKey(2))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(state.params))
    batch = jnp.arange(20, dtype=jnp.int32).reshape(4, 5) % 16
    loss, grads = loop.loss_and_grads(state, batch, jax.random.PRNGKey(3))
    assert np.isfinite(loss)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    flat = dict(nnx.to_flat_state(grads))
    for name in ("gate_low_rank", "gate_proj", "output_gate_proj", "head_norm"):
        leaves = [np.asarray(value[...]) for path, value in flat.items() if name in path]
        assert leaves, f"Missing GLA parameter group: {name}"
        assert all(np.any(value != 0) for value in leaves), f"No task gradient for {name}"


def test_gla_remat_preserves_logits_diagnostics_and_causality(tmp_path):
    config = gla_config(tmp_path, qk_head_dim=4)
    plain = build_model(config.model, 0)
    remat = build_model(replace(config.model, remat=True), 0)
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
    expected = plain(tokens, return_diagnostics=True)
    actual = remat(tokens, return_diagnostics=True)
    for observed, baseline in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(observed, baseline)
    changed = tokens.at[:, 2:].set(15)
    np.testing.assert_array_equal(plain(changed)[:, :2], expected[0][:, :2])
    graphdef, params = nnx.split(remat, nnx.Param)
    np.testing.assert_array_equal(nnx.merge(graphdef, params)(tokens), expected[0])


def test_gla_diagnostics_identify_recurrence_and_report_nonfinite_outputs(tmp_path):
    config = gla_config(tmp_path)
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = diagnostics.fixed_diagnostic_sample(config)
    report = diagnostics.attention_diagnostics(state, config, sample, provenance)
    assert report["valid"] and report["issues"] == []
    assert report["attention_mechanism"] == report["attention_kernel"] == "gla"
    assert report["attention_output_normalization"] == "per_head_rmsnorm_then_silu_output_gate"
    assert report["attention_score_scale"] == "1"
    assert report["kernel_values"] == "vector_gated_recurrence"
    assert report["gla_state_elements_per_layer"] == (
        config.model.num_heads * config.model.qk_head_dim * config.model.value_head_dim
    )
    assert len(report["layers"]) == config.model.num_layers
    for layer in report["layers"]:
        assert "normalizer_min" not in layer
        assert "kernel_min" not in layer
        assert all(value == 0 for key, value in layer.items() if key.startswith("invalid_"))
    projection = state.params["layers_0"]["attention"]["o_proj"]["kernel"]
    projection[...] = projection[...].at[0, 0].set(jnp.nan)
    broken = diagnostics.attention_diagnostics(state, config, sample, provenance)
    assert not broken["valid"]
    assert broken["layers"][0]["invalid_output_count"] > 0
    assert any("invalid_output_count" in issue for issue in broken["issues"])
    json.dumps(broken, allow_nan=False)


def test_gla_packed_training_and_fresh_resume_match_uninterrupted(tmp_path):
    config = gla_config(tmp_path / "full")
    config.data.source = "packed"
    for split, count in (("train", 16), ("eval", 4)):
        writer = PackedSequenceWriter(
            tmp_path / split, sequence_length=5, num_sequences=count, shard_sequences=3,
        )
        writer.write(np.arange(count * 5).reshape(count, 5) % 16)
        setattr(config.data, f"{split}_manifest", str(writer.finish()))
    full = loop.run_training(config, audit_checkpoint=True)
    interrupted_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    paused = loop.run_training(interrupted_config, stop_after_step=1, audit_checkpoint=True)
    # Reload the durable configuration and reconstruct the model/optimizer state.
    resumed_config = load_config(interrupted_config.train.workdir / "config.json")
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    before, after = paused["checkpoint_audit"], resumed["checkpoint_audit"]
    assert before["final_state_sha256"] == before["checkpoint_state_sha256"] == after["start_state_sha256"]
    assert before["next_update"] == after["first_update"]
    assert after["final_state_sha256"] == after["checkpoint_state_sha256"]
    assert after["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    assert resumed["restored_step"] == 1
    assert resumed["data_start_sequence"] == config.data.batch_size
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == config.optim.total_steps
            states.append(state)
        finally:
            manager.close()
    for first, second in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(first, second)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [2, 4]
    for row in records:
        assert row["attention_kernel"] == "gla"
        assert row["attention_diagnostics"]["valid"]
        assert row["attention_diagnostics"]["sample"]["eval_manifest_sha256"]
        assert row["validation_sequences"] == 4
        assert row["validation_tokens"] == 16
        assert np.isfinite(row["validation_nll"])
    json.dumps(records, allow_nan=False)


def test_run_config_without_gla_fields_resumes_unchanged_and_rejects_gla(tmp_path):
    config = tiny_config(tmp_path)
    assert config.model.attention_mechanism == "dot_product"
    loop.ensure_run_config(config, resume=True)
    path = tmp_path / "config.json"
    historical = json.loads(path.read_text())
    historical["model"] = {
        key: value for key, value in historical["model"].items()
        if key != "attention_mechanism" and not key.startswith("gla_")
    }
    path.write_text(json.dumps(historical))
    original = path.read_bytes()
    loop.ensure_run_config(config, resume=True)
    assert path.read_bytes() == original
    restored = load_config(path)
    assert restored.model.attention_mechanism == "dot_product"
    changed = replace(config, model=replace(config.model, attention_mechanism="gla"))
    with pytest.raises(ValueError, match="does not match"):
        loop.ensure_run_config(changed, resume=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize("platform,length,expected", [
    ("cpu", 64, "blocked"),
    ("gpu", 64, "fused"),
    ("gpu", 65, "blocked"),
])
def test_gla_auto_backend_selects_supported_recurrence(platform, length, expected):
    config = ModelConfig(attention_mechanism="gla", max_seq_len=length)
    assert resolve_attention_backend(config, platform=platform) == expected
    assert config.attention_backend == "auto"


@pytest.mark.parametrize("platform,length,error", [
    ("cpu", 64, "NVIDIA GPU"),
    ("gpu", 65, "divisible by 64"),
])
def test_gla_explicit_fused_backend_rejects_unsupported_execution(platform, length, error):
    config = ModelConfig(attention_mechanism="gla", attention_backend="fused", max_seq_len=length)
    with pytest.raises(ValueError, match=error):
        resolve_attention_backend(config, platform=platform)


def test_diagnostics_accept_replicated_parameters_with_multiple_visible_cpu_devices(tmp_path):
    # JAX device enumeration happens once per process, before test fixtures can
    # set XLA_FLAGS. Exercise the real runtime in a fresh two-device process.
    program = dedent("""\
        import json
        from dataclasses import replace
        from pathlib import Path
        import sys

        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        from scaffolds.transformer.tests.test_training import tiny_config
        from scaffolds.transformer.training import diagnostics
        from scaffolds.transformer.training.distributed import put_replicated
        from scaffolds.transformer.training.state import create_state

        assert jax.local_device_count() == 2
        mesh = Mesh(np.asarray(jax.local_devices()), ("data",))
        sharding = NamedSharding(mesh, P())
        checked = []
        for mechanism, degree in (("dot_product", None), ("dot_product", 3), ("gla", None)):
            config = tiny_config(Path(sys.argv[1]))
            config.model = replace(config.model, attention_mechanism=mechanism,
                                   attention_degree=degree, attention_backend="blocked")
            state = create_state(config, jax.random.PRNGKey(0))
            sample, provenance = diagnostics.fixed_diagnostic_sample(config)
            direct_layers = diagnostics._diagnostic_step(state.graphdef, state.params, sample[:, :-1])
            expected = diagnostics.attention_diagnostics(state, config, sample, provenance)
            replicated = put_replicated(state, sharding)
            assert len(jax.tree.leaves(replicated.params)[0].addressable_shards) == 2
            actual = diagnostics.attention_diagnostics(replicated, config, sample, provenance)
            assert expected["valid"] and actual["valid"]
            assert actual == expected
            for direct, mapped in zip(direct_layers, actual["layers"], strict=True):
                for name, value in direct.items():
                    np.testing.assert_array_equal(mapped[name], value)
            checked.append(actual["attention_kernel"])
        print(json.dumps(checked))
    """)
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2",
               OMP_NUM_THREADS="2", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3], env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == ["softmax", "taylor", "gla"]
