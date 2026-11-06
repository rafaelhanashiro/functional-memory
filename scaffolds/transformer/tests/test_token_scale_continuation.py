"""Continuing a head-scaled Taylor checkpoint with the per-token query scale gate."""

import copy
from dataclasses import replace
import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from scaffolds.records import _tree_descriptor
from scaffolds.transformer.configs import ContinuationConfig, save_config, validate_config
from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.tests.test_continuation import assert_equal_tree
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.continuation import initialize_continuation_state
from scaffolds.transformer.training.optim import split_continuation_parameters
from scaffolds.transformer.training.state import create_state


@pytest.fixture
def pretrained(tmp_path):
    """A fixed-Taylor parent with QK RMSNorm and per-head scaling, trained three updates."""
    parent = tiny_config(tmp_path / "parent")
    parent.model = replace(parent.model, attention_degree=2, attention_normalization=False,
                           attention_backend="blocked", attention_output_scaling="head_rms_norm",
                           qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
                           qk_norm_head_scale=True)
    parent.optim.learning_rate = parent.optim.min_learning_rate = 1e-3
    parent.optim.grad_clip_norm = 1.0
    parent.data.source = "packed"
    for split, count in (("train", 32), ("eval", 4)):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5, num_sequences=count, shard_sequences=4)
        writer.write(np.arange(count * 5).reshape(count, 5) % 16)
        setattr(parent.data, f"{split}_manifest", str(writer.finish()))
    parent.train.workdir.mkdir()
    loop.resolve_packed_data_cursor(parent, 0)
    config_path = parent.train.workdir / "config.yaml"
    save_config(parent, config_path)
    state = create_state(parent, jax.random.PRNGKey(parent.train.seed))
    batch = jnp.asarray(np.arange(20).reshape(4, 5) % 16)
    for step in range(1, 4):
        state, *_ = loop.train_step(state, batch, jax.random.fold_in(jax.random.PRNGKey(0), step))
    checkpoint = tmp_path / "checkpoint" / "3"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(checkpoint / "default", {"state": state})
    continuation = ContinuationConfig(
        parent_config=str(config_path), parent_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=_tree_descriptor(checkpoint)["sha256"],
        parent_data_identity_sha256=hashlib.sha256((parent.train.workdir / "data_identity.json").read_bytes()).hexdigest(),
        parent_step=3)
    return parent, state, continuation, batch


def child_config(pretrained, path, *, gate, new_rate=None):
    parent, _, continuation, _ = pretrained
    config = copy.deepcopy(parent)
    config.continuation = replace(continuation, new_parameter_learning_rate=new_rate)
    config.optim.total_steps = 6
    config.train.workdir = path
    if gate:
        config.model = replace(config.model, qk_norm_token_scale="cosine")
    validate_config(config)
    return config


def test_gate_starts_closed_with_the_parent_backbone_and_a_fresh_clock(pretrained, tmp_path):
    _, parent, _, batch = pretrained
    fixed, _ = initialize_continuation_state(child_config(pretrained, tmp_path / "fixed", gate=False))
    gated, audit = initialize_continuation_state(
        child_config(pretrained, tmp_path / "gated", gate=True, new_rate=1e-2))
    gate, backbone = split_continuation_parameters(gated.params)
    assert set(gate) == {"qk_norm_token_scale_kernel", "qk_norm_token_scale_amplitude"}
    assert_equal_tree(backbone, parent.params)
    assert_equal_tree(gated.opt_state[1].backbone, parent.opt_state[1])
    assert int(gated.opt_state[1].coefficients[0].count) == 0
    assert int(gated.opt_state[1].backbone[0].count) == 3
    assert audit["new_coefficient_optimizer_count"] == 0
    assert audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]
    assert np.all(np.asarray(gate["qk_norm_token_scale_amplitude"][...]) == 0)
    inputs = batch[:, :-1]
    fixed_logits = nnx.merge(fixed.graphdef, fixed.params)(inputs, deterministic=True)
    gated_logits = nnx.merge(gated.graphdef, gated.params)(inputs, deterministic=True)
    np.testing.assert_array_equal(np.asarray(gated_logits), np.asarray(fixed_logits))
    _, fixed_grads = loop.loss_and_grads(fixed, batch, jax.random.PRNGKey(0))
    _, gated_grads = loop.loss_and_grads(gated, batch, jax.random.PRNGKey(0))
    gate_grads, backbone_grads = split_continuation_parameters(gated_grads)
    for actual, expected in zip(jax.tree.leaves(backbone_grads), jax.tree.leaves(fixed_grads), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert np.any(np.asarray(gate_grads["qk_norm_token_scale_amplitude"][...]) != 0)
    assert np.all(np.asarray(gate_grads["qk_norm_token_scale_kernel"][...]) == 0)


def test_new_parameters_use_their_own_learning_rate_then_open_the_direction(pretrained, tmp_path):
    _, _, _, batch = pretrained
    gated, _ = initialize_continuation_state(
        child_config(pretrained, tmp_path / "gated", gate=True, new_rate=1e-2))
    before = np.asarray(gated.params["qk_norm_token_scale_amplitude"][...])
    _, grads = loop.loss_and_grads(gated, batch, jax.random.PRNGKey(0))
    after, *_ = loop.train_step(gated, batch, jax.random.PRNGKey(0))
    step = np.abs(np.asarray(after.params["qk_norm_token_scale_amplitude"][...]) - before)
    gradient = np.abs(np.asarray(grads["qk_norm_token_scale_amplitude"][...]))
    # Adam's first update is the learning rate times the gradient sign: the gate's own 1e-2,
    # while the backbone keeps its 1e-3 and its clock.
    np.testing.assert_allclose(step[gradient > 1e-6], 1e-2, rtol=1e-2)
    assert np.all(step[gradient == 0] == 0)
    assert int(after.opt_state[1].coefficients[0].count) == 1
    assert int(after.opt_state[1].backbone[0].count) == 4
    np.testing.assert_array_equal(np.asarray(after.params["qk_norm_token_scale_kernel"][...]),
                                  np.asarray(gated.params["qk_norm_token_scale_kernel"][...]))
    twice, *_ = loop.train_step(after, batch, jax.random.PRNGKey(1))
    assert np.any(np.asarray(twice.params["qk_norm_token_scale_kernel"][...])
                  != np.asarray(after.params["qk_norm_token_scale_kernel"][...]))


def test_rejects_a_new_parameter_rate_without_new_parameters(pretrained, tmp_path):
    with pytest.raises(ValueError, match="introduced parameters"):
        child_config(pretrained, tmp_path / "fixed", gate=False, new_rate=1e-2)


def forget_gate_config(pretrained, path, *, new_rate=1e-2, form="input"):
    """Zero-amplitude tanh query coefficients with Gated DeltaNet's forget gate, on the blocked backend."""
    parent, _, continuation, _ = pretrained
    config = copy.deepcopy(parent)
    config.continuation = replace(continuation, new_parameter_learning_rate=new_rate)
    config.optim.total_steps = 6
    config.train.workdir = path
    config.model = replace(config.model, attention_coefficients="learned", attention_coefficient_sharing="per_head",
                           attention_coefficient_scale="factorial", attention_coefficient_conditioning="query",
                           attention_coefficient_activation="tanh", attention_coefficient_amplitude_scale="factorial",
                           attention_coefficient_amplitude_initial=0.0,
                           attention_coefficient_projection_initial_std=1 / math.sqrt(config.model.hidden_size),
                           attention_forget_gate=True, attention_forget_gate_form=form, attention_backend="blocked")
    validate_config(config)
    return config


def test_constant_decay_continuation_introduces_only_the_rate(pretrained, tmp_path):
    _, parent, _, batch = pretrained
    gated, audit = initialize_continuation_state(
        forget_gate_config(pretrained, tmp_path / "constant", form="constant"))
    introduced, backbone = split_continuation_parameters(gated.params)
    assert set(introduced) == {"query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel",
                               "forget_gate_log_rate"}
    assert_equal_tree(backbone, parent.params)
    assert audit["new_coefficient_optimizer_count"] == 0
    rate = np.exp(np.asarray(introduced["forget_gate_log_rate"][...])[..., 0])
    np.testing.assert_allclose(rate, np.broadcast_to([2 ** -4, 2 ** -8], rate.shape), rtol=1e-6)
    before = np.asarray(gated.params["forget_gate_log_rate"][...])
    _, grads = loop.loss_and_grads(gated, batch, jax.random.PRNGKey(0))
    after, *_ = loop.train_step(gated, batch, jax.random.PRNGKey(0))
    step = np.abs(np.asarray(after.params["forget_gate_log_rate"][...]) - before)
    gradient = np.abs(np.asarray(grads["forget_gate_log_rate"][...]))
    np.testing.assert_allclose(step[gradient > 1e-6], 1e-2, rtol=1e-2)
    assert int(after.opt_state[1].coefficients[0].count) == 1 and int(after.opt_state[1].backbone[0].count) == 4


def test_forget_gate_continuation_starts_open_on_the_parent_backbone(pretrained, tmp_path):
    _, parent, _, batch = pretrained
    fixed, _ = initialize_continuation_state(child_config(pretrained, tmp_path / "fixed", gate=False))
    gated, audit = initialize_continuation_state(forget_gate_config(pretrained, tmp_path / "forget"))
    introduced, backbone = split_continuation_parameters(gated.params)
    assert set(introduced) == {"query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel",
                               "forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"}
    assert_equal_tree(backbone, parent.params)
    assert_equal_tree(gated.opt_state[1].backbone, parent.opt_state[1])
    assert int(gated.opt_state[1].coefficients[0].count) == 0
    assert audit["new_coefficient_optimizer_count"] == 0
    for name in ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"):
        assert np.all(np.isfinite(np.asarray(introduced[name][...])))
    rate = np.exp(np.asarray(introduced["forget_gate_log_rate"][...]))
    assert np.all(rate > 0) and np.all(rate < 16)
    # Gated DeltaNet's initialization decays from the first step, so the function differs from the parent.
    inputs = batch[:, :-1]
    fixed_logits = np.asarray(nnx.merge(fixed.graphdef, fixed.params)(inputs, deterministic=True))
    gated_logits = np.asarray(nnx.merge(gated.graphdef, gated.params)(inputs, deterministic=True))
    assert not np.allclose(gated_logits, fixed_logits, atol=1e-5)
    _, grads = loop.loss_and_grads(gated, batch, jax.random.PRNGKey(0))
    before = {name: np.asarray(gated.params[name][...]) for name in ("forget_gate_log_rate", "forget_gate_dt_bias")}
    after, *_ = loop.train_step(gated, batch, jax.random.PRNGKey(0))
    for name, value in before.items():
        gradient = np.abs(np.asarray(grads[name][...]))
        step = np.abs(np.asarray(after.params[name][...]) - value)
        np.testing.assert_allclose(step[gradient > 1e-6], 1e-2, rtol=1e-2)
    assert int(after.opt_state[1].coefficients[0].count) == 1
    assert int(after.opt_state[1].backbone[0].count) == 4
