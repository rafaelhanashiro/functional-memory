"""Pretrained weights, Adam history, data time, and exact continuation resume."""

import copy
from dataclasses import replace
import hashlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from scaffolds.records import _tree_descriptor
from scaffolds.transformer.configs import ContinuationConfig, save_config, validate_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training import tracking
from scaffolds.transformer.training.checkpointing import (
    make_checkpoint_manager, restore_latest_checkpoint, state_sha256,
)
from scaffolds.transformer.training.continuation import initialize_continuation_state
from scaffolds.transformer.training.optim import split_continuation_parameters
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter


@pytest.fixture
def pretrained(tmp_path):
    parent = tiny_config(tmp_path / "parent")
    parent.model = replace(parent.model, attention_degree=2, attention_normalization=False,
                           attention_backend="blocked", attention_output_scaling="head_rms_norm")
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


def child_config(pretrained, path, *, learned, query=False, zero_amplitude=True):
    parent, _, continuation, _ = pretrained
    config = copy.deepcopy(parent)
    config.continuation = continuation
    config.optim.total_steps = 6
    config.train.workdir = path
    if learned:
        config.model = replace(config.model, attention_coefficients="learned",
                               attention_coefficient_sharing="per_head", attention_coefficient_scale="factorial")
    if query:
        config.model = replace(config.model, attention_coefficient_conditioning="query",
                               attention_coefficient_activation="tanh",
                               attention_coefficient_amplitude_scale="factorial",
                               attention_coefficient_amplitude_initial=0.0 if zero_amplitude else 1.0,
                               attention_coefficient_projection_initial_std=(
                                   1 / math.sqrt(config.model.hidden_size) if zero_amplitude else 0.0))
    return config


def assert_equal_tree(first, second):
    assert jax.tree.structure(first) == jax.tree.structure(second)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("query", [False, True])
def test_preserved_learned_parent_keeps_coefficients_moments_and_exact_resume(pretrained, tmp_path, query):
    parent, _, _, batch = pretrained
    parent = copy.deepcopy(parent)
    parent.model = replace(parent.model, attention_coefficients="learned",
                           attention_coefficient_sharing="per_head", attention_coefficient_scale="factorial")
    if query:
        parent.model = replace(parent.model, attention_coefficient_conditioning="query",
                               attention_coefficient_activation="tanh",
                               attention_coefficient_amplitude_scale="factorial",
                               attention_coefficient_amplitude_initial=0.0,
                               attention_coefficient_projection_initial_std=0.25,
                               attention_coefficient_kernel_weight_decay=True)
    # Train the coefficient arrays so restoring initialization instead would fail.
    state = create_state(parent, jax.random.PRNGKey(parent.train.seed))
    for step in range(1, 4):
        state, *_ = loop.train_step(state, batch, jax.random.fold_in(jax.random.PRNGKey(0), step))
    config_path = parent.train.workdir / "learned-config.yaml"
    save_config(parent, config_path)
    checkpoint = tmp_path / "learned-checkpoint" / "3"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(checkpoint / "default", {"state": state})
    child = copy.deepcopy(parent)
    child.train.workdir = tmp_path / "preserved"
    child.optim.learning_rate = child.optim.min_learning_rate = 1e-4
    child.optim.total_steps = 6
    child.continuation = ContinuationConfig(
        parent_config=str(config_path), parent_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=_tree_descriptor(checkpoint)["sha256"],
        parent_data_identity_sha256=hashlib.sha256((parent.train.workdir / "data_identity.json").read_bytes()).hexdigest(),
        parent_step=3, preserve_all_parameters=True)
    restored, audit = initialize_continuation_state(child)
    assert_equal_tree(restored.params, state.params)
    assert_equal_tree(restored.opt_state, state.opt_state)
    assert audit["new_coefficient_optimizer_count"] is None
    assert audit["backbone_optimizer_count"] == 3
    assert audit["data_start_sequence"] == 12
    # The next update must match ordinary training with the new constant LR and
    # the parent's complete state, including learned-coefficient Adam moments.
    ordinary = copy.deepcopy(child)
    ordinary.continuation = ContinuationConfig()
    expected = create_state(ordinary, jax.random.PRNGKey(0)).replace(
        params=state.params, opt_state=state.opt_state, step=state.step)
    rng = jax.random.fold_in(jax.random.PRNGKey(0), 4)
    actual, *_ = loop.train_step(restored, batch, rng)
    expected, *_ = loop.train_step(expected, batch, rng)
    assert_equal_tree(actual.params, expected.params)
    assert_equal_tree(actual.opt_state, expected.opt_state)
    coefficient_name = "query_polynomial_amplitude" if query else "polynomial_coefficients"
    assert np.any(np.asarray(actual.params[coefficient_name][...]) != np.asarray(state.params[coefficient_name][...]))
    interrupted = copy.deepcopy(child)
    interrupted.train.workdir = tmp_path / "preserved-resume"
    complete = loop.run_training(child, audit_checkpoint=True)
    loop.run_training(interrupted, stop_after_step=4, audit_checkpoint=True)
    resumed = loop.run_training(interrupted, audit_checkpoint=True)
    assert complete["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["final_state_sha256"]
    changed = copy.deepcopy(child)
    if query:
        changed.model.attention_coefficient_kernel_weight_decay = False
        changed_field = "attention_coefficient_kernel_weight_decay"
    else:
        changed.model.attention_coefficient_sharing = "global"
        changed_field = "attention_coefficient_sharing"
    with pytest.raises(ValueError, match=f"parent model.{changed_field}"):
        initialize_continuation_state(changed)
    changed = copy.deepcopy(child)
    changed.continuation.new_parameter_learning_rate = 1e-3
    with pytest.raises(ValueError, match="no new parameter learning rate"):
        validate_config(changed)


@pytest.mark.parametrize("query,zero_amplitude", [(False, True), (True, True), (True, False)])
def test_matched_initial_outputs_and_exact_backbone_history(pretrained, tmp_path, query, zero_amplitude):
    _, parent, _, batch = pretrained
    fixed, fixed_audit = initialize_continuation_state(child_config(pretrained, tmp_path / "fixed", learned=False))
    learned, learned_audit = initialize_continuation_state(child_config(
        pretrained, tmp_path / "learned", learned=True, query=query, zero_amplitude=zero_amplitude))
    assert_equal_tree(fixed.params, parent.params)
    assert_equal_tree(fixed.opt_state, parent.opt_state)
    coefficients, backbone = split_continuation_parameters(learned.params)
    assert_equal_tree(backbone, parent.params)
    assert_equal_tree(learned.opt_state[1].backbone, parent.opt_state[1])
    assert int(learned.opt_state[1].coefficients[0].count) == 0
    assert int(learned.opt_state[1].backbone[0].count) == 3
    bias = coefficients["query_polynomial_bias" if query else "polynomial_coefficients"][...]
    np.testing.assert_array_equal(bias, np.ones(bias.shape))
    assert float(loop.eval_step(fixed, batch, jax.random.PRNGKey(0))) == pytest.approx(
        float(loop.eval_step(learned, batch, jax.random.PRNGKey(0))), abs=1e-6)
    for audit in (fixed_audit, learned_audit):
        assert audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]
        assert audit["parent_optimizer_sha256"] == audit["backbone_optimizer_sha256"]
        assert audit["data_start_sequence"] == 12
    # The identical attention function also gives the pretrained backbone the same
    # initial gradient, even when the new generator has a random projection.
    inputs = batch[:, :-1]
    fixed_logits = nnx.merge(fixed.graphdef, fixed.params)(inputs, deterministic=True)
    learned_logits = nnx.merge(learned.graphdef, learned.params)(inputs, deterministic=True)
    np.testing.assert_allclose(learned_logits, fixed_logits, rtol=1e-6, atol=1e-6)
    _, fixed_grads = loop.loss_and_grads(fixed, batch, jax.random.PRNGKey(0))
    _, learned_grads = loop.loss_and_grads(learned, batch, jax.random.PRNGKey(0))
    coefficient_grads, backbone_grads = split_continuation_parameters(learned_grads)
    for actual, expected in zip(jax.tree.leaves(backbone_grads), jax.tree.leaves(fixed_grads), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if query:
        # A=0 gives amplitude gradients first; after that update, W learns too.
        first_name = "query_polynomial_amplitude" if zero_amplitude else "query_polynomial_kernel"
        assert np.any(np.asarray(coefficient_grads[first_name][...]) != 0)
        after, *_ = loop.train_step(learned, batch, jax.random.PRNGKey(0))
        _, second_grads = loop.loss_and_grads(after, batch, jax.random.PRNGKey(0))
        assert np.any(np.asarray(second_grads["query_polynomial_kernel"][...]) != 0)
        twice, *_ = loop.train_step(after, batch, jax.random.PRNGKey(0))
        assert np.any(np.asarray(twice.params["query_polynomial_kernel"][...]) !=
                      np.asarray(learned.params["query_polynomial_kernel"][...]))


@pytest.mark.parametrize("query", [False, True])
def test_first_coefficient_update_uses_fresh_adam_bias_correction(pretrained, tmp_path, query):
    cfg = child_config(pretrained, tmp_path / "learned", learned=True, query=query)
    state, _ = initialize_continuation_state(cfg)
    grads = jax.tree.map(lambda value: jnp.full_like(value, 0.001), state.params)
    updates, after = state.tx.update(grads, state.opt_state, state.params)
    coefficient_grads, backbone_grads = split_continuation_parameters(grads)
    coefficient_params, _ = split_continuation_parameters(state.params)
    coefficient_updates, backbone_updates = split_continuation_parameters(updates)
    reference = optax.adamw(1e-3, b1=cfg.optim.beta1, b2=cfg.optim.beta2, weight_decay=0)
    expected, _ = reference.update(coefficient_grads, reference.init(coefficient_params), coefficient_params)
    assert_equal_tree(coefficient_updates, expected)
    assert int(after[1].coefficients[0].count) == 1
    assert int(after[1].backbone[0].count) == 4
    fixed, _ = initialize_continuation_state(child_config(pretrained, tmp_path / "fixed", learned=False))
    fixed_updates, _ = fixed.tx.update(backbone_grads, fixed.opt_state, fixed.params)
    assert_equal_tree(backbone_updates, fixed_updates)
    # The actual coefficients move; this does not accidentally freeze their branch.
    assert all(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(coefficient_updates))


@pytest.mark.parametrize("learned,query", [(False, False), (True, False), (True, True)])
def test_interrupted_continuation_preserves_cursor_state_and_baseline(pretrained, tmp_path, learned, query):
    full = child_config(pretrained, tmp_path / "full", learned=learned, query=query)
    resumed = child_config(pretrained, tmp_path / "resumed", learned=learned, query=query)
    full_result = loop.run_training(full, audit_checkpoint=True)
    paused = loop.run_training(resumed, stop_after_step=4, audit_checkpoint=True)
    resumed_result = loop.run_training(resumed, audit_checkpoint=True)
    assert full_result["start_step"] == paused["start_step"] == 3
    assert full_result["data_start_sequence"] == paused["data_start_sequence"] == 12
    assert resumed_result["start_step"] == 4 and resumed_result["data_start_sequence"] == 16
    assert full_result["checkpoint_audit"]["final_state_sha256"] == resumed_result["checkpoint_audit"]["final_state_sha256"]
    assert full_result["checkpoint_audit"]["first_update"] == paused["checkpoint_audit"]["first_update"]
    for cfg in (full, resumed):
        metrics = [json.loads(line) for line in (cfg.train.workdir / "metrics.jsonl").read_text().splitlines()]
        assert [row["step"] for row in metrics] == [3, 4, 6]
        assert metrics[0]["continuation_updates"] == 0
        assert metrics[0]["train_nll_sample_count"] == 0
        assert (cfg.train.workdir / "continuation.json").exists()


def test_changed_parent_inputs_are_rejected(pretrained, tmp_path):
    config = child_config(pretrained, tmp_path / "child", learned=True)
    changed = copy.deepcopy(config)
    changed.train.seed += 1
    with pytest.raises(ValueError, match="train.seed"):
        initialize_continuation_state(changed)
    changed = copy.deepcopy(config)
    changed.data.shuffle_seed += 1
    with pytest.raises(ValueError, match="data recipe"):
        initialize_continuation_state(changed)
    changed = copy.deepcopy(config)
    changed.continuation.parent_checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="checkpoint hash"):
        initialize_continuation_state(changed)


def test_continuations_may_train_on_the_blocked_strips_of_a_scan_parent(pretrained, tmp_path):
    """The blocked implementation is execution, like the backend: a strips child restores a parent
    that trained on the scan, with new coefficients or preserving the parent's whole model."""
    from scaffolds.transformer.training.continuation import _parent_config

    parent = pretrained[0]
    assert parent.model.attention_blocked_implementation == "scan"
    children = {}
    for implementation in ("scan", "strips"):
        child = child_config(pretrained, tmp_path / implementation, learned=True)
        child.model = replace(child.model, attention_blocked_implementation=implementation)
        children[implementation] = initialize_continuation_state(child)
    assert_equal_tree(children["strips"][0].params, children["scan"][0].params)
    assert_equal_tree(children["strips"][0].opt_state, children["scan"][0].opt_state)
    assert children["strips"][1] == children["scan"][1]
    learned = copy.deepcopy(parent)
    learned.model = replace(learned.model, attention_coefficients="learned",
                            attention_coefficient_sharing="per_head", attention_coefficient_scale="factorial")
    path = tmp_path / "learned.yaml"
    save_config(learned, path)
    preserved = copy.deepcopy(learned)
    preserved.continuation = replace(pretrained[2], parent_config=str(path), preserve_all_parameters=True,
                                     parent_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    preserved.model = replace(preserved.model, attention_blocked_implementation="strips")
    _parent_config(preserved)
    preserved.model = replace(preserved.model, attention_coefficient_sharing="global")
    with pytest.raises(ValueError, match="parent model.attention_coefficient_sharing"):
        _parent_config(preserved)


@pytest.mark.parametrize("learned,query", [(False, False), (True, False), (True, True)])
def test_ungated_continuation_cannot_silently_remove_rope(pretrained, tmp_path, learned, query):
    config = child_config(pretrained, tmp_path / "child", learned=learned, query=query)
    config.model = replace(config.model, positional_encoding="none")
    with pytest.raises(ValueError, match="parent model.positional_encoding"):
        initialize_continuation_state(config)


def test_query_continuation_rejects_non_taylor_initialization(pretrained, tmp_path):
    config = child_config(pretrained, tmp_path / "query", learned=True, query=True)
    changed = copy.deepcopy(config)
    changed.model.attention_coefficient_amplitude_initial = 1.0
    with pytest.raises(ValueError, match="exact Taylor initialization"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed.model.attention_coefficient_amplitude_scale = "none"
    with pytest.raises(ValueError, match="exact Taylor initialization"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed.model.attention_coefficient_projection_initial_std = 0.0
    with pytest.raises(ValueError, match="zero gradients and never learn"):
        validate_config(changed)
    state = create_state(config, jax.random.PRNGKey(config.train.seed))
    state.params["query_polynomial_amplitude"][...] = jnp.ones_like(
        state.params["query_polynomial_amplitude"][...])
    with pytest.raises(ValueError, match="query-dependent term"):
        initialize_continuation_state(config, state)


def test_baseline_mirror_has_no_invented_training_measurements(monkeypatch):
    payloads = []
    monkeypatch.setattr(tracking, "_log", lambda run, payload, step: payloads.append(payload))
    tracking.log_validation(None, {"step": 3, "training_tokens": 48, "validation_nll": 2.,
                                   "validation_perplexity": 7.389, "is_best": True,
                                   "parameter_count": 774})
    assert payloads[0]["val/nll"] == 2.
    assert "train/nll_window_mean" not in payloads[0]
    assert "train/tokens_per_second" not in payloads[0]


def test_packed_continuation_rejects_unpinned_parent_identity(pretrained, tmp_path):
    cfg = child_config(pretrained, tmp_path / "child", learned=False)
    cfg.train.workdir.mkdir()
    cfg.continuation = replace(cfg.continuation, parent_data_identity_sha256="0" * 64)
    with pytest.raises(ValueError, match="parent data identity hash"):
        loop.resolve_packed_data_cursor(cfg, 3)
    assert not (cfg.train.workdir / "data_identity.json").exists()


@pytest.mark.parametrize("features,zero_amplitude", [(3, False), (3, True), (5, True)])
def test_dense_continuation_preserves_backbone_and_starts_new_mixing_clock(
        pretrained, tmp_path, features, zero_amplitude):
    _, parent, _, batch = pretrained
    config = child_config(pretrained, tmp_path / "dense", learned=True, query=True,
                          zero_amplitude=zero_amplitude)
    config.model = replace(config.model, attention_coefficient_readout="dense",
                           attention_coefficient_feature_count=features,
                           attention_coefficient_readout_scale=1 / 3)
    child, audit = initialize_continuation_state(config)
    coefficients, backbone = split_continuation_parameters(child.params)
    assert "query_polynomial_mixing" in coefficients
    assert "query_polynomial_mixing" not in backbone
    assert_equal_tree(backbone, parent.params)
    assert_equal_tree(child.opt_state[1].backbone, parent.opt_state[1])
    assert int(child.opt_state[1].coefficients[0].count) == 0
    assert audit["parent_optimizer_sha256"] == audit["backbone_optimizer_sha256"]
    expected = nnx.merge(parent.graphdef, parent.params)(batch[:, :-1], deterministic=True)
    actual = nnx.merge(child.graphdef, child.params)(batch[:, :-1], deterministic=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    updated, *_ = loop.train_step(child, batch, jax.random.PRNGKey(17))
    assert int(updated.opt_state[1].coefficients[0].count) == 1
    assert int(updated.opt_state[1].backbone[0].count) == 4
