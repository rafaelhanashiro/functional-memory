"""Continuing an unnormalized exp parent with a polynomial kernel, and decaying introduced parameters."""

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


def _pretrained(tmp_path, *, degree):
    """A fixed-kernel parent (Taylor of ``degree``, or exp when None) with the canonical QK norm, trained three updates."""
    parent = tiny_config(tmp_path / "parent")
    parent.model = replace(parent.model, attention_degree=degree, attention_normalization=False,
                           attention_backend="blocked", attention_output_scaling="head_rms_norm",
                           qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
                           qk_norm_head_scale=True)
    parent.optim.learning_rate = parent.optim.min_learning_rate = 1e-3
    parent.optim.grad_clip_norm = 1.0
    parent.optim.weight_decay = 0.1
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


@pytest.fixture
def exp_parent(tmp_path):
    return _pretrained(tmp_path, degree=None)


@pytest.fixture
def taylor_parent(tmp_path):
    return _pretrained(tmp_path, degree=2)


def _child(pretrained, path, *, degree, mechanism, kernel_weight_decay=False, new_rate=1e-2):
    parent, _, continuation, _ = pretrained
    config = copy.deepcopy(parent)
    config.continuation = replace(continuation, new_parameter_learning_rate=new_rate if mechanism else None)
    config.optim.total_steps = 6
    config.train.workdir = path
    changes = {"attention_degree": degree}
    if mechanism:
        changes.update(attention_coefficients="learned", attention_coefficient_sharing="per_head",
                       attention_coefficient_scale="factorial")
    if mechanism == "query":
        changes.update(attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                       attention_coefficient_amplitude_scale="factorial", attention_coefficient_amplitude_initial=0.0,
                       attention_coefficient_projection_initial_std=1 / math.sqrt(config.model.hidden_size),
                       attention_coefficient_kernel_weight_decay=kernel_weight_decay)
    config.model = replace(config.model, **changes)
    validate_config(config)
    return config


@pytest.mark.parametrize("mechanism", [None, "static", "query"])
def test_exp_parent_continues_with_a_taylor_kernel(exp_parent, tmp_path, mechanism):
    parent_config, parent, _, batch = exp_parent
    child, audit = initialize_continuation_state(_child(exp_parent, tmp_path / "child", degree=2, mechanism=mechanism))
    introduced, backbone = split_continuation_parameters(child.params)
    assert_equal_tree(backbone, parent.params)
    assert int(child.step) == 3 and audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]
    if mechanism:
        assert jax.tree.leaves(introduced) and audit["new_coefficient_optimizer_count"] == 0
        name = "polynomial_coefficients" if mechanism == "static" else "query_polynomial_bias"
        np.testing.assert_array_equal(np.asarray(introduced[name][...]), 1.0)
    else:
        assert not jax.tree.leaves(introduced) and audit["new_coefficient_optimizer_count"] is None
    # The kernel changed from exp to the degree-two Taylor polynomial, so the function differs
    # from the parent even though every backbone weight is identical.
    inputs = batch[:, :-1]
    exp_logits = np.asarray(nnx.merge(parent.graphdef, parent.params)(inputs, deterministic=True))
    child_logits = np.asarray(nnx.merge(child.graphdef, child.params)(inputs, deterministic=True))
    assert not np.allclose(child_logits, exp_logits, atol=1e-4)
    after, *_ = loop.train_step(child, batch, jax.random.PRNGKey(0))
    assert int(after.step) == 4 and np.isfinite(float(loop.eval_step(after, batch, jax.random.PRNGKey(0))))


def test_exp_parent_rejects_an_exp_child(exp_parent, tmp_path):
    with pytest.raises(ValueError, match="polynomial degree"):
        initialize_continuation_state(_child(exp_parent, tmp_path / "child", degree=None, mechanism=None))


@pytest.mark.parametrize("kernel_weight_decay", [False, True])
def test_introduced_query_projection_decays_only_when_declared(taylor_parent, tmp_path, kernel_weight_decay):
    _, _, _, batch = taylor_parent
    child, _ = initialize_continuation_state(_child(
        taylor_parent, tmp_path / "child", degree=2, mechanism="query", kernel_weight_decay=kernel_weight_decay))
    # With every gradient zero, Adam moves nothing and only decoupled weight decay acts:
    # p <- p (1 - lr * weight_decay) for decayed arrays, unchanged otherwise.
    zero = jax.tree.map(jnp.zeros_like, child.params)
    updates, opt_state = child.tx.update(zero, child.opt_state, child.params)
    moved = {name: not np.allclose(np.asarray(updates[name][...]), 0)
             for name in ("query_polynomial_kernel", "query_polynomial_bias", "query_polynomial_amplitude")}
    assert moved["query_polynomial_kernel"] is kernel_weight_decay
    assert not moved["query_polynomial_bias"] and not moved["query_polynomial_amplitude"]
    if kernel_weight_decay:
        kernel = np.asarray(child.params["query_polynomial_kernel"][...])
        np.testing.assert_allclose(np.asarray(updates["query_polynomial_kernel"][...]), -1e-2 * 0.1 * kernel, rtol=1e-5)
