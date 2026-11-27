"""Initialize a fresh continuation from a pinned parent checkpoint.

A continuation on its parent's optimizer retains the pretrained Adam history and data time.
A fork (``continuation.optimizer: fresh``) takes the parent's weights only: its AdamW, update
clock and schedule start at zero, and it reads the parent's packed stream from the sequence
after the parent's last update.
"""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from scaffolds.records import _tree_descriptor
from scaffolds.transformer.configs import load_config, validate_config
from scaffolds.transformer.configs.base import continuation_is_fork, packed_data_offset
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.optim import (
    ContinuationAdamState, continuation_introduces_parameters, split_continuation_parameters,
)
from scaffolds.transformer.training.state import create_state


def _parent_config(config):
    continuation = config.continuation
    path = Path(continuation.parent_config)
    if hashlib.sha256(path.read_bytes()).hexdigest() != continuation.parent_config_sha256:
        raise ValueError("Continuation parent config hash changed")
    parent = load_config(path)
    if parent.continuation.parent_checkpoint:
        raise ValueError("A continuation parent must use the original shared optimizer")
    preserve = continuation.preserve_all_parameters
    if not preserve and (parent.model.attention_coefficients != "fixed"
                         or parent.model.attention_fixed_coefficients is not None):
        raise ValueError("Continuation requires a fixed Taylor or exp parent")
    if parent.model.attention_mechanism != "dot_product" or parent.model.quadratic_mode != "fixed":
        raise ValueError("Continuation requires dot-product attention")
    exp_parent = parent.model.attention_degree is None
    if not preserve and exp_parent and (parent.model.attention_normalization or config.model.attention_degree is None):
        # An unnormalized exp parent continues with a polynomial kernel of an integer degree:
        # fixed Taylor as the control, or Taylor-initialized learned coefficients.
        raise ValueError("Continuing an exp parent requires an unnormalized exp parent and a polynomial degree")
    # The backend, the blocked backend's implementation (scan or causal strips), the exponential's row-maximum
    # shift and head RMSNorm's max-abs prescale are execution settings: none changes a parameter or optimizer
    # shape (the shift and the prescale keep the function).
    execution = {"attention_backend", "attention_blocked_implementation", "attention_exp_shift",
                 "attention_head_rms_prescale"}
    allowed_model_changes = execution | {
        "attention_coefficients", "attention_coefficient_sharing", "attention_coefficient_scale",
        "attention_coefficient_conditioning", "attention_coefficient_activation",
        "attention_coefficient_amplitude_initial", "attention_coefficient_projection_initial_std",
        "attention_coefficient_amplitude_scale", "attention_coefficient_readout",
        "attention_coefficient_feature_count", "attention_coefficient_readout_scale",
        "qk_norm_token_scale", "qk_norm_token_scale_kernel_weight_decay",
        "attention_forget_gate", "attention_forget_gate_form", "attention_coefficient_kernel_weight_decay"}
    if exp_parent:
        allowed_model_changes |= {"attention_degree"}
    if (parent.model.positional_encoding == "rope" and config.model.positional_encoding == "none"
            and config.model.attention_forget_gate):
        # An explicitly requested forget-gate continuation may replace RoPE with
        # its learned distance decay. This changes the function, not the backbone
        # parameter tree: every pretrained array and its Adam history still restore.
        allowed_model_changes.add("positional_encoding")
    if preserve:
        allowed_model_changes = execution
    for key, value in asdict(parent.model).items():
        if key not in allowed_model_changes and getattr(config.model, key) != value:
            raise ValueError(f"Continuation changes parent model.{key}")
    if asdict(parent.data) != asdict(config.data):
        raise ValueError("Continuation must preserve the parent data recipe and batch size")
    for key in ("beta1", "beta2", "weight_decay", "weight_decay_policy", "grad_clip_norm",
                "coefficient_penalty", "coefficient_penalty_denominator",
                "query_coefficient_learning_rate_scale"):
        if getattr(parent.optim, key) != getattr(config.optim, key):
            raise ValueError(f"Continuation changes parent optim.{key}")
    for key in ("seed", "dtype"):
        if getattr(parent.train, key) != getattr(config.train, key):
            raise ValueError(f"Continuation changes parent train.{key}")
    if continuation.parent_step > parent.optim.total_steps:
        raise ValueError("Continuation parent step exceeds its training budget")
    if continuation.optimizer == "fresh" and continuation.data_start_sequence != (
            packed_data_offset(parent.continuation) + continuation.parent_step * config.data.batch_size):
        # The sequence after the parent's last update, wherever the parent's own stream started.
        raise ValueError("A fork's data_start_sequence must follow its parent's last update")
    # Acceptance restores on CPU with the blocked reference; backend is execution,
    # and does not change any checkpoint parameter or optimizer shape. An exp parent
    # keeps a backend that evaluates exp, which the child's polynomial kernel may not.
    parent.model.attention_backend = "blocked" if exp_parent else config.model.attention_backend
    return parent


def _check_taylor_initialization(config, coefficients):
    """Check the realized arrays, including identically zero query-dependent and token-scale terms."""
    model = config.model
    token_scale = model.qk_norm_token_scale != "none"
    token_names = {"qk_norm_token_scale_kernel", "qk_norm_token_scale_amplitude"} if token_scale else set()
    if model.attention_forget_gate:
        # The gate starts open with Gated DeltaNet's random initialization, not at the identity.
        from scaffolds.transformer.attention.forget_gate import parameter_names

        gate_names = parameter_names(getattr(model, "attention_forget_gate_form", "input"),
                                     getattr(model, "attention_forget_gate_conditioning", "query"),
                                     getattr(model, "attention_forget_gate_constant_rate", "learned"))
        token_names |= set(gate_names)
        if not all(np.all(np.isfinite(np.asarray(coefficients[name][...]))) for name in gate_names
                   if name in coefficients):
            raise ValueError("Continuation forget gate parameters are not finite")
    if model.attention_coefficients != "learned":
        if set(coefficients) != token_names:
            raise ValueError("Continuation parameter tree differs from its recipe")
        if token_scale:
            _check_token_scale_initialization(coefficients)
        return
    expected = np.asarray([1.0 if model.attention_coefficient_scale == "factorial"
                           else 1.0 / math.factorial(order)
                           for order in range(model.attention_degree + 1)], np.float32)
    query = model.attention_coefficient_conditioning == "query"
    dense = model.attention_coefficient_readout == "dense"
    readout_name = "query_polynomial_mixing" if dense else "query_polynomial_amplitude"
    names = ({"query_polynomial_bias", readout_name, "query_polynomial_kernel"}
             if query else {"polynomial_coefficients"})
    if set(coefficients) != names | token_names:
        raise ValueError("Continuation coefficient parameter tree differs from its recipe")
    if token_scale:
        _check_token_scale_initialization(coefficients)
    bias = np.asarray(coefficients["query_polynomial_bias" if query else "polynomial_coefficients"][...])
    if not np.array_equal(bias, np.broadcast_to(expected, bias.shape)):
        raise ValueError("Continuation coefficients are not exactly Taylor initialized")
    if query:
        readout = np.asarray(coefficients[readout_name][...])
        kernel = np.asarray(coefficients["query_polynomial_kernel"][...])
        if (not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(readout))
                or not (np.all(readout == 0) or np.all(kernel == 0))):
            raise ValueError("Continuation query-dependent term is not identically zero")
        expected_readout = model.attention_coefficient_amplitude_initial
        if dense:
            expected_readout *= np.eye(model.attention_degree + 1, kernel.shape[-1], dtype=np.float32)
        if not np.all(readout == expected_readout):
            raise ValueError("Continuation query amplitude or mixing differs from its initialization recipe")


def _check_token_scale_initialization(parameters):
    """The gate starts closed: zero amplitudes and a finite, nonzero direction per layer/head."""
    amplitude = np.asarray(parameters["qk_norm_token_scale_amplitude"][...])
    kernel = np.asarray(parameters["qk_norm_token_scale_kernel"][...])
    if np.any(amplitude != 0) or not np.all(np.isfinite(kernel)) or np.any(np.all(kernel == 0, axis=-1)):
        raise ValueError("Continuation token scale is not identically zero with a finite nonzero direction")


def initialize_continuation_state(config, state=None):
    """Return the child state and verified audit.

    On the parent's optimizer, existing backbone moments and their bias-correction count are
    preserved exactly and new coefficient Adam starts at zero; the child's global step is the
    parent's step, so batch order and RNG continue at the next pretrained update. A fork keeps
    the fresh optimizer state and the zero step of ``state`` (a new state of the child's recipe),
    replacing only the parameters. Checkpoint resumptions use the usual trainer restore.
    """
    validate_config(config)
    continuation = config.continuation
    if not continuation.parent_checkpoint:
        raise ValueError("No continuation parent is configured")
    fork = continuation_is_fork(continuation)
    parent_config = _parent_config(config)
    checkpoint = Path(continuation.parent_checkpoint).resolve()
    if _tree_descriptor(checkpoint)["sha256"] != continuation.parent_checkpoint_sha256:
        raise ValueError("Continuation parent checkpoint hash changed")
    rng = jax.random.PRNGKey(config.train.seed)
    parent_template = create_state(parent_config, rng)
    directory = checkpoint / "default" if (checkpoint / "default").is_dir() else checkpoint
    with ocp.StandardCheckpointer() as checkpointer:
        parent = checkpointer.restore(directory, target={"state": parent_template})["state"]
    if int(parent.step) != continuation.parent_step:
        raise ValueError("Continuation checkpoint step differs from the pinned parent step")
    if state is None:
        state = create_state(config, rng)
    new_coefficient_count = None
    introduces = continuation_introduces_parameters(config)
    if introduces:
        coefficients, backbone = split_continuation_parameters(state.params)
        if not jax.tree.leaves(coefficients):
            raise ValueError("Continuation did not introduce its new parameter arrays")
        if jax.tree.structure(backbone) != jax.tree.structure(parent.params):
            raise ValueError("Continuation backbone parameter tree differs from parent")
        _check_taylor_initialization(config, coefficients)
        params = nnx.State.merge(parent.params, coefficients)
    else:
        if jax.tree.structure(state.params) != jax.tree.structure(parent.params):
            raise ValueError("Continuation parameter tree differs from parent")
        params = parent.params
    if fork:
        # The child's own new state supplies the optimizer: AdamW initialized at count zero, with
        # zero moments, for exactly these parameters.
        if int(state.step) != 0:
            raise ValueError("A fork starts from a new state at update zero")
        opt_state = state.opt_state
        step = state.step
    elif introduces:
        new_adam = state.opt_state[1].coefficients
        new_coefficient_count = int(new_adam[0].count)
        if new_coefficient_count != 0:
            raise ValueError("New coefficient optimizer must have zero history")
        opt_state = (parent.opt_state[0], ContinuationAdamState(parent.opt_state[1], new_adam))
        step = parent.step
    else:
        opt_state, step = parent.opt_state, parent.step
    state = state.replace(params=params, opt_state=opt_state, step=step)
    backbone = split_continuation_parameters(state.params)[1] if introduces else state.params
    audit = {
        "parent_step": int(parent.step),
        "parent_config_sha256": continuation.parent_config_sha256,
        "parent_checkpoint_sha256": continuation.parent_checkpoint_sha256,
        "parent_parameters_sha256": state_sha256(parent.params),
        "backbone_parameters_sha256": state_sha256(backbone),
    }
    if fork:
        counts = [int(leaf) for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
                  if getattr(path[-1], "name", None) == "count"]
        audit.update({
            "optimizer": "fresh",
            "start_step": int(state.step),
            "optimizer_sha256": state_sha256(state.opt_state),
            "fresh_optimizer_sha256": state_sha256(state.tx.init(state.params)),
            "optimizer_counts": counts,
            "data_start_sequence": packed_data_offset(continuation) + int(state.step) * config.data.batch_size,
            "seed": config.train.seed,
        })
        if audit["optimizer_sha256"] != audit["fresh_optimizer_sha256"] or not counts or any(counts):
            raise AssertionError("A fork's optimizer is not a fresh AdamW at update zero")
        if audit["data_start_sequence"] != continuation.data_start_sequence:
            raise AssertionError("A fork's data cursor differs from its pinned start sequence")
    else:
        backbone_adam = state.opt_state[1].backbone if introduces else state.opt_state[1]
        audit.update({
            "parent_optimizer_sha256": state_sha256(parent.opt_state[1]),
            "backbone_optimizer_sha256": state_sha256(backbone_adam),
            "backbone_optimizer_count": int(backbone_adam[0].count),
            "new_coefficient_optimizer_count": new_coefficient_count,
            "data_start_sequence": int(parent.step) * config.data.batch_size,
            "seed": config.train.seed,
        })
        if audit["parent_optimizer_sha256"] != audit["backbone_optimizer_sha256"]:
            raise AssertionError("Continuation modified the pretrained optimizer history")
    if audit["parent_parameters_sha256"] != audit["backbone_parameters_sha256"]:
        raise AssertionError("Continuation modified the pretrained backbone")
    return state, audit


def packed_batch_sha256(batch) -> str:
    """One update's packed rows as int32 ``(batch_size, sequence_length)``, however micro-batched."""
    rows = np.asarray(batch, dtype=np.int32)
    return state_sha256(rows.reshape(-1, rows.shape[-1]))


def fork_batch_provenance(config, *, validated_manifest=None) -> dict:
    """The parent's last batch and the fork's first, read from the packed stream they share."""
    from scaffolds.transformer.data.packed import PackedBatchIterator

    if not continuation_is_fork(config.continuation):
        raise ValueError("Batch provenance describes a fork")
    batch_size, start = config.data.batch_size, config.continuation.data_start_sequence
    batches = {}
    for name, first in (("parent_last_batch", start - batch_size), ("first_batch", start)):
        rows = next(PackedBatchIterator(config.data.train_manifest, batch_size=batch_size,
                                        sequence_length=config.data.sequence_length, start_sequence=first,
                                        validated_manifest=validated_manifest))
        batches[name] = {"sequence_start": first, "sequence_stop": first + batch_size,
                         "sha256": packed_batch_sha256(rows)}
    return batches
