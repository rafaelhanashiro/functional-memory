"""The forks of r19dfa87cb7ed at its terminal update: pinned recipes, fork, first batch and GPUs.

Committed files only: the parent's records, the partner recipes and the adapter's pins. The fork's checkpoint and
the dataset are checked on the GPU allocation by the adapter's initialization and pilot checks.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import terminal_fork_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict

STUDY = Path(__file__).resolve().parents[1]
PARENT = STUDY / "runs" / adapter.PARENT


def test_the_pins_are_the_parents_committed_facts():
    assert hashlib.sha256((PARENT / "config.yaml").read_bytes()).hexdigest() == adapter.PARENT_CONFIG_SHA256
    assert (STUDY.parents[1] / adapter.PARENT_CONFIG).resolve() == (PARENT / "config.yaml").resolve()
    assert hashlib.sha256((PARENT / "data_identity.json").read_bytes()).hexdigest() == adapter.PARENT_DATA_IDENTITY_SHA256
    completion = json.loads((PARENT / "completion.json").read_text())
    metadata = json.loads((PARENT / "metadata.json").read_text())
    # The fork starts after the parent's final update, at the sequence its cursor stopped before.
    assert completion["final_step"] == adapter.PARENT_STEP and completion["data_start_sequence"] == 0
    assert metadata["next_train_sequence"] == adapter.DATA_START_SEQUENCE == adapter.PARENT_STEP * adapter.BATCH_SIZE
    assert json.loads((PARENT / "data_identity.json").read_text())["batch_size"] == adapter.BATCH_SIZE
    for name in adapter.PARTNERS:
        adapter.partner_config(name)  # Each partner's recipe still has its pinned bytes.
    assert len(adapter.PARENT_CHECKPOINT_SHA256) == 64


@pytest.mark.parametrize("name", tuple(adapter.PARTNERS))
def test_each_child_is_its_partners_recipe_with_the_pinned_fork(name):
    config = adapter.fork_recipe(name)
    adapter.check_recipe(config)
    assert adapter.branch(config) == name
    continuation = config_to_dict(config)["continuation"]
    assert continuation["optimizer"] == "fresh" and continuation["data_start_sequence"] == 1_937_920
    assert continuation["preserve_all_parameters"] is (name == "exp")
    assert config.optim.warmup_steps == 455 and config.optim.total_steps == 7570
    assert (config.optim.learning_rate, config.optim.min_learning_rate) == (0.002, 0.0002)
    if name == "query":
        model = config.model
        assert (model.attention_degree, model.attention_coefficient_conditioning, model.attention_coefficient_activation,
                model.attention_score_transform, model.attention_score_prefactor, model.attention_normalization,
                model.attention_output_scaling) == (8, "query", "tanh", "identity", "none", False, "head_rms_norm")
        assert model.attention_coefficient_amplitude_initial == 0.0


@pytest.mark.parametrize("name", tuple(adapter.PARTNERS))
@pytest.mark.parametrize("change, message", (
        (("optim", "weight_decay", 0.5), "optim.weight_decay"),
        (("optim", "warmup_steps", 0), "optim.warmup_steps"),
        (("train", "eval_every", 100), "train.eval_every"),
        (("train", "retain_terminal_checkpoint", False), "terminal checkpoint"),
        (("model", "attention_head_rms_prescale", "none"), "head RMSNorm"),
        (("continuation", "parent_checkpoint_sha256", "0" * 64), "parent_checkpoint_sha256"),
        (("continuation", "parent_step", 7500), "Continuation requires|fork requires|parent_step"),
))
def test_a_child_that_departs_from_its_partner_or_the_fork_is_refused(name, change, message):
    config = copy.deepcopy(adapter.fork_recipe(name))
    section, key, value = change
    setattr(config, section, replace(getattr(config, section), **{key: value}))
    with pytest.raises(ValueError, match=message):
        adapter.check_recipe(config)


def test_the_exp_child_keeps_every_parameter_and_the_query_child_its_degree():
    exp = adapter.fork_recipe("exp")
    exp.continuation = replace(exp.continuation, preserve_all_parameters=False)
    with pytest.raises(ValueError, match="preserve_all_parameters"):
        adapter.check_recipe(exp)
    query = adapter.fork_recipe("query")
    query.model = replace(query.model, attention_degree=4)
    with pytest.raises(ValueError, match="degree"):
        adapter.check_recipe(query)


@pytest.mark.parametrize("name", tuple(adapter.PARTNERS))
def test_each_child_qualifies_its_partners_kernel_with_the_fork_tests(name):
    config = adapter.fork_recipe(name)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == {"exp": "fused", "query": "mosaic"}[name]
    assert spec.check_initialization is adapter.check_initialization and spec.check_pilot is adapter.check_pilot
    assert "experiments/transformer_pp/tests/test_terminal_fork_acceptance.py" in spec.cpu_tests
    from scaffolds.transformer.acceptance import test_paths

    cpu = test_paths(spec, config_to_dict(config), resolved_backend=spec.required_backend, cpu=True)
    assert "scaffolds/transformer/tests/test_continuation_fork.py" in cpu
    gpu = test_paths(spec, config_to_dict(config), resolved_backend=spec.required_backend, cpu=False)
    assert ("scaffolds/transformer/tests/test_unnormalized_fused_attention.py" in gpu) is (name == "exp")


def _batches(first_sha256=adapter.FIRST_BATCH_SHA256, parent_sha256=adapter.PARENT_LAST_BATCH_SHA256):
    return {"parent_last_batch": {"sequence_start": 1_937_664, "sequence_stop": 1_937_920, "sha256": parent_sha256},
            "first_batch": {"sequence_start": 1_937_920, "sequence_stop": 1_938_176, "sha256": first_sha256}}


def test_the_pilot_must_read_the_parents_next_batch_from_update_zero():
    outcome = {"start_step": 0, "restored_step": None, "data_start_sequence": adapter.DATA_START_SEQUENCE,
               "checkpoint_audit": {"first_update": {"step": 1, "batch_sha256": adapter.FIRST_UPDATE_BATCH_SHA256}},
               "fork_batches": _batches()}
    facts = adapter.check_pilot(None, outcome, {})
    assert facts["data_start_sequence"] == 1_937_920 and facts["first_batch"] == _batches()["first_batch"]
    for change in ({"data_start_sequence": adapter.DATA_START_SEQUENCE - 256}, {"start_step": 7570},
                   {"restored_step": 20},
                   {"checkpoint_audit": {"first_update": {"step": 1, "batch_sha256": adapter.FIRST_BATCH_SHA256}}},
                   {"fork_batches": _batches(first_sha256=adapter.PARENT_LAST_BATCH_SHA256)},
                   {"fork_batches": _batches(parent_sha256=adapter.FIRST_BATCH_SHA256)},
                   {"fork_batches": None}):
        with pytest.raises(RuntimeError, match="update zero"):
            adapter.check_pilot(None, {**outcome, **change}, {})


def test_the_forks_train_on_one_h100():
    adapter.require_gpu_type(["recipe.yaml", "--gpu-type", "H100"])
    adapter.require_gpu_type(["recipe.yaml", "--gpu-type", "H100", "--gpu-count", "1"])
    with pytest.raises(ValueError, match="H100 only"):
        adapter.require_gpu_type(["recipe.yaml", "--gpu-type", "B300"])
    with pytest.raises(ValueError, match="one GPU"):
        adapter.require_gpu_type(["recipe.yaml", "--gpu-type", "H100", "--gpu-count", "2"])
    adapter.require_gpu_type(["recipe.yaml", "--gpu-type", "B300", "--runtime-env"])
