"""Checkpoint audits must verify real resumed state and the next consumed input."""
import hashlib
import json
import shutil
from dataclasses import replace

import jax
import numpy as np
import pytest

from scaffolds.transformer.studies.polynomial_attention.sweep import parameter_count
from scaffolds.transformer.studies.polynomial_attention.tests.test_training import tiny_config
from scaffolds.transformer.studies.polynomial_attention.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.state import create_state


WIDTHS = (8, 16, 32, 64, 128)
DEGREES = (2, 4, 8, 16)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("degree", DEGREES)
def test_actual_checkpoint_state_and_next_input_match_uninterrupted(tmp_path, width, degree):
    config = tiny_config(tmp_path / "full")
    config.model.qk_head_dim, config.model.attention_degree = width, degree
    config.data.source = "packed"
    for split, count in (("train", 20), ("eval", 4)):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5,
                                      num_sequences=count, shard_sequences=3)
        writer.write(np.arange(count * 5).reshape(count, 5) % 16)
        setattr(config.data, f"{split}_manifest", str(writer.finish()))
    full = loop.run_training(config, audit_checkpoint=True)
    restarted = replace(config, train=replace(config.train, workdir=tmp_path / "restart"))
    before = loop.run_training(restarted, stop_after_step=2, audit_checkpoint=True)
    after = loop.run_training(restarted, audit_checkpoint=True)
    a, b = before["checkpoint_audit"], after["checkpoint_audit"]
    assert a["final_state_sha256"] == a["checkpoint_state_sha256"] == b["start_state_sha256"]
    assert a["next_update"] == b["first_update"]
    assert a["next_update"]["step"] == 3
    assert a["first_update"] == full["checkpoint_audit"]["first_update"]
    assert b["final_state_sha256"] == b["checkpoint_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    assert after["data_start_sequence"] == 2 * config.data.batch_size


@pytest.mark.parametrize("width", WIDTHS)
def test_initialization_independent_of_degree_and_parameter_count(tmp_path, width):
    digests = []
    for degree in DEGREES:
        config = tiny_config(tmp_path)
        config.model.qk_head_dim, config.model.attention_degree = width, degree
        state = create_state(config, jax.random.PRNGKey(0))
        digests.append(state_sha256(state.params))
        assert sum(x.size for x in jax.tree.leaves(state.params)) == parameter_count(config.model)
    assert len(set(digests)) == 1


def test_best_only_completed_run_is_idempotent_without_resurrecting_terminal(tmp_path, monkeypatch):
    config = tiny_config(tmp_path / "retained")
    config.model.attention_degree = 2
    original_evaluate = loop.evaluate

    def worse_terminal(state, *args, **kwargs):
        return original_evaluate(state, *args, **kwargs) + (1 if int(state.step) == 4 else 0)

    monkeypatch.setattr(loop, "evaluate", worse_terminal)
    result = loop.run_training(config, audit_checkpoint=True)
    assert result["retained_best_checkpoint_step"] == 2
    root = config.train.workdir
    completion_hash = hashlib.sha256((root / "completion.json").read_bytes()).hexdigest()
    shutil.rmtree(root / "checkpoints/4")
    (root / "retention.json").write_text(json.dumps({
        "format": "key-dimension-taylor-checkpoint-retention",
        "completion_sha256": completion_hash, "best_checkpoint_step": 2,
    }))
    original_metrics = (root / "metrics.jsonl").read_bytes()
    completed = loop.run_training(config, audit_checkpoint=True)
    assert completed["final_step"] == 4
    assert (root / "metrics.jsonl").read_bytes() == original_metrics
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["retained_terminal_checkpoint_step"] is None
    assert metadata["retained_best_checkpoint_step"] == 2
    assert metadata["validation_nll"] == result["validation_nll"]


def test_state_fingerprint_rejects_nonfinite_arrays_and_detects_optimizer_changes():
    original = {"parameters": np.ones(3, np.float32), "moment": np.zeros(3, np.float32)}
    changed = {**original, "moment": np.ones(3, np.float32)}
    assert state_sha256(original) != state_sha256(changed)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        state_sha256({"parameters": np.array([np.nan], np.float32)})


def test_best_selection_agrees_with_checkpoint_manager_on_exact_tie(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    monkeypatch.setattr(loop, "evaluate", lambda *args, **kwargs: 3.0)
    result = loop.run_training(config)
    selected = json.loads((tmp_path / "best_validation.json").read_text())
    assert selected["step"] == result["retained_best_checkpoint_step"] == 4


def test_overflowing_perplexity_is_a_recorded_numerical_failure(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    monkeypatch.setattr(loop, "evaluate", lambda *args, **kwargs: 1000.0)
    with pytest.raises(FloatingPointError, match="[Pp]erplexity overflow"):
        loop.run_training(config)
    failure = json.loads((tmp_path / "numerical_failure_step_00000002.json").read_text())
    assert "perplexity overflow" in failure["error"].lower()
