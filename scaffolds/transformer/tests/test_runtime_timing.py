"""Acceptance timing isolates warm work without changing training or validation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.tests.test_continuation import child_config, pretrained
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop


def scientific_records(workdir):
    runtime_fields = {"attempt_id", "timestamp_utc", "tokens_per_second", "tokens_per_second_per_device",
                      "training_interval_seconds", "attention_backend_resume_requested"}

    def scientific(value):
        if isinstance(value, dict):
            return {key: scientific(item) for key, item in value.items() if key not in runtime_fields}
        if isinstance(value, list):
            return [scientific(item) for item in value]
        return value

    return [scientific(json.loads(line)) for line in (workdir / "metrics.jsonl").read_text().splitlines()]


def test_profiled_packed_training_and_resume_preserve_science_and_state(tmp_path):
    config = tiny_config(tmp_path / "ordinary")
    config.model.attention_degree = 2
    config.model.attention_backend = "blocked"
    config.optim.total_steps = 20
    config.train.eval_every = config.train.ckpt_every = 10
    config.data.source = "packed"
    # A single validation batch deliberately leaves no second batch to consume.
    # Profiling must replay its independent prefix rather than advance this data.
    for split, count in (("train", 84), ("eval", 4)):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5,
                                      num_sequences=count, shard_sequences=3)
        writer.write(np.arange(count * 5).reshape(count, 5) % 16)
        setattr(config.data, f"{split}_manifest", str(writer.finish()))
    ordinary = loop.run_training(config, audit_checkpoint=True)
    profiled_config = replace(config, train=replace(config.train, workdir=tmp_path / "profiled"))
    profiled = loop.run_training(profiled_config, audit_checkpoint=True, profile_timing=True)
    restarted_config = replace(config, train=replace(config.train, workdir=tmp_path / "restarted"))
    paused = loop.run_training(restarted_config, stop_after_step=10, audit_checkpoint=True, profile_timing=True)
    resumed = loop.run_training(restarted_config, audit_checkpoint=True, profile_timing=True)

    ordinary_audit, profiled_audit = ordinary["checkpoint_audit"], profiled["checkpoint_audit"]
    assert ordinary_audit == profiled_audit
    assert paused["checkpoint_audit"]["first_update"] == ordinary_audit["first_update"]
    assert paused["checkpoint_audit"]["next_update"] == resumed["checkpoint_audit"]["first_update"]
    assert paused["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert resumed["checkpoint_audit"]["final_state_sha256"] == ordinary_audit["final_state_sha256"]
    assert resumed["checkpoint_audit"]["checkpoint_state_sha256"] == ordinary_audit["checkpoint_state_sha256"]
    assert resumed["data_start_sequence"] == 40
    assert scientific_records(config.train.workdir) == scientific_records(profiled_config.train.workdir)
    assert scientific_records(config.train.workdir) == scientific_records(restarted_config.train.workdir)
    assert ordinary["validation_nll"] == profiled["validation_nll"] == resumed["validation_nll"]

    assert set(ordinary["timing"]) == {
        "train_step_seconds", "evaluation_seconds", "checkpoint_seconds", "runtime_seconds"}
    assert len(ordinary["timing"]["train_step_seconds"]) == 8
    timing = profiled["timing"]
    assert len(timing["train_step_seconds"]) == 20
    assert len(timing["diagnostic_seconds"]) == len(timing["evaluation_seconds"]) == 2
    assert len(timing["evaluation_batch_seconds"]) == 4  # Three at the first event, one at the second.
    assert len(timing["checkpoint_seconds"]) == 2
    for field in ("setup_seconds", "diagnostic_warmup_seconds", "evaluation_warmup_seconds"):
        assert 0 < timing[field] < timing["runtime_seconds"]
    for field in ("train_step_seconds", "diagnostic_seconds", "evaluation_batch_seconds", "checkpoint_seconds"):
        assert all(0 < sample < timing["runtime_seconds"] for sample in timing[field])
    for field in ("setup_seconds", "diagnostic_warmup_seconds", "evaluation_warmup_seconds"):
        assert field not in json.loads((profiled_config.train.workdir / "config.json").read_text())
    assert "profile_timing" not in (profiled_config.train.workdir / "config.yaml").read_text()


def test_profiled_continuation_preserves_baseline_and_measures_its_checkpoint(pretrained, tmp_path):
    ordinary_config = child_config(pretrained, tmp_path / "ordinary", learned=True, query=True)
    profiled_config = child_config(pretrained, tmp_path / "profiled", learned=True, query=True)
    ordinary = loop.run_training(ordinary_config, audit_checkpoint=True)
    profiled = loop.run_training(profiled_config, audit_checkpoint=True, profile_timing=True)
    assert ordinary["checkpoint_audit"] == profiled["checkpoint_audit"]
    assert scientific_records(ordinary_config.train.workdir) == scientific_records(profiled_config.train.workdir)
    assert len(profiled["timing"]["diagnostic_seconds"]) == 3  # Baseline, then two validations.
    assert len(profiled["timing"]["checkpoint_seconds"]) == 3
    assert len(ordinary["timing"]["checkpoint_seconds"]) == 2


def test_evaluation_batch_timer_covers_fetch_and_every_microbatch_without_changing_rng(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(loop, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    calls = []

    def batches():
        for batch in (np.ones((4, 3), np.int32), np.full((2, 5), 3, np.int32)):
            clock.now += 2.0
            yield batch

    def eval_step(state, batch, rng):
        clock.now += 3.0
        calls.append((np.asarray(batch), np.asarray(rng)))
        return float(batch[0, 0])

    monkeypatch.setattr(loop, "eval_step", eval_step)
    samples = []
    result = loop.evaluate(None, batches(), 2, micro_batch_size=2, eval_seed=7, batch_seconds=samples)
    assert result == 2.0  # Eight prediction tokens in each global batch.
    assert samples == [8.0, 5.0]  # Fetch + two forwards, then fetch + one forward.
    assert [batch.shape for batch, _ in calls] == [(2, 3), (2, 3), (2, 5)]
    for (_, rng), index in zip(calls, (0, 0, 1), strict=True):
        np.testing.assert_array_equal(rng, jax.random.fold_in(jax.random.PRNGKey(7), index))


@pytest.mark.parametrize("phase", ["diagnostic_warmup", "evaluation_warmup", "evaluation_profile"])
def test_profile_warmups_and_extra_samples_preserve_numerical_failure_checks(tmp_path, monkeypatch, phase):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    config.optim.total_steps = 2
    diagnostics = loop.attention_diagnostics
    evaluate = loop.evaluate
    diagnostic_calls = evaluation_calls = 0

    def checked_diagnostics(*args, **kwargs):
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        report = diagnostics(*args, **kwargs)
        if phase == "diagnostic_warmup" and diagnostic_calls == 1:
            report = {**report, "valid": False, "issues": ["injected invalid warmup"]}
        return report

    def checked_evaluation(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        result = evaluate(*args, **kwargs)
        failed_call = 1 if phase == "evaluation_warmup" else 3
        return float("nan") if phase != "diagnostic_warmup" and evaluation_calls == failed_call else result

    monkeypatch.setattr(loop, "attention_diagnostics", checked_diagnostics)
    monkeypatch.setattr(loop, "evaluate", checked_evaluation)
    with pytest.raises(FloatingPointError, match="Invalid attention diagnostics|Nonfinite validation loss"):
        loop.run_training(config, profile_timing=True)
    failure = json.loads((tmp_path / "numerical_failure_step_00000002.json").read_text())
    assert failure["step"] == 2
    assert not (tmp_path / "completion.json").exists()
    if phase == "diagnostic_warmup":
        assert diagnostic_calls == 1 and evaluation_calls == 0
        assert failure["attention_diagnostics"]["issues"] == ["injected invalid warmup"]
    else:
        assert evaluation_calls == (1 if phase == "evaluation_warmup" else 3)
        assert failure["loss"] == "NaN"
