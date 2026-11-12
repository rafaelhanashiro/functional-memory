"""Event-triggered capture of exact pre-update states for gradient spikes (training/gradient_events.py)."""
import json
import math
import shutil
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.data.loaders import make_train_iterator
from scaffolds.transformer.tests.test_continuation import child_config, pretrained  # noqa: F401 (fixture)
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import (
    gradient_event_directory, make_checkpoint_manager, restore_latest_checkpoint, restore_state, state_sha256,
)
from scaffolds.transformer.training.distributed import resolve_batch_layout
from scaffolds.transformer.training.gradient_events import (
    GradientEventCapture, GradientEventRule, read_state, select_for_retention,
)
from scaffolds.transformer.training.optim import make_learning_rate_schedule
from scaffolds.transformer.training.state import create_state

# A deliberately low threshold, so the tiny model's ordinary norm fluctuations trigger events.
RULE = {"gradient_event_factor": 1.2, "gradient_event_min_norm": 0.0, "gradient_event_history": 3,
        "gradient_event_min_history": 2, "gradient_event_max_states": 2}
# Fields that identify an event, as opposed to the execution attempt that logged it.
ATTEMPT_FIELDS = {"attempt_id", "timestamp_utc", "capture_seconds"}


def capture_config(path, *, capture=True, ckpt_every=4, micro_batch_size=2):
    cfg = tiny_config(path)
    cfg.optim.total_steps = 12
    cfg.train.eval_every, cfg.train.ckpt_every, cfg.train.micro_batch_size = 4, ckpt_every, micro_batch_size
    cfg.train = replace(cfg.train, gradient_event_capture=capture, **RULE)
    return cfg


def single_device(monkeypatch):
    """Train on one device even when the test process exposes several."""
    monkeypatch.setattr(loop, "resolve_batch_layout", partial(resolve_batch_layout, available_device_count=1))
    monkeypatch.setattr(loop, "select_data_parallel_devices", lambda layout: tuple(jax.local_devices()[:1]))


def recorder(step, updates):
    """``step`` recording every update's input and output state hashes and its returned loss and numerics."""
    def recording(state, *args, **kwargs):
        outputs = step(state, *args, **kwargs)
        updates.append({"pre": state_sha256(state), "post": state_sha256(outputs[0]),
                        "loss": np.asarray(outputs[1]), "numerics": np.asarray(outputs[2])})
        return outputs
    return recording


def event_rows(cfg):
    path = cfg.train.workdir / "gradient_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def identity(row):
    return {key: value for key, value in row.items() if key not in ATTEMPT_FIELDS}


def spike_rows(cfg):
    return [row for row in event_rows(cfg) if row["event"] == "gradient_spike"]


def current_events(rows):
    """The final trajectory's events, read from the log alone: a ``resume`` row supersedes
    the rows logged before it for updates after its restored step."""
    current = []
    for row in rows:
        if row["event"] == "resume":
            current = [earlier for earlier in current if earlier["update"] <= (row["restored_step"] or 0)]
        elif row["event"] in ("gradient_spike", "nonfinite_update"):
            current.append(row)
    return current


def capture_names(cfg):
    return sorted(path.name for path in gradient_event_directory(cfg).iterdir())


def capture_checkpoint(cfg, update):
    """A capture's step directory: named by the state's own step, inside the update's root."""
    return gradient_event_directory(cfg) / str(update) / str(update - 1)


def preempted_before(update, step):
    """``step`` stopping, as a preemption would, when it is about to run update ``update``."""
    def stopping(state, *args, **kwargs):
        if int(state.step) == update - 1:
            raise RuntimeError("simulated preemption")
        return step(state, *args, **kwargs)
    return stopping


def listed_captures(cfg):
    """The record's status and the updates of the captures it lists."""
    record = json.loads((cfg.train.workdir / "record.json").read_text())
    return record["status"], [item["update"] for item in record["diagnostic_checkpoints"]]


def expected_events(norms, *, factor=1.2, min_norm=0.0, history=3, min_history=2, warmup=0):
    """The rule, written independently: updates whose norm exceeds the threshold from earlier norms."""
    events, earlier = [], []
    for update, norm in enumerate(norms, start=1):
        recent = sorted(earlier[-history:])
        if update > warmup and math.isfinite(norm) and len(recent) >= min_history:
            middle = len(recent) // 2
            median = recent[middle] if len(recent) % 2 else (recent[middle - 1] + recent[middle]) / 2
            if norm > max(min_norm, factor * median):
                events.append(update)
        if math.isfinite(norm):
            earlier.append(norm)
    return events


def expected_kept(events, norms, max_states=2):
    """The first event and the largest later ones, equal norms keeping the earlier event."""
    later = sorted(events[1:], key=lambda update: (-norms[update - 1], update))[:max_states - 1]
    return sorted(events[:1] + later)


def rule(**overrides):
    return GradientEventRule(**{"warmup_steps": 10, "factor": 5.0, "min_norm": 1.0, "history": 100,
                                "min_history": 20, "max_states": 3, **overrides})


def test_rule_needs_post_warmup_history_and_a_norm_above_its_threshold():
    history = [0.5] * 20
    assert not rule().decide(10, 100.0, history).triggered  # The update's learning rate is still warming up.
    assert rule().decide(11, 100.0, history).triggered
    assert not rule().decide(11, 100.0, history[:19]).triggered  # Fewer than min_history earlier norms.
    at_threshold = rule().decide(11, 2.5, history)  # factor x median = 2.5 > min_norm
    assert (at_threshold.running_median, at_threshold.threshold, at_threshold.triggered) == (0.5, 2.5, False)
    assert rule().decide(11, 2.5000001, history).triggered
    quiet = [0.1] * 20  # factor x median = 0.5, below min_norm
    assert rule().decide(11, 1.0, quiet).threshold == 1.0
    assert not rule().decide(11, 1.0, quiet).triggered and rule().decide(11, 1.01, quiet).triggered
    assert not rule().decide(11, math.inf, history).triggered
    assert not rule().decide(11, math.nan, history).triggered
    # The median covers the last `history` norms; an even count averages the middle two.
    assert rule(history=3, min_history=2).decide(11, 0.0, [100.0, 1.0, 2.0, 3.0]).running_median == 2.0
    assert rule(history=3, min_history=2).decide(11, 0.0, [1.0, 4.0]).running_median == 2.5


def test_history_holds_every_finite_norm_including_events(tmp_path):
    cfg = capture_config(tmp_path)
    capture = GradientEventCapture(cfg, resolve_batch_layout(cfg, available_device_count=1),
                                   attempt_id="test", device_kind="cpu")
    norms = [1.0, 1.0, 10.0, math.nan, 1.0]
    assert [capture.observe(update, norm) is not None for update, norm in enumerate(norms, start=1)] == [
        False, False, True, False, False]
    assert list(capture.history) == [1.0, 10.0, 1.0]
    with pytest.raises(RuntimeError, match="pre-clip gradient norm"):
        capture.observe(6, None)


def test_retention_keeps_the_first_event_and_the_largest_later_events():
    kept, outcomes = [], []
    for update, norm in [(5, 3.0), (6, 2.0), (7, 4.0), (8, 1.0), (9, 2.5), (10, 9.0), (11, 4.0)]:
        keep, replaced = select_for_retention(kept, norm, 3)
        if keep:
            if replaced is not None:
                kept.remove(replaced)
            kept.append({"update": update, "gradient_norm_pre_clip": norm})
        outcomes.append((update, keep, replaced["update"] if replaced is not None else None))
    # The first event is never replaced; 11 ties the smallest later event and keeps the earlier one.
    assert outcomes == [(5, True, None), (6, True, None), (7, True, None), (8, False, None),
                        (9, True, 6), (10, True, 9), (11, False, None)]
    assert [entry["update"] for entry in kept] == [5, 7, 10]
    assert select_for_retention([{"update": 5, "gradient_norm_pre_clip": 1.0}], 100.0, 1) == (False, None)


@pytest.mark.parametrize("micro_batch_size", [2, 4])
def test_capture_leaves_training_unchanged_and_keeps_exact_pre_update_states(tmp_path, monkeypatch,
                                                                            micro_batch_size):
    # Microbatch 2 accumulates two microbatches per update; 4 is one batch per update.
    single_device(monkeypatch)
    name = "accumulated_train_step" if micro_batch_size == 2 else "train_step"
    step = getattr(loop, name)
    runs = {}
    for capture in (False, True):
        updates = []
        monkeypatch.setattr(loop, name, recorder(step, updates))
        cfg = capture_config(tmp_path / f"capture-{capture}", capture=capture, micro_batch_size=micro_batch_size)
        runs[capture] = cfg, updates, loop.run_training(cfg, audit_checkpoint=True)
    (plain_cfg, plain, plain_result), (cfg, captured, result) = runs[False], runs[True]

    # Losses, gradient norms, every state, and the final checkpoint are bitwise identical.
    assert len(plain) == len(captured) == 12
    for without, with_capture in zip(plain, captured, strict=True):
        assert without["pre"] == with_capture["pre"] and without["post"] == with_capture["post"]
        assert without["loss"].tobytes() == with_capture["loss"].tobytes()
        assert without["numerics"].tobytes() == with_capture["numerics"].tobytes()
    assert (plain_result["checkpoint_audit"]["final_state_sha256"]
            == result["checkpoint_audit"]["final_state_sha256"])
    # Without capture, nothing new is written.
    assert not (plain_cfg.train.workdir / "gradient_events.jsonl").exists()
    assert not gradient_event_directory(plain_cfg).exists()
    assert "diagnostic_checkpoints" not in json.loads((plain_cfg.train.workdir / "record.json").read_text())

    norms = [float(update["numerics"][1]) for update in captured]
    events = expected_events(norms)
    kept = expected_kept(events, norms)
    rows = event_rows(cfg)
    assert [row["update"] for row in rows] == events
    # The scenario exercises a replacement of a kept later event.
    assert len(events) >= 3 and kept != events[:2]
    assert [row["update"] for row in rows if row["replaces"] is not None]
    assert capture_names(cfg) == sorted([*map(str, kept), "state.json"])
    assert [entry["update"] for entry in read_state(gradient_event_directory(cfg))["current"]["retained"]] == kept

    layout = resolve_batch_layout(cfg, available_device_count=1)
    template = create_state(cfg, jax.random.PRNGKey(cfg.train.seed))
    schedule = make_learning_rate_schedule(cfg)
    assert float(schedule(0)) != float(schedule(1)), "the scenario needs the schedule to move between updates"
    for row in rows:
        update = row["update"]
        assert row["learning_rate"] == float(schedule(update - 1))
        assert row["gradient_norm_pre_clip"] == norms[update - 1]
        assert row["loss"] == float(captured[update - 1]["loss"])
        assert row["pre_update_state_sha256"] == captured[update - 1]["pre"]
        assert row["sequences"] == {"first": (update - 1) * 4, "end": update * 4}
        assert row["microbatch_layout"]["grad_accum_steps"] == layout.grad_accum_steps
        assert row["gradient_clip_scale"] == pytest.approx(cfg.optim.grad_clip_norm / norms[update - 1])
        if update not in kept:
            continue
        directory = gradient_event_directory(cfg) / str(update)
        assert json.loads((directory / "event.json").read_text()) == row
        assert row["capture_directory"] == f"gradient_events/{update}" and row["pre_update_step"] == update - 1
        restored = restore_state(capture_checkpoint(cfg, update), template)
        assert int(restored.step) == update - 1
        # The capture is also a one-step manager root whose step is the state's own.
        with ocp.CheckpointManager(directory, ocp.StandardCheckpointer(),
                                   ocp.CheckpointManagerOptions(read_only=True)) as manager:
            assert list(manager.all_steps()) == [update - 1]
            managed = manager.restore(update - 1, args=ocp.args.StandardRestore({"state": template}))["state"]
        assert int(managed.step) == update - 1 and state_sha256(managed) == captured[update - 1]["pre"]
        assert state_sha256(restored) == captured[update - 1]["pre"]
        # The logged coordinates rebuild the update's exact inputs, and the update itself.
        batches = make_train_iterator(replace(cfg.data, batch_size=layout.global_micro_batch_size),
                                      start_sequence=row["sequences"]["first"])
        host = np.stack([next(batches) for _ in range(layout.grad_accum_steps)])
        step_rng = jax.random.fold_in(jax.random.PRNGKey(row["seed"]), update)
        if micro_batch_size == 2:
            audit = loop._update_audit(update, jnp.asarray(host), step_rng)
            outputs = step(restored, jnp.asarray(host), jax.random.split(step_rng, layout.grad_accum_steps),
                           check_all_gradients=True, collect_telemetry=True)
        else:
            audit = loop._update_audit(update, jnp.asarray(host[0])[None], step_rng)
            outputs = step(restored, jnp.asarray(host[0]), step_rng, check_all_gradients=True,
                           collect_telemetry=True)
        assert (audit["batch_sha256"], audit["rng_sha256"]) == (row["batch_sha256"], row["rng_sha256"])
        assert state_sha256(outputs[0]) == captured[update - 1]["post"]
        assert np.asarray(outputs[1]).tobytes() == captured[update - 1]["loss"].tobytes()


def test_requeue_restores_the_norm_history_and_kept_captures(tmp_path, monkeypatch):
    single_device(monkeypatch)
    full = capture_config(tmp_path / "full")
    loop.run_training(full)
    resumed = capture_config(tmp_path / "resumed")
    assert loop.run_training(resumed, stop_after_step=4)["final_step"] == 4
    paused = read_state(gradient_event_directory(resumed))
    assert list(paused["snapshots"]) == ["4"] and len(paused["snapshots"]["4"]["history"]) == 3
    first_attempt = {row["attempt_id"] for row in event_rows(resumed)}
    assert loop.run_training(resumed)["restored_step"] == 4
    # The resumed run logs the same events, keeps the same captures, and ends with the same history.
    assert [identity(row) for row in spike_rows(resumed)] == [identity(row) for row in spike_rows(full)]
    assert [row["restored_step"] for row in event_rows(resumed) if row["event"] == "resume"] == [4]
    final, reference = (read_state(gradient_event_directory(cfg)) for cfg in (resumed, full))
    assert final["current"] == reference["current"] and final["snapshots"] == reference["snapshots"]
    assert capture_names(resumed) == capture_names(full)
    # The first event after the resume relied on norms from before it: the resumed process alone
    # had seen fewer than min_history of them.
    after = next(row for row in spike_rows(resumed) if row["attempt_id"] not in first_attempt)
    assert after["update"] - 1 - 4 < RULE["gradient_event_min_history"] <= after["history_length"]


def test_resume_discards_captures_made_after_its_checkpoint(tmp_path, monkeypatch):
    # Resume checkpoints at 2, 6, and 10 are saved asynchronously; 4, 8, and 12 validate.
    single_device(monkeypatch)
    reference = capture_config(tmp_path / "reference", ckpt_every=2)
    loop.run_training(reference)
    replacements = {row["update"]: row["replaces"] for row in event_rows(reference) if row["replaces"]}
    assert replacements == {7: 6}, "the scenario needs update 7 to replace the capture of update 6"

    cfg = capture_config(tmp_path / "run", ckpt_every=2)
    actual = loop.accumulated_train_step
    monkeypatch.setattr(loop, "accumulated_train_step", preempted_before(8, actual))
    with pytest.raises(RuntimeError, match="simulated preemption"):
        loop.run_training(cfg)
    # Update 7 replaced update 6, whose capture checkpoint 6's state still references.
    assert capture_names(cfg) == ["3", "6", "7", "state.json"]
    monkeypatch.setattr(loop, "accumulated_train_step", actual)
    assert loop.run_training(cfg)["restored_step"] == 6
    discarded = [row for row in event_rows(cfg) if row["event"] == "capture_discarded"]
    assert [row["update"] for row in discarded] == [7]
    assert discarded[0]["reason"] == "made after checkpoint step 6, from which the run repeats the update"
    # The resume row precedes what the resume discarded, and names the step it repeats from.
    rows = event_rows(cfg)
    boundary = [row for row in rows if row["event"] == "resume"]
    assert [row["restored_step"] for row in boundary] == [6]
    assert rows.index(boundary[0]) < rows.index(discarded[0])
    assert capture_names(cfg) == capture_names(reference) == ["3", "7", "state.json"]
    final, expected = (read_state(gradient_event_directory(run)) for run in (cfg, reference))
    assert final["current"] == expected["current"] and final["snapshots"] == expected["snapshots"]
    spikes = [identity(row) for row in spike_rows(cfg)]
    assert spikes[:3] == [identity(row) for row in spike_rows(reference)]
    # The repeated update 7 is the same event; the resumed process had no preceding update's telemetry.
    repeated, original = ({key: value for key, value in row.items() if key != "preceding_telemetry"}
                          for row in (spikes[3], spikes[2]))
    assert len(spikes) == 4 and repeated == original and spikes[3]["preceding_telemetry"] is None
    # The log alone tells the superseded row of update 7 from the row of the final trajectory.
    current = current_events(rows)
    assert [row["update"] for row in current] == [row["update"] for row in spike_rows(reference)] == [3, 6, 7]
    assert current[-1]["attempt_id"] == boundary[0]["attempt_id"] != spike_rows(cfg)[2]["attempt_id"]


def test_discard_reason_names_a_capture_replaced_before_the_restored_checkpoint(tmp_path, monkeypatch):
    # With a checkpoint at every update, update 7 replaces update 6's capture before checkpoint 7
    # is saved, and checkpoint 6's snapshot still references it when the run stops.
    single_device(monkeypatch)
    cfg = capture_config(tmp_path, ckpt_every=1)
    actual = loop.accumulated_train_step
    monkeypatch.setattr(loop, "accumulated_train_step", preempted_before(8, actual))
    with pytest.raises(RuntimeError, match="simulated preemption"):
        loop.run_training(cfg)
    assert capture_names(cfg) == ["3", "6", "7", "state.json"]
    monkeypatch.setattr(loop, "accumulated_train_step", actual)
    assert loop.run_training(cfg, stop_after_step=8)["restored_step"] == 7
    discarded = [row for row in event_rows(cfg) if row["event"] == "capture_discarded"]
    assert [row["update"] for row in discarded] == [6]
    assert discarded[0]["reason"] == "replaced at or before checkpoint step 7, whose state no longer keeps it"
    assert capture_names(cfg) == ["3", "7", "state.json"]


def test_restart_without_a_checkpoint_supersedes_the_earlier_attempts_events(tmp_path, monkeypatch):
    single_device(monkeypatch)
    cfg = capture_config(tmp_path)
    actual = loop.accumulated_train_step
    monkeypatch.setattr(loop, "accumulated_train_step", preempted_before(4, actual))
    with pytest.raises(RuntimeError, match="simulated preemption"):
        loop.run_training(cfg)
    assert capture_names(cfg) == ["3", "state.json"]
    monkeypatch.setattr(loop, "accumulated_train_step", actual)
    assert loop.run_training(cfg, stop_after_step=4)["restored_step"] is None
    rows = event_rows(cfg)
    assert [row["event"] for row in rows] == ["gradient_spike", "resume", "capture_discarded", "gradient_spike"]
    assert rows[1]["restored_step"] is None
    assert rows[2]["update"] == 3
    assert rows[2]["reason"] == "made by an earlier execution that left no checkpoint of this run"
    assert [row["attempt_id"] for row in current_events(rows)] == [rows[3]["attempt_id"]]
    assert capture_names(cfg) == ["3", "state.json"]


def test_continuation_baseline_checkpoint_resumes_capture(pretrained, tmp_path):
    configs = []
    for name in ("full", "resumed"):
        cfg = child_config(pretrained, tmp_path / name, learned=False)
        cfg.train = replace(cfg.train, gradient_event_capture=True, **RULE)
        configs.append(cfg)
    full, resumed = configs
    loop.run_training(full)
    # Stop at the baseline checkpoint itself, before the child's first update.
    assert loop.run_training(resumed, stop_after_step=3)["final_step"] == 3
    baseline = read_state(gradient_event_directory(resumed))
    assert baseline["snapshots"] == {"3": {"step": 3, "history": [], "retained": []}}
    assert loop.run_training(resumed)["restored_step"] == 3
    final, reference = (read_state(gradient_event_directory(cfg)) for cfg in configs)
    assert final["snapshots"]["6"]["history"] == reference["snapshots"]["6"]["history"]
    assert len(final["snapshots"]["6"]["history"]) == 3
    assert final["current"] == reference["current"]
    assert [identity(row) for row in spike_rows(resumed)] == [identity(row) for row in spike_rows(full)]
    assert [row["restored_step"] for row in event_rows(resumed) if row["event"] == "resume"] == [3]


def test_resume_requires_the_checkpoint_snapshot(tmp_path, monkeypatch):
    single_device(monkeypatch)
    cfg = capture_config(tmp_path)
    loop.run_training(cfg, stop_after_step=4)
    (gradient_event_directory(cfg) / "state.json").unlink()
    with pytest.raises(ValueError, match="No gradient-event state for checkpoint step 4"):
        loop.run_training(cfg)


def test_nonfinite_update_keeps_its_pre_update_state_outside_the_cap(tmp_path, monkeypatch):
    single_device(monkeypatch)
    cfg = capture_config(tmp_path)
    actual = loop.accumulated_train_step

    def nonfinite_update_9(*args, **kwargs):
        updated, loss, *rest = actual(*args, **kwargs)
        return (updated, jnp.where(updated.step == 9, jnp.nan, loss), *rest)

    monkeypatch.setattr(loop, "accumulated_train_step", nonfinite_update_9)
    with pytest.raises(FloatingPointError, match="at step 9"):
        loop.run_training(cfg)
    assert (tmp_path / "numerical_failure_step_00000009.json").exists()
    assert capture_names(cfg) == ["3", "7", "9", "state.json"]
    failure = event_rows(cfg)[-1]
    assert (failure["event"], failure["update"], failure["loss"], failure["retained"]) == (
        "nonfinite_update", 9, "NaN", True)
    assert failure["learning_rate"] == float(make_learning_rate_schedule(cfg)(8))
    # Update 9 started from checkpoint 8's state.
    template = create_state(cfg, jax.random.PRNGKey(0))
    with make_checkpoint_manager(cfg) as manager:
        checkpoint, checkpoint_step = restore_latest_checkpoint(manager, template)
    assert checkpoint_step == 8
    captured = restore_state(capture_checkpoint(cfg, 9), template)
    assert state_sha256(captured) == state_sha256(checkpoint) == failure["pre_update_state_sha256"]
    record = json.loads((tmp_path / "record.json").read_text())
    assert record["status"] == "numerical_failure"
    kept = {item["update"]: item for item in record["diagnostic_checkpoints"]}
    assert sorted(kept) == [3, 7, 9]
    assert kept[9]["justification"] == "pre-update state of nonfinite update 9, kept for its numerical failure"
    assert kept[9]["event"] == "nonfinite_update" and kept[9]["kind"] == "directory"

    # Training again from checkpoint 8 supersedes the failure, and with it the failure's capture.
    monkeypatch.setattr(loop, "accumulated_train_step", actual)
    assert loop.run_training(cfg)["final_step"] == 12
    assert capture_names(cfg) == ["3", "7", "state.json"]
    discarded = [row for row in event_rows(cfg) if row["event"] == "capture_discarded"]
    assert [row["update"] for row in discarded] == [9]
    assert discarded[0]["reason"] == "kept for a numerical failure that this retraining supersedes"
    assert listed_captures(cfg) == ("complete", [3, 7])


def test_a_retraining_lists_no_capture_until_it_reaches_its_own_outcome(tmp_path, monkeypatch):
    single_device(monkeypatch)
    cfg = capture_config(tmp_path)
    actual = loop.accumulated_train_step

    def nonfinite_update_9(*args, **kwargs):
        updated, loss, *rest = actual(*args, **kwargs)
        return (updated, jnp.where(updated.step == 9, jnp.nan, loss), *rest)

    monkeypatch.setattr(loop, "accumulated_train_step", nonfinite_update_9)
    with pytest.raises(FloatingPointError, match="at step 9"):
        loop.run_training(cfg)
    assert listed_captures(cfg) == ("numerical_failure", [3, 7, 9])

    # Training again from checkpoint 8 supersedes the failure and deletes its capture. Whether
    # the retraining then pauses or stops with an ordinary failure, the record lists no capture.
    monkeypatch.setattr(loop, "accumulated_train_step", actual)
    assert loop.run_training(cfg, stop_after_step=10)["restored_step"] == 8
    assert capture_names(cfg) == ["3", "7", "state.json"]
    assert listed_captures(cfg) == ("paused", [])
    monkeypatch.setattr(loop, "accumulated_train_step", preempted_before(11, actual))
    with pytest.raises(RuntimeError, match="simulated preemption"):
        loop.run_training(cfg)
    assert listed_captures(cfg) == ("failed", [])
    assert json.loads((tmp_path / "record.json").read_text()).get("numerical_failure_path") is None

    # The run's own outcome lists what is kept then, and only that.
    monkeypatch.setattr(loop, "accumulated_train_step", actual)
    assert loop.run_training(cfg)["final_step"] == 12
    assert listed_captures(cfg) == ("complete", [3, 7])


def test_replay_tool_continues_a_capture_enabled_run_without_capture(tmp_path, monkeypatch):
    # The tool copies one checkpoint without the capture root, and capture cannot change the replay.
    from scaffolds.transformer.validation.replay_checkpoint import prepare

    single_device(monkeypatch)
    cfg = capture_config(tmp_path / "run")
    loop.run_training(cfg, stop_after_step=8)
    (cfg.train.workdir / "data_identity.json").write_text("{}")  # Synthetic data has none; the tool copies it.
    replay = load_config(prepare(cfg.train.workdir, 8, tmp_path / "scratch"))
    assert replay.train.gradient_event_capture is False
    result = loop.run_training(replay, resume=True, stop_after_step=10, audit_checkpoint=True)
    assert (result["restored_step"], result["final_step"]) == (8, 10)
    assert not gradient_event_directory(replay).exists() and not (tmp_path / "scratch/gradient_events.jsonl").exists()
    continued = loop.run_training(cfg, stop_after_step=10, audit_checkpoint=True)
    assert result["checkpoint_audit"]["final_state_sha256"] == continued["checkpoint_audit"]["final_state_sha256"]


def test_finalization_keeps_and_records_only_the_kept_captures(tmp_path, monkeypatch):
    from scaffolds.records import capture_artifacts, create_run, update_record, validate_record
    from scaffolds.transformer.retention import finalize_run

    single_device(monkeypatch)
    study = tmp_path / "experiments/demo"
    study.mkdir(parents=True)
    (study / "sweep.yaml").write_text(json.dumps({"study_id": "demo", "run_ids": []}))
    recipe = config_to_dict(capture_config(tmp_path / "placeholder"))
    recipe["train"]["source_commit"] = "a" * 40
    run = create_run(study, recipe, label="capture")
    cfg = load_config(run / "config.yaml")
    monkeypatch.setattr(loop, "evaluate", lambda state, *args, **kwargs: float(state.step))
    loop.run_training(cfg)
    root = gradient_event_directory(cfg)
    (root / "5").mkdir()
    (root / "5" / "leftover").write_text("not a kept capture")
    finalize_run(cfg)
    assert capture_names(cfg) == ["3", "7", "state.json"]
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == ["4", "gradient_events"]
    record = json.loads((run / "record.json").read_text())
    kept = record["diagnostic_checkpoints"]
    assert [item["update"] for item in kept] == [3, 7]
    assert [item["pre_update_step"] for item in kept] == [2, 6]
    for item in kept:
        assert item["kind"] == "directory" and Path(item["path"]).resolve() == (root / str(item["update"])).resolve()
        assert item["justification"] == (f"pre-update state of gradient event at update {item['update']}, "
                                         f"norm {item['gradient_norm_pre_clip']:.6g}")
        assert {entry["path"] for entry in item["files"]} >= {"event.json",
                                                              f"{item['pre_update_step']}/default/_METADATA"}
    assert json.loads((run / "retention.json").read_text())["retained_gradient_event_updates"] == [3, 7]
    assert any(entry["path"].endswith("gradient_events.jsonl") for entry in record["artifacts"])
    assert {item["path"] for item in kept} <= set(validate_record(run)["verified"])
    assert finalize_run(cfg)["retained_gradient_event_updates"] == [3, 7]
    # On another cluster the captures are unavailable, not an error; changed bytes are.
    shutil.move(root / "7", tmp_path / "elsewhere")
    assert kept[1]["path"] in validate_record(run, checkpoint_hashes=False)["unavailable"]
    shutil.move(tmp_path / "elsewhere", root / "7")
    (root / "7" / "event.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_record(run)
    # A kept diagnostic checkpoint needs its justification.
    with pytest.raises(ValueError, match="requires a justification"):
        capture_artifacts(run, diagnostic_checkpoints=[{"path": root / "3", "justification": " "}])
    update_record(run, {"diagnostic_checkpoints": [{key: value for key, value in kept[0].items()
                                                    if key != "justification"}]})
    with pytest.raises(ValueError, match="without a directory descriptor and justification"):
        validate_record(run, verify_hashes=False)


def test_distributed_capture_copies_the_donated_state(tmp_path, monkeypatch):
    if len(jax.devices("cpu")) < 2:
        pytest.skip("Requires two host devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    actual = loop.make_data_parallel_functions
    runs = {}
    for capture in (False, True):
        updates = []

        def recording(*args, updates=updates, **kwargs):
            functions = actual(*args, **kwargs)

            def train_step(state, batches, rng):
                before = state_sha256(state)  # Hashed before the step donates the state.
                outputs = functions.train_step(state, batches, rng)
                updates.append({"pre": before, "post": state_sha256(outputs[0]),
                                "loss": np.asarray(outputs[1]), "numerics": np.asarray(outputs[2])})
                return outputs

            return replace(functions, train_step=train_step)

        monkeypatch.setattr(loop, "make_data_parallel_functions", recording)
        cfg = capture_config(tmp_path / f"capture-{capture}", capture=capture)
        runs[capture] = cfg, updates, loop.run_training(cfg, audit_checkpoint=True)
    (_, plain, plain_result), (cfg, captured, result) = runs[False], runs[True]
    assert result["data_parallel_devices"] == 2
    for without, with_capture in zip(plain, captured, strict=True):
        assert without["post"] == with_capture["post"]
        assert without["numerics"].tobytes() == with_capture["numerics"].tobytes()
    assert (plain_result["checkpoint_audit"]["final_state_sha256"]
            == result["checkpoint_audit"]["final_state_sha256"])
    rows = event_rows(cfg)
    assert rows and rows[0]["device_count"] == 2 and "fold_in(fold_in(step_rng, j), d)" in rows[0]["rng_derivation"]
    template = create_state(cfg, jax.random.PRNGKey(0))
    for entry in read_state(gradient_event_directory(cfg))["current"]["retained"]:
        restored = restore_state(capture_checkpoint(cfg, entry["update"]), template)
        assert state_sha256(restored) == captured[entry["update"] - 1]["pre"] == entry["pre_update_state_sha256"]
