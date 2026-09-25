"""A lost checkpoint must not restart training over an existing scientific record."""
import json
import shutil

import pytest

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import validate_resume_progress


def write_json(path, payload):
    path.write_text(json.dumps(payload))


def test_missing_all_checkpoints_rejects_resume_without_changing_records(tmp_path):
    config = tiny_config(tmp_path)
    config.train.wandb_mode = "disabled"
    loop.run_training(config, stop_after_step=2)
    metrics = (tmp_path / "metrics.jsonl").read_bytes()
    metadata = (tmp_path / "metadata.json").read_bytes()
    shutil.rmtree(tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="recorded completed progress is step 2.*no checkpoint"):
        loop.run_training(config)

    assert (tmp_path / "metrics.jsonl").read_bytes() == metrics
    assert (tmp_path / "metadata.json").read_bytes() == metadata


def test_older_remaining_checkpoint_rejects_resume_without_changing_records(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config.train.wandb_mode = "disabled"
    config.optim.total_steps = 6
    monkeypatch.setattr(loop, "evaluate", lambda state, *args, **kwargs: float(state.step))
    loop.run_training(config, stop_after_step=4)
    assert (tmp_path / "checkpoints/2").is_dir()
    metrics = (tmp_path / "metrics.jsonl").read_bytes()
    metadata = (tmp_path / "metadata.json").read_bytes()
    shutil.rmtree(tmp_path / "checkpoints/4")

    with pytest.raises(ValueError, match="recorded completed progress is step 4.*checkpoint is step 2"):
        loop.run_training(config)

    assert (tmp_path / "metrics.jsonl").read_bytes() == metrics
    assert (tmp_path / "metadata.json").read_bytes() == metadata


def test_pause_before_first_validation_still_requires_saved_checkpoint(tmp_path):
    config = tiny_config(tmp_path)
    config.train.wandb_mode = "disabled"
    loop.run_training(config, stop_after_step=1)
    assert not (tmp_path / "metrics.jsonl").exists()
    metadata = (tmp_path / "metadata.json").read_bytes()
    shutil.rmtree(tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="recorded completed progress is step 1.*no checkpoint"):
        loop.run_training(config)

    assert (tmp_path / "metadata.json").read_bytes() == metadata


def test_never_checkpointed_run_and_uncommitted_validation_allow_initialization(tmp_path):
    config = tiny_config(tmp_path)
    validate_resume_progress(config, None)
    write_json(tmp_path / "metadata.json", {
        "status": "running", "step": 0, "rng_state": {"completed_updates": 0},
    })
    write_json(tmp_path / "pending_validation.json", {"event": "validation", "step": 2})
    validate_resume_progress(config, None)


def test_failed_attempt_step_is_not_committed_progress(tmp_path):
    config = tiny_config(tmp_path)
    write_json(tmp_path / "metadata.json", {
        "status": "failed", "step": 3, "rng_state": {"completed_updates": 2},
        "numerical_failure_path": "numerical_failure_step_00000003.json",
    })
    validate_resume_progress(config, 2)
    with pytest.raises(ValueError, match="recorded completed progress is step 2"):
        validate_resume_progress(config, 1)


@pytest.mark.parametrize("record_kind", ["metrics", "best", "running", "paused"])
def test_each_durable_progress_record_prevents_rollback(tmp_path, record_kind):
    config = tiny_config(tmp_path)
    if record_kind == "metrics":
        write_json(tmp_path / "metrics.jsonl", {
            "event": "validation", "step": 4, "checkpoint_step": 4,
        })
    elif record_kind == "best":
        write_json(tmp_path / "best_validation.json", {"step": 4})
    else:
        write_json(tmp_path / "metadata.json", {"status": record_kind, "step": 4})
    with pytest.raises(ValueError, match="recorded completed progress is step 4"):
        validate_resume_progress(config, 2)
    validate_resume_progress(config, 4)


def test_completed_run_defers_to_existing_completion_and_retention_validation(tmp_path):
    config = tiny_config(tmp_path)
    write_json(tmp_path / "completion.json", {"final_step": 4})
    write_json(tmp_path / "metadata.json", {"status": "complete", "step": 4})
    write_json(tmp_path / "metrics.jsonl", {"event": "validation", "step": 4})
    validate_resume_progress(config, 2)
