"""Common best-checkpoint selection and archival behavior across mechanisms."""
import hashlib
import json
import shutil

import pytest

from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop


def test_explicit_checkpoint_storage_roundtrips_and_resumes(tmp_path):
    from scaffolds.transformer.configs import load_config, save_config
    from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, state_sha256
    from scaffolds.transformer.training.state import create_state
    from scaffolds.transformer.training.checkpointing import restore_latest_checkpoint
    import jax

    config = tiny_config(tmp_path / "results/example/runs/run")
    config.train.checkpoint_dir = tmp_path / "checkpoints/example/run"
    config_path = tmp_path / "results/example/configs/run.yaml"
    save_config(config, config_path)
    restored_config = load_config(config_path)
    assert restored_config.train.checkpoint_dir == config.train.checkpoint_dir
    loop.run_training(restored_config, stop_after_step=2)
    assert (config.train.checkpoint_dir / "2").is_dir()
    assert not (config.train.workdir / "checkpoints").exists()
    resumed = loop.run_training(restored_config)
    assert resumed["restored_step"] == 2
    assert resumed["final_step"] == 4

    uninterrupted = tiny_config(tmp_path / "reference")
    loop.run_training(uninterrupted)
    states = []
    for cfg in (restored_config, uninterrupted):
        with make_checkpoint_manager(cfg) as manager:
            state, step = restore_latest_checkpoint(manager, create_state(cfg, jax.random.PRNGKey(0)))
            assert step == 4
            states.append(state_sha256(state))
    assert states[0] == states[1]


@pytest.mark.parametrize("degree", [None, 0, 2, 3])
def test_best_metrics_match_orbax_on_exact_validation_ties(tmp_path, monkeypatch, degree):
    config = tiny_config(tmp_path)
    config.model.attention_degree = degree
    monkeypatch.setattr(loop, "evaluate", lambda *args, **kwargs: 3.0)
    outcome = loop.run_training(config)
    best = json.loads((tmp_path / "best_validation.json").read_text())
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert outcome["retained_best_checkpoint_step"] == best["step"] == 4
    assert records[-1]["is_best"] is True


def test_joint_best_only_archive_resumes_without_touching_terminal_records(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config.model.attention_degree = 2
    monkeypatch.setattr(loop, "evaluate", lambda state, *args, **kwargs: float(state.step))
    outcome = loop.run_training(config)
    assert outcome["retained_best_checkpoint_step"] == 2
    completion = (tmp_path / "completion.json").read_bytes()
    metrics = (tmp_path / "metrics.jsonl").read_bytes()
    shutil.rmtree(tmp_path / "checkpoints/4")
    (tmp_path / "retention.json").write_text(json.dumps({
        "format": "key-dimension-taylor-checkpoint-retention",
        "completion_sha256": hashlib.sha256(completion).hexdigest(), "best_checkpoint_step": 2,
    }))
    assert loop.run_training(config)["final_step"] == 4
    assert (tmp_path / "completion.json").read_bytes() == completion
    assert (tmp_path / "metrics.jsonl").read_bytes() == metrics
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["retained_terminal_checkpoint_step"] is None
