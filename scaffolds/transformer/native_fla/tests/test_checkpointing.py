import copy
import json
import random

import numpy as np
import pytest
import torch

from scaffolds.transformer.native_fla.checkpointing import (
    checkpoint_directories, latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint,
)
from scaffolds.transformer.native_fla.retention import retain_checkpoints


def _model_and_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Dropout(0.25), torch.nn.Linear(7, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=0.002)


def _update(model, optimizer, cursor):
    # All three CPU random generators affect the next update. Dropout also uses
    # the Torch RNG, so merely restoring the batch would not make this pass.
    inputs = torch.randn(3, 4) + np.random.normal() + random.random() + cursor / 100
    targets = torch.randn(3, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    return cursor + len(inputs), loss.detach().clone()


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


def _save(root, step, model, optimizer, *, best=None):
    return save_checkpoint(root / str(step), model=model, optimizer=optimizer, step=step,
                           data_cursor=3 * step, best_validation=best, config_sha256="recipe-hash")


def test_resume_restores_next_update_model_moments_rng_and_cursor(tmp_path):
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    model, optimizer = _model_and_optimizer()
    cursor, _ = _update(model, optimizer, 0)
    best = {"step": 1, "validation_nll": 2.5}
    checkpoint = save_checkpoint(tmp_path / "1", model=model, optimizer=optimizer, step=1,
                                 data_cursor=cursor, best_validation=best, config_sha256="recipe-hash",
                                 extra={"training_tokens": 6144})
    expected_cursor, expected_loss = _update(model, optimizer, cursor)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    # Construction consumes RNG and optimizer moments start empty. Resume must
    # restore both, rather than accidentally relying on the live original state.
    resumed_model, resumed_optimizer = _model_and_optimizer()
    progress = load_checkpoint(checkpoint, model=resumed_model, optimizer=resumed_optimizer,
                               expected_config_sha256="recipe-hash")
    assert progress == {"step": 1, "data_cursor": cursor, "best_validation": best,
                        "config_sha256": "recipe-hash", "extra": {"training_tokens": 6144}}
    actual_cursor, actual_loss = _update(resumed_model, resumed_optimizer, progress["data_cursor"])
    assert actual_cursor == expected_cursor
    assert torch.equal(actual_loss, expected_loss)
    _assert_equal(resumed_model.state_dict(), expected_model)
    _assert_equal(resumed_optimizer.state_dict(), expected_optimizer)


def test_interrupted_save_never_publishes_or_replaces_a_checkpoint(tmp_path, monkeypatch):
    model, optimizer = _model_and_optimizer()
    first = _save(tmp_path, 1, model, optimizer)
    original = (first / "state.pt").read_bytes()

    def fail_save(state, stream):
        stream.write(b"partial checkpoint")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        _save(tmp_path, 2, model, optimizer)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["1"]
    assert latest_checkpoint(tmp_path) == first
    assert (first / "state.pt").read_bytes() == original
    with pytest.raises(FileExistsError):
        _save(tmp_path, 1, model, optimizer)


def test_resume_rejects_wrong_recipe_and_truncated_published_state(tmp_path):
    model, optimizer = _model_and_optimizer()
    checkpoint = _save(tmp_path, 1, model, optimizer)
    with pytest.raises(ValueError, match="configuration hash"):
        load_checkpoint(checkpoint, model=model, expected_config_sha256="different-recipe")
    state_file = checkpoint / "state.pt"
    with state_file.open("r+b") as stream:
        stream.truncate(state_file.stat().st_size - 1)
    with pytest.raises(ValueError, match="Invalid native FLA checkpoint"):
        latest_checkpoint(tmp_path)


def test_retention_preserves_latest_and_best_then_only_best(tmp_path):
    model, optimizer = _model_and_optimizer()
    for step in (1, 2, 3):
        _save(tmp_path, step, model, optimizer)
    assert retain_checkpoints(tmp_path, best_step=1) == [tmp_path / "2"]
    assert checkpoint_directories(tmp_path) == [tmp_path / "1", tmp_path / "3"]
    assert retain_checkpoints(tmp_path, best_step=1, completed=True) == [tmp_path / "3"]
    assert latest_checkpoint(tmp_path) == tmp_path / "1"
    with pytest.raises(ValueError, match="retained steps are absent"):
        prune_checkpoints(tmp_path, {7})
    assert latest_checkpoint(tmp_path) == tmp_path / "1"
    with pytest.raises(ValueError, match="validation-best"):
        retain_checkpoints(tmp_path, best_step=None, completed=True)


def test_latest_ignores_unpublished_staging_but_rejects_corrupt_metadata(tmp_path):
    model, optimizer = _model_and_optimizer()
    checkpoint = _save(tmp_path, 1, model, optimizer)
    (tmp_path / ".2.interrupted").mkdir()
    assert latest_checkpoint(tmp_path) == checkpoint
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["step"] = 2
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Invalid native FLA checkpoint"):
        latest_checkpoint(tmp_path)
