import hashlib
import json

import pytest
import torch

from scaffolds.transformer.native_fla import acceptance
from scaffolds.transformer.native_fla.checkpointing import save_checkpoint
from scaffolds.transformer.native_fla.config import parameter_count, reference_recipe


def _checkpoints(tmp_path):
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    paths = [tmp_path / name / "3" for name in ("reference", "resumed")]
    for index, path in enumerate(paths):
        save_checkpoint(path, model=model, optimizer=optimizer, step=3, data_cursor=768,
                        best_validation={"step": 2 + index, "validation_nll": 3.0}, config_sha256="same-recipe")
    return paths


def _rewrite(path, state):
    torch.save(state, path / "state.pt")
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["state_size_bytes"] = (path / "state.pt").stat().st_size
    (path / "metadata.json").write_text(json.dumps(metadata))


def test_resume_audit_checks_state_and_allows_different_pilot_best_selection(tmp_path):
    first, second = _checkpoints(tmp_path)
    report = acceptance.compare_checkpoint_states(first, second)
    assert report["exact"] is True
    assert report["tensor_count"] >= 9
    assert report["step"] == 3
    assert report["data_cursor"] == 768


@pytest.mark.parametrize("corruption", ["model", "optimizer", "rng", "data_cursor"])
def test_resume_audit_detects_hidden_state_corruption(tmp_path, corruption):
    first, second = _checkpoints(tmp_path)
    state = torch.load(second / "state.pt", map_location="cpu", weights_only=False)
    if corruption == "model":
        state["model"]["weight"][0, 0] += 0.01
    elif corruption == "optimizer":
        next(iter(state["optimizer"]["state"].values()))["exp_avg"][0, 0] += 0.01
    elif corruption == "rng":
        state["rng"]["torch"][0] ^= 1
    else:
        state["data_cursor"] += 1
    _rewrite(second, state)
    report = acceptance.compare_checkpoint_states(first, second)
    assert report["exact"] is False
    assert report["difference_count"] > 0
    assert any(item["path"].startswith(corruption) for item in report["differences"])


def _summary(config):
    return {"final_step": 3, "data_cursor": 768, "parameter_count": parameter_count(config),
            "training_tokens": 3 * 256 * 2048, "validation_nll": 3.0, "validation_perplexity": 20.0,
            "last_train": {"train_loss": 3.1, "grad_norm": 0.7, "learning_rate": 3e-4,
                           "train_step_seconds": 2.0}}


def test_metrics_audit_requires_full_recipe_and_ignores_wall_clock():
    config = reference_recipe()
    first, second = _summary(config), _summary(config)
    second["last_train"]["train_step_seconds"] = 9.0
    assert acceptance.compare_pilot_summaries(first, second, config)["exact"]
    second["parameter_count"] -= 1
    second["last_train"]["train_loss"] += 1e-6
    report = acceptance.compare_pilot_summaries(first, second, config)
    assert not report["exact"]
    assert "resumed.parameter_count" in report["differences"]
    assert "last_train.train_loss" in report["differences"]


def test_matching_acceptance_identity_reuses_without_running_workers(tmp_path, monkeypatch):
    identity = {"config_sha256": "recipe", "launch_commit": "source", "gpu_type": "B300"}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = {"status": "passed", "identity_sha256": identity_hash, "resume": {"exact": True}}
    (tmp_path / "success.json").write_text(json.dumps(previous))
    monkeypatch.setattr(acceptance, "acceptance_identity", lambda path: identity)
    monkeypatch.setattr(acceptance, "_run_logged", lambda *args, **kwargs: pytest.fail("reused acceptance ran a worker"))
    assert acceptance.run_acceptance(tmp_path / "config.yaml", tmp_path) == previous
