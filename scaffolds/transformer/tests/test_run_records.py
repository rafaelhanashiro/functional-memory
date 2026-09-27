"""Canonical recipes stay immutable while execution and resume records advance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop


ROOT = Path(__file__).resolve().parents[3]


def allocated_run(tmp_path):
    from scaffolds.records import create_run

    study = tmp_path / "experiments/demo"
    study.mkdir(parents=True)
    # The JSON subset is valid YAML and keeps the records helpers stdlib-only.
    (study / "sweep.yaml").write_text(json.dumps({"study_id": "demo", "run_ids": []}))
    recipe = config_to_dict(tiny_config(tmp_path / "placeholder"))
    recipe["train"]["source_commit"] = "a" * 40
    return create_run(study, recipe, label="tiny")


def worker(run, *, stop=None):
    command = [sys.executable, "-c", """
import json, sys
from pathlib import Path
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.training.loop import run_training
root = Path(sys.argv[1])
result = run_training(load_config(root / 'config.yaml'),
                      stop_after_step=int(sys.argv[2]) if sys.argv[2] != '-' else None,
                      audit_checkpoint=True)
(root / ('audit-' + sys.argv[2] + '.json')).write_text(json.dumps(result))
""", str(run), str(stop) if stop is not None else "-"]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
    env.pop("SHARED_TRANSFORMER_SOURCE_COMMIT", None)
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((run / f"audit-{stop if stop is not None else '-'}.json").read_text())


def test_fresh_process_resume_keeps_run_id_recipe_and_exact_state(tmp_path):
    full = allocated_run(tmp_path / "full")
    restarted = allocated_run(tmp_path / "restart")
    recipe_bytes = (restarted / "config.yaml").read_bytes()
    baseline = worker(full)
    paused = worker(restarted, stop=2)
    record_before = json.loads((restarted / "record.json").read_text())
    assert record_before["status"] == "paused"
    resumed = worker(restarted)
    record = json.loads((restarted / "record.json").read_text())
    assert record["run_id"] == record_before["run_id"] == restarted.name
    assert record["status"] == "complete"
    assert (restarted / "config.yaml").read_bytes() == recipe_bytes
    assert record["config"]["sha256"] == hashlib.sha256(recipe_bytes).hexdigest()
    assert record["execution"]["resolved_config"]["model"]["attention_backend"] == "blocked"
    assert (restarted / "checkpoints/4").is_dir()
    previous, after = paused["checkpoint_audit"], resumed["checkpoint_audit"]
    assert previous["final_state_sha256"] == after["start_state_sha256"]
    assert previous["next_update"] == after["first_update"]
    assert after["final_state_sha256"] == baseline["checkpoint_audit"]["final_state_sha256"]


def test_allocated_recipe_rejects_scientific_override_and_changed_bytes(tmp_path):
    run = allocated_run(tmp_path)
    cfg = load_config(run / "config.yaml")
    cfg.model.qk_head_dim = 4
    with pytest.raises(ValueError, match="does not match the allocated run"):
        loop.ensure_run_config(cfg, resume=True)
    assert not (run / "config.json").exists()
    with (run / "config.yaml").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="bytes changed after allocation"):
        loop.ensure_run_config(load_config(run / "config.yaml"), resume=True)


def test_fresh_execution_requires_new_run_id(tmp_path):
    run = allocated_run(tmp_path)
    cfg = load_config(run / "config.yaml")
    loop.ensure_run_config(cfg, resume=False)
    with pytest.raises(ValueError, match="already initialized"):
        loop.ensure_run_config(cfg, resume=False)


def test_production_requires_preallocated_recipe(tmp_path, monkeypatch):
    root = tmp_path / "experiments/demo/runs/r0001"
    root.mkdir(parents=True)
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", "f" * 40)
    with pytest.raises(ValueError, match="preallocated, committed"):
        loop.ensure_run_config(tiny_config(root), resume=True)
    assert not (root / "config.yaml").exists()


def test_finalization_updates_authoritative_record_and_hashes(tmp_path, monkeypatch):
    from scaffolds.records import validate_record
    from scaffolds.transformer.retention import finalize_run

    run = allocated_run(tmp_path)
    cfg = load_config(run / "config.yaml")
    monkeypatch.setattr(loop, "evaluate", lambda state, *args, **kwargs: float(state.step))
    loop.run_training(cfg)
    finalize_run(cfg)
    record = json.loads((run / "record.json").read_text())
    assert record["retained_best_checkpoint_step"] == 2
    assert record["retained_terminal_checkpoint_step"] is None
    assert record["selected_checkpoint"]["step"] == 2
    assert record["selected_checkpoint"]["file_count"] > 0
    assert record["completion"]["sha256"] == hashlib.sha256((run / "completion.json").read_bytes()).hexdigest()
    assert record["metrics"]["sha256"] == hashlib.sha256((run / "metrics.jsonl").read_bytes()).hexdigest()
    assert not (run / "checkpoints/4").exists()
    assert "metadata.json" in validate_record(run)["verified"]
