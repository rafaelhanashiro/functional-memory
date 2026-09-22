import json
from pathlib import Path

import pytest

from experiments.key_dimension_taylor.retention import finalize_run
from experiments.key_dimension_taylor.tests.test_joint_plot import make_run, write_json


def retention_fixture(root):
    run = make_run(root)
    workdir = Path(run["workdir"])
    checkpoint_root = root / "experiments" / "key_dimension_taylor" / "checkpoints" / run["name"]
    checkpoint_root.mkdir(parents=True)
    for step in (18_000, 19_000, 19_074):
        (checkpoint_root / str(step)).mkdir()
    (workdir / "checkpoints").symlink_to(checkpoint_root)
    run["checkpoint_dir"] = str(checkpoint_root)
    records = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines()]
    write_json(workdir / "best_validation.json", records[1])
    write_json(workdir / "metadata.json", {"slurm_job_id": "102", "status": "complete"})
    write_json(workdir / "launch.json", {"segments": [{"slurm_job_id": "101"}, {"slurm_job_id": "102"}]})
    logs = root / "results" / "logs" / "key_dimension_taylor" / run["name"]
    logs.mkdir(parents=True)
    for job in ("100", "101", "102"):
        (logs / f"slurm-{job}.out").write_text(f"job {job}\n")
    return {"project_root": str(root), "source_commit": "source", "runs": [run]}, run, logs


def test_retains_latest_tied_best_and_final_log_preserving_terminal_and_segment_history(tmp_path):
    manifest, run, logs = retention_fixture(tmp_path)
    workdir = Path(run["workdir"])
    completion_bytes = (workdir / "completion.json").read_bytes()
    metrics_bytes = (workdir / "metrics.jsonl").read_bytes()
    record = finalize_run(manifest, run["name"])
    assert record["format"] == "key-dimension-taylor-checkpoint-retention"
    assert record["best_checkpoint_step"] == 19_000
    assert record["terminal_step"] == 19_074
    assert record["terminal_perplexity"] == 16.0
    assert [path.name for path in Path(run["checkpoint_dir"]).iterdir()] == ["19000"]
    assert [path.name for path in logs.iterdir()] == ["slurm-102.out"]
    assert record["segment_job_ids"] == ["101", "102"]
    assert len(record["launch_history"]["segments"]) == 2
    assert len(record["log_history"]) == 3
    assert all(entry.get("sha256") for entry in record["log_history"] if not entry["retained"])
    assert (workdir / "completion.json").read_bytes() == completion_bytes
    assert (workdir / "metrics.jsonl").read_bytes() == metrics_bytes
    metadata = json.loads((workdir / "metadata.json").read_text())
    assert metadata["retained_terminal_checkpoint_step"] is None
    assert metadata["slurm_job_id"] == "102"
    assert finalize_run(manifest, run["name"]) == record


def test_missing_selected_log_prevents_any_pruning(tmp_path):
    manifest, run, logs = retention_fixture(tmp_path)
    (logs / "slurm-102.out").unlink()
    with pytest.raises(ValueError, match="matching Slurm log"):
        finalize_run(manifest, run["name"])
    assert len(list(Path(run["checkpoint_dir"]).iterdir())) == 3


def test_wrong_best_selection_prevents_any_pruning(tmp_path):
    manifest, run, _ = retention_fixture(tmp_path)
    path = Path(run["workdir"]) / "best_validation.json"
    record = json.loads(path.read_text())
    record["step"] = 18_000
    write_json(path, record)
    with pytest.raises(ValueError, match="best committed"):
        finalize_run(manifest, run["name"])
    assert len(list(Path(run["checkpoint_dir"]).iterdir())) == 3


def test_historical_reference_is_never_pruned(tmp_path):
    manifest, run, _ = retention_fixture(tmp_path)
    run["retained_completed"] = True
    with pytest.raises(ValueError, match="only modify a new"):
        finalize_run(manifest, run["name"])
