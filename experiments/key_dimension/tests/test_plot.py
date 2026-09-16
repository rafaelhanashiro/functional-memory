import json

import pytest

from experiments.key_dimension.plot import aggregate, summarize_run


def fixture_run(tmp_path, dim=64):
    workdir = tmp_path / f"dk{dim}"
    workdir.mkdir()
    run = {
        "name": f"200m-dk{dim:03d}-seed0", "qk_head_dim": dim,
        "value_head_dim": 64, "parameter_count": 199906304,
        "workdir": str(workdir), "training_tokens": 1000,
        "config": {"optim": {"total_steps": 10}},
    }
    records = [
        {
            "event": "validation", "event_id": f"validation:{step}", "step": step,
            "checkpoint_step": step, "training_tokens": step * 100,
            "source_commit": "a" * 40, "qk_head_dim": dim,
            "parameter_count": 199906304, "validation_nll": nll,
            "validation_perplexity": ppl,
        }
        for step, nll, ppl in [(5, 2.0, 7.39), (10, 2.1, 8.17)]
    ]
    (workdir / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return run, workdir


def test_terminal_endpoint_requires_completion_and_uses_terminal_not_best(tmp_path):
    run, workdir = fixture_run(tmp_path)
    row, _ = summarize_run(run, "a" * 40)
    assert row["status"] == "incomplete" and row["terminal_ppl"] == ""
    (workdir / "completion.json").write_text(json.dumps({"final_step": 10, "training_tokens": 1000}))
    row, _ = summarize_run(run, "a" * 40)
    assert row["status"] == "complete"
    assert row["terminal_ppl"] == 8.17 and row["best_ppl"] == 7.39


def test_completion_without_committed_terminal_validation_is_rejected(tmp_path):
    run, workdir = fixture_run(tmp_path)
    (workdir / "metrics.jsonl").write_text((workdir / "metrics.jsonl").read_text().splitlines()[0] + "\n")
    (workdir / "completion.json").write_text(json.dumps({"final_step": 10, "training_tokens": 1000}))
    (workdir / "pending_validation.json").write_text('{"step": 10}')
    with pytest.raises(ValueError, match="committed terminal validation"):
        summarize_run(run, "a" * 40)


def test_wrong_source_is_rejected(tmp_path):
    run, _ = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        summarize_run(run, "b" * 40)


def test_aggregate_exports_only_completed_terminal_points(tmp_path):
    completed, workdir = fixture_run(tmp_path, 64)
    incomplete, _ = fixture_run(tmp_path, 2)
    (workdir / "completion.json").write_text(json.dumps({"final_step": 10, "training_tokens": 1000}))
    manifest = {"source_commit": "a" * 40, "identity": "test", "project_root": str(tmp_path), "runs": [incomplete, completed]}
    output = tmp_path / "plots"
    rows = aggregate(manifest, output)
    assert rows[0]["terminal_ppl"] == "" and rows[1]["terminal_ppl"] == 8.17
    for filename in ("summary.csv", "README.md", "ppl_vs_dk.pdf", "ppl_vs_dk.png", "learningcurves.pdf", "learningcurves.png"):
        assert (output / filename).stat().st_size > 0
