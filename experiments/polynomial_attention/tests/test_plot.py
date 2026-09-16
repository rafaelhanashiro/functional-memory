import json

import pytest

from experiments.polynomial_attention.plot import aggregate, summarize_run
from experiments.polynomial_attention.sweep import run_name


def fixture_run(tmp_path, degree=16):
    workdir = tmp_path / run_name(degree)
    workdir.mkdir()
    run = {
        "name": run_name(degree), "attention_degree": degree, "qk_head_dim": 64,
        "value_head_dim": 64, "parameter_count": 199906304,
        "workdir": str(workdir), "training_tokens": 1000,
        "config": {"optim": {"total_steps": 10}},
    }
    records = [
        {
            "event": "validation", "event_id": f"validation:{step}", "step": step,
            "checkpoint_step": step, "training_tokens": step * 100,
            "source_commit": "a" * 40, "qk_head_dim": 64, "attention_degree": degree,
            "parameter_count": 199906304, "validation_nll": nll,
            "validation_perplexity": ppl,
        }
        for step, nll, ppl in [(5, 2.0, 7.39), (10, 2.1, 8.17)]
    ]
    (workdir / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return run, workdir


def write_completion(run, workdir):
    (workdir / "completion.json").write_text(json.dumps({
        "final_step": 10, "training_tokens": 1000, "attention_degree": run["attention_degree"],
        "parameter_count": run["parameter_count"], "source_commit": "a" * 40,
    }))


def test_terminal_endpoint_requires_completion_and_uses_terminal_not_best(tmp_path):
    run, workdir = fixture_run(tmp_path)
    row, _ = summarize_run(run, "a" * 40)
    assert row["status"] == "incomplete" and row["terminal_ppl"] == ""
    write_completion(run, workdir)
    row, _ = summarize_run(run, "a" * 40)
    assert row["status"] == "complete"
    assert row["terminal_ppl"] == 8.17 and row["best_ppl"] == 7.39


def test_completion_without_committed_terminal_validation_is_rejected(tmp_path):
    run, workdir = fixture_run(tmp_path)
    (workdir / "metrics.jsonl").write_text((workdir / "metrics.jsonl").read_text().splitlines()[0] + "\n")
    write_completion(run, workdir)
    (workdir / "pending_validation.json").write_text('{"step": 10}')
    with pytest.raises(ValueError, match="committed terminal validation"):
        summarize_run(run, "a" * 40)


def test_wrong_source_and_degree_are_rejected(tmp_path):
    run, _ = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        summarize_run(run, "b" * 40)
    run["attention_degree"] = 8
    with pytest.raises(ValueError, match="identity"):
        summarize_run(run, "a" * 40)


def test_failure_status_is_preserved_and_excluded_from_terminal_points(tmp_path):
    run, workdir = fixture_run(tmp_path)
    (workdir / "metadata.json").write_text(json.dumps({"status": "failed", "error": "Invalid Taylor normalizer"}))
    row, _ = summarize_run(run, "a" * 40)
    assert row["status"] == "failed" and row["error"] == "Invalid Taylor normalizer"
    assert row["terminal_ppl"] == ""


def test_aggregate_exports_completed_zero_degree_and_softmax_reference(tmp_path, monkeypatch):
    import matplotlib.axes
    zero, zero_dir = fixture_run(tmp_path, 0)
    exact, exact_dir = fixture_run(tmp_path, None)
    incomplete, _ = fixture_run(tmp_path, 16)
    write_completion(zero, zero_dir)
    write_completion(exact, exact_dir)
    manifest = {"source_commit": "a" * 40, "identity": "test", "project_root": str(tmp_path), "runs": [zero, incomplete, exact]}
    output = tmp_path / "plots"
    reference_lines = []
    original = matplotlib.axes.Axes.axhline

    def inspect_line(self, y=0, *args, **kwargs):
        reference_lines.append((y, kwargs.get("label")))
        return original(self, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", inspect_line)
    rows = aggregate(manifest, output)
    assert rows[0]["terminal_ppl"] == 8.17 and rows[1]["terminal_ppl"] == "" and rows[2]["terminal_ppl"] == 8.17
    assert reference_lines == [(8.17, "Exact softmax")]
    for filename in ("summary.csv", "README.md", "ppl_vs_degree.pdf", "ppl_vs_degree.png", "learningcurves.pdf", "learningcurves.png"):
        assert (output / filename).stat().st_size > 0
    assert "m=0" in (output / "README.md").read_text()
