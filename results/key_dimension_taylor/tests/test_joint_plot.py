"""Grid placement, endpoints, and exports, from synthetic run directories."""
import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from results.key_dimension_taylor.plot import DEGREES, WIDTHS, aggregate, grid_rows, summarize_run
from scaffolds.transformer.analysis import run_view

TOKENS = 19_074 * 256 * 2048


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def make_run(root: Path, width=8, degree=2, *, complete=True, terminal_ppl=16.0, selected_step=None):
    name = f"200m-dk{width:03d}-{'softmax' if degree is None else f'm{degree:03d}'}-seed0"
    run = root / "experiments" / "key_dimension_taylor" / "runs" / f"r{abs(hash(name)) % 16**12:012x}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump({
        "model": {"qk_head_dim": width, "value_head_dim": 64, "attention_degree": degree},
        "data": {"batch_size": 256, "sequence_length": 2049}, "optim": {"total_steps": 19_074},
        "train": {"seed": 0, "source_commit": "source"}}))
    record = {"run_id": run.name, "label": name, "source_commit": "source"}
    if selected_step is not None:
        record["selected_checkpoint"] = {"step": selected_step, "path": f"runs/{run.name}/checkpoints/{selected_step}"}
    write_json(run / "record.json", record)
    identity = {"source_commit": "source", "qk_head_dim": width, "parameter_count": 100 + width}
    if complete:
        records = []
        for step, ppl in ((18_000, 10.0), (19_000, 10.0), (19_074, terminal_ppl)):
            records.append({"event": "validation", "step": step, "checkpoint_step": step, **identity,
                            "attention_degree": degree, "training_tokens": step * 524_288,
                            "validation_nll": math.log(ppl), "validation_perplexity": ppl})
        (run / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        write_json(run / "completion.json", {**identity, "attention_degree": degree, "final_step": 19_074,
                                             "training_tokens": TOKENS, "validation_nll": records[-1]["validation_nll"],
                                             "validation_perplexity": terminal_ppl})
        write_json(run / "metadata.json", {**identity, "status": "complete"})
    else:
        write_json(run / "metadata.json", {**identity, "status": "running"})
    return run_view(run, name)


def make_study(root):
    runs = [make_run(root, width, degree, terminal_ppl=15 + width / 100 + degree / 10)
            for width in reversed(WIDTHS) for degree in reversed(DEGREES)]
    runs += [make_run(root, width, None, terminal_ppl=14 + width / 100) for width in reversed(WIDTHS)]
    return {"key_dimensions": list(WIDTHS), "degrees": list(DEGREES), "runs": runs}


def test_placement_terminal_and_width_specific_delta(tmp_path):
    grid = grid_rows([summarize_run(run)[0] for run in make_study(tmp_path)["runs"]])
    assert [line[0]["qk_head_dim"] for line in grid] == list(WIDTHS)
    assert [row["attention_degree"] for row in grid[0]] == [*DEGREES, None]
    assert grid[0][0]["terminal_ppl"] == 15.28
    # Latest on ties when no checkpoint is recorded.
    assert (grid[0][0]["best_ppl"], grid[0][0]["best_step"]) == (10.0, 19_000)
    for line in grid:
        assert line[0]["delta_ppl_vs_softmax"] == pytest.approx(1.2)
        assert line[4]["delta_ppl_vs_softmax"] == ""


def test_retained_checkpoint_decides_ties(tmp_path):
    assert summarize_run(make_run(tmp_path, 64, 2, selected_step=18_000))[0]["best_step"] == 18_000


def test_failed_pending_and_incomplete_never_supply_terminal_metrics(tmp_path):
    pending = make_run(tmp_path, 8, 2, complete=False)
    failed_dir = Path(make_run(tmp_path, 8, 4, complete=False)["workdir"])
    write_json(failed_dir / "numerical_failure_step_00000007.json", {"step": 7, "error": "zero denominator"})
    write_json(failed_dir / "metadata.json", {"status": "failed"})
    failed = run_view(failed_dir, "failed")
    incomplete = make_run(tmp_path, 8, 8)
    Path(incomplete["workdir"], "completion.json").unlink()
    incomplete = run_view(Path(incomplete["workdir"]), "incomplete")
    for run, expected in ((pending, "incomplete"), (failed, "failed"), (incomplete, "incomplete")):
        row, _ = summarize_run(run)
        assert row["status"] == expected
        assert row["terminal_ppl"] == row["terminal_nll"] == row["terminal_step"] == ""
    assert summarize_run(failed)[0]["error"] == "zero denominator"


def test_missing_or_mismatched_terminal_rejected(tmp_path):
    run = make_run(tmp_path)
    path = Path(run["workdir"]) / "completion.json"
    completion = json.loads(path.read_text())
    completion["validation_perplexity"] = 10.0
    write_json(path, completion)
    with pytest.raises(ValueError, match="terminal validation"):
        summarize_run(run)


def test_duplicate_or_missing_grid_coordinate_rejected(tmp_path):
    rows = [summarize_run(run)[0] for run in make_study(tmp_path)["runs"]]
    with pytest.raises(ValueError, match="Duplicate"):
        grid_rows(rows + [rows[0]])
    with pytest.raises(ValueError, match="incomplete"):
        grid_rows(rows[:-1])


def test_aggregate_exports_grid_curves_csv_and_outside_runs(tmp_path):
    study = make_study(tmp_path)
    study["runs"] += [make_run(tmp_path, width, None, terminal_ppl=22 + width / 100) for width in (2, 256)]
    study["runs"].append(make_run(tmp_path, 64, 0, terminal_ppl=47.0))
    output = tmp_path / "report"
    output.mkdir()
    authored = "# Joint experiment\n\nAuthored method and reproduction command.\n"
    (output / "README.md").write_text(authored)
    rows = aggregate(study, output)
    assert len(rows) == 28
    for name in ("terminal_ppl_heatmap", "delta_ppl_heatmap", "learning_curves"):
        for suffix in ("png", "pdf"):
            assert (output / f"{name}.{suffix}").stat().st_size > 1000
    csv_rows = {row["run"]: row for row in csv.DictReader((output / "summary.csv").open())}
    assert sum(row["grid_cell"] == "True" for row in csv_rows.values()) == 25
    degree_zero, reference = csv_rows["200m-dk064-m000-seed0"], csv_rows["200m-dk064-softmax-seed0"]
    assert float(degree_zero["delta_ppl_vs_softmax"]) == pytest.approx(
        float(degree_zero["terminal_ppl"]) - float(reference["terminal_ppl"]))
    report = (output / "README.md").read_text()
    assert report.startswith(authored)
    assert "| dK | m=2 | m=4 | m=8 | m=16 | Softmax |" in report
    assert "| 2 |" not in report and "`200m-dk064-m000-seed0`" in report
    # Runs without a retained checkpoint are listed as such.
    assert "Checkpoints and logs are not retained for 28 runs" in report


def test_partial_grid_marks_untrained_cells(tmp_path):
    study = make_study(tmp_path)
    missing = study["runs"].pop(0)
    output = tmp_path / "partial"
    output.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(study, output)
    study["grid_coverage"] = "partial"
    rows = aggregate(study, output)
    assert missing["name"] not in {row["run"] for row in rows}
    assert "| — |" in (output / "README.md").read_text()
