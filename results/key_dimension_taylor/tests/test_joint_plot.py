import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from results.key_dimension_taylor.plot import DEGREES, WIDTHS, aggregate, grid_rows, summarize_run, summarize_retained_run


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def make_run(root: Path, width=8, degree=2, *, reused=False, complete=True, terminal_ppl=16.0):
    name = f"200m-dk{width:03d}-{'softmax' if degree is None else f'm{degree:03d}'}-seed0"
    workdir = root / "results" / "key_dimension_taylor" / "runs" / name
    workdir.mkdir(parents=True, exist_ok=True)
    run = {
        "name": name, "qk_head_dim": width, "attention_degree": degree, "value_head_dim": 64,
        "parameter_count": 100 + width, "source_commit": "source", "retained_completed": reused,
        "original_experiment": ("key_dimension" if degree is None else "polynomial_attention") if reused else "key_dimension_taylor",
        "workdir": str(workdir), "training_tokens": 10_000_269_312,
        "config": {"train": {"source_commit": "source", "seed": 0}, "optim": {"total_steps": 19_074}},
    }
    records = []
    if complete:
        for step, ppl in ((18_000, 10.0), (19_000, 10.0), (19_074, terminal_ppl)):
            record = {"event": "validation", "step": step, "checkpoint_step": step,
                      "source_commit": "source", "qk_head_dim": width, "parameter_count": 100 + width,
                      "training_tokens": step * 524_288, "validation_nll": math.log(ppl),
                      "validation_perplexity": ppl, "slurm_job_id": "102" if step == 19_074 else "101"}
            if degree is not None:  # Historical key-dimension metrics omit the degree.
                record["attention_degree"] = degree
            records.append(record)
        (workdir / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        terminal = records[-1]
        write_json(workdir / "completion.json", {
            "source_commit": "source", "qk_head_dim": width, "attention_degree": degree,
            "parameter_count": 100 + width, "final_step": 19_074, "training_tokens": run["training_tokens"],
            "validation_nll": terminal["validation_nll"], "validation_perplexity": terminal_ppl,
            "retained_best_checkpoint_step": 19_000, "slurm_job_id": "102",
        })
    if reused:
        run["provenance_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (workdir / "metrics.jsonl", workdir / "completion.json")}
    return run


def make_manifest(root):
    return {"project_root": str(root), "source_commit": "source", "identity": "identity",
            "runs": [make_run(root, width, degree, reused=width == 64, terminal_ppl=15 + width / 100 + degree / 10)
                     for width in reversed(WIDTHS) for degree in reversed(DEGREES)],
            "controls": [make_run(root, width, None, reused=True, terminal_ppl=14 + width / 100)
                         for width in reversed(WIDTHS)]}


def test_placement_reuse_terminal_and_width_specific_delta(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = [summarize_run(run)[0] for run in manifest["runs"] + manifest["controls"]]
    grid = grid_rows(rows)
    assert [line[0]["qk_head_dim"] for line in grid] == list(WIDTHS)
    assert [row["attention_degree"] for row in grid[0]] == [*DEGREES, None]
    assert all(row["reused"] for row in grid[3])
    assert sum(row["reused"] for row in rows) == 9
    assert grid[0][0]["terminal_ppl"] == 15.28
    assert grid[0][0]["best_ppl"] == 10.0
    assert grid[0][0]["best_step"] == 19_000
    for line in grid:
        assert line[0]["delta_ppl_vs_softmax"] == pytest.approx(1.2)
        assert line[4]["delta_ppl_vs_softmax"] == ""


def test_failed_pending_and_incomplete_never_supply_terminal_metrics(tmp_path):
    pending = make_run(tmp_path, 8, 2, complete=False)
    failed = make_run(tmp_path, 8, 4, complete=False)
    write_json(Path(failed["workdir"]) / "numerical_failure_step_00000007.json", {"error": "zero denominator"})
    incomplete = make_run(tmp_path, 8, 8)
    Path(incomplete["workdir"], "completion.json").unlink()
    for run, expected in ((pending, "pending"), (failed, "failed"), (incomplete, "incomplete")):
        row, _ = summarize_run(run)
        assert row["status"] == expected
        assert row["terminal_ppl"] == row["terminal_nll"] == row["terminal_step"] == ""


def test_missing_or_mismatched_terminal_rejected(tmp_path):
    run = make_run(tmp_path)
    path = Path(run["workdir"]) / "completion.json"
    completion = json.loads(path.read_text())
    completion["validation_perplexity"] = 10.0
    write_json(path, completion)
    with pytest.raises(ValueError, match="terminal validation"):
        summarize_run(run)


def test_historical_checksum_and_original_tied_checkpoint(tmp_path):
    run = make_run(tmp_path, 64, 2, reused=True)
    run["selected_checkpoint"] = {"step": 18_000}
    assert summarize_run(run)[0]["best_step"] == 18_000
    path = Path(run["workdir"]) / "metrics.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        summarize_run(run)


def test_duplicate_or_missing_grid_coordinate_rejected(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = [summarize_run(run)[0] for run in manifest["runs"] + manifest["controls"]]
    with pytest.raises(ValueError, match="Duplicate"):
        grid_rows(rows + [rows[0]])
    with pytest.raises(ValueError, match="incomplete"):
        grid_rows(rows[:-1])


def test_aggregate_exports_annotated_grid_curves_and_terminal_csv(tmp_path):
    manifest = make_manifest(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    authored = "# Joint experiment\n\nAuthored method and reproduction command.\n"
    (output / "README.md").write_text(authored)
    rows = aggregate(manifest, output)
    assert (output / "README.md").read_text().startswith(authored)
    assert len(rows) == 25
    for name in ("terminal_ppl_heatmap", "delta_ppl_heatmap", "learning_curves"):
        for suffix in ("png", "pdf"):
            assert (output / f"{name}.{suffix}").stat().st_size > 1000
    csv_rows = list(csv.DictReader((output / "summary.csv").open()))
    assert csv_rows[0]["qk_head_dim"] == "8"
    assert csv_rows[4]["kernel"] == "softmax"
    assert csv_rows[0]["terminal_ppl"] != csv_rows[0]["best_ppl"]
    report = (output / "README.md").read_text()
    assert "| dK | m=2 | m=4 | m=8 | m=16 | Softmax |" in report
    assert f"{rows[0]['terminal_ppl']:.4f}" in report
    assert "subtracts the softmax endpoint at the same width" in report


def test_partial_grid_marks_untrained_cells_and_keeps_outside_runs(tmp_path):
    import copy
    manifest = make_manifest(tmp_path)
    missing = manifest["runs"].pop(0)
    extra = copy.deepcopy(manifest["controls"][0])
    extra.update(name="duplicate-softmax", grid_cell=False)
    manifest["runs"].append(extra)
    output = tmp_path / "partial"
    output.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(manifest, output)
    manifest["grid_coverage"] = "partial"
    rows = aggregate(manifest, output)
    assert missing["name"] not in {row["run"] for row in rows}
    assert "duplicate-softmax" in {row["run"] for row in rows}
    csv_rows = {row["run"]: row for row in csv.DictReader((output / "summary.csv").open())}
    assert csv_rows["duplicate-softmax"]["grid_cell"] == "False"
    report = (output / "README.md").read_text()
    assert "| — |" in report
    assert "Runs retained outside the grid: `duplicate-softmax`" in report


def test_focused_heatmap_retains_supplemental_historical_runs(tmp_path):
    import copy

    manifest = make_manifest(tmp_path)
    manifest.update(key_dimensions=list(WIDTHS), degrees=list(DEGREES), grid_coverage="full")
    supplemental = [make_run(tmp_path, width, None, reused=True, terminal_ppl=22 + width / 100)
                    for width in (2, 4, 256)]
    supplemental.append(make_run(tmp_path, 64, 0, reused=True, terminal_ppl=47.0))
    duplicate = copy.deepcopy(manifest["controls"][3])
    duplicate.update(name="alternate-softmax-64", grid_cell=False)
    supplemental.append(duplicate)
    for run in supplemental:
        run["grid_cell"] = False
    manifest["runs"].extend(supplemental)

    output = tmp_path / "focused"
    output.mkdir()
    rows = aggregate(manifest, output)
    assert len(rows) == 30
    csv_rows = list(csv.DictReader((output / "summary.csv").open()))
    assert sum(row["grid_cell"] == "True" for row in csv_rows) == 25
    assert sum(row["grid_cell"] == "False" for row in csv_rows) == 5
    by_name = {row["run"]: row for row in csv_rows}
    degree_zero = by_name["200m-dk064-m000-seed0"]
    reference = by_name["200m-dk064-softmax-seed0"]
    assert float(degree_zero["delta_ppl_vs_softmax"]) == pytest.approx(
        float(degree_zero["terminal_ppl"]) - float(reference["terminal_ppl"]))
    report = (output / "README.md").read_text()
    assert "25-cell joint comparison" in report
    assert "| dK | m=2 | m=4 | m=8 | m=16 | Softmax |" in report
    assert "| 2 |" not in report and "| 256 |" not in report
    assert "`200m-dk064-m000-seed0`" in report
    assert (output / "terminal_ppl_heatmap.png").exists()


def retained_manifest(root):
    import yaml

    manifest = make_manifest(root)
    manifest.update(format="functional-memory-retained-experiment", experiment="key_dimension_taylor")
    manifest.pop("project_root")
    manifest["key_dimensions"] = [8]
    manifest["degrees"] = [2, 4, 8, 16]

    def reference(path):
        return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    for category in ("runs", "controls"):
        manifest[category] = [run for run in manifest[category] if run["qk_head_dim"] == 8]
        for run in manifest[category]:
            workdir = Path(run.pop("workdir"))
            config = run.pop("config")
            config["model"] = {key: run[key] for key in ("qk_head_dim", "value_head_dim", "attention_degree")}
            config["data"] = {"batch_size": 256, "sequence_length": 2049}
            config["train"]["workdir"] = "/historical/removed/worktree/output"
            config_path = root / "experiments/key_dimension_taylor/configs" / f"{run['name']}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config))
            write_json(workdir / "metadata.json", {"status": "complete"})
            records = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines()]
            run.update(seed=0, resolved_config=reference(config_path), metrics=reference(workdir / "metrics.jsonl"),
                       completion=reference(workdir / "completion.json"), metadata=reference(workdir / "metadata.json"),
                       terminal={key: records[-1][key] for key in ("step", "training_tokens", "validation_nll", "validation_perplexity")},
                       selected_checkpoint={"path": f"experiments/key_dimension_taylor/checkpoints/{run['name']}/18000", "step": 18000,
                                            "validation_nll": math.log(10.0), "validation_perplexity": 10.0})
            run.pop("provenance_sha256", None)
    path = root / "experiments/key_dimension_taylor/sweep.yaml"
    write_json(path, manifest)
    return path, manifest


def test_retained_manifest_uses_portable_paths_preserves_readme_and_selection(tmp_path):
    path, manifest = retained_manifest(tmp_path)
    output = tmp_path / "rendered"
    output.mkdir()
    authored = "# Authored question\n\nExact reproduction instructions.\n"
    (output / "README.md").write_text(authored + "<!-- BEGIN GENERATED RESULTS -->\nold\n<!-- END GENERATED RESULTS -->\n\nAuthored limitations.\n")
    rows = aggregate(manifest, output, manifest_path=path)
    assert len(rows) == 5
    assert all(row["best_step"] == 18000 and row["terminal_step"] == 19074 for row in rows)
    report = (output / "README.md").read_text()
    assert report.startswith(authored) and report.endswith("Authored limitations.\n")
    assert "experiments/key_dimension_taylor/sweep.yaml" in report
    assert "heatmaps do not mark it" in report
    assert len(list(output.rglob("README.md"))) == 1
    assert not list(output.rglob("REPORT.md"))
    assert (output / "terminal_ppl_heatmap.png").exists()


def test_retained_manifest_rejects_changed_bytes_and_wrong_best(tmp_path):
    _, manifest = retained_manifest(tmp_path)
    run = manifest["runs"][0]
    run["selected_checkpoint"]["step"] = 19074
    with pytest.raises(ValueError, match="validation-best"):
        summarize_retained_run(run, tmp_path)
    path = tmp_path / run["metrics"]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        summarize_retained_run(run, tmp_path)


def test_subset_grid_and_unavailable_historical_artifacts(tmp_path):
    manifest = make_manifest(tmp_path)
    manifest.update(key_dimensions=[8], degrees=[4, 2])
    manifest["runs"] = [run for run in manifest["runs"] if run["qk_head_dim"] == 8 and run["attention_degree"] in (2, 4)]
    manifest["controls"] = [run for run in manifest["controls"] if run["qk_head_dim"] == 8]
    manifest["controls"][0]["unavailable_historical_artifacts"] = [{"kind": "slurm_log", "path": "/old/cluster/slurm-7.out", "sha256": "original-checksum"}]
    failed = manifest["runs"][0]
    Path(failed["workdir"], "completion.json").unlink()
    write_json(Path(failed["workdir"]) / "preflight_failure.json", {"error": "nonfinite pilot", "reason": "numerical"})
    output = tmp_path / "subset-report"
    rows = aggregate(manifest, output)
    assert len(rows) == 3
    assert [row["attention_degree"] for row in rows] == [2, 4, None]
    assert rows[1]["status"] == "failed"
    assert rows[1]["terminal_ppl"] == rows[1]["delta_ppl_vs_softmax"] == ""
    report = (output / "README.md").read_text()
    assert "have not been locally revalidated" in report
    assert "original-checksum" in report
    assert "/old/cluster/slurm-7.out" in report


@pytest.mark.parametrize("kind,expected", [("numerical", "failed"), ("infrastructure", "incomplete"), ("validation", "incomplete")])
def test_production_failure_classification_and_completion_resolution(tmp_path, kind, expected):
    run = make_run(tmp_path)
    workdir = Path(run["workdir"])
    completion = json.loads((workdir / "completion.json").read_text())
    (workdir / "completion.json").unlink()
    write_json(workdir / "metadata.json", {"status": "paused"})
    write_json(workdir / "production_failure.json", {"failure_kind": kind, "error": "failure detail"})
    row, _ = summarize_run(run)
    assert row["status"] == expected
    assert row["failure_kind"] == kind
    assert row["error"] == "failure detail"
    if kind != "numerical":
        assert row["execution_status"] == "interrupted"
    assert row["terminal_ppl"] == ""
    write_json(workdir / "completion.json", completion)
    row, _ = summarize_run(run)
    assert row["status"] == "complete"
    assert row["failure_kind"] == row["error"] == ""
