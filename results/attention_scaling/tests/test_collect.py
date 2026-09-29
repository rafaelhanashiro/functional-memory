"""Checks for the combined heatmap's source attribution and missing-cell policy."""

from __future__ import annotations

from scaffolds.records import load_study

import hashlib
import json
import yaml
from pathlib import Path

import pytest

from results.attention_scaling.collect import collect_rows
from scaffolds.records.selections import run_sources


def _write(path: Path, value: dict | list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "sweep.yaml" and isinstance(value, dict) and "runs" in value:
        value = dict(value)
        runs = value.pop("runs")
        value["run_ids"] = []
        for run in runs:
            run = dict(run, run_id=run.get("run_id", run["name"]))
            value["run_ids"].append(run["run_id"])
            directory = path.parent / "runs" / run["run_id"]
            _write(directory / "record.json", run)
            if not (directory / "config.yaml").exists():
                # A recipe without a decay policy or qk_norm is a legacy one.
                _write(directory / "config.yaml", {"model": {"qk_head_dim": run["qk_head_dim"],
                                                             "attention_degree": run["attention_degree"]}})
    if isinstance(value, list):
        payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in value)
    else:
        payload = json.dumps(value, sort_keys=True) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def _pin(root: Path, config: dict) -> dict:
    """The config with its cells' runs pinned to their current scientific files."""
    return dict(config, runs=[{"study": cell["experiment"], "run_id": cell["run"],
                               "sources": run_sources(root, cell["experiment"], cell["run"])}
                              for cell in config["cells"]])


def _fixture(tmp_path: Path) -> dict:
    experiment = "toy_taylor"
    base = tmp_path / "experiments" / experiment
    complete_name = "r000000000001"
    complete_dir = base / "runs" / complete_name
    source = "source-commit-1"
    completed_metadata = {"status": "complete", "seed": 0, "qk_head_dim": 8,
                          "attention_degree": 2, "source_commit": source,
                          "step": 10, "training_tokens": 100}
    early_validation = {"event": "validation", "step": 5, "training_tokens": 50,
                        "validation_nll": 2.0, "validation_perplexity": 7.389,
                        "qk_head_dim": 8, "attention_degree": 2, "source_commit": source}
    last_validation = {"event": "validation", "step": 10, "training_tokens": 100,
                       "validation_nll": 3.0, "validation_perplexity": 20.086,
                       "qk_head_dim": 8, "attention_degree": 2, "source_commit": source}
    completion = {"final_step": 10, "training_tokens": 100, "validation_nll": 3.0,
                  "validation_perplexity": 20.086, "qk_head_dim": 8,
                  "attention_degree": 2, "source_commit": source}
    complete_files = {
        "metadata": ("metadata.json", _write(complete_dir / "metadata.json", completed_metadata)),
        "metrics": ("metrics.jsonl", _write(complete_dir / "metrics.jsonl", [early_validation, last_validation])),
        "completion": ("completion.json", _write(complete_dir / "completion.json", completion)),
    }
    complete_run = {"name": complete_name, "seed": 0, "qk_head_dim": 8,
                    "attention_degree": 2, "source_commit": source,
                    "terminal": {"step": 10, "training_tokens": 100,
                                 "validation_nll": 3.0, "validation_perplexity": 20.086}}
    for kind, (filename, digest) in complete_files.items():
        complete_run[kind] = {"path": f"experiments/{experiment}/runs/{complete_name}/{filename}",
                              "sha256": digest}

    failed_name = "r000000000002"
    failed_dir = base / "runs" / failed_name
    failed_metadata = {"status": "failed", "seed": 0, "qk_head_dim": 16,
                       "attention_degree": 1, "source_commit": source, "step": 7,
                       "target_steps": 10, "target_tokens": 100, "error": "nonfinite loss"}
    failure = {"step": 7, "attention_degree": 1, "error": "nonfinite loss"}
    metadata_sha = _write(failed_dir / "metadata.json", failed_metadata)
    failure_filename = "numerical_failure_step_00000007.json"
    failure_sha = _write(failed_dir / failure_filename, failure)
    failed_run = {"name": failed_name, "status": "numerical_failure", "seed": 0,
                  "qk_head_dim": 16, "attention_degree": 1, "source_commit": source,
                  "metadata": {"path": f"experiments/{experiment}/runs/{failed_name}/metadata.json",
                               "sha256": metadata_sha},
                  "failure": {"path": f"experiments/{experiment}/runs/{failed_name}/{failure_filename}",
                              "sha256": failure_sha, "step": 7}}
    _write(base / "sweep.yaml", {"experiment": experiment, "runs": [complete_run, failed_run]})
    return _pin(tmp_path, {
        "d_k": [8, 16], "terminal_step": 10, "training_tokens": 100, "seed": 0,
        "panels": [{"id": "normalized_fixed", "title": "Taylor",
                    "columns": [{"id": "m1", "label": "m=1"}, {"id": "m2", "label": "m=2"}]}],
        "cells": [{"panel": "normalized_fixed", "column": "m2", "d_k": 8,
                   "experiment": experiment, "run": complete_name},
                  {"panel": "normalized_fixed", "column": "m1", "d_k": 16,
                   "experiment": experiment, "run": failed_name}]})


def test_terminal_validation_and_failed_and_missing_cells(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    rows = collect_rows(tmp_path, config)
    assert len(rows) == 4
    assert [(row["column"], row["d_k"], row["status"]) for row in rows] == [
        ("m1", 8, "not_run"), ("m1", 16, "numerical_failure"),
        ("m2", 8, "complete"), ("m2", 16, "not_run")]
    completed = rows[2]
    assert (completed["terminal_nll"], completed["terminal_ppl"]) == (3.0, 20.086)
    assert set(completed["artifacts"]) == {"metadata", "metrics", "completion"}
    # A recipe without a decay policy or qk_norm predates the current defaults on both counts.
    assert completed["legacy"] is True
    assert completed["legacy_reasons"] == ["weight decay on all parameters", "no QK-norm"]
    assert (completed["weight_decay_policy"], completed["qk_norm"]) == (None, False)
    assert rows[0]["legacy"] is False and rows[0]["legacy_reasons"] == []
    failed = rows[1]
    assert (failed["terminal_nll"], failed["terminal_ppl"], failed["failure_step"]) == (None, None, 7)
    assert set(failed["artifacts"]) == {"metadata", "failure"}
    assert rows[0]["artifacts"] == {}


def test_current_defaults_are_read_from_the_resolved_configuration(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    run_dir = tmp_path / "experiments/toy_taylor/runs/r000000000001"
    resolved = {"optim": {"weight_decay_policy": "exclude_embeddings_norms_biases_v1"},
                "model": {"qk_head_dim": 8, "attention_degree": 2, "qk_norm": True}}
    _write(run_dir / "config.yaml", resolved)
    config = _pin(tmp_path, config)
    row = collect_rows(tmp_path, config)[2]
    assert (row["legacy"], row["legacy_reasons"], row["qk_norm"]) == (False, [], True)
    assert row["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"

    # A column that QK-norm does not apply to ignores it; a plain-score reference column rejects it.
    config["panels"][0]["columns"][1]["qk_norm"] = None
    assert collect_rows(tmp_path, config)[2]["legacy"] is False
    config["panels"][0]["columns"][1]["qk_norm"] = False
    with pytest.raises(ValueError, match="plain-score reference"):
        collect_rows(tmp_path, config)

    resolved["model"]["qk_norm"] = False
    _write(run_dir / "config.yaml", resolved)
    config = _pin(tmp_path, config)
    config["panels"][0]["columns"][1]["qk_norm"] = True
    row = collect_rows(tmp_path, config)[2]
    assert (row["legacy"], row["legacy_reasons"]) == (True, ["no QK-norm"])


def test_rejects_changed_inputs_and_reference_errors(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    run_dir = tmp_path / "experiments/toy_taylor/runs/r000000000001"
    # Provenance is not pinned; a changed measurement file is.
    record = json.loads((run_dir / "record.json").read_text())
    _write(run_dir / "record.json", dict(record, attempts=[{"node": "n1"}]))
    collect_rows(tmp_path, config)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    _write(run_dir / "metadata.json", dict(metadata, extra="mutated"))
    with pytest.raises(ValueError, match="Scientific inputs of toy_taylor/r000000000001 differ"):
        collect_rows(tmp_path, config)
    _write(run_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="Pinned runs must be exactly the cells' runs"):
        collect_rows(tmp_path, dict(config, runs=config["runs"][:1]))

    duplicate = dict(config, cells=[*config["cells"], config["cells"][0]])
    with pytest.raises(ValueError, match="Duplicate or out-of-grid"):
        collect_rows(tmp_path, duplicate)
    bad_width = _pin(tmp_path, dict(config, cells=[dict(config["cells"][0], d_k=32)]))
    with pytest.raises(ValueError, match="Duplicate or out-of-grid"):
        collect_rows(tmp_path, bad_width)


def test_rejects_terminal_metric_disagreement_even_with_fresh_hashes(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    metrics_path = tmp_path / "experiments/toy_taylor/runs/r000000000001/metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    records[-1]["validation_perplexity"] = 21.0
    _write(metrics_path, records)
    with pytest.raises(ValueError, match="completion validation"):
        collect_rows(tmp_path, _pin(tmp_path, config))


def test_declared_but_unfinished_run_is_incomplete(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    experiment = "toy_taylor"
    manifest_path = tmp_path / "experiments" / experiment / "sweep.yaml"
    manifest = load_study(manifest_path)
    name = "r000000000003"
    manifest["runs"].append({"name": name, "status": "running", "seed": 0,
                             "qk_head_dim": 8, "attention_degree": 1,
                             "source_commit": "source-commit-1"})
    _write(manifest_path, manifest)
    config["cells"].append({"panel": "normalized_fixed", "column": "m1", "d_k": 8,
                            "experiment": experiment, "run": name})
    rows = collect_rows(tmp_path, _pin(tmp_path, config))
    assert rows[0]["status"] == "incomplete"
    assert rows[0]["terminal_ppl"] is None
    assert rows[0]["artifacts"] == {}


def test_committed_comparison_sources() -> None:
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "results/attention_scaling/selection.yaml").read_text())
    rows = collect_rows(root, config)
    assert len(rows) == 45
    assert {status: sum(row["status"] == status for row in rows)
            for status in {row["status"] for row in rows}} == {"complete": 31, "numerical_failure": 2, "not_run": 12}
    current = {(row["panel"], row["column"], row["d_k"]) for row in rows
               if row["status"] != "not_run" and not row["legacy"]}
    assert current == {("references", "softmax", 64), ("references", "softmax_no_qk_norm", 64),
                       *(("references", "gla", width) for width in (8, 16, 64)),
                       ("normalized_fixed", "m4", 16), ("normalized_fixed", "m8", 16), ("normalized_fixed", "m16", 128),
                       *(("normalized_fixed", f"m{degree}", 64) for degree in (1, 2, 3, 4, 8, 16))}
    failures = {(row["panel"], row["column"]): row["failure_step"]
                for row in rows if row["status"] == "numerical_failure"}
    assert failures == {("normalized_fixed", "m1"): 31, ("normalized_fixed", "m3"): 419}
    assert all(row["d_k"] == 64 for row in rows if row["status"] == "numerical_failure")

    chart = yaml.safe_load((root / "results/attention_scaling/degree_chart.yaml").read_text())
    chart_rows = collect_rows(root, chart)
    assert len(chart_rows) == 30
    chart_failures = {(row["panel"], row["column"]): row["failure_step"]
                      for row in chart_rows if row["status"] == "numerical_failure"}
    assert chart_failures == {("normalized_learned_taylor", "m1"): 27, ("normalized_learned_taylor", "m2"): 1857,
                              ("normalized_learned_taylor", "m3"): 318, ("normalized_learned_taylor", "m4"): 1456,
                              ("normalized_fixed_hermite", "m1"): 5, ("normalized_fixed_hermite", "m2"): 3829,
                              ("normalized_fixed_hermite", "m3"): 802, ("normalized_fixed_hermite", "m4"): 1390,
                              ("normalized_fixed", "m1"): 31, ("normalized_fixed", "m3"): 419}
    legacy = {(row["panel"], row["column"]) for row in chart_rows if row["legacy"]}
    assert legacy == set()

    for panel, other, message in (("unnormalized_fixed_hermite", "unnormalized_fixed", "no fixed Hermite"),
                                  ("unnormalized_fixed", "unnormalized_fixed_hermite", "a Taylor cell uses 1/r!")):
        swapped = json.loads(json.dumps(chart))
        target = next(cell for cell in swapped["cells"] if cell["panel"] == panel and cell["column"] == "m2")
        source = next(cell for cell in chart["cells"] if cell["panel"] == other and cell["column"] == "m2")
        swapped["cells"] = [cell for cell in swapped["cells"] if cell["run"] != source["run"]]
        target.update(experiment=source["experiment"], run=source["run"])
        with pytest.raises(ValueError, match=message):
            collect_rows(root, _pin(root, swapped))

    wrong_family = json.loads(json.dumps(chart))
    quadratic = "experiments/attention_scaling"
    cell = next(cell for cell in wrong_family["cells"] if cell["panel"] == "normalized_nonnegative")
    record = next(run for run in load_study(root / quadratic)["runs"] if run["name"] == "200m-positive-global-seed0")
    cell.update(experiment="attention_scaling", run=record["run_id"])
    with pytest.raises(ValueError, match="quadratic sharing"):
        collect_rows(root, _pin(root, wrong_family))

    wrong_data = json.loads(json.dumps(config))
    wrong_data["data_identity"]["eval_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data identity eval_manifest_sha256"):
        collect_rows(root, wrong_data)
