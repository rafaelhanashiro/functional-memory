"""Checks for the combined heatmap's source attribution and missing-cell policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.attention_scaling_comparison.adapter.analysis.collect import (
    collect_rows, source_artifact_set_sha256,
)


def _write(path: Path, value: dict | list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in value)
    else:
        payload = json.dumps(value, sort_keys=True) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fixture(tmp_path: Path) -> dict:
    experiment = "toy_taylor"
    base = tmp_path / "experiments" / experiment
    complete_name = "dk8-m2-seed0"
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

    failed_name = "dk16-m1-seed0"
    failed_dir = base / "runs" / failed_name
    failed_metadata = {"status": "failed", "seed": 0, "qk_head_dim": 16,
                       "attention_degree": 1, "source_commit": source, "step": 7,
                       "target_steps": 10, "target_tokens": 100, "error": "nonfinite loss"}
    failure = {"step": 7, "attention_degree": 1, "error": "nonfinite loss"}
    metadata_sha = _write(failed_dir / "metadata.json", failed_metadata)
    failure_filename = "numerical_failure_step_00000007.json"
    failure_sha = _write(failed_dir / failure_filename, failure)
    failed_run = {"name": failed_name, "status": "numerical-failure", "seed": 0,
                  "qk_head_dim": 16, "attention_degree": 1, "source_commit": source,
                  "metadata": {"path": f"experiments/{experiment}/runs/{failed_name}/metadata.json",
                               "sha256": metadata_sha},
                  "failure": {"path": f"experiments/{experiment}/runs/{failed_name}/{failure_filename}",
                              "sha256": failure_sha, "step": 7}}
    manifest_path = base / "provenance" / "manifest.json"
    manifest_sha = _write(manifest_path, {"experiment": experiment, "runs": [complete_run, failed_run]})
    return {"d_k": [8, 16], "terminal_step": 10, "training_tokens": 100, "seed": 0,
            "panels": [{"id": "normalized_fixed", "title": "Taylor",
                        "columns": [{"id": "m1", "label": "m=1"},
                                    {"id": "m2", "label": "m=2"}]}],
            "source_manifests": {experiment: {"path": f"experiments/{experiment}/provenance/manifest.json",
                                              "sha256": manifest_sha}},
            "cells": [{"panel": "normalized_fixed", "column": "m2", "d_k": 8,
                       "experiment": experiment, "run": complete_name},
                      {"panel": "normalized_fixed", "column": "m1", "d_k": 16,
                       "experiment": experiment, "run": failed_name}]}


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
    # No resolved configuration: the run predates the current defaults on both counts.
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
    run_dir = tmp_path / "experiments/toy_taylor/runs/dk8-m2-seed0"
    resolved = {"optim": {"weight_decay_policy": "exclude_embeddings_norms_biases_v1"}, "model": {"qk_norm": True}}
    _write(run_dir / "config.json", resolved)
    config.pop("source_artifact_set_sha256", None)
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
    _write(run_dir / "config.json", resolved)
    config["panels"][0]["columns"][1]["qk_norm"] = True
    row = collect_rows(tmp_path, config)[2]
    assert (row["legacy"], row["legacy_reasons"]) == (True, ["no QK-norm"])


def test_rejects_mutated_source_and_reference_errors(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    source_path = tmp_path / config["source_manifests"]["toy_taylor"]["path"]
    original_manifest = json.loads(source_path.read_text())
    source_path.write_text(source_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Source manifest SHA256"):
        collect_rows(tmp_path, config)
    _write(source_path, original_manifest)

    duplicate = dict(config, cells=[*config["cells"], config["cells"][0]])
    with pytest.raises(ValueError, match="Duplicate or out-of-grid"):
        collect_rows(tmp_path, duplicate)
    bad_width = dict(config, cells=[dict(config["cells"][0], d_k=32)])
    with pytest.raises(ValueError, match="Duplicate or out-of-grid"):
        collect_rows(tmp_path, bad_width)


def test_rejects_terminal_metric_disagreement_even_with_fresh_hashes(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    experiment = "toy_taylor"
    manifest_path = tmp_path / config["source_manifests"][experiment]["path"]
    manifest = json.loads(manifest_path.read_text())
    metrics_path = tmp_path / manifest["runs"][0]["metrics"]["path"]
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    records[-1]["validation_perplexity"] = 21.0
    manifest["runs"][0]["metrics"]["sha256"] = _write(metrics_path, records)
    config["source_manifests"][experiment]["sha256"] = _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="completion validation"):
        collect_rows(tmp_path, config)


def test_declared_but_unfinished_run_is_incomplete(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    experiment = "toy_taylor"
    manifest_path = tmp_path / config["source_manifests"][experiment]["path"]
    manifest = json.loads(manifest_path.read_text())
    name = "dk8-m1-seed0"
    manifest["runs"].append({"name": name, "status": "running", "seed": 0,
                             "qk_head_dim": 8, "attention_degree": 1,
                             "source_commit": "source-commit-1"})
    config["source_manifests"][experiment]["sha256"] = _write(manifest_path, manifest)
    config["cells"].append({"panel": "normalized_fixed", "column": "m1", "d_k": 8,
                            "experiment": experiment, "run": name})
    rows = collect_rows(tmp_path, config)
    assert rows[0]["status"] == "incomplete"
    assert rows[0]["terminal_ppl"] is None
    assert rows[0]["artifacts"] == {}


def test_pinned_artifact_set_catches_changes_to_legacy_unhashed_files(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    experiment = "toy_taylor"
    manifest_path = tmp_path / config["source_manifests"][experiment]["path"]
    manifest = json.loads(manifest_path.read_text())
    for kind in ("metadata", "metrics", "completion"):
        del manifest["runs"][0][kind]["sha256"]
    config["source_manifests"][experiment]["sha256"] = _write(manifest_path, manifest)
    config["source_artifact_set_sha256"] = source_artifact_set_sha256(collect_rows(tmp_path, config))

    metadata_path = tmp_path / manifest["runs"][0]["metadata"]["path"]
    metadata = json.loads(metadata_path.read_text())
    metadata["extra_provenance"] = "mutated"
    _write(metadata_path, metadata)
    with pytest.raises(ValueError, match="Source artifact set SHA256"):
        collect_rows(tmp_path, config)


def test_committed_comparison_sources() -> None:
    root = Path(__file__).resolve().parents[4]
    config = json.loads((root / "experiments/attention_scaling_comparison/configs/heatmap.json").read_text())
    rows = collect_rows(root, config)
    assert len(rows) == 45
    assert {status: sum(row["status"] == status for row in rows)
            for status in {row["status"] for row in rows}} == {"complete": 31, "numerical_failure": 2, "not_run": 12}
    current = {(row["panel"], row["column"], row["d_k"]) for row in rows
               if row["status"] != "not_run" and not row["legacy"]}
    assert current == {("references", "softmax", 64), ("references", "softmax_no_qk_norm", 64),
                       *(("normalized_fixed", f"m{degree}", 64) for degree in (1, 2, 3, 4, 8, 16))}
    failures = {(row["panel"], row["column"]): row["failure_step"]
                for row in rows if row["status"] == "numerical_failure"}
    assert failures == {("normalized_fixed", "m1"): 31, ("normalized_fixed", "m3"): 419}
    assert all(row["d_k"] == 64 for row in rows if row["status"] == "numerical_failure")

    chart = json.loads((root / "experiments/attention_scaling_comparison/configs/degree_chart.json").read_text())
    chart_rows = collect_rows(root, chart)
    assert len(chart_rows) == 17
    chart_failures = {(row["panel"], row["column"]): row["failure_step"]
                      for row in chart_rows if row["status"] == "numerical_failure"}
    assert chart_failures == {("normalized_learned_taylor", "m1"): 27, ("normalized_learned_taylor", "m2"): 1857,
                              ("normalized_learned_taylor", "m3"): 318, ("normalized_learned_taylor", "m4"): 1456}
    legacy = {(row["panel"], row["column"]) for row in chart_rows if row["legacy"]}
    assert legacy == {*(("unnormalized_fixed", f"m{d}") for d in (1, 2, 3, 4)),
                      ("references", "unnormalized_exp")}

    wrong_family = json.loads(json.dumps(chart))
    quadratic = "experiments/learned_quadratic_attention/provenance/manifest.json"
    wrong_family["source_manifests"]["learned_quadratic_attention"] = {
        "path": quadratic, "sha256": hashlib.sha256((root / quadratic).read_bytes()).hexdigest()}
    wrong_family.pop("source_artifact_set_sha256")
    cell = next(cell for cell in wrong_family["cells"] if cell["panel"] == "normalized_nonnegative")
    cell.update(experiment="learned_quadratic_attention", run="200m-positive-global-seed0")
    with pytest.raises(ValueError, match="quadratic sharing"):
        collect_rows(root, wrong_family)

    wrong_data = json.loads(json.dumps(config))
    wrong_data["data_identity"]["eval_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data identity eval_manifest_sha256"):
        collect_rows(root, wrong_data)
