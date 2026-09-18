import hashlib
import json

import pytest

from experiments.key_dimension.plot import aggregate, summarize_run


def fixture_run(tmp_path, dim=64, source_commit="a" * 40):
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
            "source_commit": source_commit, "qk_head_dim": dim,
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


def fixture_extension(tmp_path):
    original_run, original_dir = fixture_run(tmp_path, 64)
    extension_run, extension_dir = fixture_run(tmp_path, 256, "b" * 40)
    for workdir in (original_dir, extension_dir):
        (workdir / "completion.json").write_text(json.dumps({"final_step": 10, "training_tokens": 1000}))
    original = {
        "source_commit": "a" * 40, "identity": "original", "project_root": str(tmp_path),
        "data_identity": {"train_manifest": {"path": "/data/train", "sha256": "d" * 64}},
        "runs": [original_run],
    }
    extension = {
        "source_commit": "b" * 40, "identity": "extension", "project_root": str(tmp_path),
        "data_identity": original["data_identity"], "runs": [extension_run],
    }
    reference_path = tmp_path / "original.json"
    reference_path.write_text(json.dumps(original))
    extension["reference_manifests"] = [{
        "path": str(reference_path), "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }]
    return extension, reference_path


def test_aggregate_combines_pinned_sources_and_preserves_reference(tmp_path):
    manifest, reference = fixture_extension(tmp_path)
    original_bytes = reference.read_bytes()
    original_metrics = (tmp_path / "dk64/metrics.jsonl").read_bytes()
    rows = aggregate(manifest)
    assert [row["qk_head_dim"] for row in rows] == [64, 256]
    assert all(row["status"] == "complete" and row["terminal_ppl"] == 8.17 for row in rows)
    readme = (tmp_path / "results/key_dimension/README.md").read_text()
    assert "2/2 runs complete" in readme
    for source in ("a" * 40, "b" * 40, "original", "extension"):
        assert f"`{source}`" in readme
    assert manifest["reference_manifests"][0]["sha256"] in readme
    assert reference.read_bytes() == original_bytes
    assert (tmp_path / "dk64/metrics.jsonl").read_bytes() == original_metrics


def test_aggregate_rejects_modified_reference_manifest(tmp_path):
    manifest, reference = fixture_extension(tmp_path)
    reference.write_text(reference.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        aggregate(manifest)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("field,value", [("qk_head_dim", 64), ("name", "200m-dk064-seed0")])
def test_aggregate_rejects_duplicate_run_identities(tmp_path, field, value):
    manifest, _ = fixture_extension(tmp_path)
    manifest["runs"][0][field] = value
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(manifest)


def test_referenced_metrics_must_match_their_own_source(tmp_path):
    manifest, reference = fixture_extension(tmp_path)
    original = json.loads(reference.read_text())
    original["source_commit"] = "c" * 40
    reference.write_text(json.dumps(original))
    manifest["reference_manifests"][0]["sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Metric identity"):
        aggregate(manifest)


def test_aggregate_rejects_incomparable_data_identity(tmp_path):
    manifest, _ = fixture_extension(tmp_path)
    manifest["data_identity"] = {"train_manifest": {"path": "/data/other", "sha256": "e" * 64}}
    with pytest.raises(ValueError, match="data identity"):
        aggregate(manifest)


@pytest.mark.parametrize("field,value", [("training_tokens", 2000), ("value_head_dim", 128)])
def test_aggregate_rejects_incomparable_recipe_or_budget(tmp_path, field, value):
    manifest, _ = fixture_extension(tmp_path)
    manifest["runs"][0][field] = value
    with pytest.raises(ValueError, match="recipe or training budget"):
        aggregate(manifest)
