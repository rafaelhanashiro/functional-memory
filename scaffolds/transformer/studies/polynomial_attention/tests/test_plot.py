import hashlib
import json
import shutil
import sys

import pytest
import yaml

from scaffolds.transformer.studies.polynomial_attention.plot import aggregate, main, summarize_run
from scaffolds.transformer.studies.polynomial_attention.sweep import run_name


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
        assert (output / ("figures" if filename.endswith((".pdf", ".png")) else "") / filename).stat().st_size > 0
    assert "m=0" in (output / "README.md").read_text()


def retained_fixture(root):
    """A retained result whose best checkpoint precedes its terminal endpoint."""
    results = root / "results/polynomial_attention"
    results.mkdir(parents=True)
    (results / "runs").mkdir()
    run, workdir = fixture_run(results / "runs")
    write_completion(run, workdir)
    config = {
        "data": {"batch_size": 10, "sequence_length": 11},
        "model": {"attention_degree": 16, "qk_head_dim": 64, "value_head_dim": 64},
        "optim": {"total_steps": 10},
        "train": {"seed": 0, "source_commit": "a" * 40, "workdir": "/removed/execution/checkout/run"},
    }
    config_path = results / f"configs/{run['name']}.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    lock_path = results / "configs/uv.lock"
    lock_path.write_text("retained dependency lock\n")

    def artifact(path):
        return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    retained = {key: run[key] for key in ("name", "attention_degree", "qk_head_dim", "value_head_dim", "parameter_count")}
    retained.update(
        seed=0, source_commit="a" * 40, resolved_config=artifact(config_path),
        metrics=artifact(workdir / "metrics.jsonl"), completion=str((workdir / "completion.json").relative_to(root)),
        selected_checkpoint={"path": f"checkpoints/polynomial_attention/{run['name']}/5",
                             "step": 5, "validation_nll": 2.0, "validation_perplexity": 7.39},
        terminal={"step": 10, "training_tokens": 1000, "validation_nll": 2.1, "validation_perplexity": 8.17},
    )
    manifest = {"format": "functional-memory-retained-experiment", "experiment": "polynomial_attention",
                "runs": [retained], "dependency_lock": artifact(lock_path),
                "dependency_lock_sha256": artifact(lock_path)["sha256"]}
    path = results / "provenance/manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_retained_cli_is_portable_without_checkpoints_or_original_checkout(tmp_path, monkeypatch):
    original = tmp_path / "original"
    retained_fixture(original)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    manifest_path = relocated / "results/polynomial_attention/provenance/manifest.json"
    monkeypatch.setattr(sys, "argv", ["plot", "--manifest", str(manifest_path)])
    main()
    import csv
    with (manifest_path.parent.parent / "summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "complete"
    assert row["terminal_ppl"] == "8.17" and row["best_ppl"] == "7.39" and row["best_step"] == "5"
    assert (manifest_path.parent.parent / "figures/learningcurves.pdf").stat().st_size > 0
    readme = (manifest_path.parent.parent / "README.md").read_text()
    assert "provenance/manifest.json" in readme and "validation-best" in readme
    assert not (relocated / "checkpoints/polynomial_attention").exists()


@pytest.mark.parametrize("field", ["resolved_config", "metrics"])
def test_retained_artifact_changes_are_rejected(tmp_path, field):
    manifest_path, manifest = retained_fixture(tmp_path)
    path = tmp_path / manifest["runs"][0][field]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        aggregate(manifest, manifest_path=manifest_path)


@pytest.mark.parametrize("field", ["selected_checkpoint", "terminal"])
def test_retained_best_and_terminal_metrics_must_match_validation_history(tmp_path, field):
    manifest_path, manifest = retained_fixture(tmp_path)
    run = manifest["runs"][0]
    if field == "selected_checkpoint":
        run[field].update(step=10, validation_nll=2.1, validation_perplexity=8.17)
    else:
        run[field].update(validation_nll=2.0, validation_perplexity=7.39)
    with pytest.raises(ValueError, match="validation-best|terminal metrics"):
        aggregate(manifest, manifest_path=manifest_path)


def test_retained_plot_checks_recorded_shared_dependency_lock(tmp_path):
    manifest_path, manifest = retained_fixture(tmp_path)
    old_lock = tmp_path / "results/polynomial_attention/configs/uv.lock"
    new_lock = tmp_path / "scaffolds/transformer/uv.lock"
    new_lock.parent.mkdir(parents=True)
    old_lock.rename(new_lock)
    manifest["dependency_lock_path"] = str(new_lock.relative_to(tmp_path))
    manifest.pop("dependency_lock")
    rows = aggregate(manifest, manifest_path=manifest_path)
    assert rows[0]["status"] == "complete"
    new_lock.write_text("changed dependency lock\n")
    with pytest.raises(ValueError, match="checksum"):
        aggregate(manifest, manifest_path=manifest_path)


def test_regeneration_preserves_authored_readme_and_canonical_paths(tmp_path):
    manifest_path, manifest = retained_fixture(tmp_path)
    results = manifest_path.parent.parent
    readme = results / "README.md"
    prefix = "# Authored study\n\nMethods that must survive.\n\n"
    suffix = "\n\n## Reproduce\nUse the recorded configuration.\n"
    readme.write_text(prefix + "<!-- BEGIN GENERATED RESULTS -->\nold output\n<!-- END GENERATED RESULTS -->" + suffix)
    aggregate(manifest, manifest_path=manifest_path)
    text = readme.read_text()
    assert text.startswith(prefix) and text.endswith(suffix)
    assert "old output" not in text and text.count("<!-- BEGIN GENERATED RESULTS -->") == 1
    aggregate(manifest, manifest_path=manifest_path)
    assert readme.read_text() == text
    assert (results / "figures/ppl_vs_degree.png").is_file()
    assert not list(results.rglob("REPORT.md"))
    assert not (results / "ppl_vs_degree.png").exists()


def test_canonical_default_precedes_legacy_manifest(tmp_path):
    from scaffolds.transformer.studies.polynomial_attention.plot import default_manifest_path
    path, _ = retained_fixture(tmp_path)
    (path.parent.parent / "provenance.json").write_text("legacy snapshot")
    assert default_manifest_path(path.parent.parent) == path


def test_explicit_historical_retention_manifest_remains_supported(tmp_path):
    canonical, manifest = retained_fixture(tmp_path)
    historical = canonical.parent.parent / "provenance.json"
    canonical.rename(historical)
    rows = aggregate(manifest, manifest_path=historical)
    assert rows[0]["status"] == "complete"
    assert "[Provenance](provenance.json)" in (historical.parent / "README.md").read_text()
