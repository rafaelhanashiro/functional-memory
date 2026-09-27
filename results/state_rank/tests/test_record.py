import csv
import json

import pytest

from results.state_rank import record
from scaffolds.records import create_run, load_study, run_directory
import yaml


def test_committed_selection_pins_current_run_records():
    """Provenance edits must update the selection without loading numerical archives."""
    selection = yaml.safe_load((record.RESULTS / "selection.yaml").read_text())
    entries = selection["runs"]
    pinned = [entry["run_id"] for entry in entries]
    selected = set(selection["collections"].values())
    for comparison in selection["comparisons"].values():
        selected.add(comparison["baseline_run_id"])
        selected.update(comparison["condition_run_ids"].values())
    assert len(pinned) == len(set(pinned))
    assert set(pinned) == selected
    assert selected <= set(load_study(record.RECORD)["run_ids"])
    for entry in entries:
        run = run_directory(record.RECORD, entry["run_id"])
        canonical = json.loads((run / "record.json").read_text())
        assert entry["record"] == (run / "record.json").relative_to(record.REPOSITORY_ROOT).as_posix()
        assert entry["record_sha256"] == record.sha256(run / "record.json"), entry["run_id"]
        assert entry["config_sha256"] == canonical["config"]["sha256"] == record.sha256(run / "config.yaml")
        assert entry["inputs"] == canonical["artifacts"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_links_map_to_canonical_record_paths():
    assert record.relocate("random_vs_pretrained/token_metrics.png") == "random_vs_pretrained-token_metrics.png"
    assert record.relocate("unified_comparison/llama/heads.pdf") == "unified_comparison-llama-heads.pdf"
    assert record.relocate("random_vs_pretrained_deltanet/summary.csv") == "summary.csv"
    assert record.relocate("unified_comparison/depth_summary.csv") == "summary.csv"
    assert record.relocate("unified_comparison/validation.json") == "../../experiments/state_rank/sweep.yaml"
    assert record.relocate("unified_comparison/comparison_statistics.json") is None
    assert record.relocate("configs/unified_comparison.resolved.json") == "../../experiments/state_rank/sweep.yaml"
    assert record.relocate("unified_comparison/per_head_terminal.csv") is None
    assert record.relocate("../../scaffolds/state_rank/") == "../../scaffolds/state_rank/"
    assert record.relocate("configs/random_vs_pretrained.json") == "configs/random_vs_pretrained.json"
    text = "See [heads](unified_comparison/llama/heads.png) and [per head](unified_comparison/per_head_terminal.csv)."
    assert record.relocate_links(text) == "See [heads](unified_comparison-llama-heads.png) and per head."


def test_generated_sections_replace_only_their_markers():
    document = ("# State rank\n\nAuthored.\n\n<!-- state-rank:unified-comparison:start -->\nold\n"
                "<!-- state-rank:unified-comparison:end -->\n\nMore authored text.\n")
    updated = record.replace_sections(document, {"unified-comparison": "new", "comparison-x": "added"})
    assert "Authored." in updated and "More authored text." in updated
    assert record.sections(updated) == {"unified-comparison": "new", "comparison-x": "added"}


def test_summary_joins_comparison_only_columns(tmp_path):
    write_csv(tmp_path / "unified_comparison/depth_summary.csv", [
        {"model": "llama", "condition": "pretrained", "layer": "0", "threshold_rank": "3"},
        {"model": "gated-deltanet", "condition": "pretrained", "layer": "0", "threshold_rank": "5"},
    ])
    write_csv(tmp_path / "random_vs_pretrained/summary.csv",
              [{"condition": "pretrained", "layer": "0", "final_log_trace_mean": "1.5", "final_key_norms_mean": "2"}])
    write_csv(tmp_path / "random_vs_pretrained_deltanet/summary.csv",
              [{"condition": "pretrained", "layer": "0", "final_log_trace_mean": "0.1"}])
    rows = record.merged_summary(tmp_path)
    assert rows[0]["final_key_norms_mean"] == "2" and rows[0]["threshold_rank"] == "3"
    assert rows[1]["final_log_trace_mean"] == "0.1" and rows[1]["final_key_norms_mean"] == ""
    write_csv(tmp_path / "random_vs_pretrained_deltanet/summary.csv",
              [{"condition": "random-seed0", "layer": "0", "final_log_trace_mean": "0.1"}])
    with pytest.raises(ValueError, match="No comparison summary row"):
        record.merged_summary(tmp_path)


def test_reconstitute_links_runs_and_verifies_inputs(tmp_path):
    record_dir = tmp_path / "record"
    baseline = create_run(record_dir, {"tokens": 512}, "baseline")
    inputs = baseline / "inputs.json"
    inputs.write_text('{"records": []}\n')
    run = create_run(record_dir, {"tokens": 512}, "comparison")
    (run / "metadata.json").write_text("{}")
    manifest = {"baseline_directory": "/elsewhere/historical/baseline",
                "inputs_sha256": record.sha256(inputs),
                "conditions": [{"name": "pretrained", "weights": "pretrained", "seed": None}]}
    sweep = yaml.safe_load((record_dir / "sweep.yaml").read_text())
    sweep["comparisons"] = {"random_vs_pretrained": {"manifest": manifest,
        "resolved_config": {"tokens": 512}, "baseline_run_id": baseline.name,
        "condition_run_ids": {"pretrained": run.name}}}
    (record_dir / "sweep.yaml").write_text(yaml.safe_dump(sweep))
    workspace = record.reconstitute(record_dir, tmp_path / "workspace")
    comparison = workspace / "random_vs_pretrained"
    assert (comparison / "pretrained").resolve() == run.resolve()
    assert json.loads((comparison / "manifest.json").read_text()) == manifest
    assert (comparison / "inputs.json").read_text() == inputs.read_text()
    assert json.loads((workspace / "configs/random_vs_pretrained.resolved.json").read_text()) == {"tokens": 512}
    manifest["inputs_sha256"] = "0" * 64
    (record_dir / "sweep.yaml").write_text(yaml.safe_dump(sweep))
    with pytest.raises(ValueError, match="retained inputs differ"):
        record.reconstitute(record_dir, tmp_path / "second")


def test_explicit_selection_controls_condition_source(tmp_path):
    study = tmp_path / "study"
    baseline = create_run(study, {"tokens": 8}, "baseline")
    (baseline / "inputs.json").write_text('{}\n')
    original = create_run(study, {"seed": 0}, "first execution")
    replacement = create_run(study, {"seed": 0}, "second execution")
    sweep = yaml.safe_load((study / "sweep.yaml").read_text())
    entry = {"manifest": {"conditions": [{"name": "random-seed0"}],
             "inputs_sha256": record.sha256(baseline / "inputs.json")},
             "resolved_config": {}, "baseline_run_id": baseline.name,
             "condition_run_ids": {"random-seed0": original.name}}
    sweep["comparisons"] = {"random_vs_pretrained": entry}
    (study / "sweep.yaml").write_text(yaml.safe_dump(sweep))
    selection = {"comparisons": {"random_vs_pretrained": {
        "baseline_run_id": baseline.name,
        "condition_run_ids": {"random-seed0": replacement.name}}}}
    workspace = record.reconstitute(study, tmp_path / "workspace", selection)
    assert (workspace / "random_vs_pretrained/random-seed0").resolve() == replacement.resolve()
