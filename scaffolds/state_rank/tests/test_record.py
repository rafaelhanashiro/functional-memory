import csv
import json

import pytest

from scaffolds.state_rank import record


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_links_map_to_canonical_record_paths():
    assert record.relocate("random_vs_pretrained/token_metrics.png") == "figures/random_vs_pretrained-token_metrics.png"
    assert record.relocate("unified_comparison/llama/heads.pdf") == "figures/unified_comparison-llama-heads.pdf"
    assert record.relocate("random_vs_pretrained_deltanet/summary.csv") == "summary.csv"
    assert record.relocate("unified_comparison/depth_summary.csv") == "summary.csv"
    assert record.relocate("unified_comparison/validation.json") == "provenance/manifest.json"
    assert record.relocate("unified_comparison/comparison_statistics.json") is None
    assert record.relocate("configs/unified_comparison.resolved.json") == "provenance/manifest.json"
    assert record.relocate("unified_comparison/per_head_terminal.csv") is None
    assert record.relocate("../../scaffolds/state_rank/") == "../../scaffolds/state_rank/"
    assert record.relocate("configs/random_vs_pretrained.json") == "configs/random_vs_pretrained.json"
    text = "See [heads](unified_comparison/llama/heads.png) and [per head](unified_comparison/per_head_terminal.csv)."
    assert record.relocate_links(text) == "See [heads](figures/unified_comparison-llama-heads.png) and per head."


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
    inputs = record_dir / "runs/llama-3.2-1b-l8-kvall-t512/inputs.json"
    inputs.parent.mkdir(parents=True)
    inputs.write_text('{"records": []}\n')
    run = record_dir / "runs/random_vs_pretrained-pretrained"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text("{}")
    manifest = {"baseline_directory": "/elsewhere/historical/llama-3.2-1b-l8-kvall-t512",
                "inputs_sha256": record.sha256(inputs),
                "conditions": [{"name": "pretrained", "weights": "pretrained", "seed": None}]}
    provenance = {"comparisons": {"random_vs_pretrained": {"manifest": manifest, "resolved_config": {"tokens": 512}}}}
    (record_dir / "provenance").mkdir()
    (record_dir / "provenance/manifest.json").write_text(json.dumps(provenance))
    workspace = record.reconstitute(record_dir, tmp_path / "workspace")
    comparison = workspace / "random_vs_pretrained"
    assert (comparison / "pretrained").resolve() == run.resolve()
    assert json.loads((comparison / "manifest.json").read_text()) == manifest
    assert (comparison / "inputs.json").read_text() == inputs.read_text()
    assert json.loads((workspace / "configs/random_vs_pretrained.resolved.json").read_text()) == {"tokens": 512}
    manifest["inputs_sha256"] = "0" * 64
    (record_dir / "provenance/manifest.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="retained inputs differ"):
        record.reconstitute(record_dir, tmp_path / "second")
