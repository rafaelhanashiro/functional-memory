"""Saved comparisons pin their run identity and scientific inputs."""
import hashlib
import json

import pytest
import yaml

from scaffolds.transformer.analysis import verify_result_selection


def test_result_selection_checks_source_hashes_and_run_ownership(tmp_path):
    run = tmp_path / "experiments/toy/runs/r000000000001"
    run.mkdir(parents=True)
    source = run / "metrics.jsonl"
    source.write_text(json.dumps({"validation_perplexity": 12.0}) + "\n")
    result = tmp_path / "results/toy"
    result.mkdir(parents=True)
    selection = {"runs": [{"study": "toy", "run_id": run.name, "sources": {
        "metrics": {"path": str(source.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}]}
    path = result / "selection.yaml"
    path.write_text(yaml.safe_dump(selection))
    assert verify_result_selection(result) == selection
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_result_selection(result)
    selection["runs"][0]["sources"]["metrics"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    selection["runs"][0]["run_id"] = "r000000000002"
    path.write_text(yaml.safe_dump(selection))
    with pytest.raises(ValueError, match="must belong"):
        verify_result_selection(result)


def test_selected_ids_change_inputs_without_changing_plotting_code(tmp_path):
    from scaffolds.records import create_run
    from scaffolds.transformer.analysis import load_selected_study, selected_run_directory

    study = tmp_path / "experiments/toy"
    first = create_run(study, {"value": 1}, label="old execution")
    second = create_run(study, {"value": 2}, label="new execution")
    result = tmp_path / "results/toy"
    result.mkdir(parents=True)

    def select(run):
        sources = {name: {"path": str((run / name).relative_to(tmp_path)),
                          "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest()}
                   for name in ("record.json", "config.yaml")}
        selection = {"runs": [{"study": "toy", "run_id": run.name, "label": "baseline", "sources": sources}]}
        (result / "selection.yaml").write_text(yaml.safe_dump(selection))

    select(first)
    assert selected_run_directory(result, study, "baseline") == first
    assert [r["run_id"] for r in load_selected_study(result, study)["runs"]] == [first.name]
    select(second)
    assert selected_run_directory(result, study, "baseline") == second
    selected = load_selected_study(result, study)["runs"]
    assert [r["run_id"] for r in selected] == [second.name]
    assert selected[0]["name"] == "baseline"
    with pytest.raises(ValueError, match="Expected one selected run"):
        selected_run_directory(result, study, first.name)
