import hashlib
import json

import yaml

from scaffolds.records.selections import refresh_selection, run_sources, verify_pinned_runs


def run(root, study, run_id, **files):
    directory = root / "experiments" / study / "runs" / run_id
    directory.mkdir(parents=True)
    for name, text in {"config.yaml": f"run: {run_id}\n", "record.json": "{}\n", **files}.items():
        (directory / name).write_text(text)
    return directory


def result(root, name="toy", **selections):
    directory = root / "results" / name
    directory.mkdir(parents=True)
    for file, selection in selections.items():
        (directory / f"{file}.yaml").write_text(yaml.safe_dump(selection, sort_keys=False, allow_unicode=True))
    return directory


def test_refresh_pins_scientific_files_and_follows_a_repointed_role(tmp_path):
    old = run(tmp_path, "toy", "r000000000001", **{"metrics.jsonl": "1\n", "wandb_run.json": "{}\n"})
    new = run(tmp_path, "toy", "r000000000002", **{"metrics.jsonl": "2\n", "completion.json": "{}\n"})
    stale = {"record.json": {"path": f"experiments/toy/runs/{old.name}/record.json", "sha256": "0" * 64}}
    directory = result(tmp_path, selection={"metric": "ppl · terminal", "runs": [
        {"study": "toy", "run_id": old.name, "label": "baseline", "sources": stale}]})

    assert refresh_selection(directory) == [
        f"selection.yaml: toy/{old.name} config.yaml pinned",
        f"selection.yaml: toy/{old.name} metrics.jsonl pinned",
        f"selection.yaml: toy/{old.name} record.json removed"]
    selection = yaml.safe_load((directory / "selection.yaml").read_text())
    assert set(selection["runs"][0]["sources"]) == {"config.yaml", "metrics.jsonl"}
    verify_pinned_runs(tmp_path, selection["runs"])
    assert "ppl · terminal" in (directory / "selection.yaml").read_text()
    assert refresh_selection(directory) == []

    # Provenance edits leave the pins alone; a corrected measurement is re-pinned.
    (old / "record.json").write_text('{"status": "complete"}\n')
    assert refresh_selection(directory) == []
    (old / "metrics.jsonl").write_text("1.5\n")
    assert refresh_selection(directory) == [f"selection.yaml: toy/{old.name} metrics.jsonl changed"]

    selection = yaml.safe_load((directory / "selection.yaml").read_text())
    selection["runs"][0]["run_id"] = new.name
    (directory / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False, allow_unicode=True))
    assert refresh_selection(directory) == [f"selection.yaml: toy/{new.name} pinned for baseline, replacing {old.name}"]
    entry = yaml.safe_load((directory / "selection.yaml").read_text())["runs"][0]
    assert entry["label"] == "baseline"
    assert entry["sources"] == run_sources(tmp_path, "toy", new.name)
    assert set(entry["sources"]) == {"config.yaml", "metrics.jsonl", "completion.json"}


def test_refresh_pins_exactly_the_runs_a_grid_names(tmp_path):
    first = run(tmp_path, "a", "r000000000001", **{"numerical_failure_step_00000009.json": "{}\n"})
    second = run(tmp_path, "b", "r000000000002")
    cells = [{"panel": "p", "column": "m1", "d_k": 8, "experiment": "a", "run": first.name, "label": "one"},
             {"panel": "p", "column": "m2", "d_k": 8, "experiment": "b", "run": second.name, "label": "two"}]
    directory = result(tmp_path, grid={"cells": cells, "runs": [
        {"study": "b", "run_id": "r0000000000ff", "label": "gone", "sources": {}}]})

    assert refresh_selection(directory) == [f"grid.yaml: a/{first.name} pinned", f"grid.yaml: b/{second.name} pinned",
                                            "grid.yaml: b/r0000000000ff unpinned"]
    runs = yaml.safe_load((directory / "grid.yaml").read_text())["runs"]
    assert [(entry["study"], entry["run_id"], entry["label"]) for entry in runs] == [
        ("a", first.name, "one"), ("b", second.name, "two")]
    assert "numerical_failure_step_00000009.json" in runs[0]["sources"]
    verify_pinned_runs(tmp_path, runs)


def test_refresh_rehashes_captured_inputs_of_array_analyses(tmp_path):
    directory = run(tmp_path, "arrays", "r000000000001", **{"metadata.json": "{}\n"})
    kept = {"metadata": {"path": f"experiments/arrays/runs/{directory.name}/metadata.json",
                         "sha256": hashlib.sha256(b"{}\n").hexdigest()},
            "numerical_data": {"path": f"experiments/arrays/runs/{directory.name}/spectra.npz", "sha256": "a" * 64}}
    (directory / "record.json").write_text(json.dumps({"artifacts": kept}))
    selection = {"study": "experiments/arrays", "runs": [
        {"run_id": directory.name, "label": "x", "config_sha256": "0" * 64, "inputs": {}}]}
    results = result(tmp_path, selection=selection)

    assert refresh_selection(results) == [
        f"selection.yaml: arrays/{directory.name} config.yaml changed",
        f"selection.yaml: arrays/{directory.name} metadata.json pinned",
        f"selection.yaml: arrays/{directory.name} spectra.npz pinned"]
    entry = yaml.safe_load((results / "selection.yaml").read_text())["runs"][0]
    assert entry["config_sha256"] == hashlib.sha256((directory / "config.yaml").read_bytes()).hexdigest()
    # The archive lives only on the home cluster, so its captured hash is kept.
    assert entry["inputs"] == kept


def test_refresh_preserves_distinct_roles_that_share_a_run(tmp_path):
    source = run(tmp_path, "toy", "r000000000001")
    directory = result(tmp_path, selection={"runs": [
        {"study": "toy", "run_id": source.name, "label": label, "sources": {}}
        for label in ("baseline", "control")]})

    refresh_selection(directory)
    entries = yaml.safe_load((directory / "selection.yaml").read_text())["runs"]
    assert [entry["label"] for entry in entries] == ["baseline", "control"]
    verify_pinned_runs(tmp_path, entries)
    assert refresh_selection(directory) == []


def test_refresh_follows_array_comparison_and_collection_roles(tmp_path):
    run_ids = [f"r{number:012x}" for number in range(1, 7)]
    for run_id in run_ids:
        run(tmp_path, "arrays", run_id, **{"record.json": '{"artifacts": {}}\n'})
    selection = {
        "study": "experiments/arrays",
        "collections": {"collection": run_ids[0]},
        "comparisons": {"comparison": {
            "baseline_run_id": run_ids[1], "condition_run_ids": {"pretrained": run_ids[2]}}},
        "runs": [{"run_id": run_id, "label": label, "config_sha256": "0" * 64, "inputs": {}}
                 for run_id, label in zip(run_ids, ("collection", "baseline", "pretrained"))],
    }
    directory = result(tmp_path, selection=selection)
    refresh_selection(directory)
    assert refresh_selection(directory) == []

    selection = yaml.safe_load((directory / "selection.yaml").read_text())
    selection["collections"]["collection"] = run_ids[3]
    comparison = selection["comparisons"]["comparison"]
    comparison["baseline_run_id"] = run_ids[4]
    comparison["condition_run_ids"]["pretrained"] = run_ids[5]
    (directory / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False, allow_unicode=True))

    changes = refresh_selection(directory)
    entries = yaml.safe_load((directory / "selection.yaml").read_text())["runs"]
    assert {entry["run_id"] for entry in entries} == set(run_ids[3:])
    assert len(entries) == 3
    for entry in entries:
        recipe = tmp_path / "experiments/arrays/runs" / entry["run_id"] / "config.yaml"
        assert entry["config_sha256"] == hashlib.sha256(recipe.read_bytes()).hexdigest()
        assert entry["inputs"] == {}
    assert len(changes) == 6
    assert refresh_selection(directory) == []
