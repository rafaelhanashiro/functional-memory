import json

import pytest

from results.state_rank import plot
from scaffolds.records import load_study, run_directory

STUDY = plot.ROOT / "experiments/state_rank"


def test_selection_pins_current_run_inputs_and_names_every_role_once():
    selection = plot.load_selection()
    labels = [entry["label"] for entry in selection["runs"]]
    assert len(labels) == len(set(labels))
    roles = [role for model in selection["analysis"]["models"] for role in (model["pretrained"], *model["random"])]
    assert sorted(roles) == sorted(labels)
    assert {entry["run_id"] for entry in selection["runs"]} <= set(load_study(STUDY)["run_ids"])
    for entry in selection["runs"]:
        run = run_directory(STUDY, entry["run_id"])
        record = json.loads((run / "record.json").read_text())
        assert entry["config_sha256"] == record["config"]["sha256"] == plot.sha256(run / "config.yaml")
        assert entry["inputs"] == record["artifacts"]


def test_summary_table_replaces_only_its_markers():
    document = ("# Title\n\nAuthored.\n\n<!-- state-rank:summary:start -->\nold\n"
                "<!-- state-rank:summary:end -->\n\nMore.\n")
    updated = plot.TABLE.sub(lambda match: match[1] + "new" + "\n" + match[2], document)
    assert updated == document.replace("\nold\n", "\nnew\n")


def test_regeneration_is_byte_identical(tmp_path):
    selection = plot.load_selection()
    if not all((plot.ROOT / entry["inputs"]["numerical_data"]["path"]).is_file() for entry in selection["runs"]):
        pytest.skip("The numerical archives are ignored by Git and live in the Kennel primary checkout")
    plot.regenerate(output=tmp_path)
    for name in ("state_rank.png", "state_rank.pdf", "summary.csv", "README.md"):
        assert (tmp_path / name).read_bytes() == (plot.RESULTS / name).read_bytes(), name
