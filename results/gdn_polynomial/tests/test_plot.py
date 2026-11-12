"""The polynomial Gated DeltaNet comparison regenerates from its pinned runs and checks their recipes."""

import copy
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

from results.gdn_polynomial.plot import ROOT, build, heatmap, load_rows


DIRECTORY = ROOT / "results" / "gdn_polynomial"
SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("terminal_ppl.png", "terminal_ppl.pdf", "validation_gap.png", "validation_gap.pdf", "summary.csv",
           "README.md")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_outcomes_and_differences():
    rows = {row["run_id"]: row for row in load_rows(ROOT, SELECTION)}
    assert Counter(row["status"] for row in rows.values()) == {"complete": 12, "cancelled": 4}
    assert rows["rd40af7956426"]["terminal_validation_perplexity"] == pytest.approx(13.179060532842437)
    assert rows["rd40af7956426"]["percent_vs_control"] == pytest.approx(-2.948, abs=1e-3)
    assert rows["ra79e66bd9121"]["percent_vs_tied_gdn"] == pytest.approx(-5.472, abs=1e-3)
    assert rows["ra98612db06d4"]["percent_vs_tied_gdn"] == pytest.approx(-6.742, abs=1e-3)
    assert rows["rf65d38ec0552"]["percent_vs_control"] == pytest.approx(2.558, abs=1e-3)
    # A run cancelled before training keeps its cell without inventing measurements.
    assert rows["rae2329afb5bd"]["curve"] == [] and rows["rae2329afb5bd"]["percent_vs_tied_gdn"] is None
    # The order-0 write makes the polynomial runs clip far more often than linear ones.
    assert rows["rd40af7956426"]["gradient_clipped_fraction"] > 0.9 > 0.01 > rows["rf7341031e3d5"]["gradient_clipped_fraction"]


@pytest.mark.parametrize("field, value, message", [
    ("degree", 3, "GDN degree"), ("erase", "tied", "erase gate"), ("write", "tanh", "linear write activation"),
    ("d_k", 16, "d_K"),
])
def test_recipe_must_match_its_declared_coordinate(field, value, message):
    selection = copy.deepcopy(SELECTION)
    next(entry for entry in selection["runs"] if entry["run_id"] == "rd40af7956426")[field] = value
    with pytest.raises(ValueError, match=message):
        load_rows(ROOT, selection)


def test_changed_inputs_and_repeated_runs_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][0]["sources"]["metrics.jsonl"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_rows(ROOT, changed)
    repeated = copy.deepcopy(SELECTION)
    repeated["runs"].append(repeated["runs"][0])
    with pytest.raises(ValueError, match="selected twice"):
        load_rows(ROOT, repeated)


def test_heatmap_marks_cancelled_cells():
    figure = heatmap(load_rows(ROOT, SELECTION), SELECTION)
    try:
        gids = {text.get_gid() for axis in figure.axes for text in axis.texts}
        assert {"cell:2:64:complete", "cell:1:8:cancelled", "cell:tied:16:complete",
                "cell:softmax:8:complete"} <= gids
        assert not any(gid and gid.startswith("cell:4:") for gid in gids)  # no degree-four column
        texts = {text.get_gid(): text.get_text() for axis in figure.axes for text in axis.texts}
        assert texts["cell:1:8:cancelled"] == "CANCELLED"
        # A coordinate with no selected run is shown as not run, not as a result.
        not_run = [gid for gid in gids if gid and gid.startswith("cell:3:64:")]
        assert not_run and "cancelled" not in not_run[0] and "complete" not in not_run[0]
    finally:
        plt.close(figure)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    build(ROOT, tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
