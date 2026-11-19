"""Regenerate the isolated SoS training diagnostic from pinned run measurements."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs
from plot_gradients import plot as plot_gradients
from plot_coefficients import plot as plot_coefficients
from plot_head_parameters import plot as plot_head_parameters


def main():
    selection = yaml.safe_load((OUT / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    assert selection["settings"]["late_training_start_exclusive"] == 1500
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((run / "config.yaml").read_text())
        model = config["model"]
        assert load_run(run)["status"] == "complete", entry["run_id"]
        assert model["attention_coefficient_parameterization"] == "sum_of_squares"
        assert model["attention_coefficient_conditioning"] == "query"
        assert model["attention_degree"] == entry["degree"]
        assert model["initialization"] != "flax_default"
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert rows[-1]["step"] == selection["settings"]["terminal_step"]
        assert all(row["attention_diagnostics"]["valid"] for row in rows)
        for key, value in selection["settings"]["model_shape"].items():
            assert model[key] == value, (entry["run_id"], key)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    plot_gradients(ROOT, OUT, selection)
    plot_coefficients(ROOT, OUT, selection)
    plot_head_parameters(ROOT, OUT, selection)
    print("Regenerated SoS training dynamics from pinned, completed runs.")


if __name__ == "__main__":
    main()
