import json
import math
from statistics import fmean

import pytest
import yaml

from experiments.transformer_pp.results.diagnostics.input_dependence.calibration_mean import summarize


def test_summary_reproduces_on_the_same_held_out_sequences(tmp_path):
    output = tmp_path / "summary.csv"
    summarize.write_summary(output)
    assert output.read_bytes() == (summarize.TOPIC / "summary.csv").read_bytes()

    selection = yaml.safe_load((summarize.TOPIC / "selection.yaml").read_text())
    settings = selection["settings"]
    first = settings["calibration_sequences"]
    evaluation_rows = list(range(first, first + settings["evaluation_sequences"]))
    for summary in summarize.collect():
        run = summarize.ROOT / "experiments/transformer_pp/runs" / summary["run_id"]
        metadata = json.loads((run / "metadata.json").read_text())
        assert metadata["context_length"] == settings["context_length"]
        assert metadata["calibration_row_range"] == [0, first]  # upper bound is exclusive
        assert sorted(metadata["recipient_rows"]) == evaluation_rows
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        for condition, field in (("native", "native_ppl"), ("calibration_mean", "mean_ppl")):
            sample, = [r["sequence_nll"] for r in records
                       if r["condition"] == condition and r["layer_scope"] == "all_layers"]
            assert summary[field] == pytest.approx(math.exp(fmean(sample)), rel=1e-12)
