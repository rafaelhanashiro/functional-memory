import json

import pytest

from experiments.transformer_pp.staging.polynomial_gradient_dynamics.report import (
    BEGIN, END, HERE, dynamics_table, generated_tables, paired_loss,
)


def test_paired_interval_uses_within_sequence_differences():
    mean, lo, hi = paired_loss([2, 11, 101], [1, 10, 100])
    assert (mean, lo, hi) == (1, 1, 1)
    mean, lo, hi = paired_loss([2, 4, 6], [1, 2, 3])
    assert mean == 2
    assert (lo, hi) == pytest.approx((0.8684143, 3.1315857))


def test_published_tables_reproduce():
    committed = (HERE / "README.md").read_text().split(BEGIN)[1].split(END)[0].strip()
    assert generated_tables() == committed


def test_dynamics_table_preserves_factorial_units_and_uses_all_heads(tmp_path):
    static = {"raw": [[[1, 1, 3], [1, 1, 5]]]}
    query = {"effective_summary": {
        "mean": [[[1, 0.8, 0.65], [1.2, 1.4, 0.8]]],
        "std": [[[0.1, 0.2, 0.2], [0.3, 0.5, 0.3]]],
    }}
    runs = {}
    for role, coefficients in (("learned_static_parent", static), ("static_dynamics", static),
                               ("query_dynamics", query)):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(json.dumps({"step": 400, "attention_diagnostics": {"coefficients": coefficients}}) + "\n")
        runs[role] = {"run_id": role, "sources": {"metrics.jsonl": {"path": path.name}}}
    result = dynamics_table(runs, tmp_path, [400], [2])
    assert result.splitlines()[-1] == "| 400 | 4.000 / 4.000 | 0.250 | 0.350 |"
