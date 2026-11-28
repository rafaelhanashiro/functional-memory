import pytest

from experiments.transformer_pp.staging.static_taylor_reset.report import (
    BEGIN, END, HERE, generated_tables, paired_loss,
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
