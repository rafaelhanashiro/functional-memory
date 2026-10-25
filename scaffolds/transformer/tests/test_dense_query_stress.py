"""Validation-only stress case for dense query coefficients; production acceptance excludes it.

Mosaic's canonical BF16 weight precision rounds the degree-2 coefficient-bias gradient more
than the blocked BF16 path once coefficients are strongly signed: on B300 and H100, 7.6-8.3%
relative L2 against FP32 versus 1.9-3.2% (validation/README.md, dense query K). The production
start and moderate coefficients pass in test_dense_query_readout.py.
"""
import pytest

from scaffolds.transformer.tests.test_dense_query_readout import check_gradients_within_bf16_rounding


@pytest.mark.gpu
@pytest.mark.xfail(strict=False, reason="degree-2 coefficient-bias gradient under strongly signed coefficients "
                   "exceeds 2.5x blocked BF16 rounding")
@pytest.mark.parametrize("features", (3, 9))
def test_gpu_degree_two_strongly_signed_coefficients(features):
    check_gradients_within_bf16_rounding(2, features, 0.3)
