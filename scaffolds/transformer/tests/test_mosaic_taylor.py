"""GPU correctness gates for the Mosaic GPU Taylor attention kernels (H100 and B300).

Use ``pytest -m gpu tests/test_mosaic_taylor.py`` inside an H100 or B300 allocation.
Outputs and q/k/v VJPs are compared with the independent FP64 dense reference and with
the Pallas-Triton kernel, using the BF16 tolerances of test_fused_attention.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.fused_taylor import fused_polynomial_attention
from scaffolds.transformer.attention.backends.mosaic.taylor import WIDTHS, taylor_attention
from scaffolds.transformer.tests.test_fused_attention import _assert_result, _compiled_forward_and_vjp, _inputs

DEGREES = (1, 2, 3, 4, 8, 16)
TOLERANCES = dict(output_rtol=0.006, gradient_rtols=(0.025, 0.025, 0.015))
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _numpy_fp64_forward_and_vjp(operands, cotangent, degree):
    """Independent FP64 output and q/k/v VJP of normalized causal Taylor attention (NumPy)."""
    q, k, v, do = (np.asarray(x.astype(jnp.float32), np.float64) for x in (*operands, cotangent))
    time, width = q.shape[1], q.shape[-1]
    scores = np.einsum("bthd,bshd->bhts", q, k) / math.sqrt(width)
    causal = np.tril(np.ones((time, time), bool))
    kernel = sum(scores ** r / math.factorial(r) for r in range(degree + 1))
    slope = sum(scores ** r / math.factorial(r) for r in range(degree))
    kernel, slope = np.where(causal, kernel, 0.0), np.where(causal, slope, 0.0)
    z = kernel.sum(-1)
    weights = kernel / z[..., None]
    out = np.einsum("bhts,bshd->bthd", weights, v)
    do_h = do.transpose(0, 2, 1, 3)
    delta = np.einsum("bhtd,bthd->bht", do_h, out)
    ds = (np.einsum("bhtd,bshd->bhts", do_h, v) - delta[..., None]) * slope / z[..., None]
    grad_q = np.einsum("bhts,bshd->bthd", ds, k) / math.sqrt(width)
    grad_k = np.einsum("bhts,bthd->bshd", ds, q) / math.sqrt(width)
    grad_v = np.einsum("bhts,bhtd->bshd", weights, do_h)
    return out, (grad_q, grad_k, grad_v)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("degree", DEGREES)
def test_outputs_and_vjp_match_fp64_and_triton(degree, width):
    operands, cotangent = _inputs((1, 512, 2, width, 64), dtype=jnp.bfloat16, seed=94)
    mosaic = _compiled_forward_and_vjp(partial(taylor_attention, attention_degree=degree))(operands, cotangent)
    assert mosaic[0].dtype == jnp.bfloat16 and all(g.dtype == jnp.bfloat16 for g in mosaic[1])
    dense64 = _numpy_fp64_forward_and_vjp(operands, cotangent, degree)
    _assert_result(mosaic, dense64, **TOLERANCES, label=f"Mosaic vs FP64, degree={degree}, width={width}")
    triton = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))(operands, cotangent)
    _assert_result(mosaic, triton, **TOLERANCES, label=f"Mosaic vs Triton, degree={degree}, width={width}")


@pytest.mark.parametrize("degree", (2, 3, 16))
def test_production_context_matches_triton(degree):
    operands, cotangent = _inputs((1, 2048, 4, 64, 64), dtype=jnp.bfloat16, seed=95)
    mosaic = _compiled_forward_and_vjp(partial(taylor_attention, attention_degree=degree))(operands, cotangent)
    triton = _compiled_forward_and_vjp(partial(fused_polynomial_attention, attention_degree=degree))(operands, cotangent)
    _assert_result(mosaic, triton, **TOLERANCES, label=f"production context, degree={degree}")


@pytest.mark.parametrize("degree", (1, 2))
def test_outputs_and_vjp_ignore_future_tokens(degree):
    operands, cotangent = _inputs((1, 512, 2, 64, 64), dtype=jnp.bfloat16, seed=96)
    prefix = 300
    attend = _compiled_forward_and_vjp(partial(taylor_attention, attention_degree=degree))
    restricted = cotangent.at[:, prefix:].set(0)
    output, grads = attend(operands, restricted)
    q, k, v = operands
    changed = (q.at[:, prefix:].multiply(-2), k.at[:, prefix:].multiply(3), v.at[:, prefix:].add(2))
    changed_output, changed_grads = attend(changed, restricted)
    np.testing.assert_array_equal(np.asarray(changed_output[:, :prefix]), np.asarray(output[:, :prefix]))
    for name, grad, changed_grad in zip("qkv", grads, changed_grads):
        np.testing.assert_array_equal(np.asarray(changed_grad[:, :prefix]), np.asarray(grad[:, :prefix]), err_msg=name)


def test_rows_with_a_zero_signed_sum_are_nan():
    # Degree 1 with s = -1 everywhere gives p(s) = 0 and a zero row sum.
    width = 64
    q = jnp.zeros((1, 512, 1, width), jnp.bfloat16).at[..., 0].set(-8.0)
    k = jnp.zeros((1, 512, 1, width), jnp.bfloat16).at[..., 0].set(1.0)
    v = jnp.ones((1, 512, 1, 64), jnp.bfloat16)
    output = taylor_attention(q, k, v, attention_degree=1)
    assert bool(jnp.isnan(output).all())


@pytest.mark.parametrize("change", ["float32", "width", "value width", "time", "degree"])
def test_unsupported_inputs_raise(change):
    q = k = jnp.zeros((1, 512, 1, 64), jnp.bfloat16)
    v = jnp.zeros((1, 512, 1, 64), jnp.bfloat16)
    degree = 2
    if change == "float32":
        q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    elif change == "width":
        q = k = jnp.zeros((1, 512, 1, 48), jnp.bfloat16)
    elif change == "value width":
        v = jnp.zeros((1, 512, 1, 128), jnp.bfloat16)
    elif change == "time":
        q = k = v = jnp.zeros((1, 200, 1, 64), jnp.bfloat16)
    else:
        degree = 0
    with pytest.raises(ValueError):
        taylor_attention(q, k, v, attention_degree=degree)
