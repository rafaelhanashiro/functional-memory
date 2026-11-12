"""GPU qualification of the blocked backend's causal strips with transformed polynomial inputs.

The causal strips (``strip_dot_product_attention``) train score-transform recipes on the blocked backend; they
are plain JAX and run on B200, B300 and H100 alike. Use ``pytest -m gpu tests/test_score_transform_strips.py``
inside a GPU allocation with ``JAX_PLATFORMS=cuda,cpu`` (acceptance's GPU stage), which gives the FP64
references the host CPU. Every case runs the production dispatch, ``attention_core`` on the blocked backend
with ``blocked_implementation="strips"`` (counted), at production precision: Q, K and V rounded to BF16 inside
the program, as QK RMSNorm and RoPE hand them over; FP32 coefficients; a BF16 cotangent; degree eight, 2,048
tokens in four 512-query strips, d_K = d_V = 64, two heads. Its output and every gradient are compared with an
independent dense FP64 reference written here (every pair, explicit phi(s)^r with phi = logaddexp(0, s) or
s sign(s), JAX FP64 autodiff on the host) from the same rounded inputs: the output and dQ, dK and dV against
their BF16 rounding floors (the reference rounded to BF16), and the FP32 coefficient gradient order by order,
so that no order's error hides behind the others'. The identity transform is the control's backbone.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.core import attention_core

pytestmark = pytest.mark.gpu
DEGREE, TIME, HEADS, WIDTH = 8, 2048, 2, 64
TRANSFORMS = ("identity", "softplus", "abs")
# Score spreads: "unit" scores ~ N(0, 1), the QK RMSNorm scale; "wide" ~ N(0, 9), with a sixth of the scores
# below -3, where softplus is close to e^s, and scores beyond +-12.
SPREADS = {"unit": 1.0, "wide": 3.0}
BF16_LEAVES = ("output", "dq", "dk", "dv")
# Bounds, as for the score-bias strips: BF16 leaves within FLOOR_FACTOR of the BF16 rounding floor in relative
# L2, and no element further from the reference than MAX_FACTOR times BF16's unit roundoff (2^-8) times the
# largest reference magnitude; the FP32 coefficient gradient below SUM_TOLERANCE in every order.
FLOOR_FACTOR = 1.2
MAX_FACTOR = 2.0
SUM_TOLERANCE = 1e-4


@pytest.fixture(autouse=True)
def _requires_a_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation (B200, B300 or H100)")


def _inputs(spread, *, seed):
    """FP32 Q, K, V before their BF16 rounding (not BF16-representable: the program rounds them), signed
    coefficients N(0, 1)/r!, and a BF16 cotangent."""
    rng = np.random.default_rng(seed)
    shape = (1, TIME, HEADS, WIDTH)
    q = SPREADS[spread] * rng.standard_normal(shape)
    k, v = rng.standard_normal(shape), rng.standard_normal(shape)
    c = rng.standard_normal((1, TIME, HEADS, DEGREE + 1)) / np.array([math.factorial(r) for r in range(DEGREE + 1)])
    cotangent = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    return [jnp.asarray(x, jnp.float32) for x in (q, k, v, c)], cotangent


def _strips(transform):
    """The production operation: Q, K, V rounded to BF16, then attention_core on the blocked strips (the identity
    transform through the untransformed call, as every untransformed recipe makes it)."""
    def attend(q, k, v, c):
        keywords = {} if transform == "identity" else {"score_transform": transform}
        return attention_core(*(x.astype(jnp.bfloat16) for x in (q, k, v)), backend="blocked",
                              blocked_implementation="strips", attention_degree=DEGREE,
                              attention_normalization=False, query_polynomial_coefficients=c,
                              attention_coefficient_activation="tanh", **keywords)
    return attend


def _dense(transform, q, k, v, c):
    """Independent all-pairs reference: W(t,i) = sum_r c_r(t) z_ti^r, z = phi(s), every power explicit."""
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    z = (scores if transform == "identity" else jnp.logaddexp(0., scores) if transform == "softplus"
         else scores * jnp.sign(scores))
    rows = lambda x: jnp.swapaxes(x, 1, 2)[..., None]  # [b, t, h] -> [b, h, t, 1]
    weights = sum(rows(c[..., r]) * z ** r for r in range(DEGREE + 1))
    causal = jnp.arange(TIME)[:, None] >= jnp.arange(TIME)[None, :]
    return jnp.einsum("bhti,bihd->bthd", jnp.where(causal, weights, 0.), v)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _rounded(operands):
    """Q, K, V rounded to BF16 as the program rounds them, and the FP32 coefficients, in FP64 NumPy."""
    return [np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32), np.float64) for x in operands[:3]] + [
        np.asarray(operands[3], np.float64)]


def _float64(transform, operands, cotangent):
    """The reference's output and VJP in FP64 on the host CPU, independent of the GPU compiler."""
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(), jax.default_matmul_precision("highest"), jax.default_device(cpu):
        exact = [jax.device_put(x, cpu) for x in _rounded(operands)]
        g = jax.device_put(np.asarray(cotangent.astype(jnp.float32), np.float64), cpu)
        output, gradients = jax.jit(partial(_vjp, partial(_dense, transform)))(exact, g)
        return [np.asarray(output), *(np.asarray(x) for x in gradients)]


def _run(transform, operands, cotangent, monkeypatch):
    """The strips' output and gradients on the GPU (the strips counted), and the FP64 reference's."""
    calls = []
    strips = reference.strip_dot_product_attention

    def counted(*args, **kwargs):
        calls.append(kwargs.get("score_transform", "identity"))
        return strips(*args, **kwargs)

    monkeypatch.setattr(reference, "strip_dot_product_attention", counted)
    output, gradients = jax.jit(partial(_vjp, _strips(transform)))(operands, cotangent)
    assert calls == [transform], "attention_core must reach the causal strips with the transform"
    assert output.dtype == jnp.bfloat16 and gradients[3].dtype == jnp.float32
    actual = [np.asarray(output.astype(jnp.float32)), *(np.asarray(x, np.float32) for x in gradients)]
    return actual, _float64(transform, operands, cotangent)


def _relative_l2(got, want):
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _compare(actual, expected, label):
    """Every leaf against its bound (the coefficient gradient order by order). Returns each leaf's measured
    error: the multiple of its BF16 floor and the largest element error for the BF16 leaves, the largest order's
    relative L2 for the coefficient gradient."""
    measured = {}
    for name, got, want in zip((*BF16_LEAVES, "dc"), actual, expected, strict=True):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        assert got.shape == want.shape and np.isfinite(got).all(), (label, name)
        if name in BF16_LEAVES:
            # The strips return BF16 output and dQ, dK, dV (cotangents of the BF16 rounding of Q, K, V).
            assert np.array_equal(got, np.asarray(jnp.asarray(got, jnp.bfloat16).astype(jnp.float32))), (label, name)
            rounded = np.asarray(jnp.asarray(want, jnp.bfloat16).astype(jnp.float32), np.float64)
            floor = _relative_l2(rounded, want)
            ratio = _relative_l2(got, want) / floor
            assert ratio < FLOOR_FACTOR, f"{label} {name}: {ratio:.3f} times the BF16 floor {floor:.3e}"
            worst = np.max(np.abs(got - want)) / (2 ** -8 * np.max(np.abs(want)))
            assert worst < MAX_FACTOR, f"{label} {name}: largest error {worst:.3f} BF16 roundoffs of the largest value"
            measured[name] = (ratio, worst)
            continue
        worst = 0.
        for order in range(want.shape[-1]):
            error = _relative_l2(got[..., order], want[..., order])
            assert error < SUM_TOLERANCE, f"{label} {name} order {order}: relative L2 {error:.3e}"
            worst = max(worst, error)
        measured[name] = worst
    print(f"\n{label}: " + " ".join(f"{name}={value[0]:.3f}x/{value[1]:.2f}" if isinstance(value, tuple)
                                      else f"{name}={value:.2e}" for name, value in measured.items()))
    return measured


@pytest.mark.parametrize("spread", tuple(SPREADS))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_strips_match_float64_for_every_transform(transform, spread, monkeypatch):
    """Unit and wide score spreads, signed coefficients: the output and dQ, dK, dV within 1.2x their BF16
    rounding floors (no element off by more than 2 x 2^-8 of the largest value), and the FP32 coefficient
    gradient within 1e-4 relative L2 in every order."""
    operands, cotangent = _inputs(spread, seed=11 + 10 * TRANSFORMS.index(transform) + tuple(SPREADS).index(spread))
    q, k = _rounded(operands)[:2]
    scores = np.einsum("bthd,bihd->bhti", q, k)[:, :, np.tril(np.ones((TIME, TIME), bool))] / math.sqrt(WIDTH)
    assert .4 < (scores < 0).mean() < .6
    if spread == "wide":
        assert (scores < -3).mean() > .14 and np.abs(scores).max() > 12
    actual, expected = _run(transform, operands, cotangent, monkeypatch)
    _compare(actual, expected, f"{transform} {spread}")
