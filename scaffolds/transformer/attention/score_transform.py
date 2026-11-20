"""Parameter-free score functions: the transform phi and the prefactor f (attention/pieces.py).

The kernel reads a parameter-free function of the score instead of the score itself, and its
weight may be multiplied by a second one. With s_ti = Q_t K_i^T / sqrt(d_K) (after QK
normalization, head scale and RoPE),

    W(t,i) = f(s_ti) K(z_ti),    z_ti = phi(s_ti),

where K is the kernel: the exponential e^z, or a polynomial sum_(r=0..m) c_r(x_t) z^r. The prefactor
f is "none" (f = 1, every historical recipe), softplus, sigmoid, swish or exp, the first three evaluated by the
same functions and derivatives as the transforms below and exp as e^s; for the exponential, f(s) e^s with
f = sigmoid or softplus is e^s / (1 + e^-s) or log(1 + e^s) e^s. For polynomials,

    W(t,i) = f(s_ti) sum_(r=0..m) c_r(x_t) z_ti^r,    z_ti = phi(s_ti),

Query coefficients keep their generator and factorial scales. Fixed degree-one coefficients [0, 1]
instead give W(t,i) = phi(s_ti), with no coefficient parameters. The transforms are

- ``identity``: z = s (every historical recipe; the kernels compute exactly as before);
- ``softplus``: z = log(1 + exp(s)), positive and increasing, z -> 0 as s -> -inf and z - s -> 0 as
  s -> +inf;
- ``sigmoid``: z = 1 / (1 + exp(-s)), bounded between zero and one for finite scores;
- ``abs``: z = |s|;
- ``swish``: z = s sigmoid(s), signed: its minimum is about -0.2785 at s = -1.2785, z -> 0 as s -> -inf
  and z - s -> 0 as s -> +inf.

With c_0 = 0 and c_r = 1/r! for r >= 1 (a Taylor start with ``attention_coefficient_constant_initial:
0``), the softplus kernel sum_(r=1..m) z^r/r! is the degree-m Taylor polynomial of e^z - 1 at z, and
e^z - 1 = e^s exactly: it is about e^s for negative scores, so a key with a very negative score gets
a weight near zero, and about T_m(s) - 1 for large positive scores. Without the transform, a very
negative score gives a large weight of the sign of (-1)^m.

Softplus is evaluated as max(s, 0) + log1p(exp(-|s|)), finite for every finite s, with derivative
sigmoid(s); abs has derivative sign(s), zero at s = 0. Both derivatives are their JVP rules: JAX's
own derivative of |s| at zero is 1, which would also make the autodiff derivative of the softplus
formula 0 at s = 0 instead of 1/2. Sigmoid's JVP preserves positive-tail gradients after its value rounds
to one, with derivative 1/4 at zero; nonfinite scores remain invalid rather than saturating to finite
weights. Swish shares sigmoid's stable logistic: its derivative sigmoid(s) + s e^-|s| / (1 + e^-|s|)^2 is 1/2 at
zero and tends to 1 as s -> +inf and to 0 as s -> -inf, and a nonfinite score is NaN, value and derivative. The
exp prefactor is e^s with JAX's exact derivative e^s; a nonfinite causal score still makes its row NaN, since
the polynomial (or the kernel) of that score is nonfinite. The blocked backend (``backends/reference.py``: the
scan and the causal strips) and the Mosaic transformed query-polynomial kernels (``backends/mosaic/api.py``
``transformed_query_polynomial_attention``, the same derivatives at s = 0, and the prefactor f with the product
rule f'(s) p(z) + f(s) p'(z) phi'(s)) evaluate the score functions; every other backend and kernel refuses a
transformed or prefactored recipe instead of computing the untransformed polynomial.

``attention_coefficient_zero_constant`` structurally omits direct c_0 (query or
static) or both query-SoS factor constants. Static direct softplus coefficients, with or
without that zero constant and with any prefactor, are broadcast over queries into the same
transformed-query kernel; its coefficient VJP sums over that broadcast. No query generator
is added to the static model. With f = phi = softplus the weight z p(z) is the input power
of a SoS core (``attention_coefficient_input_power``) for direct coefficients too, without
a structural zero coefficient: the two routes agree in exact arithmetic.
"""

import jax
import jax.numpy as jnp

TRANSFORMS = ("identity", "softplus", "abs", "sigmoid", "swish")
PREFACTORS = ("none", "softplus", "sigmoid", "exp", "swish")
# Formulas recorded in the attention identity.
FORMULAS = {"softplus": "z_ti = softplus(s_ti) = log(1 + exp(s_ti)), evaluated as max(s_ti, 0) + log1p(exp(-|s_ti|))",
            "abs": "z_ti = |s_ti|",
            "sigmoid": "z_ti = sigmoid(s_ti) = 1 / (1 + exp(-s_ti)), evaluated with exp(-|s_ti|)",
            "swish": "z_ti = swish(s_ti) = s_ti sigmoid(s_ti), sigmoid evaluated with exp(-|s_ti|)"}
PREFACTOR_FORMULAS = {
    "softplus": "f(s_ti) = softplus(s_ti) = log(1 + exp(s_ti)), evaluated as max(s_ti, 0) + log1p(exp(-|s_ti|))",
    "sigmoid": "f(s_ti) = sigmoid(s_ti) = 1 / (1 + exp(-s_ti)), evaluated with exp(-|s_ti|)",
    "exp": "f(s_ti) = exp(s_ti)",
    "swish": "f(s_ti) = swish(s_ti) = s_ti sigmoid(s_ti), sigmoid evaluated with exp(-|s_ti|)"}
# Score functions whose values can be negative for finite scores: a weight they enter is signed.
SIGNED_FUNCTIONS = ("swish",)


@jax.custom_jvp
def softplus(scores):
    """log(1 + exp(s)) as max(s, 0) + log1p(exp(-|s|)); derivative sigmoid(s)."""
    return jnp.maximum(scores, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(scores)))


@softplus.defjvp
def _softplus_jvp(primals, tangents):
    (scores,), (tangent,) = primals, tangents
    return softplus(scores), jax.nn.sigmoid(scores) * tangent


@jax.custom_jvp
def sigmoid(scores):
    """Stable logistic value; nonfinite scores remain invalid attention inputs.

    The derivative uses exp(-abs(s))/(1 + exp(-abs(s)))**2, rather than z*(1-z): positive-tail
    gradients remain representable even after the value itself rounds to one.
    """
    tail = jnp.exp(-jnp.abs(scores))
    value = jnp.where(scores >= 0.0, 1.0, tail) / (1.0 + tail)
    return jnp.where(jnp.isfinite(scores), value, jnp.nan)


@sigmoid.defjvp
def _sigmoid_jvp(primals, tangents):
    (scores,), (tangent,) = primals, tangents
    tail = jnp.exp(-jnp.abs(scores))
    slope = tail / (1.0 + tail) ** 2
    return sigmoid(scores), jnp.where(jnp.isfinite(scores), slope, jnp.nan) * tangent


@jax.custom_jvp
def swish(scores):
    """s sigmoid(s) with sigmoid's stable logistic; nonfinite scores remain invalid attention inputs (NaN)."""
    tail = jnp.exp(-jnp.abs(scores))
    value = scores * (jnp.where(scores >= 0.0, 1.0, tail) / (1.0 + tail))
    return jnp.where(jnp.isfinite(scores), value, jnp.nan)


@swish.defjvp
def _swish_jvp(primals, tangents):
    """sigmoid(s) + s e^-|s| / (1 + e^-|s|)^2: 1/2 at zero, tending to 1 as s -> +inf and to 0 as s -> -inf."""
    (scores,), (tangent,) = primals, tangents
    tail = jnp.exp(-jnp.abs(scores))
    inverse = 1.0 / (1.0 + tail)
    slope = jnp.where(scores >= 0.0, 1.0, tail) * inverse + scores * tail * inverse * inverse
    return swish(scores), jnp.where(jnp.isfinite(scores), slope, jnp.nan) * tangent


def exponential(scores):
    """e^s, the exp prefactor; JAX's derivative e^s is exact. It overflows to +inf above about 88.7 in FP32,
    which makes the weight, and so its row, nonfinite."""
    return jnp.exp(scores)


@jax.custom_jvp
def absolute(scores):
    """|s|; derivative sign(s), zero at s = 0."""
    return jnp.abs(scores)


@absolute.defjvp
def _absolute_jvp(primals, tangents):
    (scores,), (tangent,) = primals, tangents
    return jnp.abs(scores), jnp.sign(scores) * tangent


def validate(transform) -> str:
    if not isinstance(transform, str) or transform not in TRANSFORMS:
        raise ValueError(f"Unknown attention score transform: {transform!r}")
    return transform


def apply(scores, transform: str):
    """phi(scores) elementwise, in the scores' dtype. The identity returns ``scores`` itself: no operation."""
    if validate(transform) == "identity":
        return scores
    if transform == "sigmoid":
        return sigmoid(scores)
    if transform == "swish":
        return swish(scores)
    return softplus(scores) if transform == "softplus" else absolute(scores)


def validate_prefactor(prefactor) -> str:
    if not isinstance(prefactor, str) or prefactor not in PREFACTORS:
        raise ValueError(f"Unknown attention score prefactor: {prefactor!r}")
    return prefactor


def prefactor(scores, kind: str):
    """f(scores) elementwise in the scores' dtype, or None for "none" (no multiplication at all)."""
    if validate_prefactor(kind) == "none":
        return None
    return {"sigmoid": sigmoid, "softplus": softplus, "exp": exponential, "swish": swish}[kind](scores)
