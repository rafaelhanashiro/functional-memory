"""Transformed polynomial inputs of tanh query-coefficient polynomial attention.

Every order of the polynomial reads a parameter-free function of the score instead of the score
itself. With s_ti = Q_t K_i^T / sqrt(d_K) (after QK normalization, head scale and RoPE),

    W(t,i) = sum_(r=0..m) c_r(x_t) z_ti^r,    z_ti = phi(s_ti),

where c_r keeps its generator and factorial scales and phi is

- ``identity``: z = s (every historical recipe; the kernels compute exactly as before);
- ``softplus``: z = log(1 + exp(s)), positive and increasing, z -> 0 as s -> -inf and z - s -> 0 as
  s -> +inf;
- ``abs``: z = |s|.

With c_0 = 0 and c_r = 1/r! for r >= 1 (a Taylor start with ``attention_coefficient_constant_initial:
0``), the softplus kernel sum_(r=1..m) z^r/r! is the degree-m Taylor polynomial of e^z - 1 at z, and
e^z - 1 = e^s exactly: it is about e^s for negative scores, so a key with a very negative score gets
a weight near zero, and about T_m(s) - 1 for large positive scores. Without the transform, a very
negative score gives a large weight of the sign of (-1)^m.

Softplus is evaluated as max(s, 0) + log1p(exp(-|s|)), finite for every finite s, with derivative
sigmoid(s); abs has derivative sign(s), zero at s = 0. Both derivatives are their JVP rules: JAX's
own derivative of |s| at zero is 1, which would also make the autodiff derivative of the softplus
formula 0 at s = 0 instead of 1/2. The blocked backend (``backends/reference.py``: the scan and the causal
strips) and the Mosaic transformed query-polynomial kernels (``backends/mosaic/api.py``
``transformed_query_polynomial_attention``, the same derivatives at s = 0) evaluate the transform; every
other backend and kernel refuses a transformed recipe instead of computing the untransformed polynomial.

``attention_coefficient_zero_constant`` structurally omits direct c_0 (query or
static) or both query-SoS factor constants. Static softplus coefficients are
broadcast over queries into the same transformed-query kernel; its coefficient
VJP sums over that broadcast. No query generator is added to the static model.
"""

import jax
import jax.numpy as jnp

TRANSFORMS = ("identity", "softplus", "abs")
# Formulas recorded in the attention identity.
FORMULAS = {"softplus": "z_ti = softplus(s_ti) = log(1 + exp(s_ti)), evaluated as max(s_ti, 0) + log1p(exp(-|s_ti|))",
            "abs": "z_ti = |s_ti|"}


@jax.custom_jvp
def softplus(scores):
    """log(1 + exp(s)) as max(s, 0) + log1p(exp(-|s|)); derivative sigmoid(s)."""
    return jnp.maximum(scores, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(scores)))


@softplus.defjvp
def _softplus_jvp(primals, tangents):
    (scores,), (tangent,) = primals, tangents
    return softplus(scores), jax.nn.sigmoid(scores) * tangent


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
    return softplus(scores) if transform == "softplus" else absolute(scores)
