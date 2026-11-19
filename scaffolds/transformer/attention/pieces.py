"""Dot-product attention as pieces, applied in one fixed order.

    s_ti = Q_t K_i^T / sqrt(d_K)          scores, after QK normalization, head scale and RoPE
    z_ti = phi(s_ti)                       score transform (score_transform.py; identity: z = s)
    k_ti = K(z_ti)                         kernel: the exponential e^z, or a polynomial in z whose
                                           coefficients are fixed, static, or generated per query
    w_ti = f(s_ti) k_ti D_ti               prefactor f (score_transform.py; none: f = 1) and decay D
                                           (a forget gate: D = exp(L_t - L_i) or its query-distance forms)
    y_t  = sum_i w_ti v_i [/ sum_i w_ti]   unnormalized, or normalized by the causal row sum

Each piece checks only its own settings. The blocked backend (backends/reference.py) is the
reference composition. A fast kernel implements some combinations and declares which, here; a
recipe on a fast backend must lie inside its kernel's declaration, and its acceptance qualifies
that kernel against the blocked backend. The exponential applies the decay inside the exponent,
exp(phi(s) + log D), which is the same weight.
"""

from dataclasses import dataclass

from scaffolds.transformer.attention.score_transform import validate as validate_transform
from scaffolds.transformer.attention.score_transform import validate_prefactor


@dataclass(frozen=True)
class ScoreFunction:
    """The parameter-free score functions of one layer: the transform phi and the prefactor f."""

    transform: str = "identity"
    prefactor: str = "none"

    def __post_init__(self):
        validate_transform(self.transform)
        validate_prefactor(self.prefactor)

    @property
    def identity(self) -> bool:
        """phi(s) = s and f = 1: the scores enter the kernel unchanged and nothing multiplies it."""
        return self.transform == "identity" and self.prefactor == "none"


# Score functions the fused unnormalized exponential implements in its forward and backward passes
# (backends/fused_unnormalized.py): f(s) exp(phi(s)). Its Taylor truncations take the identity only.
FUSED_EXPONENTIAL_TRANSFORMS = ("identity", "softplus", "sigmoid")
FUSED_EXPONENTIAL_PREFACTORS = ("none", "softplus", "sigmoid")


def fused_exponential_problem(score_function: ScoreFunction, *, attention_degree) -> str | None:
    """Why the fused unnormalized kernel does not implement this score function, or None."""
    if score_function.identity:
        return None
    if attention_degree is not None:
        return "the fused Taylor truncations take untransformed scores without a prefactor"
    if score_function.transform not in FUSED_EXPONENTIAL_TRANSFORMS:
        return f"the fused exponential implements the transforms {', '.join(FUSED_EXPONENTIAL_TRANSFORMS)}"
    if score_function.prefactor not in FUSED_EXPONENTIAL_PREFACTORS:
        return f"the fused exponential implements the prefactors {', '.join(FUSED_EXPONENTIAL_PREFACTORS)}"
    return None


def exponential_kernel_name(score_function: ScoreFunction, *, normalized: bool) -> str:
    """The recorded kernel name of f(s) exp(phi(s)): "exp_unnormalized"/"softmax" for the identity."""
    if score_function.identity:
        return "softmax" if normalized else "exp_unnormalized"
    name = "exp" if score_function.transform == "identity" else f"exp_{score_function.transform}"
    if score_function.prefactor != "none":
        name = f"{score_function.prefactor}_times_{name}"
    return f"{name}_normalized" if normalized else f"{name}_unnormalized"
