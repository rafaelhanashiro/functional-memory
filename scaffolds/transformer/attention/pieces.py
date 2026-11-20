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

Coefficients per query: on the blocked backend, static and fixed polynomial coefficients that meet a
forget gate, a score transform, score biases or a prefactor run as every query's equal coefficients
(MultiHeadAttention._coefficients_per_query), so every unnormalized polynomial with RoPE or no positional
encoding takes any of those pieces together (configs.base._blocked_polynomial; attention_core's
composition route). Per-degree score biases shift each order's base, (phi(s) + beta_r)^r. The Mosaic
transformed query-polynomial kernels take a transform and a prefactor together, w = f(s) p(phi(s)), for the
tanh query polynomial and learned static direct coefficients run per query (mosaic_score_transform_problem).
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

# Fast kernels' declarations: each says which recipes its kernels implement (moved from ModelConfig,
# whose methods of the same names delegate here). The blocked backend implements every recipe.


def _base():
    from scaffolds.transformer.configs import base

    return base


def mosaic_supported(model) -> bool:
    """Whether a Mosaic GPU kernel implements this recipe's attention: normalized fixed
    Taylor, tanh query-conditioned polynomial attention, or an unnormalized polynomial with
    static (fixed or learned) coefficients, at its widths and degrees."""
    from scaffolds.transformer.attention.backends.mosaic.api import VALUE_WIDTH, WIDTHS, query_degree_supported

    degree = model.attention_degree
    integer_degree = isinstance(degree, int) and not isinstance(degree, bool)
    if getattr(model, "attention_kernel_basis", "polynomial") != "polynomial":
        return False  # No Mosaic kernel evaluates a B-spline basis.
    if getattr(model, "attention_score_bias", "none") != "none":
        return False  # Score biases run on their own kernels (mosaic_score_bias_supported).
    if getattr(model, "attention_score_transform", "identity") != "identity":
        return False  # Transforms run on their own kernels (mosaic_score_transform_supported).
    if getattr(model, "attention_score_prefactor", "none") != "none":
        return False  # Prefactors run on the transformed kernels too (mosaic_score_transform_supported).
    taylor = (model.attention_normalization and integer_degree and degree >= 1
              and model.attention_coefficients == "fixed" and model.quadratic_mode == "fixed"
              and model.attention_fixed_coefficients is None and model.attention_fixed_square_factors is None
              and model.attention_coefficient_parameterization == "direct"
              and model.attention_coefficient_conditioning == "static")
    query = (not model.attention_normalization and model.attention_coefficient_conditioning in ("query", "query_key")
             and model.attention_coefficient_activation in (*_base().AFFINE_ACTIVATIONS, *_base().PURE_ACTIVATIONS)
             and integer_degree and not model.attention_forget_gate
             and query_degree_supported(degree, model.qk_head_dim, model.max_seq_len))
    # Static coefficients run on the query-coefficient kernel with every query's coefficients equal;
    # with a distance gate they run on the decay kernels instead (mosaic_decay_supported).
    static = (not model.attention_normalization and model.attention_coefficient_conditioning == "static"
              and model.quadratic_mode == "fixed"
              and model.attention_coefficient_parameterization in ("direct", "sum_of_squares")
              and model.attention_fixed_square_factors is None and integer_degree
              and not model.attention_forget_gate
              and query_degree_supported(degree, model.qk_head_dim, model.max_seq_len))
    return (model.attention_mechanism == "dot_product" and (taylor or query or static)
            and getattr(model, "positional_encoding", "rope") == "rope"
            and model.qk_head_dim in WIDTHS and model.value_head_dim == VALUE_WIDTH)


def mosaic_decay_supported(model) -> bool:
    """Whether the Mosaic GPU decay kernels implement this recipe: tanh or sigmoid
    query-conditioned unnormalized polynomial attention with an input-form query-distance
    forget gate (query or endpoint query-key rates) or per-order cumulative query gates, or
    learned static coefficients with an input-free gate (the query-gate kernels with every
    query's coefficients and rates equal) or the per-order query-key gate (every query's
    coefficients equal), with RoPE or no positional encoding.
    Shared gates must fit B300's kernels (and so H100's); per-order gates must fit H100's, and
    B300 refuses the per-order shapes it cannot hold, and every per-order cumulative gate, when
    the model is traced (``query_degree_supported``).
    Only an explicit ``mosaic`` backend selects them; ``auto`` keeps its historical choice."""
    from scaffolds.transformer.attention.backends.mosaic.api import VALUE_WIDTH, WIDTHS, query_degree_supported

    degree = model.attention_degree
    integer_degree = isinstance(degree, int) and not isinstance(degree, bool)
    if getattr(model, "attention_score_bias", "none") != "none":
        return False  # Score biases run on their own kernels (mosaic_score_bias_supported).
    if getattr(model, "attention_score_transform", "identity") != "identity":
        return False  # Transforms take no forget gate.
    if getattr(model, "attention_score_prefactor", "none") != "none":
        return False  # Nor do prefactors on these kernels.
    input_free = model.attention_forget_gate_conditioning == "none"
    static = (model.attention_coefficient_conditioning == "static" and model.attention_coefficients == "learned"
              and model.attention_coefficient_parameterization == "direct")
    query = (model.attention_coefficient_conditioning == "query"
             and model.attention_coefficient_activation in (*_base().AFFINE_ACTIVATIONS, *_base().PURE_ACTIVATIONS))
    coefficients = static if input_free else query or (static and model.attention_forget_gate_conditioning == "query_key")
    return (model.attention_mechanism == "dot_product" and not model.attention_normalization
            and getattr(model, "attention_kernel_basis", "polynomial") == "polynomial"
            and coefficients
            and model.attention_forget_gate
            and (model.attention_forget_gate_mode == "query_distance"
                 or model.attention_forget_gate_mode == "cumulative" and not input_free)
            and model.attention_forget_gate_form == "input" and integer_degree
            and query_degree_supported(degree, model.qk_head_dim, model.max_seq_len,
                                       gate="query" if input_free else model.attention_forget_gate_conditioning,
                                       per_order=model.attention_forget_gate_per_order,
                                       mode=model.attention_forget_gate_mode)
            and model.positional_encoding in ("rope", "none")
            and model.qk_head_dim in WIDTHS and model.value_head_dim == VALUE_WIDTH)


def mosaic_score_bias_supported(model) -> bool:
    """Whether the Mosaic GPU score-bias kernels implement this recipe: per-degree score biases of
    the tanh query-coefficient polynomial, with no forget gate or the per-order query-key endpoint
    gate (``mosaic_score_bias_problem``). Only an explicit ``mosaic`` backend selects them; ``auto``
    never does, and score-bias recipes reject it."""
    return mosaic_score_bias_problem(model) is None


def mosaic_score_bias_problem(model) -> str | None:
    """Why the Mosaic score-bias kernels cannot run this recipe, or None. They take Q/K widths 4-128,
    value width 64, and degree 1-12 on H100, where their forward, dQ and dK/dV kernels must fit in
    shared memory (``score_bias_supported``); their entry points refuse other GPUs before tracing.
    Backend resolution checks the sequence length (a multiple of 128 rows)."""
    from scaffolds.transformer.attention.backends.mosaic.api import VALUE_WIDTH, WIDTHS

    if getattr(model, "attention_score_bias", "none") == "none":
        return "the recipe has no score biases"
    problem = _base()._score_bias_problem(model)
    if problem is None and model.qk_head_dim not in WIDTHS:
        problem = f"Q/K width {model.qk_head_dim} is not one of {WIDTHS}"
    if problem is None and model.value_head_dim != VALUE_WIDTH:
        problem = f"value width {model.value_head_dim} is not {VALUE_WIDTH}"
    if problem is None:
        from scaffolds.transformer.attention.backends.mosaic.score_bias import (
            HOPPER_TIME_MULTIPLE, SHIFT_OF_MODE, implementation_name, score_bias_supported,
        )

        gate = "endpoint" if model.attention_forget_gate else "none"  # the per-order query-key gate
        shift = SHIFT_OF_MODE.get(model.attention_score_bias)  # row (static, query) or pair (query_key)
        # Shared memory does not depend on the length; ask at the recipe's length in whole 128-row tiles.
        rows = -(-model.max_seq_len // HOPPER_TIME_MULTIPLE) * HOPPER_TIME_MULTIPLE
        if shift is None:
            problem = f"no kernel takes score bias {model.attention_score_bias!r}"
        elif not score_bias_supported(model.attention_degree, model.qk_head_dim, rows, gate=gate, shift=shift):
            problem = (f"H100's {implementation_name(gate, shift)} kernels do not hold degree "
                       f"{model.attention_degree} at Q/K width {model.qk_head_dim} in shared memory")
    return problem


def mosaic_score_transform_supported(model) -> bool:
    """Whether the Mosaic transformed query-polynomial kernels implement this recipe: the tanh query
    polynomial, a learned static direct polynomial or the fixed pointwise kernel evaluated at a transformed score
    and optionally multiplied by a prefactor (``mosaic_score_transform_problem``). Only an explicit ``mosaic``
    backend selects them; ``auto`` never does, and transformed or prefactored recipes reject it."""
    return mosaic_score_transform_problem(model) is None


def mosaic_score_transform_problem(model) -> str | None:
    """Why the Mosaic transformed query-polynomial kernels cannot run this recipe, or None. They are the
    query polynomial's kernels with phi(s) and f(s) evaluated in their tiles (W = f(s) p(phi(s)),
    ``common.score_function_family``: any transform without a prefactor; a softplus, sigmoid, exp or swish
    prefactor with the identity or softplus transform), on H100 and B200, so they take the recipes those kernels
    take (``mosaic_supported``'s query case: RoPE, Q/K widths 4-128, value width 64, the degrees
    ``query_degree_supported`` fits at the recipe's length; with a prefactor, the GPU-qualified degrees up to
    ``MAX_PREFACTOR_QUERY_DEGREE``) with diagonal tanh query coefficients, learned static direct coefficients
    (every query's equal) or fixed degree-one (0, 1) coefficients and no forget gate, score biases or key
    factors. Backend resolution checks the GPU and length."""
    from scaffolds.transformer.attention.backends.mosaic.api import (
        MAX_PREFACTOR_QUERY_DEGREE, VALUE_WIDTH, WIDTHS, query_degree_supported,
    )
    from scaffolds.transformer.attention.backends.mosaic.common import SCORE_FUNCTION_FAMILIES

    transform = getattr(model, "attention_score_transform", "identity")
    prefactor = getattr(model, "attention_score_prefactor", "none")
    if transform == "identity" and prefactor == "none":
        return "the recipe has no score transform or prefactor"
    if (transform, prefactor) not in SCORE_FUNCTION_FAMILIES:
        return (f"no kernel family evaluates the {transform} transform with the {prefactor} prefactor; prefactors "
                "take the identity or softplus transform")
    problem = _base()._transformed_coefficient_problem(model)
    if problem is None and model.positional_encoding != "rope":
        problem = "the Mosaic query-polynomial kernels run RoPE recipes"
    if problem is None and (model.attention_forget_gate or model.attention_score_bias != "none"):
        problem = "it takes no forget gate or score biases"
    if problem is None and model.qk_head_dim not in WIDTHS:
        problem = f"Q/K width {model.qk_head_dim} is not one of {WIDTHS}"
    if problem is None and model.value_head_dim != VALUE_WIDTH:
        problem = f"value width {model.value_head_dim} is not {VALUE_WIDTH}"
    if problem is None and prefactor != "none" and model.attention_degree > MAX_PREFACTOR_QUERY_DEGREE:
        problem = (f"the prefactor kernels are GPU-qualified at degrees up to {MAX_PREFACTOR_QUERY_DEGREE}, not "
                   f"{model.attention_degree}; the blocked backend takes any degree")
    if problem is None and not query_degree_supported(model.attention_degree, model.qk_head_dim, model.max_seq_len,
                                                      transform=transform, prefactor=prefactor):
        problem = (f"the kernels do not hold degree {model.attention_degree} at Q/K width {model.qk_head_dim} "
                   f"and {model.max_seq_len} tokens in shared memory")
    return problem


def path_fused_supported(model) -> bool:
    """Whether FLA's fused PaTH kernels cover this recipe's widths and degree (any GPU, length)."""
    from scaffolds.transformer.attention.backends.fla_path_attention import HEAD_DIMS, MAX_DEGREE

    degree = model.attention_degree
    return (model.positional_encoding == "path" and model.qk_head_dim in HEAD_DIMS
            and model.value_head_dim == model.qk_head_dim
            and (degree is None or (isinstance(degree, int) and 1 <= degree <= MAX_DEGREE)))


def path_kernel(model) -> str | None:
    """The PaTH attention kernel this recipe uses, or None for RoPE recipes."""
    if model.positional_encoding != "path":
        return None
    if model.attention_degree is None:
        return "softmax" if model.attention_normalization else "exp"
    return "query_polynomial"
