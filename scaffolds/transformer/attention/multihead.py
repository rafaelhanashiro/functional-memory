"""Shared Q/K/V projections, positional encoding, and output projection."""

from functools import partial
from typing import Literal
import math

import jax
import jax.numpy as jnp
from flax import nnx

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.output_scaling import (
    HEAD_RMS_PRESCALES, OUTPUT_SCALINGS, scale_attention_output,
)
from scaffolds.transformer.attention.qk_norm import (
    TOKEN_SCALE_PARAMETERS, cosine_score_operands, rms_score_operands, scale_rms_queries, token_log_scale,
)
from scaffolds.transformer.attention.rational_polynomial import DENOMINATOR_KINDS
from scaffolds.transformer.attention.spline import (
    QUERY_PARAMETERS as SPLINE_QUERY_PARAMETERS, STATIC_PARAMETERS as SPLINE_STATIC_PARAMETERS, validate_knots,
)
from scaffolds.transformer.attention.taylor import _validate_degree
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.training import telemetry

AttentionBackend = Literal["blocked", "fused", "manual", "auto", "xla", "cudnn"]


def attention_from_config(config, *, rngs: nnx.Rngs):
    """Construct the configured module without exposing mechanism fields to blocks."""
    if config.attention_mechanism == "gla":
        from scaffolds.transformer.attention.gla import GatedLinearAttention

        return GatedLinearAttention(
            config.hidden_size, config.num_heads,
            config.num_heads * config.qk_head_dim, config.num_heads * config.value_head_dim,
            gate_rank=config.gla_gate_rank, gate_tau=config.gla_gate_tau,
            chunk_size=config.gla_chunk_size, inner_chunk_size=config.gla_inner_chunk_size,
            rope_theta=config.rope_theta, rms_norm_eps=config.rms_norm_eps,
            dropout=config.dropout, use_rope=config.gla_use_rope,
            backend=config.attention_backend, rngs=rngs,
        )
    if config.attention_mechanism == "gated_deltanet":
        from scaffolds.transformer.attention.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet(
            config.hidden_size, config.num_heads, config.qk_head_dim, config.value_head_dim,
            conv_size=config.gdn_conv_size, norm_eps=config.rms_norm_eps,
            backend=config.attention_backend, erase_mode=config.gdn_erase_mode,
            write_activation=config.gdn_write_activation,
            polynomial_degree=config.gdn_polynomial_degree, rngs=rngs,
        )
    return MultiHeadAttention(
        config.hidden_size, config.num_heads, config.num_kv_heads,
        config.qk_head_dim, config.value_head_dim, rope_theta=config.rope_theta,
        dropout=config.dropout, backend=config.attention_backend,
        attention_degree=config.attention_degree, quadratic_mode=config.quadratic_mode,
        attention_normalization=config.attention_normalization,
        attention_fixed_coefficients=config.attention_fixed_coefficients,
        attention_coefficient_conditioning=config.attention_coefficient_conditioning,
        attention_coefficient_parameterization=config.attention_coefficient_parameterization,
        attention_coefficient_input_power=getattr(config, "attention_coefficient_input_power", 0),
        attention_static_sos_parameterization=getattr(config, "attention_static_sos_parameterization", "direct"),
        attention_coefficient_activation=config.attention_coefficient_activation,
        attention_coefficient_input_normalization=config.attention_coefficient_input_normalization,
        attention_coefficient_input_source=config.attention_coefficient_input_source,
        attention_coefficient_readout=config.attention_coefficient_readout,
        attention_coefficient_readout_scale=config.attention_coefficient_readout_scale,
        attention_coefficient_amplitude_scale=config.attention_coefficient_amplitude_scale,
        attention_coefficient_scale=config.attention_coefficient_scale,
        attention_coefficient_orders=config.attention_coefficient_orders,
        attention_coefficient_zero_constant=config.attention_coefficient_zero_constant,
        attention_coefficient_gate=getattr(config, "attention_coefficient_gate", "none"),
        attention_fixed_square_factors=config.attention_fixed_square_factors,
        qk_norm=config.qk_norm, qk_norm_eps=config.qk_norm_eps,
        qk_norm_type=config.qk_norm_type,
        qk_norm_head_scale=config.qk_norm_head_scale,
        qk_norm_token_scale=getattr(config, "qk_norm_token_scale", "none"),
        attention_output_scaling=config.attention_output_scaling, rms_norm_eps=config.rms_norm_eps,
        positional_encoding=config.positional_encoding,
        attention_forget_gate=getattr(config, "attention_forget_gate", False),
        attention_forget_gate_mode=getattr(config, "attention_forget_gate_mode", "cumulative"),
        attention_forget_gate_form=getattr(config, "attention_forget_gate_form", "input"),
        attention_forget_gate_constant_rate=getattr(config, "attention_forget_gate_constant_rate", "learned"),
        attention_forget_gate_conditioning=getattr(config, "attention_forget_gate_conditioning", "query"),
        attention_forget_gate_rate_activation=getattr(config, "attention_forget_gate_rate_activation", "softplus"),
        attention_blocked_implementation=getattr(config, "attention_blocked_implementation", "scan"),
        attention_kernel_basis=getattr(config, "attention_kernel_basis", "polynomial"),
        attention_spline_knots=getattr(config, "attention_spline_knots", None),
        attention_score_bias=getattr(config, "attention_score_bias", "none"),
        attention_score_transform=getattr(config, "attention_score_transform", "identity"),
        attention_score_prefactor=getattr(config, "attention_score_prefactor", "none"),
        score_mlp_width=getattr(config, "attention_score_mlp_width", None),
        score_mlp_architecture=getattr(config, "attention_score_mlp_architecture", "dynamic_readout"),
        score_mlp_form=getattr(config, "attention_score_mlp_form", "kernel"),
        score_mlp_activation=getattr(config, "attention_score_mlp_activation", "silu"),
        attention_exp_shift=getattr(config, "attention_exp_shift", "none"),
        attention_head_rms_prescale=getattr(config, "attention_head_rms_prescale", "none"),
        attention_polynomial_shift=getattr(config, "attention_polynomial_shift", "none"),
        rngs=rngs,
    )


def _forget_gate_diagnostics(log_alpha):
    """Per-head statistics of a forget gate's log alpha [batch, time, heads]."""
    return jax.lax.stop_gradient({
        "forget_gate_log_alpha_mean": jnp.mean(log_alpha, axis=(0, 1)),
        "forget_gate_log_alpha_min": jnp.min(log_alpha, axis=(0, 1)),
        "forget_gate_log_alpha_max": jnp.max(log_alpha, axis=(0, 1)),
    })


class MultiHeadAttention(nnx.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        qk_head_dim: int, value_head_dim: int, rope_theta: float = 10000.0,
        dropout: float = 0.0, backend: AttentionBackend = "blocked",
        attention_degree: int | None = None, quadratic_mode: str = "fixed",
        attention_normalization: bool = True, qk_norm: bool = False,
        qk_norm_eps: float = 1e-6, *, attention_fixed_coefficients: tuple[float, ...] | None = None,
        attention_fixed_square_factors: tuple[float, ...] | None = None,
        attention_output_scaling: str = "none", rms_norm_eps: float = 1e-6,
        attention_coefficient_conditioning: str = "static",
        attention_coefficient_parameterization: str = "direct",
        attention_coefficient_input_power: int = 0,
        attention_static_sos_parameterization: str = "direct",
        attention_coefficient_activation: str = "identity",
        attention_coefficient_input_normalization: str = "none",
        attention_coefficient_input_source: str = "post_rmsnorm",
        attention_coefficient_readout: str = "diagonal",
        attention_coefficient_readout_scale: float = 1.0,
        positional_encoding: str = "rope",
        attention_coefficient_amplitude_scale: str = "none",
        attention_coefficient_scale: str = "none",
        attention_coefficient_orders: tuple[int, ...] | None = None,
        attention_coefficient_zero_constant: bool = False,
        attention_coefficient_gate: str = "none",
        qk_norm_type: str = "cosine_log_scale",
        qk_norm_head_scale: bool = False,
        qk_norm_token_scale: str = "none",
        attention_forget_gate: bool = False,
        attention_forget_gate_mode: str = "cumulative",
        attention_forget_gate_form: str = "input",
        attention_forget_gate_conditioning: str = "query",
        attention_forget_gate_rate_activation: str = "softplus",
        attention_blocked_implementation: str = "scan",
        attention_forget_gate_constant_rate: str = "learned",
        attention_kernel_basis: str = "polynomial",
        attention_spline_knots: tuple[float, ...] | None = None,
        attention_score_bias: str = "none",
        attention_score_transform: str = "identity",
        attention_score_prefactor: str = "none",
        score_mlp_width: int | None = None,
        score_mlp_architecture: str = "dynamic_readout",
        score_mlp_form: str = "kernel",
        score_mlp_activation: str = "silu",
        attention_exp_shift: str = "none",
        attention_head_rms_prescale: str = "none",
        attention_polynomial_shift: str = "none",
        rngs: nnx.Rngs,
    ):
        if positional_encoding not in ("rope", "path", "none"):
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        # The blocked backend's execution: the historical scan or causal strips (backends.reference).
        if attention_blocked_implementation not in ("scan", "strips"):
            raise ValueError(f"Unknown blocked attention implementation: {attention_blocked_implementation}")
        if attention_blocked_implementation == "strips" and positional_encoding == "path":
            raise ValueError("Causal strips run RoPE or position-free attention; PaTH has its own blocked code")
        self.attention_blocked_implementation = attention_blocked_implementation
        from scaffolds.transformer.attention import score_mlp

        score_mlp.parameter_names(score_mlp_architecture)
        if score_mlp_width is None and score_mlp_architecture != "dynamic_readout":
            raise ValueError("A conventional score MLP requires a hidden width")
        if score_mlp_architecture == "conventional" and (score_mlp_form != "kernel" or attention_forget_gate):
            raise ValueError("A conventional score MLP requires the pure ungated kernel")
        if score_mlp_form not in score_mlp.FORMS or score_mlp_activation not in score_mlp.ACTIVATIONS:
            raise ValueError(f"Unknown score-MLP form or activation: {score_mlp_form}, {score_mlp_activation}")
        if score_mlp_width is not None:
            if isinstance(score_mlp_width, bool) or not isinstance(score_mlp_width, int) or score_mlp_width < 1:
                raise ValueError("The score-MLP width must be a positive integer")
            if (attention_normalization or attention_degree is not None or quadratic_mode != "fixed"
                    or attention_fixed_coefficients is not None or attention_fixed_square_factors is not None
                    or attention_coefficient_conditioning != "static" or attention_kernel_basis != "polynomial"
                    or attention_score_bias != "none" or attention_score_transform != "identity"):
                raise ValueError("A score MLP replaces the kernel of unnormalized attention: no degree, "
                                 "polynomial coefficients, or normalization")
            if backend not in ("blocked", "fused"):
                raise ValueError("Score-MLP attention runs on the blocked or fused backend")
            if backend == "fused" and attention_forget_gate:
                raise ValueError("A gated score MLP runs on the blocked backend: the fused kernel has no forget gate")
            if positional_encoding not in ("rope", "none"):
                raise ValueError("Score-MLP attention supports RoPE or no positional encoding, not PaTH")
            if attention_forget_gate and (attention_forget_gate_mode != "cumulative"
                                          or attention_forget_gate_form != "input"):
                raise ValueError("Score-MLP attention supports only the cumulative input forget gate")
        self.score_mlp_width = score_mlp_width
        self.score_mlp_architecture = score_mlp_architecture
        self.score_mlp_form = score_mlp_form
        if positional_encoding == "path" and backend not in ("auto", "blocked", "fused"):
            raise ValueError("PaTH attention runs on backend 'blocked' (JAX) or 'fused' (FLA's Triton kernels)")
        self.positional_encoding = positional_encoding
        _validate_degree(attention_degree)
        if not isinstance(attention_normalization, bool):
            raise ValueError("attention_normalization must be a boolean")
        if not isinstance(qk_norm, bool):
            raise ValueError("qk_norm must be a boolean")
        if qk_norm_type not in ("cosine_log_scale", "rms_norm"):
            raise ValueError(f"Unknown QK-norm type: {qk_norm_type}")
        if not isinstance(qk_norm_head_scale, bool):
            raise ValueError("qk_norm_head_scale must be a boolean")
        if qk_norm_head_scale and (not qk_norm or qk_norm_type != "rms_norm"):
            raise ValueError("QK head scaling requires enabled QK RMSNorm")
        if qk_norm_token_scale not in ("none", "cosine"):
            raise ValueError(f"Unknown QK token scale: {qk_norm_token_scale}")
        if qk_norm_token_scale != "none" and not qk_norm_head_scale:
            raise ValueError("The per-token query scale extends QK RMS head scaling")
        if qk_norm and attention_degree == 0:
            raise ValueError("QK-norm requires a kernel that reads scores; degree 0 ignores Q/K")
        if not attention_normalization and (quadratic_mode != "fixed" or backend not in ("blocked", "fused", "mosaic")):
            raise ValueError("Unnormalized attention requires a fixed exp/Taylor kernel and a blocked, fused or mosaic backend")
        if attention_output_scaling not in OUTPUT_SCALINGS:
            raise ValueError(f"Unknown attention output scaling: {attention_output_scaling}")
        if attention_output_scaling == "inverse_position" and attention_normalization:
            raise ValueError("Inverse-position output scaling applies only to unnormalized attention")
        if quadratic_mode not in ("fixed", "positive", "unrestricted"):
            raise ValueError(f"Unknown quadratic mode: {quadratic_mode}")
        if quadratic_mode != "fixed" and attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        if attention_fixed_coefficients is not None:
            if quadratic_mode != "fixed" or attention_degree is None or attention_degree < 1:
                raise ValueError("Fixed polynomial coefficients require degree >= 1 without a quadratic mode")
            if (not isinstance(attention_fixed_coefficients, (tuple, list))
                    or len(attention_fixed_coefficients) != attention_degree + 1):
                raise ValueError("Fixed polynomial coefficients must contain attention_degree + 1 coefficients")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in attention_fixed_coefficients):
                raise ValueError("Fixed polynomial coefficients must be finite numeric values")
            attention_fixed_coefficients = tuple(float(value) for value in attention_fixed_coefficients)
        if attention_fixed_square_factors is not None:
            if attention_fixed_coefficients is not None:
                raise ValueError("Specify either fixed polynomial coefficients or fixed square factors")
            if (quadratic_mode != "fixed" or not attention_normalization or attention_degree is None
                    or attention_degree < 2 or attention_degree % 2):
                raise ValueError("Fixed square factors require normalized, even-degree attention without a quadratic mode")
            if (not isinstance(attention_fixed_square_factors, (tuple, list))
                    or len(attention_fixed_square_factors) != attention_degree + 1):
                raise ValueError("Fixed square factors must contain attention_degree + 1 factors")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in attention_fixed_square_factors):
                raise ValueError("Fixed square factors must be finite numeric values")
            attention_fixed_square_factors = tuple(float(value) for value in attention_fixed_square_factors)
        if min(hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if num_heads != num_kv_heads:
            raise ValueError("This experiment requires full MHA (num_heads == num_kv_heads)")
        if qk_head_dim % 2:
            raise ValueError("qk_head_dim must be even for RoPE")
        if backend not in ("blocked", "fused", "mosaic", "manual", "auto", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {backend}")
        if backend not in ("blocked", "fused", "mosaic") and attention_degree is not None:
            raise ValueError("Taylor attention requires backend='blocked', 'fused' or 'mosaic'")
        if backend != "manual" and dropout != 0.0:
            raise ValueError("Blocked/fused attention requires dropout=0")
        if attention_coefficient_conditioning not in ("static", "query", "query_key"):
            raise ValueError("Unknown attention coefficient conditioning")
        if attention_coefficient_activation not in ("identity", *("tanh", "sigmoid", "pure_tanh", "affine_sigmoid")):
            raise ValueError("Unknown attention coefficient activation")
        if attention_coefficient_activation != "identity" and attention_coefficient_conditioning == "static":
            raise ValueError("Tanh and sigmoid coefficients require query conditioning")
        if attention_coefficient_input_normalization not in ("none", "l2"):
            raise ValueError(f"Unknown coefficient input normalization: {attention_coefficient_input_normalization}")
        if attention_coefficient_input_source not in ("post_rmsnorm", "pre_rmsnorm"):
            raise ValueError(f"Unknown coefficient input source: {attention_coefficient_input_source}")
        if (attention_coefficient_input_source == "pre_rmsnorm"
                and attention_coefficient_input_normalization != "l2"):
            raise ValueError("pre_rmsnorm coefficient input source requires L2 coefficient input normalization")
        l2_score_supported = (attention_score_transform == "identity" or (
            attention_score_transform == "softplus"
            and attention_coefficient_parameterization == "sum_of_squares"
            and (attention_coefficient_input_power == 1 or attention_score_prefactor != "none")))
        if attention_coefficient_input_normalization == "l2" and (
                attention_coefficient_conditioning != "query"
                or attention_coefficient_activation not in ("tanh", "pure_tanh")
                or attention_coefficient_parameterization not in ("direct", "sum_of_squares")
                or attention_coefficient_readout != "diagonal" or attention_coefficient_orders is not None
                or attention_coefficient_gate != "none" or attention_forget_gate
                or attention_kernel_basis != "polynomial" or attention_score_bias != "none"
                or not l2_score_supported or attention_normalization
                or positional_encoding not in ("rope", "none")
                or attention_degree is None or attention_degree < 1 or quadratic_mode != "fixed"):
            raise ValueError("L2 coefficient input normalization requires unnormalized diagonal tanh (or direct pure "
                             "tanh) query polynomial coefficients with RoPE or no positional encoding, without "
                             "selected orders, gates or score biases; only softplus SoS with an input or score "
                             "prefactor permits a score transform")
        self.attention_coefficient_input_normalization = attention_coefficient_input_normalization
        self.attention_coefficient_input_source = attention_coefficient_input_source
        if not isinstance(attention_forget_gate, bool):
            raise ValueError("attention_forget_gate must be a boolean")
        if attention_forget_gate_mode not in ("cumulative", "query_distance"):
            raise ValueError(f"Unknown attention forget gate mode: {attention_forget_gate_mode}")
        if attention_forget_gate_mode != "cumulative" and not attention_forget_gate:
            raise ValueError("A query-distance decay mode requires attention_forget_gate")
        # Softmax and the unnormalized exponential add the decay to their scores; nonlinear
        # query-coefficient polynomials multiply their weights by it. Each runs with RoPE or no
        # positional encoding on the blocked backend, or with PaTH on either PaTH backend; the
        # polynomials' query-distance decay (and its endpoint variants) also runs on Mosaic, and so
        # does their cumulative decay with query conditioning when every order has its own gate
        # (the dispatch refuses a shared cumulative gate there). The
        # query-key endpoint gate applies to all three, in the query-distance mode. Static
        # coefficients take the input-free query-distance gate (conditioning none), whose rate reads
        # no token, the per-order query-key gate, or the shared cumulative input gate on the blocked
        # backend (static FoX); they then run the query-coefficient decay operation.
        input_free_gate = attention_forget_gate_conditioning == "none"
        exponential_gate = (attention_degree is None and quadratic_mode == "fixed"
                            and attention_coefficient_conditioning == "static" and not input_free_gate)
        static_gate_polynomial = (attention_coefficient_conditioning == "static" and not attention_normalization
                                  and attention_degree is not None and attention_degree >= 1 and quadratic_mode == "fixed"
                                  and attention_fixed_coefficients is None and attention_fixed_square_factors is None
                                  and attention_coefficient_orders is None and attention_kernel_basis == "polynomial")
        # On the blocked backend every unnormalized static or fixed direct polynomial takes every gate: its
        # coefficients run per query (_coefficients_per_query).
        composed_static = (attention_coefficient_conditioning == "static" and not attention_normalization
                           and attention_degree is not None and attention_degree >= 1 and quadratic_mode == "fixed"
                           and attention_fixed_square_factors is None and attention_kernel_basis == "polynomial"
                           and attention_coefficient_parameterization == "direct" and score_mlp_width is None
                           and positional_encoding in ("rope", "none") and backend == "blocked")
        polynomial_gate = (static_gate_polynomial if input_free_gate else
                           (attention_coefficient_conditioning == "query" and attention_coefficient_activation != "identity")
                           or (static_gate_polynomial and attention_forget_gate_conditioning == "query_key")
                           or (static_gate_polynomial and attention_forget_gate_conditioning == "query"
                               and attention_forget_gate_mode == "cumulative" and attention_forget_gate_form == "input"
                               and backend == "blocked")) or composed_static
        forget_backend_supported = (
            positional_encoding == "path" and backend in ("auto", "blocked", "fused")
            or positional_encoding in ("rope", "none") and (
                backend == "blocked" or backend == "mosaic" and polynomial_gate
                and (attention_forget_gate_mode == "query_distance" or attention_forget_gate_conditioning == "query")
                and attention_forget_gate_form == "input"))
        if attention_forget_gate and not ((exponential_gate or polynomial_gate) and forget_backend_supported):
            raise ValueError("The forget gate requires softmax, the unnormalized exponential, or nonlinear query "
                             "coefficients (unnormalized static coefficients take the input-free gate, the "
                             "per-order query-key gate, or the cumulative input gate on blocked): RoPE/none "
                             "on the blocked backend (query-distance and per-order cumulative decay of polynomials "
                             "also on mosaic), or PaTH on blocked/fused/auto")
        if attention_forget_gate_mode == "query_distance" and positional_encoding == "path":
            # The PaTH reference and fused kernels apply only the cumulative decay exp(L_t - L_i).
            raise ValueError("The query-distance decay mode requires RoPE or no positional encoding on the "
                             "blocked backend; PaTH supports only cumulative FoX decay")
        if attention_forget_gate_mode == "query_distance" and not (polynomial_gate or exponential_gate):
            raise ValueError("The query-distance decay mode requires nonlinear query coefficients, softmax, or "
                             "the unnormalized exponential")
        self.attention_forget_gate = attention_forget_gate
        self.attention_forget_gate_mode = attention_forget_gate_mode
        if attention_forget_gate_form not in ("input", "constant", "sigmoid"):
            raise ValueError(f"Unknown forget gate form: {attention_forget_gate_form}")
        if attention_forget_gate_form != "input" and not attention_forget_gate:
            raise ValueError("A forget gate form requires attention_forget_gate")
        if attention_forget_gate_form == "sigmoid" and (
                attention_forget_gate_mode != "cumulative" or positional_encoding not in ("rope", "none")):
            raise ValueError("The sigmoid forget gate form is qualified as cumulative decay with RoPE or no "
                             "positional encoding on the blocked backend")
        if attention_forget_gate_form == "constant" and (
                attention_forget_gate_mode != "cumulative"
                or positional_encoding not in ("rope", "none") or not (polynomial_gate or exponential_gate)):
            raise ValueError("The constant forget gate form requires the cumulative gate on softmax, the "
                             "unnormalized exponential, or nonlinear query coefficients, with RoPE or no "
                             "positional encoding on the blocked backend")
        self.attention_forget_gate_form = attention_forget_gate_form
        if attention_forget_gate_constant_rate not in ("learned", "fixed"):
            raise ValueError(f"Unknown constant forget gate rate: {attention_forget_gate_constant_rate}")
        if attention_forget_gate_constant_rate != "learned" and (
                not attention_forget_gate or attention_forget_gate_form != "constant"):
            raise ValueError("A fixed forget gate rate requires the constant forget gate form")
        self.attention_forget_gate_constant_rate = attention_forget_gate_constant_rate
        if attention_forget_gate_conditioning not in ("query", "query_key", "none"):
            raise ValueError("Unknown forget gate conditioning")
        if attention_forget_gate_rate_activation not in ("softplus", "sigmoid"):
            raise ValueError("Unknown forget gate rate activation")
        if input_free_gate and (not attention_forget_gate or attention_forget_gate_rate_activation != "softplus"
                                or attention_coefficient_conditioning != "static"):
            raise ValueError("The input-free gate is the softplus query-distance gate of static coefficients")
        if (attention_forget_gate_conditioning != "query" or attention_forget_gate_rate_activation != "softplus") and (
                not attention_forget_gate or attention_forget_gate_form != "input"
                or attention_forget_gate_mode != "query_distance" or backend not in ("blocked", "mosaic")):
            raise ValueError("Endpoint gate variants require input-form query-distance decay on the blocked or "
                             "mosaic backend")
        if attention_forget_gate_conditioning == "query_key" and attention_forget_gate_rate_activation != "softplus":
            raise ValueError("Query-key decay is qualified only with softplus rates")
        if exponential_gate and attention_forget_gate_mode == "query_distance" and (
                attention_forget_gate_rate_activation != "softplus"):
            # ALiBi with a learned, input-dependent slope per query token: (t - i) log alpha_t joins each
            # score; query_key conditioning gives each pair its own rate, (t - i) log alpha_ti.
            raise ValueError("Query-distance decay of softmax and the unnormalized exponential takes the "
                             "input-form softplus gate with query conditioning, or its query-key endpoint "
                             "form, on the blocked backend")
        self.attention_forget_gate_conditioning = attention_forget_gate_conditioning
        self.attention_forget_gate_rate_activation = attention_forget_gate_rate_activation
        if attention_coefficient_conditioning == "query_key" and attention_coefficient_activation != "tanh":
            raise ValueError("Query-key coefficients require the tanh activation")
        if attention_coefficient_conditioning in ("query", "query_key") and (
                attention_normalization or attention_degree is None or attention_degree < 1 or quadratic_mode != "fixed"
                or attention_fixed_coefficients is not None or attention_fixed_square_factors is not None):
            raise ValueError("Query-conditioned coefficients require unnormalized polynomial attention")
        if (attention_coefficient_conditioning == "query" and attention_coefficient_activation == "identity"
                and attention_degree != 1):
            raise ValueError("Identity query coefficients require degree-one attention")
        if type(attention_coefficient_input_power) is not int or attention_coefficient_input_power not in (0, 1):
            raise ValueError("Coefficient input power must be the integer 0 or 1")
        if attention_coefficient_input_power and (
                attention_coefficient_parameterization != "sum_of_squares"
                or attention_coefficient_conditioning not in ("static", "query")
                or attention_normalization or attention_score_transform != "softplus"):
            raise ValueError("An input power requires unnormalized static or query sum-of-squares coefficients "
                             "with softplus scores")
        if attention_static_sos_parameterization not in ("direct", "relative_taylor"):
            raise ValueError("Unknown static SoS parameterization")
        if attention_static_sos_parameterization != "direct" and (
                attention_coefficient_parameterization != "sum_of_squares"
                or attention_coefficient_conditioning != "static" or attention_fixed_square_factors is not None
                or attention_normalization):
            raise ValueError("The relative Taylor factor map applies only to unnormalized learned static SoS coefficients")
        self.attention_static_sos_parameterization = attention_static_sos_parameterization
        static_sos = (attention_coefficient_parameterization == "sum_of_squares"
                      and attention_coefficient_conditioning == "static" and not attention_normalization)
        if static_sos and (
                attention_degree is None or attention_degree - attention_coefficient_input_power < 2
                or (attention_degree - attention_coefficient_input_power) % 2
                or attention_coefficient_activation != "identity" or attention_coefficient_readout != "diagonal"
                or attention_coefficient_orders is not None or attention_coefficient_gate != "none"
                or attention_coefficient_scale != "none" or attention_coefficient_amplitude_scale != "none"
                or attention_forget_gate or attention_score_bias != "none"
                or attention_score_transform not in ("identity", "softplus") or attention_score_prefactor != "none"
                or attention_kernel_basis != "polynomial" or positional_encoding not in ("rope", "none")
                or backend not in ("blocked", "mosaic")):
            raise ValueError("Unnormalized static SoS requires an even core degree, identity coefficient activation, "
                             "identity or softplus scores, no prefactor or extra scales/gates/biases, and blocked or "
                             "mosaic attention")
        if attention_coefficient_parameterization == "sum_of_squares" and attention_coefficient_conditioning in (
                "query", "query_key"):
            core_degree = attention_degree - attention_coefficient_input_power
            if (core_degree < 2 or core_degree % 2 or attention_coefficient_conditioning != "query"
                    or attention_coefficient_activation not in ("tanh", "pure_tanh")
                    or attention_coefficient_readout != "diagonal"
                    or attention_coefficient_orders is not None or attention_coefficient_gate != "none"
                    or attention_coefficient_amplitude_scale != "none"
                    or attention_score_transform not in ("identity", "softplus")):
                raise ValueError("Query sum-of-squares requires an even core degree and diagonal tanh or pure tanh "
                                 "query coefficients without selected orders, coefficient gates, or amplitude scaling")
            if (attention_score_transform == "softplus" or attention_coefficient_input_power) and (
                    attention_forget_gate or attention_score_bias != "none"):
                raise ValueError("Softplus sum-of-squares takes no forget gate or score biases")
        if attention_coefficient_readout not in ("diagonal", "dense"):
            raise ValueError("Unknown attention coefficient readout")
        if attention_coefficient_readout == "dense" and (
                attention_coefficient_conditioning != "query"
                or attention_coefficient_activation not in ("tanh", "sigmoid", "pure_tanh", "affine_sigmoid")):
            raise ValueError("Dense coefficient readout requires tanh or sigmoid query-only coefficients")
        if (not isinstance(attention_coefficient_readout_scale, (int, float))
                or isinstance(attention_coefficient_readout_scale, bool)
                or not math.isfinite(attention_coefficient_readout_scale)
                or attention_coefficient_readout_scale <= 0):
            raise ValueError("Coefficient readout scale must be finite and positive")
        if attention_coefficient_readout_scale != 1.0 and attention_coefficient_readout != "dense":
            raise ValueError("A non-unit readout scale requires dense coefficient readout")
        self.attention_coefficient_readout = attention_coefficient_readout
        self.attention_coefficient_readout_scale = attention_coefficient_readout_scale
        self.attention_coefficient_orders = attention_coefficient_orders
        self.attention_coefficient_zero_constant = attention_coefficient_zero_constant
        self.attention_coefficient_conditioning = attention_coefficient_conditioning
        self.attention_coefficient_parameterization = attention_coefficient_parameterization
        self.attention_coefficient_input_power = attention_coefficient_input_power
        if attention_coefficient_amplitude_scale not in ("none", "factorial"):
            raise ValueError(f"Unknown attention coefficient amplitude scale: {attention_coefficient_amplitude_scale}")
        if (attention_coefficient_amplitude_scale != "none"
                and attention_coefficient_activation not in ("tanh", "sigmoid", "pure_tanh", "affine_sigmoid")):
            raise ValueError("An amplitude scale applies to tanh and sigmoid coefficients only")
        self.attention_coefficient_activation = attention_coefficient_activation
        self.attention_coefficient_amplitude_scale = attention_coefficient_amplitude_scale
        if attention_coefficient_scale not in ("none", "factorial"):
            raise ValueError(f"Unknown attention coefficient scale: {attention_coefficient_scale}")
        self.attention_coefficient_scale = attention_coefficient_scale
        if attention_coefficient_gate not in ("none", "sigmoid", "gdn"):
            raise ValueError(f"Unknown attention coefficient gate: {attention_coefficient_gate!r}")
        if attention_coefficient_gate != "none" and (
                attention_coefficient_conditioning != "query" or attention_coefficient_activation != "tanh"
                or attention_coefficient_readout != "diagonal" or attention_coefficient_orders is not None
                or attention_forget_gate or positional_encoding != "rope"):
            raise ValueError("Coefficient gates require diagonal tanh query coefficients with RoPE, "
                             "without selected orders or a forget gate")
        self.attention_coefficient_gate = attention_coefficient_gate
        # The score function's basis: monomials (historical) or a fixed B-spline basis (attention/spline.py).
        if attention_kernel_basis not in ("polynomial", "bspline"):
            raise ValueError(f"Unknown attention kernel basis: {attention_kernel_basis}")
        if attention_kernel_basis == "bspline":
            attention_spline_knots = validate_knots(attention_spline_knots, attention_degree)
            if (backend != "blocked" or attention_normalization or positional_encoding == "path"
                    or quadratic_mode != "fixed" or attention_coefficient_conditioning not in ("static", "query")
                    or (attention_coefficient_conditioning == "query") != (attention_coefficient_activation == "tanh")
                    or attention_coefficient_readout != "diagonal" or attention_forget_gate
                    or attention_fixed_coefficients is not None or attention_fixed_square_factors is not None
                    or attention_coefficient_scale != "none" or attention_coefficient_amplitude_scale != "none"
                    or attention_coefficient_orders is not None or attention_coefficient_gate != "none"):
                raise ValueError("B-spline attention requires unnormalized static or diagonal tanh-query coefficients "
                                 "on the blocked backend, without a forget gate, PaTH, fixed coefficients, selected "
                                 "orders, coefficient gates, or factorial scales")
        elif attention_spline_knots is not None:
            raise ValueError("Spline knots apply to the bspline kernel basis only")
        self.attention_kernel_basis = attention_kernel_basis
        self.attention_spline_knots = attention_spline_knots
        # Per-degree score biases (attention/score_bias.py) shift each order's score of diagonal tanh query
        # coefficients; ModelConfig states the recipe-level requirements, mirrored for direct construction.
        from scaffolds.transformer.attention.score_bias import MODES as SCORE_BIAS_MODES

        if not isinstance(attention_score_bias, str) or attention_score_bias not in SCORE_BIAS_MODES:
            raise ValueError(f"Unknown attention score bias: {attention_score_bias!r}")
        # Composition on the blocked backend (attention/pieces.py): every unnormalized polynomial with RoPE or no
        # positional encoding, its static or fixed direct coefficients run per query (_coefficients_per_query),
        # takes any forget gate, score transform, score biases and prefactor. Fast kernels keep their declarations.
        blocked_polynomial = (
            backend == "blocked" and positional_encoding in ("rope", "none") and not attention_normalization
            and attention_degree is not None and attention_degree >= 1 and quadratic_mode == "fixed"
            and attention_kernel_basis == "polynomial" and score_mlp_width is None
            and attention_fixed_square_factors is None
            and (attention_coefficient_conditioning in ("query", "query_key") and attention_coefficient_activation != "identity"
                 or attention_coefficient_conditioning == "static" and attention_coefficient_parameterization == "direct"))
        if attention_score_bias != "none" and not blocked_polynomial:
            from scaffolds.transformer.attention.backends.mosaic.api import MAX_QUERY_DEGREE

            if (attention_coefficient_conditioning != "query" or attention_coefficient_activation != "tanh"
                    or attention_coefficient_readout != "diagonal" or attention_coefficient_orders is not None
                    or attention_coefficient_gate != "none" or attention_kernel_basis != "polynomial"
                    or attention_normalization or quadratic_mode != "fixed"
                    or not 1 <= attention_degree <= MAX_QUERY_DEGREE
                    or positional_encoding not in ("rope", "none") or backend not in ("blocked", "mosaic")
                    or attention_forget_gate and not (
                        attention_forget_gate_form == "input" and attention_forget_gate_mode == "query_distance"
                        and attention_forget_gate_conditioning == "query_key"
                        and attention_forget_gate_rate_activation == "softplus")):
                raise ValueError("Per-degree score biases require diagonal tanh query coefficients of unnormalized "
                                 f"polynomial attention of degree 1-{MAX_QUERY_DEGREE} with RoPE or no positional "
                                 "encoding, without selected orders, coefficient gates, or a B-spline basis, on the "
                                 "mosaic or blocked backend, and no forget gate but the per-order query-key "
                                 "endpoint gate")
        self.attention_score_bias = attention_score_bias
        # A transformed polynomial input (attention/score_transform.py); ModelConfig states the recipe-level
        # requirements, mirrored for direct construction.
        from scaffolds.transformer.attention.score_transform import validate as validate_transform

        fixed_pointwise = (attention_score_transform in ("sigmoid", "softplus", "swish")
                           and attention_degree == 1 and attention_fixed_coefficients == (0.0, 1.0)
                           and attention_fixed_square_factors is None
                           and attention_coefficient_conditioning == "static"
                           and attention_coefficient_parameterization == "direct"
                           and attention_coefficient_activation == "identity"
                           and attention_coefficient_scale == "none"
                           and attention_coefficient_amplitude_scale == "none"
                           and attention_coefficient_input_power == 0)
        # Static direct coefficients (learned; the structural zero constant optional) run per query on the
        # transformed kernels (_coefficients_per_query).
        static_transformed = (attention_coefficient_conditioning == "static"
                              and attention_coefficient_parameterization == "direct"
                              and attention_coefficient_activation == "identity"
                              and attention_fixed_coefficients is None and attention_fixed_square_factors is None
                              and attention_coefficient_input_power == 0)
        if type(attention_coefficient_zero_constant) is not bool:
            raise ValueError("Coefficient zero constant must be a boolean")
        if attention_coefficient_zero_constant and (
                attention_score_transform not in ("identity", "softplus") or positional_encoding != "rope"
                or attention_coefficient_input_power or attention_coefficient_orders is not None
                or attention_coefficient_input_normalization != "none"
                or attention_coefficient_parameterization not in ("direct", "sum_of_squares")
                or attention_fixed_coefficients is not None or attention_fixed_square_factors is not None
                or not (static_transformed or attention_coefficient_conditioning == "query"
                        and attention_coefficient_activation == "tanh")):
            raise ValueError("Structural zero constants require direct static or tanh query raw or softplus RoPE "
                             "coefficients, without a prefactor, selected orders, fixed coefficients or L2")
        # The exponential (softmax or unnormalized) composes with any score function: f(s) exp(phi(s))
        # (attention/pieces.py), on the blocked backend, or on the fused kernel when it declares it.
        from scaffolds.transformer.attention.pieces import ScoreFunction, fused_exponential_problem

        score_function = ScoreFunction(attention_score_transform, attention_score_prefactor)
        exponential_kernel = (attention_degree is None and quadratic_mode == "fixed" and score_mlp_width is None
                              and attention_fixed_coefficients is None and attention_fixed_square_factors is None
                              and attention_coefficient_conditioning == "static"
                              and attention_kernel_basis == "polynomial" and attention_score_bias == "none")
        if not score_function.identity and exponential_kernel:
            if positional_encoding not in ("rope", "none"):
                raise ValueError("Score functions of the exponential apply with RoPE or no positional encoding")
            if backend not in ("blocked", "fused"):
                raise ValueError("Score functions of the exponential run on the blocked or fused backend")
            if backend == "fused":
                problem = fused_exponential_problem(score_function, attention_degree=attention_degree)
                if problem is None and (attention_normalization or attention_forget_gate):
                    problem = "the fused kernel is the unnormalized exponential without a forget gate"
                if problem is not None:
                    raise ValueError(f"The fused backend cannot run these score functions: {problem}")
        elif attention_score_prefactor != "none" and not blocked_polynomial and attention_polynomial_shift == "none" and not (
                backend == "mosaic" and attention_score_transform in ("identity", "softplus")
                and (static_transformed or attention_coefficient_conditioning == "query"
                     and attention_coefficient_activation == "tanh")):
            raise ValueError("A score prefactor applies to the exponential (softmax or unnormalized), on the "
                             "blocked backend to any unnormalized polynomial, or on the mosaic backend to diagonal "
                             "tanh query or static direct coefficients at identity or softplus scores")
        self.attention_score_prefactor = attention_score_prefactor
        # The row-power shift's score functions and coefficient sources are its own guard's (below).
        if (validate_transform(attention_score_transform) != "identity" or attention_score_prefactor != "none") and (
                not exponential_kernel and not blocked_polynomial and attention_polynomial_shift == "none") and (
                not (fixed_pointwise or (attention_score_transform not in ("sigmoid", "swish")
                                        and (static_sos or static_transformed
                                             or (attention_coefficient_conditioning == "query"
                                                 and attention_coefficient_activation == "tanh"))))
                or attention_coefficient_readout != "diagonal" or attention_coefficient_orders is not None
                or attention_coefficient_gate != "none" or attention_kernel_basis != "polynomial"
                or attention_normalization or quadratic_mode != "fixed" or attention_degree is None
                or attention_degree < 1 or positional_encoding not in ("rope", "none")
                or backend not in ("blocked", "mosaic", *(("fused",) if attention_polynomial_shift != "none" else ()))
                or backend == "mosaic" and positional_encoding != "rope"
                or attention_forget_gate or attention_score_bias != "none"):
            raise ValueError("A score transform requires fixed static linear (0, 1) sigmoid/softplus/swish weights, "
                             "static direct or diagonal tanh query coefficients of unnormalized polynomial "
                             "attention with RoPE or no positional encoding, without selected orders, coefficient "
                             "gates, a B-spline basis, a forget gate, or score biases, on the blocked backend, or "
                             "with RoPE on the mosaic backend")
        self.attention_score_transform = attention_score_transform
        # The unnormalized exponential shifted by its causal row maximum (attention/pieces.py exp_shift_problem, which
        # ModelConfig states for recipes): exact only where head RMSNorm takes the output with its shift.
        from scaffolds.transformer.attention.pieces import EXP_SHIFTS

        if attention_exp_shift not in EXP_SHIFTS:
            raise ValueError(f"Unknown exponential shift: {attention_exp_shift!r}")
        if attention_exp_shift != "none" and (
                attention_normalization or not exponential_kernel or attention_output_scaling != "head_rms_norm"
                or positional_encoding not in ("rope", "none") or backend not in ("blocked", "fused")
                or backend == "fused" and (attention_forget_gate or not score_function.identity)):
            raise ValueError("The row-maximum shift applies to the unnormalized exponential with head RMSNorm output "
                             "scaling and RoPE or no positional encoding, on the blocked backend or, plain and "
                             "ungated, the fused kernel")
        self.attention_exp_shift = attention_exp_shift
        # Head RMSNorm in units of each row's largest magnitude (attention/output_scaling.py): the normalization's
        # own evaluation, so it needs that output scaling and nothing else.
        if attention_head_rms_prescale not in HEAD_RMS_PRESCALES:
            raise ValueError(f"Unknown head RMSNorm prescale: {attention_head_rms_prescale!r}")
        if attention_head_rms_prescale != "none" and attention_output_scaling != "head_rms_norm":
            raise ValueError("The max-abs prescale evaluates head RMSNorm output scaling, which this layer does "
                             "not use")
        self.attention_head_rms_prescale = attention_head_rms_prescale
        # A per-query polynomial in units of its row's largest power (attention/pieces.py polynomial_shift_problem,
        # which ModelConfig states for recipes): exact only where max-abs head RMSNorm takes the output with its shift.
        from scaffolds.transformer.attention.pieces import POLYNOMIAL_SHIFTS

        if attention_polynomial_shift not in POLYNOMIAL_SHIFTS:
            raise ValueError(f"Unknown polynomial shift: {attention_polynomial_shift!r}")
        if attention_polynomial_shift != "none" and (
                attention_normalization or (attention_coefficient_conditioning, attention_coefficient_parameterization)
                not in (("query", "sum_of_squares"), ("query", "direct"), ("static", "direct"))
                or attention_score_transform not in ("identity", "softplus")
                or attention_score_prefactor not in ("none", "softplus", "exp")
                or (attention_score_prefactor != "none" and attention_score_transform != "identity")
                or attention_head_rms_prescale != "max_abs" or backend not in ("blocked", "fused")):
            raise ValueError("The row-power shift evaluates unnormalized per-query polynomials (tanh query coefficients, "
                             "direct or sum of squares, or static direct ones) on the raw or softplus score, or times "
                             "softplus or exp of the raw score, under max-abs head RMSNorm, on the blocked or fused "
                             "backend")
        self.attention_polynomial_shift = attention_polynomial_shift
        if attention_coefficient_zero_constant and attention_coefficient_parameterization == "direct":
            # The same compact selection and factorial handling as explicit orders,
            # while the public recipe records the structural constraint separately.
            self.attention_coefficient_orders = tuple(range(1, attention_degree + 1))
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.rope_theta = rope_theta
        self.backend = backend
        self.attention_degree = attention_degree
        # Python floats are static module data, never optimizer parameters.
        self.attention_fixed_coefficients = attention_fixed_coefficients
        self.attention_fixed_square_factors = attention_fixed_square_factors
        self.attention_normalization = attention_normalization
        self.quadratic_mode = quadratic_mode
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type
        self.qk_norm_head_scale = qk_norm_head_scale
        self.qk_norm_token_scale = qk_norm_token_scale
        self.qk_norm_eps = qk_norm_eps
        self.attention_output_scaling = attention_output_scaling
        self.rms_norm_eps = rms_norm_eps
        self.dtype = jnp.float32
        # Preserve the reference constructor order/default initialization exactly.
        self.q_proj = nnx.Linear(hidden_size, num_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, num_kv_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, num_kv_heads * value_head_dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout)
        self.o_proj = nnx.Linear(num_heads * value_head_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None, return_diagnostics: bool = False,
        coefficients: jax.Array | None = None,
        attention_parameters: dict[str, jax.Array] | None = None,
        query_coefficients: jax.Array | None = None,
        return_coefficients: bool = False,
        key_factors: jax.Array | None = None,
        coefficient_source_input: jax.Array | None = None,
        coefficient_score_transform: str = "identity",
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array] | jax.Array]:
        """Optionally replace or return effective query coefficients [B, T, H, m+1].

        With replaced static coefficients, ``key_factors`` [B, T, H, m+1] multiply each
        key's degree-r term, running the query-key polynomial kernel.

        ``x`` remains the post-RMSNorm Q/K/V input. A pre_rmsnorm coefficient source
        must be supplied explicitly as ``coefficient_source_input`` with the same shape;
        it is unnecessary when ``query_coefficients`` bypasses the generator.
        """
        batch, seq_len, _ = x.shape
        if coefficient_source_input is not None:
            if self.attention_coefficient_input_source != "pre_rmsnorm":
                raise ValueError("coefficient_source_input requires the pre_rmsnorm coefficient input source")
            if coefficient_source_input.shape != x.shape:
                raise ValueError(f"coefficient_source_input must have shape {x.shape}, "
                                 f"got {coefficient_source_input.shape}")
        elif self.attention_coefficient_input_source == "pre_rmsnorm" and query_coefficients is None:
            raise ValueError("pre_rmsnorm coefficient input source requires coefficient_source_input")
        if return_coefficients and return_diagnostics:
            raise ValueError("return_coefficients and return_diagnostics are mutually exclusive")
        spline = self.attention_kernel_basis == "bspline"
        if spline and (query_coefficients is not None or return_coefficients or key_factors is not None):
            raise ValueError("Coefficient override and capture apply to monomial coefficients, not a B-spline basis")
        if self.attention_score_bias != "none" and (
                query_coefficients is not None or return_coefficients or key_factors is not None):
            raise ValueError("Coefficient override and capture assume sum_r c_r s^r; per-degree score biases "
                             "shift every order's score")
        if self.attention_score_transform != coefficient_score_transform and (
                query_coefficients is not None or return_coefficients or key_factors is not None):
            raise ValueError("Coefficient score transform must match the model's attention_score_transform")
        if self.attention_score_prefactor != "none" and (
                query_coefficients is not None or return_coefficients or key_factors is not None):
            raise ValueError("Coefficient override and capture assume the polynomial alone; a score prefactor "
                             "multiplies it by f(s)")
        if query_coefficients is not None or return_coefficients:
            if (self.attention_coefficient_conditioning not in ("static", "query")
                    or self.attention_normalization or self.attention_degree is None
                    or self.attention_degree < 1):
                raise ValueError("Coefficient override and capture require unnormalized static or query coefficients")
            if query_coefficients is not None:
                expected = (batch, seq_len, self.num_heads, self.attention_degree + 1)
                if query_coefficients.shape != expected:
                    raise ValueError(f"query_coefficients must have shape {expected}, got {query_coefficients.shape}")
                query_coefficients = jnp.asarray(query_coefficients, dtype=jnp.float32)
        if key_factors is not None:
            if (query_coefficients is None or key_factors.shape != query_coefficients.shape
                    or self.attention_coefficient_conditioning != "static"):
                raise ValueError("key_factors require replaced static coefficients of the same shape")
            key_factors = jnp.asarray(key_factors, dtype=jnp.float32)
        parameters = {} if attention_parameters is None else dict(attention_parameters)
        log_scale = parameters.pop("qk_norm_log_scale", None)
        q_gain = parameters.pop("qk_norm_q_gain", None)
        k_gain = parameters.pop("qk_norm_k_gain", None)
        head_log_scale = parameters.pop("qk_norm_head_log_scale", None)
        if (self.qk_norm and self.qk_norm_type == "cosine_log_scale") != (log_scale is not None):
            raise ValueError("QK-norm requires exactly its model-owned qk_norm_log_scale parameter")
        rms_enabled = self.qk_norm and self.qk_norm_type == "rms_norm"
        if rms_enabled != (q_gain is not None) or rms_enabled != (k_gain is not None):
            raise ValueError("QK RMSNorm requires exactly its model-owned qk_norm_q_gain and qk_norm_k_gain parameters")
        if self.qk_norm_head_scale != (head_log_scale is not None):
            raise ValueError("QK RMS head scaling requires exactly its model-owned qk_norm_head_log_scale parameter")
        token_scale_kernel, token_scale_amplitude = (parameters.pop(name, None) for name in TOKEN_SCALE_PARAMETERS)
        if ((self.qk_norm_token_scale != "none") != (token_scale_kernel is not None)
                or (token_scale_kernel is None) != (token_scale_amplitude is None)):
            raise ValueError("The per-token query scale requires exactly its model-owned kernel and amplitude")
        output_norm_scale = parameters.pop("output_norm_scale", None)
        if (self.attention_output_scaling == "head_rms_norm") != (output_norm_scale is not None):
            raise ValueError("Head RMSNorm output scaling requires exactly its model-owned output_norm_scale parameter")
        query_kernel = parameters.pop("query_polynomial_kernel", None)
        query_bias = parameters.pop("query_polynomial_bias", None)
        query_amplitude = parameters.pop("query_polynomial_amplitude", None)
        query_mixing = parameters.pop("query_polynomial_mixing", None)
        query_tanh_bias = parameters.pop("query_polynomial_tanh_bias", None)
        key_kernel = parameters.pop("key_polynomial_kernel", None)
        key_amplitude = parameters.pop("key_polynomial_amplitude", None)
        gate_kernel = parameters.pop("query_polynomial_gate_kernel", None)
        gate_bias = parameters.pop("query_polynomial_gate_bias", None)
        # The GDN gate's dt_bias takes the sigmoid gate's bias slot; A_log selects its multiplier.
        gate_log_rate = parameters.pop("query_polynomial_gate_log_rate", None)
        gate_dt_bias = parameters.pop("query_polynomial_gate_dt_bias", None)
        gdn_gate = self.attention_coefficient_gate == "gdn"
        if (gdn_gate != (gate_log_rate is not None) or gdn_gate != (gate_dt_bias is not None)
                or (gdn_gate and gate_bias is not None)):
            raise ValueError("The GDN coefficient gate requires exactly its model-owned kernel, A_log and dt_bias")
        if gdn_gate:
            gate_bias = gate_dt_bias
        if ((self.attention_coefficient_gate != "none") != (gate_kernel is not None)
                or (gate_kernel is None) != (gate_bias is None)):
            raise ValueError("Coefficient gates require exactly their model-owned kernel and bias")
        if gate_kernel is not None and query_coefficients is not None:
            raise ValueError("A coefficient override would bypass the coefficient gate")
        spline_parameters = {name: parameters.pop(name, None)
                             for name in (*SPLINE_STATIC_PARAMETERS, *SPLINE_QUERY_PARAMETERS)}
        spline_names = (() if not spline else SPLINE_QUERY_PARAMETERS
                        if self.attention_coefficient_conditioning == "query" else SPLINE_STATIC_PARAMETERS)
        if any(spline_parameters[name] is None for name in spline_names) or any(
                value is not None for name, value in spline_parameters.items() if name not in spline_names):
            raise ValueError("B-spline attention requires exactly its model-owned coefficient parameters")
        if self.attention_coefficient_orders is not None and self.attention_coefficient_conditioning == "static":
            from scaffolds.transformer.attention.parameters import expand_polynomial_coefficients

            parameters["polynomial_coefficients"] = expand_polynomial_coefficients(
                parameters["polynomial_coefficients"], self.attention_degree, self.attention_coefficient_orders)
        if self.attention_coefficient_scale == "factorial":
            from scaffolds.transformer.attention.parameters import factorial_scale

            if (query_bias is None) == ("polynomial_coefficients" not in parameters):
                raise ValueError("A coefficient scale needs exactly one learned coefficient array")
            if query_bias is not None:
                query_bias = factorial_scale(query_bias, self.attention_coefficient_orders)
            else:
                parameters["polynomial_coefficients"] = factorial_scale(parameters["polynomial_coefficients"])
        if spline:
            # theta per layer/head, or theta_j(x_t) = b_j + a_j tanh(w_j^T x_t) from the monomial
            # query generator's own FP32 projection; the kernel is attention/spline.py's.
            if any(value is not None for value in (query_kernel, query_bias, query_amplitude, query_mixing,
                                                   query_tanh_bias, key_kernel, key_amplitude)):
                raise ValueError("B-spline attention cannot receive monomial coefficient parameters")
            parameters["spline_knots"] = self.attention_spline_knots
            if self.attention_coefficient_conditioning == "query":
                from scaffolds.transformer.attention.input_dependent import project_coefficients

                parameters["query_spline_coefficients"] = project_coefficients(
                    x, spline_parameters["query_spline_kernel"], spline_parameters["query_spline_bias"],
                    amplitude=spline_parameters["query_spline_amplitude"], activation="tanh")
            else:
                parameters["spline_coefficients"] = spline_parameters["spline_coefficients"]
        elif self.attention_coefficient_conditioning in ("query", "query_key"):
            from scaffolds.transformer.attention.input_dependent import (
                normalize_coefficient_input, project_coefficients, project_key_factors,
            )

            if query_kernel is None or (query_bias is None) != (
                    self.attention_coefficient_activation in ("sigmoid", "pure_tanh")):
                raise ValueError("Query-conditioned attention requires its model-owned kernel and, "
                                 "except for pure sigmoid and tanh coefficients, bias")
            if (self.attention_coefficient_readout == "dense") != (query_mixing is not None):
                raise ValueError("Coefficient readout must match its model-owned mixing parameter")
            if query_mixing is not None and query_amplitude is not None:
                raise ValueError("Dense readout replaces the diagonal amplitude parameter")
            effective_coefficients = query_coefficients
            if effective_coefficients is None:
                if (self.attention_coefficient_input_normalization == "l2" and query_tanh_bias is not None
                        and self.attention_coefficient_activation != "pure_tanh"):
                    raise ValueError("L2 coefficient input normalization supports the inner tanh bias of pure tanh "
                                     "coefficients only")
                coefficient_source = (coefficient_source_input
                                      if self.attention_coefficient_input_source == "pre_rmsnorm" else x)
                coefficient_input = normalize_coefficient_input(
                    coefficient_source, self.attention_coefficient_input_normalization)
                effective_coefficients = project_coefficients(
                    coefficient_input, query_kernel, query_bias, amplitude=query_amplitude, mixing=query_mixing,
                    activation=self.attention_coefficient_activation, tanh_bias=query_tanh_bias,
                    amplitude_scale=self.attention_coefficient_amplitude_scale,
                    readout_scale=self.attention_coefficient_readout_scale,
                    orders=self.attention_coefficient_orders, gate_kernel=gate_kernel, gate_bias=gate_bias,
                    gate_log_rate=gate_log_rate)
                if self.attention_coefficient_parameterization == "sum_of_squares":
                    from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients

                    # Pure tanh generates each factor's fraction of its Taylor factor, not the affine g.
                    effective_coefficients = query_square_coefficients(
                        effective_coefficients, self.attention_degree,
                        input_power=self.attention_coefficient_input_power,
                        zero_constant=self.attention_coefficient_zero_constant,
                        pure=self.attention_coefficient_activation == "pure_tanh")
            parameters["query_polynomial_coefficients"] = effective_coefficients
            if self.attention_coefficient_conditioning == "query_key":
                if key_kernel is None or key_amplitude is None:
                    raise ValueError("Query-key attention requires its model-owned key kernel and amplitude")
                parameters["key_polynomial_factors"] = project_key_factors(x, key_kernel, key_amplitude)
            elif key_kernel is not None or key_amplitude is not None:
                raise ValueError("Query-conditioned attention cannot receive key factor parameters")
        elif any(value is not None for value in (query_kernel, query_bias, query_amplitude, query_mixing, query_tanh_bias,
                                                 key_kernel, key_amplitude, gate_kernel, gate_bias, gate_log_rate)):
            raise ValueError("Static coefficients cannot receive query coefficient parameters")
        from scaffolds.transformer.attention import score_mlp

        unit_parameters = {name: parameters.pop(name, None) for name in score_mlp.ALL_PARAMETERS}
        unit_names = score_mlp.parameter_names(self.score_mlp_architecture) if self.score_mlp_width is not None else ()
        if (any(unit_parameters[name] is None for name in unit_names)
                or any(value is not None for name, value in unit_parameters.items() if name not in unit_names)):
            raise ValueError("A score MLP requires exactly its model-owned unit parameters")
        if self.score_mlp_width is not None:
            amplitude, offset, slope, output_bias = score_mlp.project_parameters(
                x, unit_parameters, self.score_mlp_architecture)
            parameters["score_mlp_amplitude"] = amplitude
            parameters["score_mlp_offset"] = offset
            parameters["score_mlp_slope"] = slope
            parameters["score_mlp_form"] = self.score_mlp_form
            if output_bias is not None:
                parameters["score_mlp_output_bias"] = output_bias
        from scaffolds.transformer.attention.forget_gate import (
            PARAMETERS as FORGET_GATE_PARAMETERS, ALL_PARAMETERS, alibi_log_rate, constant_log_decay,
            input_free_log_decay, log_decay,
            parameter_names, project_gate_inputs, sigmoid_log_decay,
        )

        gate_names = parameter_names(self.attention_forget_gate_form, self.attention_forget_gate_conditioning,
                                     self.attention_forget_gate_constant_rate) if self.attention_forget_gate else ()
        forget_gate = {name: parameters.pop(name, None) for name in ALL_PARAMETERS}
        if any(forget_gate[name] is None for name in gate_names) or any(
                value is not None for name, value in forget_gate.items() if name not in gate_names):
            raise ValueError("The forget gate requires exactly its model-owned parameters for its form")
        log_alpha = None
        if self.attention_forget_gate:
            # The constant form's log alpha is -exp(A_log) at every token, A_log learned or fixed at
            # ALiBi's slopes (the learned rate's start); the input-free gate's is
            # -exp(A_log) softplus(dt_bias); the other input and sigmoid forms read x_t.
            if self.attention_forget_gate_form == "constant":
                log_alpha = constant_log_decay(x, alibi_log_rate(self.num_heads)
                                               if self.attention_forget_gate_constant_rate == "fixed"
                                               else forget_gate["forget_gate_log_rate"])
            elif self.attention_forget_gate_conditioning == "none":
                log_alpha = input_free_log_decay(x, forget_gate["forget_gate_log_rate"], forget_gate["forget_gate_dt_bias"])
            elif self.attention_forget_gate_form == "sigmoid":
                log_alpha = sigmoid_log_decay(x, forget_gate["forget_gate_kernel"], forget_gate["forget_gate_bias"])
            elif self.attention_forget_gate_conditioning == "query_key":
                query_input, key_input = project_gate_inputs(
                    x, forget_gate["forget_gate_kernel"], forget_gate["forget_gate_dt_bias"],
                    forget_gate["forget_gate_key_kernel"])
                parameters.update(query_gate_input=query_input, key_gate_input=key_input,
                                  gate_log_rate=forget_gate["forget_gate_log_rate"])
            else:
                log_alpha = log_decay(x, *(forget_gate[name] for name in FORGET_GATE_PARAMETERS),
                                      rate_activation=self.attention_forget_gate_rate_activation)
            # Endpoint gates keep only their token projections here; pair rates are evaluated
            # inside score blocks. Legacy modes retain their historical arithmetic exactly. The
            # cumulative sum runs over time, so per-order gates [B,T,H,R] sum each order separately.
            if log_alpha is not None:
                parameters["query_log_decay"] = (jnp.cumsum(log_alpha, axis=1)
                                                   if self.attention_forget_gate_mode == "cumulative" else log_alpha)
            parameters["query_decay_mode"] = self.attention_forget_gate_mode
        from scaffolds.transformer.attention.score_bias import (
            ALL_PARAMETERS as SCORE_BIAS_PARAMETERS, attention_operands, parameter_names as score_bias_names,
        )

        score_bias = {name: parameters.pop(name, None) for name in SCORE_BIAS_PARAMETERS}
        bias_names = score_bias_names(self.attention_score_bias)
        if any(score_bias[name] is None for name in bias_names) or any(
                value is not None for name, value in score_bias.items() if name not in bias_names):
            raise ValueError("Score biases require exactly their model-owned parameters for their mode")
        if bias_names:
            if (self.attention_forget_gate and self.backend != "blocked"
                    and getattr(parameters.get("query_gate_input"), "ndim", None) != 4):
                # The Mosaic score-bias kernels take only this gate; the blocked backend composes every gate.
                raise ValueError("Per-degree score biases take only the per-order query-key endpoint gate")
            # From the same FP32 activation as the coefficient generator: row shifts beta_r(t) for static
            # and query biases, or the query-key inputs x u, x v with b and a; pairs combine in the kernel.
            parameters.update(attention_operands(
                self.attention_score_bias, x, {name: score_bias[name] for name in bias_names}))
        kernel_coefficient_activation = self.attention_coefficient_activation
        if self.attention_coefficient_conditioning == "static" and (
                query_coefficients is not None or return_coefficients):
            if (self.attention_fixed_square_factors is not None
                    or "polynomial_square_factors" in parameters):
                raise ValueError("Static coefficient capture and override require direct polynomial coefficients")
            # Capture the represented coefficients after their one factorial division.
            # Capture alone preserves the ordinary static attention call exactly.
            # The fixed Taylor kernel represents its coefficients implicitly.
            represented = parameters.get("polynomial_coefficients")
            if represented is None:
                represented = jnp.asarray(
                    self.attention_fixed_coefficients if self.attention_fixed_coefficients is not None
                    else [1 / math.factorial(order) for order in range(self.attention_degree + 1)],
                    dtype=jnp.float32)
            effective_coefficients = jnp.broadcast_to(
                represented,
                (batch, seq_len, self.num_heads, self.attention_degree + 1))
            if query_coefficients is not None:
                effective_coefficients = query_coefficients
                parameters.pop("polynomial_coefficients", None)
                parameters["query_polynomial_coefficients"] = effective_coefficients
                # The core's tanh branch selects the general per-query polynomial
                # kernel. These are already effective coefficients: no activation,
                # generator, or further factorial scaling is applied to the table.
                kernel_coefficient_activation = "tanh"
                if key_factors is not None:
                    parameters["key_polynomial_factors"] = key_factors
        if self._coefficients_per_query(parameters, query_coefficients):
            # Coefficients per query (attention/pieces.py): static or fixed coefficients meet a forget gate, a score
            # transform, score biases or a prefactor as every query's equal coefficients on the general polynomial
            # kernel; broadcasting sums the per-query gradients back to the stored coefficients, as the log decay's
            # broadcast does to A_log and dt_bias. No activation, generator or factorial scale is applied again.
            represented = parameters.pop("polynomial_coefficients", None)
            if represented is None:
                # Fixed coefficients: the recipe's table, or the Taylor kernel's 1/r!.
                represented = jnp.asarray(
                    self.attention_fixed_coefficients if self.attention_fixed_coefficients is not None
                    else [1 / math.factorial(order) for order in range(self.attention_degree + 1)], dtype=jnp.float32)
            parameters["query_polynomial_coefficients"] = jnp.broadcast_to(
                represented.astype(jnp.float32), (batch, seq_len, self.num_heads, self.attention_degree + 1))
            kernel_coefficient_activation = "tanh"
        from scaffolds.transformer.attention.path import PATH_ARGUMENTS

        path_parameters = {name: parameters.pop(name, None) for name in PATH_ARGUMENTS}
        if (self.positional_encoding == "path") != all(value is not None for value in path_parameters.values()):
            if self.positional_encoding == "path" or any(value is not None for value in path_parameters.values()):
                raise ValueError("PaTH position encoding requires exactly its model-owned gate parameters")
        token_log_scales = None
        if self.qk_norm_token_scale != "none":
            # Bounded per-token log scale from the same attention input as the coefficient generators.
            token_log_scales = token_log_scale(x, token_scale_kernel, token_scale_amplitude, eps=self.qk_norm_eps)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.backend != "manual":
            q, k, v = (operand.astype(self.dtype) for operand in (q, k, v))
        q = q.reshape(batch, seq_len, self.num_heads, self.qk_head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.qk_head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.value_head_dim)
        if self.qk_norm:
            if self.qk_norm_type == "rms_norm":
                q, k = rms_score_operands(q, k, q_gain, k_gain, eps=self.qk_norm_eps)
                if self.qk_norm_head_scale:
                    q = scale_rms_queries(q, head_log_scale, token_log_scales)
            else:
                # Preserve the historical scaled-cosine arithmetic exactly.
                q, k = cosine_score_operands(q, k, log_scale, eps=self.qk_norm_eps)
        if self.positional_encoding == "path":
            result = self._path_attention(x, q.astype(self.dtype), k.astype(self.dtype), v, parameters,
                                          path_parameters, output_norm_scale, return_diagnostics, log_alpha)
            return (result, effective_coefficients) if return_coefficients else result
        if self.positional_encoding == "rope":
            positions = jnp.arange(seq_len)
            q = apply_rope(q, positions, self.rope_theta)
            k = apply_rope(k, positions, self.rope_theta)
        # "none": positions enter only through the causal mask (and a forget gate's decay).
        if self.backend != "manual":
            q, k = q.astype(self.dtype), k.astype(self.dtype)
        if coefficients is not None:
            if "coefficients" in parameters:
                raise ValueError("Coefficients were supplied twice")
            parameters = {**parameters, "coefficients": coefficients}
        if (self.attention_fixed_coefficients is not None and query_coefficients is None
                and "query_polynomial_coefficients" not in parameters):
            if any(name in parameters for name in ("coefficients", "polynomial_coefficients", "polynomial_square_factors",
                                                *DENOMINATOR_KINDS)):
                raise ValueError("Fixed polynomial coefficients cannot be combined with supplied attention coefficients")
            parameters["polynomial_coefficients"] = jnp.asarray(self.attention_fixed_coefficients, dtype=jnp.float32)
        if self.attention_fixed_square_factors is not None:
            if any(name in parameters for name in ("coefficients", "polynomial_coefficients", "polynomial_square_factors",
                                                *DENOMINATOR_KINDS)):
                raise ValueError("Fixed square factors cannot be combined with supplied attention coefficients")
            parameters["polynomial_square_factors"] = jnp.asarray(self.attention_fixed_square_factors, dtype=jnp.float32)
        if "polynomial_square_factors" in parameters and self.attention_fixed_square_factors is None:
            from scaffolds.transformer.attention.nonnegative_polynomial import static_square_factors

            parameters["polynomial_square_factors"] = static_square_factors(
                parameters["polynomial_square_factors"],
                self.attention_degree - self.attention_coefficient_input_power,
                parameterization=self.attention_static_sos_parameterization)
            if self.attention_coefficient_input_power:
                parameters["square_factor_input_power"] = self.attention_coefficient_input_power
        gate_statistics = None
        if return_diagnostics and gate_kernel is not None:
            from scaffolds.transformer.attention.input_dependent import (
                coefficient_gate_logits, coefficient_gate_statistics, project_coefficients,
            )

            # The same generator without its gate, so suppression can be told from compensation.
            ungated = project_coefficients(
                x, query_kernel, query_bias, amplitude=query_amplitude,
                activation=self.attention_coefficient_activation,
                amplitude_scale=self.attention_coefficient_amplitude_scale)
            gate_statistics = coefficient_gate_statistics(
                coefficient_gate_logits(x, gate_kernel, gate_bias), ungated,
                parameters["query_polynomial_coefficients"], q, k, gate_log_rate=gate_log_rate)
        if self.attention_score_transform != "identity":
            # Only transformed recipes pass it, so every other recipe's call is unchanged.
            parameters["score_transform"] = self.attention_score_transform
        if self.attention_score_prefactor != "none":
            parameters["score_prefactor"] = self.attention_score_prefactor
        if self.attention_exp_shift != "none":
            # Only a shifted recipe passes it, so every other recipe's call is unchanged.
            parameters["exp_shift"] = self.attention_exp_shift
        if self.attention_polynomial_shift != "none":
            parameters["polynomial_shift"] = self.attention_polynomial_shift
        result = attention_core(
            q, k, v, backend=self.backend, attention_degree=self.attention_degree,
            attention_normalization=self.attention_normalization,
            attention_coefficient_activation=kernel_coefficient_activation,
            quadratic_mode=self.quadratic_mode, **parameters,
            return_diagnostics=return_diagnostics, dropout=self.dropout,
            deterministic=deterministic, rngs=rngs, weights_dtype=x.dtype,
            blocked_implementation=self.attention_blocked_implementation,
        )
        y, diagnostics = result if return_diagnostics else (result, None)
        # A shifted output e^(-m) o and its row shifts m [batch, time, heads], which head RMSNorm takes.
        shifted_output = self.attention_exp_shift != "none" or self.attention_polynomial_shift != "none"
        y, output_shift = y if shifted_output else (y, None)
        if (return_diagnostics and self.attention_coefficient_input_normalization == "l2"
                and query_coefficients is None):
            from scaffolds.transformer.attention.input_dependent import coefficient_input_statistics

            diagnostics.update(coefficient_input_statistics(
                coefficient_source, coefficient_input, query_kernel,
                query_tanh_bias if self.attention_coefficient_activation == "pure_tanh" else None))
        if gate_statistics is not None:
            diagnostics.update(gate_statistics)
        if return_diagnostics and log_alpha is not None:
            diagnostics.update(_forget_gate_diagnostics(log_alpha))
        if return_diagnostics and token_log_scales is not None:
            diagnostics.update(jax.lax.stop_gradient({
                "token_scale_log_delta_mean": jnp.mean(token_log_scales, axis=(0, 1)),
                "token_scale_log_delta_std": jnp.std(token_log_scales, axis=(0, 1)),
                "token_scale_log_delta_min": jnp.min(token_log_scales, axis=(0, 1)),
                "token_scale_log_delta_max": jnp.max(token_log_scales, axis=(0, 1)),
            }))
        shifted = {} if output_shift is None else {"output_shift": output_shift}
        telemetry.record_attention_output(
            y, head_rms_norm_eps=self.rms_norm_eps if self.attention_output_scaling == "head_rms_norm" else None,
            **shifted)
        y = scale_attention_output(y, self.attention_output_scaling,
                                   output_norm_scale=output_norm_scale, eps=self.rms_norm_eps, **shifted,
                                   **self._head_rms_prescale())
        output = self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
        if return_coefficients:
            return output, effective_coefficients
        return (output, diagnostics) if return_diagnostics else output

    def _head_rms_prescale(self) -> dict:
        """Head RMSNorm's prescale as a keyword; only a prescaled recipe passes it, so every other call is
        unchanged."""
        prescale = self.attention_head_rms_prescale
        return {} if prescale == "none" else {"head_rms_prescale": prescale}

    def _coefficients_per_query(self, parameters, query_coefficients) -> bool:
        """Whether static or fixed polynomial coefficients run per query: when a forget gate, a score transform,
        score biases, a prefactor or the row-power shift meets them (static SoS factors keep their own kernel)."""
        return (query_coefficients is None and self.attention_coefficient_conditioning == "static"
                and self.attention_degree is not None and not self.attention_normalization
                and "polynomial_square_factors" not in parameters and self.attention_fixed_square_factors is None
                and self.score_mlp_width is None and self.attention_kernel_basis == "polynomial"
                and (self.attention_forget_gate or self.attention_score_transform != "identity"
                     or self.attention_score_bias != "none" or self.attention_score_prefactor != "none"
                     or self.attention_polynomial_shift != "none"))

    def _path_attention(self, x, q, k, v, parameters, path_parameters, output_norm_scale, return_diagnostics,
                        log_alpha=None):
        """PaTH-transported scores replace RoPE; the kernel, any forget gate, and output path are unchanged."""
        from scaffolds.transformer.attention.input_dependent import coefficient_statistics
        from scaffolds.transformer.attention.path import CHUNK_SIZE, path_attention, path_gates

        batch, seq_len = x.shape[:2]
        if self.num_kv_heads != self.num_heads:
            raise ValueError("PaTH attention requires one key head per query head")
        w, beta = path_gates(x, *(path_parameters[name] for name in (
            "path_w_down", "path_w_up", "path_w_conv", "path_beta_kernel", "path_beta_bias")), num_heads=self.num_heads)
        coefficients = parameters.pop("query_polynomial_coefficients", None)
        log_decay = parameters.pop("query_log_decay", None)
        if parameters.pop("query_decay_mode", "cumulative") != "cumulative":
            raise ValueError("PaTH attention supports only the cumulative forget-gate decay")
        if parameters:
            raise ValueError(f"PaTH attention does not accept {sorted(parameters)}")
        if self.attention_degree is None and coefficients is None:
            kernel = "softmax" if self.attention_normalization else "exp"
        elif (not self.attention_normalization and coefficients is not None
              and self.attention_coefficient_activation == "tanh"):
            kernel = "query_polynomial"
        else:
            raise ValueError("PaTH supports softmax, the unnormalized exponential, and tanh query-conditioned "
                             "polynomial attention, each with or without a forget gate")
        chunk = CHUNK_SIZE if seq_len % CHUNK_SIZE == 0 else seq_len
        if seq_len % CHUNK_SIZE and seq_len > CHUNK_SIZE:
            raise ValueError(f"PaTH needs a sequence length divisible by {CHUNK_SIZE} (or at most {CHUNK_SIZE})")
        reference = partial(path_attention, q, k, v, w, beta, kernel=kernel, coefficients=coefficients,
                            log_decay=log_decay, chunk_size=chunk, return_diagnostics=True)
        if self.backend == "fused":
            # "auto" is resolved before tracing; an unresolved request runs the JAX reference.
            from scaffolds.transformer.attention.backends.fla_path_attention import fused_path_attention

            y = fused_path_attention(q, k, v, w, beta, kernel=kernel, coefficients=coefficients, log_decay=log_decay)
            # Diagnostics come from the blocked reference, as for the other fused kernels.
            diagnostics = reference()[1] if return_diagnostics else {}
        else:
            y, diagnostics = reference()
        if coefficients is not None:
            diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
        diagnostics.update(jax.lax.stop_gradient({
            "path_beta_mean": jnp.mean(beta), "path_beta_min": jnp.min(beta), "path_beta_max": jnp.max(beta),
        }))
        if log_alpha is not None:
            diagnostics.update(_forget_gate_diagnostics(log_alpha))
        telemetry.record_attention_output(
            y, head_rms_norm_eps=self.rms_norm_eps if self.attention_output_scaling == "head_rms_norm" else None)
        y = scale_attention_output(y, self.attention_output_scaling,
                                   output_norm_scale=output_norm_scale, eps=self.rms_norm_eps,
                                   **self._head_rms_prescale())
        output = self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
        return (output, diagnostics) if return_diagnostics else output
