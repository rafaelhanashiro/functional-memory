"""Shared Transformer settings; experiments vary geometry and attention rules."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal


OFM_SOURCE_COMMIT = "8977f529e83efc688daf96cdcfd365ebbce2afa8"
ATTENTION_DEGREES = (0, 2, 4, 8, 16, None)


# Affine generators c = b + a*f(x W); pure generators c = f(x W), with no b or a.
AFFINE_ACTIVATIONS = ("tanh", "affine_sigmoid")
PURE_ACTIVATIONS = ("sigmoid", "pure_tanh")

@dataclass
class DataConfig:
    source: Literal["synthetic", "packed"] = "synthetic"
    dataset_name: str = ""
    dataset_config: str = ""
    split: str = "train"
    tokenizer_name: str = ""
    text_field: str = "text"
    train_manifest: str = ""
    eval_manifest: str = ""
    batch_size: int = 2
    sequence_length: int = 65
    shuffle_seed: int = 0
    synthetic_vocab_size: int = 512


@dataclass
class ModelConfig:
    kind: Literal["transformer"] = "transformer"
    vocab_size: int = 512
    max_seq_len: int = 64
    num_layers: int = 2
    hidden_size: int = 128
    num_heads: int = 4
    num_kv_heads: int = 4
    qk_head_dim: int = 32
    value_head_dim: int = 32
    intermediate_size: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    # Opt-in FP32 backbone normalization arithmetic; False preserves historical BF16 recipes.
    rms_norm_fp32: bool = False
    dropout: float = 0.0
    # Retained for recipe compatibility; this scaffold requires separate weights.
    tie_embeddings: bool = False
    remat: bool = False
    attention_backend: Literal["auto", "blocked", "fused", "mosaic", "manual", "xla", "cudnn"] = "auto"
    # How the blocked backend runs dot-product attention with RoPE or no positional encoding.
    # "scan" (historical) evaluates 128-query blocks against every key, summing dK/dV across
    # blocks in the input dtype; "strips" evaluates the same arithmetic over causal 512-query
    # strips with FP32 dK/dV sums (attention/backends/reference.py). Other resolved backends
    # ignore it. Fresh runs that can train on the blocked backend use strips
    # (fresh_recipe_uses_blocked_strips).
    attention_blocked_implementation: Literal["scan", "strips"] = "scan"
    attention_degree: int | None = None
    attention_normalization: bool = True
    # Unnormalized dot-product attention only: rescale each head's output
    # o_t = sum_(j<=t) k(s_tj) v_j before the output projection. "inverse_position"
    # divides by the number of visible keys t; "head_rms_norm" applies an RMSNorm over
    # the head's value channels with a learned gain per layer, head, and channel.
    attention_output_scaling: Literal["none", "inverse_position", "head_rms_norm"] = "none"
    attention_mechanism: Literal["dot_product", "gla", "gated_deltanet"] = "dot_product"
    # Gated DeltaNet's causal depthwise short-convolution width (FLA and paper default 4).
    gdn_conv_size: int = 4
    # Independent erasure learns a separate per-token/head sigmoid gate, initially
    # copied from beta; tied retains GDN's shared erase/write gate.
    gdn_erase_mode: Literal["tied", "independent"] = "tied"
    # Signed writes retain a separate sigmoid erase gate. Historical recipes use sigmoid.
    gdn_write_activation: Literal["sigmoid", "tanh"] = "sigmoid"
    # Degrees two and three compose polynomial memory with the original-key-space erase map.
    gdn_polynomial_degree: Literal[1, 2, 3] = 1
    gla_gate_rank: int = 16
    gla_gate_tau: float = 16.0
    gla_chunk_size: int = 64
    gla_inner_chunk_size: int = 16
    gla_use_rope: bool = True
    quadratic_mode: Literal["fixed", "positive", "unrestricted"] = "fixed"
    quadratic_sharing: Literal["global", "per_head"] = "global"
    # Degree-m kernels may learn every coefficient c_0..c_m of sum_r c_r s^r,
    # starting from the Taylor values 1/r!, with or without row normalization.
    # Both forms support fused kernels and the blocked reference.
    attention_coefficients: Literal["fixed", "learned"] = "fixed"
    # Optional fixed ascending monomial coefficients c_0..c_m. None preserves
    # the Taylor kernel; explicit values use the generic polynomial backends.
    attention_fixed_coefficients: tuple[float, ...] | None = None
    # Optional fixed factors [q1_0..q1_{m/2}, q2_0..q2_{m/2-1}] for
    # p(s) = q1(s)^2 + q2(s)^2. Evaluated without expanding near real roots.
    attention_fixed_square_factors: tuple[float, ...] | None = None
    attention_coefficient_sharing: Literal["global", "per_head"] = "global"
    # Optional ascending active monomial orders for static or diagonal tanh-query coefficients.
    # Store only these entries; all omitted orders remain exactly zero.
    attention_coefficient_orders: tuple[int, ...] | None = None
    # Query coefficients use the incoming attention activation, per layer/head. "query_key" also
    # multiplies each key's degree-r term by f_r(x_i) = 1 + a_r tanh(w_r^T x_i) from the key token.
    attention_coefficient_conditioning: Literal["static", "query", "query_key"] = "static"
    # Legacy identity is degree-one affine. Affine generators: tanh is b + a*tanh(x W) and
    # affine_sigmoid is b + a*sigmoid(x W). Pure generators have no b or a: sigmoid is
    # c_r = sigmoid(x w_r) and pure_tanh is c_r = tanh(x w_r) (dense: c_r = sum_j V_rj f(x w_j),
    # V starting at I). pure_tanh needs a random W start, since tanh(0) = 0 zeroes every coefficient.
    attention_coefficient_activation: Literal["identity", "tanh", "sigmoid", "pure_tanh", "affine_sigmoid"] = "identity"
    # Normalize only the coefficient-generator input. By default it includes the shared
    # pre-attention RMSNorm and its channel gain; pre_rmsnorm selects the raw block input.
    # Q/K/V, scores, and W's parameterization are unchanged by either choice.
    # "l2" uses x / max(||x||_2, 1e-6) in FP32; "none" preserves historical arithmetic.
    attention_coefficient_input_normalization: Literal["none", "l2"] = "none"
    attention_coefficient_input_source: Literal["post_rmsnorm", "pre_rmsnorm"] = "post_rmsnorm"
    # Forgetting Transformer decay: each query-conditioned polynomial weight is multiplied by
    # prod_(j=i+1..t) alpha_j, with Gated DeltaNet's per-head Mamba-2 gate
    # log alpha_t = -exp(A_log) softplus(x_t w_alpha + dt_bias) and its initialization.
    attention_forget_gate: bool = False
    # Cumulative FoX uses intervening gates; query_distance uses only the current query's
    # alpha_t raised to the key distance t-i, with the identical gate parameters. On softmax and
    # the unnormalized exponential (input-form softplus gate, blocked backend) it adds
    # (t - i) log alpha_t to every score: ALiBi with a learned, input-dependent slope per query
    # (with query_key conditioning, per query-key pair: (t - i) log alpha_ti).
    attention_forget_gate_mode: Literal["cumulative", "query_distance"] = "cumulative"
    # "input": Gated DeltaNet's token-dependent gate above. "constant": log alpha_t = -exp(A_log)
    # per layer/head, an ALiBi-style decay exp(-lambda_h (t - i)) with lambda_h starting at
    # ALiBi's slopes 2^(-8 h / H); only the log rate is learned. On softmax and the exponential
    # it is ALiBi's score bias -lambda_h (t - i). "sigmoid": the Forgetting Transformer's
    # original alpha_t = sigmoid(x_t w_f + b_f), w_f ~ N(0, 0.02^2) and b_f = 0.
    attention_forget_gate_form: Literal["input", "constant", "sigmoid"] = "input"
    # The constant form's rate: "learned" as above, or "fixed" at ALiBi's slopes 2^(-8 h / H) in
    # every layer, with no trainable parameter. Recipes without the field learn it.
    attention_forget_gate_constant_rate: Literal["learned", "fixed"] = "learned"
    # Endpoint-only distance gates can read both endpoints, or bound the query-only rate. "query_key"
    # (input-form query_distance gates of query polynomials, softmax, and the exponential) gives each
    # pair log alpha_ti = -exp(A_log) softplus(x_t w_alpha + x_i w_key + dt_bias), w_key starting at
    # zero; the bounded sigmoid rate is qualified for query polynomials only. "none" reads no token:
    # lambda = exp(A_log) softplus(dt_bias) per layer/head (and order), the input form without its
    # projection and with the same draws, the query-distance gate of learned static coefficients.
    attention_forget_gate_conditioning: Literal["query", "query_key", "none"] = "query"
    attention_forget_gate_rate_activation: Literal["softplus", "sigmoid"] = "softplus"
    # Independent input-form gate for every polynomial order, including zero: query-distance
    # (blocked or mosaic), or cumulative with query conditioning (blocked, or mosaic on H100), where
    # order r's term is multiplied by exp(L_tr - L_ir) with its own L_tr = sum_(j<=t) log alpha_jr.
    # Every order starts as a copy of the corresponding shared gate.
    attention_forget_gate_per_order: bool = False
    # Apply optim.weight_decay to the coefficient projections: the query W, the coefficient
    # gate's U, and, under query_key conditioning, the key W_k. b, a, the key amplitude, the
    # gate bias, and V stay excluded.
    attention_coefficient_kernel_weight_decay: bool = False
    # Dense readout mixes all degree+1 tanh or sigmoid features into every output coefficient.
    # Diagonal preserves the historical one-feature-per-coefficient generator.
    attention_coefficient_readout: Literal["diagonal", "dense"] = "diagonal"
    # Multiply only the dense input-dependent readout by this fixed scale; the
    # Taylor bias and its factorial scaling remain unchanged.
    attention_coefficient_readout_scale: float = 1.0
    # Dense readout's tanh feature count K, independent of the degree; null means degree+1.
    # V then has shape [degree+1, K] per layer/head; K other than degree+1 starts V at zero.
    attention_coefficient_feature_count: int | None = None
    # Tanh coefficients start at b = 1/r!, a = amplitude_initial, and W with independent
    # N(0, projection_initial_std^2) entries (0 keeps W at zero). a = 0 with W = 0 never learns.
    attention_coefficient_amplitude_initial: float = 1.0
    attention_coefficient_projection_initial_std: float = 0.0
    # Tanh coefficients may learn an offset inside the tanh, c_r = b_r + a_r tanh(w_r^T x_t + beta_r),
    # starting at beta_r = 0, so a constant shift of the pre-activation need not come through w_r.
    attention_coefficient_tanh_bias: bool = False
    # "sigmoid" gates every order independently per layer/head: c~_r(x_t) = 2 g_r(x_t) c_r(x_t),
    # g_r(x_t) = sigmoid(U_r^T x_t + beta_r), with U and this gate bias beta starting at zero
    # (not the inner tanh bias, which cannot be combined with it). The fixed factor 2 makes the
    # multiplier 2 g_r exactly 1 at the start, so the ungated function and its derivatives in
    # W, a and b are preserved; 2 g_r ranges over (0, 2), suppressing or amplifying an order.
    # "gdn" multiplies instead by Gated DeltaNet's decay without a distance exponent,
    # alpha_r(x_t) = exp(-exp(A_log_r) softplus(w_r^T x_t + dt_bias_r)) in (0, 1], per
    # layer/head/order. w, A_log and dt_bias take the input-form forget gate's draws, made
    # once per layer/head and copied to every order: all orders start with the same alpha,
    # which head_rms_norm output scaling cancels up to its epsilon. w decays like U.
    attention_coefficient_gate: Literal["none", "sigmoid", "gdn"] = "none"
    # Per-degree score biases (attention/score_bias.py): order r >= 1 of diagonal tanh query
    # coefficients acts on s_ti + beta_r(t,i), order zero unshifted: "static" beta_r = b_r, "query"
    # b_r + a_r tanh(x_t u_r), "query_key" b_r + a_r tanh(x_t u_r + x_i v_r), per layer/head/order,
    # starting at b = 0, a = 1, u = v = 0 (no random draws); u and v decay. With no forget gate or
    # the per-order query-key endpoint gate (each order's gate multiplies its own shifted power).
    # Trains on H100's Mosaic score-bias kernels or on the blocked backend's causal strips, which run
    # on any GPU; the blocked scan is their reference for tests and diagnostics, and allocation and
    # acceptance refuse it for training. Every other backend rejects them.
    attention_score_bias: Literal["none", "static", "query", "query_key"] = "none"
    # The polynomial's input (attention/score_transform.py): every order of diagonal tanh query
    # coefficients reads z_ti = phi(s_ti) in place of the score s_ti, W(t,i) = sum_r c_r(x_t) z_ti^r.
    # "identity" (z = s) is every historical recipe; "softplus" z = log(1 + exp(s)), so a very negative
    # score gives z near 0 and a weight near c_0; "abs" z = |s|. No parameter is added. Runs on the blocked
    # backend (training on its causal strips) or on the Mosaic transformed query-polynomial kernels
    # (mosaic_score_transform_supported), without a forget gate or score biases; every other backend
    # rejects it, and auto never resolves it.
    attention_score_transform: Literal["identity", "softplus", "abs"] = "identity"
    # The start of the order-0 offset b_0 of diagonal tanh query coefficients, c_0(x_t) = b_0 + a_0
    # tanh(x_t w_0), in place of Taylor's 1 (null). With attention_coefficient_amplitude_initial 0, c_0
    # starts at exactly this value: 0 with the softplus transform starts the kernel at
    # sum_(r=1..m) z^r/r!, the Taylor polynomial of e^z - 1 = e^s. No other array changes.
    attention_coefficient_constant_initial: float | None = None
    # Dot-product position encoding: "rope" rotates queries and keys; "path" replaces it with
    # PaTH (arXiv:2505.16381), scores transported through token Householder products
    # I - beta_t w_t w_t^T. w_t comes from a rank-path_w_rank projection, a width-path_conv_size
    # causal SiLU convolution, and L2 normalization per head; beta_t = 2 sigmoid(x_t W + b).
    positional_encoding: Literal["rope", "path", "none"] = "rope"
    path_w_rank: int = 32
    path_conv_size: int = 3
    # "factorial" divides the tanh amplitude of order r by r!, c_r = b_r + (a_r/r!) tanh(w_r^T x_t),
    # so under Adam's unit-scale steps order r's modulation moves at lr/r!, matching b_r = 1/r!.
    # Sigmoid coefficients become sigmoid(w_r^T x_t)/r! (dense: V's row r divided by r!).
    attention_coefficient_amplitude_scale: Literal["none", "factorial"] = "none"
    # "factorial" stores order r's input-independent coefficient (static c_r, or the query
    # bias b_r) as r! times its value, c_r = c~_r / r!, so under Adam it moves at lr/r!
    # and its gradient carries s^r / r! rather than s^r. It starts at the same values.
    attention_coefficient_scale: Literal["none", "factorial"] = "none"
    # The score function's basis. "polynomial" (historical): monomials, sum_r c_r s^r of degree
    # attention_degree. "bspline": f(s) = sum_j theta_j B_j(s), the fixed B-splines of degree
    # attention_degree on the distinct attention_spline_knots, clamped (each endpoint repeated
    # degree + 1 times), so len(knots) + degree - 1 coefficients (attention/spline.py). Outside
    # the end knots f continues its first or last polynomial piece. Requires learned direct
    # per_head coefficients, static or diagonal tanh query-conditioned (theta_j(x_t) =
    # b_j + a_j tanh(w_j^T x_t)), Taylor initialization (theta starts at the exact B-spline
    # coefficients of T_degree), no factorial scales or selected orders, unnormalized
    # dot-product attention without a forget gate or PaTH, on the blocked backend.
    attention_kernel_basis: Literal["polynomial", "bspline"] = "polynomial"
    attention_spline_knots: tuple[float, ...] | None = None
    # "direct" learns c_0..c_m; "sum_of_squares" learns q1, q2 with p = q1^2 + q2^2
    # (even m; normalized static coefficients, or unnormalized tanh query coefficients, whose
    # factors are generated per query and expanded to c_r(x_t)), so the kernel is nonnegative
    # by construction;
    # "rational" (row-normalized only) learns a numerator a_0..a_m and a denominator
    # 1 + b_1 s + ... + b_m s^m, so the weights are p_a(s) / sum_j p_b(s_j).
    attention_coefficient_parameterization: Literal["direct", "sum_of_squares", "rational"] = "direct"
    # Initial learned coefficients: "taylor" starts every learned polynomial at
    # sum_r s^r / r! (a rational kernel then equals normalized Taylor attention);
    # "linear" starts the numerator at s and the denominator at 1 (a_1 = 1, b_0 = 1,
    # all other coefficients 0), so a rational kernel starts at linear attention
    # divided by the number of visible keys. Not available with sum_of_squares.
    # "hermite" starts tanh query coefficients at exp(1/2) sum_{n<=m} He_n(s) / n!.
    attention_coefficient_initialization: Literal["taylor", "linear", "hermite"] = "taylor"
    # Rational kernels only: the denominator's degree (null: attention_degree) and form.
    # "direct" learns b_1..b_d of 1 + sum_r b_r s^r; "sum_of_squares" (even d) learns
    # u_1..u_k, w_1..w_k of (1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2, a nonnegative
    # denominator with constant 1 (attention/rational_polynomial.py).
    attention_denominator_degree: int | None = None
    attention_denominator_parameterization: Literal["direct", "sum_of_squares"] = "direct"
    # "fixed" pins the denominator's constant at 1; "learned" learns it with the other
    # coefficients (or factors), leaving the common scale of numerator and denominator free.
    attention_denominator_constant: Literal["fixed", "learned"] = "fixed"
    # Historical cosine scores: s = exp(gamma) * cos(q, k) with a
    # learned log scale gamma per layer and head (or one shared), initialized at
    # log(qk_norm_initial_scale); null selects sqrt(qk_head_dim).
    # QK-norm is the default for dot-product attention: unset resolves to on, except for
    # GLA (rejects it) and degree 0 (never reads Q/K). Plain scores need qk_norm: false.
    qk_norm: bool | None = None
    # rms_norm uses separate Q/K channel gains per layer, shared across heads,
    # applied before RoPE. Set sharing=per_layer and leave initial_scale=null.
    qk_norm_type: Literal["cosine_log_scale", "rms_norm"] = "cosine_log_scale"
    qk_norm_sharing: Literal["global", "per_head", "per_layer"] = "per_head"
    qk_norm_initial_scale: float | None = None
    qk_norm_eps: float = 1e-6
    # Add a positive scalar exp(gamma) per layer/head after RMSNorm's channel
    # gains. gamma starts at zero; the existing channel gains remain per layer.
    qk_norm_head_scale: bool = False
    # "cosine" multiplies each query's scale by exp(a_h cos(w_h, x_t)), a per-token log scale
    # bounded by |a_h| and read from the attention input x_t. a_h starts at zero, so the
    # model starts exactly at its static head scale; w_h is drawn N(0, 1/hidden_size) and
    # only its direction matters. Requires qk_norm_head_scale.
    qk_norm_token_scale: Literal["none", "cosine"] = "none"
    # Opt in to AdamW decay on the token-scale direction w only. Its amplitude a
    # remains exempt, and existing recipes retain their no-decay behavior.
    qk_norm_token_scale_kernel_weight_decay: bool = False
    # Weight initialization: the scaffold's Flax defaults, or an architecture's reference
    # recipe ("transformer_reference" for dot-product attention, "gla_reference" for GLA);
    # see models/initialization.py.
    initialization: Literal["flax_default", "transformer_reference", "transformer_truncated_reference", "gla_reference",
                            "gated_deltanet_reference"] = "flax_default"

    def mosaic_supported(self) -> bool:
        """Whether a Mosaic GPU kernel implements this recipe's attention: normalized fixed
        Taylor, tanh query-conditioned polynomial attention, or an unnormalized polynomial with
        static (fixed or learned) coefficients, at its widths and degrees."""
        from scaffolds.transformer.attention.backends.mosaic.api import VALUE_WIDTH, WIDTHS, query_degree_supported

        degree = self.attention_degree
        integer_degree = isinstance(degree, int) and not isinstance(degree, bool)
        if getattr(self, "attention_kernel_basis", "polynomial") != "polynomial":
            return False  # No Mosaic kernel evaluates a B-spline basis.
        if getattr(self, "attention_score_bias", "none") != "none":
            return False  # Score biases run on their own kernels (mosaic_score_bias_supported).
        if getattr(self, "attention_score_transform", "identity") != "identity":
            return False  # Transforms run on their own kernels (mosaic_score_transform_supported).
        taylor = (self.attention_normalization and integer_degree and degree >= 1
                  and self.attention_coefficients == "fixed" and self.quadratic_mode == "fixed"
                  and self.attention_fixed_coefficients is None and self.attention_fixed_square_factors is None
                  and self.attention_coefficient_parameterization == "direct"
                  and self.attention_coefficient_conditioning == "static")
        query = (not self.attention_normalization and self.attention_coefficient_conditioning in ("query", "query_key")
                 and self.attention_coefficient_activation in (*AFFINE_ACTIVATIONS, *PURE_ACTIVATIONS)
                 and integer_degree and not self.attention_forget_gate
                 and query_degree_supported(degree, self.qk_head_dim, self.max_seq_len))
        # Static coefficients run on the query-coefficient kernel with every query's coefficients equal;
        # with a distance gate they run on the decay kernels instead (mosaic_decay_supported).
        static = (not self.attention_normalization and self.attention_coefficient_conditioning == "static"
                  and self.quadratic_mode == "fixed" and self.attention_coefficient_parameterization == "direct"
                  and self.attention_fixed_square_factors is None and integer_degree
                  and not self.attention_forget_gate
                  and query_degree_supported(degree, self.qk_head_dim, self.max_seq_len))
        return (self.attention_mechanism == "dot_product" and (taylor or query or static)
                and getattr(self, "positional_encoding", "rope") == "rope"
                and self.qk_head_dim in WIDTHS and self.value_head_dim == VALUE_WIDTH)

    def mosaic_decay_supported(self) -> bool:
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

        degree = self.attention_degree
        integer_degree = isinstance(degree, int) and not isinstance(degree, bool)
        if getattr(self, "attention_score_bias", "none") != "none":
            return False  # Score biases run on their own kernels (mosaic_score_bias_supported).
        if getattr(self, "attention_score_transform", "identity") != "identity":
            return False  # Transforms take no forget gate.
        input_free = self.attention_forget_gate_conditioning == "none"
        static = (self.attention_coefficient_conditioning == "static" and self.attention_coefficients == "learned"
                  and self.attention_coefficient_parameterization == "direct")
        query = (self.attention_coefficient_conditioning == "query"
                 and self.attention_coefficient_activation in (*AFFINE_ACTIVATIONS, *PURE_ACTIVATIONS))
        coefficients = static if input_free else query or (static and self.attention_forget_gate_conditioning == "query_key")
        return (self.attention_mechanism == "dot_product" and not self.attention_normalization
                and getattr(self, "attention_kernel_basis", "polynomial") == "polynomial"
                and coefficients
                and self.attention_forget_gate
                and (self.attention_forget_gate_mode == "query_distance"
                     or self.attention_forget_gate_mode == "cumulative" and not input_free)
                and self.attention_forget_gate_form == "input" and integer_degree
                and query_degree_supported(degree, self.qk_head_dim, self.max_seq_len,
                                           gate="query" if input_free else self.attention_forget_gate_conditioning,
                                           per_order=self.attention_forget_gate_per_order,
                                           mode=self.attention_forget_gate_mode)
                and self.positional_encoding in ("rope", "none")
                and self.qk_head_dim in WIDTHS and self.value_head_dim == VALUE_WIDTH)

    def mosaic_score_bias_supported(self) -> bool:
        """Whether the Mosaic GPU score-bias kernels implement this recipe: per-degree score biases of
        the tanh query-coefficient polynomial, with no forget gate or the per-order query-key endpoint
        gate (``mosaic_score_bias_problem``). Only an explicit ``mosaic`` backend selects them; ``auto``
        never does, and score-bias recipes reject it."""
        return self.mosaic_score_bias_problem() is None

    def mosaic_score_bias_problem(self) -> str | None:
        """Why the Mosaic score-bias kernels cannot run this recipe, or None. They take Q/K widths 8-128,
        value width 64, and degree 1-12 on H100, where their forward, dQ and dK/dV kernels must fit in
        shared memory (``score_bias_supported``); their entry points refuse other GPUs before tracing.
        Backend resolution checks the sequence length (a multiple of 128 rows)."""
        from scaffolds.transformer.attention.backends.mosaic.api import VALUE_WIDTH, WIDTHS

        if getattr(self, "attention_score_bias", "none") == "none":
            return "the recipe has no score biases"
        problem = _score_bias_problem(self)
        if problem is None and self.qk_head_dim not in WIDTHS:
            problem = f"Q/K width {self.qk_head_dim} is not one of {WIDTHS}"
        if problem is None and self.value_head_dim != VALUE_WIDTH:
            problem = f"value width {self.value_head_dim} is not {VALUE_WIDTH}"
        if problem is None:
            from scaffolds.transformer.attention.backends.mosaic.score_bias import (
                HOPPER_TIME_MULTIPLE, SHIFT_OF_MODE, implementation_name, score_bias_supported,
            )

            gate = "endpoint" if self.attention_forget_gate else "none"  # the per-order query-key gate
            shift = SHIFT_OF_MODE.get(self.attention_score_bias)  # row (static, query) or pair (query_key)
            # Shared memory does not depend on the length; ask at the recipe's length in whole 128-row tiles.
            rows = -(-self.max_seq_len // HOPPER_TIME_MULTIPLE) * HOPPER_TIME_MULTIPLE
            if shift is None:
                problem = f"no kernel takes score bias {self.attention_score_bias!r}"
            elif not score_bias_supported(self.attention_degree, self.qk_head_dim, rows, gate=gate, shift=shift):
                problem = (f"H100's {implementation_name(gate, shift)} kernels do not hold degree "
                           f"{self.attention_degree} at Q/K width {self.qk_head_dim} in shared memory")
        return problem

    def mosaic_score_transform_supported(self) -> bool:
        """Whether the Mosaic transformed query-polynomial kernels implement this recipe: the tanh query
        polynomial evaluated at a transformed score (``mosaic_score_transform_problem``). Only an explicit
        ``mosaic`` backend selects them; ``auto`` never does, and transformed recipes reject it."""
        return self.mosaic_score_transform_problem() is None

    def mosaic_score_transform_problem(self) -> str | None:
        """Why the Mosaic transformed query-polynomial kernels cannot run this recipe, or None. They are the
        query polynomial's kernels with phi(s) evaluated in their tiles, on H100 and B300 alike, so they take
        the recipes those kernels take (``mosaic_supported``'s query case: RoPE, Q/K widths 8-128, value width
        64, the degrees ``query_degree_supported`` fits at the recipe's length) with diagonal tanh query
        coefficients and no forget gate or score biases. Backend resolution checks the GPU and length."""
        from scaffolds.transformer.attention.backends.mosaic.api import VALUE_WIDTH, WIDTHS, query_degree_supported

        transform = getattr(self, "attention_score_transform", "identity")
        if transform == "identity":
            return "the recipe has no score transform"
        problem = _query_coefficient_problem(self)
        if problem is None and self.positional_encoding != "rope":
            problem = "the Mosaic query-polynomial kernels run RoPE recipes"
        if problem is None and (self.attention_forget_gate or self.attention_score_bias != "none"):
            problem = "it takes no forget gate or score biases"
        if problem is None and self.qk_head_dim not in WIDTHS:
            problem = f"Q/K width {self.qk_head_dim} is not one of {WIDTHS}"
        if problem is None and self.value_head_dim != VALUE_WIDTH:
            problem = f"value width {self.value_head_dim} is not {VALUE_WIDTH}"
        if problem is None and not query_degree_supported(self.attention_degree, self.qk_head_dim, self.max_seq_len,
                                                          transform=transform):
            problem = (f"the kernels do not hold degree {self.attention_degree} at Q/K width {self.qk_head_dim} "
                       f"and {self.max_seq_len} tokens in shared memory")
        return problem

    def path_fused_supported(self) -> bool:
        """Whether FLA's fused PaTH kernels cover this recipe's widths and degree (any GPU, length)."""
        from scaffolds.transformer.attention.backends.fla_path_attention import HEAD_DIMS, MAX_DEGREE

        degree = self.attention_degree
        return (self.positional_encoding == "path" and self.qk_head_dim in HEAD_DIMS
                and self.value_head_dim == self.qk_head_dim
                and (degree is None or (isinstance(degree, int) and 1 <= degree <= MAX_DEGREE)))

    def path_kernel(self) -> str | None:
        """The PaTH attention kernel this recipe uses, or None for RoPE recipes."""
        if self.positional_encoding != "path":
            return None
        if self.attention_degree is None:
            return "softmax" if self.attention_normalization else "exp"
        return "query_polynomial"

    def __post_init__(self):
        if self.kind != "transformer":
            raise ValueError("The shared scaffold supports only kind=transformer")
        if self.tie_embeddings is not False:
            raise ValueError("model.tie_embeddings must be false: the shared scaffold requires untied embeddings")
        if not isinstance(self.rms_norm_fp32, bool):
            raise ValueError("model.rms_norm_fp32 must be a boolean")
        for name in (
            "vocab_size", "max_seq_len", "num_layers", "hidden_size", "num_heads",
            "num_kv_heads", "qk_head_dim", "value_head_dim", "intermediate_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.num_kv_heads != self.num_heads:
            raise ValueError("This scaffold uses full MHA: num_kv_heads must equal num_heads")
        if self.attention_mechanism not in ("dot_product", "gla", "gated_deltanet"):
            raise ValueError(f"Unknown attention mechanism: {self.attention_mechanism}")
        uses_rope = (self.attention_mechanism == "dot_product"
                     or (self.attention_mechanism == "gla" and self.gla_use_rope))
        if self.qk_head_dim % 2 and uses_rope:
            raise ValueError("model.qk_head_dim must be even for RoPE")
        if self.attention_backend not in ("auto", "blocked", "fused", "mosaic", "manual", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if self.positional_encoding not in ("rope", "path", "none"):
            raise ValueError(f"Unknown positional encoding: {self.positional_encoding}")
        if self.attention_blocked_implementation not in ("scan", "strips"):
            raise ValueError(f"Unknown blocked attention implementation: {self.attention_blocked_implementation}")
        if self.attention_blocked_implementation == "strips" and (
                self.attention_mechanism != "dot_product" or self.positional_encoding == "path"):
            raise ValueError("model.attention_blocked_implementation strips applies to dot-product attention with RoPE "
                             "or no positional encoding; PaTH, GLA, and Gated DeltaNet have their own blocked code")
        _validate_kernel_basis(self)
        for name in ("path_w_rank", "path_conv_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.positional_encoding == "path":
            # Softmax, or the unnormalized exponential.
            exponential = (self.attention_degree is None and self.attention_coefficients == "fixed"
                           and self.quadratic_mode == "fixed")
            query_tanh = (not self.attention_normalization and self.attention_coefficient_conditioning == "query"
                          and self.attention_coefficient_activation == "tanh")
            if self.attention_mechanism != "dot_product" or not (exponential or query_tanh):
                raise ValueError("PaTH position encoding supports softmax, unnormalized exponential, and tanh "
                                 "query-conditioned unnormalized polynomial attention")
            if self.attention_backend not in ("auto", "blocked", "fused"):
                raise ValueError("PaTH attention runs on model.attention_backend blocked (its chunked JAX "
                                 "implementation), fused (FLA's Triton kernels), or auto")
            if self.attention_backend == "fused" and not self.path_fused_supported():
                raise ValueError("Fused PaTH needs equal qk_head_dim and value_head_dim in 16, 32, 64, 128 "
                                 "and polynomial degree 1-4")
        if (self.attention_backend == "mosaic" and self.attention_score_bias == "none"
                and self.attention_score_transform == "identity"
                and not (self.mosaic_supported() or self.mosaic_decay_supported())):
            # Score-bias and score-transform recipes name their own reasons (_validate_score_bias,
            # _validate_score_transform).
            raise ValueError("Backend 'mosaic' supports normalized fixed Taylor attention, and tanh "
                             "query-conditioned or static-coefficient unnormalized polynomial attention of "
                             "degree 1-12, with qk_head_dim 8, 16, 32, 64, or 128 and value_head_dim 64; "
                             "query-conditioned attention may add a query-distance forget gate or per-order "
                             "cumulative query gates (H100 only), and learned static coefficients the input-free "
                             "one (attention_forget_gate_conditioning: none), where the decay kernels fit the "
                             "degree and width")
        if not isinstance(self.attention_normalization, bool):
            raise ValueError("model.attention_normalization must be a boolean")
        if self.attention_output_scaling not in ("none", "inverse_position", "head_rms_norm"):
            raise ValueError(f"Unknown attention output scaling: {self.attention_output_scaling}")
        if self.attention_output_scaling != "none" and self.attention_mechanism != "dot_product":
            raise ValueError("model.attention_output_scaling applies only to dot-product attention")
        if self.attention_output_scaling == "inverse_position" and self.attention_normalization:
            raise ValueError("model.attention_output_scaling inverse_position applies only to unnormalized attention")
        if self.attention_mechanism == "gla":
            if self.attention_degree is not None or self.quadratic_mode != "fixed":
                raise ValueError("GLA cannot use Taylor degrees or learned quadratic modes")
            if not self.attention_normalization:
                raise ValueError("attention_normalization is a dot-product option; GLA requires its default true")
            if self.attention_backend not in ("auto", "blocked", "fused"):
                raise ValueError("GLA requires backend='auto', 'blocked' or 'fused'")
        if self.attention_mechanism == "gated_deltanet":
            if self.attention_degree is not None or self.quadratic_mode != "fixed":
                raise ValueError("Gated DeltaNet cannot use Taylor degrees or learned quadratic modes")
            if not self.attention_normalization or self.attention_coefficients != "fixed":
                raise ValueError("Gated DeltaNet uses its own recurrence: no normalization or coefficient options")
            if self.attention_backend not in ("auto", "blocked", "fused"):
                raise ValueError("Gated DeltaNet requires backend='auto', 'blocked' or 'fused'")
        if not isinstance(self.gdn_conv_size, int) or isinstance(self.gdn_conv_size, bool) or self.gdn_conv_size <= 0:
            raise ValueError("model.gdn_conv_size must be a positive integer")
        if self.gdn_erase_mode not in ("tied", "independent"):
            raise ValueError("model.gdn_erase_mode must be tied or independent")
        if self.gdn_erase_mode != "tied" and self.attention_mechanism != "gated_deltanet":
            raise ValueError("model.gdn_erase_mode independent requires gated_deltanet")
        if self.gdn_write_activation not in ("sigmoid", "tanh"):
            raise ValueError("model.gdn_write_activation must be sigmoid or tanh")
        if self.gdn_write_activation == "tanh":
            if self.attention_mechanism != "gated_deltanet" or self.gdn_erase_mode != "independent":
                raise ValueError("model.gdn_write_activation tanh requires gated_deltanet with independent erasure")
        if type(self.gdn_polynomial_degree) is not int or self.gdn_polynomial_degree not in (1, 2, 3):
            raise ValueError("model.gdn_polynomial_degree must be 1, 2, or 3")
        if self.gdn_polynomial_degree > 1:
            if self.attention_mechanism != "gated_deltanet" or self.gdn_erase_mode != "independent":
                raise ValueError("model.gdn_polynomial_degree > 1 requires gated_deltanet with independent erasure")
        for name in ("gla_gate_rank", "gla_chunk_size", "gla_inner_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if (not isinstance(self.gla_gate_tau, (int, float)) or isinstance(self.gla_gate_tau, bool)
                or not math.isfinite(self.gla_gate_tau) or self.gla_gate_tau <= 0):
            raise ValueError("model.gla_gate_tau must be finite and positive")
        if not isinstance(self.gla_use_rope, bool):
            raise ValueError("model.gla_use_rope must be a boolean")
        if self.attention_mechanism == "gla" and self.attention_backend == "fused":
            if (self.gla_chunk_size, self.gla_inner_chunk_size) != (64, 16):
                raise ValueError("Fused GLA requires chunk size 64 and inner chunk size 16")
        if not self.attention_normalization:
            if self.quadratic_mode != "fixed":
                raise ValueError("Unnormalized attention requires the exponential or a fixed Taylor kernel")
            if self.attention_backend not in ("auto", "blocked", "fused", "mosaic"):
                raise ValueError("Unnormalized attention requires backend='auto', 'blocked', 'fused' or 'mosaic'")
        if self.attention_backend in ("manual", "xla", "cudnn") and self.attention_degree is not None:
            raise ValueError(f"Backend {self.attention_backend!r} supports exact softmax only")
        if self.quadratic_mode not in ("fixed", "positive", "unrestricted"):
            raise ValueError(f"Unknown quadratic mode: {self.quadratic_mode}")
        if self.quadratic_sharing not in ("global", "per_head"):
            raise ValueError(f"Unknown quadratic sharing: {self.quadratic_sharing}")
        if self.quadratic_mode != "fixed" and self.attention_degree != 2:
            raise ValueError("Learned quadratic modes require model.attention_degree=2")
        if self.attention_degree is not None and (
            not isinstance(self.attention_degree, int)
            or isinstance(self.attention_degree, bool)
            or self.attention_degree < 0
        ):
            raise ValueError("model.attention_degree must be null or a nonnegative integer")
        if self.attention_coefficients not in ("fixed", "learned"):
            raise ValueError(f"Unknown attention coefficients: {self.attention_coefficients}")
        if self.attention_coefficient_sharing not in ("global", "per_head"):
            raise ValueError(f"Unknown attention coefficient sharing: {self.attention_coefficient_sharing}")
        if self.attention_coefficient_conditioning not in ("static", "query", "query_key"):
            raise ValueError(f"Unknown attention coefficient conditioning: {self.attention_coefficient_conditioning}")
        activation = self.attention_coefficient_activation
        if activation not in ("identity", *AFFINE_ACTIVATIONS, *PURE_ACTIVATIONS):
            raise ValueError(f"Unknown attention coefficient activation: {activation}")
        if self.attention_coefficient_input_normalization not in ("none", "l2"):
            raise ValueError("Unknown attention coefficient input normalization: "
                             f"{self.attention_coefficient_input_normalization}")
        if self.attention_coefficient_input_source not in ("post_rmsnorm", "pre_rmsnorm"):
            raise ValueError("Unknown attention coefficient input source: "
                             f"{self.attention_coefficient_input_source}")
        if (self.attention_coefficient_input_source == "pre_rmsnorm"
                and self.attention_coefficient_input_normalization != "l2"):
            raise ValueError("pre_rmsnorm coefficient input source requires L2 coefficient input normalization")
        if self.attention_coefficient_input_normalization == "l2" and (
                self.attention_mechanism != "dot_product" or self.attention_coefficients != "learned"
                or self.attention_coefficient_sharing != "per_head"
                or self.attention_coefficient_conditioning != "query" or activation != "tanh"
                or self.attention_coefficient_parameterization not in ("direct", "sum_of_squares")
                or self.attention_coefficient_readout != "diagonal"
                or self.attention_coefficient_orders is not None or self.attention_coefficient_tanh_bias
                or self.attention_coefficient_gate != "none" or self.attention_forget_gate
                or self.attention_kernel_basis != "polynomial" or self.attention_score_bias != "none"
                or self.attention_score_transform != "identity" or self.attention_normalization
                or self.positional_encoding not in ("rope", "none")
                or self.attention_degree is None or self.attention_degree < 1 or self.quadratic_mode != "fixed"):
            raise ValueError("L2 coefficient input normalization requires unnormalized learned per-head "
                             "diagonal tanh query polynomial coefficients, direct or sum-of-squares, "
                             "with RoPE or no positional encoding, without selected orders, inner biases, gates, "
                             "score biases or score transforms")
        if self.attention_coefficient_readout not in ("diagonal", "dense"):
            raise ValueError(f"Unknown attention coefficient readout: {self.attention_coefficient_readout}")
        if self.attention_coefficient_readout == "dense" and (
                self.attention_mechanism != "dot_product" or self.attention_coefficients != "learned"
                or self.attention_coefficient_conditioning != "query"
                or activation not in (*AFFINE_ACTIVATIONS, *PURE_ACTIVATIONS)
                or self.attention_coefficient_parameterization != "direct"):
            raise ValueError("Dense coefficient readout requires learned direct nonlinear query-only coefficients")
        readout_scale = self.attention_coefficient_readout_scale
        if (not isinstance(readout_scale, (int, float)) or isinstance(readout_scale, bool)
                or not math.isfinite(readout_scale) or readout_scale <= 0):
            raise ValueError("model.attention_coefficient_readout_scale must be finite and positive")
        if readout_scale != 1.0 and self.attention_coefficient_readout != "dense":
            raise ValueError("A non-unit readout scale requires dense coefficient readout")
        features = self.attention_coefficient_feature_count
        if features is not None:
            if isinstance(features, bool) or not isinstance(features, int) or features < 1:
                raise ValueError("model.attention_coefficient_feature_count must be a positive integer or null")
            if self.attention_coefficient_readout != "dense":
                raise ValueError("model.attention_coefficient_feature_count requires dense coefficient readout")
            if features != self.attention_degree + 1 and self.attention_coefficient_amplitude_initial != 0:
                raise ValueError("A feature count other than degree+1 has no diagonal: start V at zero "
                                 "(attention_coefficient_amplitude_initial: 0)")
        if self.attention_coefficient_activation != "identity" and self.attention_coefficient_conditioning == "static":
            raise ValueError("Nonlinear coefficient activation requires query-conditioned coefficients")
        if self.attention_coefficient_conditioning == "query_key" and self.attention_coefficient_activation != "tanh":
            raise ValueError("Query-key coefficients require the tanh activation")
        if self.attention_coefficient_conditioning in ("query", "query_key"):
            square_factors = self.attention_coefficient_parameterization == "sum_of_squares"
            hermite_start = self.attention_coefficient_initialization == "hermite"
            if (self.attention_mechanism != "dot_product" or self.attention_coefficients != "learned"
                    or self.attention_coefficient_parameterization not in ("direct", "sum_of_squares")
                    or self.attention_coefficient_sharing != "per_head"
                    or self.attention_degree is None or self.attention_degree < 1 or self.attention_normalization
                    or self.attention_coefficient_initialization not in ("taylor", "hermite")):
                raise ValueError("Query-conditioned coefficients require unnormalized learned per_head polynomial "
                                 "attention, direct or sum-of-squares, with Taylor or Hermite initialization")
            if (square_factors or hermite_start) and (
                    self.attention_coefficient_conditioning != "query" or self.attention_coefficient_activation != "tanh"
                    or self.attention_coefficient_readout != "diagonal" or self.attention_coefficient_orders is not None
                    or self.attention_coefficient_gate != "none" or self.attention_coefficient_tanh_bias):
                raise ValueError("Sum-of-squares factors and the Hermite start apply to diagonal tanh query coefficients "
                                 "without selected orders, a coefficient gate, or an inner tanh bias")
            if square_factors and (hermite_start or self.attention_degree % 2 or self.attention_coefficient_scale != "none"
                                   or self.attention_coefficient_amplitude_scale != "none"):
                raise ValueError("Sum-of-squares query coefficients require an even degree, a Taylor start, and no "
                                 "factorial scale (b and a are relative to the Taylor factors)")
            if self.attention_coefficient_activation == "identity" and self.attention_degree != 1:
                raise ValueError("Query-conditioned identity coefficients require degree-one attention")
        amplitude, spread = self.attention_coefficient_amplitude_initial, self.attention_coefficient_projection_initial_std
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    for value in (amplitude, spread)) or spread < 0):
            raise ValueError("Tanh coefficient initialization needs a finite amplitude and a nonnegative spread")
        if activation in PURE_ACTIVATIONS:
            if amplitude != 1.0:
                raise ValueError(f"Pure {activation} coefficients have no amplitude; dense V starts at the identity")
            if activation == "pure_tanh" and spread == 0:
                raise ValueError("Pure tanh coefficients need a random W start: tanh(0) = 0 zeroes every coefficient")
        elif (amplitude, spread) != (1.0, 0.0):
            if activation not in AFFINE_ACTIVATIONS:
                raise ValueError("Coefficient amplitude and projection initialization apply to tanh coefficients only")
            if amplitude == 0 and spread == 0:
                raise ValueError("Tanh coefficients with a = 0 and W = 0 have zero gradients and never learn")
        if not isinstance(self.attention_coefficient_tanh_bias, bool):
            raise ValueError("model.attention_coefficient_tanh_bias must be a boolean")
        if self.attention_coefficient_tanh_bias and activation not in AFFINE_ACTIVATIONS:
            raise ValueError("A bias inside tanh applies to affine tanh or sigmoid coefficients only")
        if not isinstance(self.attention_coefficient_gate, str) or self.attention_coefficient_gate not in (
                "none", "sigmoid", "gdn"):
            raise ValueError(f"Unknown attention coefficient gate: {self.attention_coefficient_gate!r}")
        if self.attention_coefficient_gate != "none" and (
                self.attention_coefficient_conditioning != "query" or activation != "tanh"
                or self.attention_coefficient_readout != "diagonal"
                or self.attention_coefficient_orders is not None or self.attention_coefficient_tanh_bias
                or self.attention_forget_gate is not False or self.positional_encoding != "rope"):
            raise ValueError("Coefficient gates are qualified only for diagonal tanh query coefficients with RoPE, "
                             "without selected orders, an inner tanh bias, or a forget gate")
        if self.attention_coefficient_amplitude_scale not in ("none", "factorial"):
            raise ValueError(f"Unknown attention coefficient amplitude scale: {self.attention_coefficient_amplitude_scale}")
        if (self.attention_coefficient_amplitude_scale != "none"
                and activation not in (*AFFINE_ACTIVATIONS, *PURE_ACTIVATIONS)):
            raise ValueError("An amplitude scale applies to tanh and sigmoid coefficients only")
        if not isinstance(self.attention_forget_gate, bool):
            raise ValueError("model.attention_forget_gate must be a boolean")
        if self.attention_forget_gate_mode not in ("cumulative", "query_distance"):
            raise ValueError(f"Unknown attention forget gate mode: {self.attention_forget_gate_mode}")
        if self.attention_forget_gate_mode != "cumulative" and not self.attention_forget_gate:
            raise ValueError("A query-distance decay mode requires model.attention_forget_gate")
        if self.attention_forget_gate_form not in ("input", "constant", "sigmoid"):
            raise ValueError(f"Unknown forget gate form: {self.attention_forget_gate_form}")
        if self.attention_forget_gate_form != "input" and not self.attention_forget_gate:
            raise ValueError("A forget gate form requires the forget gate")
        # Softmax and the unnormalized exponential add L_t - L_i to every score (the Forgetting
        # Transformer; with PaTH, PaTH-FoX); nonlinear query-coefficient polynomials multiply their
        # weights by exp(L_t - L_i). Learned static coefficients take only the input-free
        # query-distance gate (conditioning none), on the query-coefficient decay kernels.
        exponential = (self.attention_degree is None and self.attention_coefficients == "fixed"
                       and self.quadratic_mode == "fixed" and self.attention_coefficient_conditioning == "static")
        polynomial = (not self.attention_normalization and self.attention_coefficient_conditioning == "query"
                      and activation != "identity")
        input_free = self.attention_forget_gate_conditioning == "none"
        static_polynomial = (not self.attention_normalization and self.attention_coefficients == "learned"
                             and self.attention_coefficient_conditioning == "static"
                             and self.attention_coefficient_parameterization == "direct"
                             and self.attention_coefficient_sharing == "per_head"
                             and self.attention_coefficient_initialization == "taylor"
                             and self.attention_coefficient_orders is None
                             and getattr(self, "attention_kernel_basis", "polynomial") == "polynomial")
        # Learned static coefficients also take the per-order query-key gate, whose rates alone read the tokens.
        query_key_static = (static_polynomial and self.attention_forget_gate_conditioning == "query_key"
                            and self.attention_forget_gate_per_order is True)
        gated_polynomial = static_polynomial if input_free else polynomial or query_key_static
        if not isinstance(self.attention_forget_gate_per_order, bool):
            raise ValueError("model.attention_forget_gate_per_order must be a boolean")
        if self.attention_forget_gate_per_order and (
                not self.attention_forget_gate or not gated_polynomial
                or self.attention_forget_gate_form != "input"
                or self.attention_forget_gate_rate_activation != "softplus"
                or self.positional_encoding not in ("rope", "none")
                or self.attention_backend not in ("blocked", "mosaic")):
            raise ValueError("Per-order gates require input-form softplus polynomial attention with RoPE/NoPE "
                             "on the blocked or mosaic backend: the query-distance gate, or the cumulative "
                             "query-conditioned gate")
        if self.attention_forget_gate_per_order and self.attention_forget_gate_mode == "cumulative" and (
                self.attention_forget_gate_conditioning != "query"):
            # Each order's own cumulative sum L_tr: the blocked backend (scan or strips), or H100's Mosaic
            # per-order cumulative kernels (mosaic_decay_supported).
            raise ValueError("Per-order cumulative gates take query conditioning, on the blocked backend (scan or "
                             "strips) or mosaic")
        if self.attention_forget_gate_form == "sigmoid" and (
                self.attention_forget_gate_mode != "cumulative" or self.positional_encoding not in ("rope", "none")):
            raise ValueError("The sigmoid forget gate form is qualified as cumulative decay with RoPE or no "
                             "positional encoding on the blocked backend")
        if self.attention_forget_gate_form == "constant" and (
                self.attention_forget_gate_mode != "cumulative" or self.positional_encoding not in ("rope", "none")
                or not (polynomial or exponential)):
            raise ValueError("The constant forget gate form is qualified only as cumulative decay of softmax, the "
                             "unnormalized exponential, and nonlinear query-coefficient polynomials with RoPE or no "
                             "positional encoding on the blocked backend")
        if self.attention_forget_gate_constant_rate not in ("learned", "fixed"):
            raise ValueError(f"Unknown constant forget gate rate: {self.attention_forget_gate_constant_rate}")
        if self.attention_forget_gate_constant_rate != "learned" and (
                not self.attention_forget_gate or self.attention_forget_gate_form != "constant"):
            raise ValueError("A fixed forget gate rate requires the constant forget gate form")
        if self.attention_forget_gate_conditioning not in ("query", "query_key", "none"):
            raise ValueError("Unknown forget gate conditioning")
        if self.attention_forget_gate_rate_activation not in ("softplus", "sigmoid"):
            raise ValueError("Unknown forget gate rate activation")
        if input_free and not (
                self.attention_forget_gate and static_polynomial and self.attention_forget_gate_form == "input"
                and self.attention_forget_gate_mode == "query_distance"
                and self.attention_forget_gate_rate_activation == "softplus"
                and self.positional_encoding in ("rope", "none")
                and self.attention_backend in ("blocked", "mosaic")):
            raise ValueError("The input-free gate (attention_forget_gate_conditioning: none) is qualified as the "
                             "softplus input form's query-distance decay of learned direct per_head static "
                             "coefficients (Taylor start, every order), with RoPE or no positional encoding on "
                             "the blocked or mosaic backend")
        if (self.attention_forget_gate_conditioning != "query" or self.attention_forget_gate_rate_activation != "softplus") and (
                not self.attention_forget_gate or self.attention_forget_gate_form != "input"
                or self.attention_forget_gate_mode != "query_distance"
                or self.attention_backend not in ("blocked", "mosaic")):
            raise ValueError("Endpoint gate variants require input-form query-distance decay on the blocked or "
                             "mosaic backend")
        if self.attention_forget_gate_conditioning == "query_key" and self.attention_forget_gate_rate_activation != "softplus":
            raise ValueError("Query-key decay is qualified only with softplus rates")
        forget_backend_supported = (
            self.positional_encoding == "path" and self.attention_backend in ("auto", "blocked", "fused")
            or self.positional_encoding in ("rope", "none") and self.attention_backend == "blocked"
            # Score-bias recipes reach _validate_score_bias, which names the reason a Mosaic shape cannot run.
            or self.attention_backend == "mosaic" and (self.attention_score_bias != "none"
                                                       or self.mosaic_decay_supported()))
        if self.attention_forget_gate and (
                self.attention_mechanism != "dot_product" or not (exponential or gated_polynomial)
                or not forget_backend_supported):
            raise ValueError("The forget gate applies to softmax, unnormalized exponential, and unnormalized "
                             "nonlinear query-conditioned polynomial attention (learned static coefficients take "
                             "the input-free gate): RoPE/none on the blocked backend (query-distance and per-order "
                             "cumulative decay of polynomials also on mosaic), or PaTH on blocked/fused/auto")
        if self.attention_forget_gate_mode == "query_distance" and self.positional_encoding == "path":
            # The PaTH reference and fused kernels apply only the cumulative decay exp(L_t - L_i).
            raise ValueError("The query-distance decay mode requires RoPE or no positional encoding on the "
                             "blocked backend; PaTH supports only cumulative FoX decay")
        if self.attention_forget_gate_mode == "query_distance" and not (gated_polynomial or exponential):
            raise ValueError("The query-distance decay mode applies to nonlinear query-conditioned polynomial "
                             "attention, softmax, and the unnormalized exponential")
        if self.attention_forget_gate_mode == "query_distance" and exponential and (
                self.attention_forget_gate_rate_activation != "softplus"):
            # Softmax and the exponential add (t - i) log alpha_t to every score: ALiBi with a learned,
            # input-dependent slope per query token; query_key conditioning gives each pair its own rate,
            # (t - i) log alpha_ti. The constant and sigmoid forms are rejected above.
            raise ValueError("Query-distance decay of softmax and the unnormalized exponential takes the "
                             "input-form softplus gate with query conditioning, or its query-key endpoint "
                             "form, on the blocked backend")
        if not isinstance(self.attention_coefficient_kernel_weight_decay, bool):
            raise ValueError("model.attention_coefficient_kernel_weight_decay must be a boolean")
        if (self.attention_coefficient_kernel_weight_decay
                and self.attention_coefficient_conditioning not in ("query", "query_key")):
            raise ValueError("Coefficient kernel weight decay requires query-conditioned coefficients")
        if self.attention_coefficient_scale not in ("none", "factorial"):
            raise ValueError(f"Unknown attention coefficient scale: {self.attention_coefficient_scale}")
        if self.attention_coefficient_scale != "none" and (
                self.attention_coefficients != "learned" or self.attention_normalization
                or self.attention_coefficient_parameterization != "direct"
                or (self.attention_coefficient_conditioning != "static"
                    and activation not in AFFINE_ACTIVATIONS)):
            raise ValueError("A coefficient scale applies to learned direct unnormalized coefficients, "
                             "static or tanh query-conditioned")
        if self.attention_coefficients == "learned":
            if self.attention_mechanism != "dot_product" or self.quadratic_mode != "fixed":
                raise ValueError("Learned polynomial coefficients use dot-product attention without a quadratic mode")
            if self.attention_degree is None or self.attention_degree < 1:
                raise ValueError("Learned polynomial coefficients require an integer attention_degree >= 1")
        if self.attention_coefficient_orders is not None:
            orders = self.attention_coefficient_orders
            if (self.attention_coefficients != "learned"
                    or self.attention_coefficient_parameterization != "direct"
                    or (self.attention_coefficient_conditioning != "static" and not (
                        self.attention_coefficient_conditioning == "query"
                        and self.attention_coefficient_activation == "tanh"
                        and self.attention_coefficient_readout == "diagonal"))):
                raise ValueError("Selected coefficient orders require learned direct static or diagonal tanh-query coefficients")
            if (not isinstance(orders, (tuple, list)) or not orders
                    or any(type(order) is not int or not 0 <= order <= self.attention_degree for order in orders)
                    or list(orders) != sorted(set(orders)) or orders[-1] != self.attention_degree):
                raise ValueError("attention_coefficient_orders must be strictly increasing integers from 0 to "
                                 "attention_degree, including attention_degree")
            self.attention_coefficient_orders = tuple(orders)
        if self.attention_coefficient_parameterization not in ("direct", "sum_of_squares", "rational"):
            raise ValueError(f"Unknown coefficient parameterization: {self.attention_coefficient_parameterization}")
        if self.attention_coefficient_initialization not in ("taylor", "linear", "hermite"):
            raise ValueError(f"Unknown coefficient initialization: {self.attention_coefficient_initialization}")
        if self.attention_coefficient_initialization != "taylor":
            if self.attention_coefficients != "learned":
                raise ValueError("model.attention_coefficient_initialization applies to learned polynomial coefficients only")
            if self.attention_coefficient_parameterization == "sum_of_squares":
                raise ValueError("A sum-of-squares kernel cannot start at linear attention; use the taylor initialization")
            if self.attention_coefficient_initialization == "hermite" and self.attention_coefficient_conditioning != "query":
                raise ValueError("The Hermite start applies to tanh query-conditioned coefficients")
        if self.attention_coefficient_parameterization == "rational":
            if self.attention_coefficients != "learned":
                raise ValueError("A rational kernel requires learned polynomial coefficients")
            if not self.attention_normalization:
                raise ValueError("A rational kernel's denominator is its row sum; it requires normalized attention")
        if self.attention_denominator_parameterization not in ("direct", "sum_of_squares"):
            raise ValueError(f"Unknown denominator parameterization: {self.attention_denominator_parameterization}")
        if self.attention_denominator_constant not in ("fixed", "learned"):
            raise ValueError(f"Unknown denominator constant: {self.attention_denominator_constant}")
        if ((self.attention_denominator_degree is not None or self.attention_denominator_parameterization != "direct"
             or self.attention_denominator_constant != "fixed")
                and self.attention_coefficient_parameterization != "rational"):
            raise ValueError("model.attention_denominator_degree, attention_denominator_parameterization, and "
                             "attention_denominator_constant apply to rational kernels only")
        if self.attention_denominator_degree is not None and (
                isinstance(self.attention_denominator_degree, bool)
                or not isinstance(self.attention_denominator_degree, int) or self.attention_denominator_degree < 1):
            raise ValueError("model.attention_denominator_degree must be null or an integer >= 1")
        if self.attention_coefficient_parameterization == "rational" and self.attention_denominator_parameterization == "sum_of_squares":
            degree = (self.attention_degree if self.attention_denominator_degree is None
                      else self.attention_denominator_degree)
            if degree % 2:
                raise ValueError("A sum-of-squares denominator requires an even denominator degree")
        if self.attention_fixed_coefficients is not None and self.attention_fixed_square_factors is not None:
            raise ValueError("Specify either model.attention_fixed_coefficients or attention_fixed_square_factors")
        if self.attention_fixed_coefficients is not None:
            if (self.attention_mechanism != "dot_product" or self.quadratic_mode != "fixed"
                    or self.attention_coefficients != "fixed"
                    or self.attention_coefficient_parameterization != "direct"):
                raise ValueError("Fixed polynomial coefficients require fixed/direct dot-product attention without a quadratic mode")
            if self.attention_degree is None or self.attention_degree < 1:
                raise ValueError("Fixed polynomial coefficients require an integer attention_degree >= 1")
            values = self.attention_fixed_coefficients
            if not isinstance(values, (tuple, list)) or len(values) != self.attention_degree + 1:
                raise ValueError("model.attention_fixed_coefficients must contain attention_degree + 1 coefficients")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values):
                raise ValueError("model.attention_fixed_coefficients must contain finite numeric coefficients")
            self.attention_fixed_coefficients = tuple(float(value) for value in values)
        if self.attention_fixed_square_factors is not None:
            if (self.attention_mechanism != "dot_product" or self.quadratic_mode != "fixed"
                    or self.attention_coefficients != "fixed"
                    or self.attention_coefficient_parameterization != "sum_of_squares"
                    or not self.attention_normalization):
                raise ValueError("Fixed square factors require normalized fixed/sum_of_squares dot-product attention without a quadratic mode")
            if self.attention_degree is None or self.attention_degree < 2 or self.attention_degree % 2:
                raise ValueError("Fixed square factors require an even attention_degree >= 2")
            values = self.attention_fixed_square_factors
            if not isinstance(values, (tuple, list)) or len(values) != self.attention_degree + 1:
                raise ValueError("model.attention_fixed_square_factors must contain attention_degree + 1 factors")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values):
                raise ValueError("model.attention_fixed_square_factors must contain finite numeric factors")
            self.attention_fixed_square_factors = tuple(float(value) for value in values)
        if self.attention_coefficient_parameterization == "sum_of_squares":
            if self.attention_coefficients != "learned" and self.attention_fixed_square_factors is None:
                raise ValueError("A sum-of-squares kernel requires learned polynomial coefficients or fixed square factors")
            if self.attention_degree is None or self.attention_degree < 2 or self.attention_degree % 2:
                raise ValueError("A sum-of-squares kernel requires an even attention_degree >= 2")
            if not self.attention_normalization and self.attention_coefficient_conditioning != "query":
                raise ValueError("A sum-of-squares kernel is implemented for row-normalized attention only")
        _validate_score_bias(self)
        _validate_score_transform(self)
        if self.qk_norm is None:
            self.qk_norm = self.attention_mechanism == "dot_product" and self.attention_degree != 0
        elif self.attention_degree == 0:
            self.qk_norm = False  # Degree 0 ignores Q and K, so QK-norm cannot apply.
        if not isinstance(self.qk_norm, bool):
            raise ValueError("model.qk_norm must be a boolean")
        if not isinstance(self.qk_norm_head_scale, bool):
            raise ValueError("model.qk_norm_head_scale must be a boolean")
        if self.qk_norm_head_scale and (
                not self.qk_norm or self.qk_norm_type != "rms_norm"
                or self.attention_mechanism != "dot_product"):
            raise ValueError("QK head scaling requires enabled dot-product QK RMSNorm")
        if self.qk_norm_token_scale not in ("none", "cosine"):
            raise ValueError(f"Unknown QK token scale: {self.qk_norm_token_scale}")
        if self.qk_norm_token_scale != "none" and not self.qk_norm_head_scale:
            raise ValueError("The per-token query scale requires QK RMS head scaling")
        if not isinstance(self.qk_norm_token_scale_kernel_weight_decay, bool):
            raise ValueError("model.qk_norm_token_scale_kernel_weight_decay must be a boolean")
        if self.qk_norm_token_scale_kernel_weight_decay and self.qk_norm_token_scale == "none":
            raise ValueError("Token-scale projection weight decay requires an enabled per-token query scale")
        if self.qk_norm_type not in ("cosine_log_scale", "rms_norm"):
            raise ValueError(f"Unknown QK-norm type: {self.qk_norm_type}")
        allowed_sharing = ("per_layer",) if self.qk_norm_type == "rms_norm" else ("global", "per_head")
        if self.qk_norm_sharing not in allowed_sharing:
            raise ValueError(f"Unknown QK-norm sharing: {self.qk_norm_sharing}")
        if self.qk_norm_type == "rms_norm" and self.qk_norm_initial_scale is not None:
            raise ValueError("QK RMSNorm initializes channel gains at one; qk_norm_initial_scale must be null")
        for name in ("qk_norm_initial_scale", "qk_norm_eps"):
            value = getattr(self, name)
            if name == "qk_norm_initial_scale" and value is None:
                continue
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"model.{name} must be finite and positive")
        if self.qk_norm and self.attention_mechanism != "dot_product":
            raise ValueError("QK-norm is a dot-product attention option; GLA does not support it")
        from scaffolds.transformer.models.initialization import INITIALIZATIONS, MECHANISMS
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f"Unknown initialization: {self.initialization}")
        if MECHANISMS.get(self.initialization, self.attention_mechanism) != self.attention_mechanism:
            raise ValueError(f"model.initialization {self.initialization!r} belongs to "
                             f"{MECHANISMS[self.initialization]} attention, not {self.attention_mechanism}")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.dropout != 0 and self.attention_backend != "manual":
            raise ValueError("Attention backends other than manual require dropout=0")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive")


def _validate_kernel_basis(model: ModelConfig) -> None:
    """The score-function basis: the historical monomials, or a fixed B-spline basis."""
    basis = model.attention_kernel_basis
    if basis not in ("polynomial", "bspline"):
        raise ValueError(f"Unknown attention kernel basis: {basis}")
    if basis == "polynomial":
        if model.attention_spline_knots is not None:
            raise ValueError("model.attention_spline_knots applies to the bspline kernel basis only")
        return
    from scaffolds.transformer.attention.spline import validate_knots

    degree = model.attention_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("The bspline kernel basis takes its degree from model.attention_degree, an integer >= 1")
    if model.attention_spline_knots is None:
        raise ValueError("The bspline kernel basis requires model.attention_spline_knots")
    # Distinct, strictly increasing knots; the endpoints are clamped (multiplicity degree + 1).
    model.attention_spline_knots = validate_knots(model.attention_spline_knots, degree)
    if model.attention_mechanism != "dot_product" or model.positional_encoding not in ("rope", "none"):
        raise ValueError("The bspline kernel basis applies to dot-product attention with RoPE or no positional encoding")
    if model.attention_backend != "blocked":
        raise ValueError("The bspline kernel basis runs on model.attention_backend blocked (scan or strips); "
                         "no fused, Mosaic, or cuDNN kernel implements it")
    if model.attention_normalization:
        raise ValueError("The bspline kernel basis is implemented for unnormalized attention only")
    if (model.quadratic_mode != "fixed" or model.attention_coefficients != "learned"
            or model.attention_coefficient_parameterization != "direct"
            or model.attention_coefficient_sharing != "per_head"
            or model.attention_fixed_coefficients is not None or model.attention_fixed_square_factors is not None):
        raise ValueError("The bspline kernel basis requires learned direct per_head coefficients")
    if model.attention_coefficient_initialization != "taylor":
        raise ValueError("The bspline kernel basis starts at the exact B-spline coefficients of the Taylor "
                         "polynomial: attention_coefficient_initialization must be taylor")
    if (model.attention_coefficient_scale != "none" or model.attention_coefficient_amplitude_scale != "none"
            or model.attention_coefficient_orders is not None):
        raise ValueError("B-spline coefficients are not monomial orders: no factorial coefficient or amplitude "
                         "scale and no attention_coefficient_orders")
    if model.attention_forget_gate:
        raise ValueError("The bspline kernel basis is qualified without a forget gate")
    conditioning = model.attention_coefficient_conditioning
    if conditioning == "query":
        if (model.attention_coefficient_activation != "tanh" or model.attention_coefficient_readout != "diagonal"
                or model.attention_coefficient_tanh_bias or model.attention_coefficient_feature_count is not None
                or model.attention_coefficient_readout_scale != 1.0):
            raise ValueError("Query-conditioned B-spline coefficients use the diagonal tanh generator "
                             "theta_j = b_j + a_j tanh(w_j^T x_t), without an inner tanh bias")
    elif conditioning != "static":
        raise ValueError("B-spline coefficients are static or query-conditioned (no key factors)")


def _score_bias_problem(model) -> str | None:
    """Why per-degree score biases do not apply to this recipe's attention, or None."""
    from scaffolds.transformer.attention.backends.mosaic.api import MAX_QUERY_DEGREE

    if (model.attention_mechanism != "dot_product" or model.attention_normalization
            or model.quadratic_mode != "fixed" or model.attention_coefficients != "learned"
            or model.attention_coefficient_parameterization != "direct"
            or model.attention_coefficient_sharing != "per_head"
            or model.attention_coefficient_initialization != "taylor"
            or model.attention_coefficient_conditioning != "query"
            or model.attention_coefficient_activation != "tanh"
            or model.attention_coefficient_readout != "diagonal"
            or model.attention_coefficient_orders is not None
            or model.attention_coefficient_gate != "none"
            or getattr(model, "attention_kernel_basis", "polynomial") != "polynomial"):
        return ("they apply to unnormalized dot-product attention with learned direct per_head Taylor-initialized "
                "diagonal tanh query coefficients, without selected orders, a coefficient gate, or a B-spline basis")
    degree = model.attention_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= MAX_QUERY_DEGREE:
        return f"they need an integer attention_degree from 1 to {MAX_QUERY_DEGREE}"
    if model.positional_encoding not in ("rope", "none"):
        return "they apply with RoPE or no positional encoding, not PaTH"
    if model.attention_forget_gate and not (
            model.attention_forget_gate_form == "input" and model.attention_forget_gate_mode == "query_distance"
            and model.attention_forget_gate_conditioning == "query_key"
            and model.attention_forget_gate_rate_activation == "softplus"
            and model.attention_forget_gate_per_order is True):
        return ("they take no forget gate or the per-order query-key endpoint gate (input form, query_distance "
                "mode, softplus rates, attention_forget_gate_per_order)")
    return None


def _validate_score_bias(model: ModelConfig) -> None:
    """Per-degree score biases (attention/score_bias.py) train on H100's Mosaic score-bias kernels or on the
    blocked backend's causal strips (allocation gives fresh blocked recipes the strips,
    fresh_recipe_uses_blocked_strips). The blocked scan stays valid here as their reference for tests and
    diagnostics; allocation and acceptance refuse it for training, since only the strips have a GPU
    qualification. No other backend evaluates the shifted powers, so every other backend is rejected."""
    from scaffolds.transformer.attention.score_bias import MODES

    mode = model.attention_score_bias
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"Unknown attention score bias: {mode!r}")
    if mode == "none":
        return
    problem = _score_bias_problem(model)
    if problem is not None:
        raise ValueError(f"Per-degree score biases ({mode}) do not fit this recipe: {problem}")
    if model.attention_backend not in ("mosaic", "blocked"):
        raise ValueError(f"Per-degree score biases train on attention_backend mosaic (H100) or blocked; "
                         f"{model.attention_backend!r} never runs them")
    if model.attention_backend == "mosaic":
        problem = model.mosaic_score_bias_problem()
        if problem is not None:
            raise ValueError(f"Backend 'mosaic' cannot run these score biases: {problem}")


def _query_coefficient_problem(model) -> str | None:
    """Why this recipe's attention is not the diagonal tanh query-coefficient polynomial, or None."""
    if (model.attention_mechanism != "dot_product" or model.attention_normalization
            or model.quadratic_mode != "fixed" or model.attention_coefficients != "learned"
            or model.attention_coefficient_parameterization != "direct"
            or model.attention_coefficient_sharing != "per_head"
            or model.attention_coefficient_initialization != "taylor"
            or model.attention_coefficient_conditioning != "query"
            or model.attention_coefficient_activation != "tanh"
            or model.attention_coefficient_readout != "diagonal"
            or model.attention_coefficient_orders is not None
            or model.attention_coefficient_gate != "none"
            or getattr(model, "attention_kernel_basis", "polynomial") != "polynomial"):
        return ("it applies to unnormalized dot-product attention with learned direct per_head Taylor-initialized "
                "diagonal tanh query coefficients, without selected orders, a coefficient gate, or a B-spline basis")
    degree = model.attention_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        return "it needs an integer attention_degree >= 1"
    return None


def _validate_score_transform(model: ModelConfig) -> None:
    """A score transform (attention/score_transform.py) runs on the blocked backend, for diagonal tanh query
    coefficients with RoPE or no positional encoding, without a forget gate or score biases (allocation and
    acceptance train it on the causal strips there), or on the Mosaic transformed query-polynomial kernels for
    the recipes they take (``mosaic_score_transform_problem``). A constant-coefficient start sets b_0's
    initial value of the same coefficients on any backend. The identity transform and a null start change
    nothing."""
    from scaffolds.transformer.attention.score_transform import TRANSFORMS

    transform, start = model.attention_score_transform, model.attention_coefficient_constant_initial
    if not isinstance(transform, str) or transform not in TRANSFORMS:
        raise ValueError(f"Unknown attention score transform: {transform!r}")
    if start is not None:
        if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start):
            raise ValueError("model.attention_coefficient_constant_initial must be null or a finite number")
        problem = _query_coefficient_problem(model)
        if problem is not None:
            raise ValueError(f"A constant-coefficient start (attention_coefficient_constant_initial) does not fit "
                             f"this recipe: {problem}")
        model.attention_coefficient_constant_initial = float(start)
    if transform == "identity":
        return
    problem = _query_coefficient_problem(model)
    if problem is None and model.positional_encoding not in ("rope", "none"):
        problem = "it applies with RoPE or no positional encoding, not PaTH"
    if problem is None and model.attention_forget_gate:
        problem = "it is qualified without a forget gate"
    if problem is None and model.attention_score_bias != "none":
        problem = "it does not combine with per-degree score biases"
    if problem is not None:
        raise ValueError(f"The {transform} score transform does not fit this recipe: {problem}")
    if model.attention_backend not in ("blocked", "mosaic"):
        raise ValueError(f"A score transform runs on attention_backend blocked (its scan and causal strips) or mosaic "
                         f"(the transformed query-polynomial kernels); {model.attention_backend!r} never evaluates it")
    if model.attention_backend == "mosaic":
        problem = model.mosaic_score_transform_problem()
        if problem is not None:
            raise ValueError(f"Backend 'mosaic' cannot run the {transform} score transform: {problem}")


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    # Multiply only the diagonal tanh query generator's stored a and b updates.
    # The projection W and all other parameters retain the base LR schedule.
    query_coefficient_learning_rate_scale: float = 1.0
    # Coefficient release (training/coefficient_release.py), for learned direct static or diagonal
    # tanh query coefficients storing every order 0..m: each stored entry of order r (static c_r;
    # query b_r, a_r, and column r of W) stays at its initial value for optimizer updates with
    # count k < s_r and trains for k >= s_r, on the LR schedule's clock (the update logged as
    # step k + 1). "all": s_r = coefficient_release_step for every order. "by_order": s_r =
    # coefficient_release_step + floor(r (coefficient_release_end_step - coefficient_release_step) / m),
    # lowest order first; requires 0 < release step < end step < total_steps. A frozen entry's
    # gradient is zeroed before clipping and AdamW, and its update after AdamW, so weight decay
    # cannot move it; released entries start with zero Adam moments under the global bias
    # correction and the shared LR schedule. Null (the default) trains them from the first update.
    coefficient_release_step: int | None = None
    coefficient_release_schedule: Literal["all", "by_order"] | None = None
    coefficient_release_end_step: int | None = None
    weight_decay: float = 0.1
    weight_decay_policy: Literal["exclude_embeddings_norms_biases_v1"] = "exclude_embeddings_norms_biases_v1"
    # Penalty weight lambda on learned polynomial coefficients, added to the training
    # loss as lambda * sum over polynomials of a_0^2 + sum_(r>=2) r^2 a_r^2
    # (+ sum_(r>=1) r^2 b_r^2 for a rational denominator); a_1 is unpenalized
    # (training/regularization.py). Validation loss never includes it.
    coefficient_penalty: float = 0.0
    # Whether that penalty includes a rational kernel's direct denominator coefficients
    # b_r; false penalizes the numerator only.
    coefficient_penalty_denominator: bool = True
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1
    total_steps: int = 10


@dataclass
class TrainConfig:
    seed: int = 0
    source_commit: str = ""
    dtype: Literal["float32", "bfloat16"] = "float32"
    workdir: Path = Path(".cache/transformer/smoke")
    checkpoint_dir: Path | None = None
    micro_batch_size: int = 2
    log_every: int = 10
    eval_every: int = 10
    # Optional initial evaluation cadence, fixed in the recipe for pilots/resumptions.
    eval_early_every: int | None = None
    eval_early_until: int = 0
    eval_batches: int = 1
    ckpt_every: int = 10
    # W&B mirrors committed metrics; launchers choose the mode. Name and group
    # default to the run and experiment of experiments/<experiment>/runs/<run>.
    wandb_project: str = "functional-memory"
    wandb_entity: str | None = "rafah-massachusetts-institute-of-technology"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_dir: Path | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    # Event-triggered capture of the exact pre-update state when a post-warmup update's pre-clip
    # gradient norm exceeds max(min_norm, factor x median of the previous `history` norms, once at
    # least `min_history` exist). At most `max_states` states are retained: the first event and the
    # largest later ones. Every trigger is logged (training/gradient_events.py).
    gradient_event_capture: bool = False
    gradient_event_factor: float = 5.0
    gradient_event_min_norm: float = 1.0
    gradient_event_history: int = 100
    gradient_event_min_history: int = 20
    gradient_event_max_states: int = 3


@dataclass
class ContinuationConfig:
    """A new run initialized from a pinned pretrained state, including Adam history."""
    parent_config: str = ""
    parent_config_sha256: str = ""
    parent_checkpoint: str = ""
    parent_checkpoint_sha256: str = ""
    parent_data_identity_sha256: str = ""
    parent_step: int = 0
    # Constant learning rate of the parameters the continuation introduces, which have their
    # own Adam clock; null gives them the backbone's optim.learning_rate.
    new_parameter_learning_rate: float | None = None
    # Continue the identical architecture, retaining every parameter and Adam moment.
    preserve_all_parameters: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)


def fresh_recipe_uses_blocked_strips(model: dict) -> bool:
    """Whether a fresh recipe's ``model`` mapping trains on the blocked backend's causal strips
    unless it says otherwise: dot-product attention with RoPE or no positional encoding whose
    backend is blocked, or auto (which resolves to blocked where no kernel applies). Other
    backends never run the field, so their recipes keep its inert default, scan. Allocation
    (scaffolds.records create, and launchers that allocate copied recipes) applies this rule.
    """
    return (model.get("attention_mechanism", "dot_product") == "dot_product"
            and model.get("positional_encoding", "rope") != "path"
            and model.get("attention_backend", "auto") in ("blocked", "auto"))


def validate_config(config: Config) -> None:
    """Validate loaded/overridden settings before allocating a model."""
    config.model.__post_init__()
    data, optim, train = config.data, config.optim, config.train
    continuation = config.continuation
    if config.model.attention_kernel_basis == "bspline":
        # B-spline coefficients weight basis functions, not monomial orders: the coefficient
        # penalty (a_0^2 + sum r^2 a_r^2), the query readout LR scale (which acts on the monomial
        # generator's arrays), and any coefficient release schedule (optim.coefficient_release_*,
        # which indexes orders 0..m) do not apply. Continuations introduce monomial coefficients
        # only; an identical-architecture continuation keeps every array.
        if optim.coefficient_penalty != 0:
            raise ValueError("optim.coefficient_penalty is defined on monomial coefficients, not a B-spline basis")
        if optim.query_coefficient_learning_rate_scale != 1.0:
            raise ValueError("optim.query_coefficient_learning_rate_scale applies to monomial query coefficients, "
                             "not a B-spline basis")
        if getattr(optim, "coefficient_release_schedule", None) is not None:
            raise ValueError("A coefficient release schedule indexes monomial orders 0..m; "
                             "it does not apply to a B-spline basis")
        if continuation.parent_checkpoint and not continuation.preserve_all_parameters:
            raise ValueError("A continuation cannot introduce B-spline coefficients; only an identical-architecture "
                             "continuation (continuation.preserve_all_parameters) supports the bspline basis")
    if config.model.attention_score_bias != "none":
        # Score biases cover orders 1..m beside coefficients indexed 0..m, and no qualified schedule moves
        # them: a coefficient release, a separate query-readout learning rate, and continuations do not apply.
        if getattr(optim, "coefficient_release_schedule", None) is not None:
            raise ValueError("A coefficient release schedule is not qualified with per-degree score biases")
        if optim.query_coefficient_learning_rate_scale != 1.0:
            raise ValueError("optim.query_coefficient_learning_rate_scale is not qualified with per-degree score biases")
        if continuation.parent_checkpoint:
            raise ValueError("Checkpoint continuations are not qualified with per-degree score biases")
    if (config.model.attention_score_transform != "identity"
            or config.model.attention_coefficient_constant_initial is not None):
        # Both change the function or start of the coefficients that these schedules and continuations assume
        # start at Taylor and read s^r: none is qualified with them.
        if getattr(optim, "coefficient_release_schedule", None) is not None:
            raise ValueError("A coefficient release schedule is not qualified with a score transform or a "
                             "constant-coefficient start")
        if optim.query_coefficient_learning_rate_scale != 1.0:
            raise ValueError("optim.query_coefficient_learning_rate_scale is not qualified with a score transform "
                             "or a constant-coefficient start")
        if continuation.parent_checkpoint:
            raise ValueError("Checkpoint continuations are not qualified with a score transform or a "
                             "constant-coefficient start")
    if not isinstance(train.gradient_event_capture, bool):
        raise ValueError("train.gradient_event_capture must be a boolean")
    if train.gradient_event_capture:
        if not (math.isfinite(train.gradient_event_factor) and train.gradient_event_factor > 1.0):
            raise ValueError("train.gradient_event_factor must be finite and greater than 1")
        if not (math.isfinite(train.gradient_event_min_norm) and train.gradient_event_min_norm >= 0.0):
            raise ValueError("train.gradient_event_min_norm must be finite and non-negative")
        for name in ("gradient_event_history", "gradient_event_min_history", "gradient_event_max_states"):
            value = getattr(train, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"train.{name} must be a positive integer")
        if train.gradient_event_min_history > train.gradient_event_history:
            raise ValueError("train.gradient_event_min_history cannot exceed train.gradient_event_history")
    if not isinstance(continuation.preserve_all_parameters, bool):
        raise ValueError("continuation.preserve_all_parameters must be a boolean")
    if continuation.parent_checkpoint:
        if not continuation.parent_config:
            raise ValueError("Continuation requires its parent_config")
        for name in ("parent_config_sha256", "parent_checkpoint_sha256"):
            digest = getattr(continuation, name)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"continuation.{name} must be a SHA256 digest")
        if data.source == "packed":
            digest = continuation.parent_data_identity_sha256
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("Packed continuation requires parent_data_identity_sha256")
        if type(continuation.parent_step) is not int or not 0 < continuation.parent_step < optim.total_steps:
            raise ValueError("Continuation requires 0 < parent_step < optim.total_steps")
        if optim.warmup_steps != 0 or optim.min_learning_rate != optim.learning_rate:
            raise ValueError("Continuation currently requires a constant learning rate with zero warmup")
        rate = continuation.new_parameter_learning_rate
        if continuation.preserve_all_parameters and rate is not None:
            raise ValueError("Preserving all parameters introduces no new parameter learning rate")
        if rate is not None and (not isinstance(rate, (int, float)) or isinstance(rate, bool)
                                 or not math.isfinite(rate) or rate <= 0):
            raise ValueError("continuation.new_parameter_learning_rate must be null or finite and positive")
        if rate is not None and config.model.attention_coefficients != "learned" and (
                config.model.qk_norm_token_scale == "none"):
            raise ValueError("continuation.new_parameter_learning_rate applies to introduced parameters only")
        if config.model.attention_coefficients == "learned" and not continuation.preserve_all_parameters:
            model = config.model
            if model.attention_coefficient_orders is not None:
                raise ValueError("Continuation with selected coefficient orders is not supported")
            if (model.attention_coefficient_conditioning not in ("static", "query")
                    or model.attention_coefficient_parameterization != "direct"
                    or model.attention_coefficient_initialization != "taylor"):
                raise ValueError("Continuation introduces only static or query direct Taylor-initialized coefficients")
            if model.attention_coefficient_conditioning == "query" and (
                    model.attention_coefficient_activation != "tanh"
                    or model.attention_coefficient_scale != "factorial"
                    or model.attention_coefficient_amplitude_scale != "factorial"
                    or model.attention_coefficient_tanh_bias
                    or (model.attention_coefficient_amplitude_initial != 0
                        and model.attention_coefficient_projection_initial_std != 0)):
                raise ValueError("Query continuation requires exact Taylor initialization: factorial-scaled tanh "
                                 "coefficients with no inner bias and either zero amplitude or zero projection")
        if config.model.attention_coefficient_gate != "none":
            # Neither introducing gates nor carrying them through a continuation is qualified.
            raise ValueError("Continuation with coefficient gates is not supported")
    elif any((continuation.parent_config, continuation.parent_config_sha256,
              continuation.parent_checkpoint_sha256, continuation.parent_data_identity_sha256,
              continuation.parent_step, continuation.new_parameter_learning_rate is not None,
              continuation.preserve_all_parameters)):
        raise ValueError("Incomplete continuation specification")
    if optim.weight_decay_policy != "exclude_embeddings_norms_biases_v1":
        raise ValueError("optim.weight_decay_policy must be exclude_embeddings_norms_biases_v1")
    query_scale = optim.query_coefficient_learning_rate_scale
    if (not isinstance(query_scale, (int, float)) or isinstance(query_scale, bool)
            or not math.isfinite(query_scale) or query_scale <= 0):
        raise ValueError("optim.query_coefficient_learning_rate_scale must be finite and positive")
    if query_scale != 1.0:
        model = config.model
        if (model.attention_mechanism != "dot_product" or model.attention_coefficients != "learned"
                or model.attention_coefficient_parameterization != "direct"
                or model.attention_coefficient_conditioning != "query"
                or model.attention_coefficient_activation != "tanh"
                or model.attention_coefficient_readout != "diagonal"):
            raise ValueError("optim.query_coefficient_learning_rate_scale requires learned direct "
                             "query-conditioned tanh coefficients with diagonal readout")
        if model.attention_coefficient_gate != "none":
            raise ValueError("optim.query_coefficient_learning_rate_scale does not cover coefficient gates")
        if continuation.parent_checkpoint:
            raise ValueError("optim.query_coefficient_learning_rate_scale is not supported for "
                             "checkpoint continuations; their introduced parameters use "
                             "continuation.new_parameter_learning_rate")
    if data.source not in ("synthetic", "packed"):
        raise ValueError("data.source must be synthetic or packed")
    for name, value in (
        ("data.batch_size", data.batch_size),
        ("train.micro_batch_size", train.micro_batch_size),
        ("optim.total_steps", optim.total_steps),
        ("train.log_every", train.log_every),
        ("train.eval_every", train.eval_every),
        ("train.eval_batches", train.eval_batches),
        ("train.ckpt_every", train.ckpt_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if data.batch_size % train.micro_batch_size:
        raise ValueError("data.batch_size must be divisible by train.micro_batch_size")
    if (not isinstance(train.eval_early_until, int) or isinstance(train.eval_early_until, bool)
            or train.eval_early_until < 0):
        raise ValueError("train.eval_early_until must be a nonnegative integer")
    if train.eval_early_every is None:
        if train.eval_early_until != 0:
            raise ValueError("train.eval_early_every and train.eval_early_until must be set together")
    elif (not isinstance(train.eval_early_every, int) or isinstance(train.eval_early_every, bool)
          or train.eval_early_every <= 0 or train.eval_early_until <= 0):
        raise ValueError("train.eval_early_every and train.eval_early_until must both be positive integers")
    if not 2 <= data.sequence_length <= config.model.max_seq_len + 1:
        raise ValueError("data.sequence_length must be between 2 and model.max_seq_len + 1")
    if data.source == "synthetic" and not 1 < data.synthetic_vocab_size <= config.model.vocab_size:
        raise ValueError("synthetic_vocab_size must be in [2, model.vocab_size]")
    if data.source == "packed" and not (data.train_manifest and data.eval_manifest):
        raise ValueError("Packed data requires train_manifest and eval_manifest")
    if train.dtype not in ("float32", "bfloat16"):
        raise ValueError("train.dtype must be float32 or bfloat16")
    if (config.model.attention_mechanism == "dot_product" and config.model.positional_encoding == "rope"
            and config.model.attention_backend in ("fused", "cudnn") and config.model.attention_degree is None
            and config.model.attention_normalization and train.dtype != "bfloat16"):
        raise ValueError("Fused exact-softmax attention uses cuDNN and requires train.dtype=bfloat16")
    if (config.model.positional_encoding == "path" and config.model.attention_backend == "fused"
            and train.dtype != "bfloat16"):
        raise ValueError("Fused PaTH attention requires train.dtype=bfloat16")
    if config.model.attention_backend == "mosaic" and train.dtype != "bfloat16":
        raise ValueError("Mosaic attention requires train.dtype=bfloat16")
    if config.model.attention_backend == "fused" and (
            config.model.attention_degree is not None or not config.model.attention_normalization):
        if max(config.model.qk_head_dim, config.model.value_head_dim) > 128:
            raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
    if config.model.attention_mechanism in ("gla", "gated_deltanet") and config.model.attention_backend == "fused":
        if (data.sequence_length - 1) % 64:
            raise ValueError("Fused GLA and Gated DeltaNet require prediction sequence length divisible by 64")
    if train.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError("Invalid train.wandb_mode")
    if not 0 <= optim.warmup_steps <= optim.total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if not 0 <= optim.min_learning_rate <= optim.learning_rate or optim.learning_rate <= 0:
        raise ValueError("Require 0 <= min_learning_rate <= learning_rate and learning_rate > 0")
    if not 0 <= optim.beta1 < 1 or not 0 <= optim.beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if optim.weight_decay < 0 or optim.grad_clip_norm <= 0:
        raise ValueError("Require weight_decay >= 0 and grad_clip_norm > 0")
    penalty = optim.coefficient_penalty
    if (not isinstance(penalty, (int, float)) or isinstance(penalty, bool)
            or not math.isfinite(penalty) or penalty < 0):
        raise ValueError("optim.coefficient_penalty must be a finite number >= 0")
    if penalty > 0 and config.model.attention_coefficient_conditioning != "static":
        raise ValueError("optim.coefficient_penalty requires static polynomial coefficients")
    if penalty > 0 and (config.model.attention_coefficients != "learned"
                        or config.model.attention_coefficient_parameterization not in ("direct", "rational")):
        raise ValueError("optim.coefficient_penalty requires learned direct or rational polynomial coefficients")
    if not isinstance(optim.coefficient_penalty_denominator, bool):
        raise ValueError("optim.coefficient_penalty_denominator must be a boolean")
    rational = (config.model.attention_coefficients == "learned"
                and config.model.attention_coefficient_parameterization == "rational")
    if not optim.coefficient_penalty_denominator and not rational:
        raise ValueError("optim.coefficient_penalty_denominator: false applies to rational kernels only")
    if (penalty > 0 and rational and optim.coefficient_penalty_denominator
            and config.model.attention_denominator_constant != "fixed"):
        raise ValueError("The denominator penalty is defined with its constant fixed at 1; "
                         "set optim.coefficient_penalty_denominator: false for a learned constant")
    _validate_coefficient_release(config)


def _validate_coefficient_release(config: Config) -> None:
    optim, model = config.optim, config.model
    schedule, start, end = (optim.coefficient_release_schedule, optim.coefficient_release_step,
                            optim.coefficient_release_end_step)
    if schedule is None:
        if start is not None or end is not None:
            raise ValueError("optim.coefficient_release_step and coefficient_release_end_step require "
                             "optim.coefficient_release_schedule")
        return
    if schedule not in ("all", "by_order"):
        raise ValueError(f"Unknown optim.coefficient_release_schedule: {schedule!r}")
    if start is None:
        raise ValueError("optim.coefficient_release_schedule requires optim.coefficient_release_step")
    if type(start) is not int or not 0 < start < optim.total_steps:
        raise ValueError("optim.coefficient_release_step must be an integer with "
                         "0 < coefficient_release_step < optim.total_steps")
    if schedule == "by_order":
        if end is None:
            raise ValueError("The by_order coefficient release requires optim.coefficient_release_end_step")
        if type(end) is not int or not start < end < optim.total_steps:
            raise ValueError("optim.coefficient_release_end_step must be an integer with "
                             "coefficient_release_step < coefficient_release_end_step < optim.total_steps")
    elif end is not None:
        raise ValueError("optim.coefficient_release_end_step applies to the by_order schedule only")
    static = (model.attention_coefficient_conditioning == "static"
              and model.attention_coefficient_parameterization == "direct")
    query = (model.attention_coefficient_conditioning == "query"
             and model.attention_coefficient_activation == "tanh"
             and model.attention_coefficient_readout == "diagonal"
             and not model.attention_coefficient_tanh_bias)
    if (model.attention_coefficients != "learned" or not (static or query)
            or model.attention_coefficient_initialization != "taylor"
            or model.attention_coefficient_orders is not None):
        # Every released array must store orders 0..m on its last axis, starting at Taylor.
        raise ValueError("optim.coefficient_release_schedule requires Taylor-initialized learned direct static "
                         "coefficients or diagonal tanh query coefficients without an inner tanh bias, "
                         "storing every order 0..m (no attention_coefficient_orders)")
    if config.continuation.parent_checkpoint:
        raise ValueError("optim.coefficient_release_schedule is not supported for checkpoint continuations")
