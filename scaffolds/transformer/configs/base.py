"""Shared Transformer settings; experiments vary geometry and attention rules."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal


OFM_SOURCE_COMMIT = "8977f529e83efc688daf96cdcfd365ebbce2afa8"
ATTENTION_DEGREES = (0, 2, 4, 8, 16, None)


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
    dropout: float = 0.0
    # Retained for recipe compatibility; this scaffold requires separate weights.
    tie_embeddings: bool = False
    remat: bool = False
    attention_backend: Literal["auto", "blocked", "fused", "manual", "xla", "cudnn"] = "auto"
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
    # Degrees two and three compose polynomial memory with the same original-key-space
    # erase map. Degree one preserves the historical GDN architecture.
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
    # Query coefficients use the incoming attention activation, per layer/head.
    attention_coefficient_conditioning: Literal["static", "query"] = "static"
    # Legacy identity is degree-one affine; tanh supports b + a*tanh(x W).
    attention_coefficient_activation: Literal["identity", "tanh"] = "identity"
    # "direct" learns c_0..c_m; "sum_of_squares" learns q1, q2 with p = q1^2 + q2^2
    # (even m, row-normalized only), so the kernel is nonnegative by construction;
    # "rational" (row-normalized only) learns a numerator a_0..a_m and a denominator
    # 1 + b_1 s + ... + b_m s^m, so the weights are p_a(s) / sum_j p_b(s_j).
    attention_coefficient_parameterization: Literal["direct", "sum_of_squares", "rational"] = "direct"
    # Initial learned coefficients: "taylor" starts every learned polynomial at
    # sum_r s^r / r! (a rational kernel then equals normalized Taylor attention);
    # "linear" starts the numerator at s and the denominator at 1 (a_1 = 1, b_0 = 1,
    # all other coefficients 0), so a rational kernel starts at linear attention
    # divided by the number of visible keys. Not available with sum_of_squares.
    attention_coefficient_initialization: Literal["taylor", "linear"] = "taylor"
    # Rational kernels only: the denominator's degree (null: attention_degree) and form.
    # "direct" learns b_1..b_d of 1 + sum_r b_r s^r; "sum_of_squares" (even d) learns
    # u_1..u_k, w_1..w_k of (1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2, a nonnegative
    # denominator with constant 1 (attention/rational_polynomial.py).
    attention_denominator_degree: int | None = None
    attention_denominator_parameterization: Literal["direct", "sum_of_squares"] = "direct"
    # "fixed" pins the denominator's constant at 1; "learned" learns it with the other
    # coefficients (or factors), leaving the common scale of numerator and denominator free.
    attention_denominator_constant: Literal["fixed", "learned"] = "fixed"
    # Cosine scores for any dot-product mechanism: s = exp(gamma) * cos(q, k) with a
    # learned log scale gamma per layer and head (or one shared), initialized at
    # log(qk_norm_initial_scale); null selects sqrt(qk_head_dim).
    # QK-norm is the default for dot-product attention: unset resolves to on, except for
    # GLA (rejects it) and degree 0 (never reads Q/K). Plain scores need qk_norm: false.
    qk_norm: bool | None = None
    qk_norm_sharing: Literal["global", "per_head"] = "per_head"
    qk_norm_initial_scale: float | None = None
    qk_norm_eps: float = 1e-6
    # Weight initialization: the scaffold's Flax defaults, or an architecture's reference
    # recipe ("transformer_reference" for dot-product attention, "gla_reference" for GLA);
    # see models/initialization.py.
    initialization: Literal["flax_default", "transformer_reference", "gla_reference",
                            "gated_deltanet_reference"] = "flax_default"

    def __post_init__(self):
        if self.kind != "transformer":
            raise ValueError("The shared scaffold supports only kind=transformer")
        if self.tie_embeddings is not False:
            raise ValueError("model.tie_embeddings must be false: the shared scaffold requires untied embeddings")
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
        if self.attention_backend not in ("auto", "blocked", "fused", "manual", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
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
            if self.attention_backend not in ("auto", "blocked", "fused"):
                raise ValueError("Unnormalized attention requires backend='auto', 'blocked' or 'fused'")
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
        if self.attention_coefficient_conditioning not in ("static", "query"):
            raise ValueError(f"Unknown attention coefficient conditioning: {self.attention_coefficient_conditioning}")
        if self.attention_coefficient_activation not in ("identity", "tanh"):
            raise ValueError(f"Unknown attention coefficient activation: {self.attention_coefficient_activation}")
        if self.attention_coefficient_activation != "identity" and self.attention_coefficient_conditioning != "query":
            raise ValueError("Nonlinear coefficient activation requires query-conditioned coefficients")
        if self.attention_coefficient_conditioning == "query":
            if (self.attention_mechanism != "dot_product" or self.attention_coefficients != "learned"
                    or self.attention_coefficient_parameterization != "direct"
                    or self.attention_coefficient_sharing != "per_head"
                    or self.attention_degree is None or self.attention_degree < 1 or self.attention_normalization
                    or self.attention_coefficient_initialization != "taylor"):
                raise ValueError("Query-conditioned coefficients require unnormalized learned/direct "
                                 "per_head polynomial attention with Taylor initialization")
            if self.attention_coefficient_activation == "identity" and self.attention_degree != 1:
                raise ValueError("Query-conditioned identity coefficients require degree-one attention")
        if self.attention_coefficients == "learned":
            if self.attention_mechanism != "dot_product" or self.quadratic_mode != "fixed":
                raise ValueError("Learned polynomial coefficients use dot-product attention without a quadratic mode")
            if self.attention_degree is None or self.attention_degree < 1:
                raise ValueError("Learned polynomial coefficients require an integer attention_degree >= 1")
        if self.attention_coefficient_parameterization not in ("direct", "sum_of_squares", "rational"):
            raise ValueError(f"Unknown coefficient parameterization: {self.attention_coefficient_parameterization}")
        if self.attention_coefficient_initialization not in ("taylor", "linear"):
            raise ValueError(f"Unknown coefficient initialization: {self.attention_coefficient_initialization}")
        if self.attention_coefficient_initialization != "taylor":
            if self.attention_coefficients != "learned":
                raise ValueError("model.attention_coefficient_initialization applies to learned polynomial coefficients only")
            if self.attention_coefficient_parameterization == "sum_of_squares":
                raise ValueError("A sum-of-squares kernel cannot start at linear attention; use the taylor initialization")
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
            if not self.attention_normalization:
                raise ValueError("A sum-of-squares kernel is implemented for row-normalized attention only")
        if self.qk_norm is None:
            self.qk_norm = self.attention_mechanism == "dot_product" and self.attention_degree != 0
        elif self.attention_degree == 0:
            self.qk_norm = False  # Degree 0 ignores Q and K, so QK-norm cannot apply.
        if not isinstance(self.qk_norm, bool):
            raise ValueError("model.qk_norm must be a boolean")
        if self.qk_norm_sharing not in ("global", "per_head"):
            raise ValueError(f"Unknown QK-norm sharing: {self.qk_norm_sharing}")
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


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
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


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def validate_config(config: Config) -> None:
    """Validate loaded/overridden settings before allocating a model."""
    config.model.__post_init__()
    data, optim, train = config.data, config.optim, config.train
    if optim.weight_decay_policy != "exclude_embeddings_norms_biases_v1":
        raise ValueError("optim.weight_decay_policy must be exclude_embeddings_norms_biases_v1")
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
    if not 2 <= data.sequence_length <= config.model.max_seq_len + 1:
        raise ValueError("data.sequence_length must be between 2 and model.max_seq_len + 1")
    if data.source == "synthetic" and not 1 < data.synthetic_vocab_size <= config.model.vocab_size:
        raise ValueError("synthetic_vocab_size must be in [2, model.vocab_size]")
    if data.source == "packed" and not (data.train_manifest and data.eval_manifest):
        raise ValueError("Packed data requires train_manifest and eval_manifest")
    if train.dtype not in ("float32", "bfloat16"):
        raise ValueError("train.dtype must be float32 or bfloat16")
    if (config.model.attention_mechanism == "dot_product"
            and config.model.attention_backend in ("fused", "cudnn") and config.model.attention_degree is None
            and config.model.attention_normalization and train.dtype != "bfloat16"):
        raise ValueError("Fused exact-softmax attention uses cuDNN and requires train.dtype=bfloat16")
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
