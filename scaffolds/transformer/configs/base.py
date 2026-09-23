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
    tie_embeddings: bool = True
    remat: bool = False
    attention_backend: Literal["auto", "blocked", "fused", "manual", "xla", "cudnn"] = "auto"
    attention_degree: int | None = None
    attention_normalization: bool = True
    attention_mechanism: Literal["dot_product", "gla"] = "dot_product"
    gla_gate_rank: int = 16
    gla_gate_tau: float = 16.0
    gla_chunk_size: int = 64
    gla_inner_chunk_size: int = 16
    gla_use_rope: bool = True
    quadratic_mode: Literal["fixed", "positive", "unrestricted"] = "fixed"
    quadratic_sharing: Literal["global", "per_head"] = "global"

    def __post_init__(self):
        if self.kind != "transformer":
            raise ValueError("The shared scaffold supports only kind=transformer")
        for name in (
            "vocab_size", "max_seq_len", "num_layers", "hidden_size", "num_heads",
            "num_kv_heads", "qk_head_dim", "value_head_dim", "intermediate_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.num_kv_heads != self.num_heads:
            raise ValueError("This scaffold uses full MHA: num_kv_heads must equal num_heads")
        if self.attention_mechanism not in ("dot_product", "gla"):
            raise ValueError(f"Unknown attention mechanism: {self.attention_mechanism}")
        if self.qk_head_dim % 2 and (self.attention_mechanism != "gla" or self.gla_use_rope):
            raise ValueError("model.qk_head_dim must be even for RoPE")
        if self.attention_backend not in ("auto", "blocked", "fused", "manual", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if not isinstance(self.attention_normalization, bool):
            raise ValueError("model.attention_normalization must be a boolean")
        if self.attention_mechanism == "gla":
            if self.attention_degree is not None or self.quadratic_mode != "fixed":
                raise ValueError("GLA cannot use Taylor degrees or learned quadratic modes")
            if not self.attention_normalization:
                raise ValueError("attention_normalization is a dot-product option; GLA requires its default true")
            if self.attention_backend not in ("auto", "blocked", "fused"):
                raise ValueError("GLA requires backend='auto', 'blocked' or 'fused'")
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
    workdir: Path = Path("results/shared_transformer/smoke")
    checkpoint_dir: Path | None = None
    micro_batch_size: int = 2
    log_every: int = 10
    eval_every: int = 10
    eval_batches: int = 1
    ckpt_every: int = 10
    wandb_project: str = "functional-memory-shared-transformer"
    wandb_entity: str | None = None
    wandb_run_name: str = "smoke-transformer"
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
    if config.model.attention_mechanism == "gla" and config.model.attention_backend == "fused":
        if (data.sequence_length - 1) % 64:
            raise ValueError("Fused GLA requires prediction sequence length divisible by 64")
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
