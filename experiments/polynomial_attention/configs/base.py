"""Fixed softmax recipe with Taylor degree as the sole sweep variable."""

from dataclasses import dataclass, field
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
    attention_backend: Literal["blocked", "fused"] = "blocked"
    attention_degree: int | None = None

    def __post_init__(self):
        if self.kind != "transformer":
            raise ValueError("This experiment supports only kind=transformer")
        for name in (
            "vocab_size", "max_seq_len", "num_layers", "hidden_size", "num_heads",
            "num_kv_heads", "qk_head_dim", "value_head_dim", "intermediate_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.num_kv_heads != self.num_heads:
            raise ValueError("This experiment uses full MHA: num_kv_heads must equal num_heads")
        if self.qk_head_dim % 2:
            raise ValueError("model.qk_head_dim must be even for RoPE")
        if self.attention_backend not in ("blocked", "fused"):
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if self.attention_degree is not None and (
            not isinstance(self.attention_degree, int)
            or isinstance(self.attention_degree, bool)
            or self.attention_degree < 0
            or self.attention_degree % 2
        ):
            raise ValueError("model.attention_degree must be null or a nonnegative even integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.dropout != 0:
            raise ValueError("This fixed-recipe experiment requires dropout=0")
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
    workdir: Path = Path("results/polynomial_attention/smoke_transformer")
    micro_batch_size: int = 2
    log_every: int = 10
    eval_every: int = 10
    eval_batches: int = 1
    ckpt_every: int = 10
    wandb_project: str = "functional-memory-polynomial-attention"
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
    if config.model.attention_backend == "fused" and config.model.attention_degree is None and train.dtype != "bfloat16":
        raise ValueError("Fused exact-softmax attention uses cuDNN and requires train.dtype=bfloat16")
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
