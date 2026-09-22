"""Legacy polynomial_attention defaults for the shared Transformer scaffold."""

from dataclasses import dataclass, field
from pathlib import Path

from experiments.shared_transformer.configs.base import (
    OFM_SOURCE_COMMIT,
    Config as SharedConfig,
    DataConfig,
    ModelConfig as SharedModelConfig,
    OptimConfig,
    TrainConfig as SharedTrainConfig,
    validate_config,
)


ATTENTION_DEGREES = (0, 2, 4, 8, 16, None)


@dataclass
class ModelConfig(SharedModelConfig):
    attention_backend: str = "blocked"


@dataclass
class TrainConfig(SharedTrainConfig):
    workdir: Path = Path("results/polynomial_attention/smoke_transformer")
    wandb_project: str = "functional-memory-polynomial-attention"


@dataclass
class Config(SharedConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
