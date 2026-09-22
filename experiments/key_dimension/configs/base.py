"""Legacy key_dimension defaults for the shared Transformer scaffold."""

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


KEY_DIMENSIONS = (2, 4, 8, 16, 32, 64, 128)


@dataclass
class ModelConfig(SharedModelConfig):
    attention_backend: str = "manual"


@dataclass
class TrainConfig(SharedTrainConfig):
    workdir: Path = Path("results/key_dimension/smoke_transformer")
    wandb_project: str = "functional-memory-key-dimension"


@dataclass
class Config(SharedConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
