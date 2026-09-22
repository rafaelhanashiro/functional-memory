"""Shared configuration loading with the original family defaults."""

from pathlib import Path

from scaffolds.transformer.studies.polynomial_attention.configs.base import Config
from scaffolds.transformer.configs.loader import (
    _update_dataclass,
    config_dicts_equal,
    config_to_dict,
    load_config as _load_config,
    save_config,
)


def load_config(path: str | Path) -> Config:
    return _load_config(path, config_factory=Config)
