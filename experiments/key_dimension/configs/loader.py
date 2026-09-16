from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.key_dimension.configs.base import Config, validate_config


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    config = Config()
    _update_dataclass(config, raw, context=str(config_path))
    validate_config(config)
    return config


def _update_dataclass(target: Any, values: dict[str, Any], *, context: str) -> None:
    field_names = {field.name for field in fields(target)}
    for key, value in values.items():
        if key not in field_names:
            raise ValueError(f"Unknown config key {context}.{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section {context}.{key} must be a mapping")
            _update_dataclass(current, value, context=f"{context}.{key}")
        else:
            setattr(target, key, Path(value) if isinstance(current, Path) else value)


def config_to_dict(config: Config) -> dict[str, Any]:
    values = asdict(config)
    values["train"]["workdir"] = str(config.train.workdir)
    return values


def save_config(config: Config, path: str | Path) -> None:
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
