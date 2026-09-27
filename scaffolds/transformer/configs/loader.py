from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from scaffolds.transformer.configs.base import Config, validate_config


def load_config(path: str | Path, config_factory=Config) -> Config:
    config_path = Path(path)
    with config_path.open() as stream:
        # JSON scientific notation (e.g. 1e-06) is not parsed as a number by
        # PyYAML's YAML 1.1 resolver. Resolved run configs must round-trip exactly.
        raw = json.load(stream) if config_path.suffix == ".json" else yaml.safe_load(stream)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    config = config_factory()
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
            path_value = isinstance(current, Path) or key in ("checkpoint_dir", "wandb_dir")
            setattr(target, key, Path(value) if path_value and value is not None else value)


def config_to_dict(config: Config) -> dict[str, Any]:
    values = asdict(config)
    if config.model.attention_fixed_coefficients is not None:
        values["model"]["attention_fixed_coefficients"] = list(config.model.attention_fixed_coefficients)
    values["train"]["workdir"] = str(config.train.workdir)
    for key in ("checkpoint_dir", "wandb_dir"):
        if getattr(config.train, key) is not None:
            values["train"][key] = str(getattr(config.train, key))
    return values


def config_dicts_equal(left: dict, right: dict) -> bool:
    """Compare new configs with legacy fixed configs without altering provenance."""
    normalized = []
    for value in (left, right):
        value = copy.deepcopy(value)
        value.setdefault("model", {}).setdefault("attention_degree", None)
        value["model"].setdefault("attention_normalization", True)
        value["model"].setdefault("attention_mechanism", "dot_product")
        for key, default in (("gla_gate_rank", 16), ("gla_gate_tau", 16.0),
                             ("gla_chunk_size", 64), ("gla_inner_chunk_size", 16),
                             ("gla_use_rope", True)):
            value["model"].setdefault(key, default)
        value["model"].setdefault("quadratic_mode", "fixed")
        value["model"].setdefault("quadratic_sharing", "global")
        # Runs recorded before QK-norm existed trained without it, whatever the current default.
        value["model"].setdefault("qk_norm", False)
        value["model"].setdefault("attention_coefficients", "fixed")
        fixed_coefficients = value["model"].setdefault("attention_fixed_coefficients", None)
        if isinstance(fixed_coefficients, (tuple, list)):
            value["model"]["attention_fixed_coefficients"] = list(fixed_coefficients)
        value["model"].setdefault("attention_coefficient_parameterization", "direct")
        value["model"].setdefault("attention_coefficient_sharing", "global")
        for key, default in (("qk_norm", False), ("qk_norm_sharing", "per_head"),
                             ("qk_norm_initial_scale", None), ("qk_norm_eps", 1e-6)):
            value["model"].setdefault(key, default)
        value.setdefault("train", {}).setdefault("checkpoint_dir", None)
        for key in ("wandb_group", "wandb_dir"):
            value["train"].setdefault(key, None)
        # The W&B mirror mode is an execution setting; wandb_run.json pins the run's identity.
        value["train"].pop("wandb_mode", None)
        normalized.append(value)
    return normalized[0] == normalized[1]


def save_config(config: Config, path: str | Path) -> None:
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
