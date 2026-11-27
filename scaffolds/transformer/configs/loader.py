from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from scaffolds.transformer.configs.base import Config, validate_config

SCORE_MLP_CONSTANTS = tuple(f"attention_score_mlp_{name}_initial" for name in ("amplitude", "slope", "offset"))


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
    model_values = raw.get("model", {})
    if (isinstance(model_values, dict) and "qk_norm" not in model_values
            and model_values.get("attention_mechanism", config.model.attention_mechanism)
            != config.model.attention_mechanism):
        # The factory resolves the default before YAML changes the mechanism.
        # Resolve it again for that mechanism; preserve explicit recipe choices
        # and custom-factory defaults when the mechanism itself is unchanged.
        config.model.qk_norm = None
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
    if config.model.attention_fixed_square_factors is not None:
        values["model"]["attention_fixed_square_factors"] = list(config.model.attention_fixed_square_factors)
    if config.model.attention_spline_knots is not None:
        values["model"]["attention_spline_knots"] = list(config.model.attention_spline_knots)
    for key in SCORE_MLP_CONSTANTS:
        if getattr(config.model, key) is not None:
            values["model"][key] = list(getattr(config.model, key))
    if config.model.attention_spline_initial_power_coefficients is not None:
        values["model"]["attention_spline_initial_power_coefficients"] = list(config.model.attention_spline_initial_power_coefficients)
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
        value.setdefault("continuation", {"parent_config": "", "parent_config_sha256": "",
                                          "parent_checkpoint": "", "parent_checkpoint_sha256": "",
                                          "parent_data_identity_sha256": "",
                                          "parent_step": 0})
        value.setdefault("model", {}).setdefault("attention_degree", None)
        value["model"].setdefault("attention_normalization", True)
        value["model"].setdefault("attention_output_scaling", "none")
        value["model"].setdefault("attention_exp_shift", "none")
        value["model"].setdefault("attention_head_rms_prescale", "none")
        value["model"].setdefault("attention_polynomial_shift", "none")
        value["model"].setdefault("initialization", "flax_default")
        value["model"].setdefault("rms_norm_fp32", False)
        value["model"].setdefault("gdn_conv_size", 4)
        value["model"].setdefault("gdn_erase_mode", "tied")
        value["model"].setdefault("gdn_write_activation", "sigmoid")
        value["model"].setdefault("gdn_polynomial_degree", 1)
        value["model"].setdefault("attention_mechanism", "dot_product")
        for key, default in (("gla_gate_rank", 16), ("gla_gate_tau", 16.0),
                             ("gla_chunk_size", 64), ("gla_inner_chunk_size", 16),
                             ("gla_use_rope", True)):
            value["model"].setdefault(key, default)
        value["model"].setdefault("quadratic_mode", "fixed")
        value["model"].setdefault("quadratic_sharing", "global")
        # Runs recorded before QK-norm existed trained without it, whatever the current default.
        value["model"].setdefault("qk_norm", False)
        value["model"].setdefault("qk_norm_head_scale", False)
        value["model"].setdefault("qk_norm_token_scale", "none")
        value["model"].setdefault("qk_norm_token_scale_kernel_weight_decay", False)
        value["continuation"].setdefault("new_parameter_learning_rate", None)
        value["continuation"].setdefault("preserve_all_parameters", False)
        value["continuation"].setdefault("optimizer", "parent")
        value["continuation"].setdefault("data_start_sequence", None)
        value["model"].setdefault("attention_coefficients", "fixed")
        fixed_coefficients = value["model"].setdefault("attention_fixed_coefficients", None)
        if isinstance(fixed_coefficients, (tuple, list)):
            value["model"]["attention_fixed_coefficients"] = list(fixed_coefficients)
        fixed_square_factors = value["model"].setdefault("attention_fixed_square_factors", None)
        if isinstance(fixed_square_factors, (tuple, list)):
            value["model"]["attention_fixed_square_factors"] = list(fixed_square_factors)
        value["model"].setdefault("attention_kernel_basis", "polynomial")
        spline_knots = value["model"].setdefault("attention_spline_knots", None)
        if isinstance(spline_knots, (tuple, list)):
            value["model"]["attention_spline_knots"] = [float(knot) for knot in spline_knots]
        value["model"].setdefault("attention_spline_initial_intercept", None)
        value["model"].setdefault("attention_spline_initial_slope", None)
        spline_powers = value["model"].setdefault("attention_spline_initial_power_coefficients", None)
        if isinstance(spline_powers, (tuple, list)):
            value["model"]["attention_spline_initial_power_coefficients"] = [float(coefficient) for coefficient in spline_powers]
        value["model"].setdefault("attention_coefficient_parameterization", "direct")
        value["model"].setdefault("attention_coefficient_initialization", "taylor")
        value["model"].setdefault("attention_denominator_degree", None)
        value["model"].setdefault("attention_denominator_parameterization", "direct")
        value["model"].setdefault("attention_denominator_constant", "fixed")
        value.setdefault("optim", {}).setdefault("coefficient_penalty_denominator", True)
        value["optim"].setdefault("query_coefficient_learning_rate_scale", 1.0)
        for key in ("coefficient_release_step", "coefficient_release_schedule", "coefficient_release_end_step"):
            value["optim"].setdefault(key, None)
        value["model"].setdefault("attention_coefficient_sharing", "global")
        value["model"].setdefault("attention_coefficient_conditioning", "static")
        value["model"].setdefault("attention_coefficient_activation", "identity")
        value["model"].setdefault("attention_coefficient_input_normalization", "none")
        value["model"].setdefault("attention_coefficient_input_source", "post_rmsnorm")
        value["model"].setdefault("attention_coefficient_readout", "diagonal")
        value["model"].setdefault("attention_coefficient_readout_scale", 1.0)
        value["model"].setdefault("attention_coefficient_feature_count", None)
        value["model"].setdefault("attention_coefficient_amplitude_initial", 1.0)
        value["model"].setdefault("attention_coefficient_projection_initial_std", 0.0)
        value["model"].setdefault("attention_coefficient_tanh_bias", False)
        value["model"].setdefault("attention_coefficient_gate", "none")
        value["model"].setdefault("attention_coefficient_kernel_weight_decay", False)
        value["model"].setdefault("attention_forget_gate", False)
        value["model"].setdefault("attention_forget_gate_mode", "cumulative")
        value["model"].setdefault("attention_forget_gate_form", "input")
        value["model"].setdefault("attention_forget_gate_constant_rate", "learned")
        value["model"].setdefault("attention_forget_gate_conditioning", "query")
        value["model"].setdefault("attention_forget_gate_rate_activation", "softplus")
        value["model"].setdefault("attention_forget_gate_per_order", False)
        for key, default in (("attention_score_mlp_width", None), ("attention_score_mlp_form", "kernel"),
                             ("attention_score_mlp_activation", "silu"),
                             ("attention_score_mlp_architecture", "dynamic_readout"),
                             ("attention_score_mlp_kernel_weight_decay", False)):
            value["model"].setdefault(key, default)
        for key in SCORE_MLP_CONSTANTS:
            constants = value["model"].setdefault(key, None)
            if isinstance(constants, (tuple, list)):
                value["model"][key] = list(constants)
        value["model"].setdefault("positional_encoding", "rope")
        value["model"].setdefault("attention_blocked_implementation", "scan")
        value["model"].setdefault("path_w_rank", 32)
        value["model"].setdefault("path_conv_size", 3)
        value["model"].setdefault("attention_coefficient_amplitude_scale", "none")
        value["model"].setdefault("attention_coefficient_scale", "none")
        orders = value["model"].setdefault("attention_coefficient_orders", None)
        if isinstance(orders, (tuple, list)):
            value["model"]["attention_coefficient_orders"] = list(orders)
        value.setdefault("optim", {}).setdefault("coefficient_penalty", 0.0)
        for key, default in (("qk_norm", False), ("qk_norm_type", "cosine_log_scale"),
                             ("qk_norm_sharing", "per_head"),
                             ("qk_norm_initial_scale", None), ("qk_norm_eps", 1e-6)):
            value["model"].setdefault(key, default)
        value.setdefault("train", {}).setdefault("checkpoint_dir", None)
        value["train"].setdefault("eval_early_every", None)
        value["train"].setdefault("eval_early_until", 0)
        for key in ("wandb_group", "wandb_dir"):
            value["train"].setdefault(key, None)
        for key, default in (("gradient_event_capture", False), ("gradient_event_factor", 5.0),
                             ("gradient_event_min_norm", 1.0), ("gradient_event_history", 100),
                             ("gradient_event_min_history", 20), ("gradient_event_max_states", 3),
                             ("retain_terminal_checkpoint", False)):
            value["train"].setdefault(key, default)
        value["model"].setdefault("attention_score_bias", "none")
        value["model"].setdefault("attention_score_transform", "identity")
        value["model"].setdefault("attention_score_prefactor", "none")
        value["model"].setdefault("attention_coefficient_constant_initial", None)
        value["model"].setdefault("attention_coefficient_zero_constant", False)
        # The W&B mirror mode is an execution setting; wandb_run.json pins the run's identity.
        value["train"].pop("wandb_mode", None)
        normalized.append(value)
    return normalized[0] == normalized[1]


def save_config(config: Config, path: str | Path) -> None:
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
