"""Explicit native FLA configuration with selective decay and matched token budgets.

No Transformer/JAX defaults are loaded. The upstream training reference is
Flame's generic ``flame/models/fla.toml``, not a claimed GDN-specific published
execution. Its eight-device example has a global batch of 256 sequences; GPU
count is deliberately replaced by accumulation in this runtime. The native model,
learning rates, Adam settings, and 5% warmup are preserved; selective decay and
the token budget/evaluation cadence are explicit project adaptations.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


FLA_COMMIT = "516143e31fce09925e6c39ac37148444bad176c4"
FLAME_COMMIT = "24dea9133e2166da89a563afbdcefb7c55615646"
TORCHTITAN_COMMIT = "0b44d4c437c424b6bf719661c0eb4283dc4068bc"
REFERENCE = {
    "fla_commit": FLA_COMMIT,
    "flame_commit": FLAME_COMMIT,
    "training_config": "flame/models/fla.toml",
    "torchtitan_commit": TORCHTITAN_COMMIT,
}

# Dimensions are the only architecture adaptations. In particular, preserve
# upstream per-head dimensions, projection ratios and the FFN sizing rule.
_MODEL = {
    "hidden_size": 1024,
    "num_hidden_layers": 9,
    "num_heads": 3,
    "num_v_heads": None,
    "head_dim": 256,
    "expand_v": 2.0,
    "intermediate_size": 2816,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "initializer_range": 0.02,
    "norm_eps": 1e-6,
    "use_gate": True,
    "use_short_conv": True,
    "conv_size": 4,
    "allow_neg_eigval": False,
    "attn_mode": "chunk",
    "attn": None,
    "hidden_ratio": 4,
    "hidden_act": "swish",
    "fuse_norm": True,
    "fuse_swiglu": True,
    "fuse_cross_entropy": True,
    "fuse_linear_cross_entropy": False,
    "use_l2warp": False,
    "attnres_block_size": None,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": None,
    "use_cache": False,
}
_SIZE_FIELDS = {"hidden_size", "num_hidden_layers", "num_heads", "num_v_heads", "intermediate_size"}
_OPTIM = {
    "name": "AdamW",
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-15,
    "weight_decay": 0.1,
    # Exempt embeddings, norms, biases and native _no_weight_decay parameters.
    "weight_decay_policy": "exclude_embeddings_norms_biases_v1",
    "grad_clip_norm": 1.0,
    "warmup_steps": 379,
    "total_steps": 7570,
    "schedule": "cosine",
    "implementation": "fused",
}
_DATA = {
    "source": "packed",
    "train_manifest": None,
    "eval_manifest": None,
    "batch_size": 256,
    "sequence_length": 2049,
    "shuffle_seed": 0,
}
_TRAIN = {
    "seed": 42,
    "dtype": "bfloat16",
    "micro_batch_size": 8,
    "log_every": 10,
    "eval_every": 500,
    "eval_early_every": 100,
    "eval_early_until": 1500,
    "eval_batches": 19,
    "ckpt_every": 1024,
    "source_commit": "auto",
    "workdir": None,
    "checkpoint_dir": None,
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_group": None,
    "wandb_run_name": None,
    "wandb_mode": "online",
}


def reference_recipe(*, total_steps: int = 7570) -> dict[str, Any]:
    """Return explicit scientific settings, with paths and W&B destination unset.

    The allocator must set the local dataset manifests, run paths and W&B
    destination before validating or launching this recipe. The packed corpus
    and evaluation/checkpoint cadence are project inputs, not upstream defaults.
    The default budget is 7,570 updates. An explicit token budget changes only
    the training horizon, proportional warmup and evaluation cadence. Warmup
    retains the reference's 5% fraction, rounded half up to an integer with a
    minimum of one update. Evaluation is more frequent early in training, with
    both intervals scaled proportionally from the 7,570-update schedule.
    """
    warmup_steps = _budget_warmup(total_steps)
    recipe = deepcopy({"implementation": "native_fla", "reference": REFERENCE,
                       "data": _DATA, "model": _MODEL, "optim": _OPTIM, "train": _TRAIN})
    recipe["optim"].update(total_steps=total_steps, warmup_steps=warmup_steps)
    recipe["train"].update(_budget_evaluation(total_steps))
    return recipe


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _fields(value: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    missing = expected.keys() - value.keys()
    unknown = value.keys() - expected.keys()
    if missing or unknown:
        raise ValueError(f"{name} fields differ: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _budget_warmup(total_steps: int) -> int:
    _positive_int(total_steps, "optim.total_steps")
    # Exact half-up rounding of total_steps / 20, without float or banker rounding.
    return max(1, (total_steps + 10) // 20)


def _budget_evaluation(total_steps: int) -> dict[str, int]:
    """Scale the project's evaluation cadence using exact half-up rounding."""
    _positive_int(total_steps, "optim.total_steps")

    def scaled(updates: int) -> int:
        return (2 * updates * total_steps + 7570) // (2 * 7570)

    return {"eval_every": max(1, scaled(500)),
            "eval_early_every": max(1, scaled(100)),
            "eval_early_until": scaled(1500)}


def _reference_value(value: Any, expected: Any, name: str) -> None:
    # YAML booleans must not compare equal to integers such as num_heads=1.
    if type(expected) is bool:
        same = type(value) is bool and value == expected
    elif type(expected) is int:
        same = type(value) is int and value == expected
    elif isinstance(expected, float):
        same = (type(value) in (float, int) and math.isfinite(value)
                and math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0))
    else:
        same = value == expected
    if not same:
        raise ValueError(f"{name} must match the pinned reference: {expected!r}, got {value!r}")


def validate_config(config: dict[str, Any]) -> None:
    """Reject omitted defaults and undeclared deviations from the reference.

    This entrypoint permits an explicit token budget with proportional warmup.
    Other training recipes need an explicit reference/profile, not silent overrides.
    Qualification may construct small upstream models directly from a copied
    model dictionary; it does not change a production recipe or its schedule.
    """
    config = _mapping(config, "config")
    expected = reference_recipe()
    _fields(config, expected, "config")
    _reference_value(config["implementation"], "native_fla", "implementation")
    for section in ("reference", "data", "model", "optim", "train"):
        _fields(_mapping(config[section], section), expected[section], section)
    for key, value in REFERENCE.items():
        _reference_value(config["reference"][key], value, f"reference.{key}")
    warmup_steps = _budget_warmup(config["optim"]["total_steps"])
    for key, value in _OPTIM.items():
        if key not in {"total_steps", "warmup_steps"}:
            _reference_value(config["optim"][key], value, f"optim.{key}")
    _reference_value(config["optim"]["warmup_steps"], warmup_steps, "optim.warmup_steps")

    model = config["model"]
    for key, value in _MODEL.items():
        if key not in _SIZE_FIELDS:
            _reference_value(model[key], value, f"model.{key}")
    for key in _SIZE_FIELDS - {"num_v_heads"}:
        _positive_int(model[key], f"model.{key}")
    if model["num_v_heads"] is not None:
        _positive_int(model["num_v_heads"], "model.num_v_heads")
        if model["num_v_heads"] != model["num_heads"]:
            raise ValueError("model.num_v_heads must equal num_heads: the reference uses no grouped value attention")
    if 4 * model["num_heads"] * model["head_dim"] != 3 * model["hidden_size"]:
        raise ValueError("The reference key projection width must be 0.75 * hidden_size")
    intermediate = math.ceil(int(model["hidden_size"] * model["hidden_ratio"] * 2 / 3) / 256) * 256
    if model["intermediate_size"] != intermediate:
        raise ValueError(f"model.intermediate_size must follow FLA's FFN sizing rule: {intermediate}")

    data, train = config["data"], config["train"]
    for key in ("source", "batch_size", "sequence_length", "shuffle_seed"):
        _reference_value(data[key], _DATA[key], f"data.{key}")
    for key in ("train_manifest", "eval_manifest"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"data.{key} must name the explicit packed-data manifest")
    for key in ("seed", "dtype"):
        _reference_value(train[key], _TRAIN[key], f"train.{key}")
    for key in ("micro_batch_size", "log_every", "eval_every", "eval_early_every", "eval_batches", "ckpt_every"):
        _positive_int(train[key], f"train.{key}")
    if type(train["eval_early_until"]) is not int or train["eval_early_until"] < 0:
        raise ValueError("train.eval_early_until must be a nonnegative integer")
    if data["batch_size"] % train["micro_batch_size"]:
        raise ValueError("train.micro_batch_size must divide the scientific global batch")
    for key in ("workdir", "checkpoint_dir", "source_commit"):
        if not isinstance(train[key], str) or not train[key].strip():
            raise ValueError(f"train.{key} must be explicit")
    if train["wandb_mode"] not in ("online", "offline", "disabled"):
        raise ValueError("train.wandb_mode must be online, offline or disabled")
    if train["wandb_mode"] != "disabled":
        for key in ("wandb_project", "wandb_entity", "wandb_group", "wandb_run_name"):
            if not isinstance(train[key], str) or not train[key].strip():
                raise ValueError(f"train.{key} must be explicit when W&B is enabled")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate a complete recipe without merging other defaults."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def fla_model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Return native FLA keyword arguments, including for qualification fixtures."""
    return deepcopy(config["model"])


def parameter_count(config: dict[str, Any]) -> int:
    """Count the plain upstream GDN model without importing Torch or allocating it."""
    m = config["model"]
    d, h, layers = m["hidden_size"], m["num_heads"], m["num_hidden_layers"]
    hv = m["num_v_heads"] if m["num_v_heads"] is not None else h
    k, v = m["head_dim"], int(m["head_dim"] * m["expand_v"])
    qk, values = h * k, hv * v
    mixer = d * (2 * qk + 3 * values) + 2 * d * hv
    mixer += m["conv_size"] * (2 * qk + values) + 2 * hv + v
    blocks = layers * (mixer + 3 * d * m["intermediate_size"] + 2 * d)
    embeddings = (1 if m["tie_word_embeddings"] else 2) * m["vocab_size"] * d
    return embeddings + blocks + d
