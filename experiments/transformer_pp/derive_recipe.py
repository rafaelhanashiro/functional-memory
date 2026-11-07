"""Derive a Transformer++ recipe from a partner run at another model shape or token budget.

The study fixes the recipe's rules, not its values. A derived recipe copies the partner's
recipe and recomputes only what the rules determine:

- shape: ``num_heads = num_kv_heads = width / value_head_dim`` and a SwiGLU width of
  18/7 x width rounded to a multiple of 64 (2,304 at width 896, 1,664 at width 640);
- 1x Chinchilla: 20 prediction tokens per parameter of the study's normalized-softmax
  reference model at that shape, held for every attention variant at the shape;
- batch: a shape change holds the update count by scaling the batch with that parameter
  count, rounded to a power-of-two multiple of the reference's 256 sequences; a budget
  change holds the batch and scales the updates;
- warmup: 6.01% of the updates (455 of 7,570), rounded half up;
- validation: the reference's 4,864 validation sequences (19 batches of 256) at every
  batch, with the update cadence scaled by the budget; checkpoints every 500 updates;
- learning rate: the partner's peak, or ``--peak-learning-rate``, with a floor of 0.1 x
  peak;
- initialization: a nonzero query-coefficient projection std keeps its multiple of
  1/sqrt(width), since recipes store it as an absolute value. Every other width- or
  depth-dependent initialization is derived from the config in code.

Keys the scaffold no longer defines are dropped and reported; any other difference from the
partner fails. Allocate the derived recipe with ``python -m scaffolds.records create``.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
import sys

import yaml

STUDY = Path(__file__).resolve().parent
RUNS = STUDY / "runs"
REFERENCE_RECIPE = STUDY / "recipe.yaml"

TOKENS_PER_PARAMETER = 20
REFERENCE_BATCH = 256
REFERENCE_UPDATES = 7570
REFERENCE_WARMUP = 455
REFERENCE_CADENCE = {"eval_early_every": 100, "eval_early_until": 1500, "eval_every": 500}
VALIDATION_SEQUENCES = 19 * REFERENCE_BATCH
SWIGLU_RATIO = 18 / 7
SWIGLU_MULTIPLE = 64
MIN_LEARNING_RATE_FRACTION = 0.1
SHAPE_KEYS = ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "intermediate_size")
DERIVED = {
    "model": (*SHAPE_KEYS, "attention_coefficient_projection_initial_std"),
    "data": ("batch_size",),
    "optim": ("total_steps", "warmup_steps", "learning_rate", "min_learning_rate"),
    "train": ("eval_batches", *REFERENCE_CADENCE),
}


def round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def shape(layers: int, width: int, value_head_dim: int, intermediate_size: int | None = None) -> dict[str, int]:
    """The study's proportions at a depth and width."""
    if layers < 1 or width < 1 or width % value_head_dim:
        raise ValueError(f"width {width} must be a positive multiple of the value head width {value_head_dim}")
    heads = width // value_head_dim
    if intermediate_size is None:
        intermediate_size = SWIGLU_MULTIPLE * round_half_up(SWIGLU_RATIO * width / SWIGLU_MULTIPLE)
    return {"num_layers": layers, "hidden_size": width, "num_heads": heads, "num_kv_heads": heads,
            "intermediate_size": intermediate_size}


def parameter_count(model: dict) -> int:
    from scaffolds.transformer.configs import ModelConfig
    from scaffolds.transformer.models.registry import model_parameter_count

    return model_parameter_count(ModelConfig(**model))


def reference_parameter_count(model_shape: dict[str, int] | None = None) -> int:
    """Parameters of the study's normalized-softmax reference recipe, at its own shape or another.

    The count at a shape defines that shape's 1x budget for every attention variant.
    """
    return parameter_count({**yaml.safe_load(REFERENCE_RECIPE.read_text())["model"], **(model_shape or {})})


def batch_size(reference_parameters: int, reference_shape_parameters: int) -> int:
    """Sequences per update that hold the reference's update count at this parameter count."""
    return int(REFERENCE_BATCH * 2 ** round(math.log2(reference_parameters / reference_shape_parameters)))


def schedule(reference_parameters: int, batch: int, sequence_length: int, budget: float) -> dict[str, int]:
    if budget <= 0:
        raise ValueError("the token budget must be positive")
    if VALIDATION_SEQUENCES % batch:
        raise ValueError(f"batch {batch} does not divide the {VALIDATION_SEQUENCES} validation sequences")
    updates = round_half_up(budget * TOKENS_PER_PARAMETER * reference_parameters / (batch * (sequence_length - 1)))
    return {"total_steps": updates, "warmup_steps": round_half_up(updates * REFERENCE_WARMUP / REFERENCE_UPDATES),
            "eval_batches": VALIDATION_SEQUENCES // batch,
            **{key: round_half_up(value * budget) for key, value in REFERENCE_CADENCE.items()}}


def _defined(recipe: dict) -> tuple[dict, list[str]]:
    """The recipe restricted to keys the scaffold's Config defines, and the keys dropped."""
    from scaffolds.transformer.configs import Config

    dropped = []

    def restrict(values: dict, target, prefix: str) -> dict:
        names = {field.name: getattr(target, field.name) for field in fields(target)}
        kept = {}
        for key, value in values.items():
            if key not in names:
                dropped.append(f"{prefix}{key}")
            elif is_dataclass(names[key]) and isinstance(value, dict):
                kept[key] = restrict(value, names[key], f"{prefix}{key}.")
            else:
                kept[key] = value
        return kept

    return restrict(recipe, Config(), ""), dropped


def derive(partner: dict, *, layers: int | None = None, width: int | None = None,
           intermediate_size: int | None = None, budget: float = 1.0, batch: int | None = None,
           peak_learning_rate: float | None = None) -> tuple[dict, list[str]]:
    """Return the derived recipe and the partner keys the scaffold no longer defines."""
    recipe, dropped = _defined(copy.deepcopy(partner))
    model, data, optim, train = (recipe[section] for section in ("model", "data", "optim", "train"))
    old_width = model["hidden_size"]
    if layers is None and width is None and intermediate_size is None:
        model_shape = {key: model[key] for key in SHAPE_KEYS}  # a budget change keeps the partner's shape
    else:
        model_shape = shape(model["num_layers"] if layers is None else layers, old_width if width is None else width,
                            model["value_head_dim"], intermediate_size)
    reference_parameters = reference_parameter_count(model_shape)
    if batch is None:
        batch = batch_size(reference_parameters, reference_parameter_count())
    if batch % train["micro_batch_size"]:
        raise ValueError(f"batch {batch} is not a multiple of micro_batch_size {train['micro_batch_size']}")
    derived = schedule(reference_parameters, batch, data["sequence_length"], budget)
    model.update(model_shape)
    if model.get("attention_coefficient_projection_initial_std"):
        # Keep the std's multiple of 1/sqrt(width): 1/sqrt(896) becomes 1/sqrt(640).
        gain = float(f"{model['attention_coefficient_projection_initial_std'] * math.sqrt(old_width):.12g}")
        model["attention_coefficient_projection_initial_std"] = gain * model["hidden_size"] ** -0.5
    data["batch_size"] = batch
    optim["total_steps"], optim["warmup_steps"] = derived["total_steps"], derived["warmup_steps"]
    optim["learning_rate"] = optim["learning_rate"] if peak_learning_rate is None else peak_learning_rate
    optim["min_learning_rate"] = MIN_LEARNING_RATE_FRACTION * optim["learning_rate"]
    train.update({key: derived[key] for key in ("eval_batches", *REFERENCE_CADENCE)})
    changed = _changes(_defined(copy.deepcopy(partner))[0], recipe)
    undeclared = [key for key in changed if key.split(".", 1)[1] not in DERIVED.get(key.split(".", 1)[0], ())]
    if undeclared:
        raise AssertionError(f"derivation changed keys outside the rules: {undeclared}")
    return recipe, dropped


def _changes(before: dict, after: dict, prefix: str = "") -> dict[str, tuple]:
    changes = {}
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            changes.update(_changes(old, new, f"{prefix}{key}."))
        elif old != new:
            changes[f"{prefix}{key}"] = (old, new)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--partner", required=True, help="run ID whose frozen recipe is copied")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--intermediate-size", type=int, help="override the 18/7 x width SwiGLU rule")
    parser.add_argument("--budget", type=float, default=1.0, help="multiple of 1x Chinchilla")
    parser.add_argument("--batch", type=int, help="override the shape rule's batch")
    parser.add_argument("--peak-learning-rate", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    partner = yaml.safe_load((RUNS / args.partner / "config.yaml").read_text())
    recipe, dropped = derive(partner, layers=args.layers, width=args.width, intermediate_size=args.intermediate_size,
                             budget=args.budget, batch=args.batch, peak_learning_rate=args.peak_learning_rate)
    args.output.write_text(yaml.safe_dump(recipe, sort_keys=False))
    for key, (old, new) in _changes(_defined(partner)[0], recipe).items():
        print(f"{key}: {old} -> {new}")
    for key in dropped:
        print(f"dropped {key}: not a scaffold config key")
    print(f"parameters: {parameter_count(recipe['model']):,} "
          f"(1x reference at this shape: {reference_parameter_count({k: recipe['model'][k] for k in SHAPE_KEYS}):,})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
