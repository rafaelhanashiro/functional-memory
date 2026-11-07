"""Derive a Transformer++ recipe from a partner run at another model shape or token budget.

The study fixes the recipe's rules, not its values. A derived recipe copies the partner's
recipe and recomputes only what the rules determine:

- shape: ``num_heads = num_kv_heads = width / value_head_dim`` and a SwiGLU width of
  18/7 x width rounded to a multiple of 64 (2,304 at width 896, 1,664 at width 640);
- 1x Chinchilla: 20 prediction tokens per parameter of the study's normalized-softmax
  reference model (64-wide heads) at that depth and width, held for every attention
  variant at the shape;
- batch: a shape change holds the update count by scaling the tokens per update (524,288
  at the reference) with that parameter count, rounded to a power-of-two multiple; a
  budget change keeps the partner's batch and scales the updates;
- warmup: 6.01% of the updates (455 of 7,570), rounded half up;
- validation: the reference's 9,961,472 validation tokens (4,864 sequences of 2,049) at
  every batch, with the update cadence scaled by the budget; checkpoints every 500 updates;
- learning rate: the partner's peak, or ``--peak-learning-rate``, with a floor of 0.1 x
  peak;
- initialization: a nonzero query-coefficient projection std keeps its multiple of
  1/sqrt(width), since recipes store it as an absolute value. Every other width- or
  depth-dependent initialization is derived from the config in code.

The budget defaults to the partner's own. Keys the scaffold no longer defines are dropped
and reported; any other difference from the partner fails. Validation is evaluated in whole
training batches, so a batch must divide the validation sequences: the rules reach shapes
up to batch 256 at context 2,048. Allocate the derived recipe with
``python -m scaffolds.records create``.
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
REFERENCE_TOKENS_PER_UPDATE = 256 * 2048
REFERENCE_UPDATES = 7570
REFERENCE_WARMUP = 455
REFERENCE_CADENCE = {"eval_early_every": 100, "eval_early_until": 1500, "eval_every": 500}
VALIDATION_TOKENS = 19 * REFERENCE_TOKENS_PER_UPDATE
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


def _short(value: float) -> float:
    """Drop float noise from a derived value (0.1 x 3e-4 is 3e-05, not 2.9999999999999997e-05)."""
    return float(f"{value:.12g}")


def shape(layers: int, width: int, value_head_dim: int, intermediate_size: int | None = None) -> dict[str, int]:
    """The study's proportions at a depth and width."""
    if layers < 1 or width < 1 or width % value_head_dim:
        raise ValueError(f"width {width} must be a positive multiple of the value head width {value_head_dim}")
    heads = width // value_head_dim
    if intermediate_size is None:
        intermediate_size = SWIGLU_MULTIPLE * round_half_up(SWIGLU_RATIO * width / SWIGLU_MULTIPLE)
    if intermediate_size < 1:
        raise ValueError("the SwiGLU width must be positive")
    return {"num_layers": layers, "hidden_size": width, "num_heads": heads, "num_kv_heads": heads,
            "intermediate_size": intermediate_size}


def parameter_count(model: dict) -> int:
    from scaffolds.transformer.configs import ModelConfig
    from scaffolds.transformer.models.registry import model_parameter_count

    return model_parameter_count(ModelConfig(**model))


def _reference_model() -> dict:
    return yaml.safe_load(REFERENCE_RECIPE.read_text())["model"]


def reference_parameter_count(layers: int | None = None, width: int | None = None,
                              intermediate_size: int | None = None) -> int:
    """Parameters of the study's normalized-softmax reference recipe, at its own shape or another.

    The count at a depth and width defines that shape's 1x budget for every attention variant.
    """
    reference = _reference_model()
    if layers is None and width is None and intermediate_size is None:
        return parameter_count(reference)
    model_shape = shape(layers, width, reference["value_head_dim"], intermediate_size)
    return parameter_count({**reference, **model_shape})


def tokens_per_update(reference_parameters: int) -> int:
    """Prediction tokens per update that hold the reference's update count at this 1x count."""
    exponent = round(math.log2(reference_parameters / reference_parameter_count()))
    return int(REFERENCE_TOKENS_PER_UPDATE * 2.0 ** exponent)


def budget_of(recipe: dict) -> float:
    """A recipe's token budget, as a multiple of 1x Chinchilla at its shape."""
    model, data, optim = recipe["model"], recipe["data"], recipe["optim"]
    tokens = optim["total_steps"] * data["batch_size"] * (data["sequence_length"] - 1)
    one_x = TOKENS_PER_PARAMETER * reference_parameter_count(
        model["num_layers"], model["hidden_size"], model["intermediate_size"])
    return round(tokens / one_x, 2)


def schedule(reference_parameters: int, batch: int, sequence_length: int, budget: float) -> dict[str, int]:
    if budget <= 0:
        raise ValueError("the token budget must be positive")
    if batch < 1:
        raise ValueError("the batch must be positive")
    prediction_tokens = sequence_length - 1
    validation_sequences, remainder = divmod(VALIDATION_TOKENS, prediction_tokens)
    if remainder or validation_sequences % batch:
        raise ValueError(f"batch {batch} does not divide the {validation_sequences} validation sequences; "
                         "validation runs in whole training batches")
    updates = round_half_up(budget * TOKENS_PER_PARAMETER * reference_parameters / (batch * prediction_tokens))
    return {"total_steps": updates, "warmup_steps": round_half_up(updates * REFERENCE_WARMUP / REFERENCE_UPDATES),
            "eval_batches": validation_sequences // batch,
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
           intermediate_size: int | None = None, budget: float | None = None, batch: int | None = None,
           peak_learning_rate: float | None = None) -> tuple[dict, list[str]]:
    """Return the derived recipe and the partner keys the scaffold no longer defines."""
    recipe, dropped = _defined(copy.deepcopy(partner))
    model, data, optim, train = (recipe[section] for section in ("model", "data", "optim", "train"))
    budget = budget_of(recipe) if budget is None else budget
    old_width = model["hidden_size"]
    same_shape = layers is None and width is None and intermediate_size is None
    if not same_shape:
        model.update(shape(model["num_layers"] if layers is None else layers, old_width if width is None else width,
                           model["value_head_dim"], intermediate_size))
    reference_parameters = reference_parameter_count(model["num_layers"], model["hidden_size"],
                                                     model["intermediate_size"])
    if batch is None:
        if same_shape:
            batch = data["batch_size"]  # a budget change keeps the partner's batch
        else:
            batch, remainder = divmod(tokens_per_update(reference_parameters), data["sequence_length"] - 1)
            if remainder or batch < 1:
                raise ValueError("the shape's tokens per update are not whole sequences at this context")
    if batch < 1 or batch % train["micro_batch_size"]:
        raise ValueError(f"batch {batch} is not a positive multiple of micro_batch_size {train['micro_batch_size']}")
    derived = schedule(reference_parameters, batch, data["sequence_length"], budget)
    if model.get("attention_coefficient_projection_initial_std") and model["hidden_size"] != old_width:
        # Keep the std's multiple of 1/sqrt(width): 1/sqrt(896) becomes 1/sqrt(640).
        gain = _short(model["attention_coefficient_projection_initial_std"] * math.sqrt(old_width))
        model["attention_coefficient_projection_initial_std"] = gain * model["hidden_size"] ** -0.5
    data["batch_size"] = batch
    optim["total_steps"], optim["warmup_steps"] = derived["total_steps"], derived["warmup_steps"]
    if peak_learning_rate is not None:
        if peak_learning_rate <= 0:
            raise ValueError("the peak learning rate must be positive")
        optim["learning_rate"] = peak_learning_rate
    optim["min_learning_rate"] = _short(MIN_LEARNING_RATE_FRACTION * optim["learning_rate"])
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
    parser.add_argument("--budget", type=float, help="multiple of 1x Chinchilla (default: the partner's)")
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
    model = recipe["model"]
    print(f"parameters: {parameter_count(model):,} (1x reference at this shape: "
          f"{reference_parameter_count(model['num_layers'], model['hidden_size'], model['intermediate_size']):,}); "
          f"budget {budget_of(recipe)}x")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
