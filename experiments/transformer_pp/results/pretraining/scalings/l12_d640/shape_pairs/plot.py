"""Regenerate the Transformer++ shape-pair comparison from its pinned runs.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.pretraining.scalings.l12_d640.shape_pairs.plot

Writes ``summary.csv``, ``shape_pairs.png``/``.pdf``, and the generated tables in ``README.md``:
each attention variant's validation-best and terminal perplexity at the 15-layer width-896
reference shape and at the 12-layer width-640 shape, both at 1x Chinchilla, with its difference
from softmax within each shape.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
FIELDS = (
    "run_id", "role", "variant", "shape", "status", "parameter_count", "terminal_step", "training_tokens",
    "terminal_validation_nll", "terminal_validation_perplexity", "best_validation_step", "best_validation_nll",
    "best_validation_perplexity", "best_percent_vs_softmax", "best_ratio_to_reference_shape",
    "metrics_sha256", "completion_sha256",
)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# The validated categorical palette's eight slots, one per variant. Fixed Taylor, static and
# query keep their degree-scaling hues; marker shapes repeat the hue, and every point carries its
# value because aqua, yellow and magenta sit below 3:1 contrast on the surface.
VARIANTS = {
    "softmax": ("Softmax", "Softmax", "#eb6834", "h"),
    "exp": ("Exp", "Exp", "#e87ba4", "v"),
    "fixed_taylor_m4": ("Fixed Taylor m = 4", "Fixed Taylor", "#2a78d6", "o"),
    "static_m4": ("Static m = 4", "Static", "#1baf7a", "D"),
    "query_m4_w_decay": ("Query m = 4 + W decay", "Query + W decay", "#eda100", "s"),
    "query_m8_w_decay": ("Query m = 8 + W decay", "Query m = 8 + W decay", "#e34948", "X"),
    "fox_no_pe_query_m4_w_decay": ("FoX query m = 4 + W decay", "FoX query + W decay", "#008300", "p"),
    "fox_no_pe_exp": ("FoX exp", "FoX exp", "#6250d6", "^"),
}
# Variants sit side by side at each shape so that no marker hides another.
DODGE = {variant: 0.025 * (index - 3) for index, variant in enumerate(VARIANTS)}
CANONICAL_QK = {"model.qk_norm": True, "model.qk_norm_type": "rms_norm", "model.qk_norm_sharing": "per_layer",
                "model.qk_norm_initial_scale": None, "model.qk_norm_head_scale": True,
                "model.attention_mechanism": "dot_product"}
# The settings on which the variants differ at one shape. Each variant's mechanism is declared here
# (the backend is execution, and the projection std follows the width by the shape rules); every
# other setting must equal softmax's at the same shape.
VARIANT_KEYS = {f"model.{key}" for key in (
    "attention_backend", "attention_degree", "attention_normalization", "attention_output_scaling",
    "attention_coefficients", "attention_coefficient_activation", "attention_coefficient_amplitude_initial",
    "attention_coefficient_amplitude_scale", "attention_coefficient_conditioning",
    "attention_coefficient_kernel_weight_decay", "attention_coefficient_projection_initial_std",
    "attention_coefficient_scale", "attention_coefficient_sharing", "attention_forget_gate", "positional_encoding")}
_FIXED = {"attention_coefficients": "fixed", "attention_coefficient_activation": "identity",
          "attention_coefficient_amplitude_initial": 1.0, "attention_coefficient_amplitude_scale": "none",
          "attention_coefficient_conditioning": "static", "attention_coefficient_kernel_weight_decay": False,
          "attention_coefficient_scale": "none", "attention_coefficient_sharing": "global"}
_QUERY = {"attention_degree": 4, "attention_coefficients": "learned", "attention_coefficient_activation": "tanh",
          "attention_coefficient_amplitude_initial": 0.0, "attention_coefficient_amplitude_scale": "factorial",
          "attention_coefficient_conditioning": "query", "attention_coefficient_kernel_weight_decay": True,
          "attention_coefficient_scale": "factorial", "attention_coefficient_sharing": "per_head"}
_UNNORMALIZED = {"attention_normalization": False, "attention_output_scaling": "head_rms_norm"}
SPEC = {variant: {f"model.{key}": value for key, value in spec.items()} for variant, spec in {
    "softmax": {**_FIXED, "attention_degree": None, "attention_normalization": True,
                "attention_output_scaling": "none", "attention_forget_gate": False, "positional_encoding": "rope"},
    "exp": {**_FIXED, **_UNNORMALIZED, "attention_degree": None, "attention_forget_gate": False,
            "positional_encoding": "rope"},
    "fixed_taylor_m4": {**_FIXED, **_UNNORMALIZED, "attention_degree": 4, "attention_forget_gate": False,
                        "positional_encoding": "rope"},
    "static_m4": {**_FIXED, **_UNNORMALIZED, "attention_degree": 4, "attention_coefficients": "learned",
                  "attention_coefficient_scale": "factorial", "attention_coefficient_sharing": "per_head",
                  "attention_forget_gate": False, "positional_encoding": "rope"},
    "query_m4_w_decay": {**_QUERY, **_UNNORMALIZED, "attention_forget_gate": False, "positional_encoding": "rope"},
    "query_m8_w_decay": {**_QUERY, **_UNNORMALIZED, "attention_degree": 8, "attention_forget_gate": False,
                         "positional_encoding": "rope"},
    "fox_no_pe_query_m4_w_decay": {**_QUERY, **_UNNORMALIZED, "attention_forget_gate": True,
                                   "positional_encoding": "none"},
    "fox_no_pe_exp": {**_FIXED, **_UNNORMALIZED, "attention_degree": None, "attention_forget_gate": True,
                      "positional_encoding": "none"},
}.items()}
SHAPES = {"reference": "15 × 896 (reference)", "l12_d640": "12 × 640"}
SHAPE_KEYS = {"model.num_layers", "model.hidden_size", "model.num_heads", "model.num_kv_heads",
              "model.intermediate_size", "data.batch_size", "optim.total_steps", "optim.warmup_steps",
              "train.eval_batches", "model.attention_coefficient_projection_initial_std"}
# Recipes are compared key by key. Run names and locations differ by design (the data manifests'
# contents are checked by hash), and recipes written before a feature existed lack its key, which
# then means the value below.
LOCATIONS = {"data.train_manifest", "data.eval_manifest", "train.checkpoint_dir", "train.workdir",
             "train.wandb_group", "train.wandb_run_name"}
# Execution, not science: the blocked backend's historical scan and its causal strips (which new
# runs there use) evaluate the same arithmetic, and other backends ignore the setting.
EXECUTION = {"model.attention_blocked_implementation"}
ABSENT = {
    "continuation.new_parameter_learning_rate": None, "continuation.parent_checkpoint": "",
    "continuation.parent_checkpoint_sha256": "", "continuation.parent_config": "",
    "continuation.parent_config_sha256": "", "continuation.parent_data_identity_sha256": "",
    "continuation.parent_step": 0, "continuation.preserve_all_parameters": False,
    "model.attention_coefficient_kernel_weight_decay": False,
    "model.attention_coefficient_key_kernel_weight_decay": False, "model.attention_coefficient_orders": None,
    "model.attention_forget_gate": False, "model.attention_forget_gate_conditioning": "query",
    "model.attention_forget_gate_form": "input", "model.attention_forget_gate_mode": "cumulative",
    "model.attention_forget_gate_rate_activation": "softplus", "model.qk_norm_token_scale": "none",
    "model.qk_norm_token_scale_kernel_weight_decay": False, "optim.query_coefficient_learning_rate_scale": 1.0,
    "model.attention_coefficient_gate": "none", "model.attention_forget_gate_constant_rate": "learned",
    "model.attention_forget_gate_per_order": False, "model.attention_kernel_basis": "polynomial",
    "model.attention_spline_knots": None, "optim.coefficient_release_step": None,
    "optim.coefficient_release_schedule": None, "optim.coefficient_release_end_step": None,
}


def variant_of(model: dict) -> str:
    """The attention variant a recipe trains, or an error for anything outside this comparison."""
    degree, gate = model.get("attention_degree"), model.get("attention_forget_gate", False)
    if gate and model.get("positional_encoding") != "none":
        raise ValueError("The forget-gate variants here use no positional encoding")
    if degree is None:
        if model.get("attention_normalization"):
            if gate:
                raise ValueError("Unlisted variant: softmax with a forget gate")
            return "softmax"
        if model.get("attention_coefficients", "fixed") == "fixed":
            return "fox_no_pe_exp" if gate else "exp"
        raise ValueError("Unnormalized attention without a polynomial degree must be exp")
    if degree not in (4, 8):
        raise ValueError(f"The polynomial variants here have degree 4 (or 8 for query + W decay), not {degree}")
    coefficients = model.get("attention_coefficients")
    if degree == 8:
        if (coefficients == "learned" and model.get("attention_coefficient_conditioning") == "query"
                and model.get("attention_coefficient_activation") == "tanh"
                and model.get("attention_coefficient_kernel_weight_decay") and not gate):
            return "query_m8_w_decay"
        raise ValueError("Degree 8 appears here only as query + W decay without a forget gate")
    if coefficients == "fixed" and model.get("attention_fixed_coefficients") is None and not gate:
        return "fixed_taylor_m4"
    if coefficients == "learned":
        conditioning = model.get("attention_coefficient_conditioning")
        if conditioning == "static" and model.get("attention_coefficient_sharing") == "per_head" and not gate:
            return "static_m4"
        if (conditioning == "query" and model.get("attention_coefficient_activation") == "tanh"
                and model.get("attention_coefficient_kernel_weight_decay")):
            return "fox_no_pe_query_m4_w_decay" if gate else "query_m4_w_decay"
    raise ValueError(f"Unlisted variant: {coefficients} coefficients, "
                     f"conditioning {model.get('attention_coefficient_conditioning')}, forget gate {gate}")


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _recipe(config: dict) -> dict:
    """A recipe's settings by dotted key, with absent keys filled and locations and execution dropped."""
    recipe = {**ABSENT, **_flatten(config)}
    return {key: value for key, value in recipe.items() if key not in LOCATIONS | EXECUTION}


def _check_shape(entry: dict, config: dict, settings: dict) -> None:
    """The recipe has its declared shape's proportions, batch, schedule and the shared validation set."""
    shape, run = settings["shapes"][entry["shape"]], entry["run_id"]
    model, data, optim, train = (config[section] for section in ("model", "data", "optim", "train"))
    actual = {"num_layers": model["num_layers"], "hidden_size": model["hidden_size"], "num_heads": model["num_heads"],
              "intermediate_size": model["intermediate_size"], "batch_size": data["batch_size"],
              "updates": optim["total_steps"], "warmup_steps": optim["warmup_steps"]}
    expected = {key: shape[key] for key in actual}
    if actual != expected or model["num_kv_heads"] != model["num_heads"]:
        raise ValueError(f"{run} does not have the {entry['shape']} shape: {actual}")
    if train["eval_batches"] * data["batch_size"] != settings["validation_sequences"]:
        raise ValueError(f"{run} does not evaluate the shared {settings['validation_sequences']} validation sequences")
    shared = {"qk_head_dim": settings["qk_head_dim"], "value_head_dim": settings["value_head_dim"],
              "initialization": settings["initialization"]}
    if any(model.get(key) != value for key, value in shared.items()) or train["seed"] != settings["seed"]:
        raise ValueError(f"{run} differs from the shared head widths, initialization, or seed")
    if variant_of(model) != entry["variant"]:
        raise ValueError(f"{run} trains {variant_of(model)}, but the selection says {entry['variant']}")


def check_recipes(entries: list[dict], configs: dict, settings: dict) -> None:
    """Each run has its declared shape and variant, and each pair differs only by the shape rules."""
    pairs: dict[str, dict[str, str]] = {}
    for entry in entries:
        _check_shape(entry, configs[entry["run_id"]], settings)
        if (entry["role"] == "control") != (entry["shape"] == "reference"):
            raise ValueError(f"Reference-shape runs are controls and 12-layer runs primary: {entry['run_id']}")
        if pairs.setdefault(entry["variant"], {}).setdefault(entry["shape"], entry["run_id"]) != entry["run_id"]:
            raise ValueError(f"Two runs for {entry['variant']} at {entry['shape']}")
    if set(pairs) != set(VARIANTS) or any(set(shapes) != set(SHAPES) for shapes in pairs.values()):
        raise ValueError("The selection needs exactly one run of every variant at each shape")
    missing = object()
    for variant, shapes in pairs.items():
        control, primary = (_recipe(configs[shapes[shape]]) for shape in ("reference", "l12_d640"))
        differing = sorted(key for key in set(control) | set(primary) if key not in SHAPE_KEYS
                           and control.get(key, missing) != primary.get(key, missing))
        if differing:
            raise ValueError(f"{shapes['l12_d640']} differs from its partner {shapes['reference']} beyond the "
                             f"shape rules: {', '.join(differing)}")
        # A query-coefficient projection keeps its multiple of 1/sqrt(width).
        std = "model.attention_coefficient_projection_initial_std"
        gains = [recipe[std] * recipe["model.hidden_size"] ** 0.5 for recipe in (control, primary)]
        if abs(gains[0] - gains[1]) > 1e-9:
            raise ValueError(f"{shapes['l12_d640']} rescales its projection std off 1/sqrt(width)")
    # Within each shape every variant is softmax's recipe except for its declared mechanism.
    for shape in SHAPES:
        softmax = _recipe(configs[pairs["softmax"][shape]])
        for variant, shapes in pairs.items():
            run, recipe = shapes[shape], _recipe(configs[shapes[shape]])
            wrong = {key: recipe.get(key, missing) for key, value in {**CANONICAL_QK, **SPEC[variant]}.items()
                     if recipe.get(key, missing) != value}
            if wrong:
                raise ValueError(f"{run} is not the declared {variant}: {wrong}")
            differing = sorted(key for key in set(softmax) | set(recipe) if key not in VARIANT_KEYS | SHAPE_KEYS
                               and softmax.get(key, missing) != recipe.get(key, missing))
            if differing:
                raise ValueError(f"{run} differs from softmax at {shape} beyond its variant: {', '.join(differing)}")
    # Each shape's 1x budget: 20 prediction tokens per parameter of its normalized-softmax model.
    for name, shape in settings["shapes"].items():
        tokens = 20 * shape["reference_parameters"]
        per_update = shape["batch_size"] * (configs[pairs["softmax"][name]]["data"]["sequence_length"] - 1)
        if shape["updates"] != int(tokens / per_update + 0.5):
            raise ValueError(f"The {name} shape's {shape['updates']} updates are not its 1x budget")


def load_runs(selection: dict | None = None) -> list[dict]:
    selection = selection or yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    settings, identity = selection["settings"], selection["data_identity"]
    verify_pinned_runs(ROOT, selection["runs"])
    paths = {entry["run_id"]: ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
             for entry in selection["runs"]}
    configs = {run: yaml.safe_load((path / "config.yaml").read_text()) for run, path in paths.items()}
    check_recipes(selection["runs"], configs, settings)
    rows = []
    for entry in selection["runs"]:
        path, shape = paths[entry["run_id"]], settings["shapes"][entry["shape"]]
        record = json.loads((path / "record.json").read_text())
        if record["status"] != "complete":
            raise ValueError(f"Selected run is not complete: {entry['run_id']} ({record['status']})")
        if {key: record["data_identity"][key] for key in identity} != identity:
            raise ValueError(f"Data manifests differ from the selection: {entry['run_id']}")
        completion = json.loads((path / "completion.json").read_text())
        best = json.loads((path / "best_validation.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        tokens = shape["updates"] * shape["batch_size"] * (configs[entry["run_id"]]["data"]["sequence_length"] - 1)
        if (completion["final_step"], completion["training_tokens"]) != (shape["updates"], tokens):
            raise ValueError(f"Selected run did not finish its shape's schedule: {entry['run_id']}")
        if entry["variant"] == "softmax" and metadata["parameter_count"] != shape["reference_parameters"]:
            raise ValueError(f"The {entry['shape']} softmax model does not have the shape's 1x parameter count")
        rows.append(dict(
            run_id=entry["run_id"], role=entry["role"], variant=entry["variant"], shape=entry["shape"],
            status=record["status"], parameter_count=metadata["parameter_count"],
            terminal_step=completion["final_step"], training_tokens=completion["training_tokens"],
            terminal_validation_nll=completion["validation_nll"],
            terminal_validation_perplexity=completion["validation_perplexity"], best_validation_step=best["step"],
            best_validation_nll=best["validation_nll"], best_validation_perplexity=best["validation_perplexity"],
            metrics_sha256=entry["sources"]["metrics.jsonl"]["sha256"],
            completion_sha256=entry["sources"]["completion.json"]["sha256"],
        ))
    cells = {(row["variant"], row["shape"]): row for row in rows}
    for row in rows:
        best = row["best_validation_perplexity"]
        row["best_percent_vs_softmax"] = 100 * (best / cells["softmax", row["shape"]]["best_validation_perplexity"] - 1)
        row["best_ratio_to_reference_shape"] = (
            best / cells[row["variant"], "reference"]["best_validation_perplexity"] if row["shape"] == "l12_d640" else "")
    order, shapes = list(VARIANTS), list(SHAPES)
    rows.sort(key=lambda row: (order.index(row["variant"]), shapes.index(row["shape"])))
    return rows


def _spread(levels: list[float], gap: float) -> list[float]:
    """Nudge label positions apart by at least `gap`, keeping their mean; returns input order."""
    order = sorted(range(len(levels)), key=lambda index: levels[index])
    placed = [levels[index] for index in order]
    for _ in range(200):
        moved = False
        for index in range(1, len(placed)):
            if placed[index] - placed[index - 1] < gap:
                shift = (gap - (placed[index] - placed[index - 1])) / 2
                placed[index - 1] -= shift
                placed[index] += shift
                moved = True
        if not moved:
            break
    result = [0.0] * len(levels)
    for rank, index in enumerate(order):
        result[index] = placed[rank]
    return result


def _slopes(figure, axis, rows: list[dict], value, label, *, reference: float | None = None) -> None:
    """One line per variant from the reference shape (left) to the 12-layer shape (right), labeled at both ends."""
    cells = {(row["variant"], row["shape"]): value(row) for row in rows}
    variants = [variant for variant in VARIANTS if reference is None or variant != "softmax"]
    if reference is not None:
        axis.axhline(reference, color=MUTED, linewidth=1.0, alpha=0.85, zorder=1)
    for variant in variants:
        _, _, color, marker = VARIANTS[variant]
        xs = [DODGE[variant], 1 + DODGE[variant]]
        ys = [cells[variant, "reference"], cells[variant, "l12_d640"]]
        axis.plot(xs, ys, color=color, linewidth=1.8, zorder=2, solid_capstyle="round")
        axis.plot(xs, ys, linestyle="none", marker=marker, markersize=6.5, color=color,
                  markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3)
    values = [cells[variant, shape] for variant in variants for shape in SHAPES] + (
        [reference] if reference is not None else [])
    lower, upper = min(values), max(values)
    pad = 0.08 * (upper - lower)
    axis.set_ylim(lower - pad, upper + pad)
    axis.set_xlim(-0.85, 2.3)
    height_points = axis.get_window_extent().height * 72 / figure.dpi
    gap = 11.5 * (axis.get_ylim()[1] - axis.get_ylim()[0]) / height_points
    # Value chips sit beside each shape's markers on an opaque background, so the reference line
    # and the other variants' lines pass behind them; the leader starts at the chip's near edge.
    for shape, side, x_text in (("reference", "right", min(DODGE.values()) - 0.14),
                                ("l12_d640", "left", 1 + max(DODGE.values()) + 0.14)):
        positions = _spread([cells[variant, shape] for variant in variants], gap)
        for variant, placed in zip(variants, positions):
            _, name, color, _ = VARIANTS[variant]
            text = label(cells[variant, shape]).replace("-", "\u2212") + (f"  {name}" if side == "left" else "")
            x = (0 if shape == "reference" else 1) + DODGE[variant]
            axis.annotate(text, (x, cells[variant, shape]), xytext=(x_text, placed), textcoords="data",
                          ha=side, va="center", fontsize=7.5, color=INK, zorder=4,
                          bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": GRID,
                                "linewidth": 0.6},
                          arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8, "shrinkA": 0,
                                      "shrinkB": 3, "relpos": (1, 0.5) if side == "right" else (0, 0.5)})
    if reference is not None:
        axis.annotate("Softmax", (0.0, reference), xycoords=("axes fraction", "data"), xytext=(4, 4),
                      textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    axis.set_xticks([0, 1], [SHAPES["reference"], SHAPES["l12_d640"]], fontsize=8.5, color=INK)
    axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)


def render(rows: list[dict], output: Path = DIRECTORY) -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "model-scale",
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE}):
        figure, (absolute, relative) = plt.subplots(1, 2, figsize=(12.0, 5.9))
        figure.subplots_adjust(left=0.07, right=0.985, top=0.77, bottom=0.08, wspace=0.24)
        _slopes(figure, absolute, rows, lambda row: row["best_validation_perplexity"], lambda value: f"{value:.2f}")
        absolute.set_ylabel("Validation-best perplexity (lower is better)", fontsize=9.5, color=INK)
        _slopes(figure, relative, rows, lambda row: row["best_percent_vs_softmax"], lambda value: f"{value:+.1f}%",
                reference=0.0)
        relative.set_ylabel("Difference from softmax at the same shape (%)", fontsize=9.5, color=INK)
        figure.text(0.07, 0.955, "Attention variants at two model shapes", fontsize=13, weight="bold",
                    color=INK, va="top")
        sizes = {row["shape"]: row["parameter_count"] for row in rows if row["variant"] == "softmax"}
        figure.text(0.07, 0.905, f"Transformer++ at 1× Chinchilla for each shape ({sizes['reference'] / 1e6:.1f}M "
                    f"and {sizes['l12_d640'] / 1e6:.1f}M parameters), validation-best perplexity, one run (seed 42) "
                    "per point.\nPolynomial variants have degree m = 4 unless labeled m = 8; FoX variants use the "
                    "Gated DeltaNet gate and no positional encoding.\nThe 12-layer runs halve the batch, hold the "
                    "update count within 0.3%, and keep the validation set and everything else of their partner.",
                    fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
        figure.savefig(output / "shape_pairs.png", dpi=180, metadata={"Software": "Matplotlib"})
        figure.savefig(output / "shape_pairs.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                            "Creator": "Matplotlib"})
        plt.close(figure)


def _tables(rows: list[dict]) -> str:
    cells = {(row["variant"], row["shape"]): row for row in rows}
    lines = ["**Validation-best perplexity**", "",
             "| Variant | 15 × 896 | 12 × 640 | 12 × 640 / 15 × 896 | vs softmax, 15 × 896 | vs softmax, 12 × 640 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    ranked = sorted(VARIANTS, key=lambda variant: cells[variant, "l12_d640"]["best_validation_perplexity"])
    for variant in ranked:
        reference, small = cells[variant, "reference"], cells[variant, "l12_d640"]
        gaps = ["–" if variant == "softmax" else f"{row['best_percent_vs_softmax']:+.2f}%" for row in (reference, small)]
        lines.append(f"| {VARIANTS[variant][0]} | {reference['best_validation_perplexity']:.3f} | "
                     f"{small['best_validation_perplexity']:.3f} | {small['best_ratio_to_reference_shape']:.3f} | "
                     f"{gaps[0]} | {gaps[1]} |")
    steps = {shape: sorted({row["best_validation_step"] for row in rows if row["shape"] == shape}) for shape in SHAPES}
    lines += ["", "Validation-best updates: " + "; ".join(
        f"{SHAPES[shape]} {', '.join(f'{step:,}' for step in found)}" for shape, found in steps.items()) + ".", "",
        "**Terminal perplexity and parameters**", "",
        "| Variant | 15 × 896 run | Terminal | Parameters | 12 × 640 run | Terminal | Parameters |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |"]
    for variant in ranked:
        reference, small = cells[variant, "reference"], cells[variant, "l12_d640"]
        lines.append(f"| {VARIANTS[variant][0]} | `{reference['run_id']}` | "
                     f"{reference['terminal_validation_perplexity']:.3f} | {reference['parameter_count']:,} | "
                     f"`{small['run_id']}` | {small['terminal_validation_perplexity']:.3f} | "
                     f"{small['parameter_count']:,} |")
    return "\n".join(lines)


def update_tables(rows: list[dict], output: Path = DIRECTORY) -> None:
    path = output / "README.md"
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    prefix, rest = path.read_text().split(begin)
    _, suffix = rest.split(end)
    path.write_text(prefix + begin + "\n\n" + _tables(rows) + "\n\n" + end + suffix)


def main(output: Path = DIRECTORY) -> None:
    rows = load_runs()
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, output)
    update_tables(rows, output)


if __name__ == "__main__":
    main()
