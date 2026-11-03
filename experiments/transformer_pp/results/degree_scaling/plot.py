"""Regenerate the Transformer++ degree-scaling comparison from its pinned runs.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.degree_scaling.plot

Writes ``summary.csv``, ``degree_scaling.png``/``.pdf``, and the generated tables in
``README.md``: terminal validation perplexity (update 7,570) against the polynomial
degree for each coefficient family, with validation-best values reported separately
and softmax as a reference level.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.transforms import offset_copy
from numpy.polynomial import hermite_e

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]
FIELDS = (
    "run_id", "role", "family", "degree", "status", "terminal_step", "training_tokens",
    "terminal_validation_nll", "terminal_validation_perplexity", "change_from_previous_degree_percent",
    "percent_above_softmax", "best_validation_step", "best_validation_nll", "best_validation_perplexity",
    "metrics_sha256", "completion_sha256",
)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Families take the first four slots of the validated categorical palette, Taylor, Hermite and
# learned in the attention_scaling degree chart's order; the marker shape repeats the hue as a
# second cue, and every point carries its value because slots 3 and 4 sit below 3:1 contrast.
FAMILIES = {"taylor": ("Fixed Taylor", "#2a78d6", "o"), "hermite": ("Fixed Hermite", "#eb6834", "^"),
            "static": ("Learned static", "#1baf7a", "D"), "query_w_decay": ("Query, W decay", "#eda100", "s")}
REFERENCE = "softmax"
# Families sit side by side at each degree so that no marker hides another.
DODGE = {"taylor": -0.09, "hermite": -0.03, "static": 0.03, "query_w_decay": 0.09}
CANONICAL_QK = {"qk_norm": True, "qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
                "qk_norm_initial_scale": None, "qk_norm_head_scale": True}
# Recipes are compared key by key. Run names and locations differ by design (the data
# manifests' contents are checked by hash), and recipes written before a feature existed
# lack its key, which then means the value below.
LOCATIONS = {"data.train_manifest", "data.eval_manifest", "train.checkpoint_dir", "train.workdir",
             "train.wandb_group", "train.wandb_run_name"}
ABSENT = {
    "continuation.new_parameter_learning_rate": None, "continuation.parent_checkpoint": "",
    "continuation.parent_checkpoint_sha256": "", "continuation.parent_config": "",
    "continuation.parent_config_sha256": "", "continuation.parent_data_identity_sha256": "",
    "continuation.parent_step": 0, "model.attention_forget_gate": False,
    "model.attention_coefficient_kernel_weight_decay": False, "model.qk_norm_token_scale": "none",
    "model.qk_norm_token_scale_kernel_weight_decay": False, "optim.query_coefficient_learning_rate_scale": 1.0,
}
# Within a family only the degree changes, with the Hermite coefficients it determines;
# between families the coefficient parameterization changes too, and softmax also differs
# in its kernel, normalization, and output scaling.
DEGREE_KEYS = {"model.attention_degree", "model.attention_fixed_coefficients"}
FAMILY_KEYS = DEGREE_KEYS | {f"model.attention_{key}" for key in (
    "coefficients", "coefficient_activation", "coefficient_amplitude_initial", "coefficient_amplitude_scale",
    "coefficient_conditioning", "coefficient_kernel_weight_decay", "coefficient_projection_initial_std",
    "coefficient_scale", "coefficient_sharing")}
REFERENCE_KEYS = FAMILY_KEYS | {"model.attention_backend", "model.attention_normalization",
                                "model.attention_output_scaling"}


def hermite_coefficients(degree: int) -> list[float]:
    """Monomial coefficients of exp(1/2) sum_(r<=m) He_r(s)/r!, the degree-m Hermite truncation of exp(s)."""
    return list(math.exp(0.5) * hermite_e.herme2poly([1 / math.factorial(r) for r in range(degree + 1)]))


def family_of(model: dict) -> str:
    """The coefficient family a recipe trains, or an error for anything outside this comparison."""
    degree = model.get("attention_degree")
    if degree is None:
        if model.get("attention_normalization"):
            return REFERENCE
        raise ValueError("Unnormalized attention without a polynomial degree is not in this comparison")
    coefficients = model.get("attention_coefficients")
    if coefficients == "fixed":
        fixed = model.get("attention_fixed_coefficients")
        if fixed is None:
            return "taylor"
        expected = hermite_coefficients(degree)
        if len(fixed) == len(expected) and np.allclose(fixed, expected, rtol=1e-12, atol=0.0):
            return "hermite"
        raise ValueError(f"Fixed coefficients are neither Taylor nor the Hermite truncation: {fixed}")
    if coefficients == "learned":
        conditioning = model.get("attention_coefficient_conditioning")
        if conditioning == "static" and model.get("attention_coefficient_sharing") == "per_head":
            return "static"
        if (conditioning == "query" and model.get("attention_coefficient_activation") == "tanh"
                and model.get("attention_coefficient_kernel_weight_decay")):
            return "query_w_decay"
    raise ValueError(f"Unlisted coefficient family: {coefficients}, "
                     f"conditioning {model.get('attention_coefficient_conditioning')}")


def _check_recipe(entry: dict, config: dict, settings: dict) -> None:
    """The recipe must be the selection's shared setup, declared family and degree."""
    model, optim, train = config["model"], config["optim"], config["train"]
    run = entry["run_id"]
    shared = {"qk_head_dim": settings["qk_head_dim"], "initialization": settings["initialization"], **CANONICAL_QK}
    for key, value in shared.items():
        if model.get(key) != value:
            raise ValueError(f"{run} {key}: expected {value!r}, got {model.get(key)!r}")
    if (optim["total_steps"], optim["warmup_steps"], train["seed"]) != (
            settings["terminal_step"], settings["warmup_steps"], settings["seed"]):
        raise ValueError(f"Updates, warmup, or seed differs from the selection settings: {run}")
    family = family_of(model)
    if family != entry["family"]:
        raise ValueError(f"{run} trains {family}, but the selection says {entry['family']}")
    if (entry["role"] == "reference") != (family == REFERENCE):
        raise ValueError(f"Polynomial families are primary and softmax is the reference: {run}")
    if family != REFERENCE:
        if model["attention_degree"] != entry["degree"]:
            raise ValueError(f"{run} has degree {model['attention_degree']}, but the selection says {entry['degree']}")
        if model.get("attention_normalization") or model.get("attention_output_scaling") != "head_rms_norm":
            raise ValueError(f"Polynomial runs are unnormalized with per-head output RMSNorm: {run}")


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _recipe(config: dict) -> dict:
    """A recipe's settings by dotted key, with absent keys filled and locations dropped."""
    recipe = {**ABSENT, **_flatten(config)}
    return {key: value for key, value in recipe.items() if key not in LOCATIONS}


def _compare(recipes: dict, base: str, run: str, allowed: set) -> None:
    missing = object()
    differing = sorted(key for key in set(recipes[base]) | set(recipes[run])
                       if key not in allowed and recipes[base].get(key, missing) != recipes[run].get(key, missing))
    if differing:
        raise ValueError(f"{run} differs from {base} beyond the comparison: {', '.join(differing)}")


def check_recipes(entries: list[dict], configs: dict, settings: dict) -> None:
    """Each recipe is its declared cell in the shared setup, and recipes differ only where the comparison does."""
    for entry in entries:
        _check_recipe(entry, configs[entry["run_id"]], settings)
    recipes = {run: _recipe(config) for run, config in configs.items()}
    for key, value in ABSENT.items():
        if key not in FAMILY_KEYS and any(recipe[key] != value for recipe in recipes.values()):
            raise ValueError(f"Every run must have {key} = {value!r}")
    primary = [entry for entry in entries if entry["role"] == "primary"]
    anchors = {}
    for entry in primary:
        anchor = anchors.setdefault(entry["family"], entry["run_id"])
        _compare(recipes, anchor, entry["run_id"], DEGREE_KEYS)
    first = primary[0]["run_id"]
    for anchor in anchors.values():
        _compare(recipes, first, anchor, FAMILY_KEYS)
    for entry in entries:
        if entry["role"] == "reference":
            _compare(recipes, first, entry["run_id"], REFERENCE_KEYS)


def load_runs(selection: dict | None = None) -> list[dict]:
    selection = selection or yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    settings, identity = selection["settings"], selection["data_identity"]
    verify_pinned_runs(ROOT, selection["runs"])
    paths = {entry["run_id"]: ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
             for entry in selection["runs"]}
    check_recipes(selection["runs"], {run: yaml.safe_load((path / "config.yaml").read_text())
                                      for run, path in paths.items()}, settings)
    rows = []
    for entry in selection["runs"]:
        path = paths[entry["run_id"]]
        record = json.loads((path / "record.json").read_text())
        if record["status"] != "complete":
            raise ValueError(f"Selected run is not complete: {entry['run_id']} ({record['status']})")
        manifests = {key: record["data_identity"][key] for key in identity}
        if manifests != identity:
            raise ValueError(f"Data manifests differ from the selection: {entry['run_id']}")
        completion = json.loads((path / "completion.json").read_text())
        best = json.loads((path / "best_validation.json").read_text())
        if (completion["final_step"], completion["training_tokens"]) != (
                settings["terminal_step"], settings["training_tokens"]):
            raise ValueError(f"Selected run did not finish the schedule: {entry['run_id']}")
        rows.append(dict(
            run_id=entry["run_id"], role=entry["role"], family=entry["family"], degree=entry.get("degree", ""),
            status=record["status"], terminal_step=completion["final_step"],
            training_tokens=completion["training_tokens"], terminal_validation_nll=completion["validation_nll"],
            terminal_validation_perplexity=completion["validation_perplexity"], best_validation_step=best["step"],
            best_validation_nll=best["validation_nll"], best_validation_perplexity=best["validation_perplexity"],
            metrics_sha256=entry["sources"]["metrics.jsonl"]["sha256"],
            completion_sha256=entry["sources"]["completion.json"]["sha256"],
        ))
    references = [row for row in rows if row["family"] == REFERENCE]
    if len(references) != 1:
        raise ValueError("The selection needs exactly one softmax reference")
    cells = {(row["family"], row["degree"]): row for row in rows if row["role"] == "primary"}
    expected = {(family, degree) for family in FAMILIES for degree in settings["degrees"]}
    if len(cells) != len(rows) - 1 or set(cells) != expected:
        raise ValueError("The selection needs exactly one run for every family and degree")
    softmax = references[0]["terminal_validation_perplexity"]
    for row in rows:
        previous = cells.get((row["family"], row["degree"] - 1)) if row["role"] == "primary" else None
        ppl = row["terminal_validation_perplexity"]
        row["change_from_previous_degree_percent"] = (
            100 * (ppl / previous["terminal_validation_perplexity"] - 1) if previous else "")
        row["percent_above_softmax"] = 100 * (ppl / softmax - 1) if row["role"] == "primary" else ""
    order = list(FAMILIES) + [REFERENCE]
    rows.sort(key=lambda row: (order.index(row["family"]), row["degree"] if row["role"] == "primary" else 0))
    return rows


def _spread(levels: list[float], gap: float) -> list[float]:
    """Nudge label positions apart by at least `gap`, keeping their mean; returns input order."""
    order = sorted(range(len(levels)), key=lambda index: levels[index])
    placed = [levels[index] for index in order]
    for _ in range(100):
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


def render(rows: list[dict], output: Path = DIRECTORY) -> None:
    primary = [row for row in rows if row["role"] == "primary"]
    (softmax,) = [row for row in rows if row["role"] == "reference"]
    degrees = sorted({row["degree"] for row in primary})
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "degree-scaling",
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE}):
        figure, axis = plt.subplots(figsize=(8.0, 5.6))
        figure.subplots_adjust(left=0.1, right=0.84, top=0.8, bottom=0.11)
        level = softmax["terminal_validation_perplexity"]
        axis.axhline(level, color=MUTED, linewidth=1.0, alpha=0.85, zorder=1)
        by_degree: dict[int, list[tuple[float, str]]] = {}
        handles = []
        for family, (name, color, marker) in FAMILIES.items():
            points = sorted((row["degree"], row["terminal_validation_perplexity"])
                            for row in primary if row["family"] == family)
            xs = [degree + DODGE[family] for degree, _ in points]
            ys = [value for _, value in points]
            axis.plot(xs, ys, color=color, linewidth=1.8, zorder=2, solid_capstyle="round")
            axis.plot(xs, ys, linestyle="none", marker=marker, markersize=6.5, color=color,
                      markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3)
            handles.append(plt.Line2D([], [], color=color, marker=marker, markersize=6.5, linewidth=1.8,
                                      markeredgecolor=SURFACE, markeredgewidth=0.9, label=name))
            for degree, value in points:
                by_degree.setdefault(degree, []).append((value, family))
        values = [row["terminal_validation_perplexity"] for row in primary] + [level]
        lower, upper = min(values), max(values)
        pad = 0.05 * (upper - lower)
        axis.set_xlim(min(degrees) - 0.35, max(degrees) + 0.6)
        axis.set_ylim(lower - 1.5 * pad, upper + 1.5 * pad)
        height_points = axis.get_window_extent().height * 72 / figure.dpi
        gap = 10.0 * (axis.get_ylim()[1] - axis.get_ylim()[0]) / height_points
        # Each value is a chip right of the degree's markers: a small swatch in the family's
        # color and shape, then the number, on an opaque background so it reads as a label.
        swatch = offset_copy(axis.transData, fig=figure, x=14, y=0, units="points")
        right = max(DODGE.values())
        for x, labels in sorted(by_degree.items()):
            positions = _spread([value for value, _ in labels], 0.95 * gap)
            # Labels that would sit on the softmax level move up together to clear it.
            if level - 0.6 * gap < min(positions) < level + 0.6 * gap:
                positions = [position + level + 0.6 * gap - min(positions) for position in positions]
            for (value, family), placed in zip(labels, positions):
                _, color, marker = FAMILIES[family]
                axis.annotate(f"    {value:.2f}", (x + right, placed), xytext=(9, 0), textcoords="offset points",
                              ha="left", va="center", fontsize=7.5, color=INK, zorder=4,
                              bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": GRID,
                                    "linewidth": 0.6})
                axis.plot([x + right], [placed], marker=marker, markersize=4, color=color, linestyle="none",
                          transform=swatch, clip_on=False, zorder=5)
        axis.annotate(f"Softmax {level:.2f}", (1.0, level), xycoords=("axes fraction", "data"), xytext=(6, 0),
                      textcoords="offset points", ha="left", va="center", fontsize=8, color=MUTED,
                      annotation_clip=False)
        axis.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK,
                    handlelength=2.4, borderaxespad=0.6)
        axis.set_xticks(degrees, [f"m = {degree}" for degree in degrees], fontsize=9, color=INK)
        axis.set_xlabel("Polynomial degree", fontsize=9.5, color=INK)
        axis.set_ylabel("Terminal validation perplexity (lower is better)", fontsize=9.5, color=INK)
        axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5, length=0)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
        step = primary[0]["terminal_step"]
        figure.text(0.1, 0.955, "Degree scaling of unnormalized polynomial attention", fontsize=13,
                    weight="bold", color=INK, va="top")
        figure.text(0.1, 0.905, f"Transformer++ at 1× Chinchilla, validation perplexity at update {step:,}, "
                    "one run (seed 42) per point.\nEvery polynomial run has per-head output RMSNorm and "
                    "canonical QK RMSNorm; softmax is the normalized reference.", fontsize=8.5, color=MUTED,
                    va="top", linespacing=1.35)
        figure.savefig(output / "degree_scaling.png", dpi=180, metadata={"Software": "Matplotlib"})
        figure.savefig(output / "degree_scaling.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                                "Creator": "Matplotlib"})
        plt.close(figure)


def _tables(rows: list[dict]) -> str:
    primary = [row for row in rows if row["role"] == "primary"]
    (softmax,) = [row for row in rows if row["role"] == "reference"]
    degrees = sorted({row["degree"] for row in primary})
    cells = {(row["family"], row["degree"]): row for row in primary}
    best_steps = {row["best_validation_step"] for row in primary}
    common = best_steps.pop() if len(best_steps) == 1 else None
    header = "| Family | " + " | ".join(f"m = {degree}" for degree in degrees) + " |"
    rule = "| --- |" + " ---: |" * len(degrees)

    def table(title: str, cell) -> list[str]:
        lines = [f"**{title}**", "", header, rule]
        for family, (name, _, _) in FAMILIES.items():
            lines.append(f"| {name} | " + " | ".join(cell(cells[family, degree]) for degree in degrees) + " |")
        return lines + [""]

    lowest = {degree: min(cells[family, degree]["terminal_validation_perplexity"] for family in FAMILIES)
              for degree in degrees}

    def terminal(row):
        text = f"{row['terminal_validation_perplexity']:.3f}"
        return f"**{text}**" if row["terminal_validation_perplexity"] == lowest[row["degree"]] else text

    def best(row):
        step = "" if common is not None else f" @{row['best_validation_step']:,}"
        return f"{row['best_validation_perplexity']:.3f}{step}"

    def change(row):
        value = row["change_from_previous_degree_percent"]
        return "–" if value == "" else f"{value:+.1f}%"

    step = primary[0]["terminal_step"]
    best_title = (f"Validation-best perplexity (update {common:,} in every run)" if common is not None
                  else "Validation-best perplexity (update shown)")
    lines = table(f"Terminal validation perplexity (update {step:,}); lowest per degree in bold", terminal)
    lines += table(best_title, best)
    lines += table("Change in terminal perplexity from the previous degree", change)
    lines.append(f"Softmax reference (`{softmax['run_id']}`): terminal "
                 f"{softmax['terminal_validation_perplexity']:.3f}, validation-best "
                 f"{softmax['best_validation_perplexity']:.3f} (update {softmax['best_validation_step']:,}).")
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
