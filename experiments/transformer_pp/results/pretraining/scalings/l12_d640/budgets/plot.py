"""Regenerate the Transformer++ budget comparison at the 12-layer width-640 shape from its pinned runs.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.pretraining.scalings.l12_d640.budgets.plot

Writes ``summary.csv``, ``budgets.png``/``.pdf`` and the generated tables in ``README.md``: each attention
variant at twice the shape's 1x budget against exp at 2x, its 1x run at this shape and the reference shape's
1x run at almost the same tokens; and at four times the budget against exp at 4x and its 2x run, with the gap
to exp at matched fractions of training.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import yaml

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
FIELDS = (
    "run_id", "role", "shape", "budget", "variant", "degree", "status", "parameter_count", "terminal_step",
    "training_tokens", "terminal_validation_nll", "terminal_validation_perplexity", "best_validation_step",
    "best_validation_nll", "best_validation_perplexity", "best_percent_vs_exp", "best_percent_vs_half_budget",
    "best_percent_vs_reference_shape", "metrics_sha256", "completion_sha256",
)
# Roles name the shape, the budget in multiples of the shape's 1x budget, the variant and its degree.
ROLE = re.compile(r"(?P<shape>l12_d640|reference)_(?P<budget>[124])x_(?P<variant>[a-z_]+?)(?:_m(?P<degree>\d+))?")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# The validated categorical palette's eight slots, one per variant; softmax, fixed Taylor, static, query and the
# two FoX variants keep their shape-pair hues. Exp is the dashed line every gap is measured from.
VARIANTS = {
    "fox_query": ("FoX query", "#008300"),
    "fox_exp": ("FoX exp", "#6250d6"),
    "sum_of_squares": ("sum of squares", "#e87ba4"),
    "softplus_input": ("softplus input", "#e34948"),
    "query": ("query", "#eda100"),
    "fixed_taylor": ("fixed Taylor", "#2a78d6"),
    "exp": ("exp", MUTED),
    "static": ("static", "#1baf7a"),
    "softmax": ("softmax", "#eb6834"),
}
MARKERS = {8: "o", 12: "s", None: "D"}


def parse_role(role: str) -> tuple[str, int, str, int | None]:
    match = ROLE.fullmatch(role)
    if match is None or match["variant"] not in VARIANTS:
        raise ValueError(f"Unlisted role: {role}")
    return match["shape"], int(match["budget"]), match["variant"], int(match["degree"]) if match["degree"] else None


def variant_of(model: dict) -> tuple[str, int | None]:
    """The attention variant and degree a recipe trains, or an error for anything outside this comparison."""
    degree, gate = model.get("attention_degree"), model.get("attention_forget_gate", False)
    if gate != (model.get("positional_encoding", "rope") == "none"):
        raise ValueError("The forget-gate variants here use no positional encoding and the others RoPE")
    if degree is None:
        if model.get("attention_normalization"):
            if gate:
                raise ValueError("Unlisted variant: softmax with a forget gate")
            return "softmax", None
        if model.get("attention_coefficients", "fixed") == "fixed":
            return ("fox_exp" if gate else "exp"), None
        raise ValueError("Unnormalized attention without a polynomial degree must be exp")
    if model.get("attention_normalization") or model.get("attention_output_scaling") != "head_rms_norm":
        raise ValueError("The polynomials here are unnormalized attention with head output RMSNorm")
    transform = model.get("attention_score_transform", "identity")
    parameterization = model.get("attention_coefficient_parameterization", "direct")
    coefficients, conditioning = model.get("attention_coefficients"), model.get("attention_coefficient_conditioning")
    if coefficients == "fixed" and model.get("attention_fixed_coefficients") is None and not gate:
        return "fixed_taylor", degree
    if coefficients == "learned" and conditioning == "static" and not gate:
        return "static", degree
    if (coefficients == "learned" and conditioning == "query" and model.get("attention_coefficient_activation") == "tanh"
            and model.get("attention_coefficient_kernel_weight_decay")):
        if gate:
            return "fox_query", degree
        if transform == "softplus" and parameterization == "direct":
            return "softplus_input", degree
        if transform == "identity" and parameterization == "sum_of_squares":
            return "sum_of_squares", degree
        if transform == "identity" and parameterization == "direct":
            return "query", degree
    raise ValueError(f"Unlisted variant: {coefficients} coefficients, conditioning {conditioning}, "
                     f"transform {transform}, parameterization {parameterization}, forget gate {gate}")


def _name(variant: str, degree: int | None) -> str:
    return VARIANTS[variant][0] + ("" if degree is None else f" m = {degree}")


def _check_recipe(run: str, role: tuple, config: dict, settings: dict) -> None:
    shape, budget, variant, degree = role
    model, data, optim = config["model"], config["data"], config["optim"]
    if {key: model.get(key) for key in settings["shapes"][shape]} != settings["shapes"][shape]:
        raise ValueError(f"{run} does not have the {shape} shape")
    schedule = settings["budgets"][f"{shape}_{budget}x"]
    if (optim["total_steps"], optim["warmup_steps"], data["batch_size"]) != (
            schedule["updates"], schedule["warmup_steps"], schedule["batch_size"]):
        raise ValueError(f"{run} does not train the {shape} {budget}x schedule")
    if (config["train"]["seed"], data["sequence_length"]) != (settings["seed"], settings["sequence_length"]):
        raise ValueError(f"{run} does not use the selection's seed and sequence length")
    if variant_of(model) != (variant, degree):
        raise ValueError(f"{run} trains {_name(*variant_of(model))}, not {_name(variant, degree)}")


def _curve(path: Path) -> dict[int, float]:
    """Validation perplexity by update."""
    points = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if row.get("event", "validation") == "validation" and "validation_perplexity" in row:
            points[row["step"]] = row["validation_perplexity"]
    return points


def _percent(value: float, reference: float) -> float:
    return 100 * (value / reference - 1)


def load_runs(selection: dict | None = None) -> list[dict]:
    selection = selection or yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    settings, identity = selection["settings"], selection["data_identity"]
    verify_pinned_runs(ROOT, selection["runs"])
    rows, index = [], {}
    for entry in selection["runs"]:
        run, role = entry["run_id"], parse_role(entry["role"])
        if role in index:
            raise ValueError(f"Two runs share the role {entry['role']}")
        path = ROOT / "experiments" / entry["study"] / "runs" / run
        config = yaml.safe_load((path / "config.yaml").read_text())
        _check_recipe(run, role, config, settings)
        record = json.loads((path / "record.json").read_text())
        if record["status"] != "complete":
            raise ValueError(f"Selected run is not complete: {run} ({record['status']})")
        if {key: record["data_identity"][key] for key in identity} != identity:
            raise ValueError(f"Data manifests differ from the selection: {run}")
        completion = json.loads((path / "completion.json").read_text())
        best = json.loads((path / "best_validation.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        schedule = settings["budgets"][f"{role[0]}_{role[1]}x"]
        tokens = schedule["updates"] * schedule["batch_size"] * (settings["sequence_length"] - 1)
        if (completion["final_step"], completion["training_tokens"]) != (schedule["updates"], tokens):
            raise ValueError(f"Selected run did not finish its schedule: {run}")
        if best["step"] != schedule["validation_best_step"]:
            raise ValueError(f"{run} is validation-best at update {best['step']}, not {schedule['validation_best_step']}")
        curve = _curve(path / "metrics.jsonl")
        if curve.get(schedule["updates"]) != completion["validation_perplexity"]:
            raise ValueError(f"The validation curve does not end at the terminal evaluation: {run}")
        row = dict(
            run_id=run, role=entry["role"], shape=role[0], budget=role[1], variant=role[2],
            degree="" if role[3] is None else role[3], status=record["status"],
            parameter_count=metadata["parameter_count"], terminal_step=completion["final_step"],
            training_tokens=completion["training_tokens"], terminal_validation_nll=completion["validation_nll"],
            terminal_validation_perplexity=completion["validation_perplexity"], best_validation_step=best["step"],
            best_validation_nll=best["validation_nll"], best_validation_perplexity=best["validation_perplexity"],
            metrics_sha256=entry["sources"]["metrics.jsonl"]["sha256"],
            completion_sha256=entry["sources"]["completion.json"]["sha256"], curve=curve, key=role,
        )
        rows.append(row)
        index[role] = row
    for row in rows:
        shape, budget, variant, degree = row["key"]
        exp = index.get((shape, budget, "exp", None))
        if exp is None:
            raise ValueError(f"No exp run at the {shape} shape and {budget}x")
        half = index.get((shape, budget // 2, variant, degree)) if shape == "l12_d640" and budget > 1 else None
        reference = index.get(("reference", 1, variant, degree)) if (shape, budget) == ("l12_d640", 2) else None
        best = row["best_validation_perplexity"]
        row["best_percent_vs_exp"] = _percent(best, exp["best_validation_perplexity"])
        row["best_percent_vs_half_budget"] = _percent(best, half["best_validation_perplexity"]) if half else ""
        row["best_percent_vs_reference_shape"] = (
            _percent(best, reference["best_validation_perplexity"]) if reference else "")
    return rows


def _signed(value: float, decimals: int = 2, unit: str = "%") -> str:
    text = f"{abs(value):.{decimals}f}"
    return text + unit if float(text) == 0 else ("−" if value < 0 else "+") + text + unit


def _group(rows: list[dict], budget: int) -> list[dict]:
    return [row for row in rows if (row["shape"], row["budget"]) == ("l12_d640", budget)]


def _index(rows: list[dict]) -> dict:
    return {row["key"]: row for row in rows}


def _degree(row: dict) -> str:
    return "–" if row["degree"] == "" else str(row["degree"])


def _table_2x(rows: list[dict], settings: dict) -> str:
    two = sorted(_group(rows, 2), key=lambda row: row["best_validation_perplexity"])
    exp = _index(rows)["l12_d640", 2, "exp", None]
    lines = [f"| Run | Variant | `m` | Validation-best PPL (update) | vs exp | Terminal PPL (update "
             f"{settings['budgets']['l12_d640_2x']['updates']:,}) | vs exp |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in two:
        own = row is exp
        lines.append(
            f"| `{row['run_id']}` | {VARIANTS[row['variant']][0]} | {_degree(row)} | "
            f"{row['best_validation_perplexity']:.3f} ({row['best_validation_step']:,}) | "
            f"{'–' if own else _signed(row['best_percent_vs_exp'])} | {row['terminal_validation_perplexity']:.3f} | "
            f"{'–' if own else _signed(_percent(row['terminal_validation_perplexity'], exp['terminal_validation_perplexity']))} |")
    return "\n".join(lines)


def _table_partners(rows: list[dict]) -> str:
    index = _index(rows)
    lines = ["| Run | Variant | `m` | 2x | 1x at 12-layer width-640 | 2x vs 1x | Reference shape, 1x | "
             "2x vs reference shape |", "| --- | --- | ---: | ---: | --- | ---: | --- | ---: |"]
    for row in _group(rows, 2):
        _, _, variant, degree = row["key"]
        half, reference = index.get(("l12_d640", 1, variant, degree)), index.get(("reference", 1, variant, degree))
        lines.append(
            f"| `{row['run_id']}` | {VARIANTS[variant][0]} | {_degree(row)} | {row['best_validation_perplexity']:.3f} "
            f"({row['terminal_validation_perplexity']:.3f}) | "
            + (f"`{half['run_id']}` {half['best_validation_perplexity']:.3f} | "
               f"{_signed(row['best_percent_vs_half_budget'])} | " if half else "– | – | ")
            + (f"`{reference['run_id']}` {reference['best_validation_perplexity']:.3f} "
               f"({reference['terminal_validation_perplexity']:.3f}) | {_signed(row['best_percent_vs_reference_shape'])} |"
               if reference else "– | – |"))
    return "\n".join(lines)


def _table_4x(rows: list[dict], settings: dict) -> str:
    index = _index(rows)
    four = sorted(_group(rows, 4), key=lambda row: row["best_validation_perplexity"])
    lines = [f"| Run | Variant | `m` | Validation-best PPL (update) | Terminal PPL (update "
             f"{settings['budgets']['l12_d640_4x']['updates']:,}) | vs exp | 2x validation-best | 2x vs exp | "
             "2x to 4x |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in four:
        _, _, variant, degree = row["key"]
        two, own = index["l12_d640", 2, variant, degree], variant == "exp"
        lines.append(
            f"| `{row['run_id']}` | {VARIANTS[variant][0]} | {_degree(row)} | {row['best_validation_perplexity']:.3f} "
            f"({row['best_validation_step']:,}) | {row['terminal_validation_perplexity']:.3f} | "
            f"{'–' if own else _signed(row['best_percent_vs_exp'])} | {two['best_validation_perplexity']:.3f} | "
            f"{'–' if own else _signed(two['best_percent_vs_exp'])} | {_signed(row['best_percent_vs_half_budget'])} |")
    return "\n".join(lines)


def matched_gaps(rows: list[dict], settings: dict) -> dict[tuple, list[tuple[float, float]]]:
    """Each 4x variant's gap to exp, in percent, at the 2x update s and the 4x update 2s."""
    index, gaps = _index(rows), {}
    for row in sorted(_group(rows, 4), key=lambda row: row["best_validation_perplexity"]):
        _, _, variant, degree = row["key"]
        if variant == "exp":
            continue
        pairs = []
        for two_step, four_step in settings["matched_updates"]:
            two, two_exp = index["l12_d640", 2, variant, degree], index["l12_d640", 2, "exp", None]
            pairs.append((_percent(two["curve"][two_step], two_exp["curve"][two_step]),
                          _percent(row["curve"][four_step], index["l12_d640", 4, "exp", None]["curve"][four_step])))
        gaps[variant, degree] = pairs
    return gaps


def _table_matched(rows: list[dict], settings: dict) -> str:
    columns = settings["matched_updates"]
    lines = ["| Variant | `m` | " + " | ".join(f"{two:,} / {four:,}" for two, four in columns) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |"]
    for (variant, degree), pairs in matched_gaps(rows, settings).items():
        lines.append(f"| {VARIANTS[variant][0]} | {'–' if degree is None else degree} | "
                     + " | ".join(f"{_signed(two, unit='')} / {_signed(four, unit='')}" for two, four in pairs) + " |")
    return "\n".join(lines)


def render(rows: list[dict], output: Path = DIRECTORY) -> None:
    index = _index(rows)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "l12-d640-budgets",
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE}):
        figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 6.0), gridspec_kw={"width_ratios": (1.25, 1)})
        figure.subplots_adjust(left=0.135, right=0.86, top=0.80, bottom=0.09, wspace=0.42)
        # Left: every 2x run against exp at 2x, best first.
        two = sorted((row for row in _group(rows, 2) if row["variant"] != "exp"),
                     key=lambda row: row["best_validation_perplexity"])
        for position, row in enumerate(reversed(two)):
            left.plot(row["best_percent_vs_exp"], position, MARKERS[row["key"][3]], color=VARIANTS[row["variant"]][1],
                      markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3)
        left.set_yticks(range(len(two)), [_name(*row["key"][2:]) for row in reversed(two)], fontsize=9, color=INK)
        left.axvline(0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
        left.set_xlabel("Validation-best perplexity against exp at 2× (%)", fontsize=9.5, color=INK)
        left.set_title("All variants at 2×", fontsize=10.5, color=INK, loc="left")
        # Right: the five degree-twelve RoPE polynomials from 2x to 4x.
        four = [row for row in _group(rows, 4) if row["variant"] not in ("exp", "softmax")]
        for row in sorted(four, key=lambda row: row["best_validation_perplexity"]):
            _, _, variant, degree = row["key"]
            gaps = (index["l12_d640", 2, variant, degree]["best_percent_vs_exp"], row["best_percent_vs_exp"])
            right.plot((2, 4), gaps, "-", color=VARIANTS[variant][1], linewidth=1.8, zorder=2)
            right.plot((2, 4), gaps, MARKERS[degree], color=VARIANTS[variant][1], markersize=7.5,
                       markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3)
            right.annotate(f"{VARIANTS[variant][0]}  {_signed(gaps[1])}", (4, gaps[1]), xytext=(9, 0),
                           textcoords="offset points", va="center", fontsize=9, color=INK, annotation_clip=False)
        right.axhline(0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
        right.annotate("exp", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                       textcoords="offset points", va="bottom", fontsize=9, color=MUTED)
        right.set_xlim(1.75, 4.25)
        right.set_xticks((2, 4), ("2×", "4×"), fontsize=9.5, color=INK)
        right.set_ylabel("Validation-best perplexity against exp at the same budget (%)", fontsize=9.5, color=INK)
        right.set_title("Degree-twelve RoPE polynomials, 2× to 4×", fontsize=10.5, color=INK, loc="left")
        for axis in (left, right):
            axis.grid(axis="x" if axis is left else "y", color=GRID, linewidth=0.8)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(MUTED)
            axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
        left.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _signed(value, 0, "")))
        right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _signed(value, 1, "")))
        softmax = (index["l12_d640", 2, "softmax", None], index["l12_d640", 4, "softmax", None])
        figure.text(0.02, 0.965, "Attention variants at the 12-layer width-640 shape as the budget grows", fontsize=13,
                    weight="bold", color=INK, va="top")
        figure.text(0.02, 0.915, "Transformer++, one run (seed 42) per point; 2× is the reference shape's 1× token "
                    "count. Circles are degree 8, squares degree 12, diamonds no degree.\nFoX variants use the Gated "
                    "DeltaNet gate and no positional encoding. Softmax, off the right panel, moves from "
                    f"{_signed(softmax[0]['best_percent_vs_exp'])} to {_signed(softmax[1]['best_percent_vs_exp'])} "
                    "against exp.", fontsize=8.5, color=MUTED, va="top", linespacing=1.45)
        figure.savefig(output / "budgets.png", dpi=180, metadata={"Software": "Matplotlib"})
        figure.savefig(output / "budgets.pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"})
        plt.close(figure)


def update_tables(rows: list[dict], settings: dict, output: Path = DIRECTORY) -> None:
    path, text = output / "README.md", (output / "README.md").read_text()
    blocks = {"2X RESULTS": _table_2x(rows, settings), "2X PARTNERS": _table_partners(rows),
              "4X RESULTS": _table_4x(rows, settings), "4X MATCHED POINTS": _table_matched(rows, settings)}
    for name, table in blocks.items():
        begin, end = f"<!-- BEGIN GENERATED {name} -->", f"<!-- END GENERATED {name} -->"
        prefix, rest = text.split(begin)
        _, suffix = rest.split(end)
        text = prefix + begin + "\n\n" + table + "\n\n" + end + suffix
    path.write_text(text)


def main(output: Path = DIRECTORY) -> None:
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    rows = load_runs(selection)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, output)
    update_tables(rows, selection["settings"], output)


if __name__ == "__main__":
    main()
