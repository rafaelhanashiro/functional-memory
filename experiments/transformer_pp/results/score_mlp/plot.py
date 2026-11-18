"""Regenerate the Transformer++ score-MLP comparison from its pinned runs.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.score_mlp.plot

Writes ``summary.csv``, ``score_mlp.png``/``.pdf``, and the generated table in ``README.md``: the
conventional score MLP at widths J = 4 and 8 against exp, query m = 4 and fixed Taylor m = 4, with
each run's validation-best and terminal perplexity, its difference from exp, and the fraction of
its late updates whose gradient was clipped.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import yaml

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]
FIELDS = (
    "run_id", "role", "variant", "status", "parameter_count", "terminal_step", "training_tokens",
    "terminal_validation_nll", "terminal_validation_perplexity", "best_validation_step", "best_validation_nll",
    "best_validation_perplexity", "best_percent_vs_exp", "late_clipped_fraction", "metrics_sha256",
    "completion_sha256",
)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Slots of the model-scale figure's validated categorical palette: exp, fixed Taylor and query keep
# their hues there, and the two MLP widths take its two highest-contrast remaining slots.
VARIANTS = {
    "score_mlp_j4": ("Score MLP, J = 4", "#6250d6"),
    "score_mlp_j8": ("Score MLP, J = 8", "#008300"),
    "exp": ("Exp", "#e87ba4"),
    "query_m4": ("Query m = 4", "#eda100"),
    "fixed_taylor_m4": ("Fixed Taylor m = 4", "#2a78d6"),
}
# Settings every run shares with the query control, which the MLP runs copy except for the kernel.
SHARED = (
    "model.vocab_size", "model.max_seq_len", "model.num_layers", "model.hidden_size", "model.num_heads",
    "model.num_kv_heads", "model.qk_head_dim", "model.value_head_dim", "model.intermediate_size",
    "model.positional_encoding", "model.rope_theta", "model.tie_embeddings", "model.qk_norm", "model.qk_norm_type",
    "model.qk_norm_sharing", "model.qk_norm_head_scale", "model.initialization", "data.dataset_name",
    "data.batch_size", "data.sequence_length", "data.shuffle_seed", "optim.learning_rate", "optim.min_learning_rate",
    "optim.weight_decay", "optim.weight_decay_policy", "optim.beta1", "optim.beta2", "optim.grad_clip_norm",
    "optim.warmup_steps", "optim.total_steps", "train.seed", "train.dtype", "train.eval_batches",
)
SHAPE = {"num_layers": "model.num_layers", "hidden_size": "model.hidden_size", "num_heads": "model.num_heads",
         "intermediate_size": "model.intermediate_size", "batch_size": "data.batch_size",
         "updates": "optim.total_steps", "warmup_steps": "optim.warmup_steps"}
LOCATIONS = {"train.workdir", "train.checkpoint_dir", "train.wandb_run_name"}


def variant_of(model: dict) -> str:
    """The attention variant a recipe trains, or an error for anything outside this comparison."""
    if model.get("attention_normalization") or model.get("attention_output_scaling") != "head_rms_norm":
        raise ValueError("Every variant here is unnormalized attention with head output RMSNorm")
    if model.get("attention_forget_gate") or model.get("positional_encoding", "rope") != "rope":
        raise ValueError("Every variant here uses RoPE without a forget gate")
    width = model.get("attention_score_mlp_width")
    if width is not None:
        if (model.get("attention_score_mlp_architecture"), model.get("attention_score_mlp_form", "kernel"),
                model.get("attention_score_mlp_activation", "silu")) != ("conventional", "kernel", "silu"):
            raise ValueError("The score MLP here is the conventional SiLU kernel")
        if width not in (4, 8):
            raise ValueError(f"The score MLP here has width 4 or 8, not {width}")
        return f"score_mlp_j{width}"
    degree, coefficients = model.get("attention_degree"), model.get("attention_coefficients", "fixed")
    if degree is None:
        if coefficients == "fixed":
            return "exp"
        raise ValueError("Unnormalized attention without a polynomial degree must be exp")
    if degree != 4:
        raise ValueError(f"The polynomial controls here have degree 4, not {degree}")
    if coefficients == "fixed" and model.get("attention_fixed_coefficients") is None:
        return "fixed_taylor_m4"
    if (coefficients == "learned" and model.get("attention_coefficient_conditioning") == "query"
            and model.get("attention_coefficient_activation") == "tanh"
            and not model.get("attention_coefficient_kernel_weight_decay")):
        return "query_m4"
    raise ValueError(f"Unlisted variant: {coefficients} coefficients, "
                     f"conditioning {model.get('attention_coefficient_conditioning')}")


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def check_recipes(entries: list[dict], configs: dict, settings: dict) -> None:
    """Each run trains its declared variant on the control's recipe; the two MLP runs differ only in J."""
    if sorted(entry["variant"] for entry in entries) != sorted(VARIANTS):
        raise ValueError("The selection needs exactly one run of every variant")
    flat = {entry["run_id"]: _flatten(configs[entry["run_id"]]) for entry in entries}
    runs = {entry["variant"]: entry["run_id"] for entry in entries}
    expected = {key: settings["shape"][name] for name, key in SHAPE.items()}
    expected.update({"train.seed": settings["seed"], "model.initialization": settings["initialization"]})
    for entry in entries:
        run, recipe = entry["run_id"], flat[entry["run_id"]]
        actual = variant_of(configs[run]["model"])
        if actual != entry["variant"]:
            raise ValueError(f"{run} trains {actual}, not {entry['variant']}")
        if {key: recipe.get(key) for key in expected} != expected:
            raise ValueError(f"{run} does not train the selection's shape, schedule, seed and initialization")
        changed = sorted(key for key in SHARED if recipe.get(key) != flat[runs["query_m4"]].get(key))
        if changed:
            raise ValueError(f"{run} differs from the query control outside its attention mechanism: {changed}")
    narrow, wide = flat[runs["score_mlp_j4"]], flat[runs["score_mlp_j8"]]
    changed = {key for key in set(narrow) | set(wide) if narrow.get(key) != wide.get(key)}
    if changed - LOCATIONS != {"model.attention_score_mlp_width"}:
        raise ValueError(f"The two score-MLP runs differ beyond their width: {sorted(changed - LOCATIONS)}")


def _curve(path: Path) -> list[tuple[int, float, float | None]]:
    """Validation perplexity by update, with the clipped fraction of the updates since the last evaluation."""
    points = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if row.get("event", "validation") == "validation" and "validation_perplexity" in row:
            points[row["step"]] = (row["validation_perplexity"], row.get("train_gradient_clipped_fraction"))
    return [(step, *points[step]) for step in sorted(points)]


def _late_clipped_fraction(curve: list[tuple[int, float, float | None]], start: int) -> float:
    """The fraction of updates after ``start`` whose gradient norm exceeded the clipping threshold."""
    clipped = total = 0.0
    previous = 0
    for step, _, fraction in curve:
        if previous >= start:
            if fraction is None:
                raise ValueError(f"No gradient-clipping telemetry for the updates ending at {step}")
            clipped, total = clipped + (step - previous) * fraction, total + step - previous
        previous = step
    return clipped / total


def load_runs(selection: dict | None = None) -> list[dict]:
    selection = selection or yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    settings, identity, shape = selection["settings"], selection["data_identity"], selection["settings"]["shape"]
    verify_pinned_runs(ROOT, selection["runs"])
    paths = {entry["run_id"]: ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
             for entry in selection["runs"]}
    configs = {run: yaml.safe_load((path / "config.yaml").read_text()) for run, path in paths.items()}
    check_recipes(selection["runs"], configs, settings)
    rows = []
    for entry in selection["runs"]:
        run, path = entry["run_id"], paths[entry["run_id"]]
        record = json.loads((path / "record.json").read_text())
        if record["status"] != "complete":
            raise ValueError(f"Selected run is not complete: {run} ({record['status']})")
        if {key: record["data_identity"][key] for key in identity} != identity:
            raise ValueError(f"Data manifests differ from the selection: {run}")
        completion = json.loads((path / "completion.json").read_text())
        best = json.loads((path / "best_validation.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        tokens = shape["updates"] * shape["batch_size"] * (configs[run]["data"]["sequence_length"] - 1)
        if (completion["final_step"], completion["training_tokens"]) != (shape["updates"], tokens):
            raise ValueError(f"Selected run did not finish the schedule: {run}")
        curve = _curve(path / "metrics.jsonl")
        if curve[-1][:2] != (shape["updates"], completion["validation_perplexity"]):
            raise ValueError(f"The validation curve does not end at the terminal evaluation: {run}")
        rows.append(dict(
            run_id=run, role=entry["role"], variant=entry["variant"], status=record["status"],
            parameter_count=metadata["parameter_count"], terminal_step=completion["final_step"],
            training_tokens=completion["training_tokens"], terminal_validation_nll=completion["validation_nll"],
            terminal_validation_perplexity=completion["validation_perplexity"], best_validation_step=best["step"],
            best_validation_nll=best["validation_nll"], best_validation_perplexity=best["validation_perplexity"],
            late_clipped_fraction=_late_clipped_fraction(curve, settings["late_window_start"]),
            metrics_sha256=entry["sources"]["metrics.jsonl"]["sha256"],
            completion_sha256=entry["sources"]["completion.json"]["sha256"], curve=curve,
        ))
    exp = next(row for row in rows if row["variant"] == "exp")["best_validation_perplexity"]
    for row in rows:
        row["best_percent_vs_exp"] = 100 * (row["best_validation_perplexity"] / exp - 1)
    order = list(VARIANTS)
    rows.sort(key=lambda row: order.index(row["variant"]))
    return rows


def render(rows: list[dict], start: int, output: Path = DIRECTORY) -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "score-mlp",
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE}):
        figure, axis = plt.subplots(figsize=(8.4, 5.6))
        figure.subplots_adjust(left=0.09, right=0.975, top=0.79, bottom=0.10)
        # Highest perplexity first, so the legend reads in the curves' order at the last update.
        for row in sorted(rows, key=lambda row: -row["best_validation_perplexity"]):
            name, color = VARIANTS[row["variant"]]
            points = [(step, value) for step, value, _ in row["curve"] if step >= start]
            axis.plot(*zip(*points), color=color, linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                      zorder=3 if row["role"] == "primary" else 2,
                      label=f"{name}   best {row['best_validation_perplexity']:.2f}")
        last = max(row["terminal_step"] for row in rows)
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(FixedLocator((15, 20, 30, 50, 100)))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_xlim(start - 150, last + 150)
        axis.set_xticks(range(1000, last, 1000))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
        axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
        axis.set_xlabel("Update", fontsize=9.5, color=INK)
        axis.set_ylabel("Validation perplexity (log scale, lower is better)", fontsize=9.5, color=INK)
        axis.legend(loc="upper right", frameon=False, fontsize=9, handlelength=2.4, labelcolor=INK)
        figure.text(0.09, 0.955, "A learned score MLP in place of exp", fontsize=13, weight="bold", color=INK,
                    va="top")
        figure.text(0.09, 0.905, "Transformer++, 15-layer width-896 shape, 1× Chinchilla; one run (seed 42) per curve, "
                    "from update 500.\nThe score MLP replaces exp with "
                    r"$f(s,x)=b+J^{-1/2}\,\Sigma_j\, a_j\,\mathrm{SiLU}(w_j s+U_j^\top x+b_j)$."
                    "\nEvery curve uses unnormalized attention with head output RMSNorm. Legend values are "
                    "validation-best perplexity.", fontsize=8.5, color=MUTED, va="top", linespacing=1.5)
        figure.savefig(output / "score_mlp.png", dpi=180, metadata={"Software": "Matplotlib"})
        figure.savefig(output / "score_mlp.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                          "Creator": "Matplotlib"})
        plt.close(figure)


def _table(rows: list[dict]) -> str:
    lines = ["| Variant | Run | Best | Terminal | Best vs exp | Late updates clipped | Parameters |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    ranked = sorted(rows, key=lambda row: row["best_validation_perplexity"])
    for row in ranked:
        gap = "–" if row["variant"] == "exp" else f"{row['best_percent_vs_exp']:+.1f}%"
        lines.append(f"| {VARIANTS[row['variant']][0]} | `{row['run_id']}` | {row['best_validation_perplexity']:.3f} | "
                     f"{row['terminal_validation_perplexity']:.3f} | {gap} | {100 * row['late_clipped_fraction']:.1f}% | "
                     f"{row['parameter_count']:,} |")
    lines += ["", "Validation-best updates: " + "; ".join(
        f"{VARIANTS[row['variant']][0]} {row['best_validation_step']:,}" for row in ranked) + ".", "",
        "Late updates clipped: the fraction of updates 4,000–7,570 whose gradient norm exceeded the clipping "
        "threshold of 1."]
    return "\n".join(lines)


def update_tables(rows: list[dict], output: Path = DIRECTORY) -> None:
    path = output / "README.md"
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    prefix, rest = path.read_text().split(begin)
    _, suffix = rest.split(end)
    path.write_text(prefix + begin + "\n\n" + _table(rows) + "\n\n" + end + suffix)


def main(output: Path = DIRECTORY) -> None:
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    rows = load_runs(selection)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, selection["settings"]["curve_start_update"], output)
    update_tables(rows, output)


if __name__ == "__main__":
    main()
