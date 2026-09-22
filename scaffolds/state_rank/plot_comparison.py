"""Compare pretrained and random-weight state spectra without pooling spectra.

Run ``python -m scaffolds.state_rank.plot_comparison --results DIRECTORY``.
Only complete conditions enter figures and summaries. Random-seed envelopes
show a descriptive min/max across seed means, not independent-head uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scaffolds.state_rank.artifacts import write_report_section


RANK_METRICS = (
    ("participation_ratio", "Participation ratio"),
    ("threshold_rank", "Threshold rank"),
    ("energy_rank90", "Modes for 90% energy"),
    ("energy_rank99", "Modes for 99% energy"),
)
REQUIRED_SCALAR_METRICS = tuple(key for key, _ in RANK_METRICS) + ("log_trace",)
SCALAR_METRICS = REQUIRED_SCALAR_METRICS + ("key_norms", "value_norms")
FIGURE_NAMES = ("token_metrics", "terminal_rank_depth", "normalized_spectra", "norm_energy_depth")
PRETRAINED_COLOR = "#1f567d"
RANDOM_COLOR = "#c16828"


@dataclass
class Condition:
    name: str
    weights: str
    seed: int | None
    metadata: dict
    arrays: dict[str, np.ndarray]


def _validate_arrays(name: str, arrays: dict[str, np.ndarray]) -> None:
    required = {"layers", "head_indices", "prefixes", "spectrum_prefixes", "token_ids",
                "normalized_eigenvalues", *REQUIRED_SCALAR_METRICS}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Complete condition {name!r} is missing arrays: {sorted(missing)}")
    if ("key_norms" in arrays) != ("value_norms" in arrays):
        raise ValueError(f"{name}: key/value norms must either both be present or both be omitted")
    for key in ("layers", "head_indices", "prefixes", "spectrum_prefixes"):
        values = arrays[key]
        if values.ndim != 1 or not values.size or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name}: {key} must be a nonempty integer vector")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{name}: {key} must be strictly increasing")
    layers, heads, prefixes, selected = (arrays[key] for key in (
        "layers", "head_indices", "prefixes", "spectrum_prefixes"))
    if np.any(layers < 0) or np.any(heads < 0):
        raise ValueError(f"{name}: layer/head indices must be nonnegative")
    if not np.array_equal(prefixes, np.arange(1, len(prefixes) + 1)):
        raise ValueError(f"{name}: prefixes must be all token positions from 1 to T")
    if not np.isin(selected, prefixes).all():
        raise ValueError(f"{name}: saved spectrum prefixes must be measured token positions")
    tokens = arrays["token_ids"]
    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] != len(prefixes):
        raise ValueError(f"{name}: token_ids must have shape [inputs, tokens]")
    scalar_shape = (len(layers), tokens.shape[0], len(prefixes), len(heads))
    for key in SCALAR_METRICS:
        if key not in arrays:
            continue
        values = arrays[key]
        if values.shape != scalar_shape:
            raise ValueError(f"{name}: {key} shape {values.shape} != {scalar_shape}")
        if key == "log_trace":
            valid = np.isfinite(values) | np.isneginf(values)
        else:
            valid = np.isfinite(values) & (values >= 0)
        if not valid.all():
            raise ValueError(f"{name}: invalid {key} values")
    eigenvalues = arrays["normalized_eigenvalues"]
    expected = (len(layers), tokens.shape[0], len(selected), len(heads))
    if eigenvalues.ndim != 5 or eigenvalues.shape[:-1] != expected or eigenvalues.shape[-1] < 1:
        raise ValueError(f"{name}: normalized_eigenvalues must have shape [L,I,S,H,D]")
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues < 0):
        raise ValueError(f"{name}: eigenvalues must be finite and nonnegative")
    if np.any(np.diff(eigenvalues, axis=-1) > 1e-10):
        raise ValueError(f"{name}: eigenvalues must be sorted in descending order")
    totals = eigenvalues.sum(axis=-1)
    if not np.all(np.isclose(totals, 1., rtol=1e-5, atol=1e-7) | (totals == 0)):
        raise ValueError(f"{name}: normalized eigenvalues must sum to one, or zero for a zero state")


def load_conditions(results: Path) -> tuple[dict, list[Condition], list[dict]]:
    """Load completed measurements and reject mismatched inputs/axes across them."""
    manifest = json.loads((results / "manifest.json").read_text())
    conditions, statuses = [], []
    names = set()
    for entry in manifest["conditions"]:
        name = entry["name"]
        if not isinstance(name, str) or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError(f"Invalid or duplicate condition name: {name!r}")
        names.add(name)
        weights, seed = entry["weights"], entry.get("seed")
        if weights not in ("pretrained", "random"):
            raise ValueError(f"Unknown weight condition: {weights!r}")
        directory = results / name
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        status = metadata.get("status", entry.get("status", "pending"))
        statuses.append({"name": name, "weights": weights, "seed": seed, "status": status})
        if status != "complete":
            continue
        path = directory / "metrics.npz"
        if not path.exists():
            raise ValueError(f"Complete condition {name!r} has no metrics.npz")
        if manifest.get("identity"):
            if metadata.get("identity") != manifest["identity"]:
                raise ValueError(f"{name}: condition identity differs from its manifest")
            if metadata.get("weights") != weights or metadata.get("seed") != seed:
                raise ValueError(f"{name}: condition weights or seed differ from its manifest")
            if not metadata.get("metrics_sha256"):
                raise ValueError(f"{name}: complete condition has no archive checksum")
        if metadata.get("metrics_sha256"):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != metadata["metrics_sha256"]:
                raise ValueError(f"{name}: metrics archive checksum differs from its metadata")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        _validate_arrays(name, arrays)
        if conditions:
            reference = conditions[0].arrays
            if ("key_norms" in arrays) != ("key_norms" in reference):
                raise ValueError(f"{name}: norm availability differs across conditions")
            for key in ("layers", "head_indices", "prefixes", "spectrum_prefixes", "token_ids"):
                if not np.array_equal(arrays[key], reference[key]):
                    raise ValueError(f"{name}: {key} differs from {conditions[0].name}; cannot compare")
            if arrays["normalized_eigenvalues"].shape != reference["normalized_eigenvalues"].shape:
                raise ValueError(f"{name}: spectrum dimensions differ across conditions")
        conditions.append(Condition(name, weights, seed, metadata, arrays))
    return manifest, conditions, statuses


def aggregate_condition(condition: Condition) -> dict[str, np.ndarray]:
    """Reduce input/head axes after computing each head's scientific metric."""
    arrays = condition.arrays
    curves = {key: arrays[key].mean(axis=(1, 3)) for key in SCALAR_METRICS if key in arrays}
    for kind in ("key", "value"):
        if f"{kind}_norms" in arrays:
            curves[f"{kind}_l2_rms"] = np.sqrt(np.square(arrays[f"{kind}_norms"]).mean(axis=(1, 3)))
    eigenvalues = arrays["normalized_eigenvalues"]
    relative = np.divide(eigenvalues, eigenvalues[..., :1], out=np.zeros_like(eigenvalues),
                         where=eigenvalues[..., :1] > 0)
    curves["relative_singular_values"] = np.sqrt(relative).mean(axis=(1, 3))
    return curves


def summary_rows(conditions: list[Condition]) -> list[dict]:
    rows = []
    for condition in conditions:
        curves = aggregate_condition(condition)
        arrays = condition.arrays
        for index, layer in enumerate(arrays["layers"]):
            row = {"condition": condition.name, "weights": condition.weights, "seed": condition.seed,
                   "layer": int(layer), "terminal_prefix": int(arrays["prefixes"][-1]),
                   "inputs": int(arrays["token_ids"].shape[0]), "heads": int(len(arrays["head_indices"]))}
            for key in SCALAR_METRICS:
                if key in curves:
                    row[f"final_{key}_mean"] = float(curves[key][index, -1])
            for key in ("key_l2_rms", "value_l2_rms"):
                if key in curves:
                    row[f"final_{key}"] = float(curves[key][index, -1])
            rows.append(row)
    return rows


def _plot_comparison(axis, x, conditions, values, *, positive_floor=None):
    """Each random curve is one seed's mean; the band is the range of seed means."""
    def displayed(values):
        return np.maximum(values, positive_floor) if positive_floor is not None else values

    random = [values[condition.name] for condition in conditions if condition.weights == "random"]
    if random:
        seed_values = np.stack(random)
        for curve in seed_values:
            axis.plot(x, displayed(curve), color=RANDOM_COLOR, alpha=.22, linewidth=.9)
        axis.fill_between(x, displayed(seed_values.min(axis=0)), displayed(seed_values.max(axis=0)),
                          color=RANDOM_COLOR, alpha=.15, linewidth=0, label="Random seed range")
        axis.plot(x, displayed(seed_values.mean(axis=0)), color=RANDOM_COLOR, linewidth=1.9,
                  label=f"Random mean ({len(random)} seed{'s' if len(random) != 1 else ''})")
    for condition in conditions:
        if condition.weights == "pretrained":
            axis.plot(x, displayed(values[condition.name]), color=PRETRAINED_COLOR,
                      linewidth=2.0, label="Pretrained mean")
    from matplotlib.ticker import MaxNLocator
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=.20, linewidth=.6)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure, results: Path, name: str) -> None:
    # Resolve constrained layout once, then keep PNG/PDF tight-bbox exports from
    # independently moving axes after their bounding boxes have been measured.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    for extension in ("png", "pdf"):
        figure.savefig(results / f"{name}.{extension}", dpi=180, bbox_inches="tight")


def _plot_figures(results: Path, conditions: list[Condition], comparison_layer: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = conditions[0].arrays
    layers, prefixes = arrays["layers"], arrays["prefixes"]
    layer_index = int(np.flatnonzero(layers == comparison_layer)[0])
    curves = {condition.name: aggregate_condition(condition) for condition in conditions}
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, RANK_METRICS, strict=True):
            _plot_comparison(axis, prefixes, conditions,
                             {name: values[metric][layer_index] for name, values in curves.items()})
            axis.set(title=label, xlabel="Prefix length (tokens)", ylabel="Rank / effective modes")
        axes[0, 0].legend(loc="best", frameon=False)
        figure.suptitle(f"Layer index {comparison_layer}: state rank through the input")
        _save_figure(figure, results, "token_metrics")
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, RANK_METRICS, strict=True):
            _plot_comparison(axis, layers, conditions,
                             {name: values[metric][:, -1] for name, values in curves.items()})
            axis.set(title=label, xlabel="Layer index (zero based)", ylabel="Rank / effective modes")
        axes[0, 0].legend(loc="best", frameon=False)
        figure.suptitle(f"Terminal state rank at {prefixes[-1]} tokens")
        _save_figure(figure, results, "terminal_rank_depth")
        plt.close(figure)

        selected = arrays["spectrum_prefixes"]
        # The production design has six panels. Tiny fixtures and alternate saved
        # prefix sets use the same representation without imposing fixed shapes.
        columns = min(3, len(selected))
        rows = (len(selected) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(3.35 * columns, 2.65 * rows),
                                    sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        mode_indices = np.arange(1, arrays["normalized_eigenvalues"].shape[-1] + 1)
        for index, (axis, prefix) in enumerate(zip(axes.flat, selected)):
            _plot_comparison(axis, mode_indices, conditions,
                             {name: values["relative_singular_values"][layer_index, index]
                              for name, values in curves.items()}, positive_floor=1e-6)
            axis.set(yscale="log", ylim=(8e-7, 1.2), title=f"Prefix {prefix}",
                     xlabel="Singular-value index", ylabel=r"$\sigma_i / \sigma_1$")
        for axis in list(axes.flat)[len(selected):]:
            axis.set_visible(False)
        axes[0, 0].legend(loc="best", frameon=False)
        figure.suptitle(f"Layer index {comparison_layer}: relative singular spectra (display floor 10⁻⁶)")
        _save_figure(figure, results, "normalized_spectra")
        plt.close(figure)

        panels = [("log_trace", "Mean log₁₀ state energy")]
        if "key_norms" in arrays:
            panels = [("key_l2_rms", "Key L2 norm (RMS)"), ("value_l2_rms", "Value L2 norm (RMS)")] + panels
        figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), constrained_layout=True, squeeze=False)
        for axis, (metric, label) in zip(axes.flat, panels, strict=True):
            values = {name: values[metric][:, -1] for name, values in curves.items()}
            if metric == "log_trace":
                values = {name: value / np.log(10.) for name, value in values.items()}
            _plot_comparison(axis, layers, conditions, values)
            axis.set(xlabel="Layer index (zero based)", ylabel=label)
        axes.flat[0].legend(loc="best", frameon=False)
        figure.suptitle(f"State energy at token {prefixes[-1]}" if len(panels) == 1 else
                       f"Norms at token {prefixes[-1]} and cumulative state energy")
        _save_figure(figure, results, "norm_energy_depth")
        plt.close(figure)


def _write_readme(results, manifest, conditions, statuses, comparison_layer):
    config = manifest.get("config", {})
    has_norms = bool(conditions and "key_norms" in conditions[0].arrays)
    random_count = sum(condition.weights == "random" for condition in conditions)
    lines = ["# Pretrained versus random-weight state rank", "",
             f"Complete conditions: {len(conditions)}/{len(statuses)}; completed random seeds: {random_count}.", "",
             "| Condition | Weights | Seed | Status |", "|---|---|---:|---|"]
    for status in statuses:
        seed = "—" if status["seed"] is None else status["seed"]
        lines.append(f"| {status['name']} | {status['weights']} | {seed} | {status['status']} |")
    if conditions:
        arrays = conditions[0].arrays
        lines += ["", f"Figures and `summary.csv` include complete conditions only. All included conditions use identical token IDs, layer indices, head indices, and saved spectrum prefixes.",
                  f"There are {arrays['token_ids'].shape[0]} fixed inputs, {len(arrays['prefixes'])} tokens per input, {len(arrays['head_indices'])} state heads, and {len(arrays['layers'])} measured layers. Layers use zero-based indices. Token and spectrum panels show layer index {comparison_layer}.", "",
                  "- `token_metrics.{png,pdf}`: four rank metrics at every prefix.",
                  "- `terminal_rank_depth.{png,pdf}`: terminal rank metrics across layers.",
                  "- `normalized_spectra.{png,pdf}`: relative singular spectra at saved prefixes.",
                  "- `norm_energy_depth.{png,pdf}`: " + ("terminal-token key/value norms and " if has_norms else "") + "terminal state energy.",
                  "- `summary.csv`: one row per complete condition and layer; terminal metrics retain full numerical precision."]
    else:
        lines += ["", "No complete measurements are available yet. Figures will be produced once a condition is complete."]
    tolerance_values = {condition.metadata.get("threshold_rank_singular_tolerance")
                        for condition in conditions}
    tolerance_values.discard(None)
    if len(tolerance_values) == 1:
        tolerance = next(iter(tolerance_values))
        lines += ["", f"Threshold rank counts singular values strictly greater than {tolerance:g} times the largest singular value."]
    lines += ["", "## Interpretation and aggregation", "",
              "Participation ratio is (sum(sigma²))² / sum(sigma⁴). The 90% and 99% energy ranks count the minimum number of descending modes whose squared singular values reach the respective energy fraction.", "",
              "Rank metrics are computed for each input/head state first, then averaged equally over inputs and heads. We never compute a rank from an averaged spectrum. Each faint random curve is one seed's input/head aggregate; the solid random curve is the equal-weight mean over completed seeds, and shading is the minimum–maximum range of those seed aggregates. It is a descriptive range, not a confidence interval. Inputs and heads are not treated as independent experimental replications, and corresponding pretrained/random heads are not assumed to be matched.", "",
              "For each saved input/head state, relative singular values are computed as sqrt(lambda_i / lambda_1) from the normalized eigenvalues, and only then averaged over inputs and heads. A zero state contributes zeros. Spectral values below 10⁻⁶ are clipped only for logarithmic display; saved measurements and CSV values are unchanged.", "",
              (("Key/value norm panels show sqrt(mean(||vector||₂²)) across inputs and heads at the terminal token. These are RMS vector norms, not per-component RMS and not averages across all token positions. " if has_norms else "Key/value vector norms were not collected for this native recurrent-state diagnostic. ") + "The energy panel averages the natural log of squared state Frobenius norm over input/head states, then converts to base 10; it is not log of the arithmetic mean energy. Zero state energy is represented by negative infinity."), "",
              "This is a descriptive diagnostic on a fixed, small set of inputs. The pretrained model is one checkpoint, while random weights are independent initialization seeds of the same architecture; seed ranges do not measure pretrained uncertainty. Matching the inputs and architecture isolates this comparison but does not identify a causal contribution of an individual head or establish generalization to other texts, checkpoints, model sizes, or training recipes. No language-model quality claim follows from state rank alone."]
    if config.get("model_id"):
        lines += ["", f"Model: `{config['model_id']}`; revision: `{config.get('model_revision', 'unspecified')}`."]
    if manifest.get("state_definition"):
        lines += ["", f"Measured state: {manifest['state_definition']}."]
    if manifest.get("initialization"):
        lines += ["", f"Random initialization: {manifest['initialization']}."]
    lines += ["", "Resolved configuration and planned conditions are in `manifest.json`; condition-specific seeds, source commits, and execution metadata are in each condition's `metadata.json`. Incomplete conditions remain visible in the status table and do not enter aggregates."]
    write_report_section(results, f"comparison-{results.name}", "\n".join(lines))


def plot_comparison(results: str | Path) -> dict:
    results = Path(results).expanduser().resolve()
    manifest, conditions, statuses = load_conditions(results)
    rows = summary_rows(conditions)
    metrics = (tuple(key for key in SCALAR_METRICS if key in conditions[0].arrays)
               if conditions else REQUIRED_SCALAR_METRICS)
    fields = ["condition", "weights", "seed", "layer", "terminal_prefix", "inputs", "heads",
              *(f"final_{key}_mean" for key in metrics)]
    if "key_norms" in metrics:
        fields += ["final_key_l2_rms", "final_value_l2_rms"]
    with (results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    comparison_layer = None
    if conditions:
        layers = conditions[0].arrays["layers"]
        requested = manifest.get("config", {}).get("comparison_layer", 8)
        comparison_layer = int(requested if requested in layers else layers[len(layers) // 2])
        _plot_figures(results, conditions, comparison_layer)
    else:
        for name in FIGURE_NAMES:
            for extension in ("png", "pdf"):
                (results / f"{name}.{extension}").unlink(missing_ok=True)
    _write_readme(results, manifest, conditions, statuses, comparison_layer)
    return {"complete_conditions": len(conditions), "planned_conditions": len(statuses),
            "comparison_layer": comparison_layer, "summary_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(plot_comparison(args.results), sort_keys=True))


if __name__ == "__main__":
    main()
