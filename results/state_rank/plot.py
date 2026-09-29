"""State-rank figure, per-layer summary, and README table from the selected runs.

For each model, one layer is fixed; each head is one curve of the state's
participation ratio and threshold rank against prefix length, averaged over the
input sequences (and, for random models, over seeds at the same head index).
Ranks are computed per state before any averaging. The pinned numerical archives
are ignored by Git and live on Kennel, so regeneration runs in the primary checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re

import numpy as np
import yaml

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parents[1]
METRICS = {"participation_ratio": "Participation ratio", "threshold_rank": "Threshold rank (σ > 10⁻⁴ σ₁)"}
HEAD_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TABLE = re.compile(r"(<!-- state-rank:summary:start -->\n)(?:.*?\n)?(<!-- state-rank:summary:end -->)", re.S)
FIELDS = ["model", "layer", "condition", "participation_ratio", "participation_ratio_seed_min",
          "participation_ratio_seed_max", "threshold_rank", "threshold_rank_seed_min", "threshold_rank_seed_max"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection(results: Path = RESULTS) -> dict:
    return yaml.safe_load((results / "selection.yaml").read_text())


def archives(selection: dict, root: Path = ROOT) -> dict[str, dict[str, np.ndarray]]:
    """Each role's metric arrays [layer, input, token, head], after checking its pinned hashes."""
    loaded = {}
    for entry in selection["runs"]:
        run = root / selection["study"] / "runs" / entry["run_id"]
        if sha256(run / "config.yaml") != entry["config_sha256"]:
            raise ValueError(f"{entry['label']}: recipe differs from the selection pin")
        archive = entry["inputs"]["numerical_data"]
        path = root / archive["path"]
        if sha256(path) != archive["sha256"]:
            raise ValueError(f"{entry['label']}: numerical archive differs from the selection pin")
        with np.load(path) as data:
            loaded[entry["label"]] = {name: data[name] for name in (*METRICS, "layers", "prefixes")}
    return loaded


def layer_index(arrays: dict, layer: int) -> int:
    return int(np.flatnonzero(arrays["layers"] == layer)[0])


def head_curves(arrays: dict, layer: int, metric: str) -> np.ndarray:
    """[token, head], averaged over input sequences at one layer."""
    return arrays[metric][layer_index(arrays, layer)].mean(axis=0)


def settings(selection: dict, data: dict) -> list[dict]:
    """The four panels' columns: each model's pretrained and seed-averaged random head curves."""
    columns = []
    for model in selection["analysis"]["models"]:
        pretrained = data[model["pretrained"]]
        for condition, roles in (("pretrained", [model["pretrained"]]), ("random", model["random"])):
            curves = {metric: np.mean([head_curves(data[role], model["layer"], metric) for role in roles], axis=0)
                      for metric in METRICS}
            columns.append({"model": model, "condition": condition, "seeds": len(roles),
                            "prefixes": pretrained["prefixes"], "curves": curves})
    return columns


def layer_summary(selection: dict, data: dict) -> list[dict]:
    """Final-token values per layer, averaged over sequences and heads; random models also over seeds."""
    rows = []
    for model in selection["analysis"]["models"]:
        layers = data[model["pretrained"]]["layers"]
        for index, layer in enumerate(layers):
            final = {role: {metric: float(data[role][metric][index, :, -1, :].mean()) for metric in METRICS}
                     for role in (model["pretrained"], *model["random"])}
            row = {"model": model["key"], "layer": int(layer), "condition": "pretrained"}
            row.update({metric: f"{final[model['pretrained']][metric]:.6g}" for metric in METRICS})
            rows.append({field: row.get(field, "") for field in FIELDS})
            seeds = {metric: [final[role][metric] for role in model["random"]] for metric in METRICS}
            row = {"model": model["key"], "layer": int(layer), "condition": "random"}
            for metric, values in seeds.items():
                row.update({metric: f"{np.mean(values):.6g}", f"{metric}_seed_min": f"{min(values):.6g}",
                            f"{metric}_seed_max": f"{max(values):.6g}"})
            rows.append({field: row.get(field, "") for field in FIELDS})
    return rows


def figure(columns: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "path.simplify": False})
    fig, axes = plt.subplots(len(METRICS), len(columns), figsize=(3.2 * len(columns), 5.6), sharex="col")
    for column, setting in enumerate(columns):
        model, prefixes = setting["model"], setting["prefixes"]
        title = f"{model['label']}, layer {model['layer']}\n" + (
            "pretrained" if setting["condition"] == "pretrained" else f"random ({setting['seeds']} seeds)")
        for row, metric in enumerate(METRICS):
            axis = axes[row, column]
            for head in range(setting["curves"][metric].shape[1]):
                axis.plot(prefixes, setting["curves"][metric][:, head], color=HEAD_COLORS[head], linewidth=1.5,
                          label=f"Head {head}")
            if metric == "threshold_rank":
                axis.plot(prefixes, np.minimum(prefixes, model["state_width"]), color=MUTED, linewidth=1,
                          linestyle="--", label="Rank ceiling")
            axis.grid(True, color=GRID, linewidth=0.8)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
            if row == 0:
                axis.set_title(title, color=INK, fontsize=9)
            if row == len(METRICS) - 1:
                axis.set_xlabel("Prefix length (tokens)", color=INK, fontsize=9)
            if column == 0:
                axis.set_ylabel(METRICS[metric], color=INK, fontsize=9)
    # Pretrained and random columns of one model share each row's scale.
    for start in range(0, len(columns), 2):
        for row in range(len(METRICS)):
            pair = axes[row, start:start + 2]
            top = max(axis.get_ylim()[1] for axis in pair)
            for axis in pair:
                axis.set_ylim(0, top)
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"state_rank.{suffix}", dpi=180,
                    metadata={"CreationDate": None} if suffix == "pdf" else None)
    plt.close(fig)


def table(columns: list[dict]) -> str:
    lines = ["| Setting | Layer | Participation ratio | Threshold rank |", "|---|---:|---:|---:|"]
    for setting in columns:
        model = setting["model"]
        final = {metric: setting["curves"][metric][-1] for metric in METRICS}
        name = f"{model['label']}, {'pretrained' if setting['condition'] == 'pretrained' else 'random'}"
        lines.append(f"| {name} | {model['layer']} | {final['participation_ratio'].mean():.2f} "
                     f"| {final['threshold_rank'].mean():.1f} / {model['state_width']} |")
    return "\n".join(lines)


def regenerate(results: Path = RESULTS, output: Path | None = None, root: Path = ROOT) -> None:
    output = output or results
    output.mkdir(parents=True, exist_ok=True)
    selection = load_selection(results)
    data = archives(selection, root)
    columns = settings(selection, data)
    figure(columns, output)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(layer_summary(selection, data))
    readme = (results / "README.md").read_text()
    if len(TABLE.findall(readme)) != 1:
        raise ValueError("README needs exactly one state-rank summary marker pair")
    (output / "README.md").write_text(TABLE.sub(lambda match: match[1] + table(columns) + "\n" + match[2], readme))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write outputs here instead of the result collection")
    regenerate(output=parser.parse_args().output)


if __name__ == "__main__":
    main()
