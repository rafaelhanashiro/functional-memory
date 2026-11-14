"""Compare a fixed head's tanh saturation from pinned compact activation probes.

The probes use the same held-out sequence and checkpoint update. Generator
coordinates differ between direct and sum-of-squares coefficients; averages give
each coordinate and token equal weight, not equal influence on model output.
No GPU, checkpoint, dataset or numerical archive is needed to regenerate.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs


BLUE = "#28699b"
ORANGE = "#d66a32"
INK = "#24323d"
LIGHT = "#edf1f5"


def coordinate_labels(degree: int, family: str) -> list[str]:
    if family == "direct":
        return [rf"$r={r}$" for r in range(degree + 1)]
    if family == "sum_of_squares":
        half = degree // 2
        return [rf"$U_{r}$" for r in range(half + 1)] + [rf"$V_{r}$" for r in range(half)]
    raise ValueError(f"Unknown coefficient parameterization: {family}")


def check_summary(value: dict) -> None:
    mean, std, low, high = (float(value[key]) for key in ("mean", "std", "min", "max"))
    assert all(math.isfinite(number) for number in (mean, std, low, high))
    assert low <= mean <= high and std >= 0


def load(selection: dict) -> list[dict]:
    verify_pinned_runs(ROOT, selection["runs"])
    settings = selection["settings"]
    assert settings["saturation_absolute_tanh_threshold"] == .99
    assert settings["mostly_saturated_token_fraction_threshold"] == .90
    probes = {entry["role"]: entry for entry in selection["runs"] if entry["kind"] == "probe"}
    parents = [entry for entry in selection["runs"] if entry["kind"] == "parent"]
    assert len(probes) == 2 and len(parents) == 4
    measurements = {}
    for role, entry in probes.items():
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        if load_run(run)["status"] != "complete":
            raise RuntimeError(f"Activation probe {entry['run_id']} has not completed")
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        assert len(records) == 2
        measurements[role] = {row["role"]: row for row in records}
        assert len(measurements[role]) == len(records)
    result = []
    for parent in sorted(parents, key=lambda entry: (entry["degree"], entry["family"] != "direct")):
        row = measurements[parent["probe_role"]][parent["measurement_role"]]
        run = ROOT / "experiments" / parent["study"] / "runs" / parent["run_id"]
        record = load_run(run)
        assert record["status"] == "complete"
        model = yaml.safe_load((run / "config.yaml").read_text())["model"]
        assert model["attention_coefficient_parameterization"] == parent["family"]
        assert model["attention_degree"] == parent["degree"] == row["degree"]
        assert model["attention_coefficient_activation"] == "tanh"
        assert model["attention_coefficient_conditioning"] == "query"
        assert model["attention_coefficient_kernel_weight_decay"] is True
        assert model["positional_encoding"] == "rope"
        best = json.loads((run / "best_validation.json").read_text())
        assert best["step"] == settings["checkpoint_step"]
        assert row["source_run_id"] == parent["run_id"]
        assert row["checkpoint_step"] == parent["checkpoint_step"] == settings["checkpoint_step"]
        assert row["checkpoint_sha256"] == record["selected_checkpoint"]["sha256"]
        assert row["layer"] == settings["layer_index"] and row["head"] == settings["head_index"]
        assert row["dimension"] == settings["input_dimension"]
        assert row["indexing"] == "zero_based"
        assert row["sequences"] == settings["sequences"]
        assert row["prediction_tokens"] == settings["prediction_tokens"]
        sample = row["sample"]
        assert sample["selection"] == settings["sample_selection"]
        assert sample["num_sequences"] == settings["sequences"]
        assert sample["prediction_tokens"] == settings["prediction_tokens"]
        assert sample["record_sha256"] == settings["sample_record_sha256"]
        assert row["optimizer_updates"] == 0 and row["parent_parameters_unchanged"]
        assert row["parent_parameter_sha256_before"] == row["parent_parameter_sha256_after"]
        assert row["reference_validation"]["checkpoint_parameters_match"]
        validation = row["native_capture_validation"]
        assert validation["rmsnorm_output_equals_generator_input"]
        assert validation["manual_coefficient_max_normalized_error"] <= validation["manual_coefficient_tolerance"]
        for name in ("x_norm_before", "x_norm_after"):
            check_summary(row[name])
            assert row[name]["min"] >= 0
        coordinates = row["degrees"]
        assert [value["degree"] for value in coordinates] == list(range(row["degree"] + 1))
        if parent["family"] == "sum_of_squares":
            assert row["coefficient_parameterization"] == "sum_of_squares"
            half = row["degree"] // 2
            assert [value["coordinate_label"] for value in coordinates] == (
                [f"U{r}" for r in range(half + 1)] + [f"V{r}" for r in range(half)])
        for value in coordinates:
            assert math.isfinite(value["w_norm"]) and value["w_norm"] >= 0
            for field in ("z", "tanh", "tanh_derivative"):
                check_summary(value[field])
            assert 0 <= value["tanh_derivative"]["mean"] <= 1
            fractions = [value[name] for name in ("saturation_abs_tanh_ge099",
                         "saturation_tanh_ge099", "saturation_tanh_le_neg099")]
            assert all(math.isfinite(fraction) and 0 <= fraction <= 1 for fraction in fractions)
            assert math.isclose(fractions[0], fractions[1] + fractions[2], rel_tol=0, abs_tol=1e-12)
        result.append({**row, "family": parent["family"], "label": parent["label"],
                       "probe_run_id": probes[parent["probe_role"]]["run_id"],
                       "validation_best_perplexity": best["validation_perplexity"]})
    assert {(row["family"], row["degree"]) for row in result} == {
        (family, degree) for family in ("direct", "sum_of_squares") for degree in (8, 12)}
    return result


def aggregate(row: dict, settings: dict) -> dict:
    coordinates = row["degrees"]
    norms = np.asarray([value["w_norm"] for value in coordinates])
    saturation = np.asarray([value["saturation_abs_tanh_ge099"] for value in coordinates])
    derivative = np.asarray([value["tanh_derivative"]["mean"] for value in coordinates])
    return {
        "model": row["label"], "family": row["family"], "model_degree": row["degree"],
        "source_run_id": row["source_run_id"], "probe_run_id": row["probe_run_id"],
        "checkpoint_step": row["checkpoint_step"], "layer_index": row["layer"], "head_index": row["head"],
        "sequences": row["sequences"], "prediction_tokens": row["prediction_tokens"],
        "generator_coordinates": len(coordinates),
        "validation_best_perplexity": row["validation_best_perplexity"],
        "saturation_absolute_tanh_threshold": settings["saturation_absolute_tanh_threshold"],
        "saturation_fraction_equal_coordinate_weight": float(saturation.mean()),
        "mean_tanh_derivative_equal_coordinate_weight": float(derivative.mean()),
        "mostly_saturated_token_fraction_threshold": settings["mostly_saturated_token_fraction_threshold"],
        "coordinates_saturated_on_more_than_threshold_fraction_of_tokens": int(
            np.count_nonzero(saturation > settings["mostly_saturated_token_fraction_threshold"])),
        "x_norm_before_mean": row["x_norm_before"]["mean"],
        "x_norm_after_mean": row["x_norm_after"]["mean"],
        "mean_activation_norm": row["mean_activation_norm"],
        "w_norm_mean": float(norms.mean()), "w_norm_min": float(norms.min()), "w_norm_max": float(norms.max()),
        "sample_record_sha256": row["sample"]["record_sha256"],
    }


def plot(rows: list[dict], settings: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.4), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.765, bottom=.165, hspace=.61, wspace=.15)
    for ax, row in zip(axes.flat, rows, strict=True):
        coordinates = row["degrees"]
        positions = np.arange(len(coordinates))
        positive = np.asarray([value["saturation_tanh_ge099"] for value in coordinates]) * 100
        negative = np.asarray([value["saturation_tanh_le_neg099"] for value in coordinates]) * 100
        ax.bar(positions, 100, color=LIGHT, width=.72, zorder=1)
        ax.bar(positions, positive, color=ORANGE, width=.72, zorder=2)
        ax.bar(positions, negative, bottom=positive, color=BLUE, width=.72, zorder=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b2bac3")
        ax.tick_params(colors="#4d5965", labelsize=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-.65, len(coordinates) - .35)
        ax.set_xticks(positions, labels=coordinate_labels(row["degree"], row["family"]))
        ax.grid(axis="y", color="#dce2e8", lw=.7, zorder=0)
        if row["family"] == "direct":
            ax.set_ylabel("Saturated query tokens (%)", fontsize=11)
            ax.set_xlabel("Polynomial coefficient order", fontsize=11, labelpad=8)
        else:
            ax.axvline(row["degree"] // 2 + .5, color="#aab3bc", lw=.9, ls=(0, (3, 3)))
            ax.set_xlabel("Square-factor coefficient", fontsize=11, labelpad=8)
        result = aggregate(row, settings)
        title = "Regular query" if row["family"] == "direct" else "SoS query"
        ax.set_title(f"{title} · $m={row['degree']}$", loc="left", fontsize=15,
                     fontweight="bold", color=INK, pad=38)
        ax.text(0, 1.07,
                f"All coordinates: {100 * result['saturation_fraction_equal_coordinate_weight']:.1f}% saturated"
                f"  ·  mean tanh derivative {result['mean_tanh_derivative_equal_coordinate_weight']:.3f}",
                transform=ax.transAxes, fontsize=10.7, color="#5a6672")
    fig.suptitle("Transformer++ · regular query and SoS · tanh saturation",
                 x=.055, y=.975, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(.055, .920,
             f"Layer {settings['layer_index'] + 1}, head {settings['head_index'] + 1} · checkpoint {settings['checkpoint_step']:,}"
             f" · same first held-out sequence ({settings['prediction_tokens']:,} query tokens)",
             fontsize=12.5, color="#5a6672")
    handles = [Patch(facecolor=ORANGE, label=r"$\tanh(W^\top x)\geq0.99$"),
               Patch(facecolor=BLUE, label=r"$\tanh(W^\top x)\leq-0.99$"),
               Patch(facecolor=LIGHT, label="Outside saturation threshold")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .881), frameon=False,
               ncol=3, fontsize=11, columnspacing=2.0)
    fig.text(.055, .043,
             "Saturation is the fraction of tokens with |tanh z| ≥ 0.99. Panel averages weight all generator coordinates equally.\n"
             "Regular query coordinates are expanded polynomial orders; SoS coordinates belong to U and V in p(s) = U(s)² + V(s)².\n"
             "One head, checkpoint and sequence: these measurements do not establish whole-model behavior or explain the performance difference.",
             fontsize=10, color="#687583", linespacing=1.65)
    fig.savefig(OUT / "saturation.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "saturation.pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def save_summary(rows: list[dict], settings: dict) -> None:
    summaries = [aggregate(row, settings) for row in rows]
    with (OUT / "saturation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: format(value, ".12g") if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})


def main() -> None:
    selection = yaml.safe_load((OUT / "saturation_selection.yaml").read_text())
    rows = load(selection)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        plot(rows, selection["settings"])
    save_summary(rows, selection["settings"])
    print("Regenerated matched-head saturation comparison from pinned compact measurements.")


if __name__ == "__main__":
    main()
