"""Regenerate the isolated recipe's LR curves and explicit attention comparison."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from scaffolds.records.selections import verify_pinned_runs
from scaffolds.records import validate_record

DIRECTORY = Path(__file__).resolve().parent


def collect(selection: dict, root: Path | None = None) -> list[dict]:
    root = DIRECTORY.parents[2] if root is None else root
    verify_pinned_runs(root, selection["runs"])
    rows = []
    for entry in selection["runs"]:
        run = root / "experiments" / entry["study"] / "runs" / entry["run_id"]
        # Scientific inputs are pinned above; this checks the single authoritative
        # outcome without requiring checkpoints or logs from another machine.
        validate_record(run, verify_hashes=False, checkpoint_hashes=False)
        record = json.loads((run / "record.json").read_text())
        recipe = yaml.safe_load((run / "config.yaml").read_text())
        expected = {"model": {"initialization": "transformer_truncated_reference",
                              "qk_norm": True, "qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
                              "rms_norm_fp32": True, "num_layers": 15, "hidden_size": 896,
                              "num_heads": 14, "num_kv_heads": 14, "qk_head_dim": 64, "value_head_dim": 64,
                              "intermediate_size": 2304, "vocab_size": 32000, "max_seq_len": 2048},
                    "optim": {"total_steps": 7570, "warmup_steps": 455},
                    "data": {"batch_size": 256, "sequence_length": 2049},
                    "train": {"dtype": "bfloat16", "seed": 42}}
        if any(recipe.get(section, {}).get(key) != value
               for section, fields in expected.items() for key, value in fields.items()):
            raise ValueError(f"{run.name} is not part of the 200M/4B recipe")
        path = run / "metrics.jsonl"
        measurements = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        validation = sorted((row for row in measurements if row.get("event") == "validation"), key=lambda row: row["step"])
        if any(not math.isfinite(row[key]) for row in validation
               for key in ("validation_nll", "validation_perplexity")):
            raise ValueError(f"{run.name} has a nonfinite validation measurement")
        completion = json.loads((run / "completion.json").read_text()) if (run / "completion.json").exists() else None
        if record["status"] == "complete" and (completion is None or completion.get("final_step") != 7570):
            raise ValueError(f"{run.name} lacks a full-budget terminal completion")
        if completion is not None:
            terminal = next((row for row in validation if row["step"] == completion["final_step"]), None)
            if terminal is None or any(terminal[key] != completion[key]
                                       for key in ("validation_nll", "validation_perplexity")):
                raise ValueError(f"{run.name} completion disagrees with terminal validation")
        failures = [json.loads(path.read_text()) for path in sorted(run.glob("numerical_failure_step_*.json"))]
        rows.append({"run_id": run.name, "label": entry["label"], "status": record["status"],
                     "peak_lr": recipe["optim"]["learning_rate"], "validation": validation,
                     "terminal": completion, "latest": validation[-1] if validation else None,
                     "failure_step": max((failure["step"] for failure in failures), default=None)})
    return rows


def save(fig, name):
    fig.savefig(DIRECTORY / f"{name}.png", dpi=170, metadata={"Software": "functional-memory"})
    fig.savefig(DIRECTORY / f"{name}.pdf", metadata={"Creator": "functional-memory", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def regenerate():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.compression": 9})
    lr = yaml.safe_load((DIRECTORY / "lr_selection.yaml").read_text())
    pilots = collect(lr)
    measured_pilots = [row for row in pilots
                       if any(v["step"] <= lr["pilot_steps"] for v in row["validation"])]
    if len(measured_pilots) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for row in measured_pilots:
            measurements = [v for v in row["validation"] if v["step"] <= lr["pilot_steps"]]
            ax.plot([v["step"] for v in measurements], [v["validation_nll"] for v in measurements],
                    label=f"peak {row['peak_lr']:g} ({row['status']})")
        ax.set(xlabel="Optimizer step", ylabel="Held-out negative log likelihood",
               title="200M / 4B recipe: LR screening")
        ax.legend()
        ax.grid(alpha=.2)
        save(fig, "learning_rates")
    else:
        for suffix in ("png", "pdf"):
            (DIRECTORY / f"learning_rates.{suffix}").unlink(missing_ok=True)
    selection = DIRECTORY / "selection.yaml"
    comparison = collect(yaml.safe_load(selection.read_text())) if selection.exists() else []
    columns = ("run_id", "label", "status", "peak_lr", "failure_step", "latest_step", "latest_validation_nll",
               "latest_validation_perplexity", "terminal_validation_nll", "terminal_validation_perplexity")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in [*pilots, *comparison]:
        latest, terminal = row["latest"] or {}, row["terminal"] or {}
        writer.writerow({**{k: row[k] for k in columns[:5]}, "latest_step": latest.get("step", ""),
                         **{f"latest_{k}": latest.get(k, "") for k in ("validation_nll", "validation_perplexity")},
                         **{f"terminal_{k}": terminal.get(k, "") for k in ("validation_nll", "validation_perplexity")}})
    (DIRECTORY / "summary.csv").write_text(stream.getvalue())
    if comparison:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        positions = list(range(len(comparison)))
        labels = []
        for x, row in zip(positions, comparison):
            labels.append(row["label"])
            terminal, latest = row["terminal"], row["latest"]
            if terminal and row["status"] == "complete":
                ax.scatter(x, terminal["validation_perplexity"], color="#245b8b", marker="o")
            elif latest:
                ax.scatter(x, latest["validation_perplexity"], color="#c66b2a", marker="x")
            ax.text(x, .02, row["status"], transform=ax.get_xaxis_transform(), rotation=90, fontsize=8)
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set(ylabel="Validation perplexity", title="200M / 4B attention comparison\nDots: completed; crosses: latest incomplete measurements")
        ax.grid(axis="y", alpha=.2)
        save(fig, "attention_comparison")


if __name__ == "__main__":
    regenerate()
