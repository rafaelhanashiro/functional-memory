"""Regenerate this record's summary, figure, and README results block from runs/.

Controls are read as data: the direct-coefficient normalized runs of
`qk_norm_learned_taylor` (same recipe, signed kernel) and the QK-norm softmax
baseline of `qk_norm_attention`. The report waits until every run has concluded,
either completed or failed numerically, and writes nothing before then.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from functools import partial
from scaffolds.transformer.analysis import load_selected_study, selected_run_directory

from scaffolds.transformer.analysis import verify_result_selection

from scaffolds.transformer.analysis import committed_records, save_figure, terminal_row, update_results_readme, write_csv

locate_run = partial(selected_run_directory, Path(__file__).resolve().parent)

OUTPUT = Path(__file__).resolve().parent
RECORD = OUTPUT.parents[1] / "experiments" / "nonnegative_learned_taylor"
EXPERIMENTS = RECORD.parent
DEGREES = (2, 4)
SOFTMAX = locate_run(EXPERIMENTS / "qk_norm_attention", "200m-qk-softmax-seed0")
NONNEGATIVE, SIGNED, REFERENCE, GRID, INK = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#0b0b0b"
FIELDS = ["run", "attention_degree", "status", "terminal_validation_nll", "terminal_validation_perplexity",
          "best_step", "best_validation_perplexity", "failure_step", "signed_control_failure_step",
          "qk_norm_softmax_perplexity", "perplexity_change_vs_qk_norm_softmax", "minimum_row_sum",
          "source_commit", "slurm_job_id", "hardware"]


def run_name(degree: int) -> str:
    return f"200m-qk-m{degree:03d}-nonnegative-seed0"


def signed_control(degree: int) -> Path:
    return locate_run(EXPERIMENTS / "qk_norm_learned_taylor", f"200m-qk-m{degree:03d}-normalized-seed0")


def failure_step(run_directory: Path):
    failures = sorted(Path(run_directory).glob("numerical_failure_step_*.json"))
    return json.loads(failures[-1].read_text())["step"] if failures else None


def concluded(run_directory: Path) -> bool:
    return (Path(run_directory) / "completion.json").exists() or failure_step(run_directory) is not None


def completed(record: Path = RECORD) -> bool:
    return all(concluded(locate_run(record, run_name(degree))) for degree in DEGREES)


def minimum_row_sum(records: list[dict]) -> float:
    return min(layer["min_abs_normalizer"] for record in records
               for layer in record["attention_diagnostics"]["layers"])


def rows(record: Path = RECORD) -> list[dict]:
    manifest = load_selected_study(Path(__file__).resolve().parent, record)
    hardware = {run["name"]: run.get("hardware_summary", "") for run in manifest["runs"]}
    softmax = terminal_row(SOFTMAX)["terminal_validation_perplexity"]
    result = []
    for degree in DEGREES:
        directory = locate_run(record, run_name(degree))
        records = committed_records(directory)
        row = {"run": run_name(degree), "attention_degree": degree, "hardware": hardware.get(run_name(degree), ""),
               "signed_control_failure_step": failure_step(signed_control(degree)),
               "qk_norm_softmax_perplexity": softmax, "minimum_row_sum": minimum_row_sum(records) if records else ""}
        if (directory / "completion.json").exists():
            terminal = terminal_row(directory)
            row.update(status="complete", failure_step="", **{key: terminal[key] for key in (
                "terminal_validation_nll", "terminal_validation_perplexity", "best_step",
                "best_validation_perplexity", "source_commit", "slurm_job_id")},
                perplexity_change_vs_qk_norm_softmax=terminal["terminal_validation_perplexity"] - softmax)
        else:
            metadata = json.loads((directory / "metadata.json").read_text())
            row.update(status="numerical_failure", failure_step=failure_step(directory), terminal_validation_nll="",
                       terminal_validation_perplexity="", best_step="", best_validation_perplexity="",
                       perplexity_change_vs_qk_norm_softmax="", source_commit=metadata["source_commit"],
                       slurm_job_id=metadata.get("slurm_job_id", ""))
        result.append(row)
    return result


def plot(record: Path, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(DEGREES), figsize=(9.6, 3.6), sharey=True)
    softmax = committed_records(SOFTMAX)
    for axis, degree in zip(axes, DEGREES):
        for records, color, label in (
                (committed_records(locate_run(record, run_name(degree))), NONNEGATIVE, "nonnegative (sum of squares)"),
                (committed_records(signed_control(degree)), SIGNED, "signed (direct coefficients)"),
                (softmax, REFERENCE, "QK-norm softmax")):
            if records:
                # The signed runs failed after one validation, so their points need markers to show.
                axis.plot([r["step"] / 1000 for r in records], [r["validation_perplexity"] for r in records],
                          color=color, linewidth=2, label=label, linestyle="--" if color == REFERENCE else "-",
                          marker="o" if color == SIGNED else None, markersize=6)
        failed = failure_step(signed_control(degree))
        if failed is not None:
            axis.axvline(failed / 1000, color=SIGNED, linewidth=0.8, linestyle=":")
            axis.annotate(f"signed fails @{failed:,}", (failed / 1000, 0.97), xycoords=("data", "axes fraction"),
                          xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="top")
        axis.set_title(f"m = {degree}", fontsize=10, color=INK)
        axis.set_yscale("log")
        axis.set_xlabel("Update (thousands)", fontsize=9)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("Held-out perplexity (log scale)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure, figures / "learning_curves")
    plt.close(figure)


def readme_lines(table: list[dict]) -> list[str]:
    lines = ["Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0, d_K = 64, "
             "per-head factors and QK-norm scales, against the signed direct-coefficient runs of the same recipe "
             "and the QK-norm softmax baseline. The minimum row sum is over every validation's diagnostic sequence.",
             "", "| m | Status | Terminal PPL | Δ vs QK-norm softmax | Signed control | Minimum row sum |",
             "|---:|---|---:|---:|---|---:|"]
    for row in table:
        ppl = f"{row['terminal_validation_perplexity']:.4f}" if row["status"] == "complete" else f"failed @{row['failure_step']:,}"
        delta = f"{row['perplexity_change_vs_qk_norm_softmax']:+.4f}" if row["status"] == "complete" else "—"
        signed = f"failed @{row['signed_control_failure_step']:,}" if row["signed_control_failure_step"] else "—"
        minimum = f"{row['minimum_row_sum']:.3g}" if row["minimum_row_sum"] != "" else "—"
        lines.append(f"| {row['attention_degree']} | {row['status']} | {ppl} | {delta} | {signed} | {minimum} |")
    lines += ["", "[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf)"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Directory receiving summary.csv, figures, and README.md")
    args = parser.parse_args()
    verify_result_selection(Path(__file__).resolve().parent)
    if not completed():
        print("pending: not every run has concluded")
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    table = rows()
    write_csv(output / "summary.csv", table, FIELDS)
    plot(RECORD, output)
    if output != OUTPUT and not (output / "README.md").exists():
        shutil.copy(OUTPUT / "README.md", output / "README.md")
    update_results_readme(output / "README.md", readme_lines(table))
    print(output / "summary.csv")


if __name__ == "__main__":
    main()
