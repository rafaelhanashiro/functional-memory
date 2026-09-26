"""Regenerate this record's summary, figure, and README results block from runs/.

The legacy counterparts (no QK-norm, decay on all parameters) are read as data from
`key_dimension_taylor` (m = 2, 4, 8, 16) and `odd_taylor_attention` (m = 1, 3), with
the softmax references of `qk_norm_attention` and `key_dimension_taylor`. The report
waits until every run has concluded, completed or failed numerically.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scaffolds.transformer.analysis import committed_records, save_figure, terminal_row, update_results_readme, write_csv

RECORD = Path(__file__).resolve().parents[2]
EXPERIMENTS = RECORD.parent
DEGREES = (1, 2, 3, 4, 8, 16)
QK_SOFTMAX = EXPERIMENTS / "qk_norm_attention/runs/200m-qk-softmax-seed0"
SOFTMAX = EXPERIMENTS / "key_dimension_taylor/runs/200m-softmax-seed0"
CURRENT, LEGACY, REFERENCE, GRID, INK = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#0b0b0b"
FIELDS = ["run", "attention_degree", "status", "terminal_validation_nll", "terminal_validation_perplexity",
          "best_step", "best_validation_perplexity", "failure_step", "legacy_status", "legacy_perplexity",
          "legacy_failure_step", "perplexity_change_vs_legacy", "qk_norm_softmax_perplexity",
          "perplexity_change_vs_qk_norm_softmax", "source_commit", "slurm_job_id", "hardware"]


def run_name(degree: int) -> str:
    return f"200m-m{degree:03d}-seed0"


def legacy(degree: int) -> Path:
    record = "odd_taylor_attention" if degree % 2 else "key_dimension_taylor"
    return EXPERIMENTS / record / "runs" / run_name(degree)


def failure_step(run_directory: Path):
    failures = sorted(Path(run_directory).glob("numerical_failure_step_*.json"))
    return json.loads(failures[-1].read_text())["step"] if failures else None


def concluded(run_directory: Path) -> bool:
    return (Path(run_directory) / "completion.json").exists() or failure_step(run_directory) is not None


def completed(record: Path = RECORD) -> bool:
    return all(concluded(record / "runs" / run_name(degree)) for degree in DEGREES)


def _outcome(directory: Path) -> dict:
    if (directory / "completion.json").exists():
        terminal = terminal_row(directory)
        return {"status": "complete", "failure_step": "", **terminal}
    metadata = json.loads((directory / "metadata.json").read_text())
    return {"status": "numerical_failure", "failure_step": failure_step(directory), "terminal_validation_nll": "",
            "terminal_validation_perplexity": "", "best_step": "", "best_validation_perplexity": "",
            "source_commit": metadata["source_commit"], "slurm_job_id": metadata.get("slurm_job_id", "")}


def rows(record: Path = RECORD) -> list[dict]:
    manifest = json.loads((record / "provenance/manifest.json").read_text())
    hardware = {run["name"]: run.get("hardware_summary", "") for run in manifest["runs"]}
    softmax = terminal_row(QK_SOFTMAX)["terminal_validation_perplexity"]
    result = []
    for degree in DEGREES:
        current, previous = _outcome(record / "runs" / run_name(degree)), _outcome(legacy(degree))
        both = current["status"] == previous["status"] == "complete"
        result.append({
            "run": run_name(degree), "attention_degree": degree, "hardware": hardware.get(run_name(degree), ""),
            **{key: current[key] for key in ("status", "terminal_validation_nll", "terminal_validation_perplexity",
                                             "best_step", "best_validation_perplexity", "failure_step",
                                             "source_commit", "slurm_job_id")},
            "legacy_status": previous["status"], "legacy_perplexity": previous["terminal_validation_perplexity"],
            "legacy_failure_step": previous["failure_step"],
            "perplexity_change_vs_legacy": (current["terminal_validation_perplexity"]
                                            - previous["terminal_validation_perplexity"]) if both else "",
            "qk_norm_softmax_perplexity": softmax,
            "perplexity_change_vs_qk_norm_softmax": (current["terminal_validation_perplexity"] - softmax
                                                     if current["status"] == "complete" else ""),
        })
    return result


def plot(record: Path, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True)
    references = ((committed_records(QK_SOFTMAX), "QK-norm softmax", "-"), (committed_records(SOFTMAX), "softmax", "--"))
    for axis, degree in zip(axes.flat, DEGREES):
        for records, color, label in ((committed_records(record / "runs" / run_name(degree)), CURRENT,
                                       "QK-norm, selective decay"),
                                      (committed_records(legacy(degree)), LEGACY, "legacy (no QK-norm, decay on all)")):
            if records:
                axis.plot([r["step"] / 1000 for r in records], [r["validation_perplexity"] for r in records],
                          color=color, linewidth=2, label=label)
        for records, label, style in references:
            axis.plot([r["step"] / 1000 for r in records], [r["validation_perplexity"] for r in records],
                      color=REFERENCE, linewidth=1, linestyle=style, label=label)
        for directory, color in ((record / "runs" / run_name(degree), CURRENT), (legacy(degree), LEGACY)):
            failed = failure_step(directory)
            if failed is not None:
                axis.axvline(failed / 1000, color=color, linewidth=0.8, linestyle=":")
                axis.annotate(f"fails @{failed:,}", (failed / 1000, 0.95), xycoords=("data", "axes fraction"),
                              xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="top")
        axis.set_title(f"m = {degree}", fontsize=10, color=INK)
        axis.set_yscale("log")
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Update (thousands)", fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Held-out perplexity (log)", fontsize=9)
    axes.flat[0].legend(frameon=False, fontsize=7.5)
    figure.tight_layout()
    save_figure(figure, figures / "learning_curves")
    plt.close(figure)


def readme_lines(table: list[dict]) -> list[str]:
    def outcome(ppl, failure):
        return f"{ppl:.4f}" if ppl != "" else f"failed @{failure:,}"

    lines = ["Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0, d_K = 64, "
             "under QK-norm and selective weight decay, against the legacy run of the same degree (no QK-norm, decay "
             "on all parameters) and the QK-norm softmax baseline.", "",
             "| m | Current defaults | Legacy | Δ vs legacy | Δ vs QK-norm softmax |", "|---:|---:|---:|---:|---:|"]
    for row in table:
        delta = f"{row['perplexity_change_vs_legacy']:+.4f}" if row["perplexity_change_vs_legacy"] != "" else "—"
        gap = (f"{row['perplexity_change_vs_qk_norm_softmax']:+.4f}"
               if row["perplexity_change_vs_qk_norm_softmax"] != "" else "—")
        lines.append(f"| {row['attention_degree']} | {outcome(row['terminal_validation_perplexity'], row['failure_step'])} "
                     f"| {outcome(row['legacy_perplexity'], row['legacy_failure_step'])} | {delta} | {gap} |")
    lines += ["", "[Summary CSV](summary.csv) · [Learning curves](figures/learning_curves.pdf)"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECORD, help="Directory receiving summary.csv, figures/, README.md")
    args = parser.parse_args()
    if not completed():
        print("pending: not every run has concluded")
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    table = rows()
    write_csv(output / "summary.csv", table, FIELDS)
    (output / "figures").mkdir(exist_ok=True)
    plot(RECORD, output / "figures")
    if output != RECORD and not (output / "README.md").exists():
        shutil.copy(RECORD / "README.md", output / "README.md")
    update_results_readme(output / "README.md", readme_lines(table))
    print(output / "summary.csv")


if __name__ == "__main__":
    main()
