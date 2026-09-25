"""Regenerate this record's summary, figure, and README results block from runs/.

The control is the committed old-policy softmax run of `key_dimension_taylor`
(AdamW decay on every parameter), read as data. The new run differs only in the
scaffold's canonical decay policy, which exempts the token embedding, RMSNorm
scales, and biases.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scaffolds.transformer.analysis import committed_records, save_figure, terminal_row, update_results_readme, write_csv

RECORD = Path(__file__).resolve().parents[2]
RUN = "200m-softmax-seed0"
CONTROL = RECORD.parent / "key_dimension_taylor" / "runs" / RUN
POLICY, CONTROL_POLICY = "exclude_embeddings_norms_biases_v1", "decay_all_parameters"
NEW, OLD, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
FIELDS = ["run", "weight_decay_policy", "parameters", "updates", "prediction_tokens", "terminal_validation_nll",
          "terminal_validation_perplexity", "best_step", "best_validation_nll", "best_validation_perplexity",
          "nll_change_vs_control", "perplexity_change_vs_control", "source_commit", "slurm_job_id", "hardware"]


def completed(record: Path = RECORD) -> bool:
    return (record / "runs" / RUN / "completion.json").exists()


def rows(record: Path = RECORD, control: Path = CONTROL) -> list[dict]:
    manifest = json.loads((record / "provenance/manifest.json").read_text())
    attempts = next((run for run in manifest["runs"] if run["name"] == RUN), {}).get("execution", {}).get("attempts", [])
    node = attempts[-1]["node"] if attempts else ""
    new, old = terminal_row(record / "runs" / RUN), terminal_row(control)
    result = []
    for row, policy, name in ((new, POLICY, RUN), (old, CONTROL_POLICY, f"key_dimension_taylor/{RUN}")):
        result.append({**row, "run": name, "weight_decay_policy": policy,
                       "hardware": f"{row['gpu_count']}x{row['gpu_type']} {node}".strip() if row is new
                       else f"{row['gpu_count']}x{row['gpu_type']}",
                       "nll_change_vs_control": (row["terminal_validation_nll"] - old["terminal_validation_nll"]
                                                 if row is new else ""),
                       "perplexity_change_vs_control": (row["terminal_validation_perplexity"]
                                                        - old["terminal_validation_perplexity"] if row is new else "")})
    return result


def plot_learning_curves(record: Path, control: Path, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new, old = committed_records(record / "runs" / RUN), committed_records(control)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                         gridspec_kw={"height_ratios": (3, 1.6)})
    for records, color, label in ((old, OLD, "decay on all parameters (control)"), (new, NEW, "selective decay")):
        top.plot([r["step"] for r in records], [r["validation_perplexity"] for r in records],
                 color=color, linewidth=2, label=label)
    top.set_ylabel("Held-out perplexity")
    top.set_yscale("log")
    top.legend(frameon=False, fontsize=9)
    by_step = {r["step"]: r["validation_perplexity"] for r in old}
    steps = [r["step"] for r in new if r["step"] in by_step]
    bottom.axhline(0, color=INK, linewidth=0.8)
    bottom.plot(steps, [next(r["validation_perplexity"] for r in new if r["step"] == s) - by_step[s] for s in steps],
                color=NEW, linewidth=2)
    bottom.set_ylabel("PPL difference\n(selective − control)")
    bottom.set_xlabel("Optimizer updates")
    for axis in (top, bottom):
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    save_figure(figure, figures / "learning_curves")
    plt.close(figure)


def readme_lines(table: list[dict]) -> list[str]:
    new, old = table
    lines = ["# Softmax baseline under selective weight decay", "",
             f"Terminal held-out perplexity at {new['updates']:,} updates ({new['prediction_tokens']:,} prediction "
             "tokens), seed 0, 200M-parameter softmax reference. The control is the retained old-policy run; both "
             "use the same recipe, data order, and evaluation set.", "",
             "| Weight-decay policy | Terminal NLL | Terminal PPL | Best step | Best PPL | Hardware |",
             "|---|---:|---:|---:|---:|---|"]
    for row in (new, old):
        lines.append(f"| {row['weight_decay_policy']} | {row['terminal_validation_nll']:.6f} | "
                     f"{row['terminal_validation_perplexity']:.4f} | {row['best_step']:,} | "
                     f"{row['best_validation_perplexity']:.4f} | {row['hardware']} |")
    lines += ["", f"Selective decay changes terminal NLL by {new['nll_change_vs_control']:+.6f} nats and PPL by "
              f"{new['perplexity_change_vs_control']:+.4f} ({100 * new['perplexity_change_vs_control'] / old['terminal_validation_perplexity']:+.2f}%).", "",
              "[Summary CSV](summary.csv) · [Learning curves](figures/learning_curves.pdf)", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECORD, help="Directory receiving summary.csv, figures/, README.md")
    args = parser.parse_args()
    if not completed():
        print(f"pending: {RECORD / 'runs' / RUN / 'completion.json'} does not exist yet")
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    table = rows()
    write_csv(output / "summary.csv", table, FIELDS)
    (output / "figures").mkdir(exist_ok=True)
    plot_learning_curves(RECORD, CONTROL, output / "figures")
    if output != RECORD and not (output / "README.md").exists():
        shutil.copy(RECORD / "README.md", output / "README.md")
    update_results_readme(output / "README.md", readme_lines(table))
    print(output / "summary.csv")


if __name__ == "__main__":
    main()
