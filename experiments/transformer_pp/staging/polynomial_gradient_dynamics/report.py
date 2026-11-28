"""Regenerate polynomial gradient-dynamics results from pinned compact run measurements."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
ARMS = {
    "static_native": "Learned static, trained coefficients",
    "static_reset": "Learned static, Taylor coefficients",
    "taylor_native": "Fixed Taylor, trained with Taylor coefficients",
}


def paired_loss(first: list[float], second: list[float]) -> tuple[float, float, float]:
    differences = [a - b for a, b in zip(first, second, strict=True)]
    mean = statistics.mean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    half_width = statistics.NormalDist().inv_cdf(.975) * sem
    return mean, mean - half_width, mean + half_width


def reset_tables(rows: list[dict], expected_sequences: int) -> str:
    arms = {row["condition"]: row for row in rows}
    if set(arms) != set(ARMS) or len(arms) != len(rows):
        raise ValueError("Expected one measurement per evaluation condition")
    lines = ["| Frozen checkpoint and coefficients | NLL (nats/token) | PPL |", "| --- | ---: | ---: |"]
    for arm, label in ARMS.items():
        row = arms[arm]
        losses = row["sequence_nll"]
        if len(losses) != expected_sequences or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"Incomplete or nonfinite sequence losses in {arm}")
        if not math.isclose(statistics.mean(losses), row["nll"], abs_tol=2e-6):
            raise ValueError(f"Sequence losses disagree with the full-set loss in {arm}")
        lines.append(f"| {label} | {row['nll']:.6f} | {row['perplexity']:.4f} |")

    lines.extend(["", "| Difference | Δ NLL (nats/token) | Paired 95% interval | Δ PPL | Packed rows with higher loss |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    contrasts = [
        ("Reset minus trained coefficients on the static backbone", "static_reset", "static_native"),
        ("Static-trained minus Taylor-trained backbone, both using Taylor", "static_reset", "taylor_native"),
    ]
    for label, first, second in contrasts:
        delta, lo, hi = paired_loss(arms[first]["sequence_nll"], arms[second]["sequence_nll"])
        delta_ppl = arms[first]["perplexity"] - arms[second]["perplexity"]
        worse = sum(a > b for a, b in zip(arms[first]["sequence_nll"], arms[second]["sequence_nll"], strict=True))
        lines.append(f"| {label} | {delta:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {delta_ppl:+.4f} "
                     f"| {worse:,}/{expected_sequences:,} |")
    return "\n".join(lines)


def training_table(runs: dict, root: Path) -> str:
    lines = ["| Completed training run | Run ID | Validation-best PPL (step) | Terminal PPL (step) |",
             "| --- | --- | ---: | ---: |"]
    for role, label in (
        ("learned_static_parent", "Static, original"),
        ("static_dynamics", "Static, gradient diagnostic"),
        ("query_parent", "Query-conditioned, original"),
        ("query_dynamics", "Query-conditioned, gradient diagnostic"),
        ("fixed_taylor_parent", "Fixed Taylor control"),
        ("sos_parent", "Exp-prefactored query SoS, original"),
    ):
        run = runs[role]
        best, last = (json.loads((root / run["sources"][name]["path"]).read_text())
                      for name in ("best_validation.json", "completion.json"))
        run_id = run["run_id"]
        lines.append(f"| {label} | [{run_id}](../../runs/{run_id}/) "
                     f"| {best['validation_perplexity']:.4f} ({best['step']:,}) "
                     f"| {last['validation_perplexity']:.4f} ({last['final_step']:,}) |")
    return "\n".join(lines)


def coefficient_snapshots(run: dict, root: Path, steps: list[int]) -> dict[int, dict]:
    snapshots = {}
    with (root / run["sources"]["metrics.jsonl"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            step = row["step"]
            if step in steps:
                if step in snapshots:
                    raise ValueError(f"Duplicate coefficient measurement at step {step}")
                snapshots[step] = row["attention_diagnostics"]["coefficients"]
    if set(snapshots) != set(steps):
        raise ValueError(f"Missing coefficient measurements for {run['run_id']}")
    return snapshots


def dynamics_table(runs: dict, root: Path, steps: list[int], orders: list[int]) -> str:
    static = [coefficient_snapshots(runs[role], root, steps)
              for role in ("learned_static_parent", "static_dynamics")]
    query = coefficient_snapshots(runs["query_dynamics"], root, steps)
    header = ["Update", *(f"Static $u_{r}$, original / repeat" for r in orders),
              "Query median mean shift", "Query median within-sequence SD"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---:"] * len(header)) + " |"]
    for step in steps:
        cells = [f"{step:,}"]
        for r in orders:
            means = [statistics.mean(head[r] for layer in run[step]["raw"] for head in layer) for run in static]
            cells.append(" / ".join(f"{mean:.3f}" for mean in means))
        summary = query[step]["effective_summary"]
        for statistic in ("mean", "std"):
            values = [value * math.factorial(r) for layer in summary[statistic] for head in layer
                      for r, value in enumerate(head)]
            if statistic == "mean":
                values = [abs(value - 1) for value in values]
            cells.append(f"{statistics.median(values):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generated_tables() -> str:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    settings = selection["settings"]
    runs = {entry["role"]: entry for entry in selection["runs"]}
    metrics = ROOT / runs["paired_evaluation"]["sources"]["metrics.jsonl"]["path"]
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    return "\n\n".join([
        "### Training perplexity", training_table(runs, ROOT),
        "### How the coefficients moved",
        "For each static checkpoint, $u_r=r!c_r$ initially equals one. The table averages each selected "
        "order over all 210 layer/head pairs. For the query diagnostic, each layer, head and order "
        "has a mean and standard deviation of $r!c_r(x)$ across the 2,048 positions in the same "
        "held-out sequence. The last two columns are the median absolute departure of those means "
        "from one and the median of those standard deviations.",
        dynamics_table(runs, ROOT, settings["training_steps"], settings["static_orders"]),
        "### Frozen Taylor reset", reset_tables(rows, settings["packed_sequences"]),
    ])


def main() -> None:
    report = HERE / "README.md"
    before, after = report.read_text().split(BEGIN)
    _, after = after.split(END)
    report.write_text(f"{before}{BEGIN}\n\n{generated_tables()}\n\n{END}{after}")


if __name__ == "__main__":
    main()
