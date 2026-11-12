"""Compare explicitly selected 10B-token RBF runs with existing baselines."""
from pathlib import Path
import csv
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs


def read_json(path):
    return json.loads(path.read_text())


def collect():
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    settings = selection["comparison"]
    by_role = {entry["role"]: entry for entry in selection["runs"]}
    entries = [by_role[role] for role in settings["roles"]]
    verify_pinned_runs(ROOT, entries)
    reference_root = ROOT / "experiments/attention_scaling/runs" / by_role[settings["reference_role"]]["run_id"]
    reference = yaml.safe_load((reference_root / "config.yaml").read_text())
    reference_data = read_json(reference_root / "data_identity.json")
    reference_ppl = read_json(reference_root / "completion.json")["validation_perplexity"]
    rows = []
    for entry in entries:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((run / "config.yaml").read_text())
        record = load_run(run)
        metadata = read_json(run / "metadata.json")
        data = read_json(run / "data_identity.json")
        for key in ("train_manifest_sha256", "eval_manifest_sha256"):
            assert data[key] == reference_data[key], (entry["role"], key)
        for key, value in reference["data"].items():
            if key not in ("train_manifest", "eval_manifest"):
                assert config["data"][key] == value, (entry["role"], key)
        for key in ("learning_rate", "min_learning_rate", "weight_decay", "weight_decay_policy",
                    "beta1", "beta2", "grad_clip_norm", "warmup_steps", "total_steps"):
            assert config["optim"][key] == reference["optim"][key], (entry["role"], key)
        # Older fixed-kernel recipes predate coefficient-penalty options.
        # Only the learned-coefficient control needs that additional check.
        if entry["role"] == "learned_m4_head_rms_norm":
            assert config["optim"]["coefficient_penalty"] == 0.0
        for key in ("seed", "dtype", "eval_every", "eval_batches"):
            assert config["train"][key] == reference["train"][key], (entry["role"], key)
        for key in ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "qk_head_dim",
                    "value_head_dim", "intermediate_size", "vocab_size", "max_seq_len", "tie_embeddings"):
            assert config["model"][key] == reference["model"][key], (entry["role"], key)
        gdn = entry["group"] == "architecture_reference"
        expected_init = "gated_deltanet_reference" if gdn else "transformer_reference"
        assert config["model"]["initialization"] == expected_init
        if not gdn:
            assert config["model"]["qk_norm"] is True
        if entry["group"] in ("rbf", "same_output_scaling"):
            assert config["model"]["attention_normalization"] is False
            assert config["model"]["attention_output_scaling"] == "head_rms_norm"
        measurements = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        complete = record["status"] == "complete"
        if complete:
            value = read_json(run / "completion.json")
            step = value.get("step", value.get("final_step"))
            assert step == settings["terminal_step"]
            assert value["training_tokens"] == settings["training_tokens"]
            assert measurements[-1]["step"] == step
            assert measurements[-1]["validation_perplexity"] == value["validation_perplexity"]
        else:
            assert not (run / "completion.json").exists()
            value = measurements[-1]
            step = value["step"]
        ppl = value["validation_perplexity"]
        assert math.isfinite(ppl)
        rows.append({
            "role": entry["role"], "run_id": entry["run_id"], "label": entry["label"],
            "group": entry["group"], "status": record["status"],
            "metric_scope": "terminal" if complete else "latest_pinned_validation",
            "step": step, "training_tokens": value["training_tokens"],
            "parameter_count": value.get("parameter_count", metadata.get("parameter_count")),
            "validation_nll": value["validation_nll"], "validation_perplexity": ppl,
            "terminal_delta_ppl_vs_exp_head_rms_norm": ppl - reference_ppl if complete else "",
            "best_validation_step": read_json(run / "best_validation.json")["step"],
        })
    return rows


def render(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12.6, 8.4))
    fig.subplots_adjust(left=0.42, right=0.96, top=0.82, bottom=0.18)
    positions = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13]
    colors = {"same_output_scaling": "#2B5F9E", "rbf": "#00867E",
              "normalized": "#637084", "architecture_reference": "#895AB2"}
    for row, y in zip(rows, positions, strict=True):
        terminal = row["metric_scope"] == "terminal"
        color = colors[row["group"]]
        ax.scatter(row["validation_perplexity"], y, marker="o" if terminal else "D",
                   s=78, facecolors=color if terminal else "white", edgecolors=color,
                   linewidths=1.7, zorder=3)
        label = f"{row['validation_perplexity']:.3f}"
        if not terminal:
            label += "*"
        ax.annotate(label, (row["validation_perplexity"], y), xytext=(9, 0),
                    textcoords="offset points", va="center", color=color, weight="bold")
    ax.set_yticks(positions, [r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_ylim(14, -1)
    ax.set_xlim(13.32, 14.15)
    ax.set_xlabel("Held-out perplexity · lower is better", labelpad=12)
    ax.grid(axis="x", alpha=0.18)
    ax.axvline(rows[0]["validation_perplexity"], color="#2B5F9E", alpha=0.3, linestyle=":")
    fig.text(0.035, 0.955, "RBF pretraining versus previous baselines", fontsize=22, weight="bold")
    fig.text(0.035, 0.905, "10.0B tokens · about 200M parameters · same data, optimizer schedule and seed", fontsize=13)
    fig.text(0.035, 0.865, "Teal / blue: unnormalized + head RMSNorm. Gray: normalized, without output RMSNorm.", fontsize=11, color="#536173")
    incomplete = next((r for r in rows if r["metric_scope"] != "terminal"), None)
    note = "Every point is terminal at 10.0B tokens. RBF coefficients, centers and widths are fixed."
    if incomplete:
        note = f"* 20-RBF is incomplete: {incomplete['training_tokens']/1e9:.2f}B tokens. Every filled dot is terminal at 10.0B."
    fig.text(0.035, 0.095, note, fontsize=11)
    fig.text(0.035, 0.052, "One seed; hardware/backends differ. Gated DeltaNet is a different architecture with 5.5% more parameters.", fontsize=10.5, color="#536173")
    fig.savefig(HERE / "comparison.png", dpi=160, facecolor="white")
    fig.savefig(HERE / "comparison.pdf", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def cohort_summary():
    """Report the explicitly selected shorter-budget cohorts separately."""
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    by_role = {entry["role"]: entry for entry in selection["runs"]}
    rows = []
    for cohort in selection["cohorts"]:
        entries = [by_role[role] for role in cohort["roles"]]
        verify_pinned_runs(ROOT, entries)
        for entry in entries:
            run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
            config = yaml.safe_load((run / "config.yaml").read_text())
            assert config["model"]["initialization"] == cohort["initialization"]
            assert load_run(run)["status"] == "complete"
            terminal = read_json(run / "completion.json")
            best = read_json(run / "best_validation.json")
            measurements = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            assert measurements[-1]["validation_nll"] == terminal["validation_nll"]
            assert measurements[-1]["step"] == terminal.get("step", terminal.get("final_step"))
            rows.append({
                "cohort": cohort["label"], "role": entry["role"], "run_id": entry["run_id"],
                "label": entry["label"], "status": "complete", "metric_scope": "terminal",
                "training_tokens": terminal["training_tokens"],
                "terminal_validation_nll": terminal["validation_nll"],
                "terminal_validation_perplexity": terminal["validation_perplexity"],
                "best_validation_step": best["step"],
                "best_validation_perplexity": best["validation_perplexity"],
            })
    with (HERE / "cohorts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    table = ["<!-- BEGIN RBF COHORTS -->",
             "| Recipe | Coefficients | Run ID | Tokens (B) | Terminal PPL | Best PPL (step) |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        table.append(f"| {row['cohort']} | {row['label']} | `{row['run_id']}` | "
                     f"{row['training_tokens']/1e9:.3f} | {row['terminal_validation_perplexity']:.3f} | "
                     f"{row['best_validation_perplexity']:.3f} ({row['best_validation_step']:,}) |")
    table.append("<!-- END RBF COHORTS -->")
    path = HERE / "README.md"
    text = path.read_text()
    start = text.index(table[0])
    end = text.index(table[-1], start) + len(table[-1])
    path.write_text(text[:start] + "\n".join(table) + text[end:])


def main():
    rows = collect()
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(rows)
    table = ["<!-- BEGIN RBF COMPARISON -->", "| Mechanism | Run ID | Tokens (B) | Validation PPL | Metric |",
             "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        metric = "terminal" if row["metric_scope"] == "terminal" else "incomplete: latest pinned validation"
        table.append(f"| {row['label']} | `{row['run_id']}` | {row['training_tokens']/1e9:.3f} | {row['validation_perplexity']:.3f} | {metric} |")
    table.append("<!-- END RBF COMPARISON -->")
    path = HERE / "README.md"
    text = path.read_text()
    start = text.index("<!-- BEGIN RBF COMPARISON -->")
    end = text.index("<!-- END RBF COMPARISON -->", start) + len("<!-- END RBF COMPARISON -->")
    path.write_text(text[:start] + "\n".join(table) + text[end:])
    cohort_summary()
    print(f"Verified {len(rows)} selected runs; wrote comparison and summary.")


if __name__ == "__main__":
    main()
