"""Render the study's canonical recipe as a deterministic documentation figure.

Run from any working directory; all numeric recipe settings come from recipe.yaml.
The diagram describes future-run defaults, not measurements from historical runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import yaml


STUDY = Path(__file__).resolve().parent
INK = "#183149"
MUTED = "#4d6274"
BORDER = "#d5dfe8"


def reference_parameter_count(model: dict) -> int:
    """Count this bias-free softmax reference without allocating its weights."""
    hidden, layers = model["hidden_size"], model["num_layers"]
    heads, kv_heads = model["num_heads"], model["num_kv_heads"]
    key_dim, value_dim = model["qk_head_dim"], model["value_head_dim"]
    embedding = model["vocab_size"] * hidden * (1 if model["tie_embeddings"] else 2)
    attention = hidden * (heads * key_dim + kv_heads * (key_dim + value_dim) + heads * value_dim)
    feed_forward = 3 * hidden * model["intermediate_size"]
    qk_gains = layers * 2 * key_dim
    head_scales = layers * heads if model["qk_norm_head_scale"] else 0
    return embedding + layers * (attention + feed_forward + 2 * hidden) + hidden + qk_gains + head_scales


def wrap(value: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width, break_long_words=False, break_on_hyphens=False)
                     for line in value.splitlines())


def render(recipe: Path, output: Path) -> None:
    data = yaml.safe_load(recipe.read_text())
    model, optim, train, packed = (data[key] for key in ("model", "optim", "train", "data"))
    assert model["qk_norm"] and model["qk_norm_type"] == "rms_norm"
    assert model["qk_norm_head_scale"] and model["qk_norm_sharing"] == "per_layer"
    assert model["attention_normalization"] and model["attention_degree"] is None
    assert model["initialization"] == "transformer_truncated_reference"
    length = packed["sequence_length"] - 1
    tokens_per_step = packed["batch_size"] * length
    total_tokens = optim["total_steps"] * tokens_per_step
    head_scales = model["num_layers"] * model["num_heads"]
    global_batch, microbatch = packed["batch_size"], train["micro_batch_size"]
    assert global_batch % microbatch == 0

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "svg.hashsalt": "transformer-200m-4b-recipe",
    })
    fig = plt.figure(figsize=(14, 15), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def text(x, y, value, **kwargs):
        return ax.text(x, y, value, transform=ax.transAxes, **kwargs)

    def card(x, y, width, height, title, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.004,rounding_size=0.007",
                                   linewidth=1.05, edgecolor=color, facecolor="white", transform=ax.transAxes))
        ax.add_patch(Rectangle((x + .001, y + height - .034), width - .002, .033,
                               linewidth=0, facecolor=fill, transform=ax.transAxes))
        text(x + .014, y + height - .017, title, fontsize=14, weight="bold", va="center")

    def table(x, y, width, height, title, rows, *, color="#80a8c5", fill="#ecf4fa"):
        card(x, y, width, height, title, color, fill)
        split = x + width * .37
        prepared = [(wrap(label, 24), wrap(value, 44)) for label, value in rows]
        weights = [max(a.count("\n"), b.count("\n")) + 1 + .45 for a, b in prepared]
        scale = (height - .042) / sum(weights)
        cursor = y + height - .038
        ax.plot([split, split], [y + .004, cursor], color=BORDER, lw=.65, transform=ax.transAxes)
        for index, ((label, value), weight) in enumerate(zip(prepared, weights, strict=True)):
            row_height = weight * scale
            if index % 2 == 0:
                ax.add_patch(Rectangle((x + .004, cursor - row_height), width - .008, row_height,
                                       facecolor="#f7fafc", linewidth=0, transform=ax.transAxes))
            for tx, content in ((x + .012, label), (split + .012, value)):
                text(tx, cursor - row_height / 2, content, fontsize=10.2, va="center", linespacing=1.35)
            ax.plot([x + .004, x + width - .004], [cursor - row_height, cursor - row_height],
                    color=BORDER, lw=.55, transform=ax.transAxes)
            cursor -= row_height

    ax.add_patch(FancyBboxPatch((.035, .917), .93, .06, boxstyle="round,pad=0.006,rounding_size=0.008",
                               facecolor="#183b59", linewidth=0, transform=ax.transAxes))
    text(.5, .954, "Transformer++ recipe", fontsize=26, weight="bold", ha="center", color="white")
    text(.5, .930, "Canonical reference • QK RMSNorm with learned per-head score scaling",
         fontsize=13.2, ha="center", color="white")

    card(.035, .722, .93, .169, "QK normalization and head scaling", "#3a96ad", "#e7f5f7")
    text(.054, .838, "Q path", fontsize=11.6, weight="bold")
    text(.153, .838, "Q projection  →  RMSNorm + Q channel gains  →  × α  →  RoPE", fontsize=12.2)
    text(.054, .808, "K path", fontsize=11.6, weight="bold")
    text(.153, .808, "K projection  →  RMSNorm + K channel gains  →  RoPE", fontsize=12.2)
    text(.054, .773, r"$s_{t,s,h}=\widetilde q_t^{\mathsf{T}}\widetilde k_s/\sqrt{d_K}$"
         r"     $\alpha_{\ell,h}=\exp(\gamma_{\ell,h})$,  $\gamma_{\ell,h}=0$ initially",
         fontsize=14.3)
    text(.054, .742,
         f"Separate {model['qk_head_dim']}-channel Q/K gain vectors per layer, shared across heads; initialized at 1.\n"
         f"{head_scales} learned head scales, shared across tokens and sequences; initialized at 1. No decay on either.",
         fontsize=10.4, va="center", linespacing=1.55, color=MUTED)

    table(.035, .420, .445, .279, "Model architecture", [
        ("Total parameters", f"{reference_parameter_count(model):,} (softmax reference; approximately 200M)"),
        ("Layers / model width", f"{model['num_layers']} / {model['hidden_size']}"),
        ("Query / KV heads", f"{model['num_heads']} / {model['num_kv_heads']}"),
        ("QK / value head width", f"{model['qk_head_dim']} / {model['value_head_dim']}"),
        ("FFN width / activation", f"{model['intermediate_size']:,} / SwiGLU"),
        ("Reference attention", "Causal softmax (MHA)"),
        ("Position encoding", f"RoPE, theta = {model['rope_theta']:,.0f}"),
        ("Block normalization", f"Pre-RMSNorm, epsilon = {model['rms_norm_eps']:g}"),
        ("Vocabulary / context", f"{model['vocab_size']:,} / {model['max_seq_len']:,}"),
        ("Embeddings", "Untied input embedding and LM head"),
        ("Linear biases / dropout", f"None / {model['dropout']}"),
    ])
    table(.035, .244, .445, .154, "Weight initialization", [
        ("Base distribution", "Gaussian sigma = 0.02, truncated to [-0.04, 0.04]"),
        ("Truncation", "No variance compensation"),
        ("Embeddings / Q,K,V / FFN up / LM head", "Draw from the base distribution"),
        ("Attention output / FFN down", f"Base draw × 1/sqrt({2 * model['num_layers']})"),
        ("RMSNorm gains", "1.0; head log scales start at 0.0"),
    ], color="#c6a350", fill="#fcf5e6")
    table(.035, .040, .445, .181, "Data and evaluation", [
        ("Dataset", "FineWeb-Edu, shuffled and EOS-packed"),
        ("Tokenizer", "Llama 2; 32K vocabulary"),
        ("Objective", "Next-token cross-entropy"),
        ("Model / shuffle seed", f"{train['seed']} / {packed['shuffle_seed']}"),
        ("Validation tokens", f"{train['eval_batches'] * tokens_per_step:,}"),
        ("Validation cadence", f"Every {train['eval_early_every']} updates through {train['eval_early_until']:,}; then every {train['eval_every']}; also terminal"),
        ("Retained checkpoints", "During training: latest + best; after completion: validation best"),
    ], color="#a591c7", fill="#f2edf9")

    table(.515, .475, .45, .224, "Training budget and execution", [
        ("Reference token budget", f"{total_tokens:,} (1× Chinchilla); other budgets per run"),
        ("Reference updates", f"{optim['total_steps']:,}; runs change only this for another budget"),
        ("Prediction context", f"{length:,} tokens"),
        ("Packed record length", f"{packed['sequence_length']:,} tokens → {length:,} predictions"),
        ("Global batch", f"{global_batch} sequences"),
        ("Tokens per update", f"{tokens_per_step:,}"),
        ("Microbatch / accumulation", f"{microbatch} sequences / {global_batch // microbatch} microsteps"),
        ("Default allocation", "1 GPU per run; independent jobs"),
        ("Scheduling", "Use standard + long allowances first; sweep only for overflow"),
    ], color="#75bca1", fill="#eaf7ef")
    table(.515, .175, .45, .277, "Optimization", [
        ("Optimizer", "AdamW"),
        ("Peak / minimum LR", f"{optim['learning_rate']:g} / {optim['min_learning_rate']:g}"),
        ("LR schedule", "Linear warmup → cosine decay"),
        ("Warmup", f"{optim['warmup_steps']} updates"),
        ("Betas / epsilon", f"{optim['beta1']} / {optim['beta2']} / 1e-8"),
        ("Weight decay", f"{optim['weight_decay']} on eligible weights"),
        ("Excluded from decay", "Input embeddings, normalization gains, head log scales, biases, attention coefficients"),
        ("LM head", "Untied; weight decay applies"),
        ("Gradient clipping", f"Global norm {optim['grad_clip_norm']}"),
        ("Precision", "BF16 compute; FP32 master parameters, optimizer state, normalization statistics"),
    ], color="#c9879b", fill="#fbebf0")
    card(.515, .040, .45, .113, "Attention variants and scope", "#80a8c5", "#ecf4fa")
    text(.529, .082,
         "Unnormalized variants use per-head output RMSNorm.\n"
         "Polynomial coefficients retain factorial scaling: cᵣ = uᵣ / r!.\n"
         "The established learning rate is retained. Historical run\n"
         "recipes and checkpoint resumptions are unchanged.",
         fontsize=10.4, va="center", linespacing=1.55)

    text(.5, .017, "Source: experiments/transformer_pp/recipe.yaml • Regenerate with plot_recipe.py",
         ha="center", fontsize=9, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white", metadata={"Software": "functional-memory recipe renderer"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=STUDY / "recipe.yaml")
    parser.add_argument("--output", type=Path, default=STUDY / "recipe.png")
    args = parser.parse_args()
    render(args.recipe, args.output)


if __name__ == "__main__":
    main()
