"""Write home.json, the home page's reference-model tables, from the Transformer++ recipe.

    python3 site/data/home.py

Every value comes from experiments/transformer_pp/recipe.yaml, except the parameter count, which is
the normalized softmax reference run's own record. The fixed labels (SwiGLU, FineWeb-Edu, the Llama 2
tokenizer, AdamW with cosine decay) are checked against the recipe where it names them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from cells import text
from polynomial import ROOT, RUNS

RECIPE = ROOT / "experiments/transformer_pp/recipe.yaml"
SOFTMAX = "r9040f12f3d80"  # the reference recipe with normalized softmax, at 1× Chinchilla


def power(value: float) -> str:
    """0.001 as 10<sup>−3</sup>."""
    exponent = round(math.log10(value))
    assert abs(value - 10.0 ** exponent) < 1e-12 * value, value
    return f"10<sup>−{-exponent}</sup>"


def tables(recipe: dict, parameters: int) -> list[dict]:
    model, data, optim = recipe["model"], recipe["data"], recipe["optim"]
    assert data["dataset_name"] == "HuggingFaceFW/fineweb-edu" and data["tokenizer_name"] == "NousResearch/Llama-2-7b-hf"
    assert model["qk_norm"] and model["qk_norm_type"] == "rms_norm" and model["qk_norm_head_scale"]
    assert model["num_kv_heads"] == model["num_heads"] and model["qk_head_dim"] == model["value_head_dim"]
    context = data["sequence_length"] - 1
    tokens = optim["total_steps"] * data["batch_size"] * context
    groups = (
        ("Model", (
            ("Layers × width", f"{model['num_layers']} × {model['hidden_size']}"),
            ("Heads", f"{model['num_heads']}, of width {model['qk_head_dim']}"),
            ("Feed-forward", f"SwiGLU, width {model['intermediate_size']:,}"),
            ("Vocabulary", f"{model['vocab_size']:,}"),
            ("Parameters", f"{parameters:,} (softmax)"),
        )),
        ("Attention", (
            ("Queries and keys", "RMSNorm"),
            ("Score scale", "learned per head, starts at 1"),
            ("Positions", f"RoPE, θ = {model['rope_theta']:,.0f}"),
        )),
        ("Data and budget", (
            ("Data", "FineWeb-Edu, Llama 2 tokenizer"),
            ("Batch", f"{data['batch_size']} sequences of {context:,} tokens"),
            ("Budget", f"1× Chinchilla: {optim['total_steps']:,} updates, {tokens / 1e9:.2f}B tokens"),
        )),
        ("Optimizer", (
            ("AdamW", f"β = ({optim['beta1']}, {optim['beta2']}), weight decay {optim['weight_decay']}"),
            ("Learning rate", f"{power(optim['learning_rate'])}, {optim['warmup_steps']} warmup updates, "
                              f"cosine to {power(optim['min_learning_rate'])}"),
            ("Gradient clipping", f"global norm {optim['grad_clip_norm']:g}"),
        )),
    )
    return [{"head": [name, ""], "rows": [[text(setting), text(value)] for setting, value in rows]}
            for name, rows in groups]


def main() -> None:
    recipe = yaml.safe_load(RECIPE.read_text())
    softmax = yaml.safe_load((RUNS / SOFTMAX / "config.yaml").read_text())["model"]
    assert softmax["attention_normalization"] and softmax["attention_degree"] is None
    assert all(softmax[key] == recipe["model"][key] for key in ("num_layers", "hidden_size", "intermediate_size"))
    parameters = json.loads((RUNS / SOFTMAX / "metadata.json").read_text())["parameter_count"]
    data = {"model": [{"key": "reference", "title": "Reference model", "layout": "pair",
                       "tables": tables(recipe, parameters)}]}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
