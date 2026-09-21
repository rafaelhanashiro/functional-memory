# Functional memory experiments

Experiments are grouped by the question they investigate:

- [State rank](experiments/state_rank/README.md): tokenwise spectra and effective
  rank of attention states, including pretrained Llama/Gated DeltaNet and
  matched random-versus-pretrained comparisons across every layer.
  [Rank comparison](results/state_rank/unified_comparison/REPORT.md) ·
  [Shared depth comparison](results/state_rank/unified_comparison/depth_counts.png) ·
  [Llama plots](results/state_rank/unified_comparison/llama/README.md) ·
  [Gated DeltaNet plots](results/state_rank/unified_comparison/gated-deltanet/README.md).
- [Key dimension](experiments/key_dimension/README.md): pretrain a fixed
  Transformer backbone while varying query/key width and measure perplexity.
  All eight widths from 2 through 256 completed the same 10B-token budget.
  [Results](results/key_dimension/README.md) ·
  [Retention provenance](results/key_dimension/provenance.json).
- [Polynomial attention](experiments/polynomial_attention/README.md): train the
  same 200M-reference backbone from scratch while varying the Taylor degree of
  the exponential attention kernel, with a matched exact-softmax control.
  All six variants completed the same 10B-token budget.
  [Results](results/polynomial_attention/README.md) ·
  [Retention provenance](results/polynomial_attention/provenance.json).

Each family owns its code, tests, and scientific configuration. The root
PyTorch environment serves state-rank experiments; each training family has
an independent locked JAX environment. Results live under `results/`; existing
rank results retain their original locations and provenance.

The main write-up is `overleaf/iclr/iclr2027_conference.tex`.

## State-rank checks

```bash
uv sync --frozen
uv run --frozen python -m pytest
```

See each experiment README for training, evaluation, and reproduction commands.
