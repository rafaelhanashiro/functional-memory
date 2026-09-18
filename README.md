# Functional memory experiments

Experiments are grouped by the question they investigate:

- [State rank](experiments/state_rank/README.md): tokenwise spectra and effective
  rank of attention states, including pretrained Llama/Gated DeltaNet and a
  matched random-versus-pretrained Llama comparison across every layer.
- [Key dimension](experiments/key_dimension/README.md): pretrain a fixed
  Transformer backbone while varying query/key width and measure perplexity.
  All eight widths from 2 through 256 completed the same 10B-token budget.
  [Results](results/key_dimension/README.md) ·
  [Retention provenance](results/key_dimension/provenance.json).

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
