# State-spectrum analysis

This scaffold measures tokenwise attention-state spectra, validates recurrence
and pretrained reproduction, and renders comparisons. It owns model-specific
collection and reusable analysis; studies select recipes and launch resources,
while result collections select runs and retain figures.

Sync an isolated environment in each worktree:

```bash
uv sync --project scaffolds/state_rank --locked --link-mode hardlink
```

`spectrum.py` collects a single pretrained layer. `compare_random.py` and
`compare_deltanet.py` prepare and execute pretrained-versus-random conditions.
Their `prepare_manifest` accepts a complete recipe mapping or a JSON recipe path;
`run_condition` validates its execution identity, input checksums, and prior
completed output before executing. Canonical manifests contain `run_directories`
and `run_config_sha256` mappings: each condition writes directly into its allocated
run directory and updates its authoritative record. A manifest without those
mappings retains the standalone workspace interface for historical reproduction.

`plot_comparison.py`, `unify_analysis.py`, and the spectrum plotting helpers derive
figures and summary tables from retained arrays. Temporary comparison workspaces
may reference canonical run directories without copying numerical archives.
Scientific rank definitions and tolerance checks stay in the collection and
analysis modules; launchers own cluster resources and runtime settings.

Run the CPU tests with:

```bash
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m pytest \
  -c scaffolds/state_rank/pyproject.toml scaffolds/state_rank/tests
```

Tests requiring GPU inference use the configured CUDA device and qualified pinned
model cache. A skipped GPU test is not hardware qualification.

## Frozen-head feature approximation

`head_capture.py` provides `capture_llama_heads(model, input_ids, heads)` for
FP32 post-RoPE queries, grouped-attention-matched keys/values, and native head
outputs. It checks the reconstructed causal softmax outputs against the model's
pre-output-projection activations. `prepare_wikitext_corpus` writes deterministic
document-disjoint token windows outside the repository; `load_token_corpus`
verifies both archive and per-window hashes. `cached_model_manifest` records the
pinned pretrained model and tokenizer file hashes. These helpers preserve the
pretrained model's score convention and never modify its weights.

`sparse_polynomial.py` provides a symmetric Taylor monomial basis and learned
dense, fixed-support, random-support, and query-dependent sparse feature maps.
The same feature identities are used on the query and key sides. Sparse queries
control which state rows are read; key writes remain dense. Selection budgets
include the constant feature. Its normalized-attention result exposes raw
denominators and validity, while its explicitly requested training-safe path
also returns the denominator penalty. Evaluation never silently repairs invalid
denominators.

Runtime qualification and timing records belong in `validation/`; scientific
recipes and launch resources belong to their study. These PyTorch helpers use
this scaffold's locked environment independently of Transformer pretraining.
