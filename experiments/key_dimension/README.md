# Query/key dimension and perplexity

This experiment uses the shared Transformer implementation in
[`experiments/shared_transformer`](../shared_transformer/README.md). Its configuration,
sweep, preflight, plots, and retention policy stay in this experiment directory.
Use the shared locked environment for new runs; reproduce historical runs from
the source revision and dependency lock recorded in their provenance.

The retained sweep contains eight completed decoder-only Transformer runs
with a fixed backbone and per-head query/key widths
**2, 4, 8, 16, 32, 64, 128, 256**. The primary endpoint is terminal held-out
perplexity after a shared training-token budget. All runs use seed 0 and the
same data order; this sweep does not estimate variation across independent
seeds. The width-256 run extends the original seven-run sweep with the same
recipe and data.

## Model and training recipe

The backbone has 10 layers, residual width 1024, 16 full-MHA heads, value width
64, SwiGLU intermediate width 3008, and untied 32,000-token embeddings. It uses
RMSNorm (epsilon `1e-6`), RoPE (base 10,000), no dropout, and OFM's Flax NNX
initialization. BF16 computation uses FP32 parameters and optimizer state.

| Query/key width | Trainable parameters | Terminal PPL |
| ---: | ---: | ---: |
| 2 | 179,590,144 | 22.3489 |
| 4 | 180,245,504 | 18.8928 |
| 8 | 181,556,224 | 16.7053 |
| 16 | 184,177,664 | 15.5764 |
| 32 | 189,420,544 | 14.9853 |
| 64 | 199,906,304 | 14.6898 |
| 128 | 220,877,824 | 14.3776 |
| 256 | 262,820,864 | 14.1736 |

This is a **200M-reference family**, not a parameter-matched comparison. The
formula is `178934784 + 327680 * qk_head_dim`. Value/output projection sizes,
depth, MLP width, and head count remain fixed. Standard RoPE frequencies are
defined at each model's true query/key width. Q/K/V are zero-padded only for
the fused attention call, after RoPE; attention scaling uses the true width
and the output is sliced back to 64 value channels. Increasing query/key
width also increases parameter count, so the perplexity differences do not
isolate width independently of model size.

Each run trains for 19,074 optimizer updates with a global batch of 256
sequences and 2048 prediction positions, or **10,000,269,312 prediction
tokens**. A global microbatch of 64 supplies four accumulation steps. AdamW
uses peak/minimum learning rates `3e-4`/`3e-5`, a 2,000-step linear warmup and
cosine decay, betas `(0.9, 0.95)`, weight decay `0.1` on all parameters, and
global gradient clipping at `1.0` after accumulation/device averaging.

Evaluation uses the same first 1,792 held-out packed sequences every 1,000
updates and at completion: **3,670,016 prediction tokens**. Perplexity is the
exponential of mean token negative log likelihood, not an average of sequence
perplexities. Terminal results are compared at the common token budget;
validation-best values are recorded separately. Checkpoints are saved every
500 steps and at validation. During training, the latest checkpoint supports
exact resume and the validation-best checkpoint is preserved. Retained runs
keep only the validation-best checkpoint. For all eight runs, this is also
the terminal checkpoint at step **19074**. Terminal metrics remain in the
results independently of checkpoint selection.

## Source and data provenance

The local scaffold is adapted from OFM commit
`8977f529e83efc688daf96cdcfd365ebbce2afa8`, especially its canonical
`configs/experiments/transformer_250m.yaml`. It uses family-qualified imports
and does not import the sibling checkout's code at runtime.

The source corpus is read-only at
`/home/rafah/projects/ofm/datasets/fineweb_12b/`. Its train/eval manifests
record source revisions, tokenizer revisions, shard checksums, and the
document-level train/eval partition. The existing 12B prediction-token corpus
is sufficient for this sweep. Training consumes packed 2049-token records as
2048 inputs and shifted next-token targets. Dataset identity and the exact
data cursor are checked when resuming.

The architecture is also informed by [Hyperloop Transformers](https://arxiv.org/pdf/2604.21254).
This experiment follows OFM's recipe and held-out subset; its perplexities
are not direct reproductions of that paper's reported results.

## Environment and tests

Run commands from the repository root. The JAX environment is separate from
the root PyTorch environment because their CUDA-library pins differ.

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project experiments/shared_transformer --locked --extra cuda
uv run --project experiments/shared_transformer --locked --extra cuda python -m pytest \
  -c experiments/shared_transformer/pyproject.toml experiments/key_dimension/tests
```

GPU tests require a CUDA node. Use the Slurm preflight before production;
CPU-only success does not establish cuDNN or full-context training support.

## Retained results and provenance

Use [`results/key_dimension/provenance.json`](../../results/key_dimension/provenance.json)
for the retained artifact locations. It records each run's seed, source
commit, resolved-config path and hash, selected-checkpoint path and step,
minimum validation NLL, terminal metrics, and successful Slurm log. Paths
below are relative to the repository root:

| Artifact | Retained location |
| --- | --- |
| Resolved configuration | `experiments/key_dimension/configs/runs/<run>.yaml` |
| Validation-best checkpoint | `experiments/key_dimension/checkpoints/<run>/19074/` |
| Metrics and run report | `results/key_dimension/<run>/` |
| Successful training log | `results/logs/key_dimension/<run>/slurm-<job_id>.out` |
| Combined summary and figures | `results/key_dimension/` |

Each results directory retains `metrics.jsonl`, `completion.json`,
`best_validation.json`, `metadata.json`, and `REPORT.md`. Large checkpoint
files stay on disk and are ignored by Git. They are outside the temporary
execution checkouts, which are removed after retention and merge.

The original seven runs used source commit
`5e871d2659373272e26668e10911922c7201678a`; the width-256 run used
`3d330a2f29e067b4df017ac08d12b9b6d3154b94`. Their immutable historical launch
records are `results/key_dimension/sweep.json` and
`results/key_dimension/sweep-dk256.json`. Those records preserve the original
execution-checkout and config paths, which may no longer exist. The extension
pins the original manifest's SHA-256. Preserve both launch records unchanged
and consult `provenance.json` for current config and checkpoint locations.

The [results report](../../results/key_dimension/README.md), `summary.csv`,
PPL-versus-width plot, and validation learning curves cover all eight runs.
The aggregation tool uses committed terminal validation at the common token
budget; incomplete runs are excluded from terminal-PPL comparisons. Combined
figures can be regenerated without GPUs using the extension manifest:

```bash
uv run --project experiments/shared_transformer --locked --extra cuda python -m \
  experiments.key_dimension.plot --manifest results/key_dimension/sweep-dk256.json
```

This plotting command reads retained metrics and pinned source manifests; it
does not require the historical execution checkout or original config paths.

## Reproducing a retained run

Create a new dedicated checkout at the run's recorded source commit, and copy
its retained resolved YAML config outside that checkout. Use a fresh output
directory so retained results stay intact. The example below reproduces width
256; for another width, take its run name and source commit from
`provenance.json`. Run training inside a two-H200 allocation with the same
packed dataset available. The training CLI runs in the current allocation;
it does not submit a Slurm job.

```bash
KEY_DIMENSION_PROJECT=$(git rev-parse --show-toplevel)
KEY_DIMENSION_RUN=200m-dk256-seed0
KEY_DIMENSION_SOURCE=3d330a2f29e067b4df017ac08d12b9b6d3154b94
KEY_DIMENSION_CHECKOUT="$KEY_DIMENSION_PROJECT/worktrees/reproduce-key-dimension-256"
KEY_DIMENSION_CONFIG="$KEY_DIMENSION_PROJECT/.cache/key_dimension-reproduction/$KEY_DIMENSION_RUN.yaml"
KEY_DIMENSION_OUTPUT="$KEY_DIMENSION_PROJECT/results/key_dimension-reproduction/$KEY_DIMENSION_RUN"
KEY_DIMENSION_CHECKPOINTS="$KEY_DIMENSION_PROJECT/experiments/key_dimension/checkpoints/reproduction-$KEY_DIMENSION_RUN"

git worktree add -b codex/reproduce-key-dimension-256 \
  "$KEY_DIMENSION_CHECKOUT" "$KEY_DIMENSION_SOURCE"
mkdir -p "$(dirname "$KEY_DIMENSION_CONFIG")" "$KEY_DIMENSION_OUTPUT" "$KEY_DIMENSION_CHECKPOINTS"
cp "$KEY_DIMENSION_PROJECT/experiments/key_dimension/configs/runs/$KEY_DIMENSION_RUN.yaml" \
  "$KEY_DIMENSION_CONFIG"
ln -s "$KEY_DIMENSION_CHECKPOINTS" "$KEY_DIMENSION_OUTPUT/checkpoints"
cd "$KEY_DIMENSION_CHECKOUT"
export UV_PROJECT_ENVIRONMENT="$KEY_DIMENSION_PROJECT/.envs/key_dimension-reproduction"
export UV_CACHE_DIR="$KEY_DIMENSION_PROJECT/.cache/key_dimension-reproduction/uv"
export JAX_PLATFORMS=cuda
uv sync --project experiments/key_dimension --locked --extra cuda
uv run --project experiments/key_dimension --locked --extra cuda python -m \
  experiments.key_dimension.train "$KEY_DIMENSION_CONFIG" \
  --workdir "$KEY_DIMENSION_OUTPUT" --no-resume
```

The config copy preserves the original source, seed, model, optimizer, and
data settings; `--workdir` overrides its historical output path. Synchronize
the dependency lock from the selected source checkout. Keep that checkout
unchanged while training. To resume an interrupted reproduction, use the same
config and output directory and omit `--no-resume`; the trainer restores its
latest complete training state and packed-data cursor.

For a new experiment using the current Slurm launcher, select widths with
`--key-dimensions` and use a fresh manifest under the new project's
`results/key_dimension/`. Run names and output directories must also be fresh;
an alternative manifest filename alone does not permit overwriting retained
runs. The launcher submits two-H200 jobs with requeue enabled and gates
training on successful CUDA attention checks, full-context training/evaluation,
and checkpoint-resume preflight. Each active experiment needs its own frozen
checkout. After completion and review, preserve terminal metrics, retain only
the validation-best checkpoint and successful log, clean temporary artifacts,
and merge the retained experiment before removing its checkout.
