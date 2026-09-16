# Query/key dimension and perplexity

Pretrain seven decoder-only Transformers with a fixed backbone and vary only
per-head query/key width: **2, 4, 8, 16, 32, 64, 128**. The primary endpoint is
terminal held-out perplexity after a shared training-token budget. All runs
use seed 0 and the same data order; this first sweep does not estimate
variation across independent seeds.

## Model and training recipe

The backbone has 10 layers, residual width 1024, 16 full-MHA heads, value width
64, SwiGLU intermediate width 3008, and untied 32,000-token embeddings. It uses
RMSNorm (epsilon `1e-6`), RoPE (base 10,000), no dropout, and OFM's Flax NNX
initialization. BF16 computation uses FP32 parameters and optimizer state.

| Query/key width | Trainable parameters |
| ---: | ---: |
| 2 | 179,590,144 |
| 4 | 180,245,504 |
| 8 | 181,556,224 |
| 16 | 184,177,664 |
| 32 | 189,420,544 |
| 64 | 199,906,304 |
| 128 | 220,877,824 |

This is a **200M-reference family**, not a parameter-matched comparison. The
formula is `178934784 + 327680 * qk_head_dim`. Value/output projection sizes,
depth, MLP width, and head count remain fixed. Standard RoPE frequencies are
defined at each model's true query/key width. Q/K/V are zero-padded only for
the fused attention call, after RoPE; attention scaling uses the true width
and the output is sliced back to 64 value channels.

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
500 steps and at validation. Both terminal and validation-best checkpoints
are retained, sharing one checkpoint if they coincide.

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
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/key_dimension"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project experiments/key_dimension --locked --extra cuda
uv run --project experiments/key_dimension --locked --extra cuda python -m pytest \
  -c experiments/key_dimension/pyproject.toml experiments/key_dimension/tests
```

GPU tests require a CUDA node. Use the Slurm preflight before production;
CPU-only success does not establish cuDNN or full-context training support.

## Launch and inspect the sweep

Freeze a tested commit in an execution worktree before submitting. The active
worktree must remain unchanged until all jobs are terminal or no longer
requeueable. With that checkout at `worktrees/key_dimension_200m`:

```bash
uv run --project experiments/key_dimension --locked --extra cuda python -m \
  experiments.key_dimension.sweep --execution-root "$PWD/worktrees/key_dimension_200m" --dry-run
uv run --project experiments/key_dimension --locked --extra cuda python -m \
  experiments.key_dimension.sweep --execution-root "$PWD/worktrees/key_dimension_200m" --submit
```

Submission creates one preflight and seven serial training jobs, each with two
H200s and Slurm requeue enabled. Every training job depends on successful
completion of its predecessor. The preflight requires eight passing CUDA
attention checks, full-context train/evaluation pilots at widths 2, 4, 64,
and 128, and a checkpoint-resume check. It caches only successful checks with
matching source, configuration, data, and dependency-lock identity.

The immutable run specifications and job IDs are in
`results/key_dimension/sweep.json`. Inspect the scheduler with
`squeue -u "$USER"`, the per-run reports, and the matching Slurm logs. A failed
dependency blocks subsequent runs; inspect and resolve it before resubmission.
The launcher rejects duplicate submission of an existing sweep.

Each successful training job refreshes the sweep summary and figures. They
can also be regenerated without GPUs:

```bash
uv run --project experiments/key_dimension --locked --extra cuda python -m \
  experiments.key_dimension.plot --manifest results/key_dimension/sweep.json
```

## Results and resuming

Runs are named `200m-dk002-seed0` through `200m-dk128-seed0` and write under
the main checkout's `results/key_dimension/` directory. Each run preserves
its resolved configuration, source/data identity, durable validation metrics,
completion status, concise report, and checkpoints. A completion marker is
written only after terminal evaluation and checkpoint finalization.

Slurm requeue restores the complete training state and the packed-data cursor;
it does not restart training from a validation-best checkpoint. Active runs
execute from an immutable Git worktree while writing outputs to the main
checkout. The environment and caches remain project-local.

Keep one successful combined Slurm log at
`results/logs/key_dimension/<run>/slurm-<job_id>.out`, matching metadata.
Do not delete interrupted-run checkpoints or logs while they are still needed
for debugging or exact resume. After resolution, remove failed/superseded logs.

The aggregation tool writes a summary CSV, a PPL-versus-width plot with a
logarithmic base-two x-axis, and validation learning curves. Incomplete runs
are identified explicitly and are excluded from the terminal-PPL curve.
