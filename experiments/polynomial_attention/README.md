# Taylor degree and learned attention performance

Train **six models from scratch** with attention degrees **0, 2, 4, 8, 16** and
an **exact-softmax control**. The only scientific sweep variable is the attention
kernel. There is no frozen-checkpoint replacement or per-degree tuning.

## Fixed recipe

`configs/base.yaml` preserves the key-dimension experiment's dK=64 reference:
199,906,304 parameters; 10 layers; width 1024; 16 full-MHA heads;
dK=dV=64; SwiGLU width 3008; RoPE base 10,000; RMSNorm epsilon 1e-6;
untied 32,000-token embeddings; no dropout; BF16 model computation and FP32
parameters/optimizer state. All six runs use seed 0, identical initial parameter
arrays, the same packed FineWeb corpus, tokenizer, and packed-data order.

Each run uses 19,074 updates, global batch 256, microbatch 64, four accumulation
steps, and 2,048 prediction positions: **10,000,269,312 prediction tokens**.
AdamW and the learning-rate schedule are unchanged: peak/minimum 3e-4/3e-5,
2,000 warmup updates, cosine decay, betas (0.9, 0.95), weight decay 0.1, and
gradient clipping at 1.0 after accumulation and device averaging.

Evaluation uses the same first 1,792 held-out sequences (3,670,016 prediction
tokens) every 1,000 updates and at completion. Checkpoints are saved every 500
updates and at evaluation; terminal and validation-best checkpoints survive.
This initial sweep has one seed and does not estimate seed-to-seed variation.

## Kernel and execution

`model.attention_degree: null` selects exact softmax. A nonnegative even integer
m selects `P_m(s) = sum(s**r / r!, r=0..m)` on unshifted post-RoPE scores
`s = q @ k / sqrt(dK)`. Thus m is the maximum degree, with m+1 terms.
Attention is `sum(P_m(s) * v) / sum(P_m(s))` over the causal prefix. The division
is exact. The output vocabulary softmax and next-token loss are unchanged.

All variants share the `blocked` backend: 128-position query blocks, causal
key prefixes, FP32 polynomial/normalization arithmetic, and attention-local
backward recomputation. This replaces the reference's cuDNN implementation
without changing model geometry, context, batches, or optimizer settings.
Architecture-level `remat` remains false. No tensor feature maps are materialized.

Taylor attention has no score shifting, clipping, extra Q/K normalization,
learned coefficients, or temperature. Masked pairs are zeroed without evaluating
masking sentinels. Even-degree polynomials are positive mathematically; invalid
numerical kernels/normalizers fail visibly instead of being clamped. The exact
softmax control uses stable row normalization and permits exponential underflow
to zero for individual weights.

At m=0, attention is uniform causal-prefix averaging. Q/K projections stay in
the model and parameter count, but receive no attention-derived gradient;
AdamW weight decay still applies. Increasing degree need not improve perplexity
monotonically or match exact softmax by degree 16.

## Environment and validation

Run from the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/polynomial_attention"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project experiments/polynomial_attention --locked --extra cuda
JAX_PLATFORMS=cpu uv run --project experiments/polynomial_attention --locked --extra cuda python -m pytest \
  -c experiments/polynomial_attention/pyproject.toml experiments/polynomial_attention/tests -m 'not gpu'
```

The Kennel configuration reads the original packed corpus directly from NFS at
`/data/cl/u/rafah/home/datasets/fineweb_12b`. Results, checkpoints, environments,
and caches also remain on NFS; no node-local staging is used.

CPU tests verify recipe/initialization equality, FP64-reference attention outputs
and Q/K/V gradients, causal block boundaries, prefix averaging, failure detection,
packed data, metrics transactions, and checkpoint resume. Accelerator acceptance
additionally requires all variants' full-context/full-microbatch training and
evaluation pilots on the same H100 allocation as production. The Kennel default
is two H100s; `--gpu-count 4` selects four without changing the global batch,
microbatch, accumulation count, or scientific recipe. Hardware allocation is
part of the sweep identity and is checked again when each production job starts.
Pilots measure synchronized update, validation, and checkpoint times and require
a conservative projected production runtime below six days. First-update
compilation is excluded from the steady update rate and charged once as startup.

## Launch

Freeze the tested source in an isolated execution worktree. Leave that checkout
unchanged while its jobs remain runnable or requeueable. For an execution checkout
at `worktrees/polynomial_attention_200m`:

```bash
uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
  experiments.polynomial_attention.sweep --execution-root "$PWD/worktrees/polynomial_attention_200m" --dry-run
uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
  experiments.polynomial_attention.sweep --execution-root "$PWD/worktrees/polynomial_attention_200m" --launch
```

`--launch` submits only the preflight initially. After all checks pass, that job
submits the six production jobs automatically; the first waits for successful
preflight job completion. A failed preflight submits no production jobs.
For a manual two-stage workflow, use this instead of `--launch`:

```bash
uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
  experiments.polynomial_attention.sweep --execution-root "$PWD/worktrees/polynomial_attention_200m" --preflight-only
```

The preflight runs each variant in a separate process to release compiled programs
and GPU memory between pilots. It also checks checkpoint resume. Only after every
check succeeds for the exact source/configuration/data/lock identity:

```bash
uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
  experiments.polynomial_attention.sweep --execution-root "$PWD/worktrees/polynomial_attention_200m" --submit
```

Production submission rejects a missing or unsuccessful preflight. The six
production jobs run serially on the validated H100 allocation with requeue
support. Kennel jobs use the `long` partition, 32 CPUs and 256 GB RAM; the
preflight limit is eight hours and each production limit is seven days. The
locked Python environment must be installed before submission; jobs invoke its
interpreter directly without changing dependencies. The first uses
`afterok` on preflight; subsequent jobs use `afterany` on the preceding production
job, so a recorded failure at one degree does not suppress the remaining trials.
Resumption restores the optimizer and exact data cursor. Submission records live under
`results/polynomial_attention/sweep.json`.

## Results and diagnostics

Primary endpoint: terminal held-out perplexity at the common token budget.
Aggregation writes `summary.csv`, `ppl_vs_degree.pdf`/`.png` with a horizontal
exact-softmax reference, and learning curves. Incomplete and failed runs are
identified and excluded from terminal comparisons.

At each evaluation, per-layer score/kernel/normalizer ranges and invalid counts
are measured on the fixed first held-out sequence, at full context length.
This explicitly sampled diagnostic does not replace or alter the evaluation set.
The runtime checks invalid kernels/normalizers during ordinary model execution;
nonfinite training losses fail with retained diagnostics. Throughput is recorded
but does not determine the training budget.

```bash
uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
  experiments.polynomial_attention.plot --manifest results/polynomial_attention/sweep.json
```

Keep one successful combined Slurm log per retained run at
`results/logs/polynomial_attention/<run>/slurm-<job_id>.out`, matching its metadata.
Keep checkpoints and failed/interrupted logs while debugging or resuming; remove
superseded and failed logs once resolved. Existing key-dimension runs and results
are independent of this family.
