# Taylor degree and learned attention performance

This experiment uses the shared Transformer implementation in
[`experiments/shared_transformer`](../shared_transformer/README.md). Its configuration,
sweep, preflight, plots, and retention policy stay in this experiment directory.
Use the shared locked environment for new runs; reproduce historical runs from
the source revision and dependency lock recorded in their provenance.

This completed collection trained **six models from scratch** with attention
degrees **0, 2, 4, 8, 16** and an **exact-softmax control**. The only scientific
sweep variable was the attention kernel: no frozen-checkpoint replacement or
per-degree tuning. [Results](../../results/polynomial_attention/README.md) and
[provenance](../../results/polynomial_attention/provenance.json) retain the
terminal metrics, learning curves, execution history, and selected checkpoints.

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
updates and at evaluation during training. After completion, only the
validation-best checkpoint per run survives; terminal metrics remain in the
results even when the selected checkpoint is earlier. This collection has one
seed and does not estimate seed-to-seed variation.

## Kernel and execution

`model.attention_degree: null` selects exact softmax. A nonnegative even integer
m selects `P_m(s) = sum(s**r / r!, r=0..m)` on unshifted post-RoPE scores
`s = q @ k / sqrt(dK)`. Thus m is the maximum degree, with m+1 terms.
Attention is `sum(P_m(s) * v) / sum(P_m(s))` over the causal prefix. The division
is exact. The output vocabulary softmax and next-token loss are unchanged.

The base configuration retains the `blocked` reference backend: 128-position
query blocks, causal key prefixes, FP32 polynomial/normalization arithmetic, and
attention-local backward recomputation. The retained degree-zero control used
this implementation; degrees 2–16 and softmax used the optimized `fused` backend.
The resolved configurations record this execution difference without changing
model geometry, context, batches, or optimizer settings. Architecture-level
`remat` remains false. No tensor feature maps are materialized.

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

## Opt-in fused execution

The default remains `model.attention_backend: blocked`. To select the fused
implementation, copy the base YAML, set `model.attention_backend: fused`, and
pass that file with `--base-config /path/to/fused.yaml` to the same sweep launch
command below. Integer degrees use the Pallas/Triton polynomial kernel with
FP32 Horner arithmetic. Mixed FP32/BF16 matrix products split the FP32 operand
into three BF16 components with FP32 accumulation; products of original BF16
operands use one BF16 matrix product. FP32-only inputs retain the TF32x3 path.
Truncating the leading components avoids rounding finite FP32 values near the
range limit to BF16 infinity. The split retains the FP32 significand away from
BF16 underflow; it does not promise bitwise IEEE equivalence or exact subnormal
handling. The exact softmax control uses cuDNN.
Diagnostics retain the actual fused layer outputs and collect score and
kernel statistics separately through the blocked reference implementation.
Fused sweeps must additionally pass the fused GPU correctness suite. Backend,
source, and configuration are part of sweep provenance: an existing blocked
sweep cannot be changed in place or reused as a fused preflight success.

The polynomial kernel schedules independent query tiles in parallel and streams
only causal key tiles through each program. It stores no sequence-square score
or weight arrays. Separate dQ and dK/dV kernels recompute tiles without atomic
gradient updates; the saved output and normalizers have linear sequence size.
Online normalized accumulation and division before the derivative product avoid
unnecessary overflow. The backward retains the FP32 output before its BF16 cast.
These reduction and matrix-product choices are mathematically equivalent to the
reference but not bitwise identical; FP64 and BF16 output/VJP tests bound error.

The CUDA kernel was validated on H100 and B300 GPUs, accepts FP32/BF16 Q/K/V and head widths up to 128,
and uses smaller tiles for widths above 64. Scalar diagnostics still come from
the reference implementation. Degree zero currently uses the fused tiled path
to preserve the reference's invalid-score checks; it is not a prefix-scan path.
No extra Python dependency is needed: the pinned JAX supplies the Triton compiler.

Run benchmarks inside a GPU allocation, from the repository root:

```bash
python -m experiments.polynomial_attention.benchmark_attention --output /path/to/kernel.json
python -m experiments.polynomial_attention.benchmark_train_step --output /path/to/training.json
```

The kernel benchmark uses the production per-GPU shape `[16, 2048, 16, 64]`.
The complete-update benchmark requires four GPUs by default, the production
model, global batch 256, and four accumulation steps. Both exclude compilation
and synchronize three warmups and ten measured calls. Training uses identical
initial weights and resident synthetic tokens; it excludes data loading,
validation, and checkpoint I/O. JSON reports include source hashes, hardware,
raw timing samples, and numerical comparisons. Kernel memory figures are
compiler buffer estimates, not measured allocator peaks.

For a comparison with cuDNN and a profile of complete warmed updates:

```bash
python -m experiments.polynomial_attention.benchmark_train_step --degrees 2 16 softmax --implementations fused --trace-dir /path/to/traces --output /path/to/training.json
python -m experiments.polynomial_attention.analyze_profile /path/to/traces --output /path/to/profile-summary.json
python -m experiments.polynomial_attention.tune_attention --degrees 2 16 --output /path/to/tile-tuning.json
python -m experiments.polynomial_attention.benchmark_packed_run --workdir /path/to/fresh/pilot --output /path/to/packed-pilot.json
```

Profiles capture three additional updates after timing. The analyzer counts
actual GPU stream kernels, avoiding nested host/HLO annotations. CUDA command
buffer replay can hide operation names; such GEMMs remain unattributed rather
than being assigned to the MLP or loss. Recorded GPU activity is not SM utilization.
Tile tuning rejects numerical failures and reports resource failures explicitly.
The packed pilot requires a clean committed checkout and a fresh workdir. It
uses real configured data and the production learning-rate horizon, stopping
after 100 updates. Its compressed evaluation/checkpoint cadence measures their
first and warmed costs; its total wall time must not be scaled directly to a
full production run. See the [performance reports](../../results/polynomial_attention/performance/README.md)
for the measured comparisons, hardware, and runtime assumptions.

The completed production runs used four GPUs each. Degree zero ran on H100
PCIe, degrees 2–16 on H100 SXM, and softmax continued from an H100 checkpoint on
B300. The performance reports preserve these distinctions: B300 improved
softmax substantially but was slower for degree 16 with this kernel. Select
hardware using measured full-update costs. Independent future runs may use
available partitions concurrently; `sweep` jobs can be preempted and must resume
from checkpoints.

## Environment and validation

Run from the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project experiments/shared_transformer --locked --extra cuda
JAX_PLATFORMS=cpu uv run --project experiments/shared_transformer --locked --extra cuda python -m pytest \
  -c experiments/shared_transformer/pyproject.toml experiments/polynomial_attention/tests -m 'not gpu'
```

The Kennel configuration reads the original packed corpus directly from NFS at
`/data/cl/u/rafah/home/datasets/fineweb_12b`. Results, checkpoints, environments,
and caches can also use NFS; no node-local staging is required. Completed
experiment caches and temporary execution checkouts are removed after retention.

CPU tests verify recipe/initialization equality, FP64-reference attention outputs
and Q/K/V gradients, causal block boundaries, prefix averaging, failure detection,
packed data, metrics transactions, and checkpoint resume. Accelerator acceptance
additionally requires all variants' full-context/full-microbatch training and
evaluation pilots on the intended production allocation. The historical sweep
launcher defaults to two H100s; use `--gpu-count 4` to reproduce the four-GPU
layout without changing the global batch, microbatch, accumulation count, or
scientific recipe. Its supported scheduler profiles are recorded in `sweep.py`;
it does not automatically select B300s or schedule independent runs in parallel.
Hardware allocation is part of a live sweep identity and checked at job startup.
Pilots measure synchronized update, validation, and checkpoint times and require
a conservative projected production runtime below six days. First-update
compilation is excluded from the steady update rate and charged once as startup.

## Reproduce in a fresh experiment checkout

Keep the completed collection on `main` intact. Create a dedicated branch and
separate checkout for a reproduction, using the run's `source_commit` from
`results/polynomial_attention/provenance.json` for the original implementation:

```bash
git worktree add -b codex/poly-reproduction /path/to/new/checkout <source_commit>
```

Install the locked environment for that checkout and run acceptance tests before
production. The exact per-run YAML files are retained on `main` under
`experiments/polynomial_attention/configs/runs/<run>.yaml`. They include the
original source, seed, backend, and training recipe. Their recorded
`train.workdir` is historical: always override it with a new, empty directory.
Update data manifest locations in a copied configuration if reproducing on
another system, and verify the corpus against the retained data identity.

Inside a four-GPU Slurm allocation, from the new source checkout:

```bash
rerun_workdir="$PWD/results/polynomial_attention/reproduction-m016-seed0"
test ! -e "$rerun_workdir" && \
python -m experiments.polynomial_attention.train \
  /path/to/main/experiments/polynomial_attention/configs/runs/200m-m016-seed0.yaml \
  --workdir "$rerun_workdir" --no-resume
```

Use the source revision recorded in the configuration; if changing code, record
its new commit in a copied configuration before running. For a new complete
sweep, the existing launcher can prepare, validate, and submit all variants:

```bash
python -m experiments.polynomial_attention.sweep \
  --project-root "$PWD" --execution-root "$PWD" --gpu-count 4 --dry-run
```

Use this only in a fresh experiment checkout with a fresh results directory.
Pass `--base-config /path/to/config.yaml` to select the desired backend.
`--launch` submits GPU preflight first and submits production only after it
passes; `--preflight-only` then `--submit` is the manual equivalent. `--queue`
queues production with successful-preflight dependencies. The existing launcher
runs production jobs serially; its default profile uses `long`, and its persisted
`dogo` profile uses `standard`. Review allocation and partition availability
before a new sweep. Freeze the checkout while jobs can run or requeue, and keep
the environment unchanged. Resume restores optimizer state and the data cursor.

After reviewing a reproduction, retain its validation-best checkpoint and
scientific results according to `AGENTS.md`, then merge it into `main` and
remove its temporary checkout and merged branch.

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
uv run --project experiments/shared_transformer --locked --extra cuda python -m \
  experiments.polynomial_attention.plot --manifest results/polynomial_attention/provenance.json
```

The retained layout is:

- `experiments/polynomial_attention/`: scaffold, dependency lock, tests, and
  `configs/runs/<run>.yaml` with the six resolved configurations.
- `experiments/polynomial_attention/checkpoints/<run>/`: exactly one
  validation-best checkpoint per run, stored on shared storage and ignored by
  Git. Degree zero's best checkpoint precedes its final training step.
- `results/polynomial_attention/`: `provenance.json`, per-run metrics and
  completion records, summary tables, figures, and consolidated `performance/`
  reports. Terminal metrics are independent of checkpoint selection.
- `results/logs/polynomial_attention/<run>/slurm-<job_id>.out`: one successful
  combined Slurm log per run, matching its provenance; logs remain ignored by
  Git. Validation and benchmark logs use their recorded experiment paths.

The plotting command defaults to `provenance.json` when present, otherwise to a
live `sweep.json`. Retained paths resolve relative to the repository containing
the manifest, so a copied checkout can regenerate plots without the original
execution worktrees, training datasets, or physical checkpoints. Configuration,
metric, and dependency-lock checksums are verified before aggregation; best and
terminal metrics are checked against the committed validation history. Dataset
checksums remain provenance for reproducing training and are not reread to plot.
Live sweep aggregation remains available during new experiments.

Temporary launch manifests, caches, intermediate/final checkpoint duplicates,
failed or superseded logs, and execution worktrees are removed once completion
and retention are verified. Existing key-dimension runs are a separate experiment
family.
