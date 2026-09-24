# Taylor degree and learned attention performance

This completed collection trained six models from scratch with attention
degrees **0, 2, 4, 8, 16** and an **exact-softmax control**. The only scientific
sweep variable was the attention kernel; there was no frozen-checkpoint
replacement or per-degree tuning. The shared model implementation now lives in
[`scaffolds/transformer`](../../scaffolds/transformer/README.md), with
study-specific orchestration in `scaffolds/transformer/studies/polynomial_attention/`.

<!-- BEGIN GENERATED RESULTS -->

6/6 runs complete at 10,000,269,312 prediction tokens per run.

| Attention | Parameters | Terminal PPL |
| --- | ---: | ---: |
| Degree 0 | 199,906,304 | 47.0653 |
| Degree 2 | 199,906,304 | 16.2810 |
| Degree 4 | 199,906,304 | 14.9836 |
| Degree 8 | 199,906,304 | 14.6781 |
| Degree 16 | 199,906,304 | 14.7062 |
| Exact softmax | 199,906,304 | 14.6942 |

[Machine-readable results](summary.csv) · [Terminal PPL](figures/ppl_vs_degree.pdf) · [Learning curves](figures/learningcurves.pdf)

<!-- END GENERATED RESULTS -->

Degree 8, degree 16 and exact softmax have nearly equal terminal perplexity on
this seed. The 0.028 gap between degrees 8 and 16 is not evidence of a reliable
ordering across seeds. Higher Taylor degree need not improve trained
perplexity monotonically because it changes optimization and the learned
attention patterns. The degree-zero control is causal prefix averaging; its
Q/K projections receive no attention-derived learning signal.

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
pass that file with `--base-config /path/to/fused.yaml` to the study sweep launcher. Integer degrees use the Pallas/Triton polynomial kernel with
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
python -m scaffolds.transformer.studies.polynomial_attention.benchmark_attention --output /path/to/kernel.json
python -m scaffolds.transformer.studies.polynomial_attention.benchmark_train_step --output /path/to/training.json
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
python -m scaffolds.transformer.studies.polynomial_attention.benchmark_train_step --degrees 2 16 softmax --implementations fused --trace-dir /path/to/traces --output /path/to/training.json
python -m scaffolds.transformer.studies.polynomial_attention.analyze_profile /path/to/traces --output /path/to/profile-summary.json
python -m scaffolds.transformer.studies.polynomial_attention.tune_attention --degrees 2 16 --output /path/to/tile-tuning.json
python -m scaffolds.transformer.studies.polynomial_attention.benchmark_packed_run --workdir /path/to/fresh/pilot --output /path/to/packed-pilot.json
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
full production run. See the [performance reports](#performance-measurements)
for the measured comparisons, hardware, and runtime assumptions.

The completed production runs used four GPUs each. Degree zero ran on H100
PCIe, degrees 2–16 on H100 SXM, and softmax continued from an H100 checkpoint on
B300. The performance reports preserve these distinctions: B300 improved
softmax substantially but was slower for degree 16 with this kernel. Select
hardware using measured full-update costs. Independent future runs may use
available partitions concurrently; `sweep` jobs can be preempted and must resume
from checkpoints.


## Retained runs, diagnostics and wall times

All runs use 199,906,304 parameters, query/key and value width 64, seed 0,
19,074 updates and the common 10B-token budget. At evaluation, per-layer
score/kernel/normalizer ranges and invalid counts are measured on the fixed
first held-out sequence at full context length (2,048 prediction positions).
This sampled diagnostic does not alter the evaluation set. All final sampled
diagnostics passed. Invalid numerical kernels/normalizers or nonfinite losses
fail visibly during training; throughput never determines the token budget.

| Run | Backend | Terminal NLL | Best step | Best PPL | Successful Slurm job | Allocated wall time |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| [200m-m000-seed0](runs/200m-m000-seed0/metadata.json) | blocked | 3.851536282 | 19000 | 47.032595 | `199433` | 10h 01m 08s |
| [200m-m002-seed0](runs/200m-m002-seed0/metadata.json) | fused | 2.790000975 | 19074 | 16.281036 | `199648` | 2h 48m 14s |
| [200m-m004-seed0](runs/200m-m004-seed0/metadata.json) | fused | 2.706955969 | 19074 | 14.983596 | `199649` | 2h 51m 53s |
| [200m-m008-seed0](runs/200m-m008-seed0/metadata.json) | fused | 2.686355667 | 19074 | 14.678087 | `199650` | 3h 05m 18s |
| [200m-m016-seed0](runs/200m-m016-seed0/metadata.json) | fused | 2.688271710 | 19074 | 14.706237 | `199651` | 3h 53m 30s |
| [200m-softmax-seed0](runs/200m-softmax-seed0/metadata.json) | fused | 2.687451269 | 19074 | 14.694177 | `199869` | 1h 41m 51s |

Degree zero used source `2484f00e9a32f1571b330fff5788d9a86175235f`;
the other runs used `1a8836a75836671cdf83a3ab7763425a6a88cac7`. All used
four GPUs. Degree zero used the original blocked backend on H100 PCIe;
degrees 2–16 used the optimized fused kernel on H100 SXM. Softmax's wall time
includes both H100 and B300 allocations and replay after migration. Total
training allocations were 87,714 seconds (24h 21m 54s), approximately 97.5
GPU-hours; queue waits, development benchmarks and preflight are excluded.

## Artifact layout and cluster ownership

This README is the experiment's complete narrative documentation. The
[canonical provenance](provenance/manifest.json) records the exact source,
seed, resolved configuration and checksum, terminal metrics, validation-best
checkpoint, data identity, and successful Slurm log for every run.

| Artifact | Location relative to the repository |
| --- | --- |
| Exact historical recipes and dependency lock | `results/polynomial_attention/configs/` |
| Per-run metrics and completion evidence | `results/polynomial_attention/runs/<run>/` |
| Figures | `results/polynomial_attention/figures/` |
| Derived summary CSV | `results/polynomial_attention/summary.csv` |
| Canonical provenance | `results/polynomial_attention/provenance/manifest.json` |
| Current shared Transformer implementation | `scaffolds/transformer/` |
| Study-specific sweep, preflight and plot tools | `scaffolds/transformer/studies/polynomial_attention/` |
| Canonical checkpoint layout | `checkpoints/polynomial_attention/<run>/<step>/` |
| Successful retained training log | `logs/polynomial_attention/<run>/slurm-<job_id>.out` |

The home cluster is **Kennel**; its recorded artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. Relative checkpoint and log locations are
resolved against that artifact root, not whichever clone is reading this
report. Git shares code, recipes and scientific results; it does not transfer
ignored checkpoints, datasets, environments or Slurm logs.

The retained checkpoints and successful logs live on Kennel. Degree zero keeps
step **19000**, where validation NLL was lowest; the other five runs keep step
**19074**. A clone on ORCD or another machine can regenerate the figures without
those checkpoint files. The softmax H100-to-B300 continuation stayed within
Kennel and did not change the experiment's home cluster.

The YAML files are preserved **byte for byte**. Their old absolute data and
work-directory paths describe historical execution; moving a recipe does not
make those paths portable. `configs/uv.lock` is the historical lock matching
the retained dependency hash, while `scaffolds/transformer/uv.lock` governs
new runs of the current shared scaffold. Source-hash keys inside historical
benchmark and launch records likewise refer to files at their recorded commits.

During training, keep the latest resumable checkpoint and the validation-best
checkpoint. Only after completion and review apply the validation-best-only
retention policy. Terminal metrics remain the scientific endpoint whether or
not the selected checkpoint is terminal. Keep one successful Slurm log per
retained run, benchmark and validation; remove failed or superseded logs once
debugging is resolved. Consult [AGENTS.md](../../AGENTS.md) and
[CLUSTERS.md](../../CLUSTERS.md) before a launch or migration.

## Environment, validation and figure regeneration

Run these commands from the repository root. The shared JAX environment is
separate from the root PyTorch environment because their CUDA-library pins
differ. Historical replay instead installs the lock at the recorded source.

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project scaffolds/transformer --locked --extra cuda
JAX_PLATFORMS=cpu uv run --project scaffolds/transformer --locked --extra cuda python -m pytest \
  -c scaffolds/transformer/pyproject.toml scaffolds/transformer/studies/polynomial_attention/tests -m 'not gpu'
uv run --project scaffolds/transformer --locked --extra cuda python -m \
  scaffolds.transformer.studies.polynomial_attention.plot --manifest results/polynomial_attention/provenance/manifest.json
```

Plots use committed terminal validation at the common token budget and keep
best-checkpoint metrics separate. They read canonical repository-relative
artifact paths, not the vanished original execution checkouts or config
workdirs. Recipe, metric and historical-lock checksums are verified. No dataset,
GPU, or physical checkpoint is required to regenerate retained figures.

CPU tests cover recipe/initialization equality, independent attention output
and gradient references, packed data, metrics transactions, and checkpoint
resume. GPU acceptance must additionally check the intended attention backend,
full context and microbatch, training/evaluation pilots, and checkpoint resume
on the production allocation. CPU success alone does not establish GPU support.

## Reproducing and running

For an **exact historical replay**, create a dedicated branch and separate
checkout at the run's `source_commit`, install that checkout's original locked
environment, and use its historical training entry point
`experiments.polynomial_attention.train`. Copy `configs/200m-m016-seed0.yaml` outside the frozen checkout and
provide the content-identical packed corpus. Run inside an allocation with
four GPUs matching the recorded implementation and hardware; the training CLI runs in the allocation and does not submit Slurm.
Override the historical `train.workdir` with a fresh result directory, and
keep checkpoint storage outside the temporary execution checkout. Old source
revisions predate `checkpoint_dir`; for those revisions, create a
`<fresh-workdir>/checkpoints` symlink to a fresh durable checkpoint directory
before starting. Do not rewrite the retained recipe or reuse its old outputs.

For a **new run of the shared scaffold**, use a fresh dedicated branch and
working checkout, copy the desired recipe into the new result directory,
record the new source commit and home cluster, and set these fields in the
copied configuration:

```yaml
train:
  source_commit: auto
  workdir: results/polynomial_attention_reproduction/runs/200m-m016-seed0
  checkpoint_dir: checkpoints/polynomial_attention_reproduction/200m-m016-seed0
```

Keep the complete original model, optimizer and data sections; this excerpt
only shows path/source overrides. Verify the dataset checksums, choose an
allocation on the recorded home cluster, and freeze the tested checkout before
launch. From that checkout, inside the allocation:

```bash
uv run --project scaffolds/transformer --locked --extra cuda python -m \
  scaffolds.transformer.train /path/to/fresh-resolved-config.yaml --no-resume
```

To resume, retain the same source, configuration and paths and omit
`--no-resume`. The trainer restores optimizer state and the packed-data cursor.
The family sweep launcher is `scaffolds.transformer.studies.polynomial_attention.sweep`; use its `--help` and
dry run to inspect the current scheduler profile before submission. New result
paths and run names must be fresh; changing only the manifest filename does
not authorize overwriting retained runs. A new clone supplies no ignored
artifacts. Before migrating a run, transfer and verify its checkpoints and
supporting artifacts, make matching data/environment available, update its
home cluster and locations, and avoid concurrent copies of the same run.

After review, retain the scientific results, exact recipes, best checkpoint and
successful logs in the canonical layout, merge to `main`, and remove temporary
checkouts and merged branches after their indispensable artifacts are safe.

## Performance measurements

These retained measurements document the attention implementation and hardware used
for the completed 200M-model sweep. Full training updates use four GPUs, a global
batch of 256, global microbatch 64, four accumulation steps, and 2,048 prediction
positions. Timings are medians of ten synchronized updates after three warmups;
compilation, data loading, evaluation, and checkpointing are excluded unless stated.

| Measurement | Hardware / successful job | Source | Results |
| --- | --- | --- | --- |
| Blocked versus first fused, full update | Four H100 SXM / 199623 | `3537262e5b84b5fb44c204e5e70ad767da46a109` | [Report](#blocked-baseline-complete-updates) · [raw samples](performance/blocked_baseline/benchmark.json) |
| First fused attention kernel | H100 PCIe / 199626 | `0b7a37bab00c75b1b550bd44180c804b8d91170f` | [Report](#first-fused-attention-kernel) · [raw samples](performance/attention_kernel/benchmark.json) |
| Optimized full updates and real-data pilot | Four H100 SXM / 199635 | `0f655fb7098ae73e5923fc295fe0b5c3012184ee` | [Report](#optimized-h100-throughput) · [raw samples](performance/h100_throughput/train-step.json) |
| Optimized full updates on B300 | Four B300 / 199854 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | [Report](#b300-comparison-and-continuation) · [raw samples](performance/b300/train-step.json) |

The optimized H100 update takes 0.464 seconds at degree 2 and 0.686 seconds at
degree 16, compared with 1.644 and 1.922 seconds for the original blocked code.
B300 improves softmax throughput by 1.89× and degree-2 throughput by 1.16×;
degree 16 takes 28% longer on B300 with this kernel.

The first fused validation passed 151 CPU tests, one separate two-device CPU test,
and 46 GPU tests. The optimized implementation passed 56 GPU tests on H100 and
56 on B300. All 28 H100 tile candidates passed parity. CPU validation used source
`aeebff38f742eeddbbd995763c9a3b7dc2cf8295`; each directory's metadata records source
hashes, hardware, checksums, and the successful retained Slurm log. Exploratory
H100 tuning records preserve their exact source hashes and allocation identity.

Raw timing/loss samples, tile comparisons, correctness records, and compact GPU
profile aggregates are retained. Full traces, pilot checkpoints, batch wrappers,
and temporary execution checkouts were removed after validation. Any explicitly
historical location inside a measurement is provenance, not a retained artifact.
The real-data pilot embeds its manifest identity. Reproduction tools live under
`scaffolds/transformer/studies/polynomial_attention/`.

## Blocked baseline complete updates

Successful Slurm job `199623`, on four NVIDIA H100 80GB HBM3 GPUs on `dogo.csail.mit.edu`.
Source: `3537262e5b84b5fb44c204e5e70ad767da46a109`. The source kernel is identical to the final correctness-tested kernel.

| Taylor degree | Blocked update | Fused update | Speedup |
|---|---:|---:|---:|
| 2 | 1.644 s | 0.937 s | 1.75x |
| 16 | 1.922 s | 1.227 s | 1.57x |

Medians of 10 synchronized updates after 3 warmups, with compilation excluded.
Both backends use the same 199,906,304-parameter BF16 model, 2048 prediction positions,
global microbatch 64, four accumulation steps, and global batch 256. Timings include
forward/backward, accumulation, gradient all-reduce, clipping and AdamW updates.
They exclude data loading/transfers, evaluation, diagnostics and checkpoint I/O.

Each trial restores independent device buffers from the same immutable host snapshot
of the initial FP32 parameters and optimizer state. Inputs are the same resident
synthetic token batch, with the same RNG sequence. The largest absolute loss difference
across timed updates was 5.9127808e-05; these short loss traces supplement output/VJP
correctness checks and do not establish equivalence of long training trajectories.

These are HBM3 H100 measurements. The separate attention-only benchmark uses an H100
PCIe; compare backends within each benchmark, not absolute times across the two hosts.

- [Raw timings, losses, configuration and source hashes](performance/blocked_baseline/benchmark.json)
- [Successful Slurm log](../logs/polynomial_attention/performance/blocked_baseline/slurm-199623.out)

This is the initial fused implementation. The [optimized H100 audit](#optimized-h100-throughput) records the kernel subsequently used for production. Reproduction uses `experiments.polynomial_attention.benchmark_train_step` at the source above.

## First fused attention kernel

The implementation uses independent parallel query tiles, causal key-tile loops,
FP32 unshifted Horner evaluation and normalization, TF32x3 FP32 matrix products,
and separate custom dQ/dK/dV backward kernels. No sequence-square intermediate is
stored. Online normalized accumulation avoids overflowing an unnormalized numerator;
the backward retains unrounded FP32 outputs and divides before multiplying large derivatives.

Successful Slurm job `199626`, `NVIDIA H100 PCIe` on
`mastiff.csail.mit.edu`, source `0b7a37bab00c75b1b550bd44180c804b8d91170f`.
BF16 shape `[16, 2048, 16, 64]` (batch, time, heads, channels), matching one device's
microbatch in the four-GPU experiment. Three synchronized warmups and ten synchronized
measurements per operation; compilation excluded. Each comparison uses the same inputs
and GPU. These measurements cover attention alone.

| Degree | Blocked forward | Fused forward | Blocked forward + backward | Fused forward + backward | Forward + backward speedup |
|---|---:|---:|---:|---:|---:|
| 0 | 11.84 ms | 4.88 ms | 31.32 ms | 7.51 ms | 4.17x |
| 2 | 14.31 ms | 3.56 ms | 56.00 ms | 21.92 ms | 2.55x |
| 4 | 14.30 ms | 3.68 ms | 56.09 ms | 21.86 ms | 2.57x |
| 8 | 14.37 ms | 4.21 ms | 60.58 ms | 25.33 ms | 2.39x |
| 16 | 16.85 ms | 5.33 ms | 63.07 ms | 31.28 ms | 2.02x |

- Degree 2: compiler-estimated temporary buffers 848.2 MiB blocked → 130.0 MiB fused.
- Degree 16: compiler-estimated temporary buffers 1292.5 MiB blocked → 130.0 MiB fused.

These memory figures come from `compiled.memory_analysis`, exclude input/output buffers,
and are not measured peak allocation or whole-model memory. The largest BF16 relative
L2 difference against the blocked reference was 0.000874248 for outputs and
0.00362956 for Q/K/V gradients. These are numerical comparisons, not bitwise identity.

### Correctness

- 151 CPU tests passed; the two-device accumulation test was skipped in that run.
- That accumulation test passed separately with two CPU devices.
- 46 GPU tests passed: 39 new fused checks and 7 existing GPU attention checks.
- Total: 198 test passes. Output/VJP checks include independent FP64 references,
  odd dimensions, partial tiles, causality, cancellation, invalid-score poisoning,
  BF16 production contexts, and finite large-value forward/backward overflow stress.

CPU tests ran from `aeebff38f742eeddbbd995763c9a3b7dc2cf8295` in job `199622`;
CPU test files and the model/training implementation are unchanged in the final source.
The final GPU suite and benchmark ran together from the clean source commit above.
The overflow fixture uses a transverse query component so a relative gradient check
is well conditioned; tolerances were not relaxed to accommodate extreme radial cancellation.

### Complete training updates

The [four-H100 full-update benchmark](#blocked-baseline-complete-updates)
measured 1.75x speedup for degree 2 and 1.57x for degree 16, including accumulation,
all-reduce and optimizer work. It uses H100 HBM3 GPUs rather than this PCIe H100;
compare the two backends within each benchmark. Data loading, evaluation and checkpoint
I/O are excluded from those training-step measurements.

This records the first fused implementation. The [optimized H100 audit](#optimized-h100-throughput) records the later mixed-product optimization used for production. Degree zero retains invalid-score checks and uses tiled attention rather than a prefix-scan fast path.

- [Raw timing, memory and parity measurements](performance/attention_kernel/benchmark.json)
- [Successful Slurm log](../logs/polynomial_attention/performance/attention_kernel/slurm-199626.out)
- [GPU tests](performance/attention_kernel/gpu-tests.xml), [CPU tests](performance/attention_kernel/cpu-tests.xml), [distributed test](performance/attention_kernel/distributed-tests.xml)

## Optimized H100 throughput

Attention was the main avoidable cost. The original fused implementation still spent about 70% of degree-2 and 77% of degree-16 update time inside attention. Removing needless FP32 conversions and using three BF16 components for mixed products reduced complete update time again, while retaining FP32 polynomial/normalization arithmetic and the FP32 output saved for backward.

Final source: `0f655fb7098ae73e5923fc295fe0b5c3012184ee`; successful Slurm job `199635`, four H100 SXM GPUs on `dogo`, global batch 256, microbatch 64, four accumulation steps, 2,048 prediction positions, 199,906,304 parameters. [Successful Slurm log](../logs/polynomial_attention/performance/h100_throughput/slurm-199635.out).

### Complete training update timings

| Variant | Original blocked (s) | First fused (s) | Revised fused (s) | Training only, 10B tokens | Wall-time budget |
|---|---:|---:|---:|---:|---:|
| Degree 2 | 1.644 | 0.939 | 0.464 | 2.46 h | 2.74–2.85 h |
| Degree 4 | — | — | 0.485 | 2.57 h | 2.85–2.96 h |
| Degree 8 | — | — | 0.533 | 2.83 h | 3.11–3.22 h |
| Degree 16 | 1.922 | 1.230 | 0.686 | 3.64 h | 3.92–4.03 h |
| Softmax (cuDNN) | — | 0.358 | 0.357 | 1.89 h | 2.17–2.29 h |

Compared with the original blocked full-step benchmark, the revised kernel is 3.55× faster at degree 2 and 2.80× at degree 16. The [original blocked measurements](performance/blocked_baseline/benchmark.json) are from the earlier successful four-SXM job 199623; the first-fused comparison was remeasured in allocation 199628. Final timings use three warmups and ten synchronized updates; profiles capture three additional updates after those timings.

### GPU profile

| Variant | Attention time/update | Attention share of wall time | Collectives/update |
|---|---:|---:|---:|
| 2 | 175.9 ms | 37.9% | 11.0 ms |
| 4 | 200.2 ms | 41.1% | 10.7 ms |
| 8 | 251.4 ms | 46.9% | 9.1 ms |
| 16 | 416.3 ms | 59.8% | 6.4 ms |
| softmax | 60.2 ms | 16.8% | 9.3 ms |

The comparison leaves about 0.28–0.30 seconds per update outside attention across variants. Polynomial attention retains FP32 Horner evaluation and precision-preserving mixed products, so additional attention cost relative to cuDNN is expected. This is a measured, validated implementation; no claim of global optimality is made. CUDA command-buffer replay hides MLP/loss attribution; unidentified GEMMs remain explicitly unattributed. Recorded GPU activity does not measure SM utilization.

### Numerical and launch validation

All 56 GPU tests passed: independent FP64 output/QKV-gradient references, BF16 production context, odd widths and tail tiles, causality, tiny cotangents, invalid-score behavior, and overflow cases. Direct mixed-dot tests reject one- and two-component approximations and check finite values near the FP32 range limit. BF16 components can underflow; the implementation does not promise bitwise IEEE behavior for subnormals.

All 28 tile candidates (14 configurations × two degrees) passed parity. The existing 64×64/four-warp/two-stage setting won degree 2; a degree-16 alternative improved the isolated forward/backward call by only 2.1%. The shared existing tile setting was retained. Doubling global microbatch to 128 and halving accumulation to two changed full-step speed by less than 2%, so production batch layout remains unchanged.

### Real data and runtime projection

The 100-update degree-16 pilot read the actual packed manifests directly, performed two full seven-batch validations and four checkpoint saves, and kept the 19,074-step learning-rate horizon. Its wall time was 375.5 seconds. The first seven warmed production-loop samples had a median of 0.6997 seconds/update, versus 0.6865 seconds on resident synthetic input. The warmed evaluation took 3.59 seconds, and the waited-for checkpoint took 21.49 seconds. Startup (including dataset validation and compilation) is budgeted at 256.0 seconds.

The table adds the measured data/loop difference, 20 validations, startup, and a checkpoint budget ranging from 20 waited-for validation saves to all 39 saves charged serially. The latter is conservative because nonvalidation saves can overlap training. All variants use the observed degree-16 evaluation allowance. These are projections on the measured hardware, excluding queue delays, contention changes, and retries; they are not statistical confidence bounds. The compressed pilot cadence is not scaled directly to production.

The original forecast for the five remaining variants was approximately 14.8–15.4 hours when run sequentially with four H100 SXM GPUs each. This was a pre-production estimate; the completed sweep results are recorded separately.

Retained measurements:

- [Optimized complete updates](performance/h100_throughput/train-step.json) and [first-fused remeasurement](performance/h100_throughput/baseline-train-step.json).
- [Optimized GPU profile](performance/h100_throughput/profile-summary.json) and [first-fused GPU profile](performance/h100_throughput/baseline-profile-summary.json), compacted to per-device averages with the analysis method and trace hashes.
- [Tile tuning](performance/h100_throughput/tile-tuning.json) and [microbatch-128 comparison](performance/h100_throughput/microbatch128.json), including raw samples and parity results.
- [Packed-data pilot](performance/h100_throughput/packed-pilot.json), including its data identity, [validation metrics](performance/h100_throughput/packed-validation-metrics.jsonl), and [runtime projection](performance/h100_throughput/runtime-estimate.json).
- [GPU correctness tests](performance/h100_throughput/gpu-tests.xml) and [source/artifact provenance](performance/h100_throughput/metadata.json).

Raw timing and loss samples are retained. Full traces and temporary pilot checkpoints were removed after validation. Locations explicitly labeled historical in measurements identify the original execution only; they are not retained artifacts.

## B300 comparison and continuation

Four B300s improve softmax throughput by 1.89× and degree-2 throughput by 1.16×,
but degree 16 takes 28% longer with this kernel. Hardware selection therefore
depends on the attention variant; this benchmark does not establish optimal B300 tuning.

Successful Slurm job `199854`, partition `sweep`, node `akita`, four NVIDIA B300
SXM6 AC GPUs. Source: `1a8836a75836671cdf83a3ab7763425a6a88cac7`.
[Successful Slurm log](../logs/polynomial_attention/performance/b300/slurm-199854.out).

| Attention | Four H100 SXM, ms/update | Four B300, ms/update | B300 throughput / H100 |
| --- | ---: | ---: | ---: |
| Degree 2 | 463.705 | 399.713 | 1.16× |
| Degree 16 | 686.499 | 880.229 | 0.78× |
| Softmax, cuDNN | 357.379 | 188.821 | 1.89× |

The comparison uses the same 199,906,304-parameter model, global batch 256,
global microbatch 64, four accumulation steps, and 2,048 prediction positions.
Source file hashes, initial parameter hashes, and synthetic input hashes match
[H100 reference job 199635](performance/h100_throughput/train-step.json). Each measurement
is the median of ten synchronized complete distributed training updates after
three warmups. Compilation, data loading, evaluation, and checkpointing are
excluded. These are short throughput measurements, not full production runtimes.

All 56 GPU attention correctness tests passed. The distributed benchmark
completed all three variants with finite losses. The B300 log includes a cuBLAS
TMEM warning from JAX 0.10.0's version check. Installed
`nvidia-cublas-cu12==12.9.2.10` includes the fix documented in
[NVIDIA's patch notes](https://docs.nvidia.com/cuda/cublas-patch-release-notes/#cublas-patch-release-12-9-2);
[JAX's warning condition](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax_plugins/cuda/__init__.py#L276-L290)
does not account for the backport. A communication initialization warning
resolved during warmup, outside timed measurements. No environment change was needed.

### Production continuation

The softmax run subsequently resumed from checkpoint 9,000 on four B300s in job
`199869`, preserving optimizer state and the packed-data cursor. It replayed
282 H100 updates after the saved checkpoint. Observed production updates took
about 0.21 seconds on B300 versus 0.38 on H100. The continuation completed in
38 minutes 54 seconds at step 19,074, with 10,000,269,312 training tokens and
terminal validation perplexity 14.6941766486. Scientific configuration, source,
and dataset were unchanged; hardware changes need not yield bitwise-identical updates.
Retained run/checkpoint details are in the [sweep provenance](provenance/manifest.json).

The benchmark ran in `sweep` concurrently with the four-H100 production job in
`standard`, demonstrating usable spare capacity beyond that production allocation
on 2026-09-18. Partition availability and policy are time dependent; no queue
snapshot or submission wrapper is needed to retain these scientific measurements.

- [Raw timing, loss, and configuration records](performance/b300/train-step.json).
- [Numerical comparison](performance/b300/comparison.json), [GPU tests](performance/b300/gpu-tests.xml), and [provenance](performance/b300/metadata.json).
