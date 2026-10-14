# Shared Transformer validation

This record preserves checks of the shared scaffold and runtime rather than any
experiment's scientific question: correctness and sampled performance of the
shared scaffold extraction, the Taylor attention kernel and throughput
benchmarks, and hardware and transfer qualification for the dK × Taylor-degree
recipes. Home cluster: **Kennel**. Artifact root:
`/data/cl/u/rafah/home/projects/functional-memory`. No model checkpoint is retained
for these synthetic checks. The authoritative record is
[provenance/manifest.json](provenance/manifest.json).

## Question and method

### Cubic GDN with original-key-space query transport

The cubic backend extends the same transported-score recurrence through degree
three, with a separate `beta_3=tanh(W_3 x)` write projection initialized to zero.
Original GDN weights and the initialized quadratic model's outputs are preserved.

The initial [A100 runtime measurement](runs/gdn-cubic-a100/runtime-baseline.json)
was numerically valid but took 24.6–24.9 seconds per update and exceeded the
allocation's runtime budget. The [occupancy benchmark](runs/gdn-cubic-a100/occupancy-benchmark.json)
showed that an explicit one-CTA-per-SM minimum in the Ampere PTX reduced register
spills. With four warps, forward and backward time fell from 546.8 to 153.3 ms;
the benchmark outputs and all six input gradients remained bit-for-bit identical.
The [occupancy sweep](runs/gdn-cubic-a100/occupancy-sweep.json) compared additional
launch settings; four warps and the one-CTA hint were retained for qualification.

The [full-recipe qualification](runs/gdn-cubic-a100/runtime.json) passed on 8
NVIDIA A100 80GB PCIe GPUs (`samoyed`, job `218895`, source `6945d1b8`):
196 CPU, 9 distributed, and 75 GPU tests,
a 20-update pilot, and exact fresh-process resumption from update 20 to 21.
The passed tests and initialization retained their original job provenance; the
standalone qualification reran the full pilot and exact resume with the same source.
The GPU checks include cubic outputs and all input gradients, production widths,
length-2,048 sequences, strong decay, and the quadratic limit.

Pilot validation and telemetry were finite with zero invalid attention counts.
All four coefficient projections received nonzero gradients and updates; the new
cubic coefficients had nonzero RMS in all ten layers. Steady updates took
8.794–9.069 seconds (median 8.831), with a conservative
full-run projection of 48.61 hours within the 70-hour guard
for the explicitly qualified 72-hour production allocation.
These checks qualify startup and numerical behavior on this allocation; long-run
stability, model quality, and production timing remain scientific run measurements.

### Quadratic GDN with original-key-space query transport

The quadratic backend implements `F_t(q) = alpha_t F_{t-1}(P_t q) +
v_t sum_(r=0..2) beta_{t,r}<q,k_t>^r`, with `P_t = I-lambda_t k_t k_t^T` in
the original key space. Its independent FP64 oracle updates explicit scalar,
linear, and quadratic states using full transition matrices. Tests compare
outputs and all six input gradients, including repeated keys, zero/full erasure,
the linear GDN limit, and the ordinary polynomial-attention limit.

[CPU checks](runs/gdn-quadratic-cpu/tests.json) pass the reference, initialization,
buffer ownership, gradient, checkpoint-resume, telemetry, acceptance, and run-record
contracts. The [baseline runtime check](runs/gdn-quadratic-h100/baseline-runtime.json)
on four H100 PCIe GPUs (`mastiff`, job `218413`, source `bf837570`) passed 157 CPU,
9 distributed, and 55 GPU tests, including production widths and length-2,048
strong-decay cases. Its 20-update full-recipe pilot had finite telemetry and
validation, zero invalid attention counts, and exact fresh-process resumption
from update 20 to 21. Both added coefficient projections received nonzero
gradients and produced nonzero coefficients in all ten layers.

The pilot measured 5.573–5.595 seconds per steady update and projected 30.65 hours
for the full recipe. The 24-hour allocation's 22-hour budget therefore rejected
this implementation; it is not qualified for that allocation. This is a runtime
budget failure, not a scientific numerical failure. Gradient norms were finite
(mean 9.07, maximum 11.84); every pilot update was clipped, with the constant-degree
write projection contributing most of the norm. Twenty updates do not establish
long-run stability or scientific performance.

The optimized kernel precomputes transformed chunk keys and separates diagonal
and cross-chunk geometry adjoints without changing the recurrence. On one H100
PCIe at the production per-device shape, forward and backward together measured
66.58 ms versus 121.99 ms before optimization. All 21 quadratic GPU tests pass
on source `9152f4ee`; [measurements](runs/gdn-quadratic-h100/benchmark-geometry-split.json)
retain timings and compiler resource use.

The optimized [full-recipe qualification](runs/gdn-quadratic-h100/runtime.json)
passed on four H100 PCIe GPUs (`mastiff`, job `218426`, source `46b776c9`):
157 CPU, 9 distributed, and 55 GPU tests, a 20-update pilot, and exact
fresh-process resumption from update 20 to 21. Synchronized steady updates took
3.381–3.609 seconds (median 3.433); the conservative full-run projection was
19.94 hours, within the 22-hour guard for a 24-hour allocation. These are pilot
timings, not a completed production measurement.

The optimized pilot had finite validation and telemetry, zero invalid attention
counts, and nonzero constant and quadratic coefficients in all ten layers.
Gradient norms were finite (mean 9.03, maximum 11.26); every pilot update was
clipped, with the constant-degree write projection contributing most of the
norm. The run's scientific measurements, including later stability and model
quality, remain outside this software qualification record.

The runtime qualifier uses the same 20-minute log-stall watchdog as the training
launcher and requires finite training, evaluation, exact resumption, and a pilot
projection within the production allocation before scientific training starts.

### Query-dependent tanh polynomial coefficients

The direct fused query-polynomial kernel passed all 12 H100 CUDA cases on
2026-10-08, job `210239` on `mastiff` (one H100 PCIe), source `254cab9`.
The compact [test record](runs/query-tanh-h100/tests.json) pins the source,
dependency lock, runtime settings, and JUnit evidence. The checks cover degrees
one, two, and three: FP32/BF16 forward and all parameter/input gradients against
an independent FP64 oracle, 2,048-token BF16 forward/backward comparison with the
blocked reference, and explicit failure for actual kernel overflow and nonfinite
causally visible keys. Future invalid keys do not contaminate earlier queries.

On 2026-10-09 UTC, all 16 tanh GPU cases across degrees one through four passed
within the 52-check GPU acceptance suite for run `rfee72428a45e`, job `213697`
on `akita` (four B300 GPUs), source
`ec96b6cbaf04dfefd059fafc601559652204fcc6`. The
[B300 test record](runs/query-tanh-b300/tests.json) pins the source, dependency
lock, runtime, test names, and transient acceptance JUnit checksum. Coverage
includes all input and parameter gradients in FP32 and BF16, the 2,048-token
BF16 forward/backward comparison at head width 64, and overflow/nonfinite-key
checks at every degree. This records GPU kernel qualification; full production
acceptance remains separately recorded by the run.

Run these checks in a qualified NVIDIA GPU allocation with the locked environment
and CUDA assembler settings from `launchers/kennel.sbatch`:

```bash
JAX_PLATFORMS=cuda scaffolds/transformer/.venv/bin/python -m pytest -q -m gpu \
  scaffolds/transformer/tests/test_query_tanh_polynomial.py
```

The recorded standalone result qualifies degrees one through three on H100.
Production still requires its own CPU/distributed/GPU checks, canonical initialization, full-recipe
pilot with validation, and exact fresh-process checkpoint resumption. B300 runs
must pass those checks on B300 before training.

### Query-dependent degree-one coefficients

The H100 check for input-dependent constant and linear coefficients passed all
five CUDA cases on 2026-10-07, job `209554` on `mastiff` (H100 PCIe). The compact
[test record](runs/input-dependent-m1-h100/tests.json) pins the tested source,
dependency lock, test cases, and successful log. Coverage includes an independent
FP64 forward/gradient oracle for blocked and fused FP32/BF16 execution, irregular
sequence lengths, all six Q/K/V/input/projection gradients, a 2,048-token BF16
comparison, and rejection of actual kernel overflow even when values are zero.

Run the mechanism tests in an H100 allocation with the locked environment and
the CUDA assembler settings from `launchers/kennel.sbatch`:

```bash
JAX_PLATFORMS=cuda scaffolds/transformer/.venv/bin/python -m pytest -q -m gpu \
  scaffolds/transformer/tests/test_input_dependent_coefficients.py
```

The shared acceptance gate additionally requires the mechanism's CPU/GPU tests,
canonical backbone and coefficient initialization checks, a full-recipe pilot,
and exact fresh-process checkpoint resumption in each production allocation.

### Shared scaffold

Does one shared model, attention interface, trainer, and checkpoint implementation
preserve the separate scientific implementations and their cost? Source `d493a41`
introduced the shared implementation. The mechanism interface keeps learned
parameters owned once by the model, permits global or per-layer/head coefficients,
and separates attention mathematics from reference/fused backend selection.

The recorded CPU comparisons check initialization, outputs, gradients, optimizer
updates, and restoration/continuation of checkpoints from the original source
revisions. Retained successful outputs include 16 frozen-source parity cases,
420 legacy adapter cases, shared CPU and distributed checks, and 89 fused GPU
cases. Counts are from their separate executions and are not additive coverage;
the captured pytest outputs are summarized in the manifest's `execution_evidence`.
The H100 GPU-validation allocation 200759 and benchmark allocation 200760 both
completed with exit code 0:0; captured Slurm-step stdout/stderr is retained under
`scaffolds/transformer/validation/logs/`, outside Git.

The canonical runtime integration additionally passed 20 CPU tests: four-device
replica integrity, corruption rejection, exact checkpoint continuation, execution
provenance, and launcher policy. The existing two-device accumulation comparisons
passed seven cases across fixed, learned, and odd attention. These checks do not
replace GPU qualification for a new mechanism.

## Standard gradient diagnostics

The [gradient logging check record](runs/gradient-norm-diagnostics/tests.json)
pins the source and exact commands: 67 CPU checks passed, plus six two-device
averaging and checkpoint-resume checks in a separate execution. Coverage includes
the full averaged gradient norm before clipping, zero-norm handling, spikes on
updates between log steps, offline W&B mirroring, and unchanged numerical-failure
checks. The default workspace constructed four gradient-norm panels and two
clipping panels locally; no W&B view was published. Training updates and saved
optimizer state are unchanged. These are synthetic software checks, with no
retained checkpoint or GPU throughput measurement.

## Training telemetry

Checks: `tests/test_telemetry.py` (in acceptance's CPU stages). On two CPU devices,
updates with and without telemetry give bit-identical loss, gradient norm, and
parameters. Recorded activation statistics match direct computation, and group
gradient norms combine to the global norm. Runs of the full training loop record
window summaries, gradient spikes, and numerical-failure statistics.

Cost on four B300s (`akita`, job 212284, source `b039b6a`): reports in
[`runs/telemetry-overhead-b300/`](runs/telemetry-overhead-b300/). Each recipe ran four
alternating off/on trials of 40 timed updates (after 5 warmup updates) from the same
initial state, synthetic tokens, and update RNGs. The timing includes fetching the
packed statistics every update, as the training loop does.

| Recipe (run) | Off, ms/update | On, ms/update | Overhead | Statistics |
| --- | ---: | ---: | ---: | ---: |
| Learned m = 4 + head RMSNorm (`rc2b555b905dd`) | 454.4 | 457.5 | +0.68% | 79 |
| Softmax (`rdf00da0866a8`) | 193.3 | 196.5 | +1.64% | 63 |
| Exp + head RMSNorm (`r8ad66f3912c2`) | 369.0 | 372.0 | +0.83% | 76 |
| Normalized fixed Taylor m = 4 (`r512daaa27a15`) | 442.2 | 445.7 | +0.78% | 63 |

Telemetry adds 3.0–3.5 ms per update, about one minute per 19,074-update run. Every
trial pair shows it. Losses with and without telemetry differ by 2.3e-5–7.9e-5 over
40 timed updates, the same order as repeated identical trials without it (2.4e-5–5.3e-5,
GPU nondeterminism).

On two H100 PCIe (`mastiff`, job 212211, source `57784b2`, shared with other jobs;
[`runs/telemetry-overhead-h100/`](runs/telemetry-overhead-h100/)), the same benchmark
measured +0.59%, +1.12%, +0.19%, and +1.26% for the four recipes in the order above
(updates of 1.35–1.73 s). Individual trials ranged from −0.8% to +7.9% because of the
node's other load; losses again differed by no more than between identical trials.

## Sampled performance

Benchmarks used one H100, four layers of width256, eight heads, sequence length512,
synthetic vocabulary1024, BF16 computation, five warmups, and30 measured optimizer
updates. Exact recipes and the dependency lock are in `configs/`; raw records
are in `runs/`. Data loading, gradient accumulation, diagnostics, and checkpointing
were excluded from the timed operation.

| Mechanism | Original median ms | Shared median ms | Peak allocated bytes, both |
| --- | ---: | ---: | ---: |
| taylor | 1.9526 | 1.9988 | 186,418,432 |
| global | 2.0004 | 1.8189 | 186,419,200 |
| per_head | 1.6777 | 1.8249 | 186,419,200 |

All sampled pairs have identical compiled buffer sizes and peak allocated memory.
Single short timing samples do not establish speed equality or full-scale training
throughput. BF16 trajectories differ slightly; correctness is assessed by the
independent parity tests with their explicit tolerances.


## Taylor attention kernel and throughput benchmarks

These runs measured the fused Taylor attention kernel used by the degree and
joint sweeps in the [dK × Taylor-degree record](../../../experiments/attention_scaling/README.md).
Full updates use four GPUs, global batch 256, global microbatch 64, four
accumulation steps, 2,048 prediction positions, and the 199,906,304-parameter
dK=64 model. Timings are medians of ten synchronized calls after three warmups;
compilation, data loading, evaluation, and checkpointing are excluded unless stated.

| Run | Measurement | Hardware / job | Source |
| --- | --- | --- | --- |
| [`taylor-blocked-baseline`](runs/taylor-blocked-baseline/) | Blocked versus first fused, full update | Four H100 SXM / 199623 | `3537262e` |
| [`taylor-attention-kernel`](runs/taylor-attention-kernel/) | First fused kernel, attention only | H100 PCIe / 199626 | `0b7a37ba` |
| [`taylor-throughput-h100`](runs/taylor-throughput-h100/) | Optimized full updates, profile, tile tuning, real-data pilot | Four H100 SXM / 199635 | `0f655fb7` |
| [`taylor-train-step-b300`](runs/taylor-train-step-b300/) | Optimized full updates on B300 | Four B300 / 199854 | `1a8836a7` |

The optimized H100 update takes 0.464 s at degree 2 and 0.686 s at degree 16,
compared with 1.644 s and 1.922 s for the original blocked code (3.55× and 2.80×).
Softmax (cuDNN) takes 0.357 s.

| Degree | Blocked forward | Fused forward | Blocked fwd+bwd | Fused fwd+bwd | fwd+bwd speedup |
|---|---:|---:|---:|---:|---:|
| 0 | 11.84 ms | 4.88 ms | 31.32 ms | 7.51 ms | 4.17× |
| 2 | 14.31 ms | 3.56 ms | 56.00 ms | 21.92 ms | 2.55× |
| 4 | 14.30 ms | 3.68 ms | 56.09 ms | 21.86 ms | 2.57× |
| 8 | 14.37 ms | 4.21 ms | 60.58 ms | 25.33 ms | 2.39× |
| 16 | 16.85 ms | 5.33 ms | 63.07 ms | 31.28 ms | 2.02× |

The attention-only kernel benchmark above uses the per-device BF16 shape
`[16, 2048, 16, 64]` on an H100 PCIe; compiler-estimated temporary buffers fell
from 848.2 MiB (degree 2) and 1,292.5 MiB (degree 16) to 130.0 MiB. The largest
BF16 relative L2 differences against the blocked reference were 8.7e-4 for outputs
and 3.6e-3 for Q/K/V gradients. The first fused implementation passed 151 CPU,
one two-device, and 46 GPU tests; the optimized one passed 56 GPU tests on H100
and 56 on B300.

| Variant | Blocked (s) | First fused (s) | Optimized (s) | Attention time/update | Attention share |
|---|---:|---:|---:|---:|---:|
| Degree 2 | 1.644 | 0.939 | 0.464 | 175.9 ms | 37.9% |
| Degree 4 | — | — | 0.485 | 200.2 ms | 41.1% |
| Degree 8 | — | — | 0.533 | 251.4 ms | 46.9% |
| Degree 16 | 1.922 | 1.230 | 0.686 | 416.3 ms | 59.8% |
| Softmax (cuDNN) | — | 0.358 | 0.357 | 60.2 ms | 16.8% |

The optimization removed needless FP32 conversions and split mixed FP32/BF16
products into three BF16 components, keeping FP32 polynomial and normalization
arithmetic. All 28 tile candidates passed parity; the existing 64×64 tile won
degree 2, and a degree-16 alternative improved the isolated call by only 2.1%,
so the shared setting was kept. Global microbatch 128 changed full-step speed by
under 2%. A 100-update degree-16 pilot on the real packed data took 0.6997 s per
warmed update versus 0.6865 s on resident synthetic input.

| Attention | Four H100 SXM, ms/update | Four B300, ms/update | B300 / H100 throughput |
| --- | ---: | ---: | ---: |
| Degree 2 | 463.705 | 399.713 | 1.16× |
| Degree 16 | 686.499 | 880.229 | 0.78× |
| Softmax, cuDNN | 357.379 | 188.821 | 1.89× |

Hardware choice therefore depends on the attention variant; the B300 run does
not establish optimal B300 tuning. Its log's cuBLAS TMEM warning comes from JAX
0.10.0's version check; the installed `nvidia-cublas-cu12==12.9.2.10` includes
the fix. Raw timing and loss samples, tile comparisons, correctness summaries,
and compact profile aggregates are in each run directory; full traces and pilot
checkpoints were not retained. Pre-benchmark JUnit outputs are summarized in the
manifest.

## Hardware and transfer qualification for dK × Taylor-degree recipes

Before and during the joint sweep, the shared runtime was qualified for the 20
joint recipes on several GPU types. Each check ran the real model and production
batch for 32 updates, evaluation and checkpointing, and a separate restore from
step 32 to 33, and compared initial model/input identities and attention
outputs/gradients. A completed validation job means its harness finished, not
that every coordinate passed. The recipes are `configs/kdt-qualification-*.yaml`.

| Run | Allocation / transfer mode | Full-model outcome | Jobs |
| --- | --- | --- | --- |
| [`kdt-b300-initial`](runs/kdt-b300-initial/), [`kdt-b300-native`](runs/kdt-b300-native/) | B300, native | 20/20 passed (five reuse the first qualification) | 200733, 200753 |
| [`kdt-a100-native`](runs/kdt-a100-native/) | A100, native | 1/20 passed; 19 had nonfinite validation at step 32 | 200754 |
| [`kdt-a6000-host-staging`](runs/kdt-a6000-host-staging/) | A6000, host-staged | 20/20 passed, including resume and replica audits | 200781–200786 |
| [`kdt-a100-transport`](runs/kdt-a100-transport/) | A100, host-staged | Representative dK8/m4 passed; full grid not rerun | 200853 |

H100 native execution was qualified by the joint sweep's own pre-launch acceptance
(job 200502), recorded with that experiment.

On A6000, native peer transfers produced stale or corrupt values despite
successful CUDA calls: a 72-comparison probe found six failures while host-backed
transfer was exact ([`kdt-a6000-transport`](runs/kdt-a6000-transport/)). Disabling
NCCL peer-to-peer alone fixed a collective hang but not direct JAX GPU copies.
The generic fix stages state and batches through host memory and audits every
replica; attention mathematics and precision are unchanged. Two early pilots ran
out of memory before the first update; reserving 90% of the device pool
(`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`) let the full qualification pass.

On A100 node `samoyed`, native replication left GPU0 exact but 93
model/optimizer-state leaves mismatched on each of GPUs 1–3, and all 131,136
training token positions on those replicas were corrupted. Native and
NCCL-disable-only modes failed; host staging alone and host staging with NCCL
disabled both passed replication and step 32→33 restore checks (step-32 NLL
9.8021605 and 9.8021476). An independent direct-CUDA probe, without JAX or NCCL,
found all 36 peer-enabled copies across 12 GPU pairs corrupt and all 36
peer-disabled copies exact. The GPUs were in translated IOMMU DMA-FQ domains
without NVLink, on driver 560.35.05; the IOMMU configuration is a leading
explanation, not a proven cause. This is one coordinate, not full A100
qualification. Kennel launcher policy based on these results is in
[`CLUSTERS.md`](../../../CLUSTERS.md).

## Fused GLA kernel qualification on H100 and B300

The chunked Triton GLA kernels (`attention/backends/chunked_gla_triton.py`) first
ran on GPUs for the [GLA d_K sweep](../../../experiments/attention_scaling/README.md). Their 21
GPU tests (`tests/test_gla_kernel.py`) compare fused outputs, final states, and
all four gradients with a sequential FP32 recurrence for d_K 2–128 (and a 76/152
case), zero and extreme decays, and mixed FP32/BF16 operands. Two defects were
found and fixed before production:

- **Toolkit.** jaxlib assembles the bridge's Triton PTX with the first `ptxas`
  among its CUDA roots, normally a node's `/usr/local/cuda`. `dogo`'s CUDA 12.6
  rejects Triton 3.6's PTX 8.7, so 20 of 21 cases failed on H100. Launchers set
  `CUDA_HOME` to the locked environment's CUDA 12.9 `nvidia/cuda_nvcc`.
- **B300 precision.** With BF16 operands upcast inside the kernel, Triton 3.6 on
  sm_103 lowered the IEEE FP32 products at TF32-level precision: 12 of 21 cases
  failed, and in the production mixed case (FP32 Q/K after RoPE, BF16 V) the
  output error against an FP64 recurrence was 4.0e-4 on B300 versus 1.5e-7 on
  H100. Operands now enter the kernels as FP32 (an exact conversion); both GPU
  types then match the FP64 recurrence to about 1e-7 in FP32 and agree with each
  other at printed precision.

A third failure was a test defect: at log α = −100 the exact log-decay gradient is
about 1e-51 (0 in FP32), so a relative-error threshold judged 5e-13 of noise; that
gradient is now compared with the input-gradient scale.

Every production allocation then passed the sweep's acceptance at source `4af147e`
(49 GLA CPU tests, 21 kernel tests, a 20-update full-recipe pilot on the fused
backend, and an exact fresh-process resume): H100 in jobs 201793 (first attempt)
and 201794; B300 in jobs 201792, 201793 (after requeue), 201795, and 201796.
Per-run acceptance is summarized in each GLA run's `record.json`.

The kernel is slow rather than wrong. On one H100 at the per-device production
shape (16 × 2,048 tokens, 16 heads, d_V 64), fused GLA forward+backward took 3.9,
9.9, and 25.9 ms for d_K 8, 64, and 128, against about 1.0–1.4 ms forward for all
of a layer's dense matmuls. Its IEEE FP32 products run on CUDA cores rather than
tensor cores, and its backward grows faster than its forward in d_K; end-to-end
B300 throughput fell from 1.68M tokens/s at d_K 8 to 0.61M at d_K 128. This was a
development measurement, not a retained benchmark run.

## Fused learned-polynomial kernel qualification on H100 and B300

The fused unnormalized kernel with learned per-head coefficients
(`attention/backends/fused_learned_polynomial.py`) first ran for the
[learned-coefficient unnormalized Taylor sweep](../../../experiments/attention_scaling/README.md).
It reuses the fixed unnormalized kernel's tiles and products and adds per-head
coefficient gradients. Its 26 GPU tests (`tests/test_learned_polynomial_fused_attention.py`)
compare the forward and all four VJPs, including the coefficients, with the blocked
reference and an FP64 power sum for degrees 1–4 at odd widths and tile boundaries,
full 2,048-token BF16 context, scores above 40, causality, invalid-score poisoning,
and the fixed-coefficient kernel at Taylor values. Before launch they passed on one
H100 (`dogo`, job 201994) and one B300 (`akita`, job 201993) at the pre-launch tree.
No defects were found on either GPU type.

Every production allocation then passed the sweep's acceptance at source `788a085`
(114 CPU tests, 74 GPU tests including the fixed unnormalized kernel's 48, the
initialization check, a 20-update full-recipe pilot on the fused backend, and an
exact fresh-process resume): B300 in jobs 201997, 201999, and 202000 and H100 in job
201998. On matched B300 allocations the pilots' steady update times, 0.348 s (m=1)
and 0.452 s (m=4), were within 2% of the fixed-coefficient runs' 0.342 s and 0.455 s.
These are pilot measurements, not a retained benchmark.

## QK-norm cosine scores on B300

QK-norm (`attention/qk_norm.py`) is a score option, not a kernel: the attention
module unit-normalizes Q and K and pre-scales the queries so that the unchanged
kernels compute `exp(gamma) cos(q, k)`. It first ran for the
[QK-norm attention record](../../../experiments/attention_scaling/README.md)
(exact softmax through cuDNN, `d_K = 64`). Its five GPU tests
(`tests/test_qk_norm.py`) compare cuDNN softmax and the fused Taylor, unnormalized,
and learned-polynomial kernels on cosine operands with the blocked reference at the
full 2,048-token BF16 context, including the log-scale gradient; 44 CPU tests cover
the mathematics against FP64. Before launch the GPU tests passed on one B300
(`akita`, job 202131) at the scaffold-change tree `dd6f2f4`; the production
allocation then passed the record's acceptance at source `aa5b6eb` (171 CPU tests,
5 distributed, 12 GPU tests, the softmax-backbone initialization check, a 20-update
full-recipe pilot on cuDNN at 0.202 s per update, and an exact resume) in job
202215 on four B300s. H100 is not qualified for QK-norm: the first submission's two
H100 attempts were preempted before their GPU stage. That first submission also
exposed no defect in QK-norm; it failed on B300 because its GPU stage listed
`tests/test_softmax_backends.py`, whose width-256 cuDNN cases B300 rejects
(hidden dimension above 128 with `d_qk != d_v`), a known limit outside the
recipe's widths.

## Normalized learned polynomials and QK-norm on B300 and H100

Normalized attention with learned polynomial coefficients
(`attention/learned_polynomial.py`) initially used only the blocked reference; QK-norm scores
compose with every kernel as pre-scaled operands. Both first ran for the
[QK-norm learned Taylor sweep](../../../experiments/attention_scaling/README.md).
The new GPU cases (`tests/test_learned_polynomial_normalized.py`) run the blocked
normalized kernel at the 2,048-token BF16 context on cosine operands and compare a
causal prefix, including the coefficient and log-scale gradients, with FP64; the
degree-1 comparison runs at scale 2 because at scale 8 the signed row sums of
`1 + s` make the constant coefficient's gradient a difference of near-equal FP32
sums (30% off FP64 on B300, a property of the kernel, not the code). Before launch
these 4 cases, the 5 QK-norm gates, and the 26 fused learned-polynomial cases passed
on one B300 (`akita`, job 202336; two earlier jobs failed only the scale-8 degree-1
comparison). Every production allocation then passed the sweep's acceptance at
source `696ffb5` (673-test CPU suite before launch; per allocation 202 CPU tests,
5 distributed, 83 GPU tests, the softmax-backbone initialization check, a
20-update pilot, and an exact resume): B300 in jobs 202340, 202342, 202343, and
202344, and H100 in jobs 202337, 202338, 202339, and 202380, which is also the
first H100 qualification of QK-norm. Pilot update times were 1.01–1.03 s for the
blocked normalized kernels on four B300s (1.97 s on four H100s) against 0.46–0.48 s
for the fused unnormalized kernel on either GPU.

## Fused normalized learned polynomials

`attention/backends/fused_normalized_polynomial.py` implements both direct learned
coefficients and even-degree sum-of-squares factors. Direct coefficients use two
passes to normalize by the final signed row sum; the nonnegative family evaluates
the two factors directly and uses one online forward pass. Both reverse passes
differentiate Q, K, V, and all parameters without storing a sequence-square array.
The model owns parameters and sharing; the kernel receives per-head arrays and
JAX reduces gradients through any shared broadcast.

The qualification suite compares outputs and all gradients with an independent
FP64 oracle, exercises signed and zero row sums, factor roots, masked overflow,
odd dimensions, shared coefficients, BF16 QK-norm at 2,048 tokens, and a small
Transformer's optimizer and checkpoint/resume. Polynomial values, row sums,
backward output residuals, and parameter-gradient reductions use FP32. The shared
tensor-core dot policy is unchanged. Near polynomial roots and cancelling signed
sums, small score or accumulation errors can be amplified; these cases use
conditioning-aware bounds. There is no denominator clipping or epsilon fallback.
The strict replay test uses packed tokens with distinct embedding indices within
each microbatch: repeated-index BF16 embedding-gradient accumulation can vary
with GPU atomic update order. It checks the complete final parameter/optimizer
state exactly, plus durable checkpoint hashes and the resumed batch/RNG identity.
This fixture does not claim bitwise determinism for arbitrary training data.

Qualification passed 40 GPU tests on H100 (`dogo`, job 202628) and 40 on B300
(`akita`, job 202629), including the strict replay fixture. The complete CPU suite
passed 719 tests, with 13 expected skips. The two-GPU update comparison is job
202626. Exact source commits, configuration hashes, hardware, attempts, runtime
settings, test counts, and retained log hashes are in the
[manifest](provenance/manifest.json). Existing experiment worktrees and their
resolved backends were not changed.

The rational parameterization (numerator `a_0..a_m`, denominator
`1 + sum_(r>=1) b_r s^r`) extends the same two-pass kernel: the first pass sums the
denominator polynomial, and the reverse pass carries the numerator sensitivity
`(dY . V) / Z` and the denominator sensitivity `-(dY . Y) / Z` through separate
Horner recurrences. Its qualification adds rational cases to the same suite: FP64
value and VJP comparisons at degrees 1–4 and 8, odd widths and tile boundaries, zero
denominator sums, masked overflow, shared parameters, prefix independence, BF16
QK-norm at 2,048 tokens at degrees 4 and 8, and the Transformer training/resume
fixture. On one H100 (`mastiff`, job 207486) the fused normalized-polynomial,
normalized learned-polynomial, and QK-norm GPU suites passed 63 cases, 14 of them
rational, at the scaffold source of commit `8cee781f80db`
([record](runs/rational-polynomial-h100/tests.json)); the CPU suite passed 1,055
tests with 14 expected skips.

The kernel describes each head's parameters with a static layout, so the denominator
can have its own degree and a nonnegative form, `(1 + sum u_r s^r)^2 + (sum w_r s^r)^2`.
New cases cover a linear and a nonnegative quadratic denominator with numerators of
degree 1–8 against FP64. At 2,048 tokens in BF16, one head's QK-norm log-scale
gradient nearly cancels for the quadratic denominator: the fused and blocked BF16
paths agreed to 2.5% while both were about 70% from FP64, and fused was no farther
from FP64 than blocked on any quantity. That check now compares both paths with an
FP32 evaluation of the blocked reference. On one H100 (`dogo`, job 208617) the GPU
suites passed 82 cases, 33 rational, at commit `8fdf2cbbb301`
([record](runs/rational-denominators-h100/tests.json)); the CPU suite passed 1,099
tests with 14 expected skips. A degree-8 nonnegative denominator then passed the
same suite through `launchers/qualify_normalized_polynomial.sbatch`: 81 cases, 8 of
them for that denominator, on one H100 (`dogo`, job 208968, commit `536e3e3`;
[record](runs/rational-nonnegative-degree8-h100/)). Learned denominator constants, `b_0 + b_1 s`
and `q1(s)^2 + q2^2`, then passed it with 97 cases, 16 of them for those denominators
(`dogo`, job 208982, commit `1b0f018`; [record](runs/rational-learned-constant-h100/)).

<!-- BEGIN GENERATED NORMALIZED POLYNOMIAL THROUGHPUT -->
Kernel forward and backward, including all parameter gradients:

| GPU | Family | Degree | Blocked (ms) | Fused (ms) | Speedup |
|---|---|---:|---:|---:|---:|
| H100 | Unrestricted | 1 | 42.270 | 4.485 | 9.43× |
| H100 | Unrestricted | 2 | 42.313 | 4.871 | 8.69× |
| H100 | Unrestricted | 3 | 42.393 | 5.521 | 7.68× |
| H100 | Unrestricted | 4 | 42.429 | 6.071 | 6.99× |
| H100 | Nonnegative | 2 | 47.397 | 4.557 | 10.40× |
| H100 | Nonnegative | 4 | 43.267 | 5.929 | 7.30× |
| B300 | Unrestricted | 1 | 21.712 | 7.066 | 3.07× |
| B300 | Unrestricted | 2 | 21.739 | 8.463 | 2.57× |
| B300 | Unrestricted | 3 | 21.881 | 9.470 | 2.31× |
| B300 | Unrestricted | 4 | 22.219 | 10.392 | 2.14× |
| B300 | Nonnegative | 2 | 23.881 | 7.429 | 3.21× |
| B300 | Nonnegative | 4 | 22.108 | 10.042 | 2.20× |

Complete Transformer updates on two B300 GPUs:

| Family | Degree | Blocked (ms) | Fused (ms) | Speedup |
|---|---:|---:|---:|---:|
| Unrestricted | 2 | 995.449 | 447.108 | 2.23× |
| Unrestricted | 4 | 1013.012 | 521.030 | 1.94× |
| Nonnegative | 2 | 1077.729 | 407.212 | 2.65× |
| Nonnegative | 4 | 1009.427 | 512.678 | 1.97× |
<!-- END GENERATED NORMALIZED POLYNOMIAL THROUGHPUT -->

Run the qualification from a clean, committed worktree with its own environment:

```bash
SOURCE=$(git rev-parse HEAD)
RUN=normalized-polynomial-h100
mkdir -p scaffolds/transformer/validation/logs/$RUN
sbatch --nodelist=dogo \
  --output=scaffolds/transformer/validation/logs/$RUN/slurm-%j.out \
  scaffolds/transformer/launchers/qualify_normalized_polynomial.sbatch "$SOURCE" "$RUN" kernel
# Use akita for B300. For complete 200M-model update comparisons, request
# --gres=gpu:2 and pass "updates" in place of "kernel" with a fresh RUN.
python -m scaffolds.transformer.benchmarks.summarize_normalized_polynomial --check
```

The kernel benchmark uses BF16 `[16, 2048, 16, 64]` Q/K/V, QK-norm at scale 8,
Taylor-initialized parameters, three warmups, and ten synchronized measured calls.
It includes every input/parameter gradient; compilation, diagnostics, transfers,
and the rest of the Transformer are excluded. The update benchmark uses two GPUs,
a global microbatch of 32, and four accumulation steps (batch 128), preserving the
production per-device microbatch of 16. Its synthetic tokens are already on the
devices. Memory numbers are compiler buffer estimates, not allocator peaks.

## Fixed kernels with canonical QK normalization

On 2026-09-27, source `84dedd777c2f1def16ef9ef74ea11ec6c94cdda9` passed
**85 compiled GPU tests on H100** (`dogo`, job 203015) and **85 on B300**
(`akita`, job 203016), with no failures or skips. The selection covers static
Hermite monomial coefficients, normalized degree-two square factors, fixed
unnormalized Taylor controls, the normalized polynomial kernels, and learned
per-head QK scales. Twelve BF16 cases at context 2048 and `d_K=d_V=64` compare
fused outputs and Q/K/V/log-scale gradients against the blocked implementation.
The CPU composition tests also compare against an independent FP64 Hermite
recurrence and verify backbone initialization, optimizer parameter ownership,
and checkpoint resume.

These are software checks, not training outcomes. Each scientific allocation
still runs its own full-recipe pilot and resume gate. The signed and exact-zero
normalizer policies are unchanged. Tests use seeded synthetic inputs; no
training checkpoints or external datasets are retained by these kernel checks.

The [manifest](provenance/manifest.json) records both runs, source identity,
hardware, test-result hashes, and retained log hashes. Execute
`validation/qualify_hermite_qk.sbatch` from a checkout of the qualified source
with its own locked environment; it also selects the experiment's QK composition
tests at that source. Initial submissions used:

```bash
sbatch --gres=gpu:h100:1 --nodelist=dogo \
  --output=scaffolds/transformer/validation/logs/hermite-qk-h100-%j.log \
  scaffolds/transformer/validation/qualify_hermite_qk.sbatch \
  84dedd777c2f1def16ef9ef74ea11ec6c94cdda9 H100
# For B300, use --gres=gpu:b300:1 --nodelist=akita and pass B300.
```

## Independent GDN erase/write kernel

Source `d9c5aea542be96cb6c2217e85cb965040d441f7e` failed qualification on one
H100 PCIe on Kennel (`mastiff`, job `214022`) on 2026-10-09:
26 tests passed and two failed. Both failures used identical unit keys,
unit erase strength, and no decay. Forward comparisons passed, but relative
key-gradient errors were 0.5846% in FP32 (0.5% tolerance) and 4.7263% in BF16
(2% tolerance). The precision audit below identifies cancellation amplified by
reduced-precision arithmetic in this fixture. The repaired kernels passed the
H100 qualification described under [FP32 chunk precision repair](#fp32-chunk-precision-repair).

The [record](runs/gdn-independent-h100/tests.json) pins source, dependency lock,
hardware, and JUnit evidence. The matching log is
`validation/logs/gdn-independent-214022.log` in the experiment worktree on Kennel.
The dependent production job was cancelled before allocation. No training
result follows from this software check.

### Repeated-key precision audit

On 2026-10-10, diagnostic job `215578` completed on an available RTX A6000 on
Kennel (`terrier`), source `99c415b200fa5e8767c4d438fccc7e1b288df684`.
It reproduced the H100 BF16 key-gradient error exactly. The diagnostic used the
same seed-44 fixture and compared every output/gradient channel against an
independent analytic oracle; it did not stop at the first failed tolerance.

For identical keys and queries `e1`, no decay, and full erasure, write
`a_t = beta_t v_t`, `a_0 = 0`, output cotangent `c_t`, and `s = 1/sqrt(d_K)`.
The exact output is `s a_t`; the key gradient is
`s e1 [c_t^T(a_t - 2 a_{t-1})]`, erase gradient is
`-s c_t^T a_{t-1}`, and log-decay gradient is zero. The
[CPU oracle record](runs/gdn-independent-precision-cpu/diagnostics.json)
confirms the scan and exact two-chunk equations in FP64. FP32 scan key-gradient
relative error is `8.0e-8`. The norms of the two separately accumulated key-gradient
terms sum to 6.68 times the final gradient norm, demonstrating cancellation.

The [default TF32 record](runs/gdn-precision-215578/tf32.json) and
[IEEE record](runs/gdn-precision-215578/ieee.json) distinguish product precision
from intermediate storage. All percentages below are relative L2 key-gradient
errors against the analytic oracle, with the same BF16-representable inputs:

| Implementation / arithmetic | Key-gradient error |
| --- | ---: |
| Independent gates, native BF16 | 4.726266% |
| Original tied GDN with equivalent pre-scaled values, native BF16 | 4.726266% |
| Independent gates, native BF16, IEEE default for FP32 dots | 4.726266% |
| Independent gates, promoted FP32, TF32 products | 0.494907% |
| Independent gates, promoted FP32, IEEE products | 0.000137% |

The original-kernel comparison uses the identity
`independent(q,k,v,g,beta,erase) = tied(q,k,(beta/erase)v,g,erase)` for nonzero
erase, here exactly one. Division is used only in this diagnostic, never in the
production independent-gate implementation. Original and independent kernels
produce identical output, query/key gradients, and log-decay gradients in this
BF16 fixture. Thus the observed discrepancy is shared arithmetic, not a new
effect of splitting the write and erase gates. FP32-input key-gradient error also
falls from 0.559970% with TF32 products to 0.000143% with IEEE products.

The BF16 stress fixture exposes a 9.738% erase-gradient error and a log-decay
gradient with L2 norm 0.756763 instead of zero. Promoted FP32 plus IEEE reduces
those to 0.000182% and `4.72e-5`, respectively. The latter residual exceeded the
then-existing `2e-5` absolute allowance for an exactly zero reference. This
diagnostic changed no assertion or production kernel and did not constitute a
passing qualification. The production repair and H100 qualification are recorded
below; the repaired kernels have not been qualified on B300. These synthetic
findings establish no language-model training outcome.

Reproduction from the committed worktree and its locked environment:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 scaffolds/transformer/.venv/bin/python \
  scaffolds/transformer/validation/gdn_precision_cpu.py
sbatch --nice=0 --partition=debug --nodelist=terrier --gres=gpu:a6000:1 \
  --job-name=gdn-precision \
  --output=scaffolds/transformer/validation/logs/gdn-precision-%j.log \
  scaffolds/transformer/validation/diagnose_gdn_precision.sbatch \
  99c415b200fa5e8767c4d438fccc7e1b288df684
```

The GPU launcher runs fresh processes and separate compilation caches with
`TRITON_F32_DEFAULT=tf32` and `ieee`. Explicit dot-product precision settings
inside upstream kernels are preserved. The log is
`validation/logs/gdn-precision-215578.log` in the experiment worktree on Kennel;
the compact records pin the source, scripts, dependency lock, device, and runtime.

### FP32 chunk precision repair

The repaired tied and independent kernels retain FP32 throughout the chunk
algebra, including the triangular inverse, transformed values, chunk states and
backward partials. The direct and WY key-gradient contributions combine before
the final dtype conversion. FP32 products use TF32x3, including the triangular
solve, through an explicit per-call bridge setting that participates in the
compiled-kernel cache identity. Public outputs and input gradients retain their
input dtypes. Saved Q/K/V remain in their original dtype and are promoted again
for backward recomputation, limiting retained activation memory. The recurrence,
gate parameterization and initialization are unchanged.

On 2026-10-11, source `4b7573e31a1475ef01f7e6313ab36f14a587b0ee` passed
**34 GPU tests with no failures or skips** on one NVIDIA H100 PCIe on Kennel
(`mastiff`, job `216969`). The allocation completed successfully in 21m11s,
including the kernel benchmark. The
[qualification record](runs/gdn-independent-precision-h100/tests.json) pins the
source, locked dependencies, runtime and JUnit evidence. Separately,
[74 CPU tests](runs/gdn-precision-repair-cpu/tests.json) passed at kernel source
`9b56ed2d66849e346c36f0228f49dbbebf7f3fba`, covering model/reference behavior,
bridge precision isolation and caching, and acceptance.

The [runtime snapshot](runs/gdn-independent-precision-h100/runtime.json) records
the allowlisted environment observed in the allocation's live benchmark process;
the preceding GPU tests used the same settings. The passing JUnit is published
through the supported shared-test cache under its matching acceptance identity,
allowing production acceptance to reuse this GPU-test stage while retaining its
own full-recipe pilot and resume checks.

The repeated-key regression now uses independent FP64 analytic outputs and all
gradients, both coordinate-axis and dense exactly unit keys, and tied and
independent gates. Nonzero quantities use tighter relative L2 tolerances:
`5e-5` for FP32 and `5e-3` for BF16. For the exactly zero decay gradient, the test
uses an FP32 roundoff budget scaled by dimensions and the uncancelled products:
`gamma_(V + 2C) * ||s sum_v |c_t,v a_(t-1),v|||_2`, with
`gamma_n = n*u/(1-n*u)`, FP32 unit roundoff `u`, and chunk length `C=64`.
This is an explicit numerical error budget, not a proven bound for every
operation in the composed kernel. It does not depend on the observed residual
or BF16 epsilon. The other recurrence-comparison tolerances are unchanged, and
all gradient channels are checked even when an earlier channel fails.

The [benchmark record](runs/gdn-independent-precision-h100/benchmark.json)
measures BF16 interfaces with `T=2048`, `H=16`, and `K=V=64`. Each case excludes
compilation and three warmups, then times twenty calls with all returned tensors
synchronized. Every output and gradient was finite.

| Local batch | Gate mode | Forward median (ms) | Forward + backward median (ms) |
| --- | --- | ---: | ---: |
| 1 | Tied | 0.462 | 0.883 |
| 1 | Independent | 0.453 | 0.912 |
| 32 | Tied | 4.716 | 15.777 |
| 32 | Independent | 4.872 | 16.268 |

Batch 32 matches each GPU's attention shape in the canonical two-device recipe.
The compiler estimates 3.004 GiB of temporary buffers for either batch-32
forward-plus-backward call, excluding arguments and returned tensors. This is
an attention-kernel estimate, not a measured allocator peak or full-model memory
measurement. The timings compare the two repaired kernels and do not measure
the repair's overhead against the previous implementation. Full-model memory
fit, update throughput, validation and checkpoint resume remain each scientific
allocation's acceptance checks. This repair is qualified on H100 PCIe only;
B300 qualification remains outstanding.

The retained log is `validation/logs/gdn-repair-216969.log` in the experiment
worktree on Kennel. The [manifest](provenance/manifest.json) pins the completed
log, CPU tests, H100 qualification and benchmark. Reproduction on H100:

```bash
sbatch --nice=0 --partition=debug --nodelist=mastiff --gres=gpu:h100:1 \
  --output=scaffolds/transformer/validation/logs/gdn-repair-%j.log \
  scaffolds/transformer/validation/qualify_gdn_independent.sbatch \
  4b7573e31a1475ef01f7e6313ab36f14a587b0ee H100
```

### Independent gate storage

Source `d3d93bc6ccaa8a76fc4b29124c80988ae1593405` also fixes independent-gate
initialization: the write and erase projections keep exactly equal initial values
but own separate JAX buffers. Cloning the NNX parameter object and assigning its
array directly had shared storage. A donated update then rejected the duplicate
buffer on the source device; replication created separate buffers on the other
device, which could wait indefinitely for its peer at the initial collective.
The [CPU record](runs/gdn-independent-precision-h100/alias-repair-cpu.json)
captures the reproducer and 39 passing GDN/initialization tests, including donated
updates for both initialization schemes. Existing tests verify that every original
parameter retains its tied-GDN initialization value.

The [execution trace](runs/gdn-independent-precision-h100/allocator-trace.json)
shows the pre-fix peer allocating its 50.07 GiB workspace while the source device
never entered that executable's allocation stage. After the copy fix,
[pilot 216999](runs/gdn-independent-precision-h100/alias-repair-runtime.json)
completed 20 updates, validation, and checkpoint saving on two H100 PCIe GPUs,
with finite gradient and attention diagnostics. It used the original allocator
settings: allocation on demand and the default memory fraction. Steady updates
took 1.84–1.87 seconds; the acceptance formula projects 10.99 hours for the complete
recipe, within the 22-hour budget of a 24-hour allocation. This pilot does not
replace complete production acceptance or its fresh-process resume check.

### H100 SXM allocation qualification

On two H100 80GB HBM3 GPUs, source `d3d93bc` passed 130 CPU checks, 9 distributed
checks, all 34 GPU checks, and canonical initialization. With allocation on demand
and the default memory fraction, the full-model pilot then failed a 50.28 GiB
allocation before its first update; its peer remained waiting in a collective.
The [runtime failure record](runs/gdn-independent-precision-h100/sxm-default-oom.json)
retains the matching log. This is a runtime acceptance failure with no production
training measurements. It does not qualify the default allocator on this hardware;
the completed PCIe pilot above applies to its own hardware/runtime combination.
The [90% memory-pool check](runs/gdn-independent-precision-h100/sxm-growth90-oom.json)
also passed the tests but failed its first full-model kernel launch with CUDA
out-of-memory. Neither two-GPU SXM runtime is qualified for this recipe.

The [four-GPU qualification](runs/gdn-independent-precision-h100/sxm-four-gpu.json)
passed with allocation on demand and the default memory fraction. The global batch
(256), global microbatch (64), and four accumulation steps are unchanged; each GPU
holds 16 sequences instead of 32. Job `217971` reused matching 139 CPU/distributed
and 34 GPU checks, completed 20 full-recipe updates with finite diagnostics,
validated, saved checkpoint 20, and resumed exactly to update 21 in a fresh process.
The state, next-batch, and RNG fingerprints matched the checkpoint audit.
Steady updates took 0.61–0.63 seconds; the conservative full-run projection is
4.24 hours, within the 22-hour budget. Qualification completed successfully in
8m39s. Twenty updates do not establish long-run stability or scientific performance.

## Gated DeltaNet on ORCD H200, and ORCD H100

These qualifications come from the production acceptance of the warmup-400 runs at
source `ccb1671`. Each allocation ran its CPU, distributed-CPU, and GPU tests with no failures
or skips, the initialization check, a 20-update full-recipe pilot on the fused backend,
and an exact fresh-process resume, with direct device transfer. Each run's
`record.json` holds its acceptance summary and hash.

- **Gated DeltaNet on 2×H200**, its first H200 qualification (it had run only on B300
  and H100): job 24890607 (`node5001`) passed on 2026-10-05 with 102 CPU, 5 distributed,
  and 12 GPU tests and a pilot on the `fla_chunk_gated_delta_rule_triton` kernel at
  0.81 s per update. Run `rdf74e4289926` then trained all 19,074 updates there without a
  numerical failure.
- **2×H100 on ORCD** (H100 80GB HBM3), previously used only on Kennel: normalized fixed
  Taylor `m = 1` (job 25048157, `node4109`; 187 CPU, 7 distributed, 23 GPU tests; pilot
  0.92 s per update), `m = 2` (job 25048180, `node3409`; 178, 5, 159; 0.95 s), and
  `m = 16` (job 25054702, `node2905`; 178, 5, 159; 1.39 s) passed on 2026-10-06 and 07.
  The 200M recipe fits two 80 GB devices. `m = 2` and `m = 16` resumed H200 checkpoints
  (steps 1,000 and 7,000) and trained their remaining updates at 0.99 and 1.43 s per
  update, against about 1.07 and 1.45 s on 2×H200; `m = 1` failed numerically at update
  67, a property of odd-degree normalized kernels rather than of the hardware.

The warmup-400 unnormalized fixed Hermite runs (source `13f018f`) added two more:

- **Unnormalized fixed Hermite on ORCD H100 and H200:** 310 CPU, 5 distributed, and 43 GPU
  tests passed on 2×H100 80GB HBM3 (job 25089881, 2026-10-07), and the GPU tests again on
  2×H200 (job 25089882). Later allocations reused these through the shared test results
  and ran their own initialization check, pilot (0.85–0.89 s per update), and resume.
- **2×H100 NVL, first use** (`node1709`): job 25089884 passed the initialization check, a
  pilot at 1.39 s per update, and the resume on 2026-10-08, then trained updates
  3,001–19,074 at about 1.5 s per update, about 1.6× slower than H100 80GB HBM3 or H200.
  Its GPU unit tests did not run on the NVL: the shared test results were then keyed by
  the launcher's GPU type, `H100`, which covers both models, so it reused the H100 80GB
  HBM3 results. They are now keyed by GPU model, so an NVL allocation runs its own.

Softmax without QK-norm (plain QK scores) at `d_K = 8, 16, 32, 128` (source `e39410e`)
passed 131 CPU, 5 distributed, and 1 GPU test on 2×H100 80GB HBM3 (job 25236653,
2026-10-08), and each run its own initialization check, pilot, and resume. `d_K = 16`
started on 2×H100 NVL, which again reused the H100 80GB HBM3 test results.

The warmup-400 softmax and Gated DeltaNet runs at other widths (sources `4ed7397` and
`2ab5aa1`, both with training telemetry) added:

- **Gated DeltaNet on 2×H100 80GB HBM3:** 107 CPU, 9 distributed, and 12 GPU tests passed
  (job 25356457, 2026-10-09), and the GPU tests again on 2×H200 (jobs 25426442 and
  25356460), each with its own pilot and resume.
- **Gated DeltaNet `d_K = 128` ran out of memory on 2×H100 NVL:** it uses 71–75 GiB per
  H200, and on H100 NVL (job 25356460, 2026-10-10) its first update failed to allocate
  59.8 GiB. A requeue had reused the run's H100 80GB HBM3 acceptance, keyed by the
  launcher's `H100`, so no pilot ran on the NVL. The failed device stopped while the
  other waited in a collective until the job was preempted 33 minutes later: an
  allocation failure on one device can hang a data-parallel job instead of ending it.
  Acceptance is now keyed by GPU model, so such a requeue repeats the pilot, and the
  launchers end a job whose log stops growing for 20 minutes.
- **Softmax on 2×H100 NVL** (job 25426441) ran its own GPU tests under the GPU-model key.

## B300 cuBLAS concurrency: checkpoint replay

The locked cuBLAS 12.9 is in the range (12.8–13.1) where some Blackwell kernels free
tensor memory twice; JAX warns that a cuBLAS kernel running concurrently with another
can silently corrupt data. Two B300 runs stopped on an infinite gradient norm at a
finite loss (an unnormalized learned `m = 4` run with `1/t` output scaling at update
12,637; `rdf00da0866a8` at 18,408, which then resumed on H100 and completed).
`replay_checkpoint.py` resumes a run from a retained checkpoint in a scratch directory,
leaving the run untouched; `replay_checkpoint.sbatch` repeats it in one allocation,
optionally with `--xla_gpu_disable_async_collectives=ALLCOLLECTIVES` ("sync"), which keeps
the gradient all-reduce on the compute stream. On 2026-10-04 the first run was replayed
from its update-12,500 checkpoint through update 12,900 on its source commit
`a63a416775f31a6297076b4ba261df2a7c26df41`:

| Job | GPU | Replays | Result | Updates/s |
| --- | --- | --- | --- | ---: |
| 208436 (`akita`) | 4×B300 | default, sync, default, sync | all pass 12,637 and 12,900 | 2.10 (both) |
| 208437 (`dogo`) | 4×H100 | default, default | both pass | 1.98 |

The replays track the original losses within 1e-4 through update 12,630, then train past
the failure: it does not follow from the model state. That is consistent with transient
corruption but does not prove it. Keeping collectives synchronous costs no measurable
throughput, so Kennel B300 launches set it (`launchers/kennel_runtime.sh`,
[CLUSTERS.md](../../../CLUSTERS.md)) until B300 runs use cuBLAS >= 13.2. Retained logs
(Kennel, `validation/logs/`): `replay-b300-208436.log`
(`b722bc63a32d8de642e5677fdf566126bdba1083a4e7ef03e826fb83f9e75759`) and
`replay-h100-208437.log`
(`cc643e429c1e5267bdbc3638efcdb6bca8f02f2f8ccf5c4396fe4a481660973f`). Submitted from the
primary checkout with a detached worktree at the run's commit:

```bash
sbatch -w akita --output=scaffolds/transformer/validation/logs/replay-b300-%j.log \
  scaffolds/transformer/validation/replay_checkpoint.sbatch worktrees/b300_replay \
  experiments/attention_scaling/runs/<run> 12500 12900 <scratch> default sync default sync
```

## Reproduction

From a worktree with the scaffold environment synced
(`uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink`):

```bash
PY=scaffolds/transformer/.venv/bin/python
JAX_PLATFORMS=cpu PYTHONPATH=. $PY -m pytest \
  -c scaffolds/transformer/pyproject.toml scaffolds/transformer/tests \
  --import-mode=importlib -m 'not gpu'
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=. $PY -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests/test_distributed_transfer.py --import-mode=importlib
# Within an appropriate GPU allocation:
PYTHONPATH=. $PY -m scaffolds.transformer.benchmark \
  scaffolds/transformer/validation/configs/taylor.yaml \
  --package scaffolds.transformer --output /tmp/shared-taylor-benchmark.json
```

The benchmark output is overridden on the command line. The code that produced
each retained measurement is at the source commit recorded for it in the
[manifest](provenance/manifest.json).

Benchmark and diagnostic scripts live in `scaffolds/transformer/benchmarks/`.
Inside a GPU allocation, write new measurements to a fresh run directory, for
example:

```bash
RUN_DIR=scaffolds/transformer/validation/runs/<new-benchmark-run>
mkdir -p "$RUN_DIR"
python -m scaffolds.transformer.benchmarks.benchmark_attention --output "$RUN_DIR/benchmark.json"
python -m scaffolds.transformer.benchmarks.benchmark_train_step --degrees 2 16 softmax \
  --implementations fused --trace-dir /tmp/traces --output "$RUN_DIR/train-step.json"
python -m scaffolds.transformer.benchmarks.analyze_profile /tmp/traces --output "$RUN_DIR/profile-summary.json"
python -m scaffolds.transformer.benchmarks.tune_attention --degrees 2 16 --output "$RUN_DIR/tile-tuning.json"
python -m scaffolds.transformer.benchmarks.benchmark_packed_run --workdir /tmp/fresh-pilot --output "$RUN_DIR/packed-pilot.json"
python scaffolds/transformer/benchmarks/cuda_peer_probe.py "$RUN_DIR/cuda-peer-copy.json"
```

The retained Taylor measurements were produced by the scripts at their recorded
source commits. The full A100 transfer ablation supervisor is at source
`a1f1b2f0`; its input policy is summarized in the dK × Taylor-degree manifest, and
its recipes are `configs/kdt-qualification-*.yaml`.

## Provenance limitation

The raw benchmark `source_commit` field captured parent `c2b0d6e` while the shared
refactor was uncommitted. It is preserved verbatim and is not treated as an exact
runtime source identity. Every one of the 53 Python files in the benchmarked
source was compared byte-for-byte with `2e0e88d`, and all matched. The final shared source is retained at `d493a41`, but
an immutable hash of its benchmark-time dirty tree was not recorded. These
small benchmarks are supporting evidence rather than production experiment data.
