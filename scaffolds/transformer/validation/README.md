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
[CLUSTERS.md](../../../CLUSTERS.md)); it did not prevent later failures, which cuBLAS
13.2 does (next section). Retained logs
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

## B300 CUDA 13 environment

Keeping collectives on the compute stream did not remove the B300 failures: on
2026-10-09 a four-B300 run with the flag set had MLP-only gradient norms of 1.7e13 and
1.2e16, and on 2026-10-15 the single-GPU blocked-reference tanh run `r3ef6320d0629`
(no collectives) stopped at update 649 with a finite loss (4.26) and an infinite
gradient norm. Replays from its update-500 checkpoint on its source `bb81667`:

| Job | GPU | Environment | Result |
| --- | --- | --- | --- |
| 219228 (`akita`) | 1×B300 | CUDA 12, cuBLAS 12.9.2.10 | failed at update 572 (finite loss 4.48, infinite gradient norm) |
| 219229 (`dogo`) | 2×H100 | CUDA 12 | passed through update 660 |
| 219246 (`akita`) | 1×B300 | CUDA 13.2, cuBLAS 13.2.2.2 | two replays, both passed through update 1,000 |

The failure moves between replays on B300 with CUDA 12, does not occur on H100, and
does not occur with cuBLAS 13.2, the first version JAX's runtime warning names as free of
the Blackwell tensor-memory issue. B300 jobs therefore use the `cuda13` environment
([CLUSTERS.md](../../../CLUSTERS.md)); job 219246 ran source `0695147`.

The full GPU test suite on one B300 with the `cuda13` environment (job 219252, source
`653b90f`, driver 580.126.20, one process per test file; record
[`runs/b300-cuda13/tests.json`](runs/b300-cuda13/tests.json)) passed 477 tests in 18
files. Two files had failures, neither from CUDA 13:

- `test_softmax_backends.py`: the two width-256 cuDNN cases, which B300 rejects with
  `d_qk != d_v` (the known limit above).
- `test_gated_deltanet_kernel.py`: the float32 width-128 case crashes with
  `CUDA_ERROR_LAUNCH_FAILED`; run alone in a fresh process it crashes in both the
  `cuda13` and `cuda12` environments on the same B300, so it is a B300 failure of the
  FLA kernel at that width. Widths 8–64 pass. A crash poisons the process's CUDA
  context, so the file's later cases (bfloat16, width 256, independent erase) are
  untested on B300 in either environment; GDN at width 128 or more is not qualified on
  B300.

An earlier single-process run of the same suite (job 219241) reported 411 failures, all
`CUDA_ERROR_LAUNCH_FAILED` after that GDN crash; the per-file run replaced it. Retained
logs (Kennel, `validation/logs/`): `b300-cuda12-replay-219228.log`
(`a5366eccc5d3034e87900ed85bd5b7cc17ea9f7b17ebec4345eb913d6c0611c5`),
`h100-cuda12-replay-219229.log`
(`a1bb6da1e602db37a76e2fa44668b3e5f56244dab3861ff135329185edb819e7`),
`b300-cuda13-replays-219246.log`
(`359ed32d26eeae7b61e02ad007397c17b214071584c93ee6207f8264f7f462c3`), and
`b300-cuda13-qualify-219252.log`
(`f5f6d0416796aab0100e9cd20bca8602436549410ddd8b4cb5dc0754244751d3`), with the per-file
JUnit reports in `b300-cuda13-gpu-219252/`.

```bash
sbatch -w akita --gres=gpu:b300:1 scaffolds/transformer/validation/qualify_b300_cuda13.sbatch \
  "$(git rev-parse HEAD)" - 0 0 0 \
  'scaffolds/transformer/tests/test_gated_deltanet_kernel.py::test_fused_output_and_all_gradients_match_the_recurrence[float32-0.005-128]'
sbatch -w akita --gres=gpu:b300:1 --cpus-per-task=16 --mem=128G \
  scaffolds/transformer/validation/replay_checkpoint.sbatch "$PWD" <run> 500 1000 <scratch> default default
```

## Signed linear GDN on one B300

The [benchmark record](runs/gdn-tanh-b300/record.json) and
[timing samples](runs/gdn-tanh-b300/benchmark.json) cover a bounded replay of
`rf65d38ec0552`, the degree-one, d_K=d_V=64 GDN with tanh writes and independent
sigmoid erasure. Job `219939` completed on one B300 SXM6 AC using CUDA 13.2 and
cuBLAS 13.2.2.2. All 10 signed-write GPU tests passed, including outputs and all
six gradients against the FP64 matrix-state oracle, zero erasure, and length
2,048. The corresponding CPU checks passed 83 tests; record/index tests passed 50.

The replay restored a disposable copy of checkpoint 11,500 with its optimizer
state and FineWeb data cursor. The model has 211,007,424 parameters; the original
global batch of 256 sequences, global microbatch 64, four accumulation steps,
2,048 prediction positions, and learning-rate horizon were preserved. The first
five updates were warmup; the next 30 were timed. All 35 updates were finite.
Source recipe, record, measurements, data identity, and checkpoint checksums were
unchanged. The source run remains paused, and no replay checkpoint is retained.

| Execution | Mean seconds/update | Tokens/second |
| --- | ---: | ---: |
| One B300, measured replay | 4.2255 | 124,077 |
| Four H100 PCIe, historical windows ending at steps 9,000, 10,000, 11,000 | 1.1635 | 450,624 |

One B300 delivers 27.5% of the earlier four-H100 aggregate throughput with this
microbatch setting, or about 1.10 times the throughput per allocated GPU. This is
a historical comparison, not a controlled kernel speedup: the single-GPU trainer
omits distributed telemetry, and each H100 window includes a periodic checkpoint.
B300 intervals include packed-data loading, transfer, numerical checks, ordinary
logging, and optimizer updates, but exclude compilation, validation, and final
checkpoint writing. B300 median update time was 4.1714 seconds (p10 4.1362, p90
4.3917). First-update compilation and execution took 159 seconds. No microbatch
tuning was performed. The measured rate projects about 8.9 hours for the remaining
7,574 updates, excluding startup, evaluation, and checkpoint overhead.

Reproduce from a worktree at the recorded launch commit, after syncing both locked
environments as described in `CLUSTERS.md`. Supply the retained source run directory
containing checkpoint 11,500 and a fresh output directory:

```bash
sbatch --output=scaffolds/transformer/validation/logs/gdn-tanh-b300/benchmark-%j.log \
  scaffolds/transformer/launchers/benchmark_gdn_tanh.sbatch \
  9e62cf7ce583ae0af776a319a2ebc4df3737f1ab "$SOURCE_RUN_DIRECTORY" "$OUTPUT_DIRECTORY"
```

The launcher verifies the frozen execution set, runs the signed-write GPU tests,
then calls `benchmarks/benchmark_checkpoint_replay.py`. Its input translation
removes the legacy explicit `gdn_polynomial_degree: 1` field for the current linear
GDN implementation; other polynomial degrees are rejected. The worktree's own
CUDA 13 environment supplies the dependencies. The successful log and GPU JUnit
output are retained under `validation/logs/gdn-tanh-b300/` on Kennel and their
hashes are recorded. The generic Git dirty flag in the timing report includes
mutable validation records and generated indices; the frozen execution check passed.

## GDN versus Mosaic on one B300

Job **220032** completed on one NVIDIA B300 SXM6 AC (`akita`, CUDA 13) in
18 minutes 38 seconds. Pooling 60 measured updates per mechanism:

| Mechanism | Parameters | Mean seconds/update | Tokens/second | Difference between repeat means |
| --- | ---: | ---: | ---: | ---: |
| Signed-write linear GDN, fused FLA Triton | 211,007,424 | 4.2410 | 123,624 | 0.789% |
| Cubic query-conditioned tanh, Mosaic | 200,573,344 | 1.2039 | 435,500 | 0.059% |

The Mosaic model has **3.523x higher training throughput** at this workload.
This is an architecture comparison, not a speedup from replacing GDN's backend:
Mosaic does not implement GDN. Compute-only means are 4.2373 and 1.2010 seconds,
so loading and host overhead do not explain the difference. GDN agrees with the
earlier checkpoint-replay measurement (124,077 tokens/second). Each mechanism's
two trials use the same physical GPU, runtime, data cursor and dataset identity;
the difference between repeat means is divided by their pooled mean above.
All 140 training updates were finite, all four production-context output/gradient
tests passed, and three CPU tests checked the timing harness and isolation.
No benchmark checkpoints are retained. The successful log and GPU-test XML are
retained on Kennel under `validation/logs/gdn-mosaic-b300/`, with hashes in the record.

This comparison times signed-write linear GDN (the recipe of `rf65d38ec0552`)
against cubic query-conditioned tanh attention (the recipe of `r0bc57959ef6c`,
with the Mosaic backend). Both use the shared 10-layer, width-1024 backbone,
16 heads, d_K = d_V = 64, MLP width 3008, BF16, sequence length 2048, batch 256,
microbatch 64, and four accumulation steps. Each keeps its mechanism's reference
initialization and required normalization. GDN's additional projections,
convolution and gates mean the parameter counts differ; this compares complete
training implementations at matched backbone widths, not only kernel efficiency.

`benchmarks/benchmark_training_throughput.py` starts each trial from reference
initialization on the same packed FineWeb data and seed. Five warmup updates
precede 30 synchronized update intervals. Intervals include loading, transfer,
accumulation, optimizer updates, numerical checks and ordinary logging; they
exclude compilation, validation and checkpointing. Two trials per mechanism
run sequentially in one B300 allocation, in GDN/Mosaic/Mosaic/GDN order.
The paused GDN training run is not resumed. These measurements do not assess
model quality, decoding, or recurrent-state memory savings. Microbatch size is
fixed, rather than tuned separately for each mechanism.

Inputs are `validation/configs/gdn-mosaic-b300-{gdn,mosaic}.yaml`;
reports and execution provenance are in `validation/runs/gdn-mosaic-b300/`.
The launcher checks both the shared frozen execution set and these input
recipes against the launch commit. From a committed Kennel worktree with both
locked CUDA environments synced:

```bash
mkdir -p scaffolds/transformer/validation/logs/gdn-mosaic-b300
sbatch --parsable \
  --output=scaffolds/transformer/validation/logs/gdn-mosaic-b300/benchmark-%j.log \
  scaffolds/transformer/launchers/benchmark_gdn_mosaic.sbatch "$(git rev-parse HEAD)"
```

Regenerate pooled timing summaries from the four explicit reports:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("scaffolds/transformer/validation/runs/gdn-mosaic-b300")
rates = {}
for name in ("gdn", "mosaic"):
    reports = [json.loads((root / f"{name}-{trial}.json").read_text()) for trial in (1, 2)]
    assert all(r["status"] == "complete" and r["all_finite"] for r in reports)
    seconds = [s["interval_seconds"] for r in reports for s in r["samples"][r["warmup_updates"]:]]
    rates[name] = reports[0]["tokens_per_update"] * len(seconds) / sum(seconds)
    print(name, "seconds/update", sum(seconds) / len(seconds), "tokens/sec", rates[name])
print("Mosaic / GDN throughput", rates["mosaic"] / rates["gdn"])
PY
```

## Mosaic GPU polynomial attention kernels on H100 and B300

`attention/backends/mosaic/` implements causal polynomial attention (forward, dQ, dK, dV) on
Mosaic GPU: `hopper.py` (wgmma, FlashAttention-3 structure) and `blackwell.py` (tcgen05 with
tensor memory, warp-specialized). `common.py` holds each family's elementwise mathematics, so
both families share one set of pipelines. `api.py` exposes `taylor_attention` (normalized
fixed Taylor, matching `fused_taylor`) and `query_polynomial_attention` (unnormalized, FP32
coefficients per query and head, matching `fused_query_polynomial`); unnormalized polynomials
with static fixed or learned coefficients (degree 1-8, matching `fused_unnormalized` and
`fused_learned_polynomial`) run on the query kernel with every query's coefficients equal,
broadcasting their coefficient gradients back. Scores, polynomials, row
sums, and accumulators are FP32, with the Triton kernels' failure rules: NaN rows for a
nonfinite, nonpositive (even Taylor degree), or zero (odd degree) row sum, a nonfinite score
or weight (query), or an overflowing output. They accept BF16 inputs, Q/K width 8, 16, 32,
64, or 128, value width 64, degree >= 1 (query and static: 1-12, where B300's dK/dV fits), and time a multiple of 128 (H100) or
256 (B300), and raise `ValueError` otherwise. Tensor cores contract in steps of one
shared-memory swizzle row, so Q/K tiles use 32-, 64-, and 128-byte swizzles at widths 16,
32, and 64 or more; width 8 is zero-padded to the 16-element minimum. On B300 the dK/dV kernel keeps every
query's coefficients in shared memory and runs a three-stage Q/dO ring unless that exceeds
the 227 KB per block (degree 8 at d_K = 128 and 2,048 tokens), then two
(`common.blackwell_dkv_stages`, which also bounds the supported degree: 12 at d_K = 64, 9 at
d_K = 128); dQ stages its output in each warpgroup's Q buffer.

Weights and score gradients enter the tensor cores at a weight precision chosen per setup
(`api.default_precision`, recorded as `attention_weight_precision` in run metadata):

| Setup | Weight precision |
| --- | --- |
| Tanh query-conditioned, m 1-8 | BF16: one rounding, as FlashAttention and cuDNN |
| Unnormalized static coefficients, m 1-8 | BF16 |
| Normalized Taylor, even m | BF16 |
| Normalized Taylor, odd m | FP32: three BF16 components whose sum is the FP32 value |

With BF16 weights the Taylor backward also reads the BF16 output, as FlashAttention's does.
Odd-degree normalized Taylor divides signed weights by a row sum that can cancel, which
amplifies their rounding; it keeps FP32-precision weights (the accuracy table below).

`tests/test_mosaic_taylor.py` (78 cases), `tests/test_mosaic_query.py` (69), and
`tests/test_mosaic_repeatability.py` (6) cover both weight precisions and pass on one H100
(`dogo`, CUDA 12.9.1) and one B300 (`akita`, CUDA 13.2): every degree and width against
independent NumPy FP64 outputs and VJPs (coefficient gradients per order) and against the
Triton kernels, the production context, causality, nonfinite handling, rejected inputs, the
model's dispatch, and bitwise-identical results across 20 repeats at production size.
`tests/test_mosaic_unnormalized.py` (36) checks fixed Taylor and shared or per-head learned
static coefficients against FP64 and the Triton kernels, including the summed coefficient
gradients; all cases pass on H100 and B300, including degree 8 at the production context with
d_K = 128. Acceptance passes a
run's weight precision (`--mosaic-precision`), so its GPU tests skip the other precision's
cases.

### Accuracy

Worst relative L2 error against NumPy FP64 over every Q/K width, degree, batch, and head
(`benchmarks/mosaic_precision.py`; `runs/mosaic-precision/`), next to cuDNN softmax (the
production softmax path) on the same inputs. Sharp inputs scale Q by 4, so scores reach
about +-8 and odd Taylor degrees put large negative weight on many keys. Each entry is the
worse of H100 and B300 (Mosaic's agree to the digits shown).

| Setup | Weights | Output | Q/K/V gradients | Output, sharp | Gradients, sharp |
| --- | --- | ---: | ---: | ---: | ---: |
| cuDNN softmax | BF16 | 2.2e-3 | 2.8e-3 | 2.2e-3 | 4.1e-3 |
| Query m 1-4 | BF16 | 2.5e-3 | 2.4e-3 | | |
| | FP32 | 1.7e-3 | 1.7e-3 | | |
| Taylor, even m | BF16 | 2.7e-3 | 3.3e-3 | 2.5e-3 | 3.4e-3 |
| | FP32 | 1.6e-3 | 1.8e-3 | 1.7e-3 | 1.8e-3 |
| Taylor, odd m | BF16 | 2.5e-3 | 2.9e-3 | 3.7e-3 | 8.1e-3 |
| | FP32 | 1.6e-3 | 1.8e-3 | 2.0e-3 | 2.9e-3 |

Query coefficient gradients are within 2.3e-7 at either precision. At degree 8 on B300
(`precision_mosaic_degree8.py`; `runs/mosaic-precision/degree8-*.txt`), BF16 query errors are
2.3-2.5e-3 with positive or signed coefficients, as at degree 3, and coefficient gradients
are within 6e-7.

### Speed

Forward + backward per layer at 16 x 2048 x 16 heads, value width 64
(`benchmarks/benchmark_mosaic.py`; `runs/mosaic-kernels/`), Triton time divided by Mosaic
time:

| | H100, BF16 | H100, FP32 | B300, BF16 | B300, FP32 |
| --- | ---: | ---: | ---: | ---: |
| Taylor, d_K 8-64, m 1-4 | 1.9-2.2x | 1.4-1.6x | 3.1-4.7x | 2.3-3.6x |
| Taylor, d_K 8-64, m = 16 | 3.3-3.4x | 2.5-2.7x | 9.1-12.4x | 7.2-10.1x |
| Taylor, d_K 128, m 1-16 | 4.3-4.6x | 3.2-3.7x | 4.9-5.8x | 3.9-4.4x |
| Query, d_K 64, m 1-4 | 2.0x | 1.2-1.4x | 4.2-4.4x | 3.0-3.2x |
| Query, d_K 64, m = 8 | 2.4x | 0.9x | 4.1x | 3.2x |
| Static, d_K 64, m = 3 and 8 | 2.0-2.1x | | 4.0-4.5x | |

At d_K = 64, query m = 3 takes 1.47 ms with BF16 weights on H100 (Triton 2.91 ms, cuDNN
softmax 1.45 ms) and 1.33 ms on B300 (Triton 5.74 ms, cuDNN softmax 0.77 ms); query m = 8
takes 2.44 ms (Triton 5.85 ms) and 2.63 ms (Triton 10.65 ms). FP32-precision weights at m = 8
spill registers on H100 (no setup uses them there).

One training microbatch (16 x 2048, `benchmark_mosaic_step.sbatch`; `runs/mosaic-step/`, one
directory per job), median of 30 updates, two rounds within 1%, each job's fused and Mosaic
runs on the same GPU:

| Recipe | GPU | Fused | Mosaic, BF16 | Mosaic, FP32 (separate job, fused) |
| --- | --- | ---: | ---: | ---: |
| Query m = 3, d_K = 64 | H100 SXM | 120.7 ms | 103.0 ms | 115.0 ms (120.8 ms) |
| | B300 | 108.5 ms | 59.5 ms | 67.4 ms (108.8 ms) |
| Static m = 3, d_K = 64 | H100 SXM | 118.9 ms | 99.5 ms | |
| Static m = 8, d_K = 64 | H100 SXM | 141.3 ms | 110.5 ms | |
| | B300 | 156.4 ms | 68.7 ms | |
| Query m = 8, d_K = 64 | H100 SXM | 159.8 ms | 120.4 ms | |
| | B300 | 167.0 ms | 76.6 ms | |
| | B300 | 108.6 ms | 58.8 ms | |
| Taylor m = 2, d_K = 32 | H100 SXM | 108.0 ms | 88.9 ms | 96.1 ms (107.9 ms) |
| | B300 | 77.1 ms | 49.3 ms | 55.0 ms (77.2 ms) |

`mastiff` has H100 PCIe cards (114 SMs, HBM2e), `dogo` H100 SXM; throughput is compared only
within one GPU.

### Stress

`stress_mosaic.sbatch` calls each kernel in its own process at the production per-GPU shape
with sharp scores, 100,000 times or more, and requires every 1,000th result to equal the
first bitwise; a process silent for 120 s counts as hung. On H100 the forward and dQ kernels
at both precisions (job 220106, 300,000 calls each) and dK/dV (job 220103, 100,000) finished
with identical results, except one query dQ (BF16) process that hung within its first 1,000
calls, for no cause found; twenty further 50,000-call processes of query and Taylor dQ (job
220120) and a 100,000-call rerun finished, as did every training pilot below. On B300 (job
220124) the forward, dQ, dK, and dV kernels of both families at both precisions each finished
100,000 calls with identical results. At degree 8 (`STRESS_DEGREE=8`) every B300 kernel (job
223204) and every H100 kernel with BF16 weights and every query kernel with FP32 weights (job
223203) finished 100,000 calls; H100 Taylor dQ and dK/dV with FP32 weights at degree 8 take
15-30 ms per call (register spills) and finished 2,000 calls each (job 223271). Kernels synchronize each ring slot only after both compute
warpgroups have released it; tiles past a warpgroup's causal diagonal are neither consumed
nor freed by it.

### Training pilots

`pilot_mosaic.sbatch` trains a study recipe for 1,000 updates (microbatch 16, validation
every 100 updates) with a chosen backend, from the same initialization and data; summaries are
in `runs/mosaic-pilot/`. Recipes: tanh query-conditioned m = 3, d_K = 64, head RMSNorm
(`r0bc57959ef6c`), unnormalized static per-head learned coefficients m = 3, d_K = 64, head
RMSNorm (`r5ae483d40d62`), normalized Taylor m = 2, d_K = 32 (`raf32fab4ba52`), and the
degree-8 static and tanh query recipes (`raa043e978a3d`, `rccf19c31669d`), all with 400 warmup
updates. XLA's GPU reductions are not deterministic, so two fused runs already
diverge; a repeated fused query run measures that spread. H100 runs used SXM and PCIe cards.

Validation NLL:

| Recipe, GPU | Backend | 200 | 400 | 600 | 800 | 1,000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Query m = 3, H100 | fused | 5.6465 | 4.6246 | 4.0609 | 3.6989 | 3.5157 |
| | fused, repeat | 5.6463 | 4.5959 | 4.0545 | 3.7041 | 3.5132 |
| | Mosaic, FP32 weights | 5.6463 | 4.5862 | 4.0395 | 3.6854 | 3.5085 |
| | Mosaic, BF16 weights | 5.6336 | 4.6112 | 4.0434 | 3.6922 | 3.5096 |
| | Mosaic, BF16 weights, 2 GPUs | 5.6609 | 4.6044 | 4.0542 | 3.6928 | 3.5062 |
| Query m = 3, B300 | fused | 5.6396 | 4.6164 | 4.0552 | 3.7141 | 3.5108 |
| | Mosaic, FP32 weights | 5.6601 | 4.6079 | 4.0514 | 3.6950 | 3.5120 |
| | Mosaic, BF16 weights | 5.6365 | 4.6179 | 4.0485 | 3.6903 | 3.5101 |
| Static m = 3, H100 | fused | 5.6991 | 4.7194 | 4.1582 | 3.8295 | 3.6210 |
|  | Mosaic, BF16 weights | 5.7029 | 4.7121 | 4.1590 | 3.8328 | 3.6197 |
| Static m = 3, B300 | fused | 5.6981 | 4.7252 | 4.1660 | 3.8374 | 3.6183 |
|  | Mosaic, BF16 weights | 5.7124 | 4.7339 | 4.1733 | 3.8482 | 3.6330 |
| Static m = 8, H100 | fused | 5.5341 | 4.5186 | 3.8805 | 3.6233 | 3.4569 |
|  | fused, repeat | 5.5369 | 4.5281 | 3.8924 | 3.6054 | 3.5116 |
|  | Mosaic, BF16 weights | 5.5441 | 4.5436 | 3.8896 | 3.6656 | 3.5029 |
| Query m = 8, B300 | fused | 5.5295 | 4.5161 | 3.9700 | 3.6891 | 3.5152 |
|  | fused, repeat | 5.5524 | 4.5313 | 3.9256 | 3.6979 | 3.5782 |
|  | Mosaic, BF16 weights | 5.5353 | 4.5061 | 3.9204 | 3.6512 | 3.5057 |
| Taylor m = 2, H100 | fused | 5.4085 | 4.7960 | 4.4177 | 4.1762 | 4.0295 |
| | Mosaic, FP32 weights | 5.4168 | 4.7962 | 4.4217 | 4.1681 | 4.0133 |
| | Mosaic, BF16 weights | 5.4061 | 4.8322 | 4.4402 | 4.1965 | 4.0154 |
| Taylor m = 2, B300 | fused | 5.4061 | 4.7812 | 4.4166 | 4.1713 | 4.0150 |
| | Mosaic, FP32 weights | 5.4121 | 4.7840 | 4.4128 | 4.1988 | 4.0108 |
| | Mosaic, BF16 weights | 5.4078 | 4.7791 | 4.4386 | 4.1682 | 4.0223 |

From step 200 on, the two fused query runs differ by up to 0.029; every Mosaic run stays
within 0.038 of the fused run and ends between -0.016 and +0.015 at step 1,000. Differences
change sign except for static m = 3 on B300 (+0.001 to +0.015), within the spread of two fused
runs, which also keep one sign. Degree 8 trains less reproducibly: two fused runs end 0.055
(static, H100) and 0.063 (query, B300) apart, and Mosaic stays within 0.050 of the first. Mosaic tracks the fused kernels within run-to-run variation at both
precisions, on one GPU and two.

From a committed worktree on Kennel (pilot arguments: commit, GPU type, study run, backend,
updates, and an optional label for a repeated run):

```bash
sbatch --partition=long --gres=gpu:h100:1 scaffolds/transformer/validation/pilot_mosaic.sbatch \
  "$(git rev-parse HEAD)" H100 r0bc57959ef6c mosaic 1000
sbatch --partition=long --gres=gpu:h100:1 scaffolds/transformer/validation/benchmark_mosaic_step.sbatch
sbatch --partition=long --gres=gpu:h100:1 scaffolds/transformer/validation/stress_mosaic.sbatch
python -m scaffolds.transformer.benchmarks.mosaic_precision --output errors.json  # in a GPU allocation
python -m scaffolds.transformer.benchmarks.benchmark_mosaic                      # in a GPU allocation
```

## Upstream FLA and cuDNN baselines on one B300

`validation/runs/upstream-b300/` compares the project's implementations with
direct calls to upstream libraries. The earlier GDN timings measure this
project's signed, independent-gate recurrence with FP32 chunk intermediates
and TF32x3 products. They are not measurements of stock FLA's native BF16 path.
The upstream comparison uses standard tied sigmoid GDN and the unmodified
FLA commit `516143e31fce09925e6c39ac37148444bad176c4`, also pinned by our bridge.

The comparison has three scopes:

- Attention forward and forward-plus-backward at B=64, T=2048, H=16 and
  K=V=64. CUDA PyTorch calls `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  directly, and forces cuDNN through PyTorch SDPA. JAX calls cuDNN directly
  and measures Mosaic plus the project's tied and independent GDN paths.
  Both frameworks generate identical NumPy operands; GDN uses unit Q/K,
  while dot-product attention scales Q outside timing to give order-one scores.
  Inputs, warmup, compilation and finite checks are excluded. All input
  gradients are included; the number of differentiable arguments depends
  on the mechanism. Tanh coefficient generation is outside the Mosaic kernel.
- Full-model cuDNN softmax and Mosaic use the same shared JAX trainer,
  packed data, 10-layer width-1024 backbone, MLP width 3008, 16 heads,
  K=V=64, batch 256, microbatch 64 and four accumulation steps. Each starts
  from its reference initialization and retains the optimizer horizon.
- The full upstream `GatedDeltaNetForCausalLM` uses matching dimensions,
  standard gates, BF16 autocast with FP32 parameters, its fused normalization
  and SwiGLU, upstream fused cross-entropy, and fused PyTorch AdamW. The
  model and loss implementations are unmodified. A resident real FineWeb
  batch excludes loading/transfer from this timing. This is a complete
  upstream model comparison, with framework, initialization and component
  differences; it is not a substitution into the JAX trainer.

The full-model compute comparison excludes input loading and transfers for
native FLA. JAX's `compute_seconds` includes numerical checks and may await
an asynchronous input transfer; its ordinary training-interval timing is
reported separately. Both include gradient accumulation, gradient clipping,
the optimizer, and parameter-norm computation. Native FLA checks the returned
loss and norms after timing. Native FLA retains an FP32 residual stream under
autocast; this project's residual stream is BF16. The benchmark preserves
these implementation choices.

The separate `benchmarks/upstream_runtime/` project has its own committed
lock and environment. It installs the official CUDA PyTorch wheel, leaving
the production JAX environments unchanged. Its explicit CUDA 13.2/cuBLAS
13.2.2.2 overrides apply Kennel's Blackwell correctness policy; cuDNN 9.22
and Triton 3.7.1 match the JAX comparison. No installed package is edited.
The native FLA baseline retains its upstream mixed-precision choices,
which differ from the project's FP32 repair. These timings do not establish
equivalent numerical accuracy or training quality.

Attention-only measurements from job `220076` (one B300 on `akita`, 2026-10-17):
each entry pools two rounds of 30 calls, after five warmup calls per round.
Values are synchronized wall time; compilation, autotuning and finite checks
are excluded. cuDNN dispatch is verified by compiled custom-call targets in
JAX and profiler events in PyTorch, with the PyTorch backend forced to cuDNN.

| Implementation | Forward (ms) | Forward + backward (ms) |
| --- | ---: | ---: |
| Upstream cuDNN, PyTorch SDPA | 0.706 | 2.734 |
| Upstream cuDNN, JAX | 1.539 | 3.550 |
| Upstream FLA, standard tied sigmoid GDN | 1.684 | 5.492 |
| Our Mosaic cubic query polynomial | 3.120 | 10.290 |
| Our JAX GDN bridge, standard tied sigmoid gates | 5.537 | 79.719 |
| Our JAX GDN bridge, independent tanh writes | 5.536 | 77.958 |

Optional native CUDA graph replay reduces FLA forward-plus-backward to
5.160 ms and PyTorch cuDNN to 2.590 ms. Normal upstream dispatch, shown in
the table, is the primary comparison. PyTorch's forward-only call retains
autograd context; forward-plus-backward is the stronger comparison across
frameworks. The mechanisms differentiate five, six, four and three inputs
for tied GDN, independent GDN, Mosaic and softmax, respectively.

Stock FLA's forward-plus-backward is **14.52x faster** than our bridge with
the same standard tied-gate mathematics. Independent signed gates have a
similar cost in our bridge. The comparison therefore identifies a large
implementation/precision gap; it does not attribute that gap to the signed
gates. Precision, state layout, framework and dispatch differ, so these
measurements alone cannot assign the gap to one of them. Mosaic takes
**1.87x** stock FLA's time and **2.90x** JAX cuDNN's time at this shape
(**3.76x** PyTorch cuDNN's time). These are training-kernel measurements at
context 2048, not asymptotic complexity or language-model quality results.

Full-model training compute, pooled over two fresh-process trials of 30
measured updates after five warmup updates each:

| Implementation | Parameters | Seconds/update | Tokens/s |
| --- | ---: | ---: | ---: |
| Shared model with upstream cuDNN softmax | 199,906,464 | 0.938 | 559,118 |
| Unmodified upstream FLA GatedDeltaNet model | 210,843,584 | 1.248 | 420,010 |
| Shared model with Mosaic cubic query polynomial | 200,573,344 | 1.250 | 419,436 |
| Our independent-tanh GDN, prior job `220032` | 211,007,424 | 4.237 | 123,732 |

The first three rows use the same physical B300 in job `220076`; the last
is explicit historical context from `runs/gdn-mosaic-b300/gdn-{1,2}.json`
on another B300. cuDNN is **1.33x** Mosaic's throughput. Stock FLA and Mosaic
are effectively tied: their 0.14% difference is smaller than Mosaic's 3.25%
repeat spread. cuDNN and FLA repeat spreads are 0.49% and 0.13%, respectively.
The earlier several-fold Mosaic advantage over our GDN does not describe
the stock FLA baseline. The upstream FLA full model is **3.39x** our historical
GDN compute throughput, with the mathematical and runtime differences above.

Including loading, transfers, checks and ordinary logging, the JAX training
intervals average 0.944 s/update (555,616 tokens/s) for cuDNN and 1.255 s/update
(417,663 tokens/s) for Mosaic. Native FLA's resident-batch timing has no matching
input-pipeline interval. All 210 updates across the six new full-model trials
have finite loss, gradient norm and parameter norm. cuDNN's output/gradient
parity test and production-context finite check pass. Job `220076` completed in 21m17s with exit code
zero; no checkpoints are retained. `runs/upstream-b300/record.json` pins the
eight reports, recipes, locks, retained log and correctness-test summary.

From a committed Kennel worktree, sync the two production environments as in
`CLUSTERS.md`, then sync the independent CUDA Torch environment and submit:

```bash
uv sync --project scaffolds/transformer/benchmarks/upstream_runtime --locked --link-mode hardlink
mkdir -p scaffolds/transformer/validation/logs/upstream-b300
sbatch --parsable \
  --output=scaffolds/transformer/validation/logs/upstream-b300/benchmark-%j.log \
  scaffolds/transformer/launchers/benchmark_upstream.sbatch "$(git rev-parse HEAD)"
```

Regenerate the pooled summaries and audit the explicitly named source files:

```bash
python3 scaffolds/transformer/validation/summarize_upstream.py
```

## Polynomial GDN Triton tuning on one B300

The [polynomial GDN record](runs/poly-gdn-tuning/record.json) pins the source,
locked JAX and native PyTorch environments, recipes, test evidence, and explicitly
selected reports. The completed initial profile is job `222298` on `akita`,
source `0e9faad`; completed occupancy tuning is job `222316`, source `34f2c02`.
Qualification of the corrected kernels passed in job `222319`, source `dfff226`.
Job `222322` completed production-microbatch and full-model confirmation from
source `8fd821a` in 1h22m11s with exit code zero.

This is the original-key-space polynomial GDN recurrence:
`P_t = I - erase_t k_t k_t^T`,
`F_t(q) = exp(g_t) F_(t-1)(P_t q) + v_t sum_r beta_(t,r) (q^T k_t)^r`,
with read `scale * F_t(q_t)`. The Triton implementation evaluates transported
scores without materializing the expanded polynomial state. Its training
backward still uses packed triangular workspaces and visits earlier chunks;
avoiding the expanded state does not give it FLA's linear-GDN training cost.
The full-model recipes preserve degree three, independently gated erasure,
sigmoid linear writes, and tanh constant/quadratic/cubic writes, matching the
original cubic recipes. The configurations in
`configs/poly-gdn-tuning-dk{16,64}.yaml` use reference initialization, ten
layers, hidden width 1,024, MLP width 3,008, 16 heads, value width 64, global
batch 256, microbatch 64, and 2,048 prediction positions.

The full-model comparison keeps FP32 parameters and Adam moments in both
frameworks, with BF16 matrix products. Convolution width four, SiLU/SwiGLU,
normalization epsilon `1e-6`, and optimizer hyperparameters match. BF16 conversion
boundaries differ between native autocast and JAX's explicit compute dtypes.
Native FLA also keeps its upstream initialization and fused normalization,
convolution, SwiGLU, cross-entropy, and AdamW implementations; JAX uses the frozen
GDN reference initialization and its shared XLA/Optax training path. Actual
parameter counts accompany timings. This is a comparison of complete training
implementations, with no numerical model-trajectory parity claim.

Native full-model timing repeats one GPU-resident batch. Compare its synchronized
wall compute time with JAX update compute, and report JAX completion intervals
separately: those include loading, transfers, checks, and logging. JAX compute
can also wait on an input transfer enqueued just before its timer starts. Both
compute measurements include gradient clipping and parameter/gradient norms.
Warmup and measured-update counts are explicit per report; slow JAX controls
use shorter windows, so repeat spreads and sample counts matter when assessing
small differences.

The default `tf32x3` mode retains FP32 transport/inverse arithmetic and accurate
tensor-core products. Opt-in `bf16_read` rounds the operands of value-reading
products to BF16; `bf16_values` also rounds the score-adjoint product. Both retain
FP32 accumulation and the original transport/inverse precision. The independent
FP64 and tensor-recurrence suites passed **52 strict FP32/BF16 cases** in default
precision and **26 BF16 cases in the selected `bf16_values` mode**, at two warps
and CTA guidance one in job `222319`, with no failures or skips. They cover
quadratic and cubic outputs and all six gradients, long context, signed writes,
erasure/decay extremes, causality, the linear limit, and packed-history indexing
across the signed 32-bit boundary. The initial profile separately passed 51
strict cases and 26 BF16 cases in each candidate mode, plus eleven option/smoke
cases; those smoke checks do not replace the strict tests.

Fresh-process diagnostics found a CUDA launch failure in four-warp polynomial
backward on Blackwell, while the standalone matrix product and VJP passed.
Two-warp forward and all six gradients completed, followed by the strict tests
above. Blackwell therefore defaults to two warps and rejects four/eight before
launch; one warp remains an explicit benchmark option. This qualification is
specific to the recorded B300 runtime and does not qualify another GPU/runtime.

Kernel measurements use resident BF16 operands at `B=16, T=2048, H=16, V=64`,
with `K=16/64`, an explicit output cotangent, and synchronized calls returning
the forward output and all six gradients. The synthetic fixture has unit Q/K,
Gaussian V/cotangent, negative log decay, sigmoid erasure, and independent signed
`tanh(0.2 * normal)` coefficients for all four degrees. Log decay remains FP32.
Each variant has two rounds, each with five warmups and 20 timed
calls; native FLA has 30 timed calls per round. The table reports pooled mean
milliseconds for two warps, with minimum-CTA guidance disabled:

| Key width | Cubic `tf32x3` | Cubic `bf16_read` | Cubic `bf16_values` | Native FLA GDN |
| --- | ---: | ---: | ---: | ---: |
| 16 | 65.729 | 62.985 | 36.600 | 1.678 |
| 64 | 786.436 | 852.813 | 462.775 | 1.990 |

FLA is the unmodified upstream PyTorch implementation pinned at `516143e`,
using its native tuned kernels. Its input fixture hashes match the polynomial
benchmark, but its operator is linear GDN with tied write/erase gates. It is a
performance reference, not a mathematical parity target. The table uses ordinary
synchronized wall timing for both implementations; FLA's separately recorded
CUDA-graph results are not substituted into this comparison.

All 12 initial polynomial variants passed the same-input output/gradient checks
against the two-warp `tf32x3` reference. For two-warp `bf16_values`, the worst
relative RMS difference was 0.260% at K16 and 0.258% at K64; the independent
qualification above provides the separate correctness evidence. One warp was
slower at K16 but reduced K64 `bf16_values` to 414.328 ms. Thus precision and warp
choices interact with width; this profile does not establish one optimal setting.
At K64, isolated default-precision cross-chunk and query backward calls took
433.205 and 169.575 ms. PTX assembly also reports substantial register spilling.
Stage calls use resident intermediates, so their sum is not an end-to-end time;
assembler resource figures are not measured occupancy.

The occupancy tuning matrix holds two warps fixed and crosses `tf32x3`/`bf16_values`
with `POLY_GDN_MIN_CTAS_PER_SM=0/1`. Zero leaves PTX unchanged; one adds the
explicit `.minnctapersm 1` register-allocation guidance to the polynomial kernels.
It does not alter the recurrence. The generic bridge validates this CUDA-only
option and separates its compilation and autotuning caches, preserving existing
default cache identities. Job `222316` completed with exit code zero; all eight
variants passed the same-input accuracy checks. Its mean forward-plus-backward
times, with the same shape and sampling protocol as above, are:

| Key width | Precision | CTA guidance 0, ms | CTA guidance 1, ms |
| --- | --- | ---: | ---: |
| 16 | `tf32x3` | 65.259 | 54.763 |
| 16 | `bf16_values` | 36.102 | 24.933 |
| 64 | `tf32x3` | 786.394 | 147.111 |
| 64 | `bf16_values` | 461.823 | 88.183 |

Combining value precision and CTA guidance reduces time by 2.62x at K16 and
8.92x at K64 against the same job's default-precision/CTA0 control. Native FLA
measured 1.356 and 1.919 ms on that job's same physical GPU, leaving substantial
gaps despite the improvement. The BF16 candidates' maximum relative RMS errors
remain 0.260%/0.258%; changing CTA guidance alone in `tf32x3` produced identical
benchmark outputs and gradients. Resource inspection confirms the requested PTX directives
and reduced spilling, but does not measure runtime occupancy.

The production microbatch requires 64-bit packed-history offsets. At
`B=64, K=64`, each FP32 history contains 2,214,592,512 elements (8.25 GiB), beyond
signed 32-bit element indexing. The qualified source forms these offsets in
64-bit integers; its boundary test exercises the actual store and both read
helpers across that limit.

Before candidate timing, confirmation compares the full batch with four
independently evaluated batch-16 slices, preserving the explicit cotangent and
concatenating output and all six gradients without rescaling. Two checks must
pass: production BF16 results use the declared exact-backend limits of 0.003
relative RMS and 0.01 normalized maximum error; the same production inputs and
cotangent, promoted exactly to FP32, must pass `1e-5` for both metrics in every
channel. Final BF16 rounding can expose otherwise tiny full/split differences:
diagnostic `222321` found sparse, mostly one-ULP differences, while unchanged
values evaluated in FP32 passed all seven strict checks (worst relative RMS
`8.54e-7`, normalized maximum `6.68e-7`). The report retains both checks and
verifies promotion preserves every input and cotangent value. Either failure
blocks timing; the guard is enabled by `--reference-batch-size 16` and is outside
all measured intervals.

The completed B64 kernel matrix in job `222322` passes both split-reference
gates and all eight variants' output/gradient checks. It uses the same
`T=2048, H=16, V=64` fixture and two warps, with two rounds of five warmups and
30 timed calls each. Pooled forward-plus-backward means are:

| Key width | Precision | CTA guidance 0, ms | CTA guidance 1, ms |
| --- | --- | ---: | ---: |
| 16 | `tf32x3` | 250.704 | 213.762 |
| 16 | `bf16_values` | 134.231 | 94.922 |
| 64 | `tf32x3` | 3,093.777 | 561.691 |
| 64 | `bf16_values` | 2,001.845 | 334.233 |

The selected `bf16_values`/CTA1 setting is 2.64x faster at K16 and 9.26x faster
at K64 than this job's default-precision/CTA0 control. Maximum relative RMS
error remains 0.260% at both widths. The promoted FP32 batch-split check is
identical at K16; at K64 its worst relative RMS and normalized maximum errors
are `8.54e-7` and `6.68e-7`. Native FLA's matching B64 calls average 3.819 ms
at K16 and 5.330 ms at K64 on the same physical GPU, with matching input hashes
and 60 synchronized wall samples per width. The optimized cubic kernel remains
24.85x and 62.71x slower than this linear-GDN reference. Kernel speedups alone
do not establish full-model throughput or training-quality equivalence.

The full-model table pools raw samples from two independent trials per setting,
with execution order reversed in the second trial. Each uses global batch 256,
with five warmup updates per trial except the slow K64 default control, which
uses two. Compute excludes compilation and warmup; the JAX intervals separately
include the input pipeline and ordinary checks/logging described above. The
explicit paired selections and both timing clocks are in the generated
[summary](runs/poly-gdn-tuning/summary.json), under `training_comparisons`.

| Key width | Implementation | Compute, s/update | Compute tokens/s | JAX interval, s/update | Measured updates |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 | Cubic `tf32x3`, CTA0 | 11.051 | 47,444 | 11.057 | 20 |
| 16 | Cubic `bf16_values`, CTA1 | 4.878 | 107,483 | 4.884 | 50 |
| 16 | Native FLA linear GDN, resident batch | 1.034 | 506,842 | — | 60 |
| 64 | Cubic `tf32x3`, CTA0 | 125.294 | 4,184 | 125.300 | 6 |
| 64 | Cubic `bf16_values`, CTA1 | 14.953 | 35,062 | 14.959 | 16 |
| 64 | Native FLA linear GDN, resident batch | 1.250 | 419,408 | — | 60 |

Full-model compute is 2.27x faster at K16 and 8.38x faster at K64 than the
default cubic control. Optimized cubic remains 4.72x and 11.96x slower than
native FLA, respectively. K16 trial-mean spreads are 0.75% for default cubic,
2.50% for optimized cubic, and 0.06% for native FLA; K64 spreads are below
0.31% for all three. Cubic parameter counts are 195,708,864 / 211,498,944 for
K16/K64, versus 195,053,504 / 210,843,584 for native FLA. All 266 updates across
the twelve trials, including 212 timed updates, have finite loss, gradient norm,
and parameter norm. No checkpoints are retained. These short throughput trials
do not compare training quality.

From a committed Kennel worktree at `8fd821a`, sync the CUDA13 and native Torch
environments according to `CLUSTERS.md`. Each command requests one B300 and
writes fresh job-specific reports. Run confirmation after inspecting tuning and
qualification; `bf16_values 2 1` below names the candidate under evaluation:

```bash
POLY_SOURCE=8fd821a09f6d7ca5e8cd8f65778d391abbc35d4e
mkdir -p scaffolds/transformer/validation/logs/poly-gdn-tuning
sbatch --partition=debug,standard \
  --output=scaffolds/transformer/validation/logs/poly-gdn-tuning/profile-%j.log \
  scaffolds/transformer/launchers/benchmark_poly_gdn_tuning.sbatch "$POLY_SOURCE" profile
sbatch --partition=debug,standard \
  --output=scaffolds/transformer/validation/logs/poly-gdn-tuning/tune-%j.log \
  scaffolds/transformer/launchers/benchmark_poly_gdn_tuning.sbatch "$POLY_SOURCE" tune
sbatch --partition=debug,standard \
  --output=scaffolds/transformer/validation/logs/poly-gdn-tuning/qualify-%j.log \
  scaffolds/transformer/launchers/benchmark_poly_gdn_tuning.sbatch "$POLY_SOURCE" qualify bf16_values 2 1
sbatch --partition=debug,standard \
  --output=scaffolds/transformer/validation/logs/poly-gdn-tuning/confirm-%j.log \
  scaffolds/transformer/launchers/benchmark_poly_gdn_tuning.sbatch "$POLY_SOURCE" confirm bf16_values 2 1
```

Regenerate the summary from the record's explicit report selection and hashes:

```bash
python3 scaffolds/transformer/validation/summarize_poly_gdn_tuning.py \
  --selection scaffolds/transformer/validation/runs/poly-gdn-tuning/record.json \
  --output scaffolds/transformer/validation/runs/poly-gdn-tuning/summary.json \
  --csv scaffolds/transformer/validation/runs/poly-gdn-tuning/summary.csv
```

The helper keeps incomplete/rejected variants visible, audits input identities and
sample counts, and does not pool measurements across physical GPUs. Its
`training_comparisons` pool the explicitly named trial pairs from raw samples,
retain each timing clock, and report per-trial counts and mean spread.

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
