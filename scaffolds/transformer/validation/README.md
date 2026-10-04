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
