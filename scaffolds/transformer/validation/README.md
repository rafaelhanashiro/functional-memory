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

### Row-power shifted query polynomials (2026-11-21)

`attention/shifted_polynomial.py` evaluates a per-query polynomial `sum_r c_r(t) z^r` of degree `M`, `z = s` or
`softplus(s)`, in units `e^(-m_t)` with `m_t = M log max(1, max_i |z_ti|)`, so every term is a power of
`|u| = |z| e^(-m_t/M) <= 1`; head RMSNorm's max-abs prescale takes the shift. Its fused Pallas-Triton kernel
(`backends/fused_shifted_polynomial.py`) evaluates the polynomial and its slope by Horner's rule in `u` and each
coefficient's gradient from the moments `sum_i dp_i u_i^r`. On a shiba B200 (job 10257) the kernel's eight
composition tests in `test_shifted_polynomial_attention.py` pass against FP64 (degrees 9 and 17, both transforms,
with stress rows whose `z^17` overflows FP32). Against FP64 at 2,048 tokens (BF16 Q/K/V, softplus) the output is
within 3.3e-6 of the largest entry, dQ/dK/dV within 3.4e-3 and the coefficient gradients within 4.4e-5; at the
production shape every output and gradient is finite for score scales 1 to 2,000. Where one key dominates a
row, head RMSNorm leaves that row's dominant coefficient gradient a near-cancellation, which FP32 (the blocked
reference too) resolves to about 1e-3 of the gradient's largest entry.

At the training micro-batch (16 x 2,048 tokens, 14 heads, Q/K width 4, value width 64), degree 17 with
softplus took 2.21 ms forward and 15.00 ms forward and backward with the first tiles, and 12.19 ms after a
19-setting sweep (job 10258: dQ 64 x 32 on 4 warps, dK/dV 16 queries by 32 keys on 4); degree 9 took 11.82 ms
before the sweep, and Mosaic's transformed kernel at degree 9 5.52 ms. B300 and H100 are qualified only by a
run's own acceptance on its allocated GPU.

**Coefficient sources (2026-11-22).** The kernel evaluates any per-query coefficient vector, so the shift also
takes the tanh query polynomial and static coefficients, fixed (Taylor's `1/r!` or a recipe's table) or learned,
broadcast to every query (`attention/pieces.py` `polynomial_shift_problem`). On CPU each source's degree-16 model on
the raw score equals the unshifted model, logits and every parameter gradient; the same cases compare the fused
model with the blocked reference's on GPU (`test_shifted_polynomial_attention.py`, `SOURCES`). The diagnostics'
cancellation ratio, `sum_i |w_ti| / |sum_i w_ti|` per row, is measured from the shifted weights, which the row factor
leaves unchanged; it had been set to 1, right only for a nonnegative sum of squares.

**Prefactors and pure tanh (2026-11-22).** The kernel also evaluates `f(s) p_t(s)` for `f = softplus` or `exp` of the
raw score. Each key takes its own units: the bound `b_i = log f(s_i) + M log max(1, |s_i|)` on every `|f(s_i) s_i^r|`,
the row the largest (at least zero), and `p_t(s_i) / max(1, |s_i|)^M` by Horner's rule in `s` where `|s| <= 1` and
in `1/s` over the reversed coefficients elsewhere; coefficient gradients sum ascending powers of `s` and descending
powers of `1/s` likewise. Units set by the largest `|s|` alone underflow every moderate key once a far negative key,
which the prefactor suppresses, sets them. On CPU the arithmetic matches FP64 at score scales 0.5 to 10,000 with far
negative keys, and the blocked reference matches dense FP64 (output and every gradient, degrees 9 and 17, with stress
rows whose `e^s` and `s^17` overflow FP32; for `e^s` the positive stress keys are near 100, since at `s = 1000` FP32
holds `s + M log s` only to 6e-5). On a B200 (jobs 11158, 11167) the fused kernel matches dense FP64 in the same
cases, and the fused models of pure tanh coefficients `tanh(x_hat W_r + beta_r)/r!` (the inner bias
`attention_coefficient_tanh_bias` with `attention_coefficient_tanh_bias_initial`) with each prefactor match the blocked
reference's. The fused and blocked models' logits are compared at 5e-4 of the largest: on a B200 one token of the
degree-sixteen tanh query test model differs by 4.2e-5 (each evaluation deterministic), which failed the earlier
2e-4 elementwise check. Acceptance's GPU stage for a shifted recipe takes this module's cases and leaves out the
coefficient forms' Mosaic case (`test_coefficient_forms.py`), which qualifies kernels a shifted run never trains on
and skips on a B200; on CPU the trainer resumes a pure tanh model with each prefactor to the uninterrupted state.

**Q/K width 64 on H200 (2026-11-23).** The kernel's tiles hold Q/K and value widths up to 64 unchanged.
`tests/test_shifted_polynomial_key_width.py` runs it at Q/K and value width 64: the fused programs against FP64 at degrees
8 and 16 over 257 tokens (five query tiles, a masked tail) with scores to about 20 and coefficients
`tanh(g + atanh(1/2))/r!` of either sign, and the fused pure tanh (degrees 8 and 16) and tanh query (degree 16) models
against the blocked reference's. On an ORCD H200 (job 30217370, `qualify_shifted_polynomial_width.sbatch`) they pass
with the module's width-4 GPU cases. The largest errors against FP64, relative to each reference's largest entry, are
1.4e-6 (output) and 3.7e-6 (gradients) at degree 8 and 9.4e-6 and 2.7e-5 at degree 16; a degree-16 row whose signed
terms cancel resolves its smaller outputs to 2.9e-5 (job 30214311), so the output check bounds the error by 1e-4 of the
largest entry. One layer's forward and backward at 16 x 2,048 tokens, 14 heads and BF16 Q/K/V
(`benchmarks/benchmark_shifted_polynomial_width.py`) took 3.1 / 10.9 ms at degree 8 and 4.2 / 14.7 ms at degree 16.
H100 is qualified only by a run's own acceptance, whose GPU stage takes these cases for a width-64 shifted run
(`experiments/transformer_pp/pure_tanh_dk64_acceptance.py`).

### Head RMSNorm max-abs prescale (2026-11-20)

`model.attention_head_rms_prescale: max_abs` ([README](../README.md#unnormalized-output-scaling)) evaluates head
RMSNorm on each token and head's output divided by its largest magnitude `c` (at least one), with `eps / c²`: the
plain function and gradient in real arithmetic, `c` held constant to differentiation. `none`, every historical
recipe, computes as before, bit for bit.

**The defect it removes.** The plain FP32 backward forms `(mean(y²) + eps)^(-3/2)`: a subnormal number once a head's
output RMS passes 4.4e12, zero past 1.1e15. It scales the term that removes the gradient's component along `y`.
The probe is 2,048 tokens of two heads with 64 channels, unit-RMS random directions times the scale, a cotangent
of scale 1e-4 and gains near one, against host FP64; the error is the relative L2 of the gradient in the rows, and
the radial fraction is the share of that component the gradient keeps (zero for the exact Jacobian).

| Output RMS | Plain, GPU | Its radial fraction | Plain, CPU | Prescaled, GPU | Prescaled, CPU |
| --- | --- | --- | --- | --- | --- |
| 1e-25 to 3e12 | at most 7.1e-8 | at most 1e-6 (from RMS 1) | at most 7.5e-8 | at most 1.8e-7 | at most 1.7e-7 |
| 4.4e12 to 1e13 | at most 5.7e-8 | at most 2.5e-7 | 0.125 | at most 1.8e-7 | at most 1.7e-7 |
| 3e13 | 7.0e-6 | 5.6e-5 | 0.125 | at most 1.8e-7 | at most 1.7e-7 |
| 1e14 | 6.6e-5 | 5.3e-4 | 0.125 | at most 1.8e-7 | at most 1.7e-7 |
| 2e14, 3e14, 4.5e14 | 0.0017, 0.0020, 0.0027 | 0.013, 0.016, 0.022 | 0.125 | at most 1.8e-7 | at most 1.7e-7 |
| 6e14 | 0.026 | 0.21 | 0.125 | at most 1.8e-7 | at most 1.7e-7 |
| 9e14 to 1e18 | 0.125 | 1.0 | 0.125 | at most 1.8e-7 | at most 1.7e-7 |
| 1e19 to 1e30 (the mean square overflows and the plain evaluation's guard rescales the row) | at most 6.7e-8 | 1e-7 | 6.8e-8, and 7.3e-7 at 1e30 | at most 1.8e-7 | 1.7e-7, and 7.3e-7 at 1e30 |

The GPU columns are H100's, B200's and B300's: their plain errors agree to every recorded digit, and the prescaled
ones are at most 1.2e-7 on H100 and B200 and 1.8e-7 on B300. Prescaled errors are the largest of the output, the
row gradient and the gain gradient; its radial fraction is at most 1e-6 from RMS 1 to 1e25. XLA's GPU code keeps
subnormal numbers, so the plain gradient degrades gradually there; its CPU code flushes them, so the whole term is
lost from 4.4e12 (and at 1e30 the gradient's smallest elements, below FP32's normal range, are flushed with either
evaluation). With a random cotangent the lost term is 12.5% of the gradient (`1 / sqrt(64)`); through a model it
is most of the Q/K gradient (below).

**Where the gradient ends.** The prescaled gradient is exact to rounding wherever FP32 holds the gradient itself,
which is not every scale: its elements are of size cotangent / RMS (1e-34 at an RMS of 1e30 with the probe's
cotangent), and FP32's normal range ends at 1.2e-38. The module's probe with cotangents of 1e-12 to 1e4 at RMS 1e19
to 1e37 (86 pairs, CPU, against its FP64 reference) has an error that depends on that quotient alone:

| Cotangent / RMS | 1e-33 or more | 1e-34 | 1e-35 | 1e-36 | 1e-37 | 1e-38 | 1e-39 or less |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Row gradient, relative L2 | at most 7.6e-8 | 7.3e-7 | 2.3e-5 | 6.7e-4 | 0.021 | 0.54 | 1 (all zero) |

The plain evaluation returns the same gradient bit for bit at every pair (from 1e19 its guard for an overflowing
mean square is the same division), so the bound is FP32's, not the prescale's; the output and the gain's gradient
stay within 1.9e-7. XLA's CPU code flushes numbers below the normal range to zero, and a GPU keeps them as
subnormal numbers with fewer bits (the probe at 1e30, a quotient of 1e-34: within 1.8e-7 on the three GPUs,
7.3e-7 on CPU). The module holds the row gradient to 1e-6 at a quotient of 1e-33 for each of those cotangents.

**CPU** (`JAX_PLATFORMS=cpu`, at `477cfc2d`): 3,524 tests in 62 modules passed (10 skipped for want of a second host
device): `tests/test_head_rms_prescale.py` (79 CPU cases); the output-scaling, row-maximum-shift, unnormalized
(attention, configuration, training), configuration, fixed-polynomial configuration, telemetry, acceptance (three
modules), continuation (two), model, training, initialization, composition, strips, exponential score-function,
attention, QK-norm and PaTH (three) modules; Transformer++'s 30 adapter, launcher and diagnostic modules,
attention_scaling's three, the records modules (two) and the two result modules whose recipe checks name the setting.
In FP64 the production arithmetic equals the plain function to 1e-12 of each row in value and VJP at RMS 1e-25 to 1e30
(eps 1e-12, 1e-6 and 1; a zero row and a row of mixed channel scales with them), also when the divisor is
differentiated, and the shifted form to 1e-11 on both sides of its linear range's boundary at shifts of -60 to 150. In
FP32 "none" is the pre-existing function bit for bit (output and VJP, with and without a row shift, compiled or not),
and rows of at most one are the plain arithmetic bit for bit. The composition with the row shift matches FP64 to
1.5e-7 in 32 cases (shifts of -60 to 2,000, rows of 1e-30 to 1e25), nine of which the shifted arithmetic alone gets
wrong or nonfinite; the shifted scan and strips with the prescale match the FP64 function at row maxima of 5 to 440.
Two-layer models (QK RMSNorm with head scales, Q/K width 4, RoPE unless noted) whose value projections are scaled by
1e16, so that every head output is above 1e15, against the same models in FP64:

| Kernel | Q/K gradient, prescaled, FP32 layers | BF16 layers | Q/K gradient, plain |
| --- | --- | --- | --- |
| Fixed Taylor `T_8(s)` | 8.9e-7 | 0.022 | 0.95 |
| Static `p(s)` | 8.0e-7 | 0.022 | 0.95 |
| Query `p_t(s)` | 8.0e-7 | 0.022 | 0.95 |
| Softplus SoS `z h_t(z)` | 1.5e-6 | 0.024 | 1.02 |
| Exponential with the row shift | 9.0e-7 | 0.020 | 0.95 |
| Exponential with the row shift and a fixed constant decay, no RoPE | 4.7e-7 | 0.015 | 1.27 |

Every other parameter group is within 1.2e-6 (FP32) with the prescale; BF16 layers keep the unscaled BF16 model's
distance from FP64 (0.019 to 0.028 in Q/K). The plain evaluation's coefficient gradients are off by 2.3 to 6.9 times
their norm. A PaTH exponential model, whose attention calls the normalization at its own site, keeps its unscaled
self's loss and gradients to 1.6e-4 with the values scaled by 1e16 (1.41 times the Q/K gradient's norm without the
prescale); with the prescale removed from that call, this case alone fails. Refusals, the allocation default and
flag, the continuation launcher's copies, every study adapter's partner comparison (each field but the prescale
still refused), the run index, identity, telemetry (unchanged), continuation and exact training resumes pass.
`records receipts` over transformer_pp, attention_scaling and state_rank at `477cfc2d`: all 586 unchanged (529
`same`, 57 `same rejection`).

**GPU** (`qualify_head_rms_prescale.sbatch`, one GPU on `long`, `-m gpu` over `test_head_rms_prescale.py` and the
existing Mosaic case of the unprescaled head RMSNorm):

| GPU | Job | Source | Result | JUnit seconds |
| --- | --- | --- | --- | --- |
| B200 (`shiba`, long) | 9250 | `520f5293` | 19 passed: probe, shifted composition, 8 kernel cases, 8 model cases, the control | 314 |
| B300 SXM6 (`akita`, long) | 9292 | `370afb2e` | 9 passed: probe, shifted composition, 3 kernel cases and 3 model cases at Q/K width 64, the control; 10 skipped as not admitted | 128 |
| H100 SXM (`dogo`, long) | 9291 | `370afb2e` | 19 passed: probe, shifted composition, 8 kernel cases, 8 model cases, the control | 183 |

`370afb2e` differs from `520f5293` in the scaffold README's text alone (the code and tests are identical).
At `477cfc2d` the executable code of `attention/output_scaling.py`, `attention/pieces.py` and
`attention/multihead.py` and the module's GPU cases are those of `370afb2e`: under `attention/` only comments and
docstrings differ, and the other changes are CPU cases and code outside the normalization (the run index, the
adapters' comparison, the continuation launcher, the recorded formula's clause).

H100 (CUDA 12 environment), [record](runs/head-rms-prescale-h100/record.json): the probe and the shifted
composition as on B200 (prescaled at most 1.2e-7 and 1.5e-7), and B200's kernel and model errors at both Q/K
widths: every kernel error within B200's ranges below (dQ 0.0030 to 0.0044 prescaled against 0.20 to 0.90 plain
with the scaled values) and every prescaled model group within 0.003 of its table (Q/K 0.011 to 0.020 against
0.61 to 0.86 plain).

B300 (CUDA 13 environment), [record](runs/head-rms-prescale-b300/record.json): the probe and the shifted
composition as on B200 (prescaled at most 1.8e-7 and 1.3e-7); fixed Taylor, static and query at Q/K width 64 give
B200's kernel and model errors (Q/K 0.011 prescaled against 0.81 plain with every head output above 1e15). B300 does
not admit the transformed kernels and its query kernel is not qualified at Q/K width 4, so its job skips the
softplus-SoS cases and the width-4 cases (10 of 19), which its final check allows.

B200 (CUDA 13 environment), [record](runs/head-rms-prescale-b200/record.json):

- Probe: the table above. Shifted composition: at most 1.3e-7 over the 32 cases; zero rows stay zero with finite
  gradients at every shift.
- Mosaic kernels at 2,048 tokens into the prescaled normalization, against dense FP64 of the same BF16 operands, at
  Q/K widths 4 and 64 (fixed Taylor, static and query at degree eight, softplus SoS with its input power at degree
  nine). Relative L2 over the eight cases, with the values at their own scale and scaled by 1e16 (head outputs
  above 1e15) alike: output 0.0028 to 0.0029, dQ 0.0030 to 0.0044, dK 0.0029 to 0.0034, dV 0.0029 to 0.0030, gain
  0.0020 to 0.0029, coefficients 0.0010 to 0.0045 (0.0086 for static at width 4 with the scaled values). Plain at
  the scaled values: dQ 0.20 to 0.90, dK 0.16 to 0.32, dV 0.10 to 0.14, coefficients 0.13 to 0.89, and its output
  0.0028 to 0.0029, as prescaled.
- One training gradient of two-layer BF16 models on those kernels (width 128, two heads, 512 tokens), every head
  output above 1e15, against the same models in FP64 on the blocked backend:

| Kernel, Q/K width | Q/K, prescaled | Unscaled model | Coefficients, prescaled | Q/K, plain | Coefficients, plain |
| --- | --- | --- | --- | --- | --- |
| Fixed Taylor, 4 | 0.014 | 0.014 | | 0.61 | |
| Fixed Taylor, 64 | 0.011 | 0.011 | | 0.81 | |
| Static, 4 | 0.014 | 0.014 | 0.013 | 0.61 | 0.99 |
| Static, 64 | 0.011 | 0.011 | 0.009 | 0.81 | 1.2 |
| Query, 4 | 0.014 | 0.014 | 0.013 | 0.61 | 1.0 |
| Query, 64 | 0.011 | 0.011 | 0.010 | 0.81 | 1.3 |
| Softplus SoS, 4 | 0.020 | 0.018 | 0.018 | 0.62 | 9.6 |
| Softplus SoS, 64 | 0.014 | 0.013 | 0.011 | 0.86 | 5.4 |

  V, O, gain and the other groups: 0.007 to 0.015 with the prescale, 0.32 to 0.55 without. The loss is the FP64
  model's to 3e-5 either way: the defect is in the backward alone.

This qualifies the normalization's arithmetic on the device and its composition with the Mosaic kernels, not a
training run: each production allocation still runs its adapter's CPU and GPU suites (for a prescaled recipe the
module's CPU cases and its probe and composition cases on the run's GPU), the initialization check, the pilot and a
fresh-process resume.

### Head RMSNorm backward at run checkpoints (2026-11-20)

`head_rms_norm` ([README](../README.md#unnormalized-output-scaling)) divides a head's output `y` (per token and head,
64 value channels) by `sqrt(mean(y^2) + eps)` in FP32. Its backward multiplies by `(mean(y^2) + eps)^(-3/2)`, which
is subnormal in FP32 once `mean(y^2)` passes 1.9e25 (output RMS 4.4e12) and under the smallest subnormal past 8e29
(RMS 8.9e14); the term it scales removes the gradient's component along `y`. A CPU flushes subnormals and loses the
term from RMS 4.4e12 (next section). `validation/measure_head_rms_backward.py` measures where a GPU loses it and, at
eleven run checkpoints, how far that moves the gradient of the next-token NLL. Record:
[`runs/head-rms-backward/record.json`](runs/head-rms-backward/record.json).

**Method.** Each run keeps its recipe, its precision (`train.dtype: bfloat16`) and its checkpoint; only
`attention_backend` becomes `blocked` (the recipe's scan or strips), so every kernel runs on any GPU. The gradient
with respect to every parameter is taken on the first four held-out sequences of the run's eval manifest (2,048
predictions each; the first is the diagnostic sequence), one sequence per evaluation, and averaged. One compiled
program evaluates the model with five head RMSNorms, chosen at run time:

| Evaluation | Forward | Backward |
| --- | --- | --- |
| production | `scale_attention_output` as committed at the run's source commit | its autodiff |
| stable | `(y/c) / sqrt(mean((y/c)^2) + eps/c^2)` in FP32, `c` the row's largest magnitude (at least 1) held constant: the same function of `y` for any constant `c`, so its gradient is the function's | its autodiff, every factor in FP32's normal range |
| fp64_head | NumPy FP64 on the host | FP64 on the host |
| stable_backward | production's | the stable evaluation's VJP |
| fp64_backward | production's | FP64 on the host |

The whole model in FP64 (parameters upcast, every layer and the blocked kernel in FP64; QK RMSNorm, RoPE tables and
coefficient generators keep their FP32 steps) gives a sixth gradient on the first sequence. A probe takes the VJP of
each committed version on synthetic FP32 rows of a given RMS against FP64.

**Resolution.** A BF16 model rounds differently after any last-bit change of an activation, so two evaluations
whose forwards differ in the last FP32 bit differ by the model's rounding noise: between the two exact evaluations
with their own forwards (stable, fp64_head), 1.3-1.4% of the whole gradient (1.7-2.1% of Q/K) in the Q/K width-64
runs, 1.9-2.6% (3.4-4.8%) at widths 16 and 8, and 2.4-6.1% (3.6-12%) at width 4; two compilations of the production
evaluation differ as much (1.4-3.9%, job 9109). With one forward (stable_backward against fp64_backward) the floor
is 0.4-0.6% (0.6-1.0% of Q/K). The five runs trained before `c426b65a` did not share it: their compiled production
evaluation differs in the last bit from the forward the two others take (unequal losses), so their production
columns carry the larger floor.

**The GPU keeps subnormals.** VJP of the committed head RMSNorm on 4,096 synthetic FP32 rows; "radial share" is the
part of the component along `y` left in the result (0 exact, 1 when the term is lost). RTX A6000 and A100 agree to
eleven digits, in the radial share exactly. The stable evaluation is within 7e-8 at every scale.

| Output RMS | Relative L2 error | Radial share |
| --- | --- | --- |
| up to 1e13 | under 8e-8 | under 3e-7 |
| 3e13 | 7.0e-6 | 5.6e-5 |
| 1e14 | 6.6e-5 | 5.3e-4 |
| 2e14, 3e14, 4.5e14 | 0.17%, 0.20%, 0.27% | 1.4%, 1.6%, 2.2% |
| 6e14 | 2.6% | 21% |
| 9e14 to 1e18 | 12.5% (the radial part of a random cotangent) | 1.00 |
| from the FP32 overflow of the mean square (RMS 2.3e18 over 64 channels) | before `c426b65a`: output and gradient zero; from `c426b65a`: 7e-8 | 0 |

**Runs** (job 9142, A100). Rows are (token, head) pairs over all layers and the four sequences; the counts over
1.9e25 and 8e29 include the rows whose FP32 mean square overflows.

| Run | Update | Largest output RMS | Rows over 1.9e25 | Rows over 8e29 | Layers with rows over 8e29 | Rows whose FP32 mean square overflows | NLL production / stable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r6a76e7fbde45` exp, Q/K width 64 | 7,500 | 5.3e10 | 0 | 0 | none | 0 | 2.7062 / 2.7063 |
| `r841f35a24241` softplus-input SoS (degree-8 core), Q/K width 64 | 7,500 | 9.3e9 | 0 | 0 | none | 0 | 2.6934 / 2.6939 |
| `re3515eed3576` fixed Taylor m=8, Q/K width 4 | 7,500 | 1.8e17 | 4.2% | 0.52% | 9 of 15, at most 2.1% | 0 | 3.0727 / 3.0722 |
| `r0f5e38b0d934` static m=8, Q/K width 4, in training | 3,500 | 1.5e13 | 0.0058% | 0 | none | 0 | 3.2950 / 3.2949 |
| `rd5e3ec5a4f02` query m=8, Q/K width 4, in training | 4,000 | 1.7e12 | 0 | 0 | none | 0 | 3.2124 / 3.2124 |
| `r5a25f500bfa8` softplus-input SoS (degree-8 core), Q/K width 4 | 7,500 | 1.2e16 | 1.2% | 0.076% | 6 of 15, at most 0.39% | 0 | 3.0071 / 3.0072 |
| `rcace22b32f93` fixed Taylor m=12, 12-layer width-640, 4x Chinchilla | 30,205 | 1.2e16 | 4.7% | 0.0043% | layer 0, 0.051% | 0 | 2.7404 / 2.7408 |
| `r1f935f18fe21` query SoS m=12, 12-layer width-640, 4x Chinchilla | 30,205 | 1.1e16 | 1.4% | 0.0045% | layer 0, 0.054% | 0 | 2.7377 / 2.7378 |
| `r6f84b1db6de6` unshifted exp, Q/K width 16 | 7,500 | 1.7e24 | 5.1% | 2.4% | 10 of 15, at most 11.2% | 0.40% | 2.8206 / 2.8208 |
| `r535bc84a7772` unshifted exp, Q/K width 8 | 7,500 | 8.1e30 | 13.4% | 5.8% | 11 of 15, at most 18% | 2.2% | 3.0605 / 3.0605 |
| `r7820049184ef` exp with fixed ALiBi decay, no RoPE, Q/K width 64 | 7,500 | 2.1e22 | 0.76% | 0.38% | 3 of 15, at most 5.4% | 0.091% | 2.7008 / 2.7009 |

Relative L2 difference of the production gradient from the stable-backward one, `||g_production - g|| / ||g||`, by
parameter group (Q/K: Q and K projections, QK gains and head scales; "other": embeddings, MLPs, norms, output head);
then the floors (one forward: stable_backward against fp64_backward; two forwards: stable against fp64_head) and the
distances from the FP64 model on the first sequence:

| Run | One forward | Q/K | V | Output projections | Coefficients | Other | All | Cosine Q/K / all | Floor, one forward: Q/K / all | Floor, two forwards: Q/K / all | From the FP64 model, Q/K: production / stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r6a76e7fbde45` | no | 1.8% | 1.2% | 1.3% | none | 1.2% | 1.3% | 0.99984 / 0.99992 | 0.60% / 0.43% | 1.8% / 1.3% | 1.8% / 1.7% |
| `r841f35a24241` | no | 1.6% | 1.1% | 1.3% | 2.0% | 1.2% | 1.2% | 0.99987 / 0.99992 | 0.59% / 0.43% | 1.7% / 1.3% | 1.6% / 1.6% |
| `re3515eed3576` | yes | 12.0% | 7.4% | 7.8% | none | 4.9% | 7.5% | 0.99294 / 0.99722 | 0.78% / 0.54% | 3.6% / 2.4% | 13.7% / 3.9% |
| `r0f5e38b0d934` | yes | 0.69% | 0.46% | 0.44% | 0.66% | 0.31% | 0.43% | 0.99998 / 0.99999 | 0.67% / 0.43% | 11.9% / 6.1% | 27% / 38% |
| `rd5e3ec5a4f02` | yes | 0.71% | 0.44% | 0.48% | 0.32% | 0.30% | 0.42% | 0.99998 / 0.99999 | 0.69% / 0.47% | 7.0% / 3.4% | 5.0% / 5.2% |
| `r5a25f500bfa8` | yes | 1.9% | 0.84% | 1.1% | 1.8% | 0.83% | 1.1% | 0.99982 / 0.99994 | 0.93% / 0.56% | 4.8% / 3.1% | 5.4% / 5.0% |
| `rcace22b32f93` | no | 2.3% | 1.2% | 1.3% | none | 1.2% | 1.3% | 0.99974 / 0.99992 | 0.66% / 0.41% | 1.9% / 1.3% | 2.7% / 2.1% |
| `r1f935f18fe21` | no | 2.3% | 1.3% | 1.5% | 2.3% | 1.2% | 1.3% | 0.99974 / 0.99991 | 0.66% / 0.41% | 2.1% / 1.3% | 2.7% / 2.2% |
| `r6f84b1db6de6` | yes | 854% | 230% | 264% | none | 148% | 291% | 0.179 / 0.347 | 0.87% / 0.49% | 3.4% / 1.9% | 1,450% / 3.2% |
| `r535bc84a7772` | yes | 4,310% | 1,610% | 2,110% | none | 1,160% | 1,960% | 0.011 / 0.044 | 1.0% / 0.58% | 4.8% / 2.6% | 3,550% / 4.6% |
| `r7820049184ef` | no | 23.6% | 7.8% | 8.7% | none | 6.3% | 8.7% | 0.973 / 0.996 | 0.64% / 0.44% | 1.9% / 1.4% | 31% / 3.7% |

- **Q/K width 64 (`r6a76e7fbde45` exp, `r841f35a24241` SoS): not affected.** Their largest output RMS is more
  than 500 times below the first inexact scale (3e13); production differs from the exact evaluations by rounding
  noise, and both lie 1.6-1.8% from the FP64 model. Their recorded row sums never passed 1.5e10 and 7.6e9 in training.
- **Fixed Taylor m=8 at width 4 (`re3515eed3576`): affected.** 0.52% of rows are past 8e29; the Q/K gradient is off
  by 12.0% (11-13% per sequence), the V, output-projection and other gradients by 5-8% and the head RMSNorm gains'
  by 20%; production lies 13.7% from the FP64 model where the stable evaluation lies 3.9%. Its recorded row sum
  passed 1e15 at update 3,500 (3e14 at 3,000, 1.1e17 at 7,500).
- **Softplus-input SoS at width 4 (`r5a25f500bfa8`): 1.9% of Q/K, 1.1% overall,** twice the one-forward floor and
  under the 4.8% rounding noise of its own Q/K gradient; rows past 8e29 since about update 3,000 (recorded row sum
  7.4e14).
- **Static at 3,500 and query at 4,000 (width 4): not affected yet** (largest RMS 1.5e13 and 1.7e12; production
  equals the exact backward to the floor). The static run's recorded row sum grows about fivefold per 500 updates
  (6.9e13 at 4,000); the query run's doubles (8.6e11 at 4,000).
- **12-layer width-640 at 4x Chinchilla: no error above rounding noise.** 31% and 16% of layer 0's rows are past
  1.9e25 but 0.05% past 8e29; the 2.3% Q/K difference is close to that of two exact evaluations (1.9-2.1%), and the
  FP64 model lies 2.7% from production and 2.1-2.2% from the stable evaluation. A device that flushes subnormals
  would lose the term in all of those rows.
- **Unshifted exp at widths 16 and 8: the production gradient is mostly the lost term.** Its Q/K norm is 8.7 and 43
  times the exact one (2.01 against 0.232, 13.2 against 0.307), cosine 0.18 and 0.011; the whole gradient's norm is
  3.1 and 20 times the exact one, against clipping at 1. The FP64 model agrees with the stable evaluation (3.2%,
  4.6%).
- **ALiBi exp (`r7820049184ef`): affected,** 23.6% of Q/K (13-31% per sequence), from about update 5,000 (recorded
  row sum 5.8e14); its source also zeroed the 1.4% of layer 13's rows past the FP32 overflow.

The polynomial runs were not handicapped as the unshifted exponential was: their largest error is 12% of the Q/K
gradient at one final checkpoint, against 8.5 and 43 times the gradient. A weight's score derivative is `m/s` of
the weight for a degree-`m` monomial (0.02 at `s` = 390) and the whole weight for exp, and under 1% of polynomial
rows pass 8e29.

**Limits.**

- GPUs: RTX A6000 and A100 (both Ampere), the idle ones; the runs trained on H100, B200 and B300, which were not
  measured. The two agree on the probe to eleven digits; Hopper and Blackwell remain to be probed.
- Kernels: the blocked reference in place of the Mosaic and fused kernels the runs trained on (`r7820049184ef`
  trained on the blocked strips); the head RMSNorm is each run's own source.
- One checkpoint per run and four sequences, one per evaluation (training averages 256 in micro-batches of 16); the
  raw gradient, before clipping and Adam. Earlier updates are dated from recorded row sums, which lie within a factor
  of ten of the measured output RMS on the same sequence (0.8 to 10.5).
- A gradient error is not a perplexity cost: that takes a run trained with the stable backward.
- The static and query runs were still training; their later updates are not covered.

**Jobs** (`measure_head_rms_backward.sbatch`, `long`, each started on idle resources):

| Job | Node, GPU | Runs | Evaluations | Elapsed |
| --- | --- | --- | --- | --- |
| 9142 | `samoyed`, 2 x A100 80GB PCIe, six processes | all eleven | the five, the FP64 model, the probe | 8 min 55 s |
| 9109 | `doodle`, RTX A6000, two processes | eight (the three exp runs past its deadline) | production, stable, fp64_head; the production evaluation traced alone on four; the probe | 21 min 54 s |

Job 9109 ran the script before the two one-forward evaluations existed (its source is `0193a8a8`); on its eight
runs the production-against-stable differences agree with job 9142's within rounding noise (Q/K 12.5% and 12.5% for
`re3515eed3576`, 1.8% and 1.7% for `r6a76e7fbde45`). CPU: `validation/test_measure_head_rms_backward.py`, 14 tests.
Regeneration, from a checkout with both environments synced:

```bash
sbatch --partition=long --nodelist=<node> --gres=gpu:2 --cpus-per-task=12 --mem=160G \
  --output=scaffolds/transformer/validation/logs/head-rms-backward-%j.log \
  scaffolds/transformer/validation/measure_head_rms_backward.sbatch "<run lists>" \
  --sequences 4 --probe --fp64-model --direct-check 0
python3 scaffolds/transformer/validation/record_head_rms_backward.py 9142 9109 --checkpoints <hashes.json> \
  --order r6a76e7fbde45,r841f35a24241,re3515eed3576,r0f5e38b0d934,rd5e3ec5a4f02,r5a25f500bfa8,rcace22b32f93,r1f935f18fe21,r6f84b1db6de6,r535bc84a7772,r7820049184ef
```

### Row-maximum-shifted exponential (2026-11-20)

`model.attention_exp_shift: row_max` ([README](../README.md#unnormalized-output-scaling)) evaluates the
unnormalized exponential with head RMSNorm as `y_t = sum_i e^(z_ti - m_t) v_i`, `m_t` the largest causal logit of
row `t`, and the RMSNorm as `y_t / sqrt(mean(y_t^2) + eps e^(-2 m_t))`: the unshifted function and gradient in real
arithmetic, `m_t` held constant to differentiation. The fused Pallas-Triton exponential has its own forward (a
running row maximum, as in flash attention), dQ and dK/dV kernels for it, the blocked scan and strips their own
branch; the unshifted kernels and calls are unchanged.

**CPU** (`JAX_PLATFORMS=cpu`, at `62573dde`): 1,020 tests in 18 modules passed (66 skipped): the output-scaling,
unnormalized, configuration, telemetry, acceptance, strips, exponential score-function, composition, attention,
QK-norm, unnormalized-training, initialization, model, continuation, key-width adapter and records modules, and
`tests/test_exp_row_max_shift.py` (72 CPU cases). In FP64 the unshifted function and the shifted one, with `m_t`
held constant or differentiated, agree to 1e-11 in output and VJP at row maxima of 4.5, 18 and 28. The FP32 scan and
strips match the FP64 function at row maxima of 5.5, 50, 198 and 446 (output relative L2 at most 8e-7, Q/K/V and
gain gradients 5e-6); there the unshifted output overflows from 89 on, and below that its Q/K gradients already
depart from FP64 by 0.088 (row maximum 34), 0.42 (50) and 1.7 (88) relative L2 while its output stays within 4e-7:
the head RMSNorm backward's `rsqrt` derivative flushes to zero (on CPU) once a head's output RMS passes about 5e12.
Shifted and unshifted agree in FP32 (1e-5) and, with BF16 layers, within BF16 rounding, for the attention and a
two-layer model (QK RMSNorm with head scales, RoPE); a layer at row maxima past 150 trains on finite values that
match FP64; the strips equal the scan in FP64 with every decay form and score function; refusals, the allocation
default, diagnostics, telemetry and an exact training resume (scan and strips) pass. `records receipts` over
transformer_pp, attention_scaling and state_rank: all 573 unchanged (516 `same`, 57 `same rejection`).

**GPU** (`qualify_exp_shift.sbatch`, one GPU, `-m gpu` over `test_unnormalized_fused_attention.py` and
`test_exp_row_max_shift.py`):

| GPU | Job | Source | Result | JUnit seconds |
| --- | --- | --- | --- | --- |
| B300 SXM6 (`akita`, standard) | 8461 | `62c95e0f` | 80 passed: 25 shifted fused cases, 4 shifted strips cases, the module's 51 unshifted cases | 119 |
| H100 SXM (`dogo`, long) | 8464 | `bbe4f532` | 80 passed: the same cases | 194 |

The shifted fused cases: outputs, row maxima and VJPs at Q/K widths 4, 8, 16 and 64 (value width 64, 2,048 tokens,
BF16) against the shifted scan and dense FP64 at row maxima of 3-4 and of 180-280; head RMSNorm with the offset
against the FP64 function at row maxima of 135-210 (output and the Q/K/V and gain gradients); FP32 inputs at row
maxima up to 190; future-token invariance; finite scores up to 1e30 kept finite and nonfinite ones poisoning their
rows alone; underflow and zero scores; bitwise repeatability; a two-layer model on the fused kernel against the
scan. Bounds: relative L2 0.006 (output), 0.025 (dQ, dK), 0.015 (dV) in BF16, FP32 ones scaled by the row maximum,
row maxima within 1e-5; records: [B300](runs/exp-shift-fused-b300/record.json),
[H100](runs/exp-shift-fused-h100/record.json).

Per layer and micro-batch (16 x 2,048 tokens, 14 heads, BF16), the forward with its VJP takes, unshifted and
shifted, on B300: 3.98 and 3.91 ms at Q/K width 4, 4.03 and 4.06 at 8, 4.04 and 4.02 at 16, 5.69 and 5.66 at 64;
on H100 SXM: 3.05 and 3.01 at 4, 3.02 and 2.96 at 8, 2.96 and 2.99 at 16, 3.47 and 3.47 at 64
(`benchmarks/benchmark_exp_shift.py`; 0.9-1.4 s on B300 and 0.71-0.83 s on H100 of attention per 15-layer,
16-micro-batch update): the shift costs no time.

This qualifies the kernels, not a training run: each production allocation still runs its adapter's CPU and GPU
suites, the initialization check, the pilot and a fresh-process resume.

### Score prefactors, swish and Q/K width 4 on Mosaic (2026-11-20)

The Mosaic transformed query-polynomial kernels evaluate `W = f(s) p(phi(s))` in their causal tiles
([README](../README.md#score-prefactors-on-polynomials)): the families `query_times_<f>` and
`query_softplus_times_<f>` for the prefactors `f` = softplus, sigmoid, exp and swish of the raw score, and
`query_swish` for the swish transform alone (`common.score_function_family`). Each tile forms `f` and `f'` beside
`z = phi(s)` and `phi'`, sharing `y = 2^(-abs(s) log2(e))`; the weight is `f p(z)`, the dQ pass sums
`dc_r = sum_i dP f z^r`, and `dS = p'(z) (dP f) phi'(s) + dP f'(s) p(z)`. Vectors, shared memory, precision and
failure rows are the query family's. Q/K width 4 joins every Mosaic family, zero-padded to 16 like width 8.
Admission stays H100 (sm90) and B200 (sm100); B300 is refused before tracing. Recipes take a prefactor at degrees up
to eight (`api.MAX_PREFACTOR_QUERY_DEGREE`), the degrees these suites run. The kernels and configuration code
are identical at every commit below (`7f03ebda` to `78b5c25e` changed only GPU test modules).

**CPU** (`JAX_PLATFORMS=cpu`, at `78b5c25e`): every test module of the scaffold, the
Transformer++ study and the records passed, 6,448 tests in 192 modules (1,651 GPU or distributed cases deselected; the ignored summary log `logs/score-prefactor-cpu-78b5c25e.log`, SHA256 in the H100 record). Among them `tests/test_score_prefactor.py` (243):
swish's and exp's values and JVPs against FP64 autodiff; the family registry; every new family's H100 forward, dQ
and dK/dV kernels lowering in exactly `hopper_smem_bytes` and failing one byte below at degrees 1, 8, 9 and 12;
the B200 kernels and both entry points lowering with every gradient at 2,048 tokens and Q/K widths 4-64; the tile
mathematics against explicit FP64 `f(s) p(phi(s))` and its product rule (unit, wide, zero and extreme scores,
masked entries at +-1e15); nonfinite causal scores giving NaN rows; `softplus(s) p(softplus(s))` equal to the
degree-nine input power on the scan and strips and through two SoS models with equal parameters; dispatch,
admission, the planned recipes, refusals and provenance. The jaxprs of the existing transformed families' tile
functions (forward, dV, dQ and dK at degrees 1, 4, 8, 9 and 12) equal `main`'s, and `records receipts
experiments/transformer_pp` recomputed every run's receipt with this code: 276 runs, 245 `same` and 31 `same
rejection`, none different. The full-shape initial parameters of every planned recipe match its pinned partner's
count and SHA256 (the [study README](../../../experiments/transformer_pp/README.md#score-prefactors-swish-and-narrow-keys)
lists the runs by adapter).

**GPU** (`qualify_score_prefactor.sbatch`, one GPU per job on the standard partition, both weight precisions, the
CUDA 12 environment on H100 and CUDA 13 on B200):

| GPU | Job | Suites | Source | Result | JUnit seconds |
| --- | --- | --- | --- | --- | --- |
| H100 PCIe (`mastiff`) | 8214 | prefactor1: every family against FP64 at degrees 1, 4, 8 (unit and wide scores), Q/K widths 4 and 16 | `6aefb8f8` | 84 passed | 239 |
| H100 PCIe | 8218 | prefactor2: production context against FP64 and the strips, input-power kernel, prefix, overflow, extremes, repeatability | `6aefb8f8` | 56 passed | 551 |
| H100 PCIe | 8215 | transforms1: softplus and abs at degrees 1-12 | `6aefb8f8` | 40 passed | 77 |
| H100 PCIe | 8219 | transforms2: widths 4-128, degree nine at widths 4, 8, 16, production context, failures, repeatability | `6aefb8f8` | 58 passed | 341 |
| H100 PCIe | 8233 | sos1: softplus SoS, `f(s) h_t(g(s))` at the production context, the input-power route, widths 4, 8, 16 | `4159732c` | 41 passed | 166 |
| H100 PCIe | 8243 | sos2: zero constants and static coefficients with a learnable `c_0`, with and without the softplus prefactor | `78b5c25e` | 13 passed | 44 |
| H100 PCIe | 8217 | pointwise (sigmoid, softplus, swish), strips composition, query and Taylor kernels at width 4; fused | `6aefb8f8` | 65 passed; 3 fused cases superseded | 231 |
| H100 PCIe | 8244 | fused: exp at Q/K widths 4, 8, 16 and 64, 2,048 tokens | `78b5c25e` | 4 passed | 12 |
| B300 SXM6 (`akita`) | 8252 | fused (long partition) | `78b5c25e` | 4 passed | 55 |
| B200 (`shiba`) | 8200 | the whole prefactor suite, before its split | `7f03ebda` | 138 passed; 2 exp extreme-score cases superseded (below) | 1,663 |
| B200 | 8245 | prefactor1 | `78b5c25e` | 84 passed | 685 |
| B200 | 8246 | prefactor2 | `78b5c25e` | 56 passed (requeued once: a debug job preempted the first attempt) | 923 |
| B200 | 8247 | transforms1 | `78b5c25e` | 40 passed (requeued once: a debug job preempted the first attempt) | 183 |
| B200 | 8248 | transforms2 | `78b5c25e` | 58 passed | 645 |
| B200 | 8249 | sos1 | `78b5c25e` | 41 passed | 428 |
| B200 | 8250 | sos2 | `78b5c25e` | 13 passed | 86 |
| B200 | 8251 | pointwise (sigmoid, softplus, swish), strips composition, query and Taylor kernels at width 4 | `78b5c25e` | 64 passed | 390 |

Bounds are the query kernels': relative L2 0.006 (output), 0.025 (dQ, dK), 0.015 (dV), 0.01 per coefficient
order and 0.02 for SoS generator gradients, each with its largest element error within three times the bound
times the largest reference element (the fused exponential: 0.006, 0.025, 0.025, 0.015). Records:
[H100](runs/score-prefactor-mosaic-h100/record.json), [B200](runs/score-prefactor-mosaic-b200/record.json),
[B300 fused](runs/score-prefactor-fused-b300/record.json).

Three test corrections, none of them in kernel code:

- The first prefactor runs (H100 job 8196 and B200 job 8200, `7f03ebda`) each passed 138 of 140: the two exp
  families' extreme-score case reached scores beyond 50, where the order-eight coefficient gradient
  `sum_i dP e^s s^8` is about 1e40 and overflows FP32 in any implementation (the FP64 reference agrees). Exp's wide
  rows now have standard deviation 6, scores near 30, the largest the runs record (`6aefb8f8`).
- H100 job 8216 (`6aefb8f8`) failed one coordinate of the narrow-key SoS case: at width 8 the amplitude gradient of
  square factor 4, a sum over 2,048 tokens whose terms cancel to 1/1111 of their magnitude, erred 2.8% (the kernels'
  transcendental error of about 1e-5 in its terms, amplified by the cancellation; the FP32 strips, with exact
  transcendentals, match FP64 there within 5e-5). The narrow-key case checks each generator gradient as one array;
  the reference-width cases keep the per-coordinate checks (`4159732c`).
- In job 8217 the fused exponential matched the blocked backend at 2,048 tokens for every narrow width, then XLA
  failed to compile the dense FP64 reference at a 257-token, two-head prefix (a layout RET_CHECK). The case now uses
  the width-64 case's 129-token, one-head prefix (`78b5c25e`), and passed in job 8244.

The launch source keeps these kernels. Outside them it admits a prefactor only at the degrees above, divides a head
output whose FP32 mean square overflows by its largest magnitude before the per-head RMSNorm
(`attention/output_scaling.py`; a finite mean square keeps its arithmetic bit for bit), and names the prefactor in
the diagnostics. There the CPU modules that the 17 runs' acceptance selects passed in both CPU stages (1,431 tests
in 36 modules), as did 39 further modules covering admission, diagnostics, the study's other adapters and the
records (2,426 tests), and `records receipts` over the three studies found all 573 receipts unchanged (516 `same`,
57 `same rejection`).

This qualifies the kernels, not a training run: each production allocation still runs its adapter's CPU and GPU
suites, the initialization check, the 20-update full-shape pilot and a fresh-process resume.

Reproduce from a checkout at the recorded source with its locked environments (B200: `--nodelist=shiba` without
the fused suite; B300: `--nodelist=akita` with the fused suite alone):

```bash
for suite in prefactor1 prefactor2 transforms1 transforms2 sos1 sos2 pointwise,fused; do
  sbatch --exclude=shiba,akita,dogo,collie,samoyed,doodle,terrier \
    --output=scaffolds/transformer/validation/logs/score-prefactor-%j.log \
    scaffolds/transformer/validation/qualify_score_prefactor.sbatch 78b5c25e29708a6a1582182b38e1ba1c6568b672 "$suite"
done
```

### Fixed sigmoid and softplus attention (2026-11-18)

The fixed degree-one `[0, 1]` route evaluates sigmoid or softplus inside the shared
Mosaic transformed kernels. It adds no parameters. Both full reference-shape
initializations and actual optimizer decay masks match the pinned exp control
exactly: 198,453,586 parameters. Acceptance-selector and pinned-recipe regression
checks passed 191 CPU tests. Thirteen focused transform checks passed, covering
independent NumPy values and all gradients, finite extreme tails, numerical
failure behavior, blocked scan/strips agreement, and forward/backward lowering
for H100 and B200 at the production context.

Both allocations passed 619 CPU and 10 distributed CPU checks. All 14 dedicated
Mosaic kernel GPU checks passed on B200, covering production-context BF16 forward
and all Q/K/V/coefficient gradients against independent CPU references, causality,
tails, nonfinite scores, and exact kernel repeatability. The two full-model
repeatability checks failed before the training pilot: their 256-token fixture
draws from only 64 embedding indices, and BF16 atomic accumulation makes the
embedding gradient vary by update order. Loss, logits, and all other gradients
repeat exactly.

Read-only diagnostic job `6520` reproduced this for sigmoid, softplus, and a
blocked-exp control with one compiled callable and with a fresh compilation.
Using distinct token IDs made every output and parameter gradient bitwise equal
over three repeats and a fresh compilation, for all three functions. Every
per-array relative-L2 difference against the CPU blocked model stayed below 1.2%.
The [original measurements](pointwise_repeatability.json) and
[diagnostic program](pointwise_repeatability_probe.py) record the behavior
before the input embedding precision fix.

The shared model now gathers FP32 embedding rows and casts the gathered
activations to the compute dtype. Its repeated-ID regression gives two token
occurrences the gradients 1 and 1/1024 in the same channel. Their FP32 sum
is 1.0009765625. The previous BF16 accumulation returned 1. The new CPU check
recovers the exact sum and checks forward activations against the former lookup
for both BF16 and FP32 training.

On B200, job `6665` passed all 20 focused GPU checks. These include both
embedding compute dtypes, fourteen pointwise kernel checks, and four full-model
checks across sigmoid and softplus with distinct and repeated token IDs. All
four full-model checks passed the original GPU-to-CPU tolerances and produced
bitwise-identical outputs and gradients on a second call. Job `6674` passed all
20 checks again with an additional replay through a newly jitted model function.

Every production allocation requires the full suite, initialization, a
20-update pilot, and fresh-process checkpoint resume. The pointwise acceptance
adapter admits H100 and B200.

### Softplus polynomials with structural zero constants (2026-11-18)

The compact parameterization removes c0 for learned direct query/static
polynomials and U0,V0 before query-SoS square expansion. The feature suite passed
24 CPU checks, including independent FP64 outputs and all VJPs, exact surviving
initialization, parameter counts, AdamW decay ownership, three optimizer updates,
diagnostics, negative tails, and legacy checkpoint configuration compatibility.
The pinned-recipe adapter passed 66 checks and the records suite passed 94.
Existing configuration, score-transform, softplus-SoS, and coefficient-selection
regressions passed all 238 CPU cases (four GPU cases deselected).

Nine new GPU cases cover all three families at 512 and 2,048 tokens, including
wide scores and native Mosaic dispatch. GPU numerical qualification is pending.
Each production allocation must pass these checks, the existing Mosaic query
suite, the full-shape pilot, and fresh-process resume before training. The
transformed kernels admit H100 and B200; B300 is not qualified here.

### Unit-L2 coefficient inputs (2026-11-14)

The coefficient-only normalizer passed **21 CPU checks in 64.58 seconds**, alongside
39 pinned-recipe acceptance checks, 224 existing configuration/query regression
checks, and four parameter-policy checks. Independent analytic/FP64 references
cover values and VJPs, while full-model captures check unchanged Q/K/V and parameter
identity. The gradient comparison checks each W/b/a head and generator coordinate
separately, preserves exact zero references, and detects missing tiny gradients.
The full reference-shape initial parameter hash matches the controls exactly.

Focused job `760` on one B300 in `akita` passed **all four composed native Mosaic
GPU cases in 86.59 seconds** at `9e9580d2`, using the worktree's own locked CUDA 13
environment. Direct and SoS degree-eight generators at key width 64 were checked
at both BF16 and BF16X3 precision: forward outputs and all q/k/v/x/W/b/a gradients
matched independent FP64 references under the unchanged error bounds.
[coefficient_input_l2.json](coefficient_input_l2.json) records the source, frozen
execution hash, allocation, runtime, bounds, and retained log/XML checksums.

Jobs `762` and `763` each passed 537 CPU tests, 10 distributed CPU tests,
70 GPU tests, exact initial-parameter matching, a full-shape 20-update pilot,
and a fresh-process resume from update 20 to 21 before production. Checkpoint
state, next-batch identity, RNG state, and data position matched across resume.
Both selected degree-eight recipes subsequently completed all 7,570 updates on
one B300 each. Their run records retain the acceptance summaries and audits;
this qualification covers these recipes and runtime, not other degrees.

### Degree-twelve unit-L2 coefficient inputs (2026-11-14)

The degree-twelve extension passed 30 coefficient-input CPU checks and 80 pinned
recipe checks. Both complete initial parameter fingerprints match their respective
controls: 200,905,126 parameters, including unchanged softmax-backbone arrays.
Focused job `1493` on one B300 in `akita`, using the locked CUDA 13 environment,
passed all eight native Mosaic composition checks in 92.82 seconds. These cover
degrees eight and twelve, direct and SoS generators, and BF16/BF16X3, comparing
outputs and all q/k/v/x/W/b/a gradients with independent FP64 references under the
existing bounds. [coefficient_input_l2_m12.json](coefficient_input_l2_m12.json)
records the source and evidence. Jobs `1512` and `1513` passed their full
acceptance suites: 587 CPU checks for direct query and 564 for SoS, plus 10
distributed CPU checks and 72 GPU checks each. Both passed exact control
initialization matching, the full-shape 20-update pilot, and fresh-process
optimizer/data resume from update 20 to 21. The state, next-batch identity, RNG,
and data-position audits matched. Both selected degree-twelve recipes completed
all 7,570 production updates on one B300 each; their run records retain the
acceptance summaries and audits.

### Plain Mosaic query kernels for a checkpoint diagnostic on B200 (2026-11-13)

Job `244590` on one B200 in `shiba` ran six existing query-kernel checks under the locked CUDA 13
environment at `2d7338c0`: **6 passed in 145.01 seconds**. Degrees 8 and 12 at Q/K width 64,
512 tokens and two heads matched independent NumPy FP64 and Pallas-Triton forward outputs and
Q/K/V/coefficient VJPs; degree 12 also matched Triton at 2,048 tokens and four heads. Every case
ran with both BF16 and BF16X3 weight products. The checks preceded the bounded checkpoint
forward diagnostic; they do not qualify a full training run or optimizer continuation.
[query_activation_probe_b200.json](query_activation_probe_b200.json) records the exact tests,
bounds, source and lock, allocation, runtime, and retained-log checksum.

The same six checks passed again in **108.63 seconds** for job `291` on B200 before
replaying the retained SoS degree-eight and degree-twelve checkpoints. A two-layer
CPU check with nonzero amplitudes and nonuniform RMSNorm gains confirmed that the
capture leaves both direct and SoS forward losses exactly unchanged; generator
reconstruction errors were below `6e-8`. The [SoS probe qualification](sos_activation_probe_b200.json)
records this allocation and its retained log. This remains forward-diagnostic qualification.
### Degree-twelve softplus–SoS and degree-thirteen prefactor (2026-11-15)

The shared Mosaic API admits total degree thirteen only for softplus, d_K=d_V=64,
and contexts 512 and 2,048. Ordinary, absolute-score and gated polynomial admission
retains the existing degree limit. The tiled forward and backward arithmetic is
unchanged. The existing shared-memory layouts lower for Hopper and Blackwell.

CPU qualification passed 126 algebra, differentiation, parameter-policy, miniature
training, admission and lowering checks in `test_softplus_sos.py` and
`test_score_transform_lowering.py`. The composed GPU suite now covers cores eight
and twelve, both input powers, both weight precisions, all Q/K/V and generator
x/W/a/b gradients against an independent FP64 direct-factor reference, full
2,048-token dispatch, causal prefixes, negative tails and numerical failures.
GPU numerical qualification is pending; each new run's full acceptance must pass
those checks, initialization fingerprints, full-shape throughput and fresh-process
resume before production. B300 remains unqualified for these transformed kernels.

### Softplus + query SoS on Mosaic B200 kernels (2026-11-13)

The transformed Mosaic kernels and their query sum-of-squares composition passed
GPU qualification on one **NVIDIA B200** (`shiba`, Slurm job **244553**, driver
580.196.07, GPU UUID `GPU-33b41ad0-1a5a-d04a-6f63-cc26a25f6374`) at frozen source
`52af77095aa516a0e150b16c4c902e3cc7ca82e9`. The compact
[qualification record](runs/softplus-sos-mosaic-b200/record.json) pins the source,
dependency lock, test modules, JUnit XML and retained log. The job used the locked
CUDA 13 environment: JAX/JAXlib 0.10.0, CUDA toolkit 13.2.86, cuBLAS 13.2.2.2 and
cuDNN 9.22.0.52, with the Kennel runtime's compute-stream collectives and
preallocation disabled.

**102 tests passed**, with no failures, errors or skips, in 835.426 seconds of
JUnit runtime; Slurm completed with exit `0:0`. The selection includes all 88
cases of `tests/test_mosaic_score_transform.py` and all 14 cases of
`tests/test_mosaic_softplus_sos.py`, at both `bf16` and `bf16x3` weight precision.
The separate 23-module CPU preflight passed 670 tests (45 deselected, no failures
or skips). Host initialization checks for both full-size recipes reproduced the
control's exact 200,150,806 parameter arrays; the prefactor changes no learned
parameter. Their compact results and retained-log hashes are in the same record.
The general transformed-kernel cases cover softplus and absolute-value inputs,
supported degrees and widths, production context, causal masking and explicit
numerical failures. The SoS cases cover

\[
z=\operatorname{softplus}(s),\qquad
w=A(z)^2+B(z)^2\quad\text{or}\quad w=z\bigl(A(z)^2+B(z)^2\bigr),
\]

with a degree-eight SoS core and total degrees eight and nine. Production
`attention_core` dispatch reaches Mosaic at 2,048 tokens and `d_K=d_V=64`.
An independent NumPy FP64 reference evaluates the two squares directly, without
expanded coefficients, and differentiates Q/K/V plus the query generator's input,
projection, amplitude and bias. This tests the prefactor's score-gradient product
rule as well as the coefficient path. Additional cases check the finite
negative-score baseline versus the vanishing tail, exact zero factors, causal
prefix outputs and generator gradients, and failure on an overflowing negative
causal score even when the constant coefficient is structurally zero.

The asserted relative-L2 bounds are 0.006 for output, 0.025 for dQ/dK, 0.015 for
dV, 0.01 for every expanded coefficient order in the general kernel tests, and
0.02 for the SoS generator gradients (every factor coordinate of W/a/b checked
separately). Each also bounds the largest element error by three times its
relative tolerance times the largest reference element; exact reference zeros
must remain exact. These are acceptance bounds, not reported maximum errors.

This records **B200 kernel correctness**, not a full-model training pilot or
untested B300 execution. The tests admit the Blackwell kernels in-process for
qualification. Each production run must still pass its own initialization,
optimizer pilot, checkpoint save and fresh-process resume checks on its granted
allocation.

Reproduce from a checkout with the matching frozen source and its locked CUDA 13
environment; validation logs stay ignored:

```bash
sbatch --partition=debug --nodelist=shiba --gres=gpu:1 \
  --output=scaffolds/transformer/validation/logs/softplus-sos-%j.log \
  scaffolds/transformer/validation/qualify_softplus_sos.sbatch \
  52af77095aa516a0e150b16c4c902e3cc7ca82e9
```

### Score transforms on the Mosaic query-polynomial kernels (2026-11-12)

Two Mosaic kernel families, `query_softplus` and `query_abs` (`backends/mosaic/common.py`), evaluate the
diagonal tanh query polynomial at `z = phi(s)` inside the query family's own H100 and B300 kernels
([README](../README.md#score-transforms)); `mosaic.api.transformed_query_polynomial_attention` is their entry
point and an explicit `attention_backend: mosaic` selects them (`ModelConfig.mosaic_score_transform_problem`).
Each tile computes `z` and, in the gradient passes, `phi'(s)` (`common.transformed_score`): softplus as
`(s + max(-s, 0)) + ln(1 + y)`, `y = 2^(-abs(s) log2(e))` (`ex2.approx`), `ln(1 + y)` from `lg2.approx` or,
below `y = 2^-6`, its cubic series, and `sigmoid(s) = 2^((min(s, 0) - ln(1 + y)) log2(e))`; abs as `abs(s)` and
`sign(s)` (0 at `s = 0`), with no transcendental and the query family's compiler settings. `s + max(-s, 0)` is
`max(s, 0)` for every finite score and NaN at `s = -inf`, where the limit 0 would give the pair the finite
weight `c_0`. Then `W = p(z)` (forward and dV), `dS = p'(z) (dP phi'(s))` (dQ and dK) and the dQ pass's
`dc_r = sum_i dP z^r`. Vectors, shared memory, rings, BF16 weight products, masking (masked entries enter at
`s = 0` and leave as zero) and failure rows are the query family's: a causal score that overflows to `-inf`
makes its row NaN, as in the query kernels and the blocked reference. Backend selection admits these kernels on
H100 (sm90) and the separately qualified B200 path (sm100, above), while the entry point refuses B300 (sm103)
before tracing its kernels, which also covers a model restored without backend selection. The
12-layer width-640 transform runs keep the strips (next section); the reference-shape softplus run
`ra3b54e2489f8` trains on these kernels.

**CPU tests** (`e185c0c5`, `JAX_PLATFORMS=cpu`): `tests/test_score_transform_lowering.py` 65 passed: the
families' counts and flags are the query family's (approximate math for softplus alone) and so is their support
predicate, for degrees 0–14, every width and 256, 2,048 and 4,096 tokens; every H100 kernel lowers for cuda on
the host in exactly the shared memory `hopper_smem_bytes` counts (the query family's: 106,608, 77,904 and 70,704
bytes for forward, dQ and dK/dV at degree eight) and fails one byte below, at degrees 4, 8 and 12; B300's
forward, dQ, dK and dV kernels, and both architectures' entry points with every gradient, lower at 2,048 tokens
(degree eight at Q/K widths 8, 64 and 128, degree twelve at 64); invalid operands are refused; the tile
mathematics of `common` on a 37-token causal tile and its transpose, in FP64 against explicit `phi(s)^r` and
`phi'(s)` (abs to 1e-12 of each array's largest value, softplus to 2e-6, the series' bound), at degrees 1, 4, 8
and 12 with unit and wide (`N(0, 64)`) scores, exact zeros and extremes from −120 to 40, masked entries at
±1e15; `phi` and `phi'` at special values in FP32 and FP64 (1/2 and 0 at `s = 0`, the limit at +inf, zero once
`e^s` underflows, NaN for NaN, and softplus's input NaN at `-inf`); a causal score of `-inf`, `+inf` or NaN makes
its row NaN in the tile functions of the query family and both transforms, with the pair's score gradient and
its row's coefficient sums of orders >= 1 nonfinite and every other row finite, as the blocked scan and strips
make the row of a `q.k` that overflows to ±inf; backend selection resolves a transformed recipe on `mosaic` on
sm90 and refuses it on sm10x, where the untransformed recipe still resolves to `mosaic` and the strips run the
transform, and the entry point refuses B300's kernels before tracing them and runs H100's; the identity's programs on Mosaic unchanged through `attention_core` and
`query_polynomial_attention`, and the transform reaching the transformed kernels from `attention_core` and from
every layer of a two-layer model. `tests/test_score_transform.py` 81 passed (Mosaic recipes accepted where the
query kernels run and refused with the reason elsewhere, backend resolution, allocation), `tests/test_acceptance.py`
164 and the adapter's 187. The whole CPU suite: every scaffold, study and result test but the two-device stage
and state rank's, 5,053 tests, 5,052 passed and one skipped (`test_regeneration_is_byte_identical`, whose
archives live in the primary checkout); the two-device stage 25 passed and five skipped, which need four host
devices and passed with four; state rank's suite in its own environment 225 passed and three skipped, which need
CUDA.

**Unchanged programs.** Against `c4034788` (this branch before the change), on the host: the jaxprs of every
existing family's H100 and B300 kernels (784: ten families, degrees 1, 3, 8 and 12, Q/K widths 16, 64 and 128
for Taylor and the query family, both precisions; B300's 128 score-bias kernels refuse alike); the
location-free Mosaic GPU modules lowered for cuda of the forward and VJP of every public entry point on both
architectures (248); the jaxprs of those entry points (236), of `attention_core` on backend mosaic for six
mechanisms with diagnostics (72) and of three recipes' loss-and-gradient programs on Mosaic (3); the blocked
scan's and strips' 276 programs, 40 small models (programs, losses, gradients, parameters, identities,
diagnostics), and validation, identity, parameter count and test selection of all 462 run recipes (only the two
12-layer width-640 transform recipes' refusal text for a hypothetical `fused` resolution differs); and the
seed-42 initialization fingerprints of `r95a5691d293f`, `r25fffc16c2f3`, `r227b0ed7c605` and `r8dc6e147601c`.
All identical. Against `e2d5aac4`, the set the GPU jobs below ran, `e185c0c5` changes only `query_softplus`'s
programs: the jaxprs of every family's H100 and B300 kernels at degrees 1 and 8, Q/K width 64 and both precisions
(336, 64 of them B300's score-bias refusals alike), and the lowered Mosaic modules of the query, abs and softplus
entry points' forward and VJP at degree eight on both architectures (B300 admitted for the comparison), are
identical but for softplus's 28 kernels and 4 modules.

**GPU qualification** (jobs 244010 and 244157 on `mastiff`, both on one H100 PCIe, GPU-b1e23521 at bus
61:00.0, driver 560.35.05, 350 W cap; commits `fe0c2a97` and `261dfa47`, one frozen execution set `f09ea1f1`,
under the launcher's runtime: the CUDA 12 environment and the locked toolkit's ptxas 12.9.86). The compact
record is [runs/score-transform-kernels/record.json](runs/score-transform-kernels/record.json), with the SHA256
of each JUnit, log and report in the ignored `logs/score-transform-kernels/<job>/`. The current frozen execution
set, `c90aa961` at `e185c0c5`, differs from `f09ea1f1` in softplus's input (the same for every finite score, NaN
at `-inf`), the H100-only admission (backend selection and the entry point), the tests and the scaffold README,
and job 244414's acceptance ran it on one H100 PCIe of `mastiff` before training `ra3b54e2489f8` (launch commit
`a54846c5`): its GPU stage passed 118 cases, none skipped or failed, in 485 s (`mosaic_score_transform` 50, among
them an overflowing score in three rows; `mosaic_query` 44; `mosaic_repeatability` 3;
`input_dependent_coefficients` 5; `query_tanh_polynomial` 16).

- Acceptance's GPU stage for `ra3b54e2489f8` (`-m gpu --mosaic-precision bf16` over its modules in acceptance's
  order): 116 passed, none skipped or failed, in 431 s: `mosaic_query` 44, `mosaic_repeatability` 3,
  `mosaic_score_transform` 48, `input_dependent_coefficients` 5, `query_tanh_polynomial` 16
  (`coefficient_scale` has no GPU case). The FP32-precision (`bf16x3`) cases of
  `tests/test_mosaic_score_transform.py`, which BF16 deselects, passed separately: 48.
- Job 244157 ran `tests/test_mosaic_score_transform.py` again at both precisions (48 and 48 passed) and the
  query family's Mosaic GPU modules at BF16, the
  [score-bias kernels'](#score-bias-mosaic-kernels-on-h100-2026-11-11) set: 187 passed, none skipped or failed
  (`mosaic_query` 44, `mosaic_repeatability` 3, `per_order_decay` 24, `mosaic_score_bias` 57, `score_bias` 4,
  `endpoint_decay` 30, `input_dependent_coefficients` 5, `query_tanh_polynomial` 16, `coefficient_forms` 3,
  `query_distance_decay` 1).
- Accuracy against the module's NumPy FP64 all-pairs reference (explicit `phi(s)^r`, `phi'` the logistic function
  or `sign(s)`, every gradient, the coefficient gradient order by order), largest relative L2 over degrees 1, 2,
  4, 8 and 12 at 512 tokens with unit and wide (`N(0, 9)`) scores and signed coefficients, and the production
  context (2,048 tokens, four heads, degree eight), where the production dispatch is also compared with the
  causal strips:

  | Quantity | Bound | BF16 vs FP64 | BF16 vs the strips (production) | `bf16x3` vs FP64 |
  | --- | ---: | ---: | ---: | ---: |
  | Output | .006 | 2.53e-3 | 2.59e-3 | 1.69e-3 |
  | dQ | .025 | 2.70e-3 | 2.66e-3 | 1.73e-3 |
  | dK | .025 | 2.43e-3 | 2.73e-3 | 1.71e-3 |
  | dV | .015 | 2.61e-3 | 2.59e-3 | 1.83e-3 |
  | `dc`, largest order | .01 | 1.1e-6 | 4.1e-7 | 1.1e-6 |

  The strips measure 1.55–1.70e-3 against FP64 there, their BF16 output floor; with BF16 weight products the
  kernels' errors match the query family's (2.41e-3 for the score-bias families below). Every Q/K width (8 to
  128) at degree eight, the largest degree at widths 64 and 128 with 2,048 tokens, scores of standard deviation
  12 and rows of exactly zero scores, future-token invariance (bitwise), NaN rows without poisoning earlier ones,
  five repeated calls (bitwise identical) and dispatch through `attention_core` with the transform's diagnostics
  passed for both transforms.
- Speed and registers (`benchmarks/benchmark_score_transform.py`; the production per-GPU micro-batch of 16 ×
  2,048 tokens, 14 heads, degree eight, BF16 weight products; medians of 20; "× 240" is an update's 15 layers ×
  16 micro-batches):

  | Kernels | Forward ms | dQ ms | dK/dV ms | × 240 s | dK/dV spills (bytes) |
  | --- | ---: | ---: | ---: | ---: | --- |
  | query (the control's) | 0.83 | 1.70 | 1.40 | 0.94 | 140/140 |
  | softplus | 1.17 | 2.18 | 1.79 | 1.23 | 336/336 |
  | abs | 0.82 | 1.76 | 1.45 | 0.97 | 216/216 |

  The forward and dQ kernels spill nothing; every kernel launches with 168 registers (the compute warpgroups
  raise theirs to 232) at PTX ISA 8.8. Through the entry points, forward plus every VJP takes 3.04 ms (query),
  3.84 ms (softplus) and 3.16 ms (abs). Job 244157 measured 0.93 s (query), 1.23 s (softplus) and 1.01 s (abs)
  × 240 on the same GPU.
- Full-shape pilot (`benchmarks/qualify_training.py pilot --no-references --continue-updates 1`: acceptance's pilot
  worker on a copy of `ra3b54e2489f8`, the production batch of 16 micro-batches of 16 × 2,048 tokens, 20
  updates, validation with the transform diagnostics and a checkpoint at update 20; frozen-execution check
  passed, set `f09ea1f1`): it resolved `mosaic` and `pallas_mosaic_gpu_query_polynomial_softplus_scores`;
  every update's loss and numerics were finite (10.501 at update 1, 9.026 at update 20), the validation NLL at
  update 20 was 8.994, and the attention diagnostics with the transform statistics were valid. 3.714 s per
  update (median of updates 2–20; 3.64–3.74), first update 40.2 s; acceptance projects 8.26 hours for 7,570
  updates (training 7.81, evaluation 0.21, checkpoints 0.17); allocator peak 30.3 GiB of 59.4. Acceptance's
  resume (update 21 in a fresh process from the pilot's checkpoint) passed its audit: loss 8.997, first update
  41.3 s, peak 29.9 GiB.
- Same-GPU pilots (job 244157, `--continue-updates 0`, each copy in a temporary directory outside the
  repository): the run again took 3.715 s per update (3.63–3.74; 8.27 hours; peak 30.3 GiB) and its control
  `r25fffc16c2f3` (acceptance's default adapter) 3.411 s (3.35–3.45; 7.63 hours; peak 30.3 GiB), both with
  finite losses and valid diagnostics. Softplus adds 0.30 s per update (8.9%), the kernels' difference × 240
  (0.30 s). The 350 W power cap was active in 96% of both pilots' busy one-second samples (342–344 W, median SM
  clock 1,425–1,463 MHz of 1,755, 42–65 °C). The control's pilot on another of `mastiff`'s H100 PCIe GPUs took
  3.52 s per update ([score-bias Mosaic kernels](#score-bias-mosaic-kernels-on-h100-2026-11-11)), 3% more than
  on this one.
- H100 SXM estimate (no SXM was free): the control's recipe trains at 2.32 s per update on an H100 SXM
  (`ra15cbb16beb7`, the same update at 2.5 times the budget; median over its validation windows of training
  throughput) and its even-order variant `r698f9a157a94` at 2.28 s on `dogo`, 1.47–1.50 times faster than the
  control's pilot here. Scaled by the softplus/control ratio the run takes 2.49–2.53 s per update: 5.7–5.8 hours
  with the PCIe pilot's evaluation and checkpoint time.
- Time limit: the H100 PCIe, the slower GPU type the run may receive, sets it: its projection (8.26–8.27 hours)
  plus two hours, rounded up: 11 hours, within `standard`'s 24 (an SXM alone would need 8); acceptance refuses a
  granted GPU whose own projection exceeds 9 hours.
- No B200, B300 or H100 SXM has run these kernels. B300's kernels run the same tile functions and lower on the
  host (above); backend selection, the entry point and the run's adapter admit H100 alone.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q -c scaffolds/transformer/pyproject.toml \
  --import-mode=importlib -m "not gpu" scaffolds/transformer/tests/test_score_transform_lowering.py \
  scaffolds/transformer/tests/test_score_transform.py experiments/transformer_pp/tests/test_score_transform_acceptance.py
# Inside one H100 allocation, from the worktree root, under the launcher's runtime (as for the score biases);
# SELECTION is acceptance.test_paths(...) for the run with resolved_backend="mosaic", cpu=False.
JAX_PLATFORMS=cuda,cpu "$PY" -m pytest -q -s -rA -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  -m gpu --mosaic-precision bf16 $SELECTION
JAX_PLATFORMS=cuda,cpu "$PY" -m pytest -q -s -rA -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  -m gpu --mosaic-precision bf16x3 scaffolds/transformer/tests/test_mosaic_score_transform.py
B=scaffolds.transformer.benchmarks.benchmark_score_transform
JAX_PLATFORMS=cuda "$PY" -m $B kernels --output kernels.json  # also attention; registers in its own process
QT=scaffolds.transformer.benchmarks.qualify_training
Q=(experiments/transformer_pp/runs/ra3b54e2489f8/config.yaml --directory <dir> --experiment transformer_pp
   --adapter experiments.transformer_pp.score_transform_acceptance --gpu-type H100)
JAX_PLATFORMS=cuda "$PY" -m $QT pilot "${Q[@]}" --no-references --continue-updates 1 --limit-hours 8 10 12 14 16 20 24
JAX_PLATFORMS=cuda "$PY" -m $QT resume "${Q[@]}"
# Job 244157: the control's pilot (acceptance's default adapter), and the query family's modules at BF16.
JAX_PLATFORMS=cuda "$PY" -m $QT pilot experiments/transformer_pp/runs/r25fffc16c2f3/config.yaml --directory <dir> \
  --experiment transformer_pp --gpu-type H100 --no-references --continue-updates 0
T=scaffolds/transformer/tests
JAX_PLATFORMS=cuda,cpu "$PY" -m pytest -q -s -rA -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  -m gpu --mosaic-precision bf16 $T/test_mosaic_query.py $T/test_mosaic_repeatability.py $T/test_per_order_decay.py \
  $T/test_mosaic_score_bias.py $T/test_score_bias.py $T/test_endpoint_decay.py \
  $T/test_input_dependent_coefficients.py $T/test_query_tanh_polynomial.py $T/test_coefficient_forms.py \
  $T/test_query_distance_decay.py $T/test_coefficient_scale.py
```

### Score transforms on the causal strips (2026-11-12)

`model.attention_score_transform` evaluates the diagonal tanh query polynomial at `z = phi(s)` in place of
the score `s` (`softplus`: `log(1 + exp(s))`; `abs`), and `model.attention_coefficient_constant_initial`
starts the constant offset `b_0` at another value ([README](../README.md#score-transforms)). The blocked
backend (scan and causal strips) and the Mosaic transformed query-polynomial kernels (above) evaluate a
transform; every other kernel refuses it.

**CPU tests** (`593e7267`, `JAX_PLATFORMS=cpu`): `tests/test_score_transform.py` 75 passed: the scan's
code and the strips in FP64 (to 1e-11) and the public FP32 scan and strips (to 3e-5 of each array's largest
value; the coefficient gradient order by order) against an independent all-pairs FP64 oracle
(`logaddexp(0, s)` and `s sign(s)` for the transforms, explicit powers), at degrees 1, 4 and 8, for mixed
scores up to +-24 with exact zeros, all scores in [-30, -1/2], and all in [1/2, 30]; both transforms' values and
derivatives at special points (softplus's derivative 1/2 at zero and `sigmoid(s)` elsewhere, `abs`'s 0 at
zero, infinities and NaN); masked pairs scoring +-1e15 and future values of 1e20 leave earlier rows' outputs
and the earlier tokens' gradients bitwise unchanged; `attention_core` and a two-layer model on the strips
against the scan; the softplus start (`c_0 = 0`, `c_r = 1/r!`) equal to `e^s` minus its Taylor tail to 1e-14,
and to `e^s` for scores at or below -3; the refusals of every other recipe, backend and kernel, allocation,
the model's start against the control's, exact diagnostics against NumPy, and records.
`tests/test_blocked_strips.py`, with a softplus (degree 4) and an abs (degree 8) family added to its tables:
16 transform cases of 191. `tests/test_acceptance.py` (selection) and the records' index test pass, and the
adapter's 97. The whole CPU suite (every scaffold, study and result test but the two-device stage) passed: 4,942
tests, none failed.

**Unchanged programs.** Against `feab62a8`, with the base's `tests/test_blocked_strips.py` families for both
trees: the jaxprs of `blocked_dot_product_attention` and `strip_dot_product_attention` for its 31 argument
families (forward, VJP, and the scan's diagnostics, with FP32 and BF16 Q/K/V, in 8- and 128-query scan blocks
and 16- and 512-query strips; 620 programs) and of `attention_core` for its 21 core families (forward, VJP and
diagnostics through the scan and the strips; 252 programs) are identical. So are the loss-and-gradient
programs, losses, gradients, parameters, attention identities and diagnostics of 40 small models
(softmax, exp, Taylor, learned static coefficients, the query polynomial at degrees 4 and 8 with its forget-gate,
endpoint, per-order, query-key, sigmoid and dense variants, ALiBi, PaTH, score biases, and five run recipes at a
small shape: `r95a5691d293f`, `r25fffc16c2f3`, `r227b0ed7c605`, `r8dc6e147601c`, `r3e2fa2532c23`); validation, recipe (the two new fields at their defaults), attention identity, parameter
count, and acceptance test selection (CPU and GPU, on the blocked, fused, mosaic and cudnn backends) for all 465
earlier run recipes (408 valid); and the seed-42 initialization fingerprints of `r95a5691d293f`,
`r25fffc16c2f3`, `r227b0ed7c605` and `r8dc6e147601c`.

**GPU qualification.** `tests/test_score_transform_strips.py`, the transform runs' GPU stage beside
`tests/test_score_bias_strips.py`, runs the production dispatch (`attention_core` on the strips, counted) at
production precision: Q, K, V rounded to BF16 in the program, FP32 coefficients, a BF16 cotangent, degree
eight, 2,048 tokens in four strips, `d_K = d_V = 64`, two heads; for each transform (identity, the controls'
backbone, too) with unit scores and with `N(0, 9)` scores (a sixth below -3, extremes beyond +-12), against
host FP64. Bounds: output, dQ, dK, dV within 1.2x their BF16 rounding floors and no element beyond 2 BF16
roundoffs of the largest value; the coefficient gradient within 1e-4 relative L2 per order. Called on the CPU
host, all six cases measure 1.000x their floors, largest elements 0.43–0.82 roundoffs, and coefficient
gradients 4.3–4.7e-7. On two H100 PCIe GPUs of `mastiff` (driver 560.35.05, the CUDA 12 environment under the
launcher's runtime) the stage, this module and `tests/test_score_bias_strips.py`, passed all 16 cases, none
skipped:

| Job | GPU | Commit | Passed | pytest s | JUnit SHA256 |
| --- | --- | --- | ---: | ---: | --- |
| 243502 | GPU-3729cb75, bus C1:00.0 | `593e7267` | 16 | 287 | `7097f45d9b0e4b4fd774421a18ff847b141f37886ae8a8242ebb31c0af7569e2` |
| 243529 | GPU-750d4329, bus 01:00.0 | `1fa151b7` | 16 | 168 | `24aa205cdbc030feb2601bb864509f104a2c9c34bb7eab503fb7cab97806f26b` |

The two commits share one frozen execution set (`b72a61cd`). `3c0ae7e7` changes it only by a run-index column
(`scaffolds/records`, with its test) and the scaffold README; the runs' launcher (`kennel.sbatch`), acceptance,
training and retention never call the index. The transform cases measured the same on both GPUs:
1.007–1.011x their floors, largest elements 0.47–0.86 roundoffs, coefficient gradients 7.0–9.3e-7 per order. The
score-bias cases stayed at 1.007–1.009x. No B200, B300 or H100 SXM ran the stage.

**Full-shape pilots** (`benchmarks/qualify_training.py pilot --no-references`: acceptance's pilot worker on a copy
of each recipe, the production batch of 8 micro-batches of 16 × 2,048 tokens, 20 updates, validation with the
diagnostics and a checkpoint; frozen-execution check passed). In job 243529 the pilot set acceptance's resume
horizon (`--continue-updates 1`), and `resume` then ran acceptance's resume worker in a fresh process: update 21
from the pilot's checkpoint, with acceptance's audit of the restored state, data position and RNG. Same jobs and
GPUs:

| Run | Transform | Job | s/update (median, updates 2–20) | First update s | Projected hours, 7,551 updates | Allocator peak GiB | Validation NLL at 20 | Resume, update 21 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `r1d342ca6e58c` | softplus | 243502 | 3.240 | 65.5 | 7.29 | 22.8 | 9.366 | not run |
| `r1d342ca6e58c` | softplus | 243529 | 3.329 | 84.5 | 7.48 | 22.9 | 9.366 | audit passed, loss 9.358 |
| `r17a1290cb4ac` | abs | 243502 | 3.237 | 68.0 | 7.27 | 23.6 | 9.369 | not run |
| `r17a1290cb4ac` | abs | 243529 | 3.275 | 46.8 | 7.35 | 23.6 | 9.369 | audit passed, loss 9.361 |

Every pilot resolved the blocked backend and `blocked_causal_strips`. Every update's loss and numerics were finite,
and the attention diagnostics, with the transform statistics, were valid. The two GPUs differ by up to 2.7% per
update. The allocator's limit was 59.4 GiB at the default share. The compact record is
[runs/score-transform-strips/record.json](runs/score-transform-strips/record.json): margins per case, each
pilot's and resume's numbers, and the SHA256 of each report, JUnit and log in the ignored
`logs/score-transform-strips/`.

**Memory without running** (`benchmarks/compiled_memory.py`, the trainer's one-device update at micro-batch
16, compile-only from the host with `JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1`): softplus 22.2 GiB on an H100 and
21.6 GiB on a B200; abs 22.8 and 22.4 GiB; the H100 pilots' peaks were 2.6–2.9% and 3.5% above.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q -c scaffolds/transformer/pyproject.toml \
  --import-mode=importlib -m "not gpu" scaffolds/transformer/tests/test_score_transform.py \
  scaffolds/transformer/tests/test_blocked_strips.py experiments/transformer_pp/tests/test_score_transform_acceptance.py
# Inside one GPU allocation, from the worktree root, under the launcher's runtime (as for the score biases):
JAX_PLATFORMS=cuda,cpu "$PY" -m pytest -q -s -rA -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  -m gpu scaffolds/transformer/tests/test_score_transform_strips.py scaffolds/transformer/tests/test_score_bias_strips.py
QT=scaffolds.transformer.benchmarks.qualify_training
JAX_PLATFORMS=cuda "$PY" -m $QT pilot experiments/transformer_pp/runs/<run>/config.yaml --directory <dir> \
  --experiment transformer_pp --adapter experiments.transformer_pp.score_transform_acceptance --gpu-type H100 \
  --no-references --continue-updates 1 --limit-hours 6 8 9 10 12 16 20 24
JAX_PLATFORMS=cuda "$PY" -m $QT resume experiments/transformer_pp/runs/<run>/config.yaml --directory <dir> \
  --experiment transformer_pp --adapter experiments.transformer_pp.score_transform_acceptance --gpu-type H100
JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 scaffolds/transformer/.venv/bin/python -m scaffolds.transformer.benchmarks.compiled_memory \
  experiments/transformer_pp/runs/<run>/config.yaml --target <h100|b200>
```

### B200 and the score-bias causal strips (2026-11-11)

`shiba` has eight B200 GPUs (180 GB, compute capability 10.0, driver 580.196.07). Kennel B200
allocations run in the CUDA 13 environment `.venv-cuda13` (jax 0.10.0 with the CUDA 13 plugin,
CUDA 13.2, cuBLAS 13.2.2.2, cuDNN 9.22.0.52) with collectives on the compute stream
(`--xla_gpu_disable_async_collectives=ALLCOLLECTIVES`), like B300 ([CLUSTERS.md](../../../CLUSTERS.md)).
On B200 this record qualifies one computation: the blocked backend's causal strips
(`strip_dot_product_attention`, plain JAX) with per-degree score biases
([README](../README.md#per-degree-score-biases)). No custom attention kernel (Triton, Mosaic or
cuDNN) is qualified on B200; the Mosaic score-bias kernels refuse it before tracing. The same tests and
pilots also ran on one H100 PCIe. The six recipes are degree-eight tanh query polynomials at the
reference shape (15 layers, width 896, 7,570 updates of 256 × 2,048 tokens) on the strips: the recipes
of the RoPE query control `r25fffc16c2f3` ("RoPE") and of the per-order query–key gated control
`r227b0ed7c605` ("gated"), each with a `static`, `query` or `query_key` shift. The compact record is
[runs/score-bias-strips/](runs/score-bias-strips/), keyed by recipe (`rope-static` to
`gated-query_key`): `tests.json` (each GPU test run's source, runtime, devices and JUnit SHA256) and
`training.json` (every pilot and resume, the out-of-memory failures, the compile-only memory estimates,
and the SHA256 of every report it summarizes; the reports are in the ignored `logs/score-bias-strips/`).

**GPU tests.** `tests/test_score_bias_strips.py` (SHA256
`172bf704055ff6b94c1205eba84e96457e6579142596e94964e7a11358d29cc9`), acceptance's whole GPU stage for
blocked score-bias recipes, runs the production dispatch (`attention_core` on the blocked backend,
`blocked_implementation="strips"`, the strips' calls counted) at production precision: Q, K and V
rounded to BF16 inside the program, FP32 coefficients, gate inputs and shift operands, a BF16
cotangent, degree eight, 2,048 tokens in four 512-query strips, `d_K = d_V = 64`, one sequence and two
heads. Its oracle is an independent all-pairs FP64 reference (explicit `(s_ti + beta_r)^r`, JAX FP64
autodiff on the host CPU) from the same rounded inputs; every FP32 gradient is compared order by
order. Ten cases: each mode's shift operands formed by `score_bias.attention_operands` from nonzero
`b`, `a`, `u`, `v`, on both backbones (no gate, and per-order query–key endpoint gates with rates from
about 3e-4 to 2 per token); and on both backbones, row and pair shifts with integer Q and K, which make
5.4% (row) and 3.0% (pair) of the shifted scores exactly zero and 47% and 73–74% negative. All ten
passed, none skipped, in each run (`JAX_PLATFORMS=cuda,cpu`):

| Job | GPU | Source | Runtime | Seconds (slowest case) | JUnit SHA256 |
| --- | --- | --- | --- | ---: | --- |
| 242636 | B200, `shiba` | `338568f2` with the module as committed | `.venv-cuda13`, no XLA flags; pytest directly | 216.7 (35.0) | – (pytest log) |
| 242699 | B200, `shiba` (GPU-e34bd188) | `996c4e55` | acceptance's stage under the launcher's runtime: `.venv-cuda13`, collectives flag | 130.8 (36.5) | `ba6d0652faba35a45f912769f621d29ceb42e66d4f94b9020c75c6a017a891ef` |
| 242685 | H100 PCIe, `mastiff` (GPU-cae52cf1) | `996c4e55` | acceptance's stage under the launcher's runtime: `.venv` (CUDA 12) | 153.1 (48.5) | `52186cce117bfedc9012ddf90e33e2ea5e6fb0c827687727b1ded2ab763b9160` |

Jobs 242699 and 242685 ran the stage as acceptance runs it (selection, stage identity, pytest, and
`require_passing_tests`); all six recipes select this module alone. They ran at the default allocator
share. Margins in job 242636:

| Quantity | Bound | Largest, B200 | Largest, CPU host |
| --- | ---: | ---: | ---: |
| Output, dQ, dK, dV: relative L2 over the BF16 rounding floor | 1.2 | 1.009 | 1.000 |
| Output, dQ, dK, dV: largest element error over `2^-8` times the largest magnitude | 2.0 | 0.94 | 0.94 |
| FP32 gradients (`c`, gate inputs and rates, shift operands), relative L2 per order | 1e-4 | 5.3e-6 (`damplitude`) | 1.2e-5 (`dgate_log_rate`) |

The bounds are the earlier strips GPU tests' (1.2 times the floor, 1e-4 per order) plus the element
bound; the CPU column calls the same test functions on the host.

**Full-shape training** (`benchmarks/qualify_training.py`: acceptance's pilot and resume workers on a
copy of each recipe with its own directory and W&B off; the production batch, 16 micro-batches of 16 ×
2,048 tokens with 14 heads, 20 updates, validation with the score-bias diagnostics and a checkpoint at
update 20, then update 21 in a fresh process; projection by acceptance's `project_runtime` for 7,570
updates). One B200 (job 242699, GPU-e34bd188) and one H100 PCIe (job 242685, GPU-cae52cf1) under the
launcher's runtime, at sources `996c4e55` and `eb139cca` (which adds only
`benchmarks/compiled_memory.py`); every pilot passed the frozen-execution check. Allocator share 0.75
(the default) unless marked 0.95.

| Recipe | B200 s/update | B200 hours | B200 peak GiB | H100 PCIe s/update | H100 PCIe hours | H100 PCIe peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE static | 4.29 | 9.49 | 32.0 | 11.39 | 24.83 | 32.0 |
| RoPE query | 4.43 | 9.72 | 32.5 | 11.74 | 25.65 | 32.3 |
| RoPE query_key | 13.15 | 28.31 | 49.1 | 36.75 | 78.58 | 47.1 |
| gated static | 20.24 | 43.61 | 66.9 | 46.79 | 100.06 | 56.4 |
| gated query | 20.47 | 44.07 | 67.3 | 46.79 (0.95) | 100.07 | 61.8 |
| gated query_key | 36.82 (0.95) | 78.98 | 147.1 | – | – | – |

- Seconds per update are medians over updates 2–20; hours are acceptance's projection; peaks are the
  allocator's (through the validation, checkpoint and resume audit). Every pilot resolved the blocked
  backend and `blocked_causal_strips`; every update's loss and gradient norm, the validation, and the
  attention diagnostics with the score-bias statistics were finite and valid, and the validation loss at
  update 20 agreed between the GPUs to 2e-4. Acceptance's resume audit passed for every pilot: restored
  at update 20 in a fresh process, update 21 trained from the saved state, data position and RNG.
- The allocator's limit is 133.8 GiB on the B200 and 59.4 GiB on the H100 at 0.75, 169.4 and 75.2 GiB
  at 0.95. Two pilots stopped in their first update on the H100 at 0.75, out of memory: the gated query
  recipe asking for 54.17 GiB (it passed at 0.95) and the gated query_key recipe asking for 141.28 GiB.
  The gated query_key update needs 147 GiB: beyond the B200's limit at 0.75 (its compile-only
  estimate, 145.7 GiB; not run) and beyond any 80 GB H100. The gated static recipe fits the H100 at
  0.75 with 3 GiB to spare.
- The first update, which compiles, took 73–366 s, and a resume's first update compiles again (72–369
  s), well within the 20-minute stall limit.
- On the same H100 PCIe the strips take 2.0–3.8 times the Mosaic kernels' update time (next section).
- Not measured: B300 (`akita` was down) and H100 SXM (`dogo` was fully allocated). B300's 288 GB give
  about 201 GiB at 0.75 (the limit recorded on `akita`) and 254 GiB at 0.95, enough for every recipe.

**Memory without running** (`benchmarks/compiled_memory.py`: the trainer's one-device update compiled
for an H100 SXM or a B200 from the host, without autotuning; arguments, outputs and temporaries, GiB).
The measured peaks were 0.8–5.6% above these estimates.

| Recipe | B200 | H100 |
| --- | ---: | ---: |
| RoPE static | 31.5 | 31.5 |
| RoPE query | 31.8 | 31.8 |
| RoPE query_key | 46.4 | 46.4 |
| gated static | 66.4 | 55.4 |
| gated query | 66.5 | 58.7 |
| gated query_key | 145.7 | 144.1 |
| gated query_key, micro-batch 8 | 91.0 | 119.6 |
| gated query_key, micro-batch 4 | 68.1 | 65.0 |

The gated query_key recipe does not fit an H100 at its micro-batch of 16. At micro-batch 4 (64
accumulation steps) its estimate, 65.0 GiB, would likely fit the H100's 75.2 GiB at 0.95.

```bash
# Inside one GPU allocation, from the worktree root, under the launcher's runtime:
source scaffolds/transformer/launchers/kennel_runtime.sh
PY=scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python
CUDA_HOME="$(transformer_cuda_home "$PY")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" PYTHONPATH="$PWD"
JAX_PLATFORMS=cuda,cpu "$PY" -m pytest -q -s -rA -c scaffolds/transformer/pyproject.toml \
  --import-mode=importlib -m gpu scaffolds/transformer/tests/test_score_bias_strips.py
# Each recipe and the adapter as committed at those sources, with XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# exported where marked; a resume is its own process.
Q=(<recipe>.yaml --directory <dir> --experiment transformer_pp
   --adapter experiments.transformer_pp.score_bias_acceptance --gpu-type <B200|H100>
   --limit-hours 24 36 48 72 96 120 168)
QT=scaffolds.transformer.benchmarks.qualify_training
JAX_PLATFORMS=cuda "$PY" -m $QT pilot "${Q[@]}" --no-references --continue-updates 1
JAX_PLATFORMS=cuda "$PY" -m $QT resume "${Q[@]}"
# Compile-only memory of one update, from any host:
CM=scaffolds.transformer.benchmarks.compiled_memory
"$PY" -m $CM <recipe>.yaml --target b200 --output <report>.json
```

### Score-bias Mosaic kernels on H100 (2026-11-11)

`model.attention_score_bias` (`static`, `query`, `query_key`) shifts every order `r = 1..m` of the
tanh query polynomial by its own signed bias, `(s_ti + beta_r(t,i))^r`, without a gate or under the
per-order query–key endpoint gate ([README](../README.md#per-degree-score-biases)). Four H100
Mosaic families (`attention/backends/mosaic/score_bias.py`) form each shifted power by repeated
multiplication inside the causal tiles, never expanding it into powers of `s` and never dividing by
the shifted score. B300 has no score-bias kernel: every entry point raises `ValueError` before
tracing one. Every kernel below was assembled with the locked toolkit's ptxas (CUDA 12.9.86, PTX ISA
8.8). The compact record is [runs/score-bias-kernels/](runs/score-bias-kernels/): `tests.json`
(source, lock, toolkit and its probe, GPU, JUnit SHA256, measured errors), the benchmark's
`kernels.json` and `registers.json`, and `training.json` (the training qualification's summary, with
the SHA256 of every report it reads).

**Families** at `m = 8`, `d_K = 64` and 2,048 tokens. Per batch and head, the vectors are `c` (9),
the gate's query inputs (9), and `beta` or `p` (8); the key vectors are the gate's key inputs (9) and
`q` (8); the head scalars are the gate's rates (9) and `b` and `a` (8 each), padded to 32. The dQ
kernel sums `dc` (9), the two gate gradients (18), and one (row) or three (pair) shift gradients of
8; the dK kernel sums the gate's (9) and `q`'s (8). On a CPU host each kernel lowers in exactly the
shared memory `common.hopper_smem_bytes` counts (of 232,448 bytes) and fails one byte below (degrees
4 and 8).

| Family (recipes) | Implementation | Vectors / keys / heads | dQ / dK sums | Shared memory, fwd / dQ / dK/dV (bytes) |
| --- | --- | --- | --- | --- |
| Row shift (RoPE static, query) | `pallas_mosaic_gpu_query_polynomial_row_score_bias` | 17 / 0 / 0 | 17 / 0 | 110,704 / 86,096 / 74,800 |
| Pair shift (RoPE query_key) | `pallas_mosaic_gpu_query_polynomial_pair_score_bias` | 17 / 8 / 16 | 33 / 8 | 117,872 / 99,408 / 86,064 |
| Gate + row shift (gated static, query) | `pallas_mosaic_gpu_query_polynomial_per_order_endpoint_decay_row_score_bias` | 26 / 9 / 9 | 35 / 9 | 122,992 / 105,552 / 92,208 |
| Gate + pair shift (gated query_key) | `pallas_mosaic_gpu_query_polynomial_per_order_endpoint_decay_pair_score_bias` | 26 / 17 / 25 | 51 / 17 | 129,136 / 117,840 / 100,400 |

**Speed and registers** (`benchmarks/benchmark_score_bias.py`, job 242565 on `mastiff`, one H100
PCIe, GPU-cae52cf1 at bus E1:00.0; BF16 at the production per-GPU micro-batch of 16 × 2,048 tokens
and 14 heads; medians of 30). "× 240" is the three kernels' time for an update's 15 layers × 16
micro-batches. Spills are ptxas's spill stores/loads in bytes. Every kernel launches with 168
registers per thread (384 threads); its compute warpgroups raise theirs to 232 with `setmaxnreg`.

| Family | Forward ms | dQ ms | dK/dV ms | × 240 s | Spills, fwd · dQ · dK/dV |
| --- | ---: | ---: | ---: | ---: | --- |
| Query polynomial (RoPE control) | 0.83 | 1.71 | 1.39 | 0.94 | 0/0 · 0/0 · 140/140 |
| Per-order endpoint gate (gated control) | 5.95 | 15.81 | 15.45 | 8.93 | 0/0 · 2,316/2,184 · 1,400/1,636 |
| Row shift | 2.38 | 3.78 | 5.24 | 2.74 | 0/0 · 32/36 · 260/188 |
| Pair shift | 5.64 | 10.60 | 10.98 | 6.53 | 0/0 · 564/472 · 668/600 |
| Gate + row shift | 8.24 | 15.69 | 15.76 | 9.53 | 132/120 · 584/484 · 1,168/1,128 |
| Gate + pair shift | 11.24 | 23.54 | 35.11 | 16.77 | 104/96 · 1,416/1,376 · 2,400/2,224 |

The training qualification's run of the same benchmark on another of `mastiff`'s GPUs (job 242484,
commit `6ea6e210`) gave identical ptxas reports and times within 2.5%, except the row and pair dK/dV
kernels (10% and 7% slower there).

**Toolkit.** Without `CUDA_HOME`, Mosaic assembles with the node's `/usr/local/cuda` (CUDA 12.6 on
`mastiff`), giving PTX ISA 8.5 and other machine code: the unchanged per-order endpoint dQ kernel
spills 2,580/2,412 bytes instead of 2,316/2,184, the gate + pair-shift forward kernel 212/200 instead
of 104/96 (`tests.json`, job 242565). `kennel.sbatch` pins `CUDA_HOME` to the locked toolkit, so
acceptance and training assemble with CUDA 12.9, as every check here did. `benchmark_score_bias`
records `CUDA_HOME`, its ptxas version and each kernel's PTX ISA.

**Tests.**

- CPU, commit `e068dd94`: `tests/test_score_bias.py` 97 passed (its 4 GPU cases deselected),
  `tests/test_score_bias_lowering.py` 43, `tests/test_benchmark_score_bias.py` 3.
- GPU, job 242565 at commit `e068dd94` on `mastiff` (one H100 PCIe, GPU-cae52cf1 at bus E1:00.0),
  under the launcher's runtime (CUDA 12.9.86 ptxas, `JAX_PLATFORMS=cuda,cpu`, empty compilation
  cache): the gated runs' acceptance GPU stage, `-m gpu --mosaic-precision bf16` over its 11 modules
  in acceptance's order (the RoPE runs' 7 are a subset). 187 passed, none skipped or failed, 382
  deselected, in 809 s (JUnit SHA256
  `fd67fc13dc3c4247fdfbea23ea48d5bf216d4f7cfd287dee175d1b5963995936`): `mosaic_query` 44,
  `mosaic_repeatability` 3, `per_order_decay` 24, `mosaic_score_bias` 57, `score_bias` 4,
  `endpoint_decay` 30, `input_dependent_coefficients` 5, `query_tanh_polynomial` 16,
  `coefficient_forms` 3, `query_distance_decay` 1; `coefficient_scale` has no GPU case. The 4
  FP32-precision (`bf16x3`) cases, which BF16 deselects, passed separately (JUnit SHA256
  `72713b80287bbb26fc5f243c9179c36bfa4529a4ee09b39de6a9d17f71c54083`).

**Accuracy** against an independent FP64 all-pairs reference on the host
(`tests/test_mosaic_score_bias.py`: explicit `(s + beta_r)^r`, every gradient, each order's sums
separately). Largest relative L2 over its 16 BF16 cases: the four families at 512 tokens and
`m = 4`, at 2,048 tokens and `m = 8`, and with only orders 5–8; exactly zero and negative shifted
scores; pair offsets near −1.5.

| Quantity | Largest relative L2 | Bound | Share of bound | `bf16x3` (`m = 4`, 512 tokens) |
| --- | ---: | ---: | ---: | ---: |
| Output | 2.41e-3 | .006 | 0.40 | 1.66e-3 |
| dV | 2.44e-3 | .015 | 0.16 | 1.67e-3 |
| dQ, dK | 2.41e-3 | .025 | 0.096 | 1.66e-3 |
| Per-order sums (largest: `dlog_rate`) | 9.48e-6 | .01 | 9.5e-4 | 4.30e-6 |

- Exact zeros: 5.4% of the shifted scores are exactly zero and 47.4–47.5% negative.
- Zero shift (`b = 0`, `a = 1`, `p = q = 0`, or zero rows) against the existing kernels at `m = 8`
  and 2,048 tokens, bound `2e-4` relative L2: without the gate, against
  `api.query_polynomial_attention`, output 4.1e-5, dQ 9.0e-5, dK 8.8e-5, dV 6.6e-5, and `dc`
  bitwise equal; with it, against `api.per_order_endpoint_decay_attention`, dQ, dK and `dc` bitwise
  equal, output 3.6e-5, dV 7.1e-6, and the gate gradients at most 1.8e-7.
- Bitwise: changing every input of later tokens leaves earlier outputs and every gradient unchanged;
  a nonfinite shift or gate input makes exactly its causal rows NaN; 20 forward and backward
  evaluations at the production shape are identical.

**Full-shape training** (`benchmarks/qualify_training.py`: acceptance's pilot and resume workers on a
copy of each recipe with its own directory and W&B off; the production batch, 20 updates, validation
and a checkpoint at update 20; projection by acceptance's `project_runtime` for 7,570 updates). H100
PCIe on `mastiff`, jobs 242444, 242445, 242484, 242495 and 242557, pinned toolkit. The SXM estimate
divides by 1.35, the gated control's ratio (11.72 s per update here, 8.71 s in its production on an
H100 SXM). These pilots ran the six reference-shape recipes of the previous section with
`attention_backend: mosaic` at sources `4a5805f4`, `6291769a` and `6ea6e210`; the compact record keys
them by recipe.

| Recipe | Implementation | s/update (median, updates 2–20) | Projected hours | SXM estimate, hours | Allocator peak, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| RoPE static | row shift | 5.50 | 12.13 | 9.0 | 30.6 |
| RoPE query | row shift | 5.75 | 12.63 | 9.4 | 30.9 |
| RoPE query_key | pair shift | 10.20 | 22.23 | 16.5 | 31.0 |
| gated static | gate + row shift | 12.34 | 26.75 | 19.8 | 31.0 |
| gated query | gate + row shift | 12.54 | 27.16 | 20.1 | 31.5 |
| gated query_key | gate + pair shift | 20.18 | 43.48 | 32.2 | 31.8 |
| RoPE control `r25fffc16c2f3` | query polynomial | 3.52 | 7.83 | – | 30.3 |
| gated control `r227b0ed7c605` | per-order endpoint gate | 11.72 | 25.42 | – | 30.7 |

- Every pilot resolved the Mosaic backend and its implementation; every update's loss and gradient
  norm, the validation, and the attention diagnostics with the score-bias statistics were finite and
  valid. The allocator's limit is 59.4 GiB.
- The gated pilots (GPU at bus E1:00.0, 43–65 °C) held their update time within 1%. The RoPE pilots
  (bus 81:00.0) slowed as they ran: static 5.23 to 5.61 s, query 5.45 to 5.89 s, query_key 9.28 to
  11.36 s. Of the 20-update pilots on that GPU, only the RoPE control's (3.38 to 3.56 s) has
  telemetry: its 350 W power cap was active in every busy sample, with SM clocks of 1,185–1,515 MHz
  (1,755 maximum) and 58 to 82 °C. That the score-bias RoPE pilots were slowed the same way is an
  inference; if so, their projections overstate.
- The gated pilots' update times are the gated control's plus the kernels' difference (× 240, same
  GPU): static 12.34 s against 11.72 + (9.53 − 8.93) = 12.32 s, query_key 20.18 s against 19.56 s.

**Resume and determinism.**

- Acceptance's resume audit passed for all six recipes in a fresh process: the restored state equals the
  saved one, and the next batch and RNG are equal.
- Under production XLA settings the same executable, on the same state and batch, is not bitwise
  reproducible. In the RoPE and gated query_key pilots, against an uninterrupted rerun in the same
  process, update 1 has the same loss and gradient norms 6.5e-7 and 3.1e-7 relative apart, and over
  updates 1–20 the loss differs by up to 2.1e-5 and 1.6e-3 and the gradient norm by up to 6.5e-3 and
  0.41. A fresh process's resume against the in-process continuation of the same checkpoint: update
  21 has the same loss and gradient norms 1 and 2 ulp apart; at update 22 the losses differ by 5.7e-6
  and 3.8e-6 and the gradient norms by 5.2e-6 and 3.2e-3. With an empty compilation cache, the
  gradient norms are 5 and 210 ulp apart at update 21, and at update 22 the losses differ by 2.9e-6
  and 9.5e-7 and the gradient norms by 6.2e-4 and 4.1e-3.
- With `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (4-update pilots) all six are exact: the pilot
  equals an uninterrupted rerun bitwise, and both the in-process continuation and the fresh-process
  resume reproduce updates 5 and 6 and the final state's SHA256. The kernels, restore, data cursor
  and RNG are therefore exact; the differences above come from XLA's default arithmetic.
  Deterministic ops add 46.6–50.4 s per update (55.5–67.9 s against 5.5–20.2 s, 3.4–10.1×).

**Instrumentation parity** (the RoPE and gated query_key recipes, 10 updates, every arm in one
process): gradient-event capture off; capture as the recipe sets it; capture forced from update 2 (a
copy whose only changes are `gradient_event_factor: 1.0001`, `gradient_event_min_norm: 0` and
`gradient_event_min_history: 1`, with the rule's warmup gate and factor then set to zero in the
capture code alone, so the optimizer's warmup is unchanged); and validation with attention
diagnostics every 2 updates.

- Deterministic ops: every arm's losses, update numerics, state hashes and final state equal the
  capture-off arm's bitwise. All 9 events' recorded pre-update hashes equal the capture-off states,
  the 3 kept captures (updates 2–4) restore to exactly those states, and all 5 diagnostic
  validations are valid and report the score-bias statistics.
- Production settings: every arm, including capture as the recipe sets it (which changes nothing on
  the device in 10 updates), differs from the capture-off arm from update 1 by the noise above:
  losses by at most 1.7–2.1e-5 (RoPE query_key) and 0.9–1.0e-4 (gated query_key). Kept captures
  restore exactly to their recorded hashes.
- A kept capture's synchronous save took 20.9–35.2 s (21.2–25.0 s with deterministic ops).

Replaying a captured event under production settings therefore reproduces its update only to FP32
rounding, which then grows; with deterministic ops, replays agree with each other bitwise (a
different program from the production update).

**B300.** On the CPU host, `tests/test_score_bias_lowering.py` presents B300's kernel set (and a GPU
without Mosaic kernels): every entry point raises before tracing a kernel, and B300's own forward,
dQ and dK/dV kernels refuse the four families. `test_b300_refuses_before_tracing_a_kernel` checks
the same on the H100 (4 cases, in the GPU stage above). No B300 allocation was used.

**Open.** Kernel times and pilots on an H100 SXM (`dogo` was fully allocated throughout).

```bash
# Inside one H100 allocation, from the worktree root, under the launcher's runtime:
source scaffolds/transformer/launchers/kennel_runtime.sh
PY=scaffolds/transformer/.venv/bin/python
CUDA_HOME="$(transformer_cuda_home "$PY")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" PYTHONPATH="$PWD"
T=scaffolds/transformer/tests
JAX_PLATFORMS=cuda,cpu "$PY" -m pytest -q -rA -c scaffolds/transformer/pyproject.toml \
  --import-mode=importlib -m gpu --mosaic-precision bf16 $T/test_mosaic_query.py \
  $T/test_mosaic_repeatability.py $T/test_per_order_decay.py $T/test_mosaic_score_bias.py \
  $T/test_score_bias.py $T/test_endpoint_decay.py $T/test_input_dependent_coefficients.py \
  $T/test_query_tanh_polynomial.py $T/test_coefficient_forms.py $T/test_query_distance_decay.py \
  $T/test_coefficient_scale.py
B=scaffolds.transformer.benchmarks.benchmark_score_bias
JAX_PLATFORMS=cuda "$PY" -m $B kernels --output kernels.json --repeats 30
JAX_PLATFORMS=cuda "$PY" -m $B registers --output registers.json
# Each recipe and the adapter as committed at those sources; a resume is its own process. Add
# XLA_FLAGS=--xla_gpu_deterministic_ops=true and --pilot-steps 4 for the exact checks, and the parity
# mode for the instrumentation arms.
Q=(<recipe>.yaml --directory <dir> --experiment transformer_pp
   --adapter experiments.transformer_pp.score_bias_acceptance --limit-hours 24 36 48 72)
QT=scaffolds.transformer.benchmarks.qualify_training
JAX_PLATFORMS=cuda "$PY" -m $QT pilot "${Q[@]}" --no-references
JAX_PLATFORMS=cuda "$PY" -m $QT resume "${Q[@]}"
```

### Per-order cumulative gates on Mosaic (2026-11-11)

`attention_backend: mosaic` runs per-order cumulative query gates on H100's Mosaic kernels, family
`per_order_cumulative_decay` (`api.per_order_cumulative_decay_attention`;
[README](../README.md#forget-gate)). Each causal tile forms order r's FP32 difference
`L_tr - L_ir` before scaling it to base 2; the dQ kernel sums `dc_r`, the dK/dV kernel sums `G W_r`
over each key's queries, and `dL_r = c_r dc_r` minus that sum. B300 refuses the family before
tracing a kernel, and `auto` never selects it.

On CPU, the 41 checks of `tests/test_mosaic_per_order_cumulative.py` and the 123 of
`tests/test_per_order_cumulative.py` passed: tile mathematics against FP64 at degrees 1, 4 and 12
(sums near -3,000 included), lowering in exactly the counted shared memory (at degree 12 and
`d_K = 64`, 125,040 forward, 95,312 dQ and 95,280 dK/dV bytes of 232,448; one byte less fails), the
entry point's operands, refusals and gradient assembly, dispatch, recipe rules, and acceptance
selection. Mutations that scale each sum before subtracting, drop the masked-query guard, flip
`dL`'s sign, or drop the nonfinite-to-NaN step fail 6, 1, 2 and 2 checks. Existing mechanisms are
unchanged against `ea40604b`: identical jaxprs, kernel bodies included, for 341 programs (every
existing Mosaic family's forward and VJP on H100 and B300 at degrees 1-12 and widths 8-128, the
query-polynomial dispatch on the scan, the strips and Mosaic, `attention_core`'s gates, and eight
small models' losses and gradients); identical support predicates over degrees 0-14, every width,
and 128-8,192 tokens; and identical validation, recipe, attention identity, acceptance selection,
Mosaic test options and parameter count for all 456 run recipes (399 valid). Only
`mosaic_decay_supported()` changes, to true, for the blocked per-order cumulative recipes
`r319b83140040` and `rf8fc9ab2cecc`, whose backend stays blocked.

GPU checks ran on one H100 PCIe (`mastiff`) with the locked CUDA 12.9 environment (nvcc 12.9.86,
cuBLAS 12.9.2.10, cuDNN 9.22), JAX 0.10.0 and driver 560.35.05; H100 SXM (`dogo`) was reserved.
The kernel code is identical at the three commits. Every check passed, none skipped:

| `-m gpu` module | Passed | Job, commit |
| --- | ---: | --- |
| `test_mosaic_per_order_cumulative.py` | 31 | 243152, `514db91b` |
| `test_per_order_decay.py` | 24 | 243152, `514db91b` |
| `test_endpoint_decay.py` | 30 | 243168, `db2ac208` |
| `test_input_free_gate.py` | 8 | 243168, `db2ac208` |
| `test_mosaic_query.py` | 79 | 243168, `db2ac208` |
| `test_mosaic_repeatability.py` | 6 | 243168, `db2ac208` |
| `test_per_order_cumulative.py` | 3 | 243168, `db2ac208` |

**Production context.** [`per_order_cumulative_context.py`](per_order_cumulative_context.py)
(job 243189, commit `f7500a94`) runs one micro-batch of the degree-twelve recipe's attention,
[16, 2048, 14, 64] in BF16, through `attention_core` on Mosaic and on the causal strips, against
FP64 (the test oracle's formulas evaluated on the GPU; 3e-15 from the host oracle on one head).
Q and K rows have unit RMS (QK RMSNorm's scale) and the coefficients are near Taylor. L comes from
the recipe's gate at initialization, or from trained-like gates: per head and order a mean
`|log alpha|` log-uniform on [1e-3, 70] with log-normal token spread, the range of
`rf8fc9ab2cecc`'s final per-order gates. Relative L2 × 1e3 of the output and the Q, K and V
gradients:

| Gates, scores | Lowest L | Mosaic − strips | Mosaic − FP64 | Strips − FP64 | BF16 floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initialization, unit scale | -2.4e3 | 2.62-2.65 | 2.34 | 1.67 | 1.66 |
| Trained, unit scale | -1.6e5 | 2.60-2.62 | 2.34-2.35 | 1.67 | 1.66 |
| Trained, Q doubled | -1.6e5 | 2.74-2.82 | 2.32-2.35 | 1.67-1.68 | 1.65-1.67 |

Mosaic sits at 1.39-1.42× the BF16 rounding floor, the level of every Mosaic query family with
BF16 weights (accuracy table below), and the strips at 1.01×. On the last case H100's per-order
query-distance kernels, given the same per-token `log alpha`, differ from the strips by
2.71-2.76e-3. The FP32 coefficient and log-decay gradients agree with FP64 within 1.3e-6 (Mosaic)
and 1.5e-6 (strips) order by order, the log-decay gradient measured against the larger of its norm
and `c dc`'s, and with each other within 6.8e-7. Every output and gradient is finite.

**Speed** with [`benchmarks/benchmark_decay.py`](../benchmarks/benchmark_decay.py) on the same
H100 PCIe, degree 12, 16 sequences of 2,048 tokens, 14 heads, widths 64; milliseconds, each kernel
called alone (job 243189) and the jitted attention forward and backward (job 243168):

| Family | Forward | dQ | dK/dV | Attention, forward + backward |
| --- | ---: | ---: | ---: | ---: |
| Per-order query distance, Mosaic | 3.12 | 8.69 | 6.43 | 15.58 |
| Per-order cumulative, Mosaic | 3.40 | 5.83 | 9.63 | 15.68 |
| Per-order cumulative, causal strips | | | | 74.74 |

Complete updates (16 micro-batches of 16 × 2,049 synthetic tokens and AdamW, median of five after
two warm-up updates, each recipe in its own process; job 243168):

| Recipe | Backend | s/update | Allocator peak | Compile |
| --- | --- | ---: | ---: | ---: |
| `r95bfff096cd4`, per-order cumulative, degree 12 | Mosaic | 7.70 | 29.1 GiB | 37 s |
| `r95bfff096cd4` | causal strips | 25.11 | 33.3 GiB | 128 s |
| `rfe140c073b2f`, per-order query distance, degree 12 | Mosaic | 7.62 | 28.4 GiB | 37 s |
| `r20fe6fe864bc`, per-order query distance, degree 8 | Mosaic | 5.86 | 27.8 GiB | 36 s |

The seven benchmark updates' losses agree within 1e-5 between Mosaic and the strips.
`rfe140c073b2f` trained at 5.15 s per update on H100 SXM (median of its validation windows) and
`r20fe6fe864bc` at 5.70 s on H100 PCIe, 0.97 of its benchmark. `r95bfff096cd4`'s update takes
1.06% longer than `rfe140c073b2f`'s, so it projects to 5.2 s per update on SXM, 10.9 hours for
7,570 updates, and to 7.5-7.7 s on PCIe, 15.8-16.2 hours. It trained at 8.68 s per update on H100
PCIe (job 243318, median of its 500-update validation windows), 1.13 times its benchmark. Records:
[runs/per-order-cumulative-mosaic/](runs/per-order-cumulative-mosaic/).

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m "not gpu" \
  scaffolds/transformer/tests/test_mosaic_per_order_cumulative.py \
  scaffolds/transformer/tests/test_per_order_cumulative.py
# In one H100 allocation with its environment's python as $PY, from the worktree root:
$PY -m pytest -q -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu \
  scaffolds/transformer/tests/test_mosaic_per_order_cumulative.py  # and the modules above
PYTHONPATH=$PWD $PY scaffolds/transformer/validation/per_order_cumulative_context.py --output context.json
$PY -m scaffolds.transformer.benchmarks.benchmark_decay kernels --degree 12 \
  --families per_order_query_decay per_order_cumulative_decay --output kernels.json
$PY -m scaffolds.transformer.benchmarks.benchmark_decay attention --degree 12 \
  --families per_order_query_decay per_order_cumulative_decay --output attention.json
$PY -m scaffolds.transformer.benchmarks.benchmark_decay update \
  experiments/transformer_pp/runs/r95bfff096cd4/config.yaml --output update.json  # --backend blocked: strips
```

### Query-key gates on softmax and the exponential (2026-11-10)

`model.attention_forget_gate_conditioning: query_key` now applies to the query-distance gate of
softmax and the unnormalized exponential: each pair's rate reads both of its tokens,
`log alpha_ti = -exp(A_log) softplus(x_t w_query + x_i w_key + dt_bias)`, and
`(t - i) log alpha_ti` joins its score. It runs on the blocked backend, scan or strips, with RoPE
or no positional encoding ([README](../README.md#forget-gate)).

All 159 CPU checks in `tests/test_exponential_query_key.py` passed. FP64 all-pairs references
cover softmax and exp at ragged sizes (37 tokens in 8-query blocks and 16-query strips; 5 tokens;
one token): the output and the Q, K, V, query-input, key-input, and log-rate gradients through the
scan's code and the strips in FP64 (to 1e-10) and through the public FP32 functions (relative L2
below 3e-5); the gate projections (`w_query`, `w_key`, `A_log`, `dt_bias`, 4e-5); and the attention
module from its weights. Further checks: the pair rate reads only its endpoints, with an exact unit
diagonal; a query input moves its own row and a key input the later rows; a nonfinite query input,
key input, or log rate poisons exactly its row, the rows from its token on, or its head's rows,
every other row bitwise unchanged; open and closed gates stay finite with exactly zero gate
gradients; raw scores far outside FP32's exp range never reach the exponential (a masked future
key's raw score of 200 beside causal scores of 0 or an own score of -100, a causal raw score of 200
that a closed gate decays to -1,800, and for softmax scores of 100 under an open gate), with the
output and every gradient finite and equal to FP64 through two-query blocks and strips; the scan's
gate statistics cover every causal pair; a zero key input reproduces the query-distance gate (FP32
to 2e-6, FP64 to 1e-12); constant pair rates at ALiBi's slopes reproduce the cumulative ALiBi decay
(FP32 to 5e-6, FP64 to 1e-12); at initialization (`w_key = 0`) the model's logits equal its
query-gate counterpart's bitwise and every shared gradient agrees to 2e-5; the key projection is the
only added parameter (188,160 in the Transformer++ shape) and takes weight decay; plus the accepted
and rejected recipes, identity, diagnostics, training updates, acceptance selection, and the
softmax-backbone initialization check. Mutations that drop the key input from the pair rate, or stop
invalid endpoint rates from poisoning, fail 53 and 36 of them. Mutations that exponentiate the
endpoint kernels' masked logits, take softmax's row maximum over raw or masked scores, or drop its
shift each fail 2 to 6 of the extreme-score checks, on the scan and the strips alike.
`tests/test_blocked_strips.py` adds both kernels' query-key families to its scan/strips equivalence
checks, invalid-row cases, core routing, and FP64 two-layer models (175 passed).
`tests/test_exponential_query_distance.py` (126) and `tests/test_forget_gate_attention.py` no longer
expect this conditioning to be rejected. The former gains the same 14 extreme-score checks for the
query gate, which those mutations also fail when applied to every exponential and every decayed
softmax.

Existing mechanisms are unchanged against `e29277ba`. The jaxprs are identical for 808 programs:
`blocked_dot_product_attention` and `strip_dot_product_attention` for that commit's 29 argument
families (forward, VJP, and the scan's diagnostics, with FP32 and BF16 Q/K/V, in 8- and 128-query
scan blocks and 16- and 512-query strips), and `attention_core` for its 19 core families (forward,
VJP, and diagnostics through the scan and the strips). The loss-and-gradient programs, losses,
gradients, parameters, identities, and diagnostics of 24 small kernel and forget-gate models are
identical. Validation, recipe, attention identity, acceptance test selection (CPU and GPU, for each
of the blocked, fused, mosaic, and cudnn backends), and parameter count are identical for all 446
run recipes (392 valid).

`test_gpu_strips_query_key_matches_float64_at_the_production_context` (softmax, exp) runs BF16
Q/K/V at [1, 2048, 14, 64] through the gate projections at width 896 and the strips, with pair
half-lives from under one token to the context, against a host FP64 oracle that matches FP64
autodiff on CPU. It requires the output, dQ, dK, and dV within 1.2x their BF16 rounding floor and
the FP32 gradients of `w_query`, `w_key`, `A_log`, and `dt_bias` within 1e-4. Run on CPU, the case
measures 1.00x the floor and gate-parameter errors of 1.6e-7 to 5.9e-7. GPU qualification runs in
the production run's acceptance, which selects this file for these recipes on CPU and GPU.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m "not gpu" \
  scaffolds/transformer/tests/test_exponential_query_key.py \
  scaffolds/transformer/tests/test_blocked_strips.py
```

### Data-dependent decay: query-distance gates on softmax and the exponential, per-order cumulative gates (2026-11-09)

`model.attention_forget_gate_mode: query_distance` applies to softmax and the unnormalized
exponential with the input-form softplus gate: every score gains `(t - i) log alpha_t`, ALiBi with a
learned, input-dependent slope per query token. `attention_forget_gate_per_order: true` takes the
cumulative mode for nonlinear query-coefficient polynomials: each order has its own gate and sum,
`W_ti = sum_r c_r(x_t) s_ti^r exp(L_tr - L_ir)`. Both run on the blocked backend, scan or strips
([README](../README.md#forget-gate)).

All 114 CPU checks in `tests/test_exponential_query_distance.py` passed. FP64 all-pairs references
cover softmax and exp at ragged sizes (37 tokens in 8-query blocks and 16-query strips; 5 tokens; one
token): the output and the Q, K, V, and raw log-alpha gradients through the scan's code and the
strips in FP64 (to 1e-10) and through the public FP32 functions (relative L2 below 3e-5); the gate
projection (every gate parameter, 4e-5); and the attention module from its weights. Further checks:
an invalid rate (NaN, ±inf, positive) poisons only its query's row, every other row bitwise
unchanged; open and closed gates stay finite; the gate is query-local; a constant per-head rate
reproduces the cumulative ALiBi decay (FP32 to 2e-6, FP64 to 1e-12); the parameters, their initial
values, and the Transformer++ count are cumulative FoX's; with `w_alpha = 0` the model equals
cumulative FoX (logits, and every gradient but `w_alpha`'s, to 2e-5); plus the accepted and rejected
recipes, identity, diagnostics, training updates, and acceptance selection.

All 92 CPU checks in `tests/test_per_order_cumulative.py` passed, at degrees 4 and 8 with unit-scale
scores and the coefficient and cumulative-sum gradients compared order by order. FP64 references
cover the scan and strips in FP64 (1e-10) and FP32 (3e-5), and the module's route through
`forget_gate.log_decay` and the cumulative sum (every gate parameter, 5e-5). Duplicated orders
reproduce shared cumulative FoX (FP32 to 2e-6, FP64 to 1e-12), with each shared parameter's gradient
the sum over its copies. Constant gates reproduce per-order query-distance gates (FP32 to 2e-5, FP64
to 1e-11). A nonfinite `log alpha` poisons the rows from its token on. Open and closed gates
(`log alpha` 0, -100, -1e30, and mixed by order) keep the output and every gradient finite and
within 3e-5 of FP64: each order's gap is masked before `exp`, a closed order keeps only the diagonal
term, and its `log alpha` gradient is exactly zero. The jaxprs of the scan and the strips, forward
and VJP, hold one order's `[B,H,Q,K]` factor at a time. Orders stay isolated, and the degree-zero
term gives the decay matrix `exp(L_t0 - L_i0)` with an exact diagonal. At initialization the model
equals shared cumulative FoX, and with `w_alpha = 0` it equals the per-order query-distance model.
Each extra order adds 188,580 parameters in the Transformer++ shape. At the Transformer++ widths
and context with BF16, where Mosaic runs the per-order query-distance twin, Mosaic refuses the
cumulative recipe. The suite also checks weight-decay masks, training updates of every layer, head,
and order, diagnostics (per-order half-lives, and the formula of each decay mode), and acceptance
selection.
`tests/test_blocked_strips.py` adds both families and the per-order query-distance family to its
scan/strips equivalence checks (161 passed). Its FP64 model check compares every gradient to 1e-11,
except the forget-gate parameters of the query-distance exponential, which must agree within one
FP32 epsilon per summed token (64) at their scale: the gate computes in FP32 by design, both
programs give it the same FP32 cotangent, and XLA orders that recipe's FP32 sums over the tokens
differently (2 ulps in `dt_bias`'s gradient on CPU).

Existing mechanisms are unchanged against `e6ed4e58`. The jaxprs of
`blocked_dot_product_attention` and `strip_dot_product_attention` are identical for 37 existing
argument families (666 programs: forward, VJP, and the scan's diagnostics, with FP32 and BF16
Q/K/V, at two block and strip sizes). Validation, recipe, attention identity, acceptance test
selection, and parameter count are identical for all 442 run recipes (388 valid). The
loss-and-gradient programs, values, and diagnostics of 20 small kernel and forget-gate models are
identical.

`test_gpu_strips_query_distance_matches_float64_at_the_production_context` (softmax, exp) and
`test_gpu_strips_per_order_cumulative_matches_float64_at_the_production_context` (degrees 4, 8) run
BF16 Q/K/V at [1, 2048, 14, 64] through the strips against host FP64 oracles, which match FP64
autodiff on CPU. They require the output, dQ, dK, and dV within 1.2x their BF16 rounding floor, and
the FP32 log-decay gradients (with the coefficient gradients, per order) within 1e-4. Run on CPU,
these cases measure 1.00x the floor and FP32 gradient errors of 1.4e-7 to 5.1e-7. GPU qualification
runs in each production run's acceptance on H100 and B300; acceptance selects these files for the
recipes that use either mechanism.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m "not gpu" \
  scaffolds/transformer/tests/test_exponential_query_distance.py \
  scaffolds/transformer/tests/test_per_order_cumulative.py \
  scaffolds/transformer/tests/test_blocked_strips.py
```

### Degree-twelve query-distance gates on Mosaic (2026-11-09)

At degree twelve, `d_K = 64`, and 2,048 tokens, H100's per-order kernels lower for cuda on a
CPU host in exactly their counted shared memory (one byte less fails): 114,800 (forward),
94,288 (dQ), and 78,896 (dK/dV) bytes for query gates, and 126,064, 109,648, and 96,304 for
query–key gates, of 232,448. CPU tests lower the production per-order entry points (forward
and all gradients) at degree twelve for H100 and check that B300 refuses degree twelve before
tracing a kernel. The shared query-decay entry point at degree twelve lowers the same way for
both H100 and B300 (checked once on a CPU host; lowering checks shared memory only).

`tests/test_per_order_decay.py -m gpu` adds degree-twelve FP64 and duplicated-order cases with
the degree-eight conventions; on B300 the FP64 cases expect the refusal, and the query–key
duplicated-order case at 1,024 tokens skips. `tests/test_endpoint_decay.py -m gpu` adds the
shared query gate at degree twelve and, above degree four (the degree-eight case too),
compares the coefficient gradient order by order: in its FP64 reference the highest order
holds 97% (degree eight) and 98% (degree twelve) of that gradient's norm, and at degree twelve
orders 0–8 each hold under 0.13%. On CPU, with the FP64 reference at the degree-twelve
per-order test inputs, dropping order 12, giving order 12 order 0's gate, or dropping orders
9–12 each fails that test's checks for both gates: the affected orders' coefficient and gate
gradients miss by 100–450× their tolerance, and dropping orders 9–12 also moves the output by
0.0082–0.0083 against its 0.006 bound (dropping order 12 alone moves it by 3.3e-4–3.7e-4).
Those per-order sums come from the dQ kernel. The output and the q, k and v gradients, which the
forward and dK/dV kernels also compute, barely move with the top orders at degree twelve, so
both tests add a case that keeps only orders 9–12: there, dropping order 12 moves them by
1.4–7.6 (per-order gates) and 4.8–15 (shared gate) times their tolerances.
At commit `41f33aa9` both files passed on H100 (`dogo`, job 240451: 54 GPU checks).
Production acceptance passed its GPU checks on H100 SXM (`dogo`, jobs 240470 and 240471).

### Input-free distance gate for static coefficients (2026-11-09)

`model.attention_forget_gate_conditioning: none` gives learned static polynomial coefficients
a learned rate that reads no token, `lambda = exp(A_log) softplus(dt_bias)` per layer and head,
or per layer, head and order with `attention_forget_gate_per_order: true`; each weight is
multiplied by `exp(-(t - i) lambda)`. The coefficients and the negated rates are broadcast to
every query and run through the existing query-coefficient decay operation: the blocked scan or
strips, or the Mosaic `query_decay` and `per_order_query_decay` kernels.

`tests/test_input_free_gate.py` checks on CPU the output and the gradients of q, k, v, the
coefficients, `A_log` and `dt_bias` against FP64 through the scan and the strips at degrees 4
and 8, shared and per order; duplicated per-order gates against the shared gate; the
parameters and their initialization, the weight-decay exemptions, training steps,
diagnostics, and the accept and reject rules; and that historical recipes are unchanged. Its
GPU cases compare the Mosaic path with FP64 at 2,048 tokens and width 64 with unit-scale scores,
each order's coefficient and gate gradients compared separately, plus duplicated orders against
the shared kernels and a two-layer model against the FP32 blocked model. At commit `cc36a392`
the GPU cases passed on H100 (`dogo`, job 240451: 8 GPU checks), and production acceptance
passed its GPU checks on H100 SXM (`dogo`, jobs 240497–240500). B300 refuses the degree-eight
per-order gate before building a kernel.

### ALiBi on softmax and the exponential (2026-11-08)

`model.attention_forget_gate_form: constant` applies to softmax and the unnormalized
exponential as well as to query polynomials: the cumulative decay adds `-m_h (t - i)` to every
score, which for softmax is ALiBi. `attention_forget_gate_constant_rate: fixed` holds `m_h` at
ALiBi's slopes `2^(-8 h / H)` with no parameter; recipes without the field learn the rate as
before ([README](../README.md#forget-gate)).

All 70 CPU checks in `tests/test_constant_decay.py` passed. FP64 references cover the slopes
(the fixed log rate is the FP32 array the learned rate's initializer stores); the fixed rate's
empty parameter set and parameter counts; softmax and exp outputs and the Q, K, V, and log-rate
gradients through the scan and the strips (five 8-query blocks or strips of 37 tokens, FP32
relative error below 2e-5 against FP64 ALiBi); the attention module, from its projection
weights, against FP64 ALiBi under both implementations; learned and fixed models agreeing at
initialization (identical backbone, decays, logits, and shared gradients to 1e-5); historical
recipes without the field; accepted and rejected recipes; acceptance selection and the
softmax-backbone initialization check; diagnostics; and training updates. Halving the fixed
log rate fails 9 of them. In FP32 the cumulative decay at 2,048 tokens and 14 heads meets
`|L_t - L_i + m_h (t - i)| <= 2.2e-4` for `t - i <= 32`, summed sequentially or as a tree: a
relative weight error near 1e-4, against the BF16 output rounding of 1.7e-3.

`test_gpu_strips_alibi_matches_float64_at_the_production_context` (softmax and exp) runs BF16
Q/K/V at [1, 2048, 14, 64] through the fixed decay and the strips and requires the output, dQ,
dK, and dV each within 1.2x its BF16 rounding floor against a host FP64 oracle; the oracle
matches FP64 autodiff to 1e-12 on CPU. At commit `93f47769` both cases passed in the production
runs' acceptance on B300 (`akita`, job 239928) and H100 SXM (`dogo`, job 239929), with the
full-recipe pilot and resume checks.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  scaffolds/transformer/tests/test_constant_decay.py -m "not gpu"
```

### Architecture-aware per-order Mosaic support (2026-11-08)

Per-order recipes validate for the Mosaic backend when H100's kernels fit them
(`api.hopper_supported`). B300 refuses shapes its kernels cannot hold
(`api.blackwell_supported`) with a `ValueError` before tracing a kernel. Every other family
keeps its predicate: over
degrees 0–14, all Q/K widths, and 128–8,192 tokens, shared-family support is unchanged, and
B300's per-order acceptance equals the previous predicate (degree 7 query and degree 6
query–key at `d_K = 64` and 2,048 tokens).

`common.hopper_smem_bytes` counts each Hopper kernel's shared memory exactly. Lowered for
cuda on a CPU host, all 288 combinations of six families, degrees 1, 4, 8, and 12, Q/K widths
16–128, and the forward, dQ, and dK/dV kernels fit the counted bytes and fail one byte below.
The largest, per-order query–key at degree 12 and width 128, uses 167,024 of 232,448 bytes.
Per-order query at degree eight and width 64 uses 110,704 (forward), 86,096 (dQ), and 74,800
(dK/dV). CPU tests lower the production per-order entry points (forward and all gradients)
at degree eight for H100 and at B300's largest degree, and check the refusal one degree
higher. Lowering checks shared memory only, not register allocation, numerics, or speed.
`tests/test_per_order_decay.py -m gpu` adds degree-eight FP64 and duplicated-order cases, with
unit-scale scores so that every order contributes and each order's coefficient and gate
gradients compared separately, and `tests/test_endpoint_decay.py -m gpu` the shared query gate at
degree eight. They passed on H100 (`mastiff`, job 239920: 46 GPU checks), and production
acceptance passed its GPU checks on H100 (jobs 239930 and 239931).

### Coefficient release schedule (2026-11-08)

`optim.coefficient_release_schedule` (`all` or `by_order`), `coefficient_release_step`,
and `coefficient_release_end_step` hold learned static coefficients, or the diagonal
tanh query generator's `b`, `a`, and `W`, at their initial values per polynomial order
until each order's release update ([README](../README.md#coefficient-release-schedule)).

All 79 checks in `tests/test_coefficient_release.py` passed on CPU (one data-parallel
case on two host devices). Before release, frozen entries stay bitwise at their
initial values, including a weight-decayed `W`; frozen gradients leave clipping, every
Adam moment, and every other update bitwise identical to a step without coefficient
gradients, and the single-batch, accumulated, and data-parallel train steps log the
masked pre-clip norm. `by_order` first moves each order at exactly its listed update;
the masks depend only on the update count; released entries start from zero moments
under the global bias correction; with `query_coefficient_learning_rate_scale`, released
`b` and `a` take exactly the scaled update while `W`, frozen entries, and every other
array match the unscaled optimizer bitwise; a checkpoint resume across a release matches
an uninterrupted run bitwise, and each release reaches W&B at its logged step; a missing
or renamed coefficient array is an error rather than trainable; and without a schedule
the optimizer state, updates, and train step are bitwise unchanged, with or without the
query learning-rate scale. Each of these changes failed at least one check: removing
either mask, the single-batch or accumulated train step's masked gradient, or the
signed-zero update fill; releasing one update late; dropping the release when the query
learning-rate scale is set; scaling `W` with that scale; or logging the release one W&B
step early. The full CPU suite passed: 2,208 single-device tests, and 23
two-device tests with five four-device cases skipped.
The acceptance pilot (20 updates) and its resume update end before a release at update
455, so for such recipes the first release on a GPU happens in production.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  scaffolds/transformer/tests/test_coefficient_release.py
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  scaffolds/transformer/tests/test_coefficient_release.py -k distributed
```

### Per-order coefficient gates on H200 (2026-11-07)

`model.attention_coefficient_gate: sigmoid` multiplies each diagonal tanh query
coefficient by `2 sigmoid(x_t U_r + beta_r)` per layer, head and order inside the shared
generator, so the Mosaic query kernel receives the gated coefficients unchanged. `U` and
the gate bias start at exact zeros, so `2 g_r = 1` reproduces the ungated model, its
parameters and its derivatives in `W`, `a` and `b`; `U` decays with the coefficient
kernel decay and the gate bias never does.

The CPU checks in `tests/test_coefficient_gates.py` compare gated and ungated models at
degrees 8 and 10 bitwise (parameters, logits, and the raw gradients of every shared
parameter), require finite nonzero gate gradients for every layer, head and order, verify
the decay masks with an actual AdamW update, compare the gated generator and all its
gradients with an FP64 oracle, check the gate diagnostics against a NumPy oracle, and
resume a short training run exactly. Two GPU cases per degree compare the gated
generator through the Mosaic query kernel with FP64 at context 2,048 and width 64,
including the gate kernel and bias gradients per order, and the gated BF16 Mosaic model's
loss and gradient direction with the FP32 blocked model.

[H200 qualification](coefficient_gates_h200.json) passed for the two production recipes
(`r77698776d82c` at degree 8 on an H200 NVL, `r8c4a6ecdf4d2` at degree 10 on an H200):
510 CPU and 9 distributed CPU checks, 72 GPU checks each including the four gate cases
(the first Mosaic query qualification at degree 10 and on H200), the study adapter's
exact initial parity with the ungated recipe, and the full-recipe pilot and
checkpoint-resume checks. Pilot training cost 2.54 and 2.51 s per update.

### Per-order query-distance gates (2026-11-07)

The blocked polynomial backend supports independent softplus distance gates for
every order, including zero, conditioned on the query or query–key endpoints.
The focused CPU suites pass independent FP64 output and all-parameter-gradient
checks, matched full-model starts, per-order statistics, zero-order decay,
causality, invalid inputs, WD masks, and historical checkpoint compatibility.
The existing shared-gate and endpoint regressions also pass. Recipes are guarded
against unrelated changes, and full-model initialization is recovered by
collapsing the duplicated gate axes.

`tests/test_per_order_decay.py -m gpu` checks both variants at 2,048 tokens with
64-channel BF16 Q/K/V against FP64, including every gate gradient, on the blocked
backend and on the Mosaic per-order kernels (families `per_order_query_decay` and
`per_order_endpoint_decay`). At commit `b38c116f` these checks passed with the shared
decay, Mosaic query and repeatability suites (88 GPU checks, `--mosaic-precision bf16`)
on H100 (`dogo`, job 239602) and B300 (`akita`, job 239603). Production acceptance passed
its 113 GPU checks on both GPUs (jobs 239609 and 239610), and its pilot and resume checks
on H100. Throughput is recorded in the query_distance_decay topic's
`per_order_throughput.json`. Each run retains the authoritative acceptance result.

### Selected polynomial orders (2026-11-03)

`model.attention_coefficient_orders` stores only the requested monomial orders
for learned static/direct or diagonal tanh-query polynomials. Static coefficients
expand before factorial scaling; query generators project only the retained W
columns, scale b and a by actual-order factorials, and expand their effective
coefficients. Thus `[0, 2, 4, 6, 8]` has five static coefficients or five query
features per head and remains even after optimizer updates. Query projection W
retains its configured weight decay. Unset recipes retain their prior shapes.

The CPU checks in `tests/test_coefficient_orders.py` compare
compact and expanded models' logits and every parameter gradient, check an
independent factorial-scaled derivative and actual Adam updates, and cover
initialization, counts, diagnostics, acceptance, configuration compatibility,
and invalid settings. Query-specific checks compare all model gradients, an
independent coefficient formula and gradients, AdamW updates, and diagnostic
scaling. Two GPU cases compare static and per-query degree-eight Mosaic outputs
and Q/K/V/stored-coefficient gradients with FP64 at context 2,048 and width 64.
The launcher selects these cases for restricted recipes; GPU qualification is
required along with the full-recipe pilot/resume acceptance before production.

All 22 selected-order CPU checks and 23 existing coefficient-scaling regressions
passed. [H100 qualification](selected_orders_h100.json) passed for both static
and query coefficients: 50 and 70 GPU checks, respectively, including the two
restricted-order FP64 comparisons. Both full-recipe initialization, training,
evaluation and checkpoint-resume checks passed before production.

On 2026-11-07 the order checks were parametrized over a second set, the intercept with
the odd orders `[0, 1, 3, 5, 7]` at degree 7, so omitted orders, parity (`p(z) - p(0)`
odd rather than even), factorials and the penalty weights (`a_1` unpenalized) follow
each set; the two Mosaic FP64 cases run for both sets. All 32 CPU checks passed.
[H200 qualification](selected_orders_h200.json) passed for the odd-order query recipe
`r0bdac8e1b97b` (the first Mosaic query qualification on H200): 501 CPU and 9
distributed CPU checks, 72 GPU checks including the two odd-order degree-seven FP64
comparisons, and the full-recipe initialization, pilot and checkpoint-resume checks, at
2.08 s per training update.

### FoX with PaTH polynomial attention (2026-11-01)

The query-polynomial PaTH reference and its six fused polynomial kernels accept
cumulative FP32 log decay. Each causal weight is multiplied by `exp(L_t-L_i)`;
the backward accumulates the query and key contributions to `dL`, then automatic
differentiation propagates through the cumulative sum to the forget gate.
Future-key differences are masked before exponentiation. The ungated kernel
specializations retain their original arithmetic.

Forty-seven focused CPU checks passed: 13 new independent mathematical checks,
10 existing PaTH reference regressions, nine gated Triton interpreter checks,
six ungated interpreter regressions, seven combined-model checks, and two
positional-encoding checks. These cover all seven operand gradients, cumulative
gate derivatives, chunk and split boundaries, zero/strong gates, causality,
backbone initialization, parameter counts, and actual zero-gradient AdamW updates.
Eight new GPU cases cover degree four and width 64 through context 2,048.
On one H100 80GB HBM3, the shared acceptance of `r74495b47d226` (job 232534,
source `43dd39f4`) passed these GPU tests, the existing PaTH tests, a full-recipe
pilot and fresh-process resume. B300 is not qualified. CPU/interpreter success
alone does not qualify a GPU.

### Continuation projection decay and FoX without RoPE (2026-10-31)

The optional `qk_norm_token_scale_kernel_weight_decay` policy decays only the
query-temperature direction; its default remains false. The continuation validator
permits an explicit RoPE-to-none transition for a forget-gate child while preserving
every backbone parameter and its Adam history. Other continuations reject that
positional-encoding change.

Forty-five targeted CPU checks passed: 13 continuation-allocation tests, 16 token-scale
and parameter-policy tests, five FoX/restoration/positional-encoding checks, and 11
acceptance/initialization checks. Zero-gradient updates verify the actual AdamW change
on query-temperature `w`, coefficient `W`, and forget-gate `w_alpha`, with amplitudes
and biases exempt. Restoration tests verify exact parent weights and optimizer
hashes, independent optimizer clocks, and finite subsequent training. The query
temperature's decay flag leaves initialization and forward values unchanged.
Shared acceptance includes these mechanism checks before its full-recipe GPU pilot
and fresh-process resume; CPU checks alone do not qualify a GPU allocation.

At source `22be3818`, both degree-four continuations passed all acceptance stages
on one H100 with the locked CUDA 12 environment. The allocations used global batch
256, microbatch eight, and 2,048 prediction tokens per sequence. Test counts below
are CPU / distributed CPU / GPU; every case passed with no skips or errors.

| Mechanism and job | Hardware and backend | Passing tests | Runtime projection | Production trainer runtime |
| --- | --- | ---: | ---: | ---: |
| FoX without RoPE, decay on coefficient and forget-gate projections (`230954`) | `mastiff`, H100 PCIe, blocked | 517 / 9 / 24 | 9.63 h | 9.46 h |
| Query-temperature, decay on its direction (`230957`) | `dogo`, H100 SXM, Mosaic | 334 / 9 / 48 | 1.60 h | 1.51 h |

Both restored exactly the pretrained backbone parameters and Adam history at step
7,500, with new-parameter optimizer count zero. Their 20-update GPU pilots and
fresh-process resumptions matched checkpoint-state, next-batch, and RNG hashes;
the production restoration audits matched the qualified restorations. The actual
configured decay masks select only the requested projection arrays, leaving
amplitudes, coefficient biases, and forget-gate rate/bias parameters exempt.
Both completed all 2,000 production updates with 21 finite validation measurements
and no numerical-failure record. The runtime projections include pilot elapsed
time once and exclude the separate two-hour allocation reserve; production trainer
times exclude acceptance. These executions qualify the two H100 configurations,
not B300 or other untested hardware/backend combinations.

### Acceptance runtime projection (2026-10-29)

Acceptance times cold diagnostic/evaluation calls separately, then collects warm
diagnostic calls and at least three full global evaluation batches. Median warm
training-update, diagnostic, and evaluation-batch times scale with their respective
production counts. Checkpoint writes use their largest measured time. The disposable
pilot's complete elapsed time is included once; its updates do not replace production
updates. The two-hour allocation reserve remains. `runtime-projection.json` records
the cost breakdown even when an allocation is rejected.

The acceptance and continuation checks passed 110 cases. Timing, training, and
continuation checks passed 32 cases on one CPU device; five explicitly distributed
cases were skipped there. All six new timing checks and those five distributed
cases passed with two forced CPU devices. The timing checks compare exact model,
optimizer, batch/RNG, and checkpoint hashes; preserve validation measurements and
the packed-data cursor; cover continuation baselines; and reject invalid warmup or
extra-probe outputs. A broader two-device invocation hit an existing mocked-diagnostics
fixture in `test_training.py` that assumes one device, so only the targeted two-device
checks are claimed here.

The GPU qualification harness runs a 20-update pilot, measures a full warmed
validation on its checkpoint, and audits the next update in a fresh process.
The input recipe stays immutable; successful qualification removes the disposable
checkpoints. The [H100 qualification](runtime_estimate_gpu.json) uses the 200M
degree-four query-attention recipe, global batch 256, microbatch 16, context 2,048,
and 19 held-out batches. In job 229866 at source `057a8b71`, cold diagnostics took
21.48 seconds and the cold evaluation batch 17.13 seconds; warm diagnostics took
0.069 seconds and the median warm evaluation batch 1.063 seconds. The new estimate
for one full validation was 20.26 seconds; a separately restored process measured
18.67 seconds (8.54% overprediction). Exact fresh-process resume checks passed,
and the disposable checkpoints were removed. This is one sampled validation on H100 SXM,
not a claim of full-training wall-time accuracy or B300 qualification.

Run from a committed checkout, after checking live scheduler allowances:

```bash
sbatch --partition=standard --gres=gpu:1 --cpus-per-task=16 --mem=128G \
  --exclude=collie,samoyed,doodle,terrier,shiba \
  --output=scaffolds/transformer/validation/logs/runtime-estimate-%j.log \
  scaffolds/transformer/validation/runtime_estimate.sbatch \
  "$(git rev-parse HEAD)" <recipe.yaml> <new-qualification.json>
```

### Query coefficient learning-rate multiplier (2026-10-29)

CPU qualification passed 17 focused checks for
`optim.query_coefficient_learning_rate_scale`: only the stored query bias and
amplitude updates scale, after shared clipping and Adam normalization; projection
and backbone updates, Adam moments, and the optimizer state layout remain unchanged.
Checks span warmup through the cosine floor, verify the independent projection
weight-decay flag and no decay on bias/amplitude, reject unsupported recipes, and
preserve frozen resume settings. Configuration, training, parameter-policy, and
continuation regressions passed another 92 checks, including accumulated-gradient
equivalence and exact checkpoint resume; five multi-device cases were skipped on
the single CPU device. Production GPU pilots and resume checks remain mandatory.

Both single-H100 executions at source `9c39f921` passed the shared CPU,
distributed-CPU, GPU, initialization, pilot, and resume acceptance stages: the
projection-decay configuration on H100 80GB HBM3, and the 3x readout-LR configuration
on H100 PCIe. Both completed 7,570 production updates with finite validation losses
using the Mosaic query kernel and the locked CUDA 12 environment. These executions
qualify the optimizer options on those H100 variants; they do not add B300 evidence.

### Query-key projection decay (2026-10-30)

With `model.attention_coefficient_kernel_weight_decay: true`, query-key coefficients
decay the key projection `W_k` as well as the query `W`; `b`, `a`, and the key amplitude
stay undecayed. On CPU, a focused check of the parameter declarations and one AdamW
step passed with the flag off and on: both projections shrink by exactly
`1 - lr * 0.1` while `b`, `a`, and `a_k` keep their values. With the key projection
left undecayed, the check fails. The query-key, parameter-policy, coefficient-scale,
QK head-scale, sigmoid-coefficient, and query learning-rate suites passed 147 checks;
16 GPU checks were deselected. Production acceptance repeats the query-key CPU and GPU
tests on each allocation.

Integration with the forget-gate continuation implementation passed 48 CPU checks
across `test_query_coefficient_learning_rate.py`, `test_parameter_policy.py`,
`test_continuation.py`, `test_token_scale_continuation.py`, and
`test_continuation_acceptance.py`, including the separate Adam state for introduced
gate parameters and the a,b-only learning-rate selection.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/tmp \
  scaffolds/transformer/.venv/bin/python -m pytest -q \
  scaffolds/transformer/tests/test_query_coefficient_learning_rate.py \
  scaffolds/transformer/tests/test_config.py \
  scaffolds/transformer/tests/test_training.py \
  scaffolds/transformer/tests/test_parameter_policy.py \
  scaffolds/transformer/tests/test_continuation.py
```

### Coefficient forms and the forget gate

The [record](runs/coefficient-forms/qualification.json) pins the qualified source. On
CPU, all 102 test files passed (16 contain only GPU cases). Independent FP64
references verify pure tanh $c_r = \tanh(w_r^\top x)$, affine sigmoid
$b + a\,\sigma(w^\top x)$, and blocked attention whose weights carry a Gated DeltaNet
forget gate $\prod_{j=i+1}^{t}\alpha_j$, including derivatives through the cumulative log
decay. Acceptance's softmax controls drop the gate. On one B300 (job 227513, CUDA 13) and
one H100 PCIe (job 227514), 57 checks each passed, including BF16 production-backend loss and
gradients against FP32 and BF16 blocked references for all three runs.

### Teacher-activation coefficient distillation

The native capture/student adapters and independent dense regressions have 13
passing CPU checks: preserved backbone and Taylor initialization, actual teacher
activations after earlier free layers, production coefficient and gradient parity,
independent clipping/Adam, synthetic teacher recovery, finite checks, and exact
optimizer resume. Nine study protocol checks cover selected-data integrity,
pooled metrics, and actual Orbax recovery. See
[CPU qualification](coefficient_distillation_cpu.json). Full-size GPU qualification
is a mandatory startup gate recorded in each execution's acceptance artifact;
the CPU checks alone do not qualify a GPU/runtime combination.

[B300 qualification](coefficient_distillation_b300.json) passed on CUDA 13 with
Mosaic: capture and native logits/losses matched exactly across 256 sequences,
native coefficient projection matched exactly, the frozen backbone hash was
unchanged, and the next Adam update after checkpoint restoration was identical.

### Pretrained Taylor continuation

The continuation runtime preserves pretrained parameters, backbone Adam moments and
bias-correction count, and the packed-data/RNG cursor. Newly introduced static
coefficients have fresh independent Adam history; clipping acts on the combined
gradient. CPU tests cover exact parent restoration, matching starting objectives,
the first coefficient update against an independent Adam reference, unchanged
backbone-update semantics, interrupted packed training with exact resume, baseline
metrics, and rejection of changed source identities. See
[CPU test record](runs/continuation/cpu-tests.json) and the separately recorded
[full-size parent restore](runs/continuation/cpu-parent-initialization.json).
[B300 qualification](runs/continuation/gpu-b300.json) passed for both fixed Taylor
and learned static degree-eight continuations of the 200M model on one GPU with
CUDA 13 and Mosaic. Both restored identical backbone parameters and Adam history,
used matching data and RNG, passed finite 20-update pilots, and resumed exactly in
a fresh process. Each passed 47 GPU tests; fixed and learned cases additionally
passed 256 and 324 CPU tests respectively, plus nine distributed CPU tests each.

Query-conditioned continuations also preserve that state while introducing fresh
Adam history for bias, amplitude, and projection parameters. The
[query continuation CPU tests](runs/query-continuation/cpu-tests.json) cover matching
initial logits and backbone gradients, amplitude-first learning with initially
zero amplitudes and random projections, subsequent projection updates, and exact
interrupted resume. They also cover the alternative exact-Taylor initialization
with zero projections and unit amplitudes. The
[200M degree-four restore](runs/query-continuation/cpu-parent-initialization.json)
verified identical pretrained parameter and Adam hashes, with backbone count 7,500
and new coefficient count zero. [H100 qualification](runs/query-continuation/gpu-h100.json)
passed for degree-four fixed and query-conditioned continuations on one H100 PCIe
with CUDA 12 and Mosaic. The fixed and query cases passed 261 and 439 CPU tests,
respectively, nine distributed CPU tests each, and 47 and 68 GPU tests. Both passed
restored-state initialization, a 20-update pilot, and exact fresh-process resume,
using identical initial backbone/Adam histories and data/RNG positions. GPU
qualification remains the acceptance gate before each new allocation's production
training.

### Static coefficient capture and override

The [CPU validation record](runs/static-coefficient-override/tests.json) records
nine override checks and 37 runner, coefficient-scaling, and QK RMSNorm checks.
The static coefficient adaptation checks include
signed coefficients with 14 heads, QK RMSNorm, and truncated reference
initialization. Capture leaves the ordinary static forward unchanged and returns
the represented FP32 coefficients after factorial scaling. Replaying that table
uses the general per-query polynomial path without another factorial division or
activation. Checks cover JIT replay, independent query/example updates, gradients
through frozen later layers, an improving coefficient update, and unchanged
pretrained parameters. Run `JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python
-m pytest -q scaffolds/transformer/tests/test_coefficient_override.py` to repeat.
Each fit qualifies itself on its own GPU before production; its `oracle_acceptance.json`
holds the measurements.

### Direct query-coefficient capture and override

On 2026-10-23, the [CPU validation record](runs/coefficient-override-cpu/tests.json)
recorded 62 passing joint checks (8 override API, 4 fitting-runner, 50 records),
plus 3 existing compatibility checks. The synthetic degree-eight model preserves its original
forward when captured effective coefficients are replayed, supports JIT, and
propagates finite coefficient gradients through the frozen layers. Optimizing
only the coefficient table leaves every pretrained parameter unchanged. Independent
examples have matching gradients, Adam states, and updates when evaluated alone,
together, or split into microbatches. The runner checks also verify plateau timing,
checkpoint continuation, and durable learning-rate decisions. The new API requires
`model.remat: false`; existing model rematerialization remains unchanged.
The optimizer check differentiates an independent objective in factorial-scaled
increments, then verifies clipping, Adam moments, and effective coefficient updates.

The [B300 replay diagnostic](runs/coefficient-override-cpu/capture-replay-b300.json)
on `akita`, job `224096`, source `f47580a`, found exact logit and loss replay with
the captured effective coefficients. Multiplying and dividing those coefficients
by factorials changed them by at most `1.49e-8`, but amplified the maximum logit
difference to `0.28125`. The runner therefore stores the effective FP32 coefficients
directly and applies factorial scaling to optimizer gradients and updates. This
diagnostic performed no coefficient fitting.

With exact coefficient storage, separately compiled evaluation differed from the
capturing forward by at most `0.0003443` nats/token on
[B300, batch 1](runs/coefficient-override-cpu/external-replay-b300.json) and
`0.0007401` on [H100, batch 8](runs/coefficient-override-cpu/external-replay-h100.json).
These checks used source `d5823fd`, jobs `224097` and `224098`, and failed the
initial `1e-5` NLL comparison. Independently compiled BF16 forwards use an absolute
per-example tolerance of `0.002` nats/token (`rtol=0`); internal capture/replay
retains the tighter check. These measurements establish the discrepancy without
identifying its compiler-level cause.

Exact commands, pass counts, source file hashes, and the dependency-lock hash are
in the record. Repeat the new checks from the repository root:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  scaffolds/transformer/tests/test_coefficient_override.py \
  experiments/attention_scaling/tests/test_coefficient_oracle.py
```

GPU fitting qualification passed on [B300, batch 1](runs/coefficient-override-cpu/gpu-b300.json)
and [H100, batch 8](runs/coefficient-override-cpu/gpu-h100.json), jobs `224100` on
`akita` and `224101` on `dogo`, source `2e77d06`. Internal replay was exact. Repeated
independent forwards produced identical logits, and independent-graph logit
relative RMS errors were below `0.005` on both GPUs. Every example's evaluation
and training-forward NLL difference was below `0.002`. Both devices passed finite
gradient and improving-update checks, coefficient checkpoint round trips, and
array-exact next updates after restoration. Improvement and plateau measurements
use the same compiled training forward, so cross-graph offsets are not counted
as fitting gains.

The validation record pins these immutable qualification copies, their source run
IDs, jobs, hardware, source commit, and checksums. Each production allocation
repeats its own checks in the run's `oracle_acceptance.json`. Scientific fitting
outcomes remain under the runs; this record contains software qualification only.

### Selectable QK RMSNorm, truncated initialization, and pilot cadence

The shared scaffold now supports Q/K RMSNorm with independent learned channel
gains per layer, shared across heads, before RoPE; fixed standard attention scaling
remains in the backend. The opt-in `transformer_truncated_reference` initialization
conditions a Gaussian with standard deviation 0.02 on ±0.04 without variance
compensation, then scales residual projections by `1/sqrt(2L)`.
`rms_norm_fp32` selects FP32 backbone norm arithmetic. Existing named initializers,
cosine QK normalization, and historical precision defaults remain unchanged.

CPU verification on 2026-10-23 passed 81 QK/config/parameter-policy/telemetry tests
(four GPU-only cases were skipped), 23 selected initialization/norm/allocation
tests, and 48 acceptance/evaluation-cadence tests. These cover forward values and
gradients, gain ownership and decay exclusions, conditional-Gaussian support and
variance, legacy recipe identity, and scratch-pilot evaluation without mutating
the production recipe. Runtime projections count the actual evaluation and
checkpoint schedules, including the early cadence. Reproduce with:

```bash
export JAX_PLATFORMS=cpu
scaffolds/transformer/.venv/bin/python -m pytest -q scaffolds/transformer/tests/test_qk_rms_norm.py scaffolds/transformer/tests/test_config.py scaffolds/transformer/tests/test_parameter_policy.py scaffolds/transformer/tests/test_telemetry.py
scaffolds/transformer/.venv/bin/python -m pytest -q scaffolds/transformer/tests/test_initialization.py scaffolds/transformer/tests/test_normalization.py scaffolds/records/tests/test_records.py -k 'initialization or reference or norm'
scaffolds/transformer/.venv/bin/python -m pytest -q scaffolds/transformer/tests/test_acceptance.py scaffolds/transformer/tests/test_evaluation_cadence.py
```

This is CPU correctness evidence. Full-context GPU kernels, production memory and
throughput, and checkpoint continuation are qualified separately in each run's
acceptance record before production begins.

### B300 query-key production qualification

The fused query-key polynomial kernel is qualified on B300 at the Transformer++ shape:
BF16, context 2048, Q/K and value width 64, degrees 2, 4, 8 and 12, with the tuned
32-by-32 backward tiles at degrees 4, 8 and 12. Job `233107` on one B300 (`akita`,
CUDA 13 environment), source `1d9bc8f8` on main, passed all 19 GPU-marked query-key
checks in 5.7 minutes ([record](runs/query-key-b300/qualification.json),
[results](runs/query-key-b300/gpu-tests.xml)):

- every gradient against FP64 references, for FP32 and BF16 inputs, through the
  per-head output RMSNorm;
- full-context agreement with the blocked reference and the original 64-by-64 geometry;
- causality of outputs and gradients, and isolation of nonfinite future keys and factors;
- the composition with QK RMSNorm in the attention backend.

This establishes kernel correctness only. It measures neither throughput nor
full-budget stability, and each production allocation still runs its own acceptance.
Rerun it from a clean checkout or detached worktree of the source with the command in
[`qualify_query_key_b300.sbatch`](qualify_query_key_b300.sbatch).

### Query-key polynomial backward tiles

The one-pass fused query-key polynomial kernel uses 32-by-32 backward tiles for
H100 and B300 BF16 inputs with context 2048, Q/K and value width 64, and degree 4, 8 or 12.
Forward keeps its original 64-by-64 tiles; other devices and shapes retain the
original geometry. No polynomial, precision, or coefficient-scaling rule changes.

On one H100 80GB HBM3 on Kennel (`dogo`), the production-shaped benchmark
`[batch=32, time=2048, heads=16, width=64]` measured:

| Degree | Original forward + backward | Tuned forward + backward | Speedup |
| --- | ---: | ---: | ---: |
| 8 | 76.414 ms | 28.356 ms | 2.695x |
| 12 | 168.779 ms | 42.819 ms | 3.942x |

The [production comparison](runs/query-key-tuning/production.json), source
`618f442`, job `223789`, compares the actual mixed forward/backward geometry with
the original geometry. The [tile sweep](runs/query-key-tuning/tuning.json), source
`c53c02c`, job `223788`, measures seven shared forward/backward geometries per degree.
Inputs have QK-normalized scores with scale 8, synthetic full-factorial query
coefficients and varying key factors. Each timing includes a real forward output
and all five input gradients, uses resident arrays and a runtime cotangent, and
reports the median of ten synchronized calls after three warmups. Compilation and
parity checks are excluded. All candidates pass the recorded numerical gates,
including each coefficient/factor degree and head. Timings cover attention alone,
not projections, output RMSNorm, optimizer updates, or complete training steps.
They do not establish B300 performance or parity on retained trained checkpoints.

The [qualification record](runs/query-key-tuning/qualification.json) records all
22 passing checks in job `223789`: 12 selector/fallback checks and 10 H100 checks
covering independent FP64 comparisons, signed coefficients, output RMSNorm and
all input/gain gradients, context-2048 parity with the blocked and original
implementations, causality, and nonfinite masking. The forward output remains
byte-identical to the original kernel in the production-context comparisons.

**Degree 4** uses the same 32-by-32 backward from source `b81bb7b`. On the Transformer++
attention shape `[batch=16, time=2048, heads=14, width=64]`, the
[H100](runs/query-key-m4/h100-tuning.json) (job `225050`, `dogo`) and
[B300](runs/query-key-m4/b300-tuning.json) (job `225051`, `akita`) benchmarks measured:

| GPU | Original forward + backward | Production forward + backward | Speedup |
| --- | ---: | ---: | ---: |
| H100 80GB HBM3 | 13.717 ms | 8.961 ms | 1.53x |
| B300 | 20.025 ms | 8.156 ms | 2.46x |

Every geometry each sweep compiled passed its numerical gates (B300 cannot fit
64-by-128 tiles in tensor memory); the fastest (64-by-32 on H100, 7.930 ms; 32-by-64 on B300, 7.771 ms) beat the production
geometry by at most 11.5% and are not selected. In both allocations the tiling test file,
now at degrees 4, 8 and 12, passed all 58 GPU checks
([qualification](runs/query-key-m4/qualification.json)). Timings cover attention alone.

Repeat within an H100 allocation, choosing a fresh validation output directory:

```bash
RUN_DIR=scaffolds/transformer/validation/runs/query-key-tuning-repeat
python -m scaffolds.transformer.benchmarks.tune_query_key \
  --degrees 8 12 --candidates 64x64x4x2 --include-production \
  --output "$RUN_DIR/production.json"
python -m pytest -q scaffolds/transformer/tests/test_query_key_tiling.py
```

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

### Training telemetry on one GPU

Since `6755d6b` the single-device update records the same statistics. Checks in
`tests/test_telemetry.py`: on one CPU device, updates with and without telemetry are
bit-identical, for one batch and for accumulated microbatches; on two CPU devices, one
batch's statistics on one device agree with data-parallel training's to 1e-4; the full
loop records window summaries, gradient spikes, and numerical-failure statistics with
and without accumulation; remat training records none. The telemetry-off single-device
programs are unchanged: their jaxprs match the previous code's for both output
scalings, with and without update checks and accumulation.

Cost on one B300 (`akita`, jobs 241876 and 241899, training code of `6755d6b`; reports in
[`runs/telemetry-overhead-single-gpu/`](runs/telemetry-overhead-single-gpu/)), with the
benchmark above in its one-device mode: alternating off/on trials (four of 60 or 40
timed updates, two of 20 for the blocked and per-order recipes, after 5 warmup updates)
from the same initial state, synthetic tokens, and update RNGs, including the loop's
per-update packing and fetch of the statistics:

| Recipe (run) | Off, ms/update | On, ms/update | Overhead | Per run | Statistics |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12-layer width-640 query m = 8 + W decay, Mosaic (`r95a5691d293f`) | 437.6 | 440.0 | +0.55% | 18 s | 92 |
| Softmax, cuDNN (`r9040f12f3d80`) | 905.4 | 910.9 | +0.61% | 42 s | 86 |
| Exp with a query–key gate, blocked strips (`rd3501a40e020`) | 3,686.6 | 3,679.6 | −0.19% | – | 104 |
| Per-degree query–key gates m = 4, per-order Mosaic (`r03be769d2069`) | 10,918.7 | 10,920.7 | +0.02% | 16 s | 107 |

Telemetry adds 2.1–5.5 ms per update where it is measurable, under a minute per
7,570-update run. Losses with and without it differ by no more than between repeated
identical trials (0.0056–0.0111 over the timed updates for the cuDNN and Mosaic recipes,
whose updates are nondeterministic; 2e-5 for the blocked strips and per-order kernels).
The reports mark their source dirty only because earlier reports in the directory were
untracked; every hashed source file matches `6755d6b`. One-GPU H100 timing was not run;
two H100 PCIe measured +0.19–1.26% above.

### Single-device gradient accumulation

Before 7e5c962f the single-device update dispatched one program per microbatch and
added each microbatch's gradients with one kernel per parameter; the data-parallel step
already accumulated inside one compiled `lax.scan`. The single-device step now does the
same. Timing on one B300 (`akita`, CUDA 13), 100 updates on the packed training data,
seconds per update between updates 40 and 100, 16 microbatches per update
([`runs/single-device-accumulation/`](runs/single-device-accumulation/)):

| Recipe | Step compute (16 x benchmark) | Before | After |
| --- | ---: | ---: | ---: |
| 10 x 1024 softmax (`rdf00da0866a8`) | 0.75 s | 1.27 s (414k tokens/s) | 0.83 s (629k tokens/s) |
| 15 x 896 softmax (`r8ccda981be2b`) | 0.93 s | 1.58 s (331k tokens/s) | 1.02 s (516k tokens/s) |

Both recipes now run within 10% of their step compute, a 1.5x speedup for every
single-GPU run. The Transformer++ acceptance passed its CPU, distributed, GPU, hardware,
initialization and pilot stages on the new step.

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

## Fused PaTH attention kernels

`attention/backends/fla_path_attention.py` runs PaTH attention (`positional_encoding: path`,
`attention_backend: fused`) on flash-linear-attention's Triton kernels (commit `516143e`)
through `attention/backends/triton_port.py`. Softmax launches FLA's eleven
`parallel_path_attn` kernels unchanged; the tanh query polynomial replaces the six
softmax-dependent kernels with a fork (`fla_path_polynomial_kernels.py`). The reference is
the JAX implementation `attention/path.py` (`blocked`). Records: `runs/path-kernels/<gpu>/`
(tests, benchmark, GDN port) and `runs/path-pilot/` (training pilots).

### Tests

| GPU | Job, source | Fused PaTH GPU tests | GDN GPU tests | CPU |
| --- | --- | ---: | ---: | --- |
| H100 SXM (`dogo`, `.venv`) | 222686, `e39f466` | 35 passed | 48 passed | full suite passed |
| B300 (`akita`, `.venv-cuda13`) | 223582, `6db92b3` | 35 passed | 40 passed (widths 128 and 256 deselected) | full suite passed (`6db92b3`) |

The GPU tests compare outputs and every gradient (dq, dk, dv, dw, dbeta, dc) with the
FP32-HIGHEST JAX implementation in BF16 at the production context (T = 2048, width 64) for
softmax and m = 1-4, at lengths 128-1,536 around the 128-token query blocks and 512-token
splits, at widths 16, 32, and 128, and check causality, NaN rows, repeatability at the
production batch, and the model's dispatch. FLA's `parallel_path_bwd_dq` kernel, unchanged
and in the fork, fails to launch (`CUDA_ERROR_LAUNCH_FAILED`) at width 128 on B300, so the
fused path rejects width 128 on Blackwell and `auto` resolves it to `blocked`. CPU tests
run the same launches in Triton's interpreter in FP32 against the JAX implementation,
including FLA's A100- and Ada-class tile branches. FLA's GDN kernel crashes at width 128 in
FP32 on B300 and poisons the process ([CLUSTERS.md](../../../CLUSTERS.md)), so GDN cases at
widths 128 and 256 are not run there.

### Accuracy

Relative L2 error against FP32 HIGHEST, BF16 q/k/v, B = 2, T = 2048, 4 heads, width 64,
scores `8 cos(q, k)` (H100, job 223344):

| Kernel | Implementation | Output | dq, dk | dv | dw, dbeta | dc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | fused | 2.7e-3 | 4.1e-3 | 2.9e-3 | 7.1e-3, 7.6e-3 | |
| | JAX | 1.6e-3 | 1.7e-3 | 1.7e-3 | 8.7e-4, 9.3e-4 | |
| m = 1-4 | fused | 2.5-2.7e-3 | 5.4-5.7e-3 | 2.6-2.9e-3 | 8.2-8.7e-3 | 1.2-3.6e-3 |
| | JAX | 1.7e-3 | 1.8e-3 | 1.7e-3 | 1.0-1.1e-3 | 1.7-4.0e-4 |

B300 (job 223581) matches these to two digits. FLA rounds intermediates to BF16 and
stores dw and dbeta in BF16. Outputs, dq, and dc repeat bitwise over 10 runs at the
production batch on both GPUs; dk, dv, dw, and dbeta, summed with FP32 atomics, vary by up
to 4e-5, 2e-5, and 1e-3 (relative L2).

### Forget gate and the exponential

The unnormalized exponential is FLA's softmax output times its row sum, and the forget gate
(softmax and exponential, PaTH-FoX) enables FLA's own gate in six kernels with the gate
`L sqrt(d_K)`. The JAX implementation with the decay matches FLA's PyTorch reference given
FLA's per-token gate `log alpha sqrt(d_K)` (`test_decayed_softmax_matches_the_authors_reference`).
On B300 (`akita`, `.venv-cuda13`, job 233303), the 63 GPU tests of `test_forget_gate_attention.py`,
`test_path_attention.py`, and `test_path_attention_fused.py` pass: outputs and every gradient,
including the decay's, at the production context, at lengths 128-1,536, causality, and the
model's dispatch for softmax, exp, and both with the gate. The CPU suites pass (job 233367).
On H100 SXM (`dogo`, `.venv`), the acceptance of `rafcf22b799ed` (job 233373) passed 117 GPU
tests, a wider selection that includes these suites.

Merged with the FoX-PaTH polynomial (merge commit `19b9cdea`: one `log_decay` for every
kernel, which FLA's gate receives as `L sqrt(d_K)` and the polynomial fork as `L`), one H100
SXM (`dogo`, `.venv`, job 235146) passed the 71 GPU tests of `test_forget_gate_attention.py`,
`test_path_attention.py`, `test_path_attention_fused.py`, and `test_fox_path_attention_fused.py`
(`test_fox_path_attention.py` and `test_fox_path_model.py` have no GPU cases). A 72nd case
compared the fused and blocked gated-polynomial models in BF16: the fused `path_beta_bias`
gradient's error against FP32, 0.149, exceeded twice the blocked model's (0.055), so that
model is compared in FP32 by the interpreter only. The CPU suites of the touched mechanisms
pass (813 tests), as do the records and study tests (180). Log:
`validation/logs/fox-merge-qualify-235146.log` (Kennel).

Relative L2 error against FP32 HIGHEST, BF16 q/k/v, B = 2, T = 2048, 4 heads, width 64,
scores `2 cos(q, k)`, decay drawn from Gated DeltaNet's initial range (B300, job 233347):

| Kernel | Implementation | Output | dq, dk | dv | dw, dbeta | dL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | fused | 2.2e-3 | 3.6e-3 | 2.3e-3 | 6.9e-3, 7.0e-3 | |
| | JAX | 1.6e-3 | 1.7e-3, 1.8e-3 | 1.7e-3 | 8.8e-4 | |
| Softmax, gate | fused | 2.1e-3 | 2.8e-3 | 2.4e-3 | 4.9e-3, 5.2e-3 | 2.0e-3 |
| | JAX | 1.7e-3 | 1.7e-3 | 1.7e-3 | 5.9e-4, 6.0e-4 | 3.0e-4 |
| Exp | fused | 3.0e-3 | 5.8e-3, 5.7e-3 | 2.9e-3 | 8.6e-3, 8.8e-3 | |
| | JAX | 1.7e-3 | 1.8e-3 | 1.7e-3 | 1.1e-3, 1.0e-3 | |
| Exp, gate | fused | 2.8e-3 | 3.6e-3, 3.5e-3 | 2.9e-3 | 5.6e-3, 5.9e-3 | 1.7e-3 |
| | JAX | 1.7e-3 | 1.7e-3 | 1.7e-3 | 5.8e-4, 5.9e-4 | 5.5e-5 |

### Speed

Forward and backward per layer, B = 32, T = 2048, 16 heads, width 64, H100 SXM (job 223344):

| Kernel | Fused | JAX | Speedup | RoPE-free reference |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 17.8 ms | 119.3 ms | 6.7x | 3.1 ms (cuDNN) |
| m = 2 | 14.4 ms | 114.5 ms | 7.9x | 4.0 ms (Mosaic) |
| m = 4 | 15.9 ms | 112.2 ms | 7.1x | 4.7 ms (Mosaic) |

B300 (job 223581):

| Kernel | Fused | JAX | Speedup | RoPE-free reference |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 13.7 ms | 65.2 ms | 4.7x | 2.3 ms (cuDNN) |
| m = 2 | 13.7 ms | 61.3 ms | 4.5x | 3.6 ms (Mosaic) |
| m = 4 | 15.0 ms | 60.0 ms | 4.0x | 4.6 ms (Mosaic) |

FLA's own `parallel_path_attn` in PyTorch (`benchmark_upstream_path.py`, CUDA 13, torch
2.9.1, Triton 3.7.1, same B300 job) takes 3.60 ms forward and 13.34 ms forward and backward
for softmax, against the bridge's 4.20 ms and 13.74 ms. On the same inputs (B = 2, 4 heads)
the two agree to relative L2 2e-4 (output), 3e-4 to 1.2e-3 (dq, dk, dv), and 3.5e-3 (dw,
dbeta), below either's error against FP32, but not bitwise: dk, dv, dw, and dbeta use
atomics, and the outputs differ by up to 1e-3 on values up to 4 (cause not isolated; each
runtime autotunes the UT-transform kernels independently).

A full update of the 200M recipe on one GPU: 3.0-3.1 s fused and 11.9 s JAX on H100 SXM
(pilots, microbatch 16); 2.16-2.32 s fused and 5.9-6.2 s JAX on B300 (the m = 2 and m = 4
runs, microbatch 32).

### Training pilots

`pilot_path.sbatch` trains `rb901a10c4777`'s recipe (tanh m = 2) for 1,000 updates on one
H100 SXM, microbatch 16, from its initialization and data. Validation NLL:

| Implementation | 100 | 300 | 500 | 700 | 900 | 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JAX | 6.7490 | 5.1931 | 4.5408 | 4.1220 | 3.8334 | 3.7188 |
| Fused | 6.7465 | 5.1782 | 4.5320 | 4.1274 | 3.8501 | 3.7373 |
| Fused, repeat | 6.7463 | 5.1874 | 4.5303 | 4.1256 | 3.8415 | 3.7258 |

The fused runs differ from the JAX run by -0.015 to +0.019, and from each other by up to
0.025: the implementations agree within run-to-run variation.

### Gated DeltaNet on `triton_port`

`compare_gdn_port.py` lowers the GDN forward and backward (tied and independent gates,
BF16 and FP32) with the bridge before (`7fb44dc`) and after the move onto `triton_port`:
all 10 Triton custom calls per case are identical, including their compiled kernels, and
outputs and gradients are bitwise equal on H100 and B300
(`runs/path-kernels/<gpu>/gdn-port.json`).

### Reproduction

From a committed worktree on Kennel with both environments synced (for the comparison with
FLA's own PyTorch kernels on B300, also `uv sync --project
scaffolds/transformer/benchmarks/upstream_runtime --locked --link-mode hardlink`); the third
argument is `tests` (fused PaTH and GDN GPU tests and the GDN launch comparison) or
`benchmark`:

```bash
sbatch --partition=debug --nodelist=dogo --gres=gpu:h100:1 \
  --output=scaffolds/transformer/validation/logs/path-kernels/%j.log \
  scaffolds/transformer/validation/qualify_path_kernels.sbatch "$(git rev-parse HEAD)" H100 tests
sbatch --partition=long --nodelist=akita --gres=gpu:b300:1 \
  --output=scaffolds/transformer/validation/logs/path-kernels/%j.log \
  scaffolds/transformer/validation/qualify_path_kernels.sbatch "$(git rev-parse HEAD)" B300 benchmark
sbatch --partition=long --nodelist=dogo --gres=gpu:h100:1 \
  --output=scaffolds/transformer/validation/logs/path-pilot/%j.log \
  scaffolds/transformer/validation/pilot_path.sbatch "$(git rev-parse HEAD)" H100 rb901a10c4777 fused 1000 "" 16
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

### Frozen-backbone query-generator fitting

The shared `training.coefficient_fit` runtime supports native query-generator
parameter updates as well as free coefficient tables. Its
[CPU qualification](runs/query-generator-fit/cpu.json) covers original-forward
identity, selective generator updates, native Adam coordinates without duplicate
factorial scaling, downstream activation recomputation, group-gradient accounting,
and exact checkpoint resume. Six generator checks and four existing free-table
checks pass. Each fit qualifies itself on its own GPU before production; its `oracle_acceptance.json`
holds the measurements.

## Coefficients shared across fitting sequences

The [CPU record](runs/shared-batch-coefficient-fit/cpu.json) qualifies one position-dependent
coefficient table broadcast across sequences: batch-mean initialization, mean gradients
accumulated before one factorial-coordinate Adam update, microbatch invariance,
frozen source parameters, and exact checkpoint resume. Existing generator and
independent-coefficient checks also pass. The [H100 qualification](runs/shared-batch-coefficient-fit/gpu-h100.json)
passes original coefficient replay over all 256 sequences, repeated shared-table
forward execution, an improving mean-gradient update, an unchanged source tree,
and exact checkpoint resume for the 15-layer degree-eight model.

## Static polynomial attention with cumulative FoX

Learned direct static coefficients take the cumulative input gate on the blocked backend, with RoPE
or no positional encoding, through the query-coefficient decay operation with every query's
coefficients equal. `tests/test_static_forget_gate.py` checks the forward values and the Q, K, V,
coefficient, input and gate-parameter gradients against an independent FP64 oracle with an explicit
product of gates, and checks parameter ownership, the zero-decay limit, causal masking, diagnostics and
the rejected configurations; acceptance runs it for every such recipe:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest -q \
  scaffolds/transformer/tests/test_static_forget_gate.py -m 'not gpu'
```

Its GPU case compares production at the default matmul precision with the host FP64 oracle at 2,048
tokens, BF16 activations and width 64. It passed on one H100 PCIe with the locked CUDA 12 environment
at source `a7c6831c`, with relative errors below 1.8% for the output and every gradient. On CPU the
current implementation reproduces that source's parameters, loss and gradients bitwise for the recipe
of `r17fd7f040bff` at a 2-layer width-128 shape.

## Static coefficient fitting

The [CPU record](runs/static-coefficient-fit/cpu.json) qualifies one coefficient
vector per layer and head, shared across all positions and sequences. Checks
cover mean initialization over both axes, equivalence to a constant positional
table, direct full-batch gradients, one factorial-coordinate Adam update after
accumulation, a frozen source tree, and exact resume for single-sequence and
batch fitting. Existing coefficient-fitting modes retain their checks. Each GPU
execution must pass its forward, update, and resume qualification before fitting.

The [single-sequence H100 qualification](runs/static-coefficient-fit/gpu-h100-single.json)
passes original replay, repeated static-table forward, an improving gradient update,
frozen source parameters, and exact checkpoint resume at shape `[15,14,9]`.

The [256-sequence H100 qualification](runs/static-coefficient-fit/gpu-h100-batch.json)
also passes with eight-sequence microbatches and one mean-gradient Adam update
per full batch, using the same static tensor shape.

## Exact fixed-linear Mosaic attention

On 2026-11-02, all 78 existing fixed-polynomial and configuration CPU checks
passed for the fixed-coefficient path. They cover forward/QKV gradients,
coefficients absent from trainable parameters, unchanged initialization, and
constants preserved through training and checkpoint resume.

`test_mosaic_unnormalized.py::test_fixed_linear_full_context_and_head_rmsnorm_match_independent_fp64`
adds an exact `[0,1]` test at 2,048 tokens and key/value width 64. It uses BF16
operands prepared with canonical QK RMSNorm, per-head scaling and RoPE; an
independent host NumPy FP64 reference checks signed linear attention followed by
head RMSNorm, including analytic Q/K/V and output-gain gradients. Its two CPU
cross-checks against float64 automatic differentiation passed, including inputs
small enough for the RMSNorm epsilon to dominate. The tolerance is the existing
4% composition tolerance. The test is automatically selected for Mosaic
unnormalized-attention GPU acceptance. [H100 qualification](runs/reference-oracle-recheck/fixed-linear-gpu.json)
passed in job 233097 on 2026-11-02 with the original 4% tolerance.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  scaffolds/transformer/tests/test_fixed_polynomial.py \
  scaffolds/transformer/tests/test_fixed_polynomial_config.py -m 'not gpu'
# In a supported GPU allocation, with that allocation's environment:
python -m pytest -q scaffolds/transformer/tests/test_mosaic_unnormalized.py \
  -k fixed_linear_full_context
```

## Query-dependent distance decay

The blocked backend supports `attention_forget_gate_mode: query_distance`:
the existing FP32 log gate multiplies the already evaluated polynomial by
`exp((t-i) * log_alpha_t)`. Cumulative FoX remains the historical default.

CPU qualification on 2026-11-02 passed 15 new scaffold checks and 28 experiment
recipe checks. These cover an independent float64 forward and all operand/gate
gradients, padded blocks, query-local dependence, zero and strong decay, invalid
gates, legacy defaults, identical initialization, factorial coefficients, and
the actual AdamW decay update. Three existing cumulative-FoX regressions passed;
the run-record suite passed all 63 checks, and 104 existing acceptance tests passed.

The new GPU check uses 2,048 tokens, 64-channel BF16 Q/K/V, and FP32 coefficients
and attenuation. It compares the output and all five operand gradients to
independent host NumPy FP64 algebra and analytic gradients, with relative L2
error below 0.012 for each. Small-case NumPy outputs and gradients are checked
against CPU float64 automatic differentiation, including a length-one sequence
whose gate gradient is exactly zero.
[H100 qualification](runs/reference-oracle-recheck/query-distance-gpu.json) passed in job
233097 on 2026-11-02 with the original 0.012 tolerance. Production is gated on
this test, the complete
shared acceptance suite, an exact full-parameter initialization hash match to
the control, and a real-model training/evaluation/checkpoint-resume pilot.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  scaffolds/transformer/tests/test_query_distance_decay.py \
  experiments/transformer_pp/tests/test_query_distance_decay_acceptance.py -m 'not gpu'
# Run the GPU test inside the allocated job using its selected CUDA environment:
python -m pytest -q scaffolds/transformer/tests/test_query_distance_decay.py -m gpu
```

## Score MLP on the blocked and fused backends

`attention_score_mlp_width` replaces exp in unnormalized attention with
`J` SiLU units per layer and head (`attention/score_mlp.py`). It runs on the
blocked backend and, without the forget gate, on the fused Pallas Triton kernel
(`attention/backends/fused_score_mlp.py`). XLA cannot compile the dense FP32 HIGHEST
definition at 2,048 tokens on H100 (an internal `layout_normalization.cc`
`ReshapeIsBitcast` RET_CHECK), so the GPU checks compare each implementation with
host NumPy FP64 algebra and analytic gradients on the same BF16 values. CPU tests
check that oracle against float64 autodiff of the dense definition (both forms,
`J` = 1, 4, 8, with and without the gate, lengths 1 and 19).

### Score MLP on the blocked backend

[H100 qualification](runs/score-mlp/gpu-qualification.json) passed in job 234802
on 2026-11-04 (2,048 tokens, 4 heads, 64-channel BF16 Q/K/V, FP32 units; relative
L2 error below 0.012 for the output and every gradient):

| Case | Output | dQ | dK | dV | da, db, dw (dL) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kernel, `J` = 4 | 1.7e-3 | 1.7e-3 | 4.6e-3 | 4.6e-3 | ≤ 2e-7 |
| kernel, `J` = 8 | 1.7e-3 | 1.7e-3 | 4.6e-3 | 4.6e-3 | ≤ 3e-7 |
| exp residual, `J` = 4 | 1.7e-3 | 1.7e-3 | 4.6e-3 | 4.6e-3 | ≤ 5e-7 |
| exp residual, `J` = 8 | 1.7e-3 | 1.7e-3 | 4.6e-3 | 4.6e-3 | ≤ 4e-7 |
| exp residual, `J` = 8, gated | 1.7e-3 | 1.6e-3 | 2.3e-3 | 2.4e-3 | ≤ 4e-7 |

[Throughput](runs/score-mlp/throughput.json) at the Transformer++ attention shape
(micro-batch 16, 14 heads, 2,048 tokens), forward and backward of one layer on an
H100 PCIe (job 234788):

| Kernel | ms | Attention s/update |
| --- | ---: | ---: |
| Plain blocked exp | 40.8 | 9.8 |
| Exp residual, `J` = 4 | 160.2 | 38.5 |
| Exp residual, `J` = 8 | 187.3 | 45.0 |

### Score MLP on the fused backend

CPU tests (`test_score_mlp_fused.py`, 102 cases) run the kernel's three Pallas programs
in the interpreter with exact FP32 products against float64 autodiff of the dense
definition: both forms, `J` = 1, 2, 3, 4, 8, 1 to 160 tokens across 32-token query and
key tiles, Q/K widths 1 to 128 (including 17) with other value widths, FP32 and BF16
operands, two batches and heads. The output and every gradient are compared whole and
per head, `a` and `b` also per unit, and each `dw_hj` against the absolute sum of its
terms (in one case a unit's terms cancel to 1/1,219 of that sum). They also check exact
causality, that `a = 0` gives exp with exactly zero `db` and `dw`, NaN row poisoning,
input validation, selection, and a two-layer model against the blocked model.

Masked and padded entries are evaluated at score zero and get zero dS in the backward,
as in the VJP of the causal mask. Three VJP checks, interpreted and compiled, cover them:
a future key in the query's own tile whose FP32 score or exp overflows, against float64
and the blocked path; `f(0) = 1 + sum_j a_tj SiLU(b_tj)` overflowing FP32 at masked and
padded entries while every causal value is finite, against the host FP64 definition and
the blocked path; and a NaN unit, whose NaN entries in every gradient are the blocked
path's, with everything outside its row's reach finite and equal to the blocked path's.

[H100 qualification](runs/score-mlp/fused-gpu-qualification.json) passed in job 235164
on dogo (H100 SXM) at commit `b1712a0c`: all 41 GPU tests of both score-MLP test files,
36 of them for the fused kernel. At 2,048 tokens, 4 heads, and 64-channel BF16 Q/K/V,
errors against the FP64 oracle are below 0.012 (output, dQ, dK, dV) and 1e-4 (`a`, `b`,
`w`) whole, per head, and per unit:

| Case | Output | dQ | dK | dV | da, db, dw |
| --- | ---: | ---: | ---: | ---: | ---: |
| kernel, `J` = 4 | 1.7e-3 | 1.7e-3 | 1.7e-3 | 1.7e-3 | ≤ 3e-7 |
| kernel, `J` = 8 | 1.7e-3 | 1.7e-3 | 1.7e-3 | 1.7e-3 | ≤ 4e-7 |
| exp residual, `J` = 4 | 1.7e-3 | 1.7e-3 | 1.7e-3 | 1.7e-3 | ≤ 5e-7 |
| exp residual, `J` = 8 | 1.7e-3 | 1.7e-3 | 1.7e-3 | 1.7e-3 | ≤ 4e-7 |

- Against the blocked path on the same GPU: output and dQ ≤ 9e-4, dK and dV ≤ 4.9e-3
  (the blocked path's own dK and dV errors against the oracle are 4.6e-3), units ≤ 3e-7.
- Two calls return bitwise identical outputs and gradients.
- The exp residual at `a = 0` matches the fused exp kernel to 5e-5 (its 64x64 tiles
  sum in another FP32 order), with `db` and `dw` exactly zero.
- 131 tokens, two batches, three heads; Q/K and value widths 33/17 and 72/40 at `J` = 3,
  and 128/128 at `J` = 3 and 8: FP32 ≤ 1.8e-6; BF16 ≤ 1.8e-3, units ≤ 8e-7.
- Masked entries (FP32): the overflowing in-tile score or exp, ≤ 4e-7 from the FP64
  oracle; the overflowing `f(0)`, finite, within 1.1e-7 of the FP64 definition and equal
  to the blocked path; the NaN unit, NaN exactly where the blocked path has NaN, the rest
  within 5.1e-7 of it.
- Model shape (micro-batch 16, 14 heads, 2,048 tokens, `J` = 8): finite, with the same
  agreement with the blocked path.
- A two-layer model at 256 tokens (FP32, HIGHEST precision): the same loss, parameter
  gradients within 2.9e-5 of the blocked model.

[Throughput](runs/score-mlp/fused-throughput.json) at the Transformer++ attention shape
(micro-batch 16, 14 heads, 2,048 tokens), median forward and backward of one layer, both
backends on one H100 SXM (dogo, job 235165, commit `b1712a0c`); attention time per update
is 15 layers times 16 micro-batches:

| Kernel | Fused ms | Fused s/update | Blocked ms | Blocked s/update |
| --- | ---: | ---: | ---: | ---: |
| Plain exp | 3.3 | 0.79 | 25.8 | 6.2 |
| kernel, `J` = 4 | 15.5 | 3.7 | 97.2 | 23.3 |
| kernel, `J` = 8 | 33.7 | 8.1 | 119.1 | 28.6 |
| Exp residual, `J` = 4 | 16.8 | 4.0 | 104.3 | 25.0 |
| Exp residual, `J` = 8 | 34.7 | 8.3 | 126.2 | 30.3 |

The kernel uses 32x32 tiles, 4 warps, and 2 stages. In the record's sweep this is the
fastest setting for the exp residual at `J` = 8 and within 5% of it for the kernel form
at `J` = 8 (3 stages: 32.2 ms); at `J` = 4, 64x32 tiles are 8% faster for the exp
residual (15.5 ms) and 12% for the kernel form (13.5 ms). With 64x64 tiles the `J` = 8
backward is 3-4x slower (exp residual `J` = 8: 116.7 against 28.3 ms), consistent with
register pressure from the per-unit tiles.

Repeat from a committed worktree on Kennel: the CPU tests, then each GPU step within its
own H100 allocation, which writes that step's record:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  scaffolds/transformer/tests/test_score_mlp_attention.py \
  scaffolds/transformer/tests/test_score_mlp_fused.py -m 'not gpu'
mkdir -p scaffolds/transformer/validation/logs/score-mlp
for step in tests benchmark; do
  sbatch --partition=debug --gres=gpu:h100:1 --exclude=collie,samoyed,doodle,terrier,shiba \
    --output=scaffolds/transformer/validation/logs/score-mlp/%j.log \
    scaffolds/transformer/validation/qualify_score_mlp_fused.sbatch "$(git rev-parse HEAD)" "$step"
done
```

## Continuing learned models

The [CPU qualification](runs/learned-continuation/cpu.json) checks that static and
query-coefficient continuations preserve the complete parent model and Adam state,
match ordinary continued updates, and resume exactly after interruption. Model changes
and a separate new-parameter learning rate are rejected in this mode. Full recipe GPU
acceptance remains mandatory before training.

## Blocked softmax attention at 2,048 tokens

Which plain JAX/XLA (non-fused) causal softmax should become the blocked basis for the
blocked-only mechanisms at 2,048 tokens, keeping FP32 scores and softmax arithmetic,
production's NaN rows for nonfinite causal scores, and a 15-layer model on one 80 GB H100?
The candidates are in [`blocked_attention_benchmark.py`](blocked_attention_benchmark.py);
no scaffold code or default changed. Measured on 2026-11-04 on `dogo` (H100 SXM, driver
560.35.05, JAX 0.10.0, cuDNN 9.22) at commit `8df8ce2f`, all in job 235720:
[kernel record](runs/blocked-attention/kernel-h100-235720.json) and
[model record](runs/blocked-attention/model-h100-235720.json).

| Candidate | Design |
| --- | --- |
| `production-bN` | `blocked_dot_product_attention(query_block_size=N)`: checkpointed scan of N-query blocks, each against all 2,048 keys |
| `lean-f32-scan-*-b128` | production's scan with fewer passes: a max shift without a gradient, output (P V)/l |
| `lean-{f32,bf16}-skip-{ckpt,nockpt}-bN` | the same strips unrolled in Python, so strip b multiplies only keys [0, end_b); JAX autodiff, with or without `jax.checkpoint` per strip |
| `lean-f32up-*` | the same on FP32-upcast Q/K/V, so the strips' dK/dV sum in FP32 |
| `flash-{f32,bf16}-{rows,cols}-bN` | custom VJP over the unrolled strips: the forward keeps O and the row log-sum-exp; the backward recomputes S and P = exp(S - lse), dS = P (dP - rowsum(dO O)), over query strips (rows; sums dK, dV) or key strips (cols; sums dQ) |
| `flash-{f32,bf16}-saved-bN` | the same, keeping each strip's normalized weights from the forward instead of recomputing them |
| `dense-f32scores-bf16w` | one dense block: FP32 scores and softmax, BF16 weights, no checkpoint |
| `dense-manual` | `softmax_attention(backend="manual")`: BF16 scores and weights, no checkpoint |
| `cudnn` | `softmax_attention(backend="cudnn")`, the production softmax path |

`f32`: FP32 softmax weights enter the products with upcast V, dO, Q and K: production's
precision (FP32 scores and softmax, TF32 products; see below), though the candidates compute
(P V) / l where production computes (P / l) V. `bf16`: P (and dS) are rounded to BF16 before
the products, with FP32 accumulation, and delta = rowsum(dO O) uses O rounded to BF16, as in
cuDNN and FlashAttention. Every blocked candidate returns NaN for any row with a nonfinite
causal score. Two compiler behaviours shape the custom backward:

- It starts with an optimization barrier on its residuals and cotangent. Without it XLA merges
  the backward's score products with the forward's and keeps every layer's scores alive:
  `flash-f32-rows-b512` without the barrier (commit `3782b98e`, job 235223,
  [record](runs/blocked-attention/model-h100-235223-no-barrier.json)) ran out of memory at
  micro-batch 16 and peaked at 34.1 GB at 8.
- With the default `xla_allow_excess_precision`, XLA GPU removes an f32 -> bf16 -> f32
  conversion pair when the second conversion consumes the first, so such a pair does not
  reliably round. Residuals stored in BF16 and upcast in the backward (O for delta, saved
  weights) are therefore rounded explicitly with `reduce_precision`, which XLA keeps; a
  compile-only H100 test checks both.

### Accuracy

Relative L2 error x 1e3 against a host NumPy FP64 forward and analytic VJP, the same fixed
reference for every candidate whatever its precision. Accuracy and nonfinite-input programs run
at [2, 2048, 4, 64] (BF16 inputs; sharp: Q x 4, score standard deviation 4), compiled without
GEMM autotuning (`xla_gpu_autotune_level=0`); the timed programs use XLA's defaults at
[16, 2048, 14, 64] and are scored as well (second table). The oracle matches float64 JAX
autodiff to 6.8e-16 ([CPU check](runs/blocked-attention/oracle-cpu.json)). The first row is the
oracle rounded to BF16, the best a BF16 result can be.

| Variant | Output | dQ | dK | dV | Sharp output | Sharp dQ | Sharp dK | Sharp dV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle rounded to BF16 | 1.61 | 1.65 | 1.65 | 1.64 | 1.66 | 1.66 | 1.66 | 1.65 |
| `cudnn` | 2.04 | 2.48 | 2.43 | 2.30 | 1.73 | 3.50 | 3.23 | 2.27 |
| `dense-manual` | 3.74 | 4.15 | 4.18 | 3.75 | 10.07 | 12.43 | 12.51 | 10.06 |
| `dense-f32scores-bf16w` | 2.05 | 2.65 | 2.64 | 2.09 | 1.77 | 3.73 | 3.52 | 1.77 |
| `production-b128` | 1.62 | 1.67 | 3.27 | 3.08 | 1.67 | 1.68 | 4.04 | 3.96 |
| `production-b512` | 1.62 | 1.67 | 2.58 | 2.51 | 1.67 | 1.68 | 2.79 | 2.77 |
| `production-b2048` | 1.62 | 1.67 | 1.67 | 1.65 | 1.67 | 1.68 | 1.68 | 1.66 |
| `lean-f32-skip-ckpt-b512` | 1.62 | 1.68 | 2.59 | 2.51 | 1.66 | 1.73 | 2.81 | 2.77 |
| `lean-f32up-skip-ckpt-b512` | 1.62 | 1.68 | 1.68 | 1.66 | 1.66 | 1.73 | 1.72 | 1.67 |
| `lean-bf16-skip-nockpt-b512` | 2.05 | 2.65 | 3.29 | 2.81 | 1.77 | 3.73 | 4.17 | 2.83 |
| `flash-f32-{rows,cols}`, every strip size | 1.62 | 1.67 | 1.67 | 1.65 | 1.66 | 1.68 | 1.68 | 1.66 |
| `flash-bf16-{rows,cols}`, every strip size | 2.05 | 2.48 | 2.44 | 2.30 | 1.77 | 3.51-3.52 | 3.24 | 2.27 |
| `flash-f32-saved-b512` | 1.62 | 1.67 | 1.67 | 1.65 | 1.67 | 1.70 | 1.69 | 1.66 |
| `flash-bf16-saved-b512` | 2.30 | 2.89 | 2.87 | 2.30 | 2.27 | 3.96 | 3.68 | 2.27 |

- Production's blocked backward sums each block's dK and dV in BF16: autodiff returns a BF16
  cotangent for K and V in every scan iteration. The error grows with the number of blocks
  (16 blocks: 3.3e-3, twice the floor and above cuDNN; one block: the floor). Autodiff over
  unrolled BF16 strips inherits this; FP32-upcast operands or the custom VJP remove it.
- The same sums apply to every kernel trained through `blocked_dot_product_attention`, whose
  query blocks are always 128. Against an FP32 reference of the same kernel (one block,
  `HIGHEST` matmul precision) on the same BF16 inputs at [2, 2048, 4, 64], dK and dV error
  relative to the BF16 floor:

  | Kernel | 16 blocks of 128: dK / dV | One block: dK / dV |
  | --- | ---: | ---: |
  | softmax | 1.96x / 1.85x | 1.01x / 1.01x |
  | query polynomial m = 4, unnormalized | 2.79x / 2.79x | 1.01x / 1.01x |
  | normalized learned polynomial m = 2 | 1.88x / 1.79x | 1.01x / 1.01x |

  Output and dQ stay at 1.01x. These are the numerics of every BF16 run trained on the
  blocked scan: in `attention_scaling`, the normalized learned m = 1-4 and the query or
  query-key coefficient runs; in Transformer++, every forget-gate run (softmax, exponential and
  query polynomials; the 4,096-token runs have 32 blocks, not measured). PaTH, GDN and GLA have
  separate blocked code.
- JAX's default precision runs FP32 x FP32 products as TF32 on H100: an FP32 matmul has
  2.9e-4 relative error (2.2e-7 at `HIGHEST`), and with FP32 inputs production and every
  FP32-weight candidate have 1.6-3.1e-4. "FP32 weights" are therefore TF32 weights; with BF16
  outputs this stays below the output rounding (`flash-f32` is at the floor).
- `flash-bf16` is within 2% of cuDNN's errors (output 2.05 vs 2.04, sharp output 1.77 vs
  1.73; every gradient within 0.5%); the BF16-score textbook path is 6-8x the floor on sharp
  inputs. `flash-bf16-saved` reuses the normalized weights rounded to BF16 in dS and is less
  accurate than cuDNN (dQ 2.89 vs 2.48, sharp output 2.27 vs 1.73).

The timed programs, scored on their first sequence (other inputs, so their own floor):

| Variant | Output | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| Oracle rounded to BF16 | 1.61 | 1.66 | 1.67 | 1.66 |
| `cudnn` | 2.03 | 2.48 | 2.45 | 2.31 |
| `production-b128` | 1.62 | 1.67 | 3.24 | 3.10 |
| `flash-f32-rows-b512` | 1.61 | 1.67 | 1.68 | 1.67 |
| `flash-bf16-rows-b512` | 2.04 | 2.48 | 2.45 | 2.31 |
| `flash-f32-saved-b512` | 1.62 | 1.68 | 1.68 | 1.67 |
| `flash-bf16-saved-b512` | 2.27 | 2.92 | 2.90 | 2.31 |
| `lean-f32up-skip-ckpt-b512` | 1.61 | 1.69 | 1.70 | 1.68 |

Relative to their floors, the timed programs' errors are within about 1% of the accuracy
programs' for these variants (production's dK: 1.8%) and within 6% for all 43.

### Nonfinite inputs

One value injected at token 1,000 of one head, on the GPU at [2, 2048, 4, 64]
([CPU tests](test_blocked_attention_benchmark.py) check the same against production). Entries
are the nonfinite output rows of that head; every other head stays finite, and every variant's
gradients are nonfinite in every case.

| Injection | Production's rule | `production-*`, scans, single block | Causal-skip strips | `cudnn` | `dense-manual` |
| --- | --- | --- | --- | --- | --- |
| Q NaN | 1000 | 1000 | 1000 | 1000 | 1000 |
| K NaN | >= 1000 | >= 1000 | >= 1000 | >= 1000 | >= 1000 |
| V NaN or +inf | >= 1000 | all | >= start of its strip (896, 768, 512, 0 for 128, 256, 512, 1,024 queries) | >= 896 | all |
| K = -inf (scores -inf) | >= 1000 | >= 1000 | >= 1000 | none | none |
| Overflowing score (1e40/8) | 1000 | 1000 | 1000 | 1000 | 1000 |
| Huge finite scores (1e20 in every channel) | 1000 | 1000 | 1000 | 1000 | 1000 |

Masked weights are exact zeros and 0 x NaN = NaN, so a bad V value also poisons the earlier
rows that share its product: every row when each block multiplies all keys, the rows of its
strip with causal skipping. cuDNN and the textbook path treat a -inf score as a zero weight;
the blocked candidates keep production's rule. On CPU, rows with huge finite scores
(|s| ~ 1e20) also became nonfinite after a fused multiply-add rounded S x scale differently in
two fusions, identically in production and the candidates.

### Speed and compiled memory, one layer

[16, 2048, 14, 64] BF16, median of 20 synchronized calls after 3 warmups. Temp is the
compiler's estimate for one layer's forward + backward; it excludes what a model holds across
layers (each layer's residuals until its backward). One Transformer++ update runs 240 layer
micro-batches (15 layers x 16), so attention s/update = forward + backward ms x 0.24.

| Variant | Forward ms | Fwd + bwd ms | Temp GB | Attention s/update |
| --- | ---: | ---: | ---: | ---: |
| `cudnn` | 0.52 | 1.59 | 0.12 | 0.38 |
| `dense-manual` | 3.55 | 11.40 | 5.64 | 2.74 |
| `dense-f32scores-bf16w` | 6.73 | 14.49 | 5.94 | 3.48 |
| `production-b128` | 9.28 | 37.37 | 1.01 | 8.97 |
| `production-b512` | 8.42 | 35.56 | 3.28 | 8.53 |
| `production-b2048` | 7.99 | 22.76 | 12.34 | 5.46 |
| `lean-f32-scan-ckpt-b128` | 8.14 | 31.31 | 1.01 | 7.51 |
| `lean-f32-skip-ckpt-b256` | 4.99 | 15.46 | 2.46 | 3.71 |
| `lean-f32-skip-nockpt-b512` | 4.94 | 11.46 | 5.17 | 2.75 |
| `lean-f32up-skip-ckpt-b512` | 5.33 | 16.52 | 4.55 | 3.97 |
| `lean-bf16-skip-nockpt-b512` | 3.93 | 9.53 | 3.82 | 2.29 |
| `flash-f32-cols-b512` | 5.14 | 13.61 | 2.76 | 3.27 |
| `flash-f32-rows-b512` | 5.06 | 13.69 | 2.44 | 3.29 |
| `flash-f32-saved-b512` | 4.87 | 13.21 | 3.32 | 3.17 |
| `flash-bf16-cols-b512` | 3.89 | 10.17 | 2.98 | 2.44 |
| `flash-bf16-rows-b512` | 4.02 | 10.33 | 3.32 | 2.48 |
| `flash-bf16-saved-b512` | 4.25 | 9.74 | 2.25 | 2.34 |

- Strip size: `flash-f32-cols` takes 14.71, 13.41, 13.61, 15.17 and 19.87 ms at 128, 256,
  512, 1,024 and 2,048 queries; `flash-bf16-cols` 11.75, 10.65, 10.17, 11.83 and 15.40 ms.
  Small strips pay a fixed cost per strip; large ones compute more of the masked triangle.
  Rows and cols are within 2% at 256 and 512 queries.
- Keeping the weights (`saved`) is 1-9% faster than recomputing them (`rows`: 1.2% and 3.5% in
  FP32 at 256 and 512 queries, 8.9% and 5.8% in BF16), but holds each strip's normalized
  weights from a layer's forward to its backward: 2.35 GB per layer in FP32 and 1.17 GB in BF16
  at 512 queries and micro-batch 16, which a 15-layer model keeps for every layer and this
  temp does not show (full model below).
- Profiles of 3 calls: the GPU idles 0.3-0.8 ms per call, and GEMMs take about half the
  strip variants' kernel time (41-51%) and a third of production's.

  | Variant | GEMM ms | Other ms | Est. GEMM GB | Est. other GB | GEMM TB/s | GEMM TFLOP/s | Other TB/s |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `production-b128` | 12.25 | 24.46 | 34.4 | 73.0 | 2.81 | 69 | 2.99 |
  | `flash-f32-cols-b128` | 7.13 | 6.96 | 20.4 | 19.6 | 2.87 | 63 | 2.81 |
  | `flash-f32-rows-b512` | 6.42 | 6.47 | 18.7 | 19.4 | 2.91 | 82 | 3.00 |
  | `flash-bf16-rows-b512` | 4.72 | 5.09 | 13.5 | 15.6 | 2.87 | 111 | 3.05 |
  | `flash-bf16-saved-b512` | 3.81 | 5.47 | 11.0 | 15.2 | 2.88 | 118 | 2.77 |
  | `lean-f32up-skip-ckpt-b512` | 7.94 | 8.00 | 22.0 | 24.2 | 2.77 | 76 | 3.02 |

  Times are per call over all kernels, by kernel name (cuDNN's 1.27 ms are all its own fused
  kernels). The estimate sums each kernel's operand and result bytes and its GEMM FLOPs from
  the optimized HLO (slices and in-place updates at their size, loop bodies times their trip
  counts); its GEMM FLOPs equal the analytic count. Both classes move about 2.8-3.1 TB/s, near
  the H100 SXM's 3.35 TB/s, while the GEMMs run at 63-118 TFLOP/s of the TF32 and BF16 tensor
  cores' 495 and 990: plausibly bandwidth-bound throughout (cache reuse is not modeled).
- First compiles of a strip size, forward + backward: 102-122 s at 128 queries, 53-81 s at
  256, 27-32 s at 512 (GEMM autotuning of each strip shape; later variants reuse it); the scans
  and cuDNN take 10 s or less.
- Repeated at the end of the job: cuDNN 1.49 ms (first 1.59), production 37.32 ms (37.37).

### Full model

The Transformer++ softmax recipe
([`r9040f12f3d80`](../../../experiments/transformer_pp/runs/r9040f12f3d80/config.yaml), 15 layers)
on synthetic tokens: one forward + backward micro-step of the scaffold's
`training.loop.loss_and_grads` (no optimizer update), with softmax attention routed to each
candidate inside the benchmark; `production-b128` and `cudnn` use the model's own backends.
Each case runs in a fresh process. Peak is the allocator peak including the train state
(parameters and AdamW moments), under the default allocator limit of 63.8 GB;
s/update = micro-step x 256 / micro-batch, median of 10 after 2 warmups.

| Variant | Micro-batch | Micro-step ms | s/update | Peak GB | Compile s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cudnn` | 16 / 8 / 4 | 107.3 / 59.1 / 33.3 | 1.72 / 1.89 / 2.13 | 26.3 / 15.6 / 9.9 | 29-32 |
| `production-b128` | 16 / 8 / 4 | 638.3 / 335.7 / 179.4 | 10.21 / 10.74 / 11.48 | 27.8 / 15.9 / 9.8 | 42-50 |
| `flash-f32-rows-b512` | 16 / 8 / 4 | 277.2 / 147.4 / 80.5 | 4.44 / 4.72 / 5.15 | 29.1 / 17.0 / 10.5 | 60-67 |
| `flash-bf16-rows-b512` | 16 / 8 / 4 | 232.1 / 120.9 / 67.7 | 3.71 / 3.87 / 4.33 | 27.8 / 16.4 / 10.3 | 59-65 |
| `lean-f32up-skip-ckpt-b512` | 16 / 8 / 4 | 323.5 / 171.9 / 95.0 | 5.18 / 5.50 / 6.08 | 34.5 / 19.7 / 12.2 | 66-74 |
| `flash-f32-saved-b512` | 16 / 8 / 4 | OOM / 145.6 / 79.4 | - / 4.66 / 5.08 | - / 34.1 / 19.1 | 61-72 |
| `flash-bf16-saved-b512` | 16 / 8 / 4 | 221.9 / 117.9 / 64.6 | 3.55 / 3.77 / 4.14 | 44.7 / 24.9 / 14.5 | 62-70 |
| `lean-bf16-skip-nockpt-b512` | 16 / 8 / 4 | OOM / 117.6 / 64.5 | - / 3.76 / 4.13 | - / 42.9 / 23.3 | 80-97 |

Each completed routed case records the calls its candidate received while the step was
traced: 15, one per layer. The timings agree: at micro-batch 16, each micro-step differs from
production's by 15 times the one-layer forward + backward difference, to within 2%, and
compile times and peaks differ from production's:

| Variant | Micro-step minus production's, ms | 15 x one-layer difference, ms |
| --- | ---: | ---: |
| `cudnn` | -531.0 | -536.7 |
| `flash-f32-rows-b512` | -361.1 | -355.2 |
| `flash-bf16-rows-b512` | -406.2 | -405.5 |
| `lean-f32up-skip-ckpt-b512` | -314.8 | -312.6 |
| `flash-bf16-saved-b512` | -416.5 | -414.4 |

Losses (10.5088 at micro-batch 16) and gradient norms (2.811) agree to four digits across
cases, a sanity check of each candidate's correctness that would also hold without routing.

Saved weights stay alive from each layer's forward to its backward. At micro-batch 8,
`flash-f32-saved-b512` peaks 17.1 GB above `flash-f32-rows-b512` (15 layers x 1.17 GB of FP32
weights: 17.6 GB), and at 16 it runs out of memory (52.7 GiB of temporaries);
`flash-bf16-saved-b512` fits at 16 with 44.7 GB, 16.9 GB above `flash-bf16-rows-b512`, for 4%.
Without checkpoints, autodiff keeps every layer's weights: at micro-batch 16 its 52.4 GiB of
temporaries did not fit.

### Recommendation

- **Blocked basis: `flash-f32-rows-b512`.** The custom VJP over four unrolled 512-query
  strips with causal key prefixes; the forward keeps O and the row log-sum-exp, and the
  backward recomputes each strip behind the barrier and sums dK and dV in FP32. It keeps
  production's precision (FP32 scores and softmax, TF32 products; it normalizes after the
  product) and NaN-row rule, puts the output and every gradient at the BF16 rounding floor
  (more accurate than production's BF16 block sums and than cuDNN), fits micro-batch 16 at
  29.1 GB, and cuts the forward + backward of an update from 10.21 s to 4.44 s, 2.3x (cuDNN:
  1.72 s).
- **cuDNN numerics: `flash-bf16-rows-b512`**, the same code with BF16 weights: 3.71 s per
  update, within 2% of cuDNN's errors.
- 256 and 512 queries per strip are within 5% in speed; 512 compiles in about half the time.
  Rows and cols tie; rows shares the forward's layout.
- **Without a hand-written backward: `lean-f32up-skip-ckpt-b512`**, checkpointed autodiff over
  the same strips on FP32-upcast operands (at the floor; 5.18 s per update, 34.5 GB at
  micro-batch 16), for mechanisms whose derivative is impractical by hand, such as the score MLP.
- Not recommended: production's scan at any block size (BF16 dK/dV sums; one dense block
  avoids them at 12 GB per layer), autodiff without checkpoints (out of memory at
  micro-batch 16), the dense paths, and saved weights (`flash-f32-saved` runs out of memory
  at micro-batch 16 for a 1-4% gain; `flash-bf16-saved` is less accurate than cuDNN).
- The remaining 2.0-2.6x gap to cuDNN per update is plausibly memory bandwidth: plain XLA
  writes every [strip x keys] score, weight and score-gradient block to HBM and reads it back,
  at an estimated 2.8-3.1 TB/s.

Integration without changing historical behaviour:

1. Move the strip forward and custom VJP into a scaffold backend beside
   `blocked_dot_product_attention`, which stays unchanged, with these CPU tests and a GPU
   FP64-oracle test at 2,048 tokens.
2. Select it with a new recipe field whose default is the existing scan (for example
   `model.attention_blocked_implementation: scan | strips`, with strip size and weight
   precision). Historical recipes and resumptions keep their numerics, including the BF16
   dK/dV sums; new runs opt in, or get it from `scaffolds.records create` once qualified.
3. Keep `return_diagnostics=True` on the existing blocked function, as the fused paths do.
4. Port each blocked-only mechanism into the strip template: its kernel, normalization and
   invalid-row rules in the forward, and in the backward, behind the barrier:
   - unnormalized kernels: dS = f'(S) dP;
   - normalized kernels: dS = (f'(S) / l) (dP - delta), with delta = rowsum(dO O) and l the
     row sum (softmax: P (dP - delta) with P = exp(S - lse));
   - rational kernels p_a(S) / L with L = sum_j p_b(S_j): dS = (p_a'(S) dP - p_b'(S) delta) / L;

   plus its coefficient gradients. Write each intended rounding of a value that is upcast
   again with `reduce_precision` (`_rounded_to`). Check each port's NumPy oracle against
   float64 `jax.vjp`, as `oracle_check` does, and the port against its oracle. Use the
   `lean-f32up` template where the derivative is impractical.
5. Strips assume a static sequence length, and the model compiles in 59-67 s instead of
   42-50 s.
6. Qualify on H100 and B300 (B300 was not measured), then acceptance and a pilot.

### Tests and regeneration

The 138 CPU tests in [`test_blocked_attention_benchmark.py`](test_blocked_attention_benchmark.py)
pass. They check every candidate's output and dQ/dK/dV against the FP64 oracle in float64 (to
1e-11) at T = 1, 32 and 37 with strips of 1, 4, 8, 16 and 64 (ragged last strips), float32
parity with production, BF16 dtypes and rounding, causality, the nonfinite cases above
against production, FP32 dK/dV sums, the backward's barrier, option and shape validation, and
the model routing: a two-layer model routed to the candidates matches the scaffold's blocked
loss and every parameter gradient, with one routed call per layer. They also check the
explicit BF16 rounding (bitwise a conversion round trip, and kept in compile-only H100
programs, where XLA removes a plain conversion pair), the HLO traffic and FLOP estimate (a
synthetic module, and compile-only H100 programs against the analytic GEMM FLOPs), the
profile's class split, and the block-sum check. The compile-only tests skip without the CUDA
plugin.

```bash
PY=scaffolds/transformer/.venv/bin/python
JAX_PLATFORMS=cpu PYTHONPATH=. $PY -m pytest -q -c scaffolds/transformer/pyproject.toml \
  --import-mode=importlib scaffolds/transformer/validation/test_blocked_attention_benchmark.py
JAX_PLATFORMS=cpu PYTHONPATH=. $PY -m scaffolds.transformer.validation.blocked_attention_benchmark \
  oracle-check --output oracle-cpu.json
# One H100 (sbatch --partition=debug,standard --gres=gpu:h100:1 --nodelist=dogo --time=01:30:00
# --cpus-per-task=8 --mem=120G), PYTHONPATH=$PWD JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false,
# both stages in one job (the kernel stage's budget leaves the model stage its time):
$PY -m scaffolds.transformer.validation.blocked_attention_benchmark kernel --output kernel.json \
  --budget-minutes 52 \
  --profile-variants cudnn production-b128 flash-f32-cols-b128 flash-f32-rows-b512 \
    flash-bf16-rows-b512 flash-bf16-saved-b512 lean-f32up-skip-ckpt-b512 \
  --float32-input-variants production-b128 dense-manual flash-f32-rows-b512 flash-bf16-rows-b512 \
    flash-f32-saved-b512 lean-f32up-skip-ckpt-b512 lean-f32-skip-ckpt-b512
$PY -m scaffolds.transformer.validation.blocked_attention_benchmark model --output model.json \
  --variants cudnn production-b128 flash-f32-rows-b512 flash-bf16-rows-b512 \
    lean-f32up-skip-ckpt-b512 flash-f32-saved-b512 flash-bf16-saved-b512 \
    lean-bf16-skip-nockpt-b512 \
  --micro-batches 16 8 4 --warmup 2 --iterations 10
# All tables of the records, including the 43 kernel variants:
$PY -m scaffolds.transformer.validation.blocked_attention_benchmark summarize \
  --kernel-record kernel.json --model-record model.json --output summary.md
```

## Blocked causal strips

`model.attention_blocked_implementation: strips` (scaffold README, "Blocked attention
implementations") evaluates the blocked scan's own per-block arithmetic over causal 512-query
strips with FP32 K/V cotangent sums: the `lean-f32up-skip-ckpt-b512` form above, generalized
to every mechanism of `blocked_dot_product_attention` rather than hand-written per mechanism.
The scan is unchanged (normalized jaxpr text identical to `bc3274e0` in 2,661 configurations).
CPU checks: `tests/test_blocked_strips.py` (float64 equivalence with the scan's code across 25
argument families at ragged strips, NaN rules, BF16 accuracy, a two-layer model, records and
acceptance) and `tests/test_blocked_strips_h100.py` (compile-only H100: no GEMM reads an FP32
value whose BF16 rounding XLA removed, and per-layer memory).

Measured on 2026-11-08 at commit `6ea739f6` on one H100 PCIe (`mastiff`, jobs 239691 and
239696) and one B300 (`akita`, jobs 239692 and 239695) with
[`blocked_strips_qualification.py`](blocked_strips_qualification.py) and
[`qualify_blocked_strips.sbatch`](qualify_blocked_strips.sbatch); records in
[runs/blocked-strips/](runs/blocked-strips/). `test_blocked_strips_h100.py` passed on both GPUs.

**Accuracy.** BF16 inputs at [2, 2048, 4, 64], relative L2 error × 1e3 against the scan's
arithmetic in FP64; the floor is that reference rounded to BF16. The H100 and B300 values agree
to the digits shown.

| Mechanism | Floor dK / dV | Scan dK / dV | Strips dK / dV | Output and dQ (both) |
| --- | ---: | ---: | ---: | ---: |
| FoX softmax | 1.66 / 1.66 | 2.13 / 2.12 | 1.67 / 1.67 | 1.67 / 1.68 |
| FoX exponential | 1.66 / 1.66 | 2.10 / 2.12 | 1.67 / 1.67 | 1.67 / 1.67 |
| FoX query m = 4 | 1.66 / 1.66 | 2.11 / 2.12 | 1.67 / 1.67 | 1.67 / 1.67 |
| Static m = 4, unnormalized | 1.66 / 1.66 | 4.62 / 4.63 | 1.67 / 1.67 | 1.67 / 1.67 |

The strips sit at the floor (1.008×, the TF32 products); the scan's BF16 block sums give
1.27-1.29× under the forget gate's decay and 2.8× without it. FP32 gradients of the log decay
and the coefficients agree with FP64 to below 1e-6 for both.

**Full model.** One forward and backward micro-step of the recipe's 15-layer model
(`training.loop.loss_and_grads`, micro-batch 16, synthetic tokens), each in a fresh process;
s/update = median micro-step × 16.

| Recipe | GPU | Scan s/update | Strips s/update | Speedup | Compile s (scan / strips) | Peak GB (scan / strips) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FoX softmax, no RoPE (`r5adc29a91e3b`) | H100 PCIe | 17.42 | 11.67 | 1.49× | 47 / 74 | 30.1 / 30.8 |
| FoX softmax, no RoPE (`r5adc29a91e3b`) | B300 | 6.06 | 3.83 | 1.58× | 47 / 103 | 29.5 / 30.4 |
| FoX query m = 4, no RoPE (`r82a80d3091e2`) | H100 PCIe | 15.25 | 11.73 | 1.30× | 64 / 86 | 31.6 / 30.3 |
| FoX query m = 4, no RoPE (`r82a80d3091e2`) | B300 | 5.27 | 3.91 | 1.35× | 119 / 93 | 31.0 / 30.3 |

Loss (10.5163 and 10.5100) and gradient norm (3.0317 and 2.0181) agree between the
implementations to five digits. H100 SXM (`dogo`) was fully allocated; the PCIe card is about
1.5× slower in absolute terms, not in the ratio.

### Quartic B-spline score function

`model.attention_kernel_basis: bspline` (scaffold README, "B-spline score function") runs in
the strips' shared per-block arithmetic. Qualified on 2026-11-08 at commit `927995ed` on one
B300 (`akita`, job 239705) and one H100 SXM (`dogo`, job 239706) with the same script: the
spline GPU precision tests (`test_spline_attention.py -m gpu`, static and query) and
`test_blocked_strips_h100.py` passed on both, and the kernel stage puts static and query
spline outputs and gradients at the BF16 floor (1.007-1.008×; the scan's block sums give
2.7×). Full-model micro-steps on the strips, micro-batch 16:

| Recipe | B300 s/update | H100 SXM s/update | Peak GB |
| --- | ---: | ---: | ---: |
| static quartic spline (`r7e4193fcafbd`) | 4.53 | 8.56 | 30.5 |
| query quartic spline + W decay (`rf303a5f94487`) | 5.16 | 9.86 | 32.3-32.5 |
| static m = 4 polynomial on strips (`r4339a245820e`) | 2.75 | 5.42 | 29.0 |
| query m = 4 polynomial on strips (`rca97cc9db8f2`) | 3.32 | 6.57 | 31.0 |

Compilation takes 1.4-3.1 minutes. Initial losses agree with the polynomial controls' to five
digits (both start at T4).

```bash
# One GPU per type, from the worktree root (validation files are outside the frozen check):
sbatch --partition=debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --output=scaffolds/transformer/validation/logs/blocked-strips/%x-%j.log \
  scaffolds/transformer/validation/qualify_blocked_strips.sbatch "$(git rev-parse HEAD)"
```

## Per-order FoX gradient failure investigation (2026-11-07)

The per-order query and query–key gate recipes stopped in GPU acceptance, before
training, in `test_per_order_decay.py::test_gpu_per_order_forward_and_backward`.
That test exercises both gate variants for either recipe. The recorded B300
failure is `CUDA_ERROR_ILLEGAL_ADDRESS` while synchronizing the compiled output
and VJP, after the explicit FP64 reference had completed. Later test failures in
the same process follow the poisoned CUDA context.

Fresh-process H100 diagnostics reproduce the query–key backward failure without
running the reference at all. This establishes that the problem is not confined
to B300 or caused by the reference calculation. Qualification used the unchanged
implementation at `971b2ea6`, JAX 0.10.0, the locked CUDA 12 environment, an H100 on
Kennel `dogo` with driver 560.35.05, and Slurm diagnostic jobs 239099, 239102 and
239108. Inputs match the acceptance test: B=2, T=2048, H=2, dK=dV=64, degree 4,
BF16 Q/K/V, FP32 coefficients/gates and gate input width 7.

| Isolated computation | H100 outcome |
| --- | --- |
| Query-only output and all gradients, highest/default matmul precision | Pass |
| Query-only output and all gradients against the FP64 reference | Pass; maximum relative L2 error 0.0025572 |
| Query–key forward only | Pass |
| Query–key output and all gradients, highest/default precision | Illegal address |
| Query–key output and all gradients, CUDA command buffers disabled | Illegal address |
| Query–key output and all gradients against FP64, priority fusion disabled | Pass; maximum relative L2 error 0.0026623 |

Compute Sanitizer reports **stack overflow** in `loop_add_fusion_5`. The optimized
HLO maps this to accumulation of the query-gate input gradient, shape
`[2,2048,2,5]`, inside the reverse scan, with operation
`transpose(jvp())/while/body/closed_call/add_any`. Five order-specific gradients
are padded, repeatedly inserted into and read from the full gradient buffer, and
added to the scan carry. This is a compiler-generated fusion failure, not a
cuBLAS matrix multiplication failure. Disabling XLA's `priority-fusion` pass
preserves the mathematics and removes the observed crash. The worst numerical
errors above are BF16 gradients; the passing query–key gate/projection gradients
have relative L2 error at most 7.56e-7.

The diagnostic switch is **not** enabled in training. The targeted implementation
fix at `4ef08db2` slices the query-gate time block once per scan iteration, before
selecting each order. Its transpose combines order gradients locally before one
full-buffer update. H100 qualification job **239129** passes both separately
identified query/query–key GPU tests, including initialization and decay masks,
with normal compiler optimizations. The independent query–key FP64 comparison
has maximum relative L2 error 0.0026623; Compute Sanitizer reports **zero errors**
for the fixed backward pass at training's default matmul precision. Recipes,
runtime settings and mathematical definitions are unchanged. Full per-run
acceptance still gates production training on its actual GPU allocation.
The targeted CPU suite passes 73 tests covering per-order and shared endpoint
outputs/gradients, initialization, decay masks and the experiment's recipe guard.
The B300 diagnostic job 239100 was cancelled while pending; it produced no
diagnostic result.

Reproduce inside a correctly configured single-H100 allocation using the
worktree's locked environment (each invocation is a fresh process):

```bash
PY=scaffolds/transformer/.venv/bin/python
PROBE=scaffolds/transformer/validation/per_order_crash_probe.py
JAX_PLATFORMS=cuda PYTHONPATH=. $PY "$PROBE" --phase original
JAX_PLATFORMS=cuda PYTHONPATH=. $PY "$PROBE" --phase forward --endpoint
JAX_PLATFORMS=cuda PYTHONPATH=. $PY "$PROBE" --phase backward --endpoint
XLA_FLAGS=--xla_disable_hlo_passes=priority-fusion JAX_PLATFORMS=cuda PYTHONPATH=. \
  $PY "$PROBE" --phase original --endpoint
```

Raw logs, sanitizer output and HLO dumps are retained under the ignored
`validation/logs/per-order-debug/H100-<job>/` directories.

## Raw-input L2 coefficient source (2026-11-15)

The coefficient generator can read the raw residual before pre-attention RMSNorm,
normalized to unit L2. Q/K/V retain the existing RMSNorm path. The source is an
explicit operand; no inversion of learned channel gains is used.
[Qualification](coefficient_raw_input_l2.json) records the frozen source, commands,
lock, and evidence. Twenty-three CPU checks cover source/gradient isolation,
independent FP64 outputs and gradients, historical defaults, initialization, and
source-correct diagnostics. The pinned experiment adapter passed 85 checks.
[Full-shape initialization](coefficient_raw_input_initialization.json) confirms
200,905,126 parameters and the exact completed-control fingerprints for both
m12 forms, including every shared backbone array.

Debug job 2420 passed all four native Mosaic B300 cases (m12 direct and SoS,
BF16 and BF16X3), checking gradients through the raw coefficient and RMSNorm QKV
branches against independent FP64 references. The locked CUDA 13 environment was
used. The focused kernel cases use 512 tokens; each production allocation still
requires its own full-shape 2,048-token pilot and fresh-process resume. This
qualifies implementation behavior, not a validation-perplexity improvement.

```bash
sbatch --partition=debug --nodelist=akita --gres=gpu:b300:1 \
  --cpus-per-task=8 --mem=64G --time=00:20:00 \
  --output=scaffolds/transformer/validation/logs/coefficient_raw_input_gpu.log \
  --error=scaffolds/transformer/validation/logs/coefficient_raw_input_gpu.log \
  scaffolds/transformer/validation/qualify_coefficient_raw_input.sbatch \
  "$PWD" "$(git rev-parse HEAD)"
```

## Softplus-prefactor SoS activation probe (2026-11-16)

The [qualification record](softplus_sos_activation_probe_b200.json) covers native
Mosaic forward probes of retained softplus-times-SoS checkpoints with core degrees
8 and 12 (total degrees 9 and 13) at d_K=64 and 2,048 tokens on one B200.
Four independent FP64 forward/generator-VJP checks passed, including production
sequence shapes. A two-layer CPU capture check at both degrees preserved the
forward loss exactly with nonzero amplitudes and nonuniform RMSNorm gains;
70 related CPU tests passed and four hardware-only cases were skipped. This
qualifies the forward diagnostic, not training or optimizer continuation.

## Raw-L2 softplus-times-SoS composition (2026-11-16)

The [qualification record](softplus_sos_raw_l2.json) pins source and lock for
core degrees 8 and 12 (total degrees 9 and 13). Six CPU composition checks pass:
raw coefficient input routing, unchanged initialization, independent direct-square
attention outputs and gradients, shared residual gradients, Q/K/V-only RMSNorm
gain gradients, and core-coordinate diagnostics. Seventeen existing rejection
cases and all 119 matched-recipe acceptance checks also pass.

Both one-B200 allocations (jobs 3796 and 3797) passed **858 CPU, 11 distributed
CPU, and 150 GPU checks each**, with zero failures, errors, or skips. The GPU
selection includes both composed raw-L2/softplus/SoS-prefactor BF16 cases at 512
tokens and existing 2,048-token transformed-kernel checks. Both full-shape initial
parameter fingerprints match their controls exactly. Each 20-update native
Mosaic pilot resumed in a fresh process at update 21 with the exact checkpoint
state, next batch, and RNG; Q/K/V retain the shared RMSNorm path.

This qualifies the composition on B200 with the locked CUDA 13 environment and
BF16 attention weights. The joint BF16X3 cases were deselected by the production
acceptance policy, so this record makes no BF16X3 or H100 qualification claim.
The composition changes no Mosaic arithmetic.

### Conventional score MLP (2026-11-16)

The conventional score MLP has static learned readouts, query-dependent hidden
offsets, a separate output bias and explicit `1/sqrt(J)` scaling. J=4 and J=8
pass the independent FP64 output/gradient oracle, causality, initialization and
weight-decay checks through the blocked scan, strips and interpreted fused paths
(14 CPU checks). The host analytic oracle is checked against separate FP64
autodiff. Configuration regressions pass 143 checks. The fused path preserves its
FP32 accumulator until the output-bias contribution is added and then casts once.

`tests/test_plain_score_mlp.py` also qualifies both widths on GPU at 2,048 tokens,
width64 and BF16 Q/K/V, checking every MLP and Q/K/V/input derivative. Each
production recipe's acceptance runs these checks, the historical fused checks,
a full-model pilot and fresh-process checkpoint resume before training.
[Qualification record](runs/plain-score-mlp/record.json).

## Affine B-spline initialization (2026-11-10)

The CPU spline suite passes **98 tests** (two GPU tests deselected) with the locked
environment:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest \
  scaffolds/transformer/tests/test_spline_attention.py -q -m 'not gpu' --maxfail=1
```

Affine initialization computes exact coefficients from Greville abscissae for any
supported degree and knot vector. Checks cover exact polynomial reproduction on
every piece; FP32/FP64 values and derivatives at knots and in the polynomial tails;
static and query model logits against the equivalent fixed monomial function;
backbone initialization, FP32 parameter storage, decay masks, serialization and
historical Taylor compatibility. An Adam update at learning rate `1e-4` moves every
coefficient when its basis is active; after the initially zero query amplitude
moves, task gradients reach the query projection. Existing output/gradient checks
against independent Cox–de Boor/de Boor references also pass. The attention backend
computation is unchanged. GPU acceptance for the new initialization is pending
and remains required on the actual training allocation.

## Polynomial B-spline initialization (2026-11-10)

The locked CPU environment passes **126 spline tests** and **120 shared acceptance
tests**. The spline command above now includes exact polynomial-to-B-spline
blossoming for several degrees and knot vectors, independent Cox–de Boor values,
FP32/FP64 values and score derivatives at knots and in the tails, model-logit parity
with the matching fixed monomial kernel, and unchanged Taylor/affine starts,
backbone draws, decay masks and historical config loading.

For a degree-four basis on `[-32, -6, 0, 6, 32]`, initializing from `[1, 1, 0.5]`
produces `[481, 559/2, 605/6, -10, 6, 815/6, 661/2, 545]` before FP32 rounding.
With all eight basis functions active, actual static and query Adam updates at
`1e-4` change every coefficient. Query amplitudes move from zero and subsequently
pass finite task gradients to their projection. The production pilot callback
can check finite updates from its real validation diagnostics before acceptance
passes; shared tests verify its recorded facts, fail-before-success behavior, and
backward compatibility for adapters without a callback.

GPU qualification now selects four production-precision cases: static/query
coefficients near the Taylor/quadratic starts, at 2,048 tokens and width 64,
comparing outputs and Q/K/V/coefficient gradients against FP64. This new
initialization still requires those GPU checks, full-recipe initialization,
pilot coefficient-update evidence, runtime projection, and exact checkpoint
resume on each production allocation. No new kernel arithmetic is introduced.

## Binomial quartic B-spline initialization (2026-11-10)

The locked CPU environment passes **141 spline tests** (six GPU cases deselected)
with the command above. The initializer matrix includes
`(1 + s/4)^4 / 16`, supplied as ascending power coefficients
`[1/16, 1/16, 3/128, 1/256, 1/4096]`. On the same degree-four basis and knots,
its exact spline coefficients are
`[2401/16, 343/32, -49/32, 35/64, -45/64, 405/32, 3645/32, 6561/16]`;
each is exactly representable in FP32. The maximum coefficient magnitude is
410.0625 and maximum FP32 spacing is 0.000030517578125.

Checks include exact reproduction on every piece, FP32/FP64 values and derivatives
through fourth order at the knots, in both polynomial tails, and at the
fourth-order zero `s = -4`. Static and query model logits match the corresponding
fixed monomial kernel with the actual output RMSNorm epsilon. Actual Adam updates
at learning rate `1e-4` move all eight coefficients when their basis functions are
active; query amplitudes move from zero and then pass task gradients to W.
Existing initialization, decay-mask, serialization and historical-recipe checks
also cover this start.

Common output scaling is not assumed to cancel exactly: a separate test verifies
`RMSNorm(y/16, eps) = RMSNorm(y, 256*eps)` in the implemented arithmetic, including
zero and low-energy rows. The requested `eps = 1e-6` is retained. For an unscaled
head-output RMS of 1, 0.1, or 0.01, the scaled/unscaled normalized amplitudes are
approximately 0.999873, 0.987490, and 0.532642. These are analytic examples, not
measurements of the production score/output distribution.

GPU acceptance selects six production-precision cases: Taylor, quadratic and
binomial starts, each with static and query coefficients. Outputs and
Q/K/V/coefficient gradients are compared against FP64 at the production context
and head width. These allocation checks, the actual-recipe pilot and its observed
coefficient movement, and exact checkpoint resumption remain required before
training. No production arithmetic or normalization setting changed.

## Static relative-Taylor SoS on Mosaic (2026-11-17)

The [qualification record](static_sos.json) pins source, lock, test evidence and
full-size initialization for static SoS cores 8 and 12 on raw scores, softplus
scores, and softplus scores with a multiplicative prefactor (total degrees 9/13).
Shared static factors use the query generator's relative-Taylor bias scaling;
no input-dependent projection or factor decay is introduced.

H100 job 4702 and B200 job 4703 each passed all **22 GPU checks**, without skips,
against independent host-FP64 direct-square references. Coverage includes outputs,
Q/K/V and individual factor gradients, reductions into shared factors, causality,
BF16/BF16X3 arithmetic, and 512/2,048-token shapes. H100 used the locked CUDA 12
environment; B200 used CUDA 13. CPU acceptance passed **573 tests** plus **11
distributed tests**; the focused diagnostics/backend checks also passed.

Actual full-size initialization matches every shared backbone array and initial
effective factor of the query controls, with 198,455,476 parameters for core 8 and
198,456,316 for core 12. Static factors are excluded from decay. This qualifies
kernel behavior and initialization; each production allocation still runs its full
acceptance, timing pilot and fresh-process checkpoint-resume check before training.

Reproduce GPU qualification from the frozen worktree with
`scaffolds/transformer/validation/qualify_static_sos.sbatch`, passing the worktree
root and launch commit, on one H100 or B200. The qualification record contains
the CPU commands and hashes of the retained ignored GPU logs and JUnit reports.

The degree-four extension uses unchanged production arithmetic (source
`31f1b784`): total degrees four/five for softplus without/with prefactor. Its
30 recipe/initialization tests and 10 focused CPU numerical tests pass. Eight
additional GPU cases cover both transformed forms, BF16/BF16X3, 2,048-token
production dispatch, shared-factor reductions and causal gradients. GPU execution
and full-model pilot/resume remain required inside each allocation; the launch
alone is not a completed hardware qualification.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml \
  experiments/transformer_pp/tests/test_static_sos_acceptance.py
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests/test_static_sos.py -k m4
```
