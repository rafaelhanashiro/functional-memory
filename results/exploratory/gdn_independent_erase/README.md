# Independent GDN erase and write gates

Opened 2026-10-09. Home cluster: **Kennel**.

Does giving Gated DeltaNet separate sigmoid erase and write gates improve its
language-model performance under the canonical 400-step-warmup recipe?

## Runs and method

| Role | Run ID | Erase gate | Status |
| --- | --- | --- | --- |
| Primary | `rf7341031e3d5` | Independent learned projection | Qualifying SXM memory settings; production queued |
| Control | `rf640fc13f3a8` | Shared with the write gate | Complete, 19,074 updates |

The control is explicitly selected for its canonical GDN initialization, degree-one
key space, width, data, training recipe, and current telemetry on Kennel. Its terminal
validation perplexity is 13.582856. Selection does not depend on its performance.

Runtime qualification `217802` is running at nice 0 on Kennel's debug partition
with two H100 SXM GPUs on `dogo`. Production job `217803` is queued on standard
at nice 0, dependent on successful qualification. Both use preallocation disabled
and a memory fraction of 0.90. Full acceptance must pass, including a 20-update
full-recipe pilot, finite validation, and exact fresh-process checkpoint
resumption, before training. No production training measurements or W&B run exist.

The default memory-fraction runtime passed 139 CPU and 34 GPU tests but failed
its SXM pilot before the first update with a 50.28 GiB allocation error;
the [runtime diagnostic](../../../scaffolds/transformer/validation/runs/gdn-independent-precision-h100/sxm-default-oom.json)
records that failure. It is not a numerical training result. Full-recipe pilot
`216999` passed 20 updates, finite validation, and checkpoint verification on
two H100 PCIe GPUs on `mastiff`, with preallocation disabled and the memory
fraction unset. That pilot does not qualify the SXM hardware or the new runtime.
Source: `d3d93bc6ccaa8a76fc4b29124c80988ae1593405`.
The [scaffold validation record](../../../scaffolds/transformer/validation/README.md#fp32-chunk-precision-repair)
contains the kernel accuracy evidence.

The primary uses repaired FP32 chunk intermediates and TF32x3 products. The
historical control used the earlier reduced-precision kernel, so a performance
difference between these executions cannot be attributed solely to gate separation.
A matched control using the repaired arithmetic would be needed for that conclusion.

Both models use the existing GDN block: linear projections, width-four causal
depthwise convolutions and SiLU for Q/K/V, per-head L2 normalization of Q/K, Mamba-2
scalar decay, gated output RMSNorm, and no RoPE. The new update, in keys-first layout,
is `S_t = alpha_t S_{t-1} + k_t (beta_t v_t - lambda_t alpha_t S_{t-1}^T k_t)^T`.
Both `beta_t = sigmoid(W_beta x_t)` and `lambda_t = sigmoid(W_lambda x_t)` are scalar
per token, layer, and head. The bias-free erase projection starts as a copy of the
write projection in a separate device buffer and is optimized independently. The
copy preserves canonical values and the original RNG draws. All original model
parameters have
the same initialization as tied GDN. The key feature space is unchanged; this run
does not introduce polynomial features.

The recipe has 10 layers, width 1024, 16 heads, `d_K = d_V = 64`, MLP width 3008,
sequence length 2,048, BF16 training, effective batch 256, global microbatch 64,
seed 0, and 19,074 updates (10.0B prediction tokens) on the same packed FineWeb data.
AdamW uses learning rates 3e-4 to 3e-5, 400 warmup updates, betas 0.9/0.95,
weight decay 0.1 with the canonical exclusions, and gradient clipping at 1.
The extra projection adds 163,840 parameters, bringing the total to 211,007,424.
GDN's internal QK normalization and gated output RMSNorm are active even though the
dot-product-only recipe fields `qk_norm` and `attention_output_scaling` are disabled.

The new run records standard every-update activation, gradient, and actual-update
telemetry, plus erase/write gate values and their difference. The original control
has the earlier standard telemetry; newly separated gate groups cannot be compared
directly against a combined historical group.

Run recipes, measurements, provenance, and launch instructions live in
[the attention study](../../../experiments/attention_scaling/README.md#independent-gdn-erase-gate).
The [shared scaffold](../../../scaffolds/transformer/README.md#gated-deltanet) defines
the implementation. No outcome is available for the independent-gate run yet.

## Open decision

Determine whether independent erasure improves validation loss, how far the erase
and write gates separate, and whether the new mechanism belongs in the maintained
attention comparison. The topic remains exploratory until that decision is made.
