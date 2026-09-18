# Fused polynomial attention: complete training updates

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

- [Raw timings, losses, configuration and source hashes](benchmark.json)
- [Successful Slurm log](../../../logs/polynomial_attention/performance/blocked_baseline/slurm-199623.out)

This is the initial fused implementation. The [optimized H100 audit](../h100_throughput/REPORT.md) records the kernel subsequently used for production. Reproduction uses `experiments.polynomial_attention.benchmark_train_step` at the source above.
