# B300 comparison and Kennel scheduling audit

Four B300s improve softmax throughput by 1.89x and degree-2 throughput by 1.16x, but degree 16 takes 28% longer with the current kernel. Hardware selection should therefore depend on the attention variant. This benchmark does not establish that the polynomial kernel is tuned optimally for B300.

Successful Slurm job: `199854`, partition `sweep`, node `akita`, four NVIDIA B300 SXM6 AC GPUs. Source: clean frozen commit `1a8836a75836671cdf83a3ab7763425a6a88cac7`. Log: `/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_b300/slurm-199854.out`.

| Attention | Four H100 SXM, ms/update | Four B300, ms/update | B300 throughput / H100 |
|---|---:|---:|---:|
| Degree 2 | 463.705 | 399.713 | 1.16x |
| Degree 16 | 686.499 | 880.229 | 0.78x |
| Softmax, cuDNN | 357.379 | 188.821 | 1.89x |

The comparison uses the same 199,906,304-parameter model, global batch 256, global microbatch 64, four accumulation steps, and 2,048 prediction positions. Source file hashes, initial parameter hashes, and synthetic input hashes match the H100 reference job `199635`. Each measurement is the median of ten synchronized complete distributed training updates after three warmups. Compilation, data loading, evaluation, and checkpointing are excluded. These are short throughput measurements, not full B300 production runtimes or an energy-efficiency comparison.

All 56 GPU attention correctness tests passed. The distributed benchmark completed all three variants with finite losses. The B300 log includes a cuBLAS TMEM warning from JAX 0.10.0's version-range check. Installed `nvidia-cublas-cu12==12.9.2.10` includes the fix documented in [NVIDIA's cuBLAS 12.9.2 patch notes](https://docs.nvidia.com/cuda/cublas-patch-release-notes/#cublas-patch-release-12-9-2); [JAX's warning condition](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax_plugins/cuda/__init__.py#L276-L290) does not account for that backport. No environment change was needed for this issue. A first-use communication initialization warning resolved during warmup and is outside timed measurements.

## Scheduling findings

Live configuration and the supplied Kennel quickstart distinguish these options:

- `standard`: four GPUs per user, 24-hour maximum, higher priority than `long` and `sweep`.
- `long`: seven-day maximum and intended for larger experiments, subject to the documented shared-capacity policy. A four-B300 test-only submission succeeded. Its partition QoS name is absent from the current QoS listing; do not infer unlimited allocation from that discrepancy.
- `sweep`: seven-day maximum, lowest priority, intended for spare capacity, immediate preemption and requeue. The successful B300 benchmark ran concurrently with the four-H100 production job in `standard`, confirming that the standard allowance does not prevent an additional sweep allocation.
- `debug`: two GPUs per user and two-hour maximum; unsuitable for these four-GPU runs.
- `conference`: documentation requires arranging deadline access with Yoon.

Partitions share physical GPUs. At the initial audit, akita's eight B300s and mastiff's eight H100 PCIe GPUs were unallocated. This is a time-dependent snapshot, not a guarantee of future availability.

The current production launcher restricts all runs to a single H100 scheduler profile and serial dependencies. Independent future experiments should retain four GPUs per run while using concurrent standard and opportunistic sweep allocations, with B300 preferred for softmax and measured hardware choices for polynomial degrees. This needs per-run scheduler profiles and hardware validation; a single final aggregation job or a lock should protect shared plots. Existing training resumes checkpoints, optimizer state, and the packed-data cursor, but preemption can lose updates since the most recent checkpoint.

## Softmax continuation

Following the comparison, the remaining softmax run was moved from H100 job `199652` to four B300s on `akita`, job `199869` in `standard`. Its finalized step-9,000 checkpoint preserves 4,718,592,000 training tokens, optimizer state, and the next packed-data sequence (2,304,000). H100 was stopped only after checkpoint and validation commit verification; 282 subsequent uncheckpointed updates will be replayed. The resumed process confirmed step 9,000, packed-data sequence 2,304,000, and 199,906,304 parameters, and then advanced normally at about 0.21 seconds/update (versus about 0.38 on H100). No model, training configuration, source, or dataset changed. The original sweep manifest and queue submission remain unchanged, with the override recorded in the run's `hardware_migration.json` and current `launch.json`.

The continuation wrapper checks the original preflight identity, protected configuration/data hashes, benchmark source, four allocated B300 devices, and absence of the old writer before resuming. It preserves the prior attempt inside the new Slurm log and removes the superseded standalone log only after successful completion. A hardware change does not promise bitwise-identical floating-point updates.

Artifacts: `train-step.json`, `comparison.json`, `gpu-tests.xml`, `cluster-snapshot.json`, `metadata.json`, `run.sbatch`, and `resume-softmax.sbatch`. Continuation details: `/data/cl/u/rafah/home/projects/functional-memory/results/polynomial_attention/200m-softmax-seed0/hardware_migration.json`.
