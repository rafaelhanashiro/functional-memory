# B300 comparison

Four B300s improve softmax throughput by 1.89× and degree-2 throughput by 1.16×,
but degree 16 takes 28% longer with this kernel. Hardware selection therefore
depends on the attention variant; this benchmark does not establish optimal B300 tuning.

Successful Slurm job `199854`, partition `sweep`, node `akita`, four NVIDIA B300
SXM6 AC GPUs. Source: `1a8836a75836671cdf83a3ab7763425a6a88cac7`.
[Successful Slurm log](../../../logs/polynomial_attention/performance/b300/slurm-199854.out).

| Attention | Four H100 SXM, ms/update | Four B300, ms/update | B300 throughput / H100 |
| --- | ---: | ---: | ---: |
| Degree 2 | 463.705 | 399.713 | 1.16× |
| Degree 16 | 686.499 | 880.229 | 0.78× |
| Softmax, cuDNN | 357.379 | 188.821 | 1.89× |

The comparison uses the same 199,906,304-parameter model, global batch 256,
global microbatch 64, four accumulation steps, and 2,048 prediction positions.
Source file hashes, initial parameter hashes, and synthetic input hashes match
[H100 reference job 199635](../h100_throughput/train-step.json). Each measurement
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

## Production continuation

The softmax run subsequently resumed from checkpoint 9,000 on four B300s in job
`199869`, preserving optimizer state and the packed-data cursor. It replayed
282 H100 updates after the saved checkpoint. Observed production updates took
about 0.21 seconds on B300 versus 0.38 on H100. The continuation completed in
38 minutes 54 seconds at step 19,074, with 10,000,269,312 training tokens and
terminal validation perplexity 14.6941766486. Scientific configuration, source,
and dataset were unchanged; hardware changes need not yield bitwise-identical updates.
Retained run/checkpoint details are in the [sweep provenance](../../provenance.json).

The benchmark ran in `sweep` concurrently with the four-H100 production job in
`standard`, demonstrating usable spare capacity beyond that production allocation
on 2026-09-18. Partition availability and policy are time dependent; no queue
snapshot or submission wrapper is needed to retain these scientific measurements.

- [Raw timing, loss, and configuration records](train-step.json).
- [Numerical comparison](comparison.json), [GPU tests](gpu-tests.xml), and [provenance](metadata.json).
