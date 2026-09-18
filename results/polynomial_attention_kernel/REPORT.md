# Fused polynomial attention kernel

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

## Correctness

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

## Complete training updates

The [four-H100 full-update benchmark](../polynomial_attention_train_step/REPORT.md)
measured 1.75x speedup for degree 2 and 1.57x for degree 16, including accumulation,
all-reduce and optimizer work. It uses H100 HBM3 GPUs rather than this PCIe H100;
compare the two backends within each benchmark. Data loading, evaluation and checkpoint
I/O are excluded from those training-step measurements.

Enable with `model.attention_backend: fused`. Existing sweep identities and results
remain unchanged; the experiment sweep has not been restarted. Degree zero retains
invalid-score checks and still uses tiled attention rather than a prefix-scan fast path.

- [Raw timing, memory and parity measurements](benchmark.json)
- [Successful Slurm log](../logs/polynomial_attention_kernel/slurm-199626.out)
- [GPU tests](gpu-tests.xml), [CPU tests](cpu-tests.xml), [distributed test](distributed-tests.xml)
- [Submission script](run.sbatch)
