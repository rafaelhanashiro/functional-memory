# Polynomial attention performance measurements

These retained measurements document the attention implementation and hardware used
for the completed 200M-model sweep. Full training updates use four GPUs, a global
batch of 256, global microbatch 64, four accumulation steps, and 2,048 prediction
positions. Timings are medians of ten synchronized updates after three warmups;
compilation, data loading, evaluation, and checkpointing are excluded unless stated.

| Measurement | Hardware / successful job | Source | Results |
| --- | --- | --- | --- |
| Blocked versus first fused, full update | Four H100 SXM / 199623 | `3537262e5b84b5fb44c204e5e70ad767da46a109` | [Report](blocked_baseline/REPORT.md) · [raw samples](blocked_baseline/benchmark.json) |
| First fused attention kernel | H100 PCIe / 199626 | `0b7a37bab00c75b1b550bd44180c804b8d91170f` | [Report](attention_kernel/REPORT.md) · [raw samples](attention_kernel/benchmark.json) |
| Optimized full updates and real-data pilot | Four H100 SXM / 199635 | `0f655fb7098ae73e5923fc295fe0b5c3012184ee` | [Report](h100_throughput/REPORT.md) · [raw samples](h100_throughput/train-step.json) |
| Optimized full updates on B300 | Four B300 / 199854 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | [Report](b300/REPORT.md) · [raw samples](b300/train-step.json) |

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
[`experiments/polynomial_attention/`](../../../experiments/polynomial_attention/README.md).
