# Output scaling for unnormalized attention

Opened 2026-10-01. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) or discard.

Dropping row normalization costs 2–4 PPL. Can rescaling each head's unnormalized
output before the output projection recover it? If so, signed kernels, which fail
when normalized, can be explored without normalization.

`model.attention_output_scaling` ([scaffold](../../../scaffolds/transformer/README.md#unnormalized-output-scaling)):
`inverse_position` divides query `t`'s output by its `t` visible keys;
`head_rms_norm` applies a per-head RMSNorm with a learned gain. All runs: `d_K = 64`,
QK-norm, selective decay, 19,074 updates (10.0B tokens), seed 0. Runs and roles:
[`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Kernel | No scaling | 1/t | Head RMSNorm |
| --- | ---: | ---: | ---: |
| exp | 18.304 | 15.045 | **14.495** |
| Learned m = 4 | 16.384 | fail @12,637 | **14.709** |

References: softmax 14.457; normalized m = 4: fixed Taylor 14.896, learned
nonnegative 14.742.

- **Head RMSNorm recovers normalization.** exp lands 0.04 from softmax: for a
  positive kernel, RMSNorm of the unnormalized output equals RMSNorm of the softmax
  output. Learned m = 4 improves by 1.68 and beats both normalized m = 4 kernels
  while staying signed.
- **1/t recovers most of it for exp** (3.26 of the 3.85 gap to softmax). Learned
  m = 4 with 1/t tracked well below the unscaled run (16.40 vs 18.09 at update
  12,000) until its gradient norm overflowed at update 12,637 with a finite loss
  (2.79). The cause is undiagnosed; its update-12,500 checkpoint is kept for a replay.

## Limits

- One seed; differences of a few tenths are not established.
- exp with head RMSNorm trained on H100; the other new runs on B300.
