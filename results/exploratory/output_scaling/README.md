# Output scaling for unnormalized attention

Opened 2026-10-01. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) or discard.

Dropping normalization costs 2–4 PPL. Can rescaling each head's unnormalized
output before the output projection recover it? If so, signed kernels, which fail
when normalized, can be explored without normalization.

`model.attention_output_scaling` ([scaffold](../../../scaffolds/transformer/README.md#unnormalized-output-scaling)):
`inverse_position` divides query `t`'s output by its `t` visible keys;
`head_rms_norm` applies a per-head RMSNorm with a learned gain. All runs: `d_K = 64`,
QK-norm, selective decay, 19,074 updates (10.0B tokens), seed 0. Each scaled run exists
with Flax defaults and with the transformer reference initialization
([attention scaling](../../attention_scaling/README.md#initialization)). Runs and roles:
[`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Kernel | Initialization | No scaling | 1/t | Head RMSNorm |
| --- | --- | ---: | ---: | ---: |
| exp | reference | 15.885 | 14.196 | **13.441** |
| exp | Flax defaults | 18.304 | 15.045 | 14.495 |
| Learned m = 4 | reference | 14.720 | 14.056 | **13.662** |
| Learned m = 4 | Flax defaults | 16.384 | fail @12,637 | 14.709 |

References, reference initialization (Flax defaults in parentheses): softmax 13.514
(14.457); normalized m = 4: fixed Taylor 13.774 (14.896), learned nonnegative 13.689
(14.742).

- **Head RMSNorm recovers normalization.** exp lands at softmax on either
  initialization (−0.07 with the reference initialization, +0.04 with Flax defaults):
  for a positive kernel, RMSNorm of the unnormalized output equals RMSNorm of the
  softmax output. Learned m = 4 stays signed, lands 0.15 above softmax, and beats both
  normalized m = 4 kernels on either initialization (by 0.03 and 0.11 with the
  reference initialization).
- **The reference initialization helps every variant,** by 0.85–1.05 for the scaled
  runs and 1.66–2.42 unscaled. The ranking is unchanged: head RMSNorm, then 1/t, then
  no scaling (learned m = 4: 13.662, 14.056, 14.720).
- **Learned m = 4 with 1/t no longer fails.** With Flax defaults its gradient norm
  overflowed at update 12,637 with a finite loss; with the reference initialization it
  finished at 14.056, still 0.39 behind head RMSNorm. The Flax-default run's
  update-12,500 checkpoint is kept for a replay.

## Limits

- One seed; differences of a few tenths are not established.
- With the reference initialization, the 1/t runs trained on H100 and the head-RMSNorm
  runs on B300; with Flax defaults, exp with head RMSNorm trained on H100 and the other
  scaled runs on B300.
