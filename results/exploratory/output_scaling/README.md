# Output scaling for unnormalized attention

Opened 2026-10-01. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) or discard.

Dropping normalization costs 2–4 PPL. Can rescaling each head's unnormalized
output before the output projection recover it? If so, signed kernels, which fail
when normalized, can be explored without normalization.

`model.attention_output_scaling` ([scaffold](../../../scaffolds/transformer/README.md#unnormalized-output-scaling)):
`inverse_position` divides query `t`'s output by its `t` visible keys;
`head_rms_norm` applies a per-head RMSNorm with a learned gain. All runs: `d_K = 64`,
QK-norm, selective decay, transformer reference initialization, 19,074 updates (10.0B
tokens), seed 0. Runs and roles: [`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Kernel | No scaling | 1/t | Head RMSNorm |
| --- | ---: | ---: | ---: |
| exp | 15.885 | 14.196 | **13.441** |
| Learned m = 4 | 14.720 | 14.056 | **13.662** |
| softmax (normalized exp) | 13.514 | — | 13.433 |

References: softmax 13.514; normalized m = 4: fixed Taylor 13.774, learned nonnegative
13.689.

- **Head RMSNorm recovers normalization.** exp lands at softmax (−0.07): for a
  positive kernel, RMSNorm of the unnormalized output equals RMSNorm of the softmax
  output. Learned m = 4 stays signed, lands 0.15 above softmax, and beats both
  normalized m = 4 kernels (by 0.03 and 0.11).
- **Softmax with head RMSNorm repeats exp with head RMSNorm.** The per-head RMSNorm
  cancels softmax's positive row sum, so the two are one model up to `rms_norm_eps`; the
  run differs in its kernel (cuDNN softmax) and GPU count. It finished at 13.433 against
  13.441, and from update 5,000 on the validation curves stay within 0.022 (mean
  −0.002). Numerics alone therefore move one run by about 0.01. The 0.07 by which exp
  with head RMSNorm beats softmax is the gain from adding head RMSNorm to softmax
  (13.433 against 13.514), not from dropping normalization.
- **Head RMSNorm beats 1/t, which beats no scaling,** for both kernels (learned m = 4:
  13.662, 14.056, 14.720; exp: 13.441, 14.196, 15.885).

## Limits

- One seed; differences of a few tenths are not established.
- The 1/t runs trained on H100 and the head-RMSNorm runs on B300 (softmax with head
  RMSNorm on 2 GPUs, the others on 4).
