# Output scaling for unnormalized attention

Opened 2026-10-01. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) or discard.

Dropping normalization costs 1–3 PPL. Can rescaling each head's unnormalized
output before the output projection recover it? If so, signed kernels, which fail
when normalized, can be explored without normalization.

`model.attention_output_scaling` ([scaffold](../../../scaffolds/transformer/README.md#unnormalized-output-scaling)):
`inverse_position` divides query `t`'s output by its `t` visible keys;
`head_rms_norm` applies a per-head RMSNorm with a learned gain. All runs: `d_K = 64`,
QK-norm, selective decay, transformer reference initialization, 19,074 updates (10.0B
tokens) with 400 warmup updates, seed 0. Runs and roles: [`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Kernel | No scaling | 1/t | Head RMSNorm |
| --- | ---: | ---: | ---: |
| exp | 15.663 | 13.933 | **13.411** |
| Learned m = 4 | 14.742 | 13.816 | **13.639** |

References: softmax 13.731; normalized m = 4: fixed Taylor 13.927, learned nonnegative
13.815.

- **Head RMSNorm recovers normalization and more.** For a positive kernel, RMSNorm of
  the unnormalized output equals RMSNorm of the softmax output (up to `rms_norm_eps`), so
  exp with head RMSNorm is softmax with head RMSNorm: it lands 0.32 below softmax.
  Learned m = 4 stays signed, lands 0.09 below softmax and 0.23 above exp with head
  RMSNorm, and beats both normalized m = 4 kernels (by 0.29 and 0.18).
- **Head RMSNorm beats 1/t, which beats no scaling,** for both kernels (learned m = 4:
  13.639, 13.816, 14.742; exp: 13.411, 13.933, 15.663).

## Learned m = 1–4 with head RMSNorm

Each head-RMSNorm run is paired with the no-scaling run of the same degree.

| Learned degree | No scaling | Head RMSNorm | Difference |
| --- | ---: | ---: | ---: |
| m = 1 | 17.874 | 17.554 | −0.32 |
| m = 2 | 15.461 | 14.630 | −0.83 |
| m = 3 | 14.630 | 13.988 | −0.64 |
| m = 4 | 14.742 | **13.639** | −1.10 |

- **Head RMSNorm helps at every degree, most at high degree:** from 0.32 at m = 1 to
  1.10 at m = 4.
- **It restores the degree order.** Without output scaling m = 3 beats m = 4; with head
  RMSNorm m = 4 leads m = 3 by 0.35.
- **No loss spikes at any degree** (largest training-loss jump 0.2 nats), including
  m = 1, which spikes by up to 7.3 nats without output scaling.

## Limits

- One seed; differences of a few tenths are not established.
- The exp runs trained on 4 H100s; learned m = 4 with 1/t and without scaling on 4
  B300s.
- Head-RMSNorm line: m = 1 trained on 2 H100 PCIe GPUs (mastiff GRES 6–7, not its
  faulty card) with `attention_coefficient_conditioning: static`, the plain
  learned-coefficient case; m = 2, 3, and 4 on 4 GPUs, each across H100 and B300 after
  requeues. m = 2 and 3 ran a later scaffold commit that only adds gradient-norm logging.
