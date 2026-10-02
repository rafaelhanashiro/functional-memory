# Reference initializations

Opened 2026-10-02. Open decision: rerun the canonical comparisons with each
architecture's reference initialization, or discard.

Every canonical run uses the scaffold's Flax defaults (Linear weights truncated normal
with variance `1/fan_in`, no depth scaling). Do they handicap the models against the
recipes their authors train with? Two `d_K = 64` runs repeat the canonical recipes and
change only `model.initialization`
([scaffold](../../../scaffolds/transformer/README.md#weight-initialization)):

- **softmax** (QK-norm): `transformer_reference`, GPT-2/Megatron: N(0, 0.02), attention
  and MLP output projections N(0, 0.02/√(2L)).
- **GLA**: `gla_reference`, official GLA code: N(0, 0.02), Xavier gain 2^-2.5 on the
  query, key, and gate projections.

Runs and roles: [`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Model | Flax defaults | Reference init | Change |
| --- | ---: | ---: | ---: |
| Softmax | 14.457 | **13.514** | −0.94 |
| GLA | 15.895 | **15.376** | −0.52 |

Gap by update (reference minus Flax defaults):

| Update | 1,000 | 5,000 | 10,000 | 15,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | −38.5 | −2.28 | −1.18 | −0.98 | −0.94 |
| GLA | −8.56 | −1.86 | −0.80 | −0.56 | −0.52 |

- **The reference initializations train better, not just faster.** The gap shrinks
  early but levels off: softmax stays near −0.95 over the last 4,000 updates.
- **Every canonical figure is affected.** All heatmap and degree-chart runs except
  Gated DeltaNet use Flax defaults, so their perplexities are inflated by an amount
  that may differ by mechanism; Gated DeltaNet
  ([attention scaling](../../attention_scaling/README.md)) uses its paper's
  initialization.

## Limits

- One seed per run.
- The softmax check trained on B300 and the GLA check on H100; the canonical GLA run
  trained on ORCD.
