# Reference initializations

Opened 2026-10-02. Open decision: rerun the canonical comparisons with each
architecture's reference initialization, or discard.

Every canonical run uses the scaffold's Flax defaults (Linear weights truncated normal
with variance `1/fan_in`, no depth scaling). Do they handicap the models against the
recipes their authors train with? Softmax at every `d_K` and GLA at `d_K = 64` repeat
the canonical recipes and change only `model.initialization`
([scaffold](../../../scaffolds/transformer/README.md#weight-initialization)):

- **softmax** (QK-norm): `transformer_reference`, GPT-2/Megatron: N(0, 0.02), attention
  and MLP output projections N(0, 0.02/√(2L)).
- **GLA**: `gla_reference`, official GLA code: N(0, 0.02), Xavier gain 2^-2.5 on the
  query, key, and gate projections.

Runs and roles: [`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Model | d_K | Flax defaults | Reference init | Change |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 8 | 17.224 | **15.639** | −1.59 |
| Softmax | 16 | 15.591 | **14.335** | −1.26 |
| Softmax | 32 | 14.865 | **13.762** | −1.10 |
| Softmax | 64 | 14.457 | **13.514** | −0.94 |
| Softmax | 128 | 14.225 | **13.303** | −0.92 |
| GLA | 64 | 15.895 | **15.376** | −0.52 |

Gap by update at `d_K = 64` (reference minus Flax defaults):

| Update | 1,000 | 5,000 | 10,000 | 15,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | −38.5 | −2.28 | −1.18 | −0.98 | −0.94 |
| GLA | −8.56 | −1.86 | −0.80 | −0.56 | −0.52 |

- **The reference initializations train better, not just faster.** The gap shrinks
  early but levels off: softmax stays near −0.95 over the last 4,000 updates.
- **Softmax gains at every width, most at small `d_K`** (−1.59 at 8 to −0.92 at 128).
- **Against Gated DeltaNet** (paper initialization), reference-initialized softmax
  is ahead at `d_K >= 32` (by 0.19, 0.19, 0.08), level at 16 (+0.05), and behind
  at 8 (+0.79).
- **Every canonical figure is affected.** All heatmap and degree-chart runs except
  Gated DeltaNet use Flax defaults, so their perplexities are inflated by an amount
  that may differ by mechanism; Gated DeltaNet
  ([attention scaling](../../attention_scaling/README.md)) uses its paper's
  initialization.

## Limits

- One seed per run.
- Softmax trained on B300 except `d_K = 8` (H100); the GLA check trained on H100 and
  its canonical run on ORCD.
