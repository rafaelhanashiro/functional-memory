# Gated DeltaNet across key width

Opened 2026-10-02. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) (a heatmap column and a
degree-chart reference) or discard; depends on the
[initialization decision](../reference_init/README.md).

Gated DeltaNet replaces attention on the shared 200M backbone: 10 layers, width 1024,
16 heads, value width 64, the same data, optimizer, and 10B-token budget, sweeping key
width `d_K`. It uses FLA's chunked kernels, the paper's initialization
(`gated_deltanet_reference`), and no QK-norm or RoPE
([scaffold](../../../scaffolds/transformer/README.md#gated-deltanet)). Parameter counts
are within 0.2M of GLA's at every width. Runs and roles: [`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity (softmax and GLA are the canonical runs, on Flax
defaults):

| d_K | Gated DeltaNet | Softmax | GLA |
| ---: | ---: | ---: | ---: |
| 8 | **14.847** | 17.224 | 17.458 |
| 16 | **14.290** | 15.591 | 16.686 |
| 32 | **13.952** | 14.865 | 16.164 |
| 64 | **13.705** | 14.457 | 15.895 |
| 128 | **13.380** | 14.225 | 15.538 |

With every model on its reference initialization (`d_K = 64` only): softmax **13.514**,
Gated DeltaNet 13.705, GLA 15.376.

- **Gated DeltaNet improves with every doubling of `d_K`,** from 14.85 to 13.38, with
  no sign of saturation at 128.
- **It beats the canonical softmax and GLA runs at every width,** but those use Flax
  defaults. On equal footing at `d_K = 64` it trails softmax by 0.19 and beats GLA by
  1.67.
- **Its advantage is largest at small `d_K`** (2.4 below canonical softmax at 8,
  0.85 at 128).

## Limits

- One seed per cell; differences of a few tenths are not established.
- The width comparison against softmax and GLA mixes initializations except at `d_K = 64`.
- `d_K = 128` trained on H100 to update 13,500 and resumed on B300; the others trained
  on B300.
