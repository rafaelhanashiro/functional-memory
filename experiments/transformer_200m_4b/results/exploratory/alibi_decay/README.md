# ALiBi-style constant decay without RoPE (m = 4)

Opened: **2026-10-29**. Home cluster: **Kennel**.

The [forget gate](../forget_gate/README.md) multiplies every attention weight by a factor
that depends on the tokens between key and query, `prod_(j=i+1..t) alpha_j`, and is the one
degree-four mechanism that beats fixed Taylor in pretraining; its
[continuation](../forget_gate_continuation/README.md) learned sparse, content-selected
resets. Is the token dependence the ingredient, or does any learnable decay with distance
do the same? This topic replaces the gate by a constant per layer and head,
`log alpha_t = -lambda_h`, so every weight is multiplied by `exp(-lambda_h (t - i))`: ALiBi's
linear bias in log space, with `lambda_h` learned. Like the forget-gate run it compares
with, it uses **no positional encoding**, so the decay is the only source of position.
The runs live in the [200M / 4B study](../../../README.md) and enter no canonical
comparison.

## Runs

| Role | Run | Label | Position |
| --- | --- | --- | --- |
| Primary: constant decay | `r8d4fc24ef537` | `query-m4-alibi-no-rope` | decay only |
| Control: forget gate | `r82a80d3091e2` | `query-m4-forget-gate-no-rope` | gate only |
| Reference: forget gate | `r111843b4c7cc` | `query-m4-forget-gate` | RoPE + gate |
| Reference: fixed Taylor | `r816ab0fbe119` | `fixed-m4-head-scale` | RoPE |

The primary copies the control's recipe and changes one setting,
`attention_forget_gate_form: constant`: degree-four tanh query coefficients
`c_r(x) = (b_r + a_r tanh(W_r x)) / r!` at zero amplitude (Taylor at initialization),
canonical QK RMSNorm with per-head scaling, head RMSNorm output scaling, the blocked
reference backend (Mosaic has no decay), and the study's 4B recipe: 7,570 updates of 256
sequences, learning rate 0.001 with cosine decay to 0.0001, warmup 455, weight decay 0.1.
The decay adds one learned log rate `A_h` per layer and head (**210 parameters**, no weight
decay), with `lambda_h = exp(A_h)` starting at ALiBi's geometric slopes `2^(-8 h / 14)` for
heads `h = 1..14`, the same in every layer: 0.67 down to 0.0039, half-lives from about one
token to 177. The gate control instead has Gated DeltaNet's token-dependent gate with its
standard initialization (188,580 parameters).

Launch commands are in the [study README](../../../README.md#alibi-style-constant-decay-without-rope).

## Findings and open decision

The primary completed its 7,570 updates on one B300 (Kennel `akita`, `long`; job 229720,
11 h 26 min). The control, the other session's run, completed on the same node.

| Run | Position | Terminal PPL | Validation-best PPL / step | vs fixed Taylor |
| --- | --- | ---: | --- | ---: |
| Fixed Taylor `r816ab0fbe119` | RoPE | 14.724 | 14.722 / 7,500 | — |
| Forget gate `r111843b4c7cc` | RoPE + gate | 14.320 | 14.318 / 7,500 | −2.7% |
| **Constant decay** `r8d4fc24ef537` | decay only | **14.485** | 14.482 / 7,500 | **−1.6%** |
| **Forget gate** `r82a80d3091e2` | gate only | **13.908** | 13.906 / 7,500 | **−5.5%** |

| Validation PPL at update | 1,000 | 2,500 | 5,000 | 7,500 |
| --- | ---: | ---: | ---: | ---: |
| Constant decay | 25.61 | 18.85 | 15.77 | 14.48 |
| Forget gate, no RoPE | 23.90 | 17.89 | 15.12 | 13.91 |

- **A learnable decay with distance helps on its own**: the constant decay beats fixed Taylor
  with RoPE by 1.6% while carrying no positional encoding at all.
- **But it recovers only about 30% of the gate's gain.** The token-dependent gate without RoPE
  reaches 13.91, 4.0% below the constant decay, with the gap present from update 1,000 and
  roughly constant afterwards. The span dependence, not the decay per se, is the main
  ingredient, consistent with the sparse content-selected resets the gate learned in the
  continuation.
- **Removing RoPE helps the gate**: 13.91 without against 14.32 with, the lowest perplexity
  of any degree-four model in this study and below unnormalized exp with RoPE
  (`r6a76e7fbde45`, 14.26). RoPE and the gate both carry position; the gate alone carries it
  better.
- **What the decay learned.** At the end, `lambda_h` has per-head medians over layers close to
  the ALiBi initialization for the fast heads (1.15 against 0.67, 0.62 against 0.45, 0.44
  against 0.30) and below it for the slow heads (0.002 against 0.004–0.006), with half-lives
  of **1.0 / 19 / 303** tokens at the 10th / 50th / 90th percentiles: 21% of heads forget within
  two tokens, 38% retain beyond 64, 4% beyond 512. Per-layer median half-lives are 12–26
  tokens except layers 3 (51) and 14 (86). The query coefficients opened to mean `|a_r|` 0.18
  (maximum 0.97).

**Open decision:** discard the topic, or promote the comparison with the forget gate into a
positional-structure collection together with the no-RoPE and continuation results. The
control's measurements live on its owner's branch and are pinned here once that branch
reaches `main`.

Measurements live only in each selected run. Refresh the selection's pins after
intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/alibi_decay
```
