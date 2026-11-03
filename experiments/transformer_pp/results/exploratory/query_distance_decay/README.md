# Query-dependent distance decay

Opened 2026-11-02.

**Question.** Can each query select its own context range more effectively than
the cumulative FoX gate, without accumulating gates from intervening positions?

## Matched comparison

The control is FoX NoPE + W decay, `r4a80e28ce0a7`. The new run,
`r3e26412b2811`, trains from scratch using its recipe. The only mathematical
change is the multiplicative decay applied to the existing polynomial weights:

\[
\begin{aligned}
\lambda_{t,h} &= \exp(A_{\log,h})\,
  \operatorname{softplus}(w_{\alpha,h}^{\mathsf T}x_t+b_{\mathrm{dt},h}),
&\alpha_{t,h}&=\exp(-\lambda_{t,h}),\\
D^{\mathrm{FoX}}_{t,i,h}&=\prod_{j=i+1}^{t}\alpha_{j,h},
&D^{\mathrm{query}}_{t,i,h}&=\exp[-(t-i)\lambda_{t,h}],\\
\operatorname{weight}_{t,i,h}&=D_{t,i,h}\,p_{t,h}(s_{t,i,h}),
&s_{t,i,h}&=q_{t,h}^{\mathsf T}k_{i,h}/\sqrt{d_K},\qquad i\leq t.
\end{aligned}
\]

The gate uses the same pre-attention RMSNorm activation \(x_t\), with parameters
per layer and head. Attenuation is computed in the log domain in FP32. It
multiplies the polynomial output; distance is not inserted into its argument.
The unnormalized degree-four query polynomial, factorial scaling of both its
stored bias and amplitude, and attention-output head RMSNorm are unchanged.

Both runs use Transformer++ with no positional encoding, \(d_K=64\), the same
truncated-reference initialization, seed 42, 2,048-token context, global batch
256, and 7,570 updates. Both retain QK RMSNorm with per-head positive scaling,
peak/minimum learning rates 0.001/0.0001, 455 warmup updates, AdamW
\((\beta_1,\beta_2)=(0.9,0.95)\), and weight decay 0.1. Decay applies to the gate
projection \(w_\alpha\) and polynomial projection \(W\); polynomial \(a,b\), gate
\(A_{\log},b_{\mathrm{dt}}\), embeddings, normalization gains, and biases remain
exempt. The frozen configurations contain the complete matched recipes.

The [study README](../../../README.md) gives execution and launch details.
Home cluster: Kennel. Each run uses one GPU. Qualification covers independent
explicit attention and gate-gradient references, initialization identity, decay
masks, and the shared acceptance sequence before production training.

## Results

The selection pins both runs explicitly. Best PPL is the lowest validation PPL
over all evaluations; terminal PPL is the completion record at update 7,570.

<!-- BEGIN GENERATED RESULTS -->
| Variant | Run | Status | Best validation PPL | Best update | Terminal PPL |
| --- | --- | --- | ---: | ---: | ---: |
| FoX NoPE + W decay | `r4a80e28ce0a7` | complete | 13.8734 | 7500 | 13.8755 |
| Query-dependent distance decay + W decay | `r3e26412b2811` | complete | 14.1977 | 7500 | 14.2025 |

Query-dependent minus cumulative FoX PPL (negative favors query-dependent decay):
validation-best **+0.3243** (**+2.337%**); terminal **+0.3271** (**+2.357%**).
<!-- END GENERATED RESULTS -->

- Query-dependent distance decay trails cumulative FoX at all 28 evaluations:
  by 2.34% at the best update (7,500) and 2.36% at update 7,570. The relative
  gap peaks at 5.9% (update 600) and stays between 2.3% and 2.6% from update
  3,000 on.
- Learned ranges are not shorter. At update 7,500 the gate half-life
  (`mean_half_life_tokens`: \(\ln 0.5\) over the mean \(\log\alpha\) on the
  diagnostic sample) has the same median over the 210 layer-heads, 2.9 tokens;
  42 heads exceed 100 tokens with query-dependent decay against 31 with FoX.

One seed is tested, so this comparison cannot quantify seed-to-seed variability.
The methods differ only in how the same gate values attenuate distance; it is not
a comparison of additional gate capacity or separately tuned optimization.

## Open decision

Retain query-dependent distance decay as a FoX variant, or discard this
exploratory topic and its run; on the one tested seed it trails cumulative FoX
throughout training.

## Regeneration

Scientific measurements remain under the selected runs. The comparison uses
committed files only; checkpoints, logs, and cluster-specific paths are not needed.
Refresh pins after an intentional update of a selected run's scientific outputs:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/query_distance_decay
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.query_distance_decay.compare
```

The command regenerates [summary.csv](summary.csv) and the results section above.
