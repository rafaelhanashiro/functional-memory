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

The selection pins both runs explicitly. Validation-best and terminal metrics
are separate: terminal PPL is reported only from a completed run's completion
record at update 7,570. An incomplete run may show its best observed validation
PPL, but no final comparison or terminal value is inferred from it.

<!-- BEGIN GENERATED RESULTS -->
| Variant | Run | Status | Best validation PPL | Best update | Terminal PPL |
| --- | --- | --- | ---: | ---: | ---: |
| FoX NoPE + W decay | `r4a80e28ce0a7` | complete | 13.8734 | 7500 | 13.8755 |
| Query-dependent distance decay + W decay | `r3e26412b2811` | submitted | — | — | — |

The query-dependent run is not complete; a final PPL comparison is pending.
<!-- END GENERATED RESULTS -->

One seed is tested, so this comparison cannot quantify seed-to-seed variability.
The methods differ only in how the same gate values attenuate distance; it is not
a comparison of additional gate capacity or separately tuned optimization.

## Open decision

Retain query-dependent distance decay as a useful FoX variant, or discard this
exploratory topic after comparing completed validation-best and terminal PPL.

## Regeneration

Scientific measurements remain under the selected runs. The comparison uses
committed files only; checkpoints, logs, and cluster-specific paths are not needed.
Refresh pins after an intentional update of a selected run's scientific outputs:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/query_distance_decay
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.query_distance_decay.compare
```

The command regenerates [summary.csv](summary.csv) and the results section above.
