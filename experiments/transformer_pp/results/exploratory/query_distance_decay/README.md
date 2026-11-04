# Query-dependent distance decay

Opened 2026-11-02.

**Question.** Can endpoint-dependent distance decay match cumulative FoX without
accumulating gates from intervening positions? Does key dependence or a bounded
positive decay rate improve the query-only version?

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

## Endpoint gate ablations

Added 2026-11-04. Both runs start from scratch with the complete recipe of
`r3e26412b2811`: degree four, query-only polynomial coefficients, no positional
encoding, projection weight decay, canonical QK RMSNorm with per-head scaling,
2,048-token context, batch 256, seed 42, and the 1× budget of 7,570 updates.
Their attention weight is still `exp(-(t-i)*lambda) * p_t(score)`.

| Variant | Run | Rate |
| --- | --- | --- |
| Endpoint query–key | `r02f984133092` | `lambda[t,i,h] = a[h] * softplus(wq[h]·x[t] + wk[h]·x[i] + b[h])` |
| Bounded query-only | `r4f169e5370da` | `lambda[t,h] = a[h] * sigmoid(wq[h]·x[t] + b[h])` |

Here `a[h] = exp(A_log[h])`; all parameters are per layer and head, and `x` is
the pre-attention RMSNorm activation. The sigmoid-rate form can equivalently be
written `a * (1 + tanh(u/2))/2`. This changes the positive rate from softplus to
sigmoid; it does not replace the attenuation itself with a signed tanh.
Neither variant uses gates at positions between query and key.

The endpoint key projection starts at zero, reproducing the query-only control
exactly at initialization. It adds 188,160 trainable parameters. Both gate
projections receive weight decay 0.1; `A_log` and the bias remain exempt.

The bounded-rate variant has the same parameter count as the control. It uses
the same random rate and projection draws. If the control's initial zero-input
rate factor is `dt = softplus(b_old)`, its bias becomes `logit(dt)` so that
`sigmoid(b_new) = dt`. Thus zero-input rates and half-lives match per head.
Rates on nonzero inputs are not exactly matched: this difference is part of the
nonlinearity ablation, and qualification records a deterministic synthetic-input
comparison. The learned positive `a` bounds the rate across inputs for each head;
it is itself trainable.

Both run on the shared Mosaic decay kernels (see [Attention kernels](#attention-kernels)),
which compute the same weights as the control's blocked reference: decay is evaluated
inside the causal tiles, and pair rates and gate gradients never leave the chip.
Qualification checks explicit attention and gradients against FP64, zero-key recovery,
invalid-gate and saturation behavior, initialization, optimizer masks, a full-recipe GPU
pilot, and checkpoint resume.
The [study launch section](../../../README.md#endpoint-gate-ablations) identifies
the adapter and exact launch command. These are exploratory pretraining runs,
not fixed-batch overfits or checkpoint continuations.

## Attention kernels

Added 2026-11-04. The endpoint runs use Mosaic GPU kernels that evaluate the decay
inside the causal attention tiles (`scaffolds/transformer/attention/backends/mosaic`,
families `query_decay` and `endpoint_decay`), for the forward pass and for both backward
passes. For \(W = p\,D\) and \(G = dO\,V^{\mathsf T}\), the dQ pass accumulates
\(dS = G D p'(s)\), \(dc_r = \sum_i G D s^r\) and the per-query gate sums
\(\sum_i G W (t-i)\), or for endpoint gates \(\sum_i G W (t-i)\sigma(x)\) and
\(\sum_i G W (t-i)\operatorname{softplus}(x)\). The dK pass accumulates the per-key
\(\sum_t G W (t-i)\sigma(x)\). No pairwise rate, decay, or gradient leaves the chip.
Softplus and sigmoid use base-2 hardware exponentials and logarithms, with a series for
\(\log_2(1+e^{-|x|})\) at small arguments. Gates are validated per token: an invalid
input or rate is NaN and poisons exactly the causal rows it enters. The diagonal weight
is exactly unattenuated, and large distances may underflow to zero. Weights and score
gradients enter the tensor cores in BF16, as for every Mosaic query-polynomial run. The
control's blocked reference keeps them in FP32; the GPU tests show that this rounding is
the only difference between the two paths.

**Correctness (H100).** `scaffolds/transformer/tests/test_endpoint_decay.py` (`-m gpu`)
checks every variant against host FP64 at 2,048 tokens: the output, dq, dk, dv, the
coefficient gradients, and every gate gradient. It also checks zero-key recovery of the
query-only control, invalid-gate poisoning, saturation, and an exact unit diagonal. The
production dispatch, with gate projections from activations, matches FP64 through the full
chain rule. A two-layer model matches the FP32 blocked model's loss and gradient direction.
Acceptance runs these tests on each allocated GPU. **B300 is unqualified.** The Blackwell
kernels lower for a CUDA target and pass Mosaic's shared-memory checks, but `akita` could not
initialize CUDA on 2026-11-04: its NVSwitch fabric manager had failed. The runs exclude it.

**Throughput** ([throughput.json](throughput.json), commit `6de6510b`). Every row ran on one GPU
of the named node. A full update is 16 accumulated microbatches of 16 × 2,049 tokens plus
AdamW, on the frozen recipe with `model.attention_backend` as shown. The gate-free floor is the
query polynomial + W decay recipe `rca97cc9db8f2` (RoPE) on Mosaic.

| Recipe | dogo (H100 SXM): blocked → Mosaic | mastiff (H100 PCIe): blocked → Mosaic |
| --- | --- | --- |
| Endpoint query–key (`r02f984133092`) | 13.42 → 2.50 s/update (5.4×) | 20.26 → 3.84 s/update (5.3×) |
| Bounded query rate (`r4f169e5370da`) | 8.69 → 2.24 s/update (3.9×) | 14.77 → 3.40 s/update (4.3×) |
| Query-only control (`r3e26412b2811`) | 8.70 → 2.24 s/update (3.9×) | 14.74 → 3.41 s/update (4.3×) |
| Gate-free floor | 1.94 s/update | 3.04 s/update |

For 7,570 updates, the endpoint run projects 5.3 h (SXM) or 8.1 h (PCIe), and the bounded run
4.7 h or 7.1 h. The blocked pilots projected 46 h and 32 h on PCIe. Peak allocator memory is
26.8 GiB on Mosaic, against 27.1 GiB (query-only) and 29.0 GiB (endpoint) blocked. The gates
cost 15% (query-only) and 29% (endpoint) over the gate-free floor on SXM, and 12% and 26% on
PCIe. Sigmoid and softplus rates cost the same.

**Remaining bottlenecks.** The table gives warmed attention times on SXM at the per-GPU shape
(16 × 2,048 tokens, 14 heads, ms; forward / forward+backward; then the Mosaic kernels).

| Family | Blocked | Mosaic | Mosaic forward / dQ / dK-dV |
| --- | ---: | ---: | --- |
| No gate | 6.9 / 19.7 | 0.67 / 1.61 | 0.55 / 0.85 / 0.82 |
| Query-only decay | 7.0 / 21.4 | 0.77 / 2.73 | 0.67 / 1.01 / 1.73 |
| Endpoint decay | 11.2 / 35.7 | 1.08 / 3.36 | 0.92 / 1.66 / 1.71 |

Attention is now about 0.8 s of the endpoint update's 2.5 s. The rest is the model outside
attention, the same as the floor. The decay dK/dV kernel costs about twice the gate-free one,
even for the single extra exponential of query-only decay. Recomputing instead of reusing the
decay tile did not change it, so holding that tile across the matrix products is not the cause; profiling the Hopper dK/dV step is the
next step. The endpoint dQ pass costs about twice the gate-free one, for its softplus and sigmoid.

## Results

The selection pins all four runs explicitly. Best PPL is the lowest validation PPL
over all evaluations; terminal PPL is the completion record at update 7,570.

<!-- BEGIN GENERATED RESULTS -->
| Variant | Run | Status | Best validation PPL | Best update | Terminal PPL |
| --- | --- | --- | ---: | ---: | ---: |
| FoX NoPE + W decay | `r4a80e28ce0a7` | complete | 13.8734 | 7500 | 13.8755 |
| Query-dependent distance decay + W decay | `r3e26412b2811` | complete | 14.1977 | 7500 | 14.2025 |
| Endpoint query-key distance decay + W decay | `r02f984133092` | planned | — | — | — |
| Bounded query rate + W decay | `r4f169e5370da` | planned | — | — | — |

Query-dependent minus cumulative FoX PPL (negative favors query-dependent decay):
validation-best **+0.3243** (**+2.337%**); terminal **+0.3271** (**+2.357%**).

Endpoint query–key and bounded-rate variants compare against the query-dependent control.
Incomplete runs have no terminal PPL; their best-so-far metrics do not establish a final comparison.
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

Determine whether endpoint key dependence or the bounded-rate nonlinearity
closes the gap with cumulative FoX. The completed query-only softplus-rate run
trails cumulative FoX throughout training; the two new ablations are pending.

## Regeneration

Scientific measurements remain under the selected runs. The comparison uses
committed files only; checkpoints, logs, and cluster-specific paths are not needed.
Refresh pins after an intentional update of a selected run's scientific outputs:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/query_distance_decay
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.query_distance_decay.compare
```

The command regenerates [summary.csv](summary.csv) and the results section above.
