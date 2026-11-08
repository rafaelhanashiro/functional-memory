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

**Correctness.** `scaffolds/transformer/tests/test_endpoint_decay.py` (`-m gpu`)
checks every variant against host FP64 at 2,048 tokens: the output, dq, dk, dv, the
coefficient gradients, and every gate gradient. It also checks zero-key recovery of the
query-only control, invalid-gate poisoning, saturation, and an exact unit diagonal. The
production dispatch, with gate projections from activations, matches FP64 through the full
chain rule. A two-layer model matches the FP32 blocked model's loss and gradient direction.
Acceptance runs these tests on each allocated GPU. They pass on H100 and on B300 (`akita`).

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

**Per-order kernels.** Added 2026-11-08 for the [independent per-order gates](#independent-per-order-gates).
Families `per_order_query_decay` and `per_order_endpoint_decay` evaluate
\(W = \sum_r c_r s^r D_r\) with \(D_r = \exp(-(t-i)\lambda_r)\), one order at a time inside the
same causal tiles. The dQ and dK passes accumulate \(dS = G \sum_r r\,c_r D_r s^{r-1}\),
\(dc_r = \sum_i G D_r s^r\), and the gate sums above for each order, with
\(W_r = c_r D_r s^r\) and the order's own gate argument \(x_r\) in place of \(W\) and \(x\).
`scaffolds/transformer/tests/test_per_order_decay.py` (`-m gpu`) checks the output and every
gradient at 2,048 tokens against FP64 autodiff of the explicit all-pairs formula. It also
checks that duplicated orders reproduce the shared-gate kernels, zero-key recovery of the
per-order query gates, invalid-gate poisoning, the exact unit diagonal, and a two-layer model
against the FP32 blocked model. Its degree-four cases pass on H100 (`dogo`) and B300 (`akita`).
H100's kernels stage the per-order vectors with each tile, so their shared memory does not grow
with the sequence and fits every degree to 12. B300's forward, dQ and dK/dV keep every query's
and key's per-order vectors for the whole sequence; at \(d_K=64\) and 2,048 tokens they hold
degree 7 (query) and 6 (query–key) and refuse larger degrees before building a kernel.

Throughput ([per_order_throughput.json](per_order_throughput.json), commit `b38c116f`),
measured as above:

| Recipe | dogo (H100 SXM): blocked → Mosaic | mastiff (H100 PCIe): blocked → Mosaic |
| --- | --- | --- |
| Per-order query–key (`r03be769d2069`) | 22.47 → 5.19 s/update (4.3×) | 39.72 → 7.70 s/update (5.2×) |
| Per-order query (`r507193d47450`) | 11.58 → 2.76 s/update (4.2×) | 19.57 → 4.38 s/update (4.5×) |
| Endpoint query–key control (`r02f984133092`) | 2.50 s/update | 3.98 s/update |
| Query-only control (`r3e26412b2811`) | 2.24 s/update | 3.48 s/update |
| Gate-free floor | 1.95 s/update | 3.07 s/update |

For 7,570 updates, training projects 10.9 h (SXM) or 16.2 h (PCIe) for the per-order
query–key run, and 5.8 h or 9.2 h for the per-order query run. Blocked, they would take 47 h
and 24 h on SXM. Peak allocator memory is 27.0 GiB on Mosaic, against 27.3–27.5 GiB (query)
and 28.3–29.6 GiB (query–key) blocked. Over their shared-gate controls, per-order gates cost
23% (query) and 108% (query–key) on SXM, and 26% and 94% on PCIe.

| Family (SXM, ms) | Blocked | Mosaic | Mosaic forward / dQ / dK-dV |
| --- | ---: | ---: | --- |
| Per-order query decay | 11.7 / 27.8 | 1.28 / 4.25 | 1.15 / 2.28 / 2.09 |
| Per-order endpoint decay | 18.8 / 67.6 | 3.13 / 12.0 | 2.94 / 6.12 / 6.12 |

Attention is about 2.9 s of the per-order query–key update's 5.2 s. Its dQ and dK/dV passes
cost 3.7× and 3.5× the shared endpoint gate's, because each order adds its own softplus,
sigmoid and exponential to every weight. Nsight Compute also shows register spills in both
backward passes, more in dK/dV: 182M local loads and 125M local stores per query–key launch,
against 88M and 45M in dQ. The dQ order loop keeps five FP32 tiles live (s, dP, distance,
\(s^r\) and the running dS sum) beside its accumulator; the dK/dV loop also carries W beside
the dK and dV accumulators. Holding fewer tiles across orders is the next step for these
kernels. The per-order query kernels cost 1.6× the shared query-decay kernels.

**B300.** On `akita` (B300, same file), the Blackwell decay kernels are the largest remaining
bottleneck. Per-order query and query–key updates take 4.80 and 10.94 s/update on Mosaic
(blocked: 6.56 and 16.73), against 1.22 s for the gate-free floor, 1.61 s for the query-only
control and 2.84 s for the endpoint control. Both per-order recipes are slower there than on
either H100, and the query–key run's 23.0 h of training would not fit a 24-hour allocation, so
the per-order runs train on H100 only. At the per-GPU attention shape, Mosaic forward+backward
takes 15.7 ms (per-order query) and 39.6 ms (per-order query–key) on B300, against 2.23 ms
without a gate. The backward passes dominate: dQ 9.14 and 18.6 ms, dK 5.34 and 15.9 ms, and
dV 3.11 and 6.14 ms. The shared endpoint gate's dQ pass also takes 5.26 ms on B300, against
1.67 ms on H100 SXM.

## Results

The selection pins all eight runs explicitly. Best PPL is the lowest validation PPL
over all evaluations; terminal PPL is the completion record at update 7,570.

<!-- BEGIN GENERATED RESULTS -->
| Variant | Run | Status | Best validation PPL | Best update | Terminal PPL |
| --- | --- | --- | ---: | ---: | ---: |
| FoX NoPE + W decay | `r4a80e28ce0a7` | complete | 13.8734 | 7500 | 13.8755 |
| Query-dependent distance decay + W decay | `r3e26412b2811` | complete | 14.1977 | 7500 | 14.2025 |
| Endpoint query-key distance decay + W decay | `r02f984133092` | complete | 14.0265 | 7500 | 14.0319 |
| Bounded query rate + W decay | `r4f169e5370da` | complete | 14.1969 | 7500 | 14.2003 |
| Per-order query distance + W decay | `r507193d47450` | running | — | — | — |
| Per-order query-key distance + W decay | `r03be769d2069` | running | — | — | — |
| Query-dependent distance decay + W decay, m = 8 | `r9320e5cd3071` | planned | — | — | — |
| Per-order query distance + W decay, m = 8 | `r20fe6fe864bc` | planned | — | — | — |

Query-dependent minus cumulative FoX PPL (negative favors query-dependent decay):
validation-best **+0.3243** (**+2.337%**); terminal **+0.3271** (**+2.357%**).

Endpoint query–key and bounded-rate variants compare against the query-dependent control.
Each per-order variant compares against its corresponding shared-gate control.
The degree-eight shared gate compares against the degree-four query-dependent run.
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

- Endpoint key dependence helps but does not reach FoX. The query–key gate
  (`r02f984133092`) beats the query-only control at 26 of 28 evaluations:
  by 1.21% at the best update (14.026 vs 14.198) and 1.20% at update 7,570.
  From update 3,000 on it leads by 1.0–1.3% and closes 42–54% of the control's
  gap to FoX. FoX still leads at all 28 evaluations, by 1.10% at the best update.
- The bounded rate ties softplus. The sigmoid-rate run (`r4f169e5370da`) ends
  at 14.197 vs 14.198 (best; −0.006%), within ±0.3% of the control from update
  3,000 on.

One seed is tested, so this comparison cannot quantify seed-to-seed variability.
The four completed runs differ in their decay rule, without separately tuned optimization.
The per-order extensions below increase gate capacity; they and the degree-eight pair are
still pending. The two
ablations run on the Mosaic kernels, the two earlier runs on the blocked reference.
Their only numerical difference is BF16 rounding of attention weights and score
gradients (see [Attention kernels](#attention-kernels)). Both ablations were preempted
once and resumed from their latest checkpoint.

## Independent per-order gates

Added 2026-11-07 at the user's request. Each degree-r contribution has its own
positive distance-decay rate, including r=0:

`weight[t,i] = sum_(r=0..4) c_r(x_t) s_ti^r exp(-(t-i) lambda[t,i,r])`.

| Variant | Run | Shared-gate control | Rate |
| --- | --- | --- | --- |
| Query, NoPE, WD | `r507193d47450` | `r3e26412b2811` | `exp(A_r) softplus(wq_r·x_t + b_r)` |
| Query–key, NoPE, WD | `r03be769d2069` | `r02f984133092` | `exp(A_r) softplus(wq_r·x_t + wk_r·x_i + b_r)` |

All gate parameters are per layer, head, and order. Every order starts as an
exact copy of the shared gate's parameters; key projections start at zero.
This preserves the control's initial attention function up to floating-point
summation differences. Polynomial coefficients remain query-only and unchanged.
Both runs retain the controls' Transformer++ recipe: degree four, NoPE, QK
RMSNorm with learned per-head scale, output head RMSNorm, 2,048 tokens, global
batch 256, seed 42, and 7,570 updates. WD 0.1 applies to coefficient W and gate
Wq/Wk; polynomial a,b and gate A,bias remain exempt. Counts are 200,339,386
(query) and 201,280,186 (query–key).

Both run on the Mosaic per-order decay kernels (families `per_order_query_decay`
and `per_order_endpoint_decay`), which evaluate each order's decay inside the causal
tiles, one order at a time, and accumulate every per-order gate gradient on chip.
Their BF16 tensor-core weights and score gradients match the query–key control;
the query control used the blocked reference's FP32 arithmetic, a numerical
implementation difference to retain when interpreting results.
Qualification checks independent FP64 outputs/gradients, every order's decay,
causality and invalid values, duplicated initialization, optimizer masks,
full-model initialization recovery, and the shared GPU pilot/resume checks.
Rates and half-lives are retained per order in the run's diagnostics. These
runs test whether different orders benefit from different learned ranges.
They do not test cumulative per-order forgetting or a larger polynomial degree.

### Degree eight

Added 2026-11-08. The query-gate comparison at degree eight, where the per-order run
gives each of nine orders its own gate:

| Variant | Run | Recipe |
| --- | --- | --- |
| Shared query gate | `r9320e5cd3071` | `r3e26412b2811` at degree eight |
| Per-order query gates | `r20fe6fe864bc` | `r9320e5cd3071` with per-order gates (`r507193d47450` at degree eight) |

Everything else is the degree-four recipe: NoPE, query-only tanh coefficients with
factorial scaling, WD 0.1 on W and the gate projections, QK RMSNorm with learned per-head
scale, output head RMSNorm, 2,048 tokens, batch 256, seed 42, and 7,570 updates. The
shared-gate run starts from exactly the parameters of cumulative FoX at degree eight
(`r9353f554b917`, the same gate parameters): 200,339,386. The per-order run copies that
gate to all nine orders (201,848,026 parameters). The shared-gate run compares against
the degree-four query-dependent run, and the per-order run against the shared-gate run.

Both use the Mosaic decay kernels (`query_decay`, `per_order_query_decay`) on one H100
each. B300 cannot hold the per-order gates at degree eight (see [Attention kernels](#attention-kernels)),
and its decay kernels are slower than either H100's. Degree-eight GPU qualification:
`test_per_order_decay.py` checks the per-order output and every gradient against FP64 and
duplicated orders against the shared-gate kernels; `test_endpoint_decay.py` checks the
shared query gate against FP64.

## Open decision

Endpoint key dependence closes about half of the query-only gap to cumulative
FoX, and the bounded rate changes nothing. Evaluate the per-order extensions against their matched shared
gates, at degree four and degree eight.
Then decide whether to run additional seeds or close the topic by discarding it
or promoting it as its own collection.

## Regeneration

Scientific measurements remain under the selected runs. The comparison uses
committed files only; checkpoints, logs, and cluster-specific paths are not needed.
Refresh pins after an intentional update of a selected run's scientific outputs:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/query_distance_decay
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.query_distance_decay.compare
```

The command regenerates [summary.csv](summary.csv) and the results section above.
