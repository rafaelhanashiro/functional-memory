# Exploratory Transformer++ diagnostics

Open questions within the [Transformer++ study](../../README.md). Each topic owns its
findings and explicit selection; its runs remain under the study's `runs/`.
These diagnostics enter no canonical pretraining comparison. A topic closes by
promotion into a maintained collection or by discarding its results and runs.

| Topic | Question | Opened |
| --- | --- | --- |
| [order_gates](order_gates/README.md) | Do independent query-dependent gates on each polynomial order improve query + W decay attention without a distance-dependent forget gate? At m = 8 and 10 the sigmoid gates end 8.4% and 7.1% worse than the ungated control; the gates lock low orders off early. The per-order Gated DeltaNet gate without the distance exponent (m = 4, RoPE) ends 8.7% worse the same way, against 2.2% better with the exponent: per-order gating helps only through the distance decay. | 2026-11-07 |
| [spline_attention](spline_attention/README.md) | Can quartic B-splines on a fixed knot grid replace the degree-four polynomial? Static 14.745 and query 14.746 tie their m = 4 controls (14.767, 14.709); the coefficients barely leave T4 and scores reach ±150. | 2026-11-08 |
| [coefficient_release](coefficient_release/README.md) | Do degree-eight learned coefficients held at Taylor through warmup, then released all at once or order by order, reach fixed Taylor? None does; query released at 455 ends 14.229, 0.011 above Taylor and 0.094 below its control. | 2026-11-08 |
| [even_powers](even_powers/README.md) | At 1×, both even-only models finish near PPL 14.62 and beat full m=4 but trail full m=8 by about 2% (static and query); keeping the intercept with only odd orders is 9.8% worse than even-only. Does the even-only basis merit further study? | 2026-11-03 |
| [query_optimization](query_optimization/README.md) | Does projection weight decay or 3x learning rate on a,b improve degree-four query/tanh attention? At 2.5× W decay lowers query perplexity by 0.1–0.3% at m = 4 and 8 and leaves key–query level. | 2026-10-29 |
| [coefficient_oracle](coefficient_oracle/README.md) | How do coefficient fits reveal capacity and conditioning, and can a dense generator distill the free solution and generalize to fresh sequences? | 2026-10-23 |
| [coefficient_forms](coefficient_forms/README.md) | Does the query generator's form matter at m = 4: pure tanh, and affine sigmoid against affine tanh? | 2026-10-27 |
| [forget_gate](forget_gate/README.md) | Does a FoX forget gate improve tanh query attention at m = 4 (with PaTH and projection decay) and m = 8, softmax, and exp, and does the original sigmoid gate match Gated DeltaNet's? | 2026-10-27 |
| [token_budget](token_budget/README.md) | Do attention-variant comparisons change between 1x and 2.5x Chinchilla? | 2026-10-27 |
| [token_query_scale_continuation](token_query_scale_continuation/README.md) | Does a bounded per-token query scale with direction weight decay improve degree-four Taylor continuation? | 2026-10-28 |
| [forget_gate_continuation](forget_gate_continuation/README.md) | Does FoX improve Taylor continuation, including its no-RoPE recipe with projection weight decay? | 2026-10-28 |
| [learned_coefficient_continuation](learned_coefficient_continuation/README.md) | How does introducing learned static/query coefficients after Taylor or exp pretraining compare with continuing models whose coefficients were learned from the start? | 2026-10-29 |
| [dense_readout_scale](dense_readout_scale/README.md) | Does the dense m = 4 readout need a 1/sqrt(K) scale under the canonical QK setup? | 2026-10-29 |
| [alibi_decay](alibi_decay/README.md) | Does a constant ALiBi-style decay per head, without RoPE, match the token-dependent forget gate without RoPE in pretraining? ALiBi's fixed slopes on softmax (13.899) and exp (13.901) beat RoPE by 4.4% and 2.5% but trail the forget gate by 1.6% and 2.0%. Learned input-free gates for static coefficients replace RoPE (−3.1% at m = 4, −1.1% at m = 8) but trail the query-coefficient gate runs; per-order input-free gates do not help. | 2026-10-29 |
| [query_width_degree](query_width_degree/README.md) | How do key width `d_K` (8–64) and degree `m` (2, 4, 8) shape validation PPL for decayed-W tanh query coefficients? | 2026-10-31 |
| [context_length](context_length/README.md) | How do the variants extrapolate from 2,048 to 16,384 tokens, and how do they compare trained at 4,096? | 2026-11-01 |
| [fixed_linear](fixed_linear/README.md) | How much does the exponential kernel contribute beyond QK and attention-output RMSNorm? | 2026-11-02 |
| [query_distance_decay](query_distance_decay/README.md) | Can endpoint dependence, bounded rates, or independent per-order gates improve query-distance decay? Per-order gates improve both shared gates, by 2.2% (query) and 2.3% (query–key); per-order query–key (13.708) beats cumulative FoX by 1.2%. At degree eight, per-order query gates gain 1.1% (13.780 vs 13.939), at twelve 0.7% (13.750 vs 13.851), 0.9% above exp with the forget gate. Per-order query–key gates reach 13.712 at degree eight and 13.681 at twelve, 0.5% below their per-order query twins but 0.80% and 0.57% above exp with the query–key gate (13.604). | 2026-11-02 |
