# Exploratory Transformer++ diagnostics

Open questions within the [Transformer++ study](../../README.md). Each topic owns its
findings and explicit selection; its runs remain under the study's `runs/`.
These diagnostics enter no canonical pretraining comparison. A topic closes by
promotion into a maintained collection or by discarding its results and runs.

| Topic | Question | Opened |
| --- | --- | --- |
| [query_optimization](query_optimization/README.md) | Does projection weight decay or 3x learning rate on a,b improve degree-four query/tanh attention? At 2.5× W decay lowers query perplexity by 0.1–0.3% at m = 4 and 8 and leaves key–query level. | 2026-10-29 |
| [coefficient_oracle](coefficient_oracle/README.md) | How do coefficient fits reveal capacity and conditioning, and can a dense generator distill the free solution and generalize to fresh sequences? | 2026-10-23 |
| [taylor_static_continuation](taylor_static_continuation/README.md) | Does learning static coefficients after Taylor pretraining improve matched continuation? | 2026-10-24 |
| [coefficient_forms](coefficient_forms/README.md) | Does the query generator's form matter at m = 4: pure tanh, and affine sigmoid against affine tanh? | 2026-10-27 |
| [forget_gate](forget_gate/README.md) | Does a Gated DeltaNet forget gate improve m = 4 tanh query attention? | 2026-10-27 |
| [token_budget](token_budget/README.md) | Do attention-variant comparisons change between 1x and 2.5x Chinchilla? | 2026-10-27 |
| [token_query_scale_continuation](token_query_scale_continuation/README.md) | Does a bounded per-token query scale with direction weight decay improve degree-four Taylor continuation? | 2026-10-28 |
| [forget_gate_continuation](forget_gate_continuation/README.md) | Does FoX improve Taylor continuation, including its no-RoPE recipe with projection weight decay? | 2026-10-28 |
| [learned_coefficient_continuation](learned_coefficient_continuation/README.md) | Do learned static or query (decayed-W) degree-four coefficients help when introduced after fixed-Taylor or exp pretraining, with the canonical QK norm? | 2026-10-29 |
| [taylor_query_continuation](taylor_query_continuation/README.md) | Does learning query-dependent coefficients after degree-four Taylor pretraining improve matched continuation? | 2026-10-24 |
| [dense_readout_scale](dense_readout_scale/README.md) | Does the dense m = 4 readout need a 1/sqrt(K) scale under the canonical QK setup? | 2026-10-29 |
| [query_width_degree](query_width_degree/README.md) | How do key width `d_K` (8–64) and degree `m` (2, 4, 8) shape validation PPL for decayed-W tanh query coefficients? | 2026-10-31 |
| [context_length](context_length/README.md) | How do the variants use context beyond their 2,048-token training length, and how do they compare trained at 4,096? | 2026-11-01 |
