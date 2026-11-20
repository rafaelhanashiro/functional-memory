# Transformer++ results

Results of the [Transformer++ study](../README.md), grouped as on the results site: pretraining
comparisons under `pretraining/`, analyses of trained models and of training under `diagnostics/`.
Each folder has a README (question, runs, findings, limits) and a selection naming its runs, pinned
by `python -m scaffolds.records refresh <folder>`; a folder with figures or tables also holds the
code that regenerates them, with tests. Recipes, measurements, records and retained artifacts stay
under `../runs/`, and one run may serve several folders. Perplexities are validation-best unless a
README says otherwise.

## Pretraining

### Polynomial

| Folder | Question and findings |
| --- | --- |
| [degree_scaling](pretraining/polynomial/degree_scaling/README.md) | Validation perplexity against degree `m = 1-8` and 12 for fixed Taylor, fixed Hermite, learned static, and query coefficients with projection weight decay (Taylor and query only at 12), with softmax as the reference level. |
| [query_width_degree](pretraining/polynomial/query_width_degree/README.md) | How do key width `d_K` (8–64) and degree `m` shape decayed-W tanh query coefficients? |

### Scalings

| Folder | Question and findings |
| --- | --- |
| [l12_d640](pretraining/scalings/l12_d640/README.md) | Every pretraining comparison at the 12-layer width-640 shape, indexed in its README. |
| 12-layer width-640: [shape_pairs](pretraining/scalings/l12_d640/shape_pairs/README.md) | Do the attention-variant comparisons hold at this shape? Eight variants at 1×, each against its reference-shape partner. |
| 12-layer width-640: [budgets](pretraining/scalings/l12_d640/budgets/README.md) | How do the variants compare at 2× and 4× this shape's budget, against exp and their runs at smaller budgets? |
| 12-layer width-640: [softplus_sos](pretraining/scalings/l12_d640/softplus_sos/README.md) | The two degree-twelve softplus–SoS compositions at 1×, 2× and 4×: the prefactor form's lead over exp shrinks from 0.88% to 0.27%, and at 4× it ties raw-score SoS. |
| 12-layer width-640: [score_transforms](pretraining/scalings/l12_d640/score_transforms/README.md) | The degree-eight query polynomial at softplus(s) reaches 17.488 (−0.59% against its control); at the score's absolute value 17.941 (+1.98%). |
| 12-layer width-640: [score_biases](pretraining/scalings/l12_d640/score_biases/README.md) | A signed score shift per polynomial order: static shifts tie (+0.07%), query-dependent shifts lose 3.55%. |
| [token_budget](pretraining/scalings/token_budget/README.md) | Do attention-variant comparisons change between 1× and 2.5× Chinchilla? |
| [context_length](pretraining/scalings/context_length/README.md) | How do the variants compare trained at 4,096 tokens instead of 2,048? |

### FoX / ALiBi

| Folder | Question and findings |
| --- | --- |
| [forget_gate](pretraining/fox_alibi/forget_gate/README.md) | Does a FoX forget gate improve tanh query attention, softmax and exp, and does the original sigmoid gate match Gated DeltaNet's? Per-order cumulative gates beat the shared gate; at m = 8 they reach 13.633, level with FoX exp (13.628). |
| [alibi_decay](pretraining/fox_alibi/alibi_decay/README.md) | Does a constant ALiBi-style decay per head, without RoPE, match the token-dependent forget gate? On softmax (13.899) and exp (13.901) it beats RoPE by 4.4% and 2.5% but trails the forget gate by 1.6% and 2.0%. |
| [query_distance_decay](pretraining/fox_alibi/query_distance_decay/README.md) | Can endpoint dependence, bounded rates, or per-order gates improve query-distance decay? Per-order gates gain 2.2–2.3%; per-order query–key (13.708) beats cumulative FoX by 1.2%. |

### Polynomial parametrizations

| Folder | Question and findings |
| --- | --- |
| Dense: [dense_readout_scale](pretraining/parametrizations/dense/dense_readout_scale/README.md) | Does the dense m = 4 readout need a 1/sqrt(K) scale, and does a pure sigmoid generator beat affine tanh? |
| Activations: [coefficient_forms](pretraining/parametrizations/activations/coefficient_forms/README.md) | Does the query generator's form matter at m = 4? Affine sigmoid (14.756) ties affine tanh (14.774); pure tanh is 53% worse and per-degree pure sigmoid 8–12% worse; none beats fixed Taylor (14.722). |
| SoS / Softplus: [sum_of_squares](pretraining/parametrizations/sos/sum_of_squares/README.md) | Do nonnegative sum-of-squares query coefficients beat fixed Taylor and exp as the degree grows? 14.196, 14.165 and 14.168 at m = 8, 12 and 16: 0.43–0.65% below exp. |
| SoS / Softplus: [static_sos](pretraining/parametrizations/sos/static_sos/README.md) | Do static SoS coefficients match query-conditioned SoS, with raw scores, softplus inputs, and the softplus prefactor? |
| SoS / Softplus: [softplus_sos](pretraining/parametrizations/sos/softplus_sos/README.md) | Does composing softplus inputs with query SoS help, and should the negative-score tail vanish? The degree-twelve prefactor form reaches 14.093 (−1.15% against exp). |
| SoS / Softplus: [softplus_inputs](pretraining/parametrizations/sos/softplus_inputs/README.md) | Does the query polynomial improve at softplus(s) in place of s? 14.262 at m = 8 (−0.42% against its control, level with exp) and 14.222 at m = 12 (−0.02%). |

### Alternative attention mechanisms

| Folder | Question and findings |
| --- | --- |
| Fixed kernels: [fixed_linear](pretraining/alternative_attention/fixed_kernels/fixed_linear/README.md) | How much does the exponential kernel contribute beyond QK and attention-output RMSNorm? |
| Fixed kernels: [pointwise_attention](pretraining/alternative_attention/fixed_kernels/pointwise_attention/README.md) | Can a fixed function of the score replace or multiply exp? Alone, softplus (15.744) and sigmoid (16.135) trail exp (14.257) by 10% and 13%; as prefactors, softplus(s) exp(s) 14.132 (−0.88%) and sigmoid(s) exp(s) 14.186 (−0.50%) beat it, and exp(softplus(s)) 14.329 (+0.51%) does not. |
| MLP: [score_mlp](pretraining/alternative_attention/mlp/score_mlp/README.md) | A conventional SiLU MLP of the score and the query token in place of exp, at widths `J = 4` and 8, against exp, query `m = 4` and fixed Taylor `m = 4`. |
| Splines: [spline_attention](pretraining/alternative_attention/splines/spline_attention/README.md) | A learned quartic B-spline of the score, static or query-dependent, in place of the degree-four polynomial, from four initial functions: started at Taylor it ties the polynomial; every other start is 6–22% behind. |

### Other runs

| Folder | Question and findings |
| --- | --- |
| [even_powers](pretraining/other/even_powers/README.md) | Even-only models finish near 14.62, beating full m = 4 but trailing full m = 8 by about 2%; odd orders with the intercept are 9.8% worse than even-only. |
| [order_gates](pretraining/other/order_gates/README.md) | Do independent query-dependent gates on each polynomial order help query + W decay attention? At m = 8 and 10 they end 8.4% and 7.1% worse than the ungated control. |
| [coefficient_input_norm](pretraining/other/coefficient_input_norm/README.md) | Does coefficient-input L2 reduce tanh saturation and improve query and SoS perplexity? It improves query m = 8 and regresses SoS m = 8 and both m = 12 forms. |
| [zero_constant](pretraining/other/zero_constant/README.md) | Does a structurally zero constant help? On softplus inputs SoS reaches 14.128 (−1.47% against a learned constant), static 14.192, query 14.250 (a tie); on raw scores it hurts: query +18.4%, static +21.4%, SoS +3.7% against raw-score controls. |
| [query_optimization](pretraining/other/query_optimization/README.md) | Does projection weight decay or 3× learning rate on a, b improve degree-four query attention? At 2.5×, W decay lowers query perplexity by 0.1–0.3% at m = 4 and 8. |
| [coefficient_release](pretraining/other/coefficient_release/README.md) | Do degree-eight learned coefficients held at Taylor through warmup, then released, reach fixed Taylor? None does; query released at update 455 ends 14.229, 0.011 above Taylor. |

## Diagnostics

| Group | Folder | Question and findings |
| --- | --- | --- |
| Overfitting | [coefficient_oracle](diagnostics/overfitting/coefficient_oracle/README.md) | How do coefficient fits reveal capacity and conditioning, and can a dense generator distill the free solution and generalize to fresh sequences? Its fitting perplexity is on the batch it fits, not validation perplexity. |
| Continued pretraining | [learned_coefficient_continuation](diagnostics/continued_pretraining/learned_coefficient_continuation/README.md) | How does introducing learned static or query coefficients after Taylor or exp pretraining compare with coefficients learned from the start? |
| Continued pretraining | [forget_gate_continuation](diagnostics/continued_pretraining/forget_gate_continuation/README.md) | Does FoX improve Taylor continuation, including its no-RoPE recipe with projection weight decay? |
| Continued pretraining | [token_query_scale_continuation](diagnostics/continued_pretraining/token_query_scale_continuation/README.md) | Does a bounded per-token query scale with direction weight decay improve degree-four Taylor continuation? |
| Length extrapolation | [extrapolation_ablation](diagnostics/length_extrapolation/extrapolation_ablation/README.md) | Why does query attention break past 2,048 tokens on long documents: keys beyond the training distance, their RoPE phases, or their number? |
| Learning dynamics | [query_parameter_dynamics](diagnostics/learning_dynamics/query_parameter_dynamics/README.md) | How do one middle-layer head's a, b, and W norms evolve in regular query polynomials at degrees 8 and 12, and do its coefficients vary across queries? |
| Learning dynamics | [sos_training_dynamics](diagnostics/learning_dynamics/sos_training_dynamics/README.md) | How do one middle-layer head's a, b, and W norms evolve in the completed sum-of-squares query-polynomial runs at degrees 8 and 12? |
| Learning dynamics | [gated_gradient_spikes](diagnostics/learning_dynamics/gated_gradient_spikes/README.md) | What causes rare gradient spikes in gated polynomial attention? The evidence and reproduction limits are established; the cause remains open. |
| Input dependence | [calibration_mean](diagnostics/input_dependence/calibration_mean/README.md) | Replacing every head's query-conditioned polynomial with its calibration mean increases held-out PPL by 1.21–1.37% across the three degree-eight-core checkpoints. |
