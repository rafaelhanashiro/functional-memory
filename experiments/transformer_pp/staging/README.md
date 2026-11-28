# Staging

Finished Transformer++ results not yet placed in [results](../results/README.md), the folders that mirror the
results site. Each topic holds what a results folder holds: a README (question, runs, findings, limits, date opened),
a selection pinned by `scaffolds.records refresh`, and any figures with the code and tests that regenerate them, so
placing a topic is a move. The user places topics, in a session together; until then nothing here is on the site.
The suggested homes are starting points for that session, not decisions.

## New results

| Topic | Question | Result | Opened | Suggested home |
| --- | --- | --- | --- | --- |
| [exp_prefactor_core_degree](exp_prefactor_core_degree/README.md) | Does `e^s h_t(z)` gain from a degree-twelve SoS core at `d_K` 64? | No: 14.086 and 14.036 at the recipe's and double rate against the degree-eight core's 14.067 and 14.001 (+0.13%, +0.24%); both cores beat exp by 1.2–1.5%. | 2026-11-25 | SoS / Softplus, beside sos_score_prefactors |
| [exp_prefactor_raw_l2](exp_prefactor_raw_l2/README.md) | Does raw-input L2 on the coefficient generator help `e^s h_t(z)` at `d_K` 64? | No: 14.088, 14.050 and 14.188 at one, two and four times the rate against the plain input's 14.067, 14.001 and 14.179 (+0.06–0.35%). | 2026-11-26 | Other: coefficient_input_norm, as its `e^s h_t(z)` part |
| [learning_rate](learning_rate/README.md) | Is the recipe's learning rate the best one at `d_K` 4 and 64? | At `d_K` 4 a larger rate helps, the SoS cores most: exp is best at double (−2.7%), `z h_t(z)` reaches 18.203 at four times, 2.5% below exp's best. At `d_K` 64 the recipe's rate is about right: across nine kernels double gains at most 0.5% (best `exp(s) h_t(z)`, 14.001), four times costs the other eight kernels 0.6–1.6%, and the query polynomial loses about 3% at both rates, with gradient spikes. | 2026-11-22 | Scalings, as a learning-rate page; or Other runs |
| [narrow_exp_prefactors](narrow_exp_prefactors/README.md) | Do the softplus and sigmoid prefactors help the exponential at `d_K` 4? | No: softplus(s) exp(s) is 1.9% behind exp (19.552) and trains unstably; sigmoid(s) exp(s) ties it (19.240). | 2026-11-26 | Fixed kernels: pointwise_attention, beside the exponential with score functions |
| [narrow_key_budget](narrow_key_budget/README.md) | With twice the training tokens, does `z h_t(z)` (SoS core 16) keep its lead over exp at `d_K` 4? | Yes, by less: 16.505 against exp's 16.614 (−0.66%), both at double the rate; at 1x the lead is 2.5%. At four times the rate `z h_t(z)` trains unstably (16.741). | 2026-11-24 | Scalings: token_budget, as its `d_K` 4 part; or beside narrow_keys |
| [narrow_softplus_prefactor_sos](narrow_softplus_prefactor_sos/README.md) | At `d_K` 4, does `softplus(s) h_t(s)` serve as well as `z h_t(z)`? | At four times the rate, yes: 18.541 against 18.569, and 0.68% below exp's best. At the recipe's rate it is 1.9% behind `z h_t(z)` and unstable (19.841). | 2026-11-25 | Beside narrow_keys; or SoS / Softplus with sos_score_prefactors |
| [narrow_keys](narrow_keys/README.md) | How do exp, softmax, `z h_t(z)` and a degree-eight ladder compare as the Q/K width narrows to 16, 8 and 4? | exp is best at `d_K` 4 and 0.5% behind `z h_t(z)` at 8 and 16; the raw-score polynomials trail exp by 1.2–6.2%. With a degree-16 core, `z h_t(z)` ties exp at `d_K` 4 (19.192). | 2026-11-20 | Scalings, as a key-width page; or Polynomial, beside query_width_degree |
| [pure_tanh_prefactors](pure_tanh_prefactors/README.md) | At `d_K` 4, how close to exp do bounded pure tanh coefficients come, alone or under a prefactor, and do they beat the affine generator at `d_K` 64? | None reaches exp (19.190); the closest, `z h_t(z)` with pure tanh square factors, is 19.385 (+1.0%), and `e^s P_t(s)` at degree 16 19.483 (+1.5%). At `d_K` 64 pure tanh trails the affine generator (+1.53% at degree 8, +0.23% at 16) and at degree 16 has more gradient spikes (14 against 5). | 2026-11-22 | Polynomial parametrizations: Activations; the affine degree-16 run at `d_K` 64 also fills a cell of sos/sum_of_squares |
| [sos_score_prefactors](sos_score_prefactors/README.md) | Is softplus the best prefactor `f(s)` in front of a query SoS core? | exp and softplus tie (14.067–14.097); sigmoid is 0.4–1.2% worse; swish 17% worse and unstable. | 2026-11-20 | SoS / Softplus, beside softplus_sos |
| [softplus_input_prefactors](softplus_input_prefactors/README.md) | Do static coefficients or a factor `z` help the direct polynomial of `z = softplus(s)`? | The factor helps: static `z p(z)` 14.114 and query `z p_t(z)` 14.159 beat exp by 1.00% and 0.69%; static `p(z)` (14.351) does not. | 2026-11-20 | SoS / Softplus, beside softplus_inputs |
| [raw_score_prefactors](raw_score_prefactors/README.md) | Does a prefactor of the raw score help the query polynomial `p_t(s)`? | `e^s p_t(s)` and `softplus(s) p_t(s)` both reach 14.118: 1.43% below the control and 0.98% below exp. | 2026-11-25 | SoS / Softplus, beside softplus_inputs |
| [static_taylor_reset](static_taylor_reset/README.md) | Do learned static coefficients improve predictions on the backbone that learned them? | At degree eight, resetting the static checkpoint to Taylor raises PPL from 14.294 to 15.532. The Taylor-trained backbone reaches 14.218 with the same Taylor coefficients. | 2026-11-27 | Learning dynamics |
| [swish](swish/README.md) | Does swish, `s sigmoid(s)`, replace exp as a fixed score function? | No: 18.393, 29% behind exp. | 2026-11-20 | Fixed kernels, beside pointwise_attention |

## Replacements

Reruns that replace a run already placed in `results/`, staged under `replacements/<topic>/` so a changed result is
seen before it is taken in; they are resolved with the user right away. None is waiting.
