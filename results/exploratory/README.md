# Exploratory results

Experiments whose place in the canonical results is not yet decided. Each topic is
either promoted into a canonical result collection or deleted with its runs; see
[AGENTS.md](../../AGENTS.md).

| Topic | Question | Opened |
| --- | --- | --- |
| [gradient_spikes](gradient_spikes/README.md) | Are the gradient spikes of learned coefficients with head RMSNorm also in the references, exp with head RMSNorm, and learned coefficients without RMSNorm? | 2026-10-09 |
| [input_dependent_coefficients](input_dependent_coefficients/README.md) | Do affine or tanh query-dependent polynomial coefficients improve unnormalized attention with output RMSNorm and warmup 400? | 2026-10-07 |
| [gdn_independent_erase](gdn_independent_erase/README.md) | Completed: separated gates give a numerical near-tie with the historical GDN control; arithmetic and hardware differ. A matched comparison and promotion remain open. | 2026-10-09 |
| [output_scaling](output_scaling/README.md) | Can rescaling unnormalized attention outputs replace normalization? | 2026-10-01 |
| [rational_polynomial](rational_polynomial/README.md) | Does a separate learned denominator polynomial let normalized learned polynomial attention train through where the shared signed polynomial failed? | 2026-10-04 |
| [sparse_polynomial_heads](sparse_polynomial_heads/README.md) | Can fixed or query-dependent sparse polynomial features reproduce frozen pretrained softmax heads at a small active-feature budget? | 2026-10-15 |
