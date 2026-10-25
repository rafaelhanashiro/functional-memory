# Exploratory results

Experiments whose place in the canonical results is not yet decided. Each topic is
either promoted into a canonical result collection or deleted with its runs; see
[AGENTS.md](../../AGENTS.md).

| Topic | Question | Opened |
| --- | --- | --- |
| [gradient_spikes](gradient_spikes/README.md) | Are the gradient spikes of learned coefficients with head RMSNorm also in the references, exp with head RMSNorm, and learned coefficients without RMSNorm? | 2026-10-09 |
| [path_position_encoding](path_position_encoding/README.md) | Does replacing RoPE with PaTH improve softmax and query-conditioned tanh polynomial attention? | 2026-10-19 |
| [rational_polynomial](rational_polynomial/README.md) | Does a separate learned denominator polynomial let normalized learned polynomial attention train through where the shared signed polynomial failed? | 2026-10-04 |
| [sparse_polynomial_heads](sparse_polynomial_heads/README.md) | Can fixed or query-dependent sparse polynomial features reproduce frozen pretrained softmax heads at a small active-feature budget? | 2026-10-15 |
| [rbf_basis](rbf_basis/README.md) | How does the number of fixed scalar RBFs affect unnormalized-attention quality and cost with per-head RMSNorm? Three 10B-token runs are complete and one is running; four fresh 1× Chinchilla jobs are submitted. Fixed-RBF quality is compared with the existing exp, polynomial, softmax, and Gated DeltaNet baselines. Two 9-RBF runs add static and query-dependent coefficient learning at 1× Chinchilla; promotion remains open. | 2026-10-21 |
