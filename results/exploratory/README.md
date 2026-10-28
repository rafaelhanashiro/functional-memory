# Exploratory results

Experiments whose place in the canonical results is not yet decided. Each topic is
either promoted into its own canonical result collection or deleted with its runs; see
[AGENTS.md](../../AGENTS.md).

| Topic | Question | Opened |
| --- | --- | --- |
| [4B coefficient oracle](../../experiments/transformer_200m_4b/results/exploratory/coefficient_oracle/README.md) | How does sharing coefficients across sequences and query positions affect fitting with a frozen 4B backbone? | 2026-10-23 |
| [4B coefficient forms](../../experiments/transformer_200m_4b/results/exploratory/coefficient_forms/README.md) | Does the query generator's form matter at m = 4: pure tanh, and affine sigmoid against affine tanh? | 2026-10-27 |
| [4B forget gate](../../experiments/transformer_200m_4b/results/exploratory/forget_gate/README.md) | Does a Gated DeltaNet forget gate improve m = 4 tanh query attention? | 2026-10-27 |
| [coefficient_oracle](coefficient_oracle/README.md) | How much can free per-query polynomial coefficients improve a fixed pretrained model on one sequence or a full batch? | 2026-10-23 |
| [path_position_encoding](path_position_encoding/README.md) | Does replacing RoPE with PaTH improve softmax and query-conditioned tanh polynomial attention? | 2026-10-19 |
| [sparse_polynomial_heads](sparse_polynomial_heads/README.md) | Can fixed or query-dependent sparse polynomial features reproduce frozen pretrained softmax heads at a small active-feature budget? | 2026-10-15 |
