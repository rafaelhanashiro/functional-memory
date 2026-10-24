# Exploratory results

Experiments whose place in the canonical results is not yet decided. Each topic is
either promoted into its own canonical result collection or deleted with its runs; see
[AGENTS.md](../../AGENTS.md).

| Topic | Question | Opened |
| --- | --- | --- |
| [4B coefficient oracle](../../experiments/transformer_200m_4b/results/exploratory/coefficient_oracle/README.md) | How does sharing coefficients across sequences and query positions affect fitting with a frozen 4B backbone? | 2026-10-23 |
| [coefficient_oracle](coefficient_oracle/README.md) | How much can free per-query polynomial coefficients improve a fixed pretrained model on one sequence or a full batch? | 2026-10-23 |
| [path_position_encoding](path_position_encoding/README.md) | Does replacing RoPE with PaTH improve softmax and query-conditioned tanh polynomial attention? | 2026-10-19 |
| [sparse_polynomial_heads](sparse_polynomial_heads/README.md) | Can fixed or query-dependent sparse polynomial features reproduce frozen pretrained softmax heads at a small active-feature budget? | 2026-10-15 |
