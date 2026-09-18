# Polynomial attention: taylor degree 8

Status: complete.

- Attention: taylor; degree: 8; backend: fused.
- Query/key width: 64; value width: 64.
- Parameters: 199906304; seed: 0.
- Updates: 19,074 / 19,074.
- Prediction tokens: 10,000,269,312 / 10,000,269,312.
- Source: `1a8836a75836671cdf83a3ab7763425a6a88cac7`; Slurm job: `199650`.
- Latest validation NLL: 2.686356; PPL: 14.678087.

Attention diagnostics at update 19074: valid on the fixed first held-out sequence (2048 prediction positions).

Primary result: terminal PPL at the common prediction-token budget.
Retained validation-best checkpoint: [19074](../../../experiments/polynomial_attention/checkpoints/200m-m008-seed0/19074/), validation PPL 14.678087. Terminal metrics remain the primary result.
