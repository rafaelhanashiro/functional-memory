# Polynomial attention: taylor degree 4

Status: complete.

- Attention: taylor; degree: 4; backend: fused.
- Query/key width: 64; value width: 64.
- Parameters: 199906304; seed: 0.
- Updates: 19,074 / 19,074.
- Prediction tokens: 10,000,269,312 / 10,000,269,312.
- Source: `1a8836a75836671cdf83a3ab7763425a6a88cac7`; Slurm job: `199649`.
- Latest validation NLL: 2.706956; PPL: 14.983596.

Attention diagnostics at update 19074: valid on the fixed first held-out sequence (2048 prediction positions).

Primary result: terminal PPL at the common prediction-token budget.
Retained terminal checkpoint: `19074`; validation best: `19074`.
