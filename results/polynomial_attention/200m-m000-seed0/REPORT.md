# Polynomial attention: taylor degree 0

Status: complete.

- Attention: taylor; degree: 0; backend: blocked.
- Query/key width: 64; value width: 64.
- Parameters: 199906304; seed: 0.
- Updates: 19,074 / 19,074.
- Prediction tokens: 10,000,269,312 / 10,000,269,312.
- Source: `2484f00e9a32f1571b330fff5788d9a86175235f`; Slurm job: `199433`.
- Latest validation NLL: 3.851536; PPL: 47.065313.

Degree zero uses uniform causal-prefix averaging. Q/K parameters remain in the model and parameter count; their task-loss gradients are zero, but AdamW weight decay still updates them.

Attention diagnostics at update 19074: valid on the fixed first held-out sequence (2048 prediction positions).

Primary result: terminal PPL at the common prediction-token budget.
Retained validation-best checkpoint: [19000](../../../experiments/polynomial_attention/checkpoints/200m-m000-seed0/19000/), validation PPL 47.032595. Terminal metrics remain the primary result.
