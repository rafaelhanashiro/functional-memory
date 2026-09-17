# Polynomial attention: taylor degree 0

Status: running.

- Attention: taylor; degree: 0; backend: blocked.
- Query/key width: 64; value width: 64.
- Parameters: 199906304; seed: 0.
- Updates: 0 / 19,074.
- Prediction tokens: 0 / 10,000,269,312.
- Source: `2484f00e9a32f1571b330fff5788d9a86175235f`; Slurm job: `199416`.

Degree zero uses uniform causal-prefix averaging. Q/K parameters remain in the model and parameter count; their task-loss gradients are zero, but AdamW weight decay still updates them.
