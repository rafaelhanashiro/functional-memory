# Sigmoid and softplus attention

Opened 2026-11-18. Home cluster: Kennel.

Does replacing the exponential score function with sigmoid or softplus improve
Transformer++ at the 15-layer width-896 reference shape?

The explicit [selection](selection.yaml) compares sigmoid `r88d8654a8a1d` and
softplus `r266c5dd1f935` with exp `r6a76e7fbde45`. For each head,
`s_ti = q_t^T k_i / sqrt(64)` and `u_t = sum_(i<=t) phi(s_ti) v_i`, followed by
per-head RMSNorm. Attention is unnormalized. The two new functions are
`phi(s) = sigmoid(s)` and `phi(s) = log(1 + exp(s))`, evaluated stably.

Both preserve the control's RoPE, QK RMSNorm with learned positive per-head
score scales, model shape, initialization, data, seed 42, optimizer, global batch
256, microbatch 16, context 2,048, and 7,570-update budget. They contain exactly
the same 198,453,586 trainable parameters as exp. No polynomial coefficients,
score offsets, or additional gates are learned.

The shared scaffold expresses each function as fixed `p(z) = z`, coefficients
`[0, 1]`, after the selected score transform. Its Mosaic transformed attention
kernels compute the causal weighted sum. The exp control used its fused
exponential backend; this implementation difference is recorded and qualified,
not a scientific hyperparameter change.

The [study launch instructions](../../../README.md#sigmoid-and-softplus-attention)
use one B200 per run in `standard`. Before production, acceptance verifies the
pinned control recipe, every initialized parameter and optimizer decay mask,
forward and backward kernel correctness, the full-recipe training pilot, and
fresh-process checkpoint resume. Run records and logs hold the current status.

Both runs are currently `acceptance_failed`, before any training update. CPU and
attention-kernel GPU checks passed; the full-model exact-repeatability fixture
fails because repeated token IDs cause nondeterministic BF16 embedding-gradient
accumulation. A read-only diagnostic reproduced the same behavior with exp and
confirmed exact full-model repeats with distinct token IDs. The
[scaffold validation record](../../../../../scaffolds/transformer/validation/README.md#fixed-sigmoid-and-softplus-attention-2026-11-18)
documents the evidence. The test fixture needs correction before qualification
and launch can continue; the recipes and frozen execution code are unchanged.

No training conclusions yet. The open decision is whether either fixed nonlinear
function merits further comparison with exp and learned SoS attention. Report
validation-best and terminal PPL separately; one seed does not measure variance.

Canonical measurements remain with the runs. Refresh the explicit source pins
when updating this comparison:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/pointwise_attention
```
