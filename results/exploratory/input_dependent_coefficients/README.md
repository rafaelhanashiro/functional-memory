# Input-dependent constant and linear coefficients

Opened 2026-10-07. Home cluster: **Kennel**.

Does letting the current token choose the constant and linear coefficients improve
unnormalized degree-one attention over input-independent learned coefficients,
when both use per-head output RMSNorm and 400 warmup steps?

## Runs and method

| Role | Run ID | Coefficients | Job | Status |
| --- | --- | --- | --- | --- |
| Primary | `r9cf9589b9855` | Query-dependent affine projection | `209648` | Production training |
| Control | `r03bfa6fd7b2e` | Input-independent, learned per layer and head | `209556` | Production training |

Explicit inputs and roles are in [selection.yaml](selection.yaml); scientific run
files live under [the attention study](../../../experiments/attention_scaling/README.md).
The primary computes `c_0(x_t), c_1(x_t)` from the incoming attention activation
(after the block input RMSNorm), independently per layer and head. Its coefficient
projection starts at zero weights and bias `(1, 1)`. The control starts with the
same constant coefficients. Both use signed weights `c_0 + c_1 s`, no attention
normalization, and per-head output RMSNorm before the output projection.

Both use the canonical `transformer_reference` initialization, QK normalization,
`d_K = d_V = 64`, 10 layers, width 1024, 16 heads, seed 0, BF16 training,
effective batch 256, global microbatch 64, the same FineWeb data order, 19,074
updates (10.0B tokens), and learning rates 3e-4 to 3e-5 with 400 warmup updates.
Coefficient parameters are FP32 and excluded from weight decay in both variants.
The predictor adds 327,680 weights; the backbone and its initial weights match.
Each run uses two H100 GPUs on Kennel's `standard` partition at nice 0.
The primary uses `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`; the control uses the default
memory fraction. This runtime setting does not change either scientific recipe.

The [shared scaffold](../../../scaffolds/transformer/README.md#input-dependent-linear-coefficients)
defines the mechanism. Exact launch commands are in the
[study README](../../../experiments/attention_scaling/README.md#input-dependent-coefficient-comparison).

## Findings and limitations

Both runs passed CPU, distributed, and GPU tests, canonical initialization checks,
full-recipe pilots with validation, and exact fresh-process checkpoint resumption.
Both have started production training with finite losses. No production validation
comparison is available yet. The control isolates query
conditioning from the effects of output RMSNorm and warmup length.
This comparison uses one seed and does not add
GDN's write gates, forgetting, or delta correction. Report terminal validation
perplexity separately from validation-best perplexity; keep incomplete runs and
numerical failures visible. These runs are outside the canonical plot selections.

## Regeneration

From the experiment worktree, refresh the derived index and pinned inputs:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records index experiments/attention_scaling
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/input_dependent_coefficients
```

Metrics and coefficient diagnostics stay in each run's `metrics.jsonl`.

## Open decision

After the pair finishes, decide whether to promote the comparison into the
maintained attention results or discard the exploratory topic and its runs.
