# Input-dependent polynomial coefficients

Opened 2026-10-07. Home cluster: **Kennel**.

Does letting the current token choose polynomial coefficients improve
unnormalized attention with per-head output RMSNorm and 400 warmup steps?
The comparison includes an affine degree-one predictor and tanh predictors at
degrees one, two, and three.

## Runs and method

| Role | Run ID | Coefficients | Job | Status |
| --- | --- | --- | --- | --- |
| Affine reference, m = 1 | `r9cf9589b9855` | Query-dependent affine projection | `209648` | Complete, 19,074 updates |
| Static reference, m = 1 | `r03bfa6fd7b2e` | Input-independent, learned per layer and head | `209556` | Complete, 19,074 updates |
| Primary, m = 1 | `r80300a4cd5fa` | `b_r + a_r tanh(w_r^T x_t)` | — | Prepared |
| Primary, m = 2 | `rbfe1750cc24c` | `b_r + a_r tanh(w_r^T x_t)` | — | Prepared |
| Primary, m = 3 | `r0bc57959ef6c` | `b_r + a_r tanh(w_r^T x_t)` | — | Prepared |

Explicit inputs and roles are in [selection.yaml](selection.yaml); scientific run
files live under [the attention study](../../../experiments/attention_scaling/README.md).
The affine reference computes `c_0(x_t), c_1(x_t)` from the incoming attention activation
(after the block input RMSNorm), independently per layer and head. Its coefficient
projection starts at zero weights and bias `(1, 1)`. The control starts with the
same constant coefficients. Both use signed weights `c_0 + c_1 s`, no attention
normalization, and per-head output RMSNorm before the output projection.

All runs use the canonical `transformer_reference` initialization, QK normalization,
`d_K = d_V = 64`, 10 layers, width 1024, 16 heads, seed 0, BF16 training,
effective batch 256, global microbatch 64, the same FineWeb data order, 19,074
updates (10.0B tokens), and learning rates 3e-4 to 3e-5 with 400 warmup updates.
Coefficient parameters are FP32 and excluded from weight decay.
The affine predictor adds 327,680 weights; the backbone and its initial weights
match the static reference. Each completed reference used two H100 GPUs on Kennel's
`standard` partition at nice 0.
The affine reference uses `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`; the static reference uses the default
memory fraction. This runtime setting does not change either scientific recipe.

The tanh runs learn `c_r(x_t) = b_r + a_r tanh(w_r^T x_t)` for every order
`r = 0, ..., m`, separately per layer and head, using the same incoming attention
activation. There is no bias inside tanh. Initial values `b_r = 1/r!`, `a_r = 1`,
and `w_r = 0` reproduce the degree-`m` Taylor kernel exactly. The modulation is
bounded by the learned amplitude for fixed parameters; its bound can change as
the amplitudes learn. These runs request four H100 or B300 GPUs on Kennel's
`standard` or `long` partition at nice 0 and use direct fused polynomial attention.
Every hardware allocation must pass the shared acceptance checks before production.

The [shared scaffold](../../../scaffolds/transformer/README.md#input-dependent-polynomial-coefficients)
defines the mechanism. Exact launch commands are in the
[study README](../../../experiments/attention_scaling/README.md#input-dependent-coefficient-comparison).

## Findings and limitations

The completed affine and static references passed CPU, distributed, and GPU tests, canonical initialization checks,
full-recipe pilots with validation, and exact fresh-process checkpoint resumption.
Both completed on 2026-10-07 with finite attention diagnostics at every recorded
validation. The static control performs better at the end of training:

| Coefficients | Terminal validation NLL | Terminal perplexity | Best validation perplexity | Best checkpoint |
| --- | ---: | ---: | ---: | ---: |
| Query-dependent affine | 2.941899 | 18.951802 | 18.951802 | 19,074 |
| Input-independent learned | 2.865257 | 17.553569 | 17.553569 | 19,074 |

The affine predictor is worse by 1.398234 perplexity (7.97%) and 0.076642 validation
NLL. It leads at updates 1,000 and 2,000, then trails the control at every recorded
validation from update 3,000 onward. Terminal and validation-best metrics coincide
for each run; only their validation-best checkpoints are retained.

The final training sample mean NLL is also worse: 2.976463 versus 2.898437
(difference 0.078026). These means cover seven logged training batches over updates
19,001–19,074, rather than the whole training dataset. Their differences from
validation NLL are similar, 0.034564 and 0.033180. The result therefore does not
show a larger training–validation gap as the main explanation for the affine
predictor's deficit; it does not distinguish optimization from representation
limitations.

The matched static reference isolates query conditioning from output RMSNorm and
warmup length in the affine comparison. This is one seed and one affine predictor, so it does not settle whether
other input-dependent parameterizations help. Neither variant adds GDN's write
gates, forgetting, or delta correction. These runs remain outside the canonical
plot selections.

The three tanh runs have no results yet. Degree one allows comparison with both
completed references; degrees two and three also change polynomial capacity, so
their comparison with the static degree-one reference does not isolate conditioning.

## Regeneration

From a checkout containing these committed run measurements, refresh the derived
index and pinned inputs:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records index experiments/attention_scaling
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/input_dependent_coefficients
```

Metrics and coefficient diagnostics stay in each run's `metrics.jsonl`.

## Open decision

Keep the affine pair as the explicit comparison while evaluating whether a
bounded input-dependent coefficient function changes this outcome. Then decide
whether to promote the comparison into the maintained attention results or
discard the exploratory topic and its runs.
