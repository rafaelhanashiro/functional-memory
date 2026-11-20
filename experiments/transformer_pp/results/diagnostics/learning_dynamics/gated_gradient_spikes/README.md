# Gradient spikes in gated polynomial attention

Opened **2026-11-10**. Home cluster for the targeted replay: **Kennel**.

What causes rare gradient spikes in Transformer++ gated polynomial attention,
and do these events impede optimization? This diagnostic starts from
the completed per-order query–key gate run `r03be769d2069`. It distinguishes
recorded training behavior from hypotheses about the backward pass. It ranks no
mechanisms.

## Historical evidence

The [selection](selection.yaml) explicitly pins eight completed runs and their
scientific inputs. The [summary](history_summary.csv) reports updates **1,501–7,570**,
6,070 updates per run. The quantity is the global gradient norm after gradient
accumulation and before clipping at one. Every update contributes to each
reporting window's maximum, mean, and clipped fraction. The analysis checks
contiguous windows, their update counts, a common clipping threshold, completion,
and the pinned input hashes before writing the table.

| Run | Largest norm | Clipped updates | Clipped fraction | Best validation PPL |
| --- | ---: | ---: | ---: | ---: |
| Query m=4, ungated, RoPE (`rca97cc9db8f2`) | 26.257 | 8 | 0.132% | 14.7085 |
| Query m=8, ungated, RoPE (`r25fffc16c2f3`) | 3.842 | 5 | 0.082% | 14.3228 |
| Shared query–key gate m=4, NoPE (`r02f984133092`) | 25.727 | 47 | 0.774% | 14.0265 |
| Per-order query–key gates m=4, NoPE (`r03be769d2069`) | 102.760 | 44 | 0.725% | 13.7084 |
| Shared query gate m=4, NoPE (`r3e26412b2811`) | 55.553 | 96 | 1.582% | 14.1977 |
| Per-order query gates m=4, NoPE (`r507193d47450`) | 1,425.372 | 54 | 0.890% | 13.8918 |
| Exp query–key gate, NoPE (`rd3501a40e020`) | 1.516 | 1 | 0.016% | 13.6036 |
| Exp ungated, RoPE (`r6a76e7fbde45`) | 1.678 | 1 | 0.016% | 14.2572 |

All eight runs completed 7,570 updates; every validation-best checkpoint is at
7,500. Terminal PPL is reported separately in the CSV. These runs use the same
15-layer width-896 Transformer++ shape, 2,048-token training context, global batch
256, seed 42, and the same training budget. Gate definitions and the matched
shared/per-order comparisons are described in the
[query-distance topic](../../../pretraining/fox_alibi/query_distance_decay/README.md). The exp query–key
baseline is described in the [decay topic](../../../pretraining/fox_alibi/alibi_decay/README.md).

- Gated polynomials have substantially more clipped updates than the selected
  ungated polynomials or exp runs after update 1,500. Nevertheless, clipping is
  infrequent: below 1% for both query–key variants and the per-order query variant.
- Ungated query attention is not spike-free: the degree-four run reaches 26.257.
  The degree-eight ungated run shown in the original comparison has a much smaller
  maximum, 3.842. Degree must therefore remain visible in the comparison.
- The per-order query–key model improves PPL over its shared-gate control despite
  a larger maximum gradient norm, and has slightly fewer clipped updates
  (44 versus 47). The maximum alone does not rank optimization quality.
- No selected run has a recorded numerical failure. The historical statistics
  establish rare large gradient excursions, not a causal explanation of the remaining exp gap.

### What the largest plotted event identifies

The per-order query–key maximum of **102.760** belongs to the interval
**updates 4,001–4,500**. The interval midpoint is not the spike's update. Only the
maximum value is retained by the compact metrics; these files contain no exact
peak step or associated batch. Training NLL is sampled every ten updates, while
validation is measured at the interval endpoint.

| Endpoint | Sampled training NLL in the preceding window | Validation NLL |
| --- | ---: | ---: |
| 4,000 | 2.779434 | 2.754505 |
| 4,500 | 2.757445 | 2.725935 |
| 5,000 | 2.733009 | 2.700291 |

The peak window has five clipped updates. Both loss summaries improve through
it; no persistent loss reversal is visible at this resolution. The sampled
training loss can miss the offending batch, and these observations cannot rule
out a transient excursion or an adverse effect on optimizer state.

## Backward-pass audit

The source audit found no concrete error in the per-order query–key Mosaic
custom VJP. The base-two exponential conversion, log-rate chain rule, causal
mask, and degree-zero term are consistent with the forward definition. This is
an inspection result, not a precision comparison on a trained spike.

For a single attenuation factor, let
\(D=\exp[-d\exp(A)\operatorname{softplus}(z)]\) and
\(u=d\exp(A)\operatorname{softplus}(z)\), with causal distance \(d\geq0\).
Then
\(|\partial D/\partial A|=u e^{-u}\leq1/e\), and
\(|\partial D/\partial z|=u e^{-u}\sigma(z)/\operatorname{softplus}(z)\leq1/e\).
Thus the distance factor alone does not make these scalar derivatives unbounded.
The full gradients also include polynomial terms, value vectors, sums over
tokens, and downstream derivatives; this bound does not bound the model gradient.

A candidate mechanism is cancellation among signed polynomial/value
contributions followed by sensitivity in head RMSNorm. Independently gated
orders can change those cancellations. Small output RMS by itself is
insufficient evidence: the relevant row must also carry a large backward
contribution, and harmless common rescaling can cancel through normalization.
A targeted check must compare corresponding forward operands and cotangents,
distinguish cancellation across orders from cancellation across keys, and
separate mathematical sensitivity from finite-precision error.

A simple limiting case explains why large scores are not required. With one
effective key, fixed value vector \(v\), fixed normalization gain \(g\), and
\(R=\operatorname{RMS}(v)>0\), the normalized output is
\[
z=\frac{p(s)(g\odot v)}{\sqrt{p(s)^2R^2+\epsilon}},\qquad
\frac{\partial z}{\partial s}
=\frac{\epsilon p'(s)(g\odot v)}{(p(s)^2R^2+\epsilon)^{3/2}}.
\]
At a simple polynomial zero, this derivative is
\(p'(s)(g\odot v)/\sqrt\epsilon\). A small forward weight need not imply a
small derivative. For \(p(s)=e^s\), the derivative instead has Euclidean norm at
most \(2\|g\odot v\|/(3\sqrt3 R)\), independent of epsilon. The same polynomial
argument applies to coefficient parameters through \(\partial p/\partial\theta\).
Strong locality and independent degree gates could expose or change such zeros.
This is a hypothesis, not an attribution of a recorded event: multiple-key value
cancellation, learned gains, gate/score projections, and downstream cotangents
are absent from this limiting calculation.

## Limits and next diagnostic

The comparison has one seed and mixed execution. The primary per-order query–key
and query runs trained uninterrupted on H100 SXM; the shared query–key control
used H100 PCIe, and both exp controls and the ungated degree-eight control used
B300. Shared query-distance attention used the blocked implementation; the other
polynomial distance-gate cases in this table used Mosaic. The ungated controls
also use RoPE, whereas the distance-gated runs use NoPE. These differences prevent
attributing every gradient difference to the gate alone.

The compact historical measurements lack parameter-group gradient norms,
pre-output-RMSNorm vector norms, and contributions and cancellations by polynomial
order. Existing scalar attention-weight cancellation statistics do not measure
cancellation among value vectors or among the polynomial orders.

Only the validation-best checkpoint at update 7,500 remains for `r03be769d2069`.
It supports a bounded **7,500-to-7,570 replay**, but cannot recreate the historical
4,001–4,500 event at its original weights. A replay must first establish agreement
with the original model and distinguish native training gradients from
instrumented or different-precision evaluations.

The open question is whether evidence localizes the spikes to genuine attention
or normalization sensitivity, numerical arithmetic, or another parameter group,
and whether a targeted intervention improves optimization. Reduced gradient
maxima alone will not demonstrate a quality improvement. Once the question is
resolved, decide whether to keep this record or delete it.

## Regenerate

From a checkout containing the pinned runs:

```bash
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.diagnostics.learning_dynamics.gated_gradient_spikes.analyze
```

If the analysis worktree does not contain those completed run records, supply a
checkout that does with `--repository /path/to/checkout`. The output remains next
to this script. Scientific measurements are read from their canonical run
directories; no logs, checkpoints, or GPU are needed for the historical summary.
The selection fingerprints were generated with
`scaffolds.records.selections.run_sources`; the script verifies them with
`verify_pinned_runs` on every regeneration.
