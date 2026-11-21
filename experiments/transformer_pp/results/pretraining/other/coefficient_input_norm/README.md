# Coefficient-input normalization

Opened **2026-11-14**. This Transformer++ exploration asks whether limiting the
input to the tanh coefficient generator reduces saturation and improves perplexity.
It contains four completed post-RMSNorm L2 runs, at degrees eight and twelve,
and two completed degree-twelve raw-input L2 runs at the 15-layer width-896 shape.
Two further raw-input L2 runs for softplus-times-SoS cores of degree eight and
twelve also completed, each matched to its original-input parent.

| Coefficient form | Unit-L2 input run | Original-input control |
| --- | --- | --- |
| Query, degree 8 | `r25a5f78841b2` | `r25fffc16c2f3` |
| SoS query, degree 8 | `r717267afcb78` | `reeefbcc19abf` |
| Query, degree 12 | `r7e34dec9dbd5` | `r0fceea0627c8` |
| SoS query, degree 12 | `r6b17c3463a60` | `r3d52b7ab43d1` |

For the current gain-applied pre-attention RMSNorm activation `x`, only the
coefficient generator uses `x_hat = x / max(||x||_2, 1e-6)` in FP32. Direct query
uses `(b_r + a_r*tanh(W_r^T x_hat))/r!`; SoS uses
`F_j = f_j*(1 + b_j + a_j*tanh(W_j^T x_hat))/2`, then expands `U^2 + V^2`.
The Q/K/V inputs and normalization, RoPE, scores, output scaling, and residual
path retain the control recipe. W is unnormalized and trainable with weight
decay 0.1; a and b have no weight decay. W initialization, all learning rates,
seed 42, global batch 256, 2,048 prediction tokens per sequence, 7,570 updates,
and the controls' reference initialization are retained. All use native Mosaic.

The [explicit selection](selection.yaml) pins the matched runs and controls.
The [study README](../../../../README.md#coefficient-input-l2-experiment) gives
execution commands and acceptance requirements. Configurations and compact
measurements live only in the selected runs.

## Completed degree-eight comparison

Both runs completed all 7,570 updates and retained the validation-best checkpoint
at update 7,500. Lower perplexity is better.

| Coefficient form | Best PPL, original / L2 | Terminal PPL, original / L2 | Best-PPL change, L2 − original |
| --- | ---: | ---: | ---: |
| Query | 14.3228 / **14.2604** | 14.3278 / **14.2636** | −0.0624 |
| SoS query | **14.1955** / 14.2707 | **14.1996** / 14.2748 | +0.0752 |

At the terminal diagnostic, saturation averaged over every layer, head, token,
and generator coordinate is **0% for query and 0.0250% for SoS**, using
`abs(tanh(z)) >= 0.99`. The corresponding mean tanh derivatives are 0.9490 and
0.8836. These are all-model averages on one fixed sequence; the original-run
probes measured only layer 8, head 1, so their saturation percentages are not a
matched all-model comparison. The most saturated SoS coordinate reaches 15.38%
on this sequence despite its small aggregate average.

All-update gradient telemetry shows one clipped update after update 700 in each
run, at update 6,376: norms 1.3333 (query) and 1.4469 (SoS), against threshold 1.
Neither run encountered a numerical failure. Low coefficient-input saturation
therefore accompanies a small query improvement but a small SoS regression in
these single-seed runs; removing saturation is not sufficient to improve both.

## Completed degree-twelve comparison

Both degree-twelve runs completed all 7,570 updates and retained their best
checkpoint at update 7,500. Each has 200,905,126 parameters and matches its own
control except for coefficient-input normalization.

| Coefficient form | Best PPL, original / L2 | Terminal PPL, original / L2 | Best-PPL change, L2 − original |
| --- | ---: | ---: | ---: |
| Query | **14.2256** / 14.2592 | **14.2296** / 14.2636 | +0.0336 |
| SoS query | **14.1649** / 14.2815 | **14.1684** / 14.2857 | +0.1166 |

At update 7,500, the new diagnostics use the same first held-out 2,048-token
sequence as the original probes: record SHA256
`26d5803c7dc95fe7f5dd7015ae4195205246bda089792b2e0f478a29398df8a6`.
The matched layer-8/head-1 comparison therefore controls the checkpoint step,
head, sequence, and saturation threshold `abs(tanh(z)) >= 0.99`.

| Coefficient form | Selected-head saturation, original / L2 | Selected-head mean tanh derivative, L2 |
| --- | ---: | ---: |
| Query | 53.5382% / **0%** | 0.9934 |
| SoS query | 61.9028% / **0%** | 0.9679 |

The original probe values and explicit parents are recorded in the
[saturation selection](saturation_selection.yaml).
Across all layers/heads/coordinates, terminal sampled saturation is **0% for
query and 0.0113% for SoS**. Mean tanh derivatives are 0.9514 and 0.9202. These
all-model averages have no matched all-model original-input probe. The most
saturated SoS coordinate reaches 8.45% of sampled tokens. Mean learned W-vector
norms reach 2.05 for query and 2.52 for SoS; W remains trainable and unnormalized.

All-update gradient telemetry after update 500 covers 7,070 updates per run.

| Coefficient form | Clipped updates, original / L2 | Maximum pre-clip norm, original / L2 |
| --- | ---: | ---: |
| Query | 2 / 5 | 11.6393 / 17.9544 |
| SoS query | 1 / 3 | 1.5031 / 1.5540 |

The query L2 maximum occurred at update 1,249; its other recorded spikes were
2.4136 at 1,410, 2.8980 at 5,858, and 1.7640 at 6,376. The remaining clipped
update lies in 901–1,000, with maximum 1.4232. SoS L2 had two clipped updates
within 501–600 (maximum 1.1059) and one at 6,376 (1.5540). Both runs completed
without numerical failure. The cutoff is 500 because the reporting window
401–500 crosses the end of warmup at 455; exact post-warmup counts cannot be
reconstructed from that window's aggregate.

Thus this L2 intervention removes saturation in the matched head and keeps it
very low in the full sampled model, but does not improve m=12 perplexity or
clipping frequency. This does not isolate saturation from changes to coefficient
variation and optimization caused by changing the input scale.

## Completed degree-twelve raw-input comparison

The raw-input variants normalize the residual activation `h` directly:
`h_hat = h / max(||h||_2, 1e-6)`, with FP32 normalization and projection.
Only the coefficient generator uses `h_hat`; Q/K/V still use `RMSNorm(h)`.
In contrast, the completed post-RMSNorm L2 controls normalize the learned-gain
activation. Away from the epsilon floor, their input is
`(g * h) / ||g * h||_2`. The new comparison removes those channel gains from
the coefficient branch, including its gradient contribution to the RMSNorm gain.
W, a, b, and the complete optimizer/data/initialization recipe remain matched.

| Coefficient form | Raw-input L2 run | Post-RMSNorm L2 control | Original-input reference |
| --- | --- | --- | --- |
| Query, degree 12 | `r1f30febd7de5` | `r7e34dec9dbd5` | `r0fceea0627c8` |
| SoS query, degree 12 | `r2bf71d7b4831` | `r6b17c3463a60` | `r3d52b7ab43d1` |

Both raw-input runs completed all 7,570 updates on Kennel, with their best
validation checkpoint at update 7,500. Lower perplexity is better.

| Coefficient form | Original input, best / terminal PPL | Post-RMSNorm L2, best / terminal PPL | Raw-input L2, best / terminal PPL |
| --- | ---: | ---: | ---: |
| Query | **14.2256 / 14.2296** | 14.2592 / 14.2636 | 14.2440 / 14.2495 |
| SoS query | **14.1649 / 14.1684** | 14.2815 / 14.2857 | 14.2089 / 14.2138 |

Raw-input L2 improves best PPL over post-RMSNorm L2 by **0.0152 for query and
0.0726 for SoS**, but remains worse than the original input by **0.0184 and
0.0440**, respectively. Thus bypassing the channel gains recovers part of the
post-RMSNorm L2 regression in these single-seed comparisons; it does not establish
an improvement over the original coefficient input.

All-update telemetry covers updates 501–7,570, or 7,070 updates per run.
The 28 validation events have unique steps, with contiguous reporting windows;
resumed training contributes no duplicate windows to these counts.

| Coefficient form | Clipped updates, original / post-L2 / raw-L2 | Maximum pre-clip norm, original / post-L2 / raw-L2 |
| --- | ---: | ---: |
| Query | 2 / 5 / 4 | 11.6393 / 17.9544 / 1.2936 |
| SoS query | 1 / 3 / 2 | 1.5031 / 1.5540 / 2.1993 |

Neither raw-input run had a numerical failure. Query's largest norm is much
smaller than its two controls' maxima, while SoS's maximum is larger;
these observations do not establish a uniform stability benefit. Both raw-input
maxima occur at update 6,376. The cutoff at 500 follows the reporting-window
limitation described above.

At update 7,500, the same layer-8/head-1 probe (indices 7, 0) and held-out
sequence used above give **0% saturation for both raw-input variants**, as for
their post-RMSNorm L2 controls. Their mean tanh derivatives are **0.6574 for
query and 0.9517 for SoS**, compared with 0.9934 and 0.9679 after RMSNorm.
The corresponding original-input saturation remains 53.5382% and 61.9028%.
Terminal saturation averaged over all sampled layer/head/token/coordinate
positions is **0% for raw-input query and 0.0421% for raw-input SoS**, versus
0% and 0.0113% after RMSNorm. The raw SoS aggregate hides a coordinate at
47.07% saturation (layer 7, head 14, factor U2). All reported diagnostics are
finite. These are descriptive measurements on one sequence, equally weighted
across generator coordinates, not model-contribution-weighted sensitivities.

## Raw-input L2 for softplus-times-SoS

Both runs completed all 7,570 updates on one B200 each on Kennel, retaining
the validation-best checkpoint at update 7,500. They retain `w(s) = u h(u)`,
with `u = softplus(s)` and a query-conditioned SoS core `h = U² + V²`. Core degrees eight and twelve give
total degrees nine and thirteen after the prefactor.

| Core degree | Raw-input L2 run | Original-input control | Control best PPL |
| --- | --- | --- | ---: |
| 8 | `r35805940a380` | `r841f35a24241` | 14.0953 |
| 12 | `r5198d688e9d4` | `rf637de9b65b1` | 14.0934 |

The only scientific recipe changes are
`attention_coefficient_input_normalization: l2` and
`attention_coefficient_input_source: pre_rmsnorm`: the coefficient projection
receives `h_raw / max(||h_raw||_2, 1e-6)` in FP32, while Q/K/V retain the
pre-attention RMSNorm path. Both preserve their parent's 15-layer width-896
shape, RoPE, native Mosaic, initial parameters, W decay 0.1, no a/b decay,
seed 42, optimizer, data, and 7,570-update budget. The comparison asks whether
raw-input L2 improves these stronger softplus-times-SoS controls.

| Core degree | Best PPL, original / raw L2 | Terminal PPL, original / raw L2 | Best-PPL change, raw L2 − original |
| --- | ---: | ---: | ---: |
| 8 | **14.0953** / 14.1313 | **14.0992** / 14.1353 | +0.0360 |
| 12 | **14.0934** / 14.0967 | **14.0972** / 14.1006 | +0.0033 |

Raw-input L2 slightly worsens the degree-eight result and is nearly tied at
degree twelve. Both completed without numerical failure. Across updates
501–7,570, each raw-L2 run clipped twice versus once for its original-input
control. The maximum pre-clip norms were 2.0195 versus 1.2121 at m=8 and
2.7499 versus 1.8003 at m=12. Thus the intervention does not improve clipping
frequency or maximum gradient norm in either matched comparison.

At checkpoint 7,500, the matched layer-8/head-1 saturation drops from
**26.2207% to 0%** at m=8 and **18.5697% to 0%** at m=12. Raw-L2 mean tanh
derivatives in this head are 0.9425 and 0.9634. Averaging all sampled
layers/heads/coordinates gives 0.0637% and 0.0395% saturation, respectively.
These small means hide a strongly saturated U2 coordinate in layer 7, head 2:
88.77% at m=8 and 44.04% at m=12. Terminal all-model averages are 0.0619%
and 0.0402%. All diagnostics are finite; each describes the same single held-out
sequence, with equal weighting rather than amplitude or contribution weighting.
The original controls have no matched all-model probe. Reduced saturation in
the matched head therefore accompanies a small regression or near tie, rather
than evidence that removing saturation improves these models.

## Saturation by coordinate

The [selected-head bars](saturation_by_coordinate.png) compare the original
coefficient input, L2 after RMSNorm, and raw-input L2 for query and SoS at
degrees eight and twelve. They also include the softplus-times-SoS runs
`r841f35a24241` (core degree eight) and `rf637de9b65b1` (core degree twelve):
`w(s) = u h(u)`, where `u = softplus(s)` and `h = U² + V²`. The prefactor makes
the total degrees nine and thirteen. Their completed raw-input L2 variants
now occupy the Raw L2 cells; post-RMSNorm L2 remains **Not run** for this form.
The original-input matched-head mean saturation is **26.22%** at m=8 and
**18.57%** at m=12, with best validation PPL **14.0953** and **14.0934**. The native probe
`rfadcac37b6d0` checks parameter identity and captures the factor generators
before expansion. This head has less saturation at m=12 for this variant; the
single-sequence comparison does not establish a causal link to PPL.

Each bar stacks positive saturation (`tanh(z) >= 0.99`, blue) and negative
saturation (`tanh(z) <= -0.99`, orange). Its total height and the panel mean
remain the fraction of tokens with `abs(tanh(z)) >= 0.99`. Signed fractions
come directly from the original-input probes. Historical L2 telemetry stores
only the combined fraction; exact zero in this head implies that both tails
are exactly zero, and the plotting code rejects nonzero totals without signed
measurements. All panels use layer 8, head 1,
checkpoint 7,500, the same first held-out 2,048-token sequence, and a linear
0–100% axis. Every coordinate in this head has exactly zero measured saturation
in all eight completed L2 runs. Raw-input L2 has not been run at degree eight
for ordinary query or SoS; those cells say **Not run**, not zero.
Each completed panel also shows its run's **best validation PPL** to four decimal
places, read from the pinned training `best_validation.json`. This is the run's
validation evaluation, not the single-sequence diagnostic perplexity. The coordinate
table retains the PPL and its checkpoint step alongside total and signed saturation,
identifies measured tails versus tails inferred from an exact zero total, and includes
explicit score-transform, prefactor-power, core-degree, and total-degree fields.

The most saturated original query and degree-twelve SoS coordinates are
predominantly one-sided on this sequence. For example, SoS m=12 U2 and V4
are 100% negatively saturated, while U3 is 100% positively saturated. These
factor generators are approximately constant across this sample; other
coordinates occupy both tails. Tail fractions do not measure the amplitude
of coefficient variation or establish behavior on other sequences.

SoS bars describe the U/V factor generators before square expansion; they are
not individual expanded polynomial orders. The bars weight tokens and selected
coordinates equally, without amplitude or contribution weighting, and describe
one sequence. The [explicit plot selection](saturation_selection.yaml) pins the
training measurements and the original-input probes.

The PNG, matching PDF, and [coordinate table](saturation_by_coordinate.csv)
regenerate from committed measurements without checkpoints, data, or a GPU:

```bash
python experiments/transformer_pp/results/pretraining/other/coefficient_input_norm/saturation_plot.py
```

## Measurements and limits

At scheduled evaluations, the existing fixed diagnostic sequence records all
layers/heads/coordinates: raw and normalized input norms, projection mean/std/min/max,
`abs(tanh(z)) >= 0.99` fractions, and mean `1-tanh(z)^2`, alongside a/b/W norms,
effective coefficient moments, and ordinary gradient-norm telemetry. The native
model forward supplies these generator activations. Attention-weight diagnostics
use the existing blocked reference alongside the native forward. SoS coordinate
labels refer to U/V factor coefficients, not expanded monomial orders.

A drop in saturation alone is not success: coefficient variation and model
quality matter.
The original controls have only the retained-checkpoint saturation probes, so
full saturation trajectories are available for the new runs, not retrospectively
for every control checkpoint. One seed per cell cannot establish small effects.

## Open question

The degree-eight result is mixed. At degree twelve, raw-input L2 improves on
post-RMSNorm L2 for both forms, but both L2 choices remain worse than the original
input in this single-seed comparison. Keep coefficient-input L2 as an exploration;
the results do not justify replacing the original input normalization by default.
All eight L2 runs are complete. For softplus-times-SoS, raw-input L2 slightly
worsens m=8 and is nearly tied at m=12, with no clipping improvement despite
removing saturation in the selected head. This also leaves the original
coefficient input as the default. The open question is whether an intermediate
input scale warrants a separate experiment; single-seed results cannot resolve
very small quality differences.

Refresh the selected scientific inputs after new measurements:

```bash
python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/other/coefficient_input_norm
```
