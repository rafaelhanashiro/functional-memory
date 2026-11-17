# Score MLP

Opened 2026-11-04.

**Question.** Can a learned, input-dependent score function `f(x_t, s)` replace exp in
unnormalized attention with attention-output RMSNorm? Method and launch commands are in
the [study README](../../../README.md#score-mlp).

## Runs

The historical runs copy `query-m4-head-scale` (`r2b22f2ff6461`) and changes only `f`, on the fused
score-MLP kernel ([qualification and throughput](../../../../../scaffolds/transformer/validation/README.md)).
Validation-best and terminal (update 7,570) perplexity:

| Run | Label | Form | `J` | Outcome | Best | Terminal |
| --- | --- | --- | ---: | --- | ---: | ---: |
| `re65b12334c5e` | `score-mlp-exp-residual-j4` | `exp(s) (1 + sum_j a_j SiLU(w_j s + b_j))` | 4 | complete | 17.792 | 17.802 |
| `r98e7f3e60a59` | `score-mlp-exp-residual-j8` | `exp(s) (1 + sum_j a_j SiLU(w_j s + b_j))` | 8 | cancelled at update 1,547 | – | – |
| `r4239d082c182` | `score-mlp-kernel-j4` | `sum_j a_j SiLU(w_j s + b_j)`, SiLU fit to exp | 4 | cancelled at update 581 | – | – |

`r4239d082c182` is recorded on branch `claude/score-mlp-kernel`. `re65b12334c5e` trained on
an H100 SXM to update 934 and on an H100 PCIe after two preemptions.

## Findings

- The exp residual at `J = 4` ends 24.8% above exp, 20.9% above fixed Taylor `m = 4`
  and 20.4% above query-only.
- All three runs destabilize. The exp residuals track exp through update 400, then
  their gradient norm spikes to 3-15x the controls' between updates 350 and 900 and
  clipping cuts most updates. The pure form spikes from update 60 (57.0 at 110, 81.6 at
  410). Validation PPL at update 1,500: `J = 4` 29.07, `J = 8` 28.90, against exp 21.59.
- Cause: the unbounded readouts `a_tj = alpha_j + x_t A_j` and `b_tj = beta_j + x_t B_j`.
  Under Adam, `x_t A_j` moves about 700 times faster than `alpha_j`. Within about 100
  updates the correction `C = sum_j a_j SiLU(w_j s + b_j)` is no longer small (`|C| > 1`
  for 88-99.6% of score pairs at update 1,500), and `1 + C` changes sign from token to
  token. The gradient concentrates on the 2.5% of rows where it nearly cancels (97% of
  the gradient with respect to `a`). Each spike is one micro-batch, mostly one layer-0
  head plus token-embedding rows. Afterwards the model learns at about 0.4 of exp's pace
  and never grows its QK scale (head scale 1.02 against exp's 1.18 at update 1,000).
- Not the cause: the kernel (a replay of 48 real updates with the trainer reproduces the
  spikes), the data (no correlation with batch difficulty), weight decay, or negative
  weights as such (the stable query control has a similar fraction).
- The evidence comes from the update-1,500 checkpoint of `r98e7f3e60a59` (kept for this
  reason), Adam moments in snapshot checkpoints, and the GPU replay.

## Controls

Validation-best perplexity, 1x Chinchilla:

| Run | Label | Best PPL |
| --- | --- | ---: |
| `r2b22f2ff6461` | `query-m4-head-scale` | 14.774 |
| `r816ab0fbe119` | `fixed-m4-head-scale` | 14.722 |
| `r6a76e7fbde45` | `exp-head-scale` | 14.257 |

## Conventional MLP baseline

The conventional baseline asks whether an MLP can replace the query-dependent
polynomial directly:

`f(s,x) = b_out + J^(-1/2) sum_j a_j SiLU(w_j s + U_j^T x + b_j)`.

The readout weights are learned per layer/head and do not depend on the token.
The existing pre-attention RMSNorm activation supplies `x`; no extra input norm
is applied. Attention stays unnormalized with QK RMSNorm, per-head positive score
scale, RoPE and head RMSNorm on its output. Both widths use the 15-layer width-896
Transformer++ shape and the same 7,570-update token budget as the controls above.

Initialization is random, with score slopes and readout weights drawn from N(0,1),
activation weights from N(0,1/896), and all biases zero. Width scaling is explicit.
The mechanism RNG preserves the backbone's reference initialization. Ordinary
weight decay applies to MLP weights and exempts biases. These runs change the
architecture and initialization together; they test an expressive MLP baseline,
not a controlled diagnosis of one cause of the older runs' instability.

| Run | Label | J | Parameters | Status |
| --- | --- | ---: | ---: | --- |
| `rea8b942672f4` | `score-mlp-conventional-j4` | 4 | 199,208,956 | B300 acceptance |
| `rfc68c6d26f40` | `score-mlp-conventional-j8` | 8 | 199,964,116 | B300 acceptance |

## Open decision

Compare the conventional J=4 and J=8 MLPs with the polynomial and exponential
controls, inspecting validation perplexity, pre-output-norm activations and gradient
norms before deciding whether this is a useful baseline.
