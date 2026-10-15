# Polynomial memory with GDN query geometry

Opened 2026-10-13. Home cluster: **Kennel**.

Do quadratic and cubic writes improve a GDN memory whose learned
key-space transformation is unchanged?

## Runs and method

| Role | Run ID | Memory function | Status |
| --- | --- | --- | --- |
| Primary | `rd40af7956426` | Constant, linear, and quadratic | Running; update 10,000 recorded |
| Cubic | `r83c3394eb730` | Constant through cubic | Paused at user request; job 218896 held, resumable at update 7,000 |
| Control | `rf7341031e3d5` | Linear, independent erase gate | Complete |

The explicitly selected control uses the repaired arithmetic, canonical GDN
initialization, independent erase gate, and the same data and optimizer recipe.
Its terminal and validation-best perplexity is 13.579409 at update 19,074.
The primary is running in job `218427` on four H100 PCIe GPUs on `mastiff`,
using `standard` at nice 0. Full qualification `218426` passed correctness,
20 training updates, validation, and exact checkpoint resumption; the optimized
pilot projected 19.94 hours for the complete recipe. Neither run is added to a
canonical result selection by this topic.

At update 10,000 (5.243 billion training tokens), the primary's validation
perplexity is 14.5621 versus 14.9685 for the linear control at the same step,
a 2.71% reduction. The primary is lower at each of the first ten validation
points. This is an incomplete learning-curve comparison, not a terminal result.
Recent training intervals average about 3.9 seconds per update, implying roughly
21 hours total if the pace holds. The control ran on H100 SXM GPUs; elapsed time
is not a matched hardware comparison.

All recorded numerical metrics are finite and attention diagnostics are valid.
There are 23 detected gradient spikes through update 10,000, handled by clipping;
the largest norm was 313.18 at update 6,376. Validation loss decreased at every
recorded evaluation. Updates 9,001–10,000 had no detected spikes; the gradient
norm averaged 1.683, maximum 8.03, with 96.1% of updates clipped. The linear
control averaged 0.190 in that interval with no clipping. This higher gradient
activity remains a stability concern despite improving validation loss.
The added quadratic coefficients were active in every layer by update 4,000;
their RMS then ranged from 0.87 to 0.97. Long-run stability and terminal
performance remain open.

Quadratic live mirror: [W&B run](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/d4ee18120782434cba349e45398d8b6c).


With GDN's L2-normalized keys and queries, the update is

```text
P_t = I - lambda_t k_t k_t^T
F_t(q) = alpha_t F_{t-1}(P_t q)
         + v_t [beta_{t,0} + beta_{t,1} <q,k_t> + beta_{t,2} <q,k_t>^2]
read_t = F_t(q_t) / sqrt(d_K)
```

The same `P_t` transforms both key indices of the quadratic state, with one shared
scalar decay `alpha_t`. The degree-zero state is unaffected by `P_t`. The scalar
decay and final `1/sqrt(d_K)` scale are outside the polynomial. There is no division
by a causal attention sum. GDN's short convolutions, SiLU, Q/K L2 normalization,
output gate, and gated output RMSNorm are retained.

The write coefficients are per token, head, and layer. The linear write remains
`beta_1 = sigmoid(W_b x) = (1 + tanh(W_b x / 2))/2`. The added signed writes are
`beta_0 = tanh(W_0 x)` and `beta_2 = tanh(W_2 x)`, with separate zero-initialized
projections. Their derivatives at initialization are nonzero. The original
projections retain their exact canonical initialization and RNG draws; the erase
projection starts as a distinct-buffer copy of the linear write projection.
Consequently the initial function equals the independent-erase GDN control.

The recipe keeps `d_K = d_V = 64`, 16 heads, 10 layers, hidden width 1024, MLP
width 3008, BF16, prediction length 2,048, effective batch 256, global microbatch
64, seed zero, and 19,074 updates on packed FineWeb. The shared AdamW recipe uses
400 warmup updates and learning rates 3e-4 to 3e-5. The new run records the current
activation, gradient, and parameter-update telemetry, plus separate degree-wise
write-coefficient diagnostics.

## Implementation and interpretation

The exact unrolling uses scores
`s_ti = q_t^T P_t ... P_{i+1} k_i`, with the identity product when `i = t`:

```text
F_t(q_t) = sum_(i<=t) exp(sum_(j=i+1..t) g_j)
          [beta_{i,0} + beta_{i,1} s_ti + beta_{i,2} s_ti^2] v_i
```

When `lambda = 0` and `alpha = 1`, this is unnormalized polynomial attention
with coefficients conditioned on the writing token. When the constant and
quadratic writes are zero it reduces to independent-erase GDN. A rank-one erase
in an expanded polynomial feature space would define a different recurrence.

The fused training backend evaluates this transported-score expression with
chunk-local triangular solves, FP32 intermediates, and TF32x3 products. It avoids
storing an expanded quadratic recurrent state and has quadratic sequence-length
work. Correctness and hardware checks belong to the shared scaffold's validation
record; they are not scientific training results.

Recipes and execution provenance live in
[the attention study](../../../experiments/attention_scaling/README.md#quadratic-gdn-query-geometry).
[selection.yaml](selection.yaml) pins the explicit primary and control.
Refresh its scientific inputs with
`python -m scaffolds.records refresh results/exploratory/gdn_polynomial`.

The cubic extension adds `beta_3 = tanh(W_3 x)`, with a separate zero-initialized
projection and the same `P_t` applied to all three key indices. It has
211,498,944 parameters, 163,840 more than quadratic. Both higher-degree runs
start at the same reference-initialized linear function. The cubic recipe keeps
the quadratic run's data, optimizer, warmup, output RMSNorm, and batch layout.
Its dedicated checks cover the cubic forward and all input gradients, exact
quadratic nesting at zero cubic write, noncommuting transition order, and
production sequence length and head widths. Full
[A100 qualification](../../../scaffolds/transformer/validation/runs/gdn-cubic-a100/runtime.json)
passed correctness, pilot validation, and exact checkpoint resumption. The pilot
projects 48.61 hours within the requested 72-hour production allocation.
This qualifies startup; long-run stability and performance remain open.

Cubic is paused at the user's request as of October 14, 2026 (EDT), with
Slurm job `218896` requeued in a held state. The latest durable checkpoint is
update 7,000, with validation perplexity 15.637199 and valid attention
diagnostics. Training reached logged update 7,260 before the pause; resumption
will replay those 260 updates from checkpoint 7,000. This is an incomplete run,
with no numerical failure. Its checkpoint, log, source, environment, and W&B
identity remain in the original Kennel worktree for resumption.

Cubic live mirror: [200m-dk064-gdn-cubic-geometry-warmup400-seed0](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/911b85e2ddde4223972bde38199aa936).

## Open decision

After completed training, compare terminal and validation-best loss, stability,
and use of the new degree-wise writes against the selected linear control.
Decide whether the mechanism warrants further matched runs and promotion into
the maintained attention comparison. One seed does not establish statistical
improvement, and the two mechanisms use different training kernels.
