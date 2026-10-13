# Quadratic memory with GDN query geometry

Opened 2026-10-13. Home cluster: **Kennel**.

Does adding quadratic and constant writes improve a GDN memory whose learned
key-space transformation is unchanged?

## Runs and method

| Role | Run ID | Memory function | Status |
| --- | --- | --- | --- |
| Primary | `rd40af7956426` | Constant, linear, and quadratic | Prepared for qualification |
| Control | `rf7341031e3d5` | Linear, independent erase gate | Complete |

The explicitly selected control uses the repaired arithmetic, canonical GDN
initialization, independent erase gate, and the same data and optimizer recipe.
Its terminal and validation-best perplexity is 13.579409 at update 19,074.
The primary has no measured training outcome yet. Neither run is added to a
canonical result selection by this topic.

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

## Open decision

After completed training, compare terminal and validation-best loss, stability,
and use of the new degree-wise writes against the selected linear control.
Decide whether the mechanism warrants further matched runs and promotion into
the maintained attention comparison. One seed does not establish statistical
improvement, and the two mechanisms use different training kernels.
