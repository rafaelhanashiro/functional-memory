# Query coefficient optimization

Opened 2026-10-29.

## Question and method

Does weight decay on the query projection W, or a higher learning rate on the
coefficient offsets b and amplitudes a, improve degree-four query/tanh attention?

Every run uses unnormalized attention with per-head output RMSNorm and

\[
c_r(x_t)=\frac{b_r+a_r\tanh(w_r^\top x_t)}{r!},\qquad r=0,\ldots,4.
\]

The stored parameters initialize at b=1, a=0, and independent projection entries
drawn from N(0, 1/896). Parameters are separate per layer and head. The complete
coefficient retains its factorial scaling in every condition. QK RMSNorm uses
per-layer channel gains and separate learned positive per-head score scales.

The existing control and the two fresh training runs share seed 42, initialization,
data order, global batch 256, 2,048 prediction tokens per sequence, and 7,570 updates
(3,968,860,160 prediction tokens). The base LR peaks at 0.001, warms up for 455
updates, and follows cosine decay to 0.0001. Both new runs use one GPU on Kennel.
Execution and launch instructions are in the [study README](../../../README.md).

| Role | Run | W decay | a,b LR multiplier | W LR multiplier |
| --- | --- | ---: | ---: | ---: |
| Control | `r2b22f2ff6461` | 0 | 1 | 1 |
| Projection decay | `rca97cc9db8f2` | 0.1 | 1 | 1 |
| Faster a,b | `rb74fa2ebede6` | 0 | 3 | 1 |

The faster-a,b run uses peak LR 0.003 and minimum LR 0.0003 for only
`query_polynomial_bias` and `query_polynomial_amplitude`, across all orders,
layers, and heads. The multiplier acts on Adam updates after global gradient
clipping and moment normalization. W and every backbone parameter keep the base
schedule. Neither run decays a or b. The backbone decay policy is unchanged.

## Findings and limits

Both optimizer ablations completed all 7,570 updates successfully. Lower validation
PPL is better; every validation-best checkpoint below is at update 7,500.

| Condition | Terminal PPL | Validation-best PPL | Terminal change from control |
| --- | ---: | ---: | ---: |
| Control | 14.779161 | 14.773754 | 0 |
| W decay 0.1 | 14.713560 | 14.708539 | -0.065600 (-0.44%) |
| 3x LR on a,b | 14.924409 | 14.919761 | +0.145249 (+0.98%) |

Projection decay gives a small improvement in this comparison. Tripling the a,b
learning rate worsens held-out PPL; this run does not support adopting that multiplier.
The same ordering holds for terminal and validation-best metrics. Coefficient
diagnostics remain with the run measurements; these PPL differences alone do not
establish a mechanism involving saturation or coefficient variation.

The W-decay job took 5h 15m on an H100 SXM and the higher-LR job took 7h 29m on an
H100 PCIe. These are complete Slurm allocations including acceptance checks,
validation, and checkpointing. Different GPU variants prevent attributing the
wall-clock difference to the optimizer change.

This is a single-seed comparison. The historical control accumulates microbatches
in a host loop; the new runs use the shared trainer's compiled accumulation loop.
The averaging, clipping, and AdamW mathematics agree, but floating-point training
trajectories need not be identical. Small differences warrant confirmation.

## Projection decay at 2.5×

Does W decay still help at 2.5× Chinchilla, at degree 8, and with key–query
coefficients? Four fresh runs each copy a completed 2.5× run from the
[token_budget topic](../token_budget/README.md) (18,925 updates, warmup 1,138) and
change only `model.attention_coefficient_kernel_weight_decay: true`, which applies the
shared weight decay 0.1 to the projections W. With key–query coefficients
`[b_r + a_r tanh(w_r^T x_t)] [1 + a^k_r tanh(w_k_r^T x_i)] / r!`, both the query
projection `w_r` and the key projection `w_k_r` decay; `b`, `a`, and `a^k` never decay.
Every run keeps the canonical per-head QK scale, seed 42, and its control's backend
(Mosaic for query, the fused Triton kernel for key–query).

| Variant | Control without decay | W decay 0.1 |
| --- | --- | --- |
| Query m = 4 | `reb9b5be4ff1b` (`query-m4-2.5x`) | `r77b617af91db` (`query-m4-w-decay-2.5x`) |
| Query m = 8 | `r77bca86ef253` (`query-m8-2.5x`) | `ra15cbb16beb7` (`query-m8-w-decay-2.5x`) |
| Key–query m = 4 | `r751984d1ca00` (`query_key-m4-2.5x`) | `rb6313357059e` (`query_key-m4-w-decay-2.5x`) |
| Key–query m = 8 | `rb7e83a4b653f` (`query_key-m8-2.5x`) | `r5bd803b3e562` (`query_key-m8-w-decay-2.5x`) |

All four are submitted on ORCD, one H100 each, like the m = 8 and key–query controls
(launch commit `8ef54e8`). The query m = 4 control trained on a Kennel B300 with the
pre-7e5c962f training loop, so that pair also differs in hardware and loop.

## Selection and open decision

The [selection](selection.yaml) names the control and both optimizer ablations; the
2.5× pairs join it when they finish.
After collecting run measurements, refresh their scientific input hashes with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/query_optimization
```

The 2.5× pairs test whether the small W-decay improvement holds; once they finish,
decide whether W decay becomes the default for new query and key–query runs. Neither
change is a default yet. Promote this comparison into its own canonical collection, or
discard the topic. These runs enter no canonical comparison while the topic is open.
