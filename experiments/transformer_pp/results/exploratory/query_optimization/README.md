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

## Selection and open decision

The [selection](selection.yaml) names the control and both optimizer ablations.
After collecting run measurements, refresh their scientific input hashes with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/query_optimization
```

Decide whether the small W-decay improvement warrants confirmation; neither change
is a new default. Promote this comparison into its own canonical collection, or
discard the topic. These runs enter no canonical comparison while the topic is open.
