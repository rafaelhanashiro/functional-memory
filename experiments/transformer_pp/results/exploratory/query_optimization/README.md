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

The existing control and the three fresh training runs share seed 42, initialization,
data order, global batch 256, 2,048 prediction tokens per sequence, and 7,570 updates
(3,968,860,160 prediction tokens). The base LR peaks at 0.001, warms up for 455
updates, and follows cosine decay to 0.0001. Each new run uses one GPU on Kennel.
Execution and launch instructions are in the [study README](../../../README.md).

| Role | Run | W decay | a,b LR multiplier | W LR multiplier |
| --- | --- | ---: | ---: | ---: |
| Control | `r2b22f2ff6461` | 0 | 1 | 1 |
| Projection decay | `rca97cc9db8f2` | 0.1 | 1 | 1 |
| Faster a,b | `rb74fa2ebede6` | 0 | 3 | 1 |
| W, a, b decay | `r68cac6f03d77` | 0.1, also on a and b | 1 | 1 |

The faster-a,b run uses peak LR 0.003 and minimum LR 0.0003 for only
`query_polynomial_bias` and `query_polynomial_amplitude`, across all orders,
layers, and heads. The multiplier acts on Adam updates after global gradient
clipping and moment normalization. W and every backbone parameter keep the base
schedule. The W, a, b decay run adds the same decay 0.1 on a and b to the W-decay run
(`attention_coefficient_amplitude_weight_decay` and `attention_coefficient_bias_weight_decay`,
switches defined only at its source commit). The backbone decay policy is unchanged.

## Findings and limits

All three optimizer ablations completed all 7,570 updates successfully. Lower validation
PPL is better; every validation-best checkpoint below is at update 7,500.

| Condition | Terminal PPL | Validation-best PPL | Terminal change from control |
| --- | ---: | ---: | ---: |
| Control | 14.779161 | 14.773754 | 0 |
| W decay 0.1 | 14.713560 | 14.708539 | -0.065600 (-0.44%) |
| 3x LR on a,b | 14.924409 | 14.919761 | +0.145249 (+0.98%) |
| W, a, b decay 0.1 | 14.757708 | 14.750170 | -0.021453 (-0.15%) |

Projection decay gives a small improvement in this comparison. Decaying a and b as well
gives back most of it, ending between W-only decay and the control. Tripling the a,b
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

## Projection decay across generators

Four degree-four runs apply the same W decay of 0.1 to PaTH, dense tanh features, and
query-key coefficients. Each matches its control except for the decay flags; for dense,
$V$ and $b$ stay exempt, and query-key decays both its query and key projections
(`attention_coefficient_key_kernel_weight_decay`, defined only at the run's source commit). Best validation perplexity; terminal values
are within 0.007 of it.

| Variant | Control, no W decay | W decay 0.1 | Change |
| --- | ---: | ---: | ---: |
| PaTH, query | 14.528 (`r74a50aca921e`) | 14.398 (`r1f2c24c109f9`) | -0.9% |
| Dense K = 5, $s = 1/\sqrt5$ | 14.710 (`rcaff0dfb095f`) | 14.739 (`rdacb58b7981c`) | +0.2% |
| Dense K = 9, $s = 1/3$ | 14.782 (`r04a5b2a3e8dd`) | 14.798 (`r753a7e8d3212`) | +0.1% |
| Query-key | 14.722 (`r501f5cca0d4c`) | 14.818 (`r79d829a16029`) | +0.7% |

- W decay helps PaTH query by 0.9%, twice its gain with RoPE at 1x.
- It leaves the dense generators level or slightly worse, and hurts query-key by 0.7%, more
  than at 2.5x (+0.07%).
- The K = 9 control used the older host gradient-accumulation loop; its W-decay run used
  the compiled loop, so that pair also differs in loop.

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

All four ran on ORCD, one H100 each, from launch commit `7662886`, as did the m = 8 and
key–query controls (launch commit `8ef54e8`). The query m = 4 control trained on a
Kennel B300 with the pre-7e5c962f training loop, so that pair also differs in hardware
and loop.

Terminal validation perplexity; every run's validation best is its terminal checkpoint.

| Pair | Control | W decay 0.1 | Change | W decay vs fixed Taylor at the degree |
| --- | ---: | ---: | ---: | ---: |
| Query m = 4 | 12.723783 | 12.689001 | -0.034782 (-0.27%) | -0.17% (12.710060, `r4f1682c1b54d`) |
| Query m = 8 | 12.475082 | 12.461680 | -0.013402 (-0.11%) | -0.11% (12.475191, `r0ea4c732c5ac`) |
| Key–query m = 4 | 12.664493 | 12.673930 | +0.009437 (+0.07%) | -0.28% |
| Key–query m = 8 | 12.449164 | 12.433672 | -0.015492 (-0.12%) | -0.33% |

- Decay lowers query perplexity at both degrees, by less than at 1x (-0.44%), and leaves
  key–query level: +0.07% at m = 4 and -0.12% at m = 8.
- Decay about halves the projections: the mean per-(layer, head, order) `||w_r||` ends at
  1.33–1.45 against 2.31–2.53 without decay, and the key–query key projection at
  1.25–1.39 against 2.17–2.38; the mean stored `|a_r|` is 6–10% larger with decay.
- Key–query m = 8 with decay, 12.434, is the lowest 2.5x perplexity in the study, 0.12%
  below exp (12.449, `rfe12e5c36e94`).
- Each pair is one seed, and every difference is under 0.3%, within the unresolved range;
  the query m = 4 pair also differs in hardware and training loop. Decay acts over 2.5x
  more updates here than at 1x (summed learning rate times 0.1: 1.04 against 0.41), so the
  1x and 2.5x gains measure different decay strengths.

## Selection and open decision

The [selection](selection.yaml) names the 1x control and both optimizer ablations, and
the four 2.5× pairs (roles `*_2p5x_control` and `*_2p5x_w_decay`).
After collecting run measurements, refresh their scientific input hashes with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/query_optimization
```

At 2.5× the W-decay improvement holds for query coefficients at a smaller size and is
absent for key–query. Decide whether W decay becomes the default for new query runs,
and whether a second seed is worth its cost; neither change is a default yet. Promote
this comparison into its own canonical collection, or discard the topic. These runs enter no canonical comparison while the topic is open.
