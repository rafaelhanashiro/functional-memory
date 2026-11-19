# Fixed linear attention

Opened 2026-11-02.

**Question.** How much does the exponential score kernel contribute beyond QK
normalization and attention-output head RMSNorm in Transformer++?

Run `rd434701f3492` replaces the exp control `r6a76e7fbde45`'s kernel by the
fixed polynomial `p(s) = s`: `[c_0, c_1] = [0, 1]`. The coefficients are
configuration constants, absent from the trainable parameter tree and optimizer.
For each head, the pre-normalization output is
`u_t = sum_(i<=t) (q_t^T k_i / sqrt(d_K)) v_i`; head RMSNorm follows the sum.
Weights may be negative. No division by the attention sum or forget gate is used.

The run trains from scratch with the control's initialization and seed 42.
It preserves RoPE, QK RMSNorm with per-head positive scaling, output head RMSNorm,
15 layers, hidden width 896, 14 heads, key/value width 64, FFN width 2304,
2,048-token context, batch 256 with microbatches of 16, and 7,570 updates.
Peak/minimum LR remain 0.001/0.0001, warmup is 455 updates, and AdamW uses the
same 0.1 decay with embeddings, normalization parameters, and biases exempt.
The complete frozen recipes are under the selected runs.

Execution uses the existing Mosaic static-polynomial implementation on one GPU;
the exp control used its fused exponential kernel. This is a configuration of
the shared Transformer scaffold, with no new model or attention implementation.
Standard acceptance checks constants staying outside the optimizer, unchanged
initialization, forward/gradient correctness, and training/checkpoint resume.
An additional exact-linear Mosaic check covers the full production context.
The [study README](../../../../../README.md#fixed-linear-attention) gives launch commands.
Home cluster: Kennel.

## Findings

| Run | `p(s)` | Best PPL (update) | Terminal PPL (update 7,570) |
| --- | --- | ---: | ---: |
| Exp control `r6a76e7fbde45` | `exp(s)` | **14.257** (7,500) | 14.262 |
| Fixed linear `rd434701f3492` | `s` | 20.717 (7,500) | 20.726 |

- Removing the exponential raises validation-best PPL by 45.3% (+0.374 nats per
  token); the terminal gap is the same.
- Both runs complete all 7,570 updates without numerical failure. The fixed-linear
  run passed full acceptance, including the exact-linear full-context check, on one
  H100 PCIe.
- One seed; seed-to-seed variability is not measured.

## Open question

Keep fixed linear as a baseline, or delete the topic.

The [selection](selection.yaml) pins the runs and scientific source files.
Measurements remain under each run; no copies are stored here. Refresh the pins
when intentionally updating the selected measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/alternative_attention/fixed_kernels/fixed_linear
```
